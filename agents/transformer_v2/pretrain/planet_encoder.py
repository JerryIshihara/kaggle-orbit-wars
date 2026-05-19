"""Single-entity (planet+comet) encoder pretraining.

Loads CSVs produced by ``planet_featurizer.save_episode_planet_csv``
and trains :class:`PlanetEncoder` + per-label heads against the
``ENCODER_PRETRAIN_LABELS`` multi-task objective.

Symmetric with ``fleet_encoder`` (the fleet pretrain in this same
package) — same dataset / model / training-loop scaffolding, just
different feature dim and label set.

Run from the repo root:

    python -m agents.transformer_v2.pretrain.planet_encoder \
        --epochs 30 --d-model 64 --batch-size 4096

Outputs (under ``data/runs/planet/<timestamp>/``):

* ``planet_encoder_best.pt`` — checkpoint at lowest val mean loss
* ``planet_encoder_last.pt`` — last-epoch checkpoint
* ``log.json`` — per-epoch train + val per-head losses & accuracies
* ``test_summary.json`` — per-head test metrics from the best ckpt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..paths import PLANET_DATASET_DIR, PLANET_RUNS_DIR
from ..featurizer.planet_featurizer import (
    ENCODER_LABEL_HEADS,
    ENCODER_PRETRAIN_LABELS,
    EXTRAP_MASK_COLS,
    EXTRAP_TARGET_COLS,
    N_EXTRAP_HORIZONS,
    PLANET_RAW_DIM,
)
from ..encoder.planet_encoder import PlanetEncoder


# ---------- Dataset ----------
class PlanetCsvDataset(Dataset):
    """Concatenate one or more episode CSVs into per-entity tensors."""

    FEATURE_COLS = tuple(f"f{i:03d}" for i in range(PLANET_RAW_DIM))

    def __init__(self, csv_paths: list[Path]):
        # Pre-compute which scalar (i.e., non-trajectory) head names we
        # actually need to load — trajectory is handled separately.
        scalar_labels = [
            n for n in ENCODER_PRETRAIN_LABELS
            if ENCODER_LABEL_HEADS[n]["type"] != "masked_regression"
        ]
        feats: list[list[float]] = []
        labels: dict[str, list[float]] = {n: [] for n in scalar_labels}
        traj_targets: list[list[float]] = []
        traj_masks: list[list[int]] = []

        for path in csv_paths:
            with path.open() as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    feats.append([float(row[c]) for c in self.FEATURE_COLS])
                    for name in scalar_labels:
                        spec = ENCODER_LABEL_HEADS[name]
                        v = row[name]
                        labels[name].append(
                            float(v) if spec["type"] == "regression" else int(v)
                        )
                    traj_targets.append([float(row[c]) for c in EXTRAP_TARGET_COLS])
                    traj_masks.append([int(row[c]) for c in EXTRAP_MASK_COLS])

        self.features = torch.tensor(feats, dtype=torch.float32)
        self.labels: dict[str, torch.Tensor] = {}
        for name in scalar_labels:
            spec = ENCODER_LABEL_HEADS[name]
            if spec["type"] == "categorical":
                self.labels[name] = torch.tensor(labels[name], dtype=torch.long)
            else:
                self.labels[name] = torch.tensor(labels[name], dtype=torch.float32)

        # Trajectory: targets stay flat (B, 2H); the model reshapes
        # (B, H, 2). Mask is (B, H) and broadcasts over the xy axis.
        if traj_targets:
            self.labels["extrap_trajectory"] = torch.tensor(traj_targets, dtype=torch.float32)
            self.labels["extrap_mask"] = torch.tensor(traj_masks, dtype=torch.float32)

        # ``is_comet`` flag for trajectory-head sample weighting. The
        # featurizer puts ``is_comet`` at feature column 0 (the very
        # first dim of ``to_vector``); just slice it out.
        if self.features.numel():
            self.labels["is_comet"] = self.features[:, 0]

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], {k: v[idx] for k, v in self.labels.items()}


# ---------- Model ----------
class TrajectoryDecoder(nn.Module):
    """Two-layer MLP decoder that maps a per-entity embedding to the
    next ``n_horizons`` future displacements ``(dx, dy)``.

    Why two layers and not one: a single ``Linear`` from ``d_model`` is
    expressive enough for orbiting planets (motion is linear in the
    encoded angle/radius), but comets follow elliptical paths whose
    forward extrapolation is genuinely nonlinear in the encoded state.
    The hidden ``GELU`` lets the decoder learn that without forcing the
    encoder to bake out the trajectory itself.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_horizons: int = 10,
        hidden: int | None = None,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1 (got {num_layers})")
        self.n_horizons = n_horizons
        h = hidden or d_model
        layers: list[nn.Module] = []
        in_dim = d_model
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2 * n_horizons))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Returns flat (B, 2 * n_horizons); caller reshapes for loss.
        return self.net(z)


class StratifiedTrajectoryDecoder(nn.Module):
    """Three sub-decoders, each specialized to a horizon range.

    Hypothesis from the comet-extrapolation experiments: a single decoder
    is forced to balance close-range precision (h=1) against long-range
    drift (h=10), and the multi-task pressure plus information bottleneck
    means it can't be sharp at both extremes simultaneously. Splitting
    into three sub-heads (short/mid/long) lets each one specialize on
    its slice of the horizon spectrum without competing for parameters.

    Defaults assume ``n_horizons=10``: short=h1..h3, mid=h4..h6, long=h7..h10.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_horizons: int = 10,
        hidden: int | None = None,
        splits: tuple[int, ...] = (3, 3, 4),
    ):
        super().__init__()
        if sum(splits) != n_horizons:
            raise ValueError(
                f"splits {splits} sum to {sum(splits)} but n_horizons={n_horizons}"
            )
        self.n_horizons = n_horizons
        self.splits = splits
        h = hidden or d_model
        # One sub-MLP per slice; each emits ``2 * slice_len`` outputs.
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, h),
                nn.GELU(),
                nn.Linear(h, 2 * s),
            )
            for s in splits
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Concat sub-head outputs in horizon order so the downstream
        # reshape (B, n_horizons, 2) lines up with EXTRAP_HORIZONS.
        return torch.cat([head(z) for head in self.heads], dim=-1)


class PlanetEncoderPretrainModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        decoder: str = "vanilla",
        encoder_num_layers: int = 2,
        head_num_layers: int = 1,
        head_hidden: int | None = None,
        separate_traj_decoders: bool = False,
        use_traj_branch: bool = True,
    ):
        super().__init__()
        if head_num_layers < 1:
            raise ValueError(
                f"head_num_layers must be >= 1 (got {head_num_layers})"
            )
        # v7 encoder is a two-branch MLP with fixed structure; the legacy
        # ``encoder_num_layers`` kwarg is kept for compat but no longer
        # routed (the v7 PlanetEncoder doesn't take it).
        self.encoder = PlanetEncoder(d_model=d_model, use_traj_branch=use_traj_branch)
        head_hidden = head_hidden if head_hidden is not None else d_model
        self.separate_traj_decoders = separate_traj_decoders

        def _scalar_head(out_dim: int) -> nn.Module:
            """N-layer MLP head for scalar / categorical / binary labels."""
            if head_num_layers == 1:
                return nn.Linear(d_model, out_dim)
            layers: list[nn.Module] = []
            in_dim = d_model
            for _ in range(head_num_layers - 1):
                layers.append(nn.Linear(in_dim, head_hidden))
                layers.append(nn.GELU())
                in_dim = head_hidden
            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)

        def _traj_decoder(n_horizons: int) -> nn.Module:
            if decoder == "stratified":
                return StratifiedTrajectoryDecoder(
                    d_model=d_model, n_horizons=n_horizons,
                )
            if decoder == "vanilla":
                return TrajectoryDecoder(
                    d_model=d_model, n_horizons=n_horizons,
                    num_layers=head_num_layers,
                )
            raise ValueError(f"unknown decoder kind: {decoder!r}")

        self.heads = nn.ModuleDict()
        for name in ENCODER_PRETRAIN_LABELS:
            spec = ENCODER_LABEL_HEADS[name]
            if spec["type"] == "categorical":
                self.heads[name] = _scalar_head(spec["n_classes"])
            elif spec["type"] == "binary":
                self.heads[name] = _scalar_head(1)
            elif spec["type"] == "regression":
                self.heads[name] = _scalar_head(1)
            elif spec["type"] == "masked_regression":
                if separate_traj_decoders:
                    # Two specialized decoders, gated at output by
                    # ``is_comet``. Each decoder only sees gradient from
                    # its own entity type because the gating coefficient
                    # for the other type is zero.
                    self.heads[name] = nn.ModuleDict({
                        "planet": _traj_decoder(spec["n_horizons"]),
                        "comet":  _traj_decoder(spec["n_horizons"]),
                    })
                else:
                    self.heads[name] = _traj_decoder(spec["n_horizons"])
            else:
                raise ValueError(f"unknown head type: {spec['type']!r}")

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        x = features.unsqueeze(1)              # (B, 1, PLANET_RAW_DIM)
        tokens = self.encoder(x)               # (B, 1, d_model)
        z = tokens.squeeze(1)                  # (B, d_model)
        is_comet = features[..., 0:1]          # (B, 1) — channel 0 is is_comet flag

        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            spec = ENCODER_LABEL_HEADS[name]
            if (
                spec["type"] == "masked_regression"
                and self.separate_traj_decoders
            ):
                # Gated route: each decoder produces (B, 2*H); combine
                # with is_comet so each row only carries gradient through
                # its matching decoder.
                pred_p = head["planet"](z)
                pred_c = head["comet"](z)
                out[name] = is_comet * pred_c + (1.0 - is_comet) * pred_p
            else:
                out[name] = head(z)
        return out


# ---------- Loss / metrics ----------
# Per-horizon training weight for the trajectory MSE loss. Set to a
# non-uniform schedule (e.g., (2,1,...,1,2)) to push the optimizer to
# fit endpoints more aggressively — typically the boundary horizons
# (h=1 close-precision, h=N long-range drift) are the hardest cells
# for a multi-task encoder. Eval still computes unweighted MSE for
# fair comparison.
HORIZON_LOSS_WEIGHTS: tuple[float, ...] = (1.0,) * N_EXTRAP_HORIZONS

# Per-head loss weights. After dropping ``extrap_xy_long`` (the sparse
# long head), no head needs explicit re-weighting; keep the dict as a
# future hook.
HEAD_LOSS_WEIGHTS: dict[str, float] = {}

# Per-sample weighting inside trajectory heads. Comets are ~38% of the
# dataset and have structurally harder trajectories. Up-weight comet
# samples in trajectory MSE so the encoder allocates capacity for both
# orbital and ballistic dynamics. Set to 1.0 for parity-of-treatment.
COMET_TRAJECTORY_WEIGHT: float = 2.0


def _masked_traj_mse(
    pred_flat: torch.Tensor,         # (B, 2H)
    target_flat: torch.Tensor,       # (B, 2H)
    mask: torch.Tensor,              # (B, H)
    n_horizons: int,
    *,
    reduction: str = "mean",
    horizon_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,    # (B,) optional
) -> torch.Tensor:
    """Per-step masked MSE over a packed (B, 2H) trajectory.

    Reshapes both prediction and target to (B, H, 2), broadcasts the
    (B, H) mask over the xy axis, and averages only over valid
    elements. ``reduction='sum'`` returns the un-normalized squared-
    error sum (useful for accumulating across batches in eval).

    If ``horizon_weights`` is given (shape ``(H,)``), each horizon's
    squared-error contribution is scaled by its weight and the
    denominator is also weighted, so the "mean" reduction stays a
    weighted average rather than a re-scaled sum. Eval should pass
    ``horizon_weights=None`` so reported MSE is unweighted.

    If ``sample_weights`` is given (shape ``(B,)``), each row's
    contribution is scaled in both numerator and denominator. Used to
    up-weight comet rows so the encoder allocates capacity to their
    harder trajectories. Eval should pass ``sample_weights=None`` for
    unbiased reporting.
    """
    B = pred_flat.shape[0]
    pred = pred_flat.view(B, n_horizons, 2)
    target = target_flat.view(B, n_horizons, 2)
    m = mask.unsqueeze(-1)                     # (B, H, 1)
    sq = (pred - target).pow(2) * m            # (B, H, 2)

    if horizon_weights is not None:
        w_h = horizon_weights.to(sq.device).view(1, n_horizons, 1)   # (1, H, 1)
        sq = sq * w_h
        denom_mask = m * w_h
    else:
        denom_mask = m

    if sample_weights is not None:
        w_s = sample_weights.to(sq.device).view(B, 1, 1)
        sq = sq * w_s
        denom_mask = denom_mask * w_s

    denom = denom_mask.sum() * 2.0             # 2 = xy dims per step

    if reduction == "sum":
        return sq.sum()
    return sq.sum() / denom.clamp(min=1.0)


def _per_head_loss(name: str, logit: torch.Tensor, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    spec = ENCODER_LABEL_HEADS[name]
    if spec["type"] == "categorical":
        return F.cross_entropy(logit, targets[name])
    if spec["type"] == "binary":
        return F.binary_cross_entropy_with_logits(logit.squeeze(-1), targets[name])
    if spec["type"] == "regression":
        return F.mse_loss(logit.squeeze(-1), targets[name])
    if spec["type"] == "masked_regression":
        mask = targets[spec["mask_field"]]
        H = spec["n_horizons"]
        weights = (
            torch.as_tensor(HORIZON_LOSS_WEIGHTS, dtype=logit.dtype)
            if len(HORIZON_LOSS_WEIGHTS) == H else None
        )
        # Comet-up-weighting on trajectory heads. ``is_comet`` is at
        # feature column 0 (planet_featurizer.to_vector first slot).
        sample_w = None
        if COMET_TRAJECTORY_WEIGHT != 1.0 and "is_comet" in targets:
            is_comet = targets["is_comet"].float()
            sample_w = 1.0 + (COMET_TRAJECTORY_WEIGHT - 1.0) * is_comet
        return _masked_traj_mse(
            logit, targets[name], mask, H,
            horizon_weights=weights, sample_weights=sample_w,
        )
    raise ValueError(f"unknown head type: {spec['type']!r}")


def compute_total_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: dict[str, float] = {}
    total: torch.Tensor | None = None
    for name, logit in preds.items():
        l = _per_head_loss(name, logit, targets)
        w = HEAD_LOSS_WEIGHTS.get(name, 1.0)
        contrib = w * l
        total = contrib if total is None else total + contrib
        losses[name] = float(l.detach())
    assert total is not None
    return total, losses


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, dict[str, float]]:
    model.eval()
    state = {n: {"loss_sum": 0.0, "n": 0, "correct": 0} for n in ENCODER_PRETRAIN_LABELS}
    # Per-horizon accumulators keyed by masked-regression head name so
    # multiple trajectory-style labels can coexist (e.g. the dense
    # ``extrap_trajectory`` head + the sparse ``extrap_xy_long`` head).
    traj_sse: dict[str, torch.Tensor] = {}
    traj_n: dict[str, torch.Tensor] = {}
    for name in ENCODER_PRETRAIN_LABELS:
        spec = ENCODER_LABEL_HEADS[name]
        if spec["type"] == "masked_regression":
            H = spec["n_horizons"]
            traj_sse[name] = torch.zeros(H)
            traj_n[name] = torch.zeros(H)

    for features, targets in loader:
        features = features.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        preds = model(features)
        for name, logit in preds.items():
            spec = ENCODER_LABEL_HEADS[name]
            if spec["type"] == "masked_regression":
                tgt = targets[name]
                mask = targets[spec["mask_field"]]
                bs = tgt.shape[0]
                H = spec["n_horizons"]
                # Sum-reduction loss for averaging at the end
                sse = _masked_traj_mse(logit, tgt, mask, H, reduction="sum")
                state[name]["loss_sum"] += float(sse)
                # Number of valid (xy) elements
                n_valid = float(mask.sum() * 2.0)
                state[name]["n"] += int(n_valid) if n_valid > 0 else 0
                # Per-horizon breakdown
                pred_h = logit.view(bs, H, 2).cpu()
                tgt_h = tgt.view(bs, H, 2).cpu()
                m_h = mask.cpu()
                per_step_sq = ((pred_h - tgt_h).pow(2).sum(-1) * m_h)  # (B, H)
                traj_sse[name] += per_step_sq.sum(dim=0)
                traj_n[name] += m_h.sum(dim=0) * 2.0
                continue

            t = targets[name]
            bs = t.shape[0]
            if spec["type"] == "categorical":
                l = F.cross_entropy(logit, t, reduction="sum")
                state[name]["correct"] += int((logit.argmax(-1) == t).sum())
            elif spec["type"] == "binary":
                l = F.binary_cross_entropy_with_logits(logit.squeeze(-1), t, reduction="sum")
                state[name]["correct"] += int(((logit.squeeze(-1) > 0).float() == t).sum())
            else:
                l = F.mse_loss(logit.squeeze(-1), t, reduction="sum")
            state[name]["loss_sum"] += float(l)
            state[name]["n"] += bs

    summary: dict[str, dict[str, float]] = {}
    for name, s in state.items():
        spec = ENCODER_LABEL_HEADS[name]
        n = max(1, s["n"])
        entry: dict[str, float] = {"loss": s["loss_sum"] / n}
        if spec["type"] in ("categorical", "binary"):
            entry["acc"] = s["correct"] / n
        if spec["type"] == "masked_regression":
            per_h = (traj_sse[name] / traj_n[name].clamp(min=1.0)).tolist()
            entry["per_horizon_mse"] = [round(v, 5) for v in per_h]
        summary[name] = entry
    return summary


def _format_summary(summary: dict[str, dict[str, float]], indent: int = 4) -> str:
    pad = " " * indent
    lines = []
    for name, m in summary.items():
        acc = f"  acc={m['acc']:.3f}" if "acc" in m else ""
        lines.append(f"{pad}{name:<28s}  loss={m['loss']:.4f}{acc}")
    return "\n".join(lines)


# ---------- Train loop ----------
def train(
    *,
    data_dir: Path,
    out_dir: Path,
    d_model: int = 128,
    encoder_num_layers: int = 2,
    head_num_layers: int = 1,
    head_hidden: int | None = None,
    batch_size: int = 4096,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 1,
    analyze_every: int = 5,
    device: str | None = None,
    num_workers: int = 0,
    seed: int = 0,
    decoder: str = "vanilla",
    separate_traj_decoders: bool = False,
    traj_loss_weight: float = 1.0,
    use_traj_branch: bool = True,
    horizon_loss_max: float = 1.0,
) -> Path:
    # Per-head loss weights live in a module-level dict that
    # ``compute_total_loss`` consults. Override the trajectory weight
    # here so the optimizer prioritizes encoder capacity for the
    # trajectory head when ``traj_loss_weight > 1``.
    if traj_loss_weight != 1.0:
        HEAD_LOSS_WEIGHTS["extrap_trajectory"] = float(traj_loss_weight)
    elif "extrap_trajectory" in HEAD_LOSS_WEIGHTS:
        # Reset to default so a previous override in the same Python
        # process doesn't leak into this run.
        HEAD_LOSS_WEIGHTS.pop("extrap_trajectory")
    # Per-horizon ramp: linear interpolation from 1.0 at h=1 to
    # ``horizon_loss_max`` at h=N. >1 pushes the optimizer to fit late
    # horizons more aggressively (where RMSE is highest). Eval still
    # reports unweighted MSE for fair comparison.
    global HORIZON_LOSS_WEIGHTS
    if horizon_loss_max != 1.0:
        H = N_EXTRAP_HORIZONS
        HORIZON_LOSS_WEIGHTS = tuple(
            1.0 + (horizon_loss_max - 1.0) * (i / (H - 1)) for i in range(H)
        )
    else:
        HORIZON_LOSS_WEIGHTS = (1.0,) * N_EXTRAP_HORIZONS
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    train_ds = PlanetCsvDataset([data_dir / n for n in manifest["train"]])
    val_ds = PlanetCsvDataset([data_dir / n for n in manifest["val"]])
    test_ds = PlanetCsvDataset([data_dir / n for n in manifest["test"]])
    print(
        f"[planet-pretrain] device={device}  d_model={d_model}  batch={batch_size}  epochs={epochs}\n"
        f"[planet-pretrain] rows: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    model = PlanetEncoderPretrainModel(
        d_model=d_model, decoder=decoder,
        encoder_num_layers=encoder_num_layers,
        head_num_layers=head_num_layers,
        head_hidden=head_hidden,
        separate_traj_decoders=separate_traj_decoders,
        use_traj_branch=use_traj_branch,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "planet_encoder_best.pt"
    last_path = out_dir / "planet_encoder_last.pt"
    config = {
        "d_model": d_model,
        "encoder_num_layers": encoder_num_layers,
        "head_num_layers": head_num_layers,
        "head_hidden": head_hidden if head_hidden is not None else d_model,
        "decoder": decoder,
        "separate_traj_decoders": separate_traj_decoders,
        "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": epochs,
        "feature_dim": PLANET_RAW_DIM,
        "labels": list(ENCODER_PRETRAIN_LABELS),
        "label_heads": {k: dict(v) for k, v in ENCODER_LABEL_HEADS.items()},
        "head_loss_weights": dict(HEAD_LOSS_WEIGHTS),
        "traj_loss_weight": traj_loss_weight,
        "use_traj_branch": use_traj_branch,
        "horizon_loss_max": horizon_loss_max,
        "horizon_loss_weights": list(HORIZON_LOSS_WEIGHTS),
    }

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        running_per_head: dict[str, float] = {n: 0.0 for n in ENCODER_PRETRAIN_LABELS}
        n_batches = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            preds = model(features)
            total, per_head = compute_total_loss(preds, targets)
            opt.zero_grad()
            total.backward()
            opt.step()
            running_total += float(total.detach())
            for k, v in per_head.items():
                running_per_head[k] += v
            n_batches += 1
        train_total = running_total / max(1, n_batches)
        train_per_head = {k: v / max(1, n_batches) for k, v in running_per_head.items()}

        entry: dict[str, Any] = {
            "epoch": epoch, "train_total": train_total,
            "train_per_head": train_per_head,
            "elapsed_s": round(time.time() - t0, 2),
        }
        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate(model, val_loader, device)
            mean = sum(m["loss"] for m in val.values()) / len(val)
            entry["val_mean_loss"] = mean
            entry["val"] = val
            print(
                f"[epoch {epoch:>3d}/{epochs}]  train_total={train_total:.4f}  "
                f"val_mean={mean:.4f}  ({entry['elapsed_s']}s)"
            )
            if mean < best_val:
                best_val = mean
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, best_path)
        if epoch % analyze_every == 0 or epoch == epochs:
            print(f"  per-head val (epoch {epoch}):")
            print(_format_summary(entry.get("val", {}) or evaluate(model, val_loader, device)))
        log.append(entry)
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, last_path)
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    print("\n[planet-pretrain] evaluating best checkpoint on test set...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_summary = evaluate(model, test_loader, device)
    print(_format_summary(test_summary, indent=2))
    (out_dir / "test_summary.json").write_text(json.dumps(test_summary, indent=2))
    print(f"\n[planet-pretrain] outputs in {out_dir}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PLANET_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-num-layers", type=int, default=2)
    parser.add_argument("--head-num-layers", type=int, default=1)
    parser.add_argument("--head-hidden", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--analyze-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--decoder", type=str, default="vanilla",
        choices=("vanilla", "stratified"),
        help="Trajectory decoder architecture: vanilla = single 2-layer "
             "MLP; stratified = three sub-MLPs over h1..3 / h4..6 / h7..10.",
    )
    parser.add_argument(
        "--separate-traj-decoders", action="store_true",
        help="Build two trajectory decoders (planet / comet) gated by "
             "is_comet at output. Each decoder only sees gradient from "
             "its own entity type via output masking.",
    )
    parser.add_argument(
        "--traj-loss-weight", type=float, default=1.0,
        help="Multiplier on the extrap_trajectory head's loss "
             "contribution in compute_total_loss. >1 pushes the encoder "
             "to allocate more capacity to trajectory at the expense of "
             "scalar / categorical heads.",
    )
    parser.add_argument(
        "--no-traj-branch", action="store_true",
        help="Ablation: drop the trajectory anchor branch from the "
             "encoder. The model must derive future positions from "
             "scalar kinematics alone (x, y, vx, vy, ω, sun-relative).",
    )
    parser.add_argument(
        "--horizon-loss-max", type=float, default=1.0,
        help="Per-horizon weight ramp endpoint: weights interpolate "
             "linearly from 1.0 at h=1 to this value at h=N. >1 biases "
             "the optimizer toward long horizons.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (PLANET_RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"))
    train(
        data_dir=args.data_dir,
        out_dir=out_dir,
        d_model=args.d_model,
        encoder_num_layers=args.encoder_num_layers,
        head_num_layers=args.head_num_layers,
        head_hidden=args.head_hidden,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        analyze_every=args.analyze_every,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        decoder=args.decoder,
        separate_traj_decoders=args.separate_traj_decoders,
        traj_loss_weight=args.traj_loss_weight,
        use_traj_branch=not args.no_traj_branch,
        horizon_loss_max=args.horizon_loss_max,
    )


if __name__ == "__main__":
    main()
