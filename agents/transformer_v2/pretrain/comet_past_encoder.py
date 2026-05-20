"""Comet trajectory-extrapolation pretraining (past → future).

The model sees 35 past timestamp slots (each (dx, dy, valid) relative
to current position) and must predict the next 30 future displacements.
No scalar features, no future anchors — pure trajectory extrapolation.

Encoder: 3-Linear MLP ``105 → 128 → 128 → 128`` + LayerNorm
Decoder: 3-Linear MLP ``128 → 128 → 128 → 60`` (30 horizons × 2 dims)

Run::

    python -m agents.transformer_v2.pretrain.comet_past_encoder \\
        --data-dir data/datasets/comet_only_40k_past \\
        --out-dir data/runs/comet/past_d128_40k_lr1e4_120ep \\
        --d-model 128 --epochs 120 --lr 1e-4 --batch-size 256
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

from ..featurizer.planet_featurizer import (
    EXTRAP_HORIZONS,
    N_SPEED_BUCKETS,
    N_SUN_BUCKETS,
    SCALAR_DIM,
)

N_PAST = 35
PAST_CHANNELS = 3
PAST_DIM = N_PAST * PAST_CHANNELS    # 105
N_EXTRAP = len(EXTRAP_HORIZONS)      # 30
EXTRAP_DIM = 2 * N_EXTRAP            # 60
SCALAR_FEAT_DIM = SCALAR_DIM         # 18 — matches planet specialist
INPUT_DIM = SCALAR_FEAT_DIM + PAST_DIM    # 123 when both are present


SCALAR_FEAT_COLS: list[str] = [f"f{i:03d}" for i in range(SCALAR_FEAT_DIM)]
PAST_COLS: list[str] = []
for _k in range(1, N_PAST + 1):
    PAST_COLS.append(f"past_dx_t{_k:02d}")
    PAST_COLS.append(f"past_dy_t{_k:02d}")
    PAST_COLS.append(f"past_valid_t{_k:02d}")
EXTRAP_TARGET_COLS: list[str] = []
EXTRAP_MASK_COLS: list[str] = []
for _h in EXTRAP_HORIZONS:
    EXTRAP_TARGET_COLS.append(f"extrap_dx_h{_h:02d}")
    EXTRAP_TARGET_COLS.append(f"extrap_dy_h{_h:02d}")
    EXTRAP_MASK_COLS.append(f"extrap_mask_h{_h:02d}")

# Scalar tasks mirroring the planet specialist (same names, same
# normalizations). The dataset builder writes these alongside the
# trajectory labels.
SCALAR_HEADS: dict[str, dict[str, Any]] = {
    "distance_to_sun_bucket": {"type": "categorical", "n_classes": N_SUN_BUCKETS},
    "speed_bucket":           {"type": "categorical", "n_classes": N_SPEED_BUCKETS},
    "recon_x_norm":           {"type": "regression"},
    "recon_y_norm":           {"type": "regression"},
    "recon_vx_norm":          {"type": "regression"},
    "recon_vy_norm":          {"type": "regression"},
}


class CometPastDataset(Dataset):
    """Loads past-path features (+ optional 18-dim scalar features) +
    extrap labels/masks + the 6 scalar labels. Both the scalar input
    columns and the scalar label columns are auto-detected — datasets
    without them load fine for single-task / path-only models.
    """

    def __init__(self, csv_paths: list[Path]):
        feats: list[list[float]] = []
        scalar_feats: list[list[float]] = []
        tgts: list[list[float]] = []
        masks: list[list[int]] = []
        scalar_lists: dict[str, list[float]] = {n: [] for n in SCALAR_HEADS}
        has_scalar_input = False
        has_scalar_labels = False
        for path in csv_paths:
            with path.open() as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    feats.append([float(row[c]) for c in PAST_COLS])
                    tgts.append([float(row[c]) for c in EXTRAP_TARGET_COLS])
                    masks.append([int(row[c]) for c in EXTRAP_MASK_COLS])
                    if SCALAR_FEAT_COLS[0] in row:
                        has_scalar_input = True
                        scalar_feats.append(
                            [float(row[c]) for c in SCALAR_FEAT_COLS]
                        )
                    if SCALAR_HEADS and "distance_to_sun_bucket" in row:
                        has_scalar_labels = True
                        for name, spec in SCALAR_HEADS.items():
                            v = row[name]
                            if spec["type"] == "categorical":
                                scalar_lists[name].append(float(int(v)))
                            else:
                                scalar_lists[name].append(float(v))
        self.past_features = torch.tensor(feats, dtype=torch.float32)
        self.targets = torch.tensor(tgts, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.has_scalar_input = has_scalar_input
        if has_scalar_input:
            self.scalar_features = torch.tensor(
                scalar_feats, dtype=torch.float32,
            )
            # Concatenate scalar in front of trajectory so the model's
            # input vector is [scalar(18) ‖ trajectory(105)].
            self.features = torch.cat(
                [self.scalar_features, self.past_features], dim=-1,
            )
        else:
            self.scalar_features = torch.empty(
                (self.past_features.shape[0], 0), dtype=torch.float32,
            )
            self.features = self.past_features
        self.scalar: dict[str, torch.Tensor] = {}
        if has_scalar_labels:
            for name, spec in SCALAR_HEADS.items():
                if spec["type"] == "categorical":
                    self.scalar[name] = torch.tensor(
                        scalar_lists[name], dtype=torch.long,
                    )
                else:
                    self.scalar[name] = torch.tensor(
                        scalar_lists[name], dtype=torch.float32,
                    )
        self.has_scalar = has_scalar_labels

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx):
        scalar = {k: v[idx] for k, v in self.scalar.items()}
        return self.features[idx], self.targets[idx], self.masks[idx], scalar


def _mlp(in_dim: int, out_dim: int, *, n_hidden: int, hidden: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.GELU())
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class CometEncoder(nn.Module):
    """The shared comet encoder: ``(B, input_dim) → (B, d_model)`` token.

    This is the SINGLE class used both inside :class:`CometPastModel`
    during L0 pretrain and as a standalone L0 encoder downstream (the
    entity-pretrain stack and the runtime agent both consume it). The
    body is a 3-Linear MLP (2 × Linear+GELU + final Linear) followed
    by LayerNorm — pure scalar feature → token, no trajectory head.
    """

    def __init__(self, d_model: int = 128, input_dim: int = INPUT_DIM):
        super().__init__()
        self.d_model = int(d_model)
        self.input_dim = int(input_dim)
        # 3-Linear MLP: 2 (Linear→GELU) + final Linear.
        self.encoder = _mlp(input_dim, d_model, n_hidden=2, hidden=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


class CometDecoder(nn.Module):
    """Pretrain-only decoder: ``(B, d_model) → {trajectory, scalar heads}``.

    Trajectory branch is a 3-Linear MLP producing the
    30 future ``(dx, dy)`` slots. When ``multi_task=True``, six scalar
    heads are added (categorical and regression) that mirror the planet
    specialist's auxiliary tasks. Used only during L0 pretrain; the
    entity-stack / agent path doesn't consume this module.
    """

    def __init__(self, d_model: int = 128, multi_task: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.multi_task = bool(multi_task)
        self.decoder = _mlp(d_model, EXTRAP_DIM, n_hidden=2, hidden=d_model)
        if multi_task:
            self.scalar_heads = nn.ModuleDict()
            for name, spec in SCALAR_HEADS.items():
                out_dim = spec["n_classes"] if spec["type"] == "categorical" else 1
                # 3-Linear head mirrors the planet specialist.
                self.scalar_heads[name] = _mlp(
                    d_model, out_dim, n_hidden=2, hidden=d_model,
                )
        else:
            self.scalar_heads = None

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {"extrap_trajectory": self.decoder(z)}
        if self.scalar_heads is not None:
            for name, head in self.scalar_heads.items():
                out[name] = head(z)
        return out


def _remap_legacy_comet_state_dict(state: dict) -> dict:
    """Map the legacy flat-layout state_dict to the composed layout.

    Old layout (CometPastModel == single nn.Module with all submodules at
    the top level):

        encoder.X            decoder.X
        norm.X               scalar_heads.NAME.X

    New layout (CometPastModel == CometEncoder + CometDecoder):

        encoder.encoder.X    decoder.decoder.X
        encoder.norm.X       decoder.scalar_heads.NAME.X
    """
    out: dict = {}
    for k, v in state.items():
        if k.startswith("encoder.") or k.startswith("norm."):
            out[f"encoder.{k}"] = v
        elif k.startswith("decoder.") or k.startswith("scalar_heads."):
            out[f"decoder.{k}"] = v
        else:
            out[k] = v
    return out


class CometPastModel(nn.Module):
    """Pretrain composite: :class:`CometEncoder` → :class:`CometDecoder`.

    Saved state_dict layout: ``encoder.encoder.*``, ``encoder.norm.*``,
    ``decoder.decoder.*``, ``decoder.scalar_heads.NAME.*``.

    Legacy ckpts (``encoder.*`` / ``norm.*`` / ``decoder.*`` / ``scalar_heads.*``
    at the top level) are auto-remapped by :meth:`load_state_dict` so prior
    ``comet_past_best.pt`` files keep loading. After this refactor, new ckpts
    written by the pretrain script use the composed layout.
    """

    def __init__(
        self,
        d_model: int = 128,
        multi_task: bool = True,
        input_dim: int = INPUT_DIM,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.multi_task = bool(multi_task)
        self.encoder = CometEncoder(d_model=d_model, input_dim=input_dim)
        self.decoder = CometDecoder(d_model=d_model, multi_task=multi_task)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Detect legacy flat layout via canonical key absence: the new layout
        # has "encoder.encoder.0.weight", the old one has "encoder.0.weight"
        # at the top level (no "encoder.encoder.*").
        if (
            "encoder.encoder.0.weight" not in state_dict
            and "encoder.0.weight" in state_dict
        ):
            state_dict = _remap_legacy_comet_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


def _masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor,
                *, reduction: str = "mean",
                horizon_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Masked MSE over (B, 2H) flattened trajectory.

    If ``horizon_weights`` is given (shape ``(H,)``), per-horizon
    contributions are scaled in both numerator and denominator so the
    ``mean`` reduction stays a weighted average (not just a re-scaled
    sum). Eval should pass ``horizon_weights=None``.
    """
    B = pred.shape[0]
    p = pred.view(B, N_EXTRAP, 2)
    t = tgt.view(B, N_EXTRAP, 2)
    m = mask.unsqueeze(-1)                  # (B, H, 1)
    sq = (p - t).pow(2) * m                  # (B, H, 2)
    if horizon_weights is not None:
        w = horizon_weights.to(sq.device).view(1, N_EXTRAP, 1)
        sq = sq * w
        denom_mask = m * w
    else:
        denom_mask = m
    if reduction == "sum":
        return sq.sum()
    denom = denom_mask.sum() * 2.0           # 2 = xy
    return sq.sum() / denom.clamp(min=1.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, Any]:
    model.eval()
    sse_h = torch.zeros(N_EXTRAP)
    n_h = torch.zeros(N_EXTRAP)
    total_sse = 0.0
    total_n = 0
    scalar_state: dict[str, dict[str, float]] = {
        n: {"loss_sum": 0.0, "n": 0, "correct": 0} for n in SCALAR_HEADS
    }
    for feats, tgts, masks, scalar in loader:
        feats = feats.to(device)
        tgts = tgts.to(device)
        masks = masks.to(device)
        preds = model(feats)
        traj_pred = preds["extrap_trajectory"]
        sse = _masked_mse(traj_pred, tgts, masks, reduction="sum")
        total_sse += float(sse)
        total_n += int(masks.sum() * 2)
        B = traj_pred.shape[0]
        per_h = ((traj_pred.view(B, N_EXTRAP, 2) - tgts.view(B, N_EXTRAP, 2)).pow(2)
                 .sum(-1) * masks).cpu()
        sse_h += per_h.sum(dim=0)
        n_h += masks.cpu().sum(dim=0) * 2.0
        for name, spec in SCALAR_HEADS.items():
            if name not in preds or name not in scalar:
                continue
            logit = preds[name]
            tgt = scalar[name].to(device)
            bs = tgt.shape[0]
            if spec["type"] == "categorical":
                l = F.cross_entropy(logit, tgt, reduction="sum")
                scalar_state[name]["correct"] += int((logit.argmax(-1) == tgt).sum())
            else:
                l = F.mse_loss(logit.squeeze(-1), tgt, reduction="sum")
            scalar_state[name]["loss_sum"] += float(l)
            scalar_state[name]["n"] += bs
    per_horizon = (sse_h / n_h.clamp(min=1.0)).tolist()
    out: dict[str, Any] = {
        "extrap_trajectory": {
            "loss": total_sse / max(1, total_n),
            "per_horizon_mse": [round(v, 6) for v in per_horizon],
        }
    }
    for name, s in scalar_state.items():
        if s["n"] == 0:
            continue
        entry: dict[str, float] = {"loss": s["loss_sum"] / s["n"]}
        if SCALAR_HEADS[name]["type"] == "categorical":
            entry["acc"] = s["correct"] / s["n"]
        out[name] = entry
    return out


def train(
    *,
    data_dir: Path,
    out_dir: Path,
    d_model: int = 128,
    batch_size: int = 256,
    epochs: int = 120,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str | None = None,
    seed: int = 1729,
    num_workers: int = 0,
    horizon_loss_max: float = 1.0,
    traj_loss_weight: float = 1.0,
    multi_task: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    # Per-horizon loss schedule: linear ramp from 1.0 at h=1 to
    # ``horizon_loss_max`` at h=N. >1 biases the optimizer toward late
    # horizons. Normalized so mean weight = 1.0 (so total trajectory
    # loss magnitude doesn't shift, only its per-horizon distribution).
    if horizon_loss_max != 1.0:
        H = N_EXTRAP
        raw = torch.tensor(
            [1.0 + (horizon_loss_max - 1.0) * (i / (H - 1)) for i in range(H)],
            dtype=torch.float32,
        )
        horizon_weights: torch.Tensor | None = raw / raw.mean()
    else:
        horizon_weights = None

    manifest = json.loads((data_dir / "manifest.json").read_text())
    train_ds = CometPastDataset([data_dir / n for n in manifest["train"]])
    val_ds = CometPastDataset([data_dir / n for n in manifest["val"]])
    test_ds = CometPastDataset([data_dir / n for n in manifest["test"]])
    print(f"[comet-past] device={device}  d_model={d_model}  "
          f"batch={batch_size}  epochs={epochs}")
    print(f"[comet-past] rows: train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             num_workers=num_workers)

    multi_task_effective = multi_task and train_ds.has_scalar
    if multi_task and not train_ds.has_scalar:
        print("[comet-past] dataset lacks scalar columns — falling back to "
              "single-task (extrap only).")
    input_dim = train_ds.input_dim
    print(f"[comet-past] input_dim={input_dim}  "
          f"(scalar_input={train_ds.has_scalar_input}, "
          f"multi_task={multi_task_effective})")
    model = CometPastModel(d_model=d_model,
                           multi_task=multi_task_effective,
                           input_dim=input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)

    config = {
        "d_model": d_model, "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": epochs,
        "n_past": N_PAST, "past_channels": PAST_CHANNELS,
        "past_dim": PAST_DIM, "n_extrap": N_EXTRAP,
        "input_dim": input_dim,
        "scalar_feat_dim": SCALAR_FEAT_DIM if train_ds.has_scalar_input else 0,
        "horizon_loss_max": horizon_loss_max,
        "horizon_weights": (horizon_weights.tolist()
                            if horizon_weights is not None else None),
        "traj_loss_weight": traj_loss_weight,
        "multi_task": multi_task_effective,
    }
    best_val = float("inf")
    best_path = out_dir / "comet_past_best.pt"
    last_path = out_dir / "comet_past_last.pt"
    log: list[dict[str, Any]] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for feats, tgts, masks, scalar in train_loader:
            feats = feats.to(device)
            tgts = tgts.to(device)
            masks = masks.to(device)
            preds = model(feats)
            traj_loss = _masked_mse(
                preds["extrap_trajectory"], tgts, masks,
                horizon_weights=horizon_weights,
            )
            total_loss = traj_loss_weight * traj_loss
            if multi_task_effective:
                for name, spec in SCALAR_HEADS.items():
                    if name not in preds:
                        continue
                    tgt = scalar[name].to(device)
                    logit = preds[name]
                    if spec["type"] == "categorical":
                        scalar_loss = F.cross_entropy(logit, tgt)
                    else:
                        scalar_loss = F.mse_loss(logit.squeeze(-1), tgt)
                    total_loss = total_loss + scalar_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            running += float(total_loss.detach())
            n_batches += 1
        train_loss = running / max(1, n_batches)
        val = evaluate(model, val_loader, device)
        val_traj_loss = val["extrap_trajectory"]["loss"]
        entry = {
            "epoch": epoch, "train_loss": train_loss,
            "val_traj_loss": val_traj_loss,
            "elapsed_s": round(time.time() - t0, 2),
        }
        log.append(entry)
        scalar_brief = ""
        if multi_task_effective:
            parts = []
            for name in SCALAR_HEADS:
                if name in val:
                    if "acc" in val[name]:
                        parts.append(f"{name[:14]}.acc={val[name]['acc']:.3f}")
                    else:
                        parts.append(f"{name[:14]}.l={val[name]['loss']:.4f}")
            scalar_brief = "  " + " ".join(parts)
        print(f"[epoch {epoch:>3d}/{epochs}]  train={train_loss:.5f}  "
              f"val_traj={val_traj_loss:.5f}{scalar_brief}  "
              f"({entry['elapsed_s']}s)")
        # Best ckpt selected on trajectory loss (mirrors planet specialist).
        if val_traj_loss < best_val:
            best_val = val_traj_loss
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "config": config}, best_path)
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "config": config}, last_path)
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    print("\n[comet-past] evaluating best checkpoint on test set...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device)
    print(f"  test traj loss = {test['extrap_trajectory']['loss']:.6f}")
    for name in SCALAR_HEADS:
        if name in test:
            extra = f"  acc={test[name].get('acc', 0):.3f}" if "acc" in test[name] else ""
            print(f"  {name:<28s}  loss={test[name]['loss']:.4f}{extra}")
    (out_dir / "test_summary.json").write_text(json.dumps(test, indent=2))
    print(f"\n[comet-past] outputs in {out_dir}")
    return best_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/datasets/comet_only_40k_past"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument(
        "--horizon-loss-max", type=float, default=1.0,
        help="Per-horizon weight ramp endpoint: weights interpolate "
             "linearly from 1.0 at h=1 to this value at h=N, then are "
             "normalized so the mean weight is 1.0. >1 biases the "
             "optimizer toward long horizons without inflating total "
             "trajectory loss magnitude.",
    )
    ap.add_argument(
        "--traj-loss-weight", type=float, default=1.0,
        help="Multiplier on the extrap_trajectory loss in the total "
             "(traj + scalar) multi-task loss.",
    )
    ap.add_argument(
        "--single-task", action="store_true",
        help="Disable scalar heads even if the dataset has those columns.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or Path(
        f"data/runs/comet/past_d{args.d_model}_40k_lr{args.lr:g}_{args.epochs}ep"
    )
    train(
        data_dir=args.data_dir, out_dir=out_dir,
        d_model=args.d_model, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        device=args.device, seed=args.seed,
        horizon_loss_max=args.horizon_loss_max,
        traj_loss_weight=args.traj_loss_weight,
        multi_task=not args.single_task,
    )


if __name__ == "__main__":
    main()
