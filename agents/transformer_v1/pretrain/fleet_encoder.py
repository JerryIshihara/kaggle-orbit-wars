"""Single-fleet encoder pretraining.

Loads CSVs produced by ``featurizer.fleet_featurizer.save_episode_fleet_csv``
and trains :class:`FleetEncoder` + per-label heads against the
``ENCODER_PRETRAIN_LABELS`` multi-task objective. The encoder will later
be the per-element projection inside a fleet deep-set, so the goal of
this stage is to make sure the projection preserves the bits a
permutation-invariant aggregator will need (mission, size, direction,
position, kinematics) — see ``ENCODER_LABEL_HEADS`` for the wired heads.

Run from the repo root:

    python -m agents.transformer_v1.pretrain.fleet_encoder \
        --epochs 30 --d-model 64 --batch-size 1024

Outputs (under ``data/runs/fleet/<timestamp>/``):

* ``fleet_encoder_best.pt`` — checkpoint at lowest val mean loss.
* ``fleet_encoder_last.pt`` — last-epoch checkpoint.
* ``log.json`` — per-epoch train + val per-head losses & accuracies.
* ``test_summary.json`` — per-head test metrics from the best ckpt.
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

from ..featurizer import (
    ENCODER_LABEL_HEADS,
    ENCODER_PRETRAIN_LABELS,
    FLEET_RAW_DIM,
)
from ..paths import FLEET_DATASET_DIR, FLEET_RUNS_DIR
from ..encoder.fleet_encoder import FleetEncoder


# ---------- Dataset ----------
class FleetCsvDataset(Dataset):
    """Concatenate one or more episode CSVs into per-fleet tensors.

    All rows from all CSVs are loaded eagerly. With ~250k rows × 47 cols
    this is ~50 MB of floats — comfortably in RAM. Categorical labels
    cast to ``long``; binary cast to ``float`` (BCE-with-logits expects
    float targets); regression cast to ``float``.
    """

    FEATURE_COLS = tuple(f"f{i:03d}" for i in range(FLEET_RAW_DIM))

    def __init__(self, csv_paths: list[Path]):
        feats: list[list[float]] = []
        labels: dict[str, list[float]] = {n: [] for n in ENCODER_PRETRAIN_LABELS}
        for path in csv_paths:
            with path.open() as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    feats.append([float(row[c]) for c in self.FEATURE_COLS])
                    for name in ENCODER_PRETRAIN_LABELS:
                        spec = ENCODER_LABEL_HEADS[name]
                        v = row[name]
                        labels[name].append(
                            float(v) if spec["type"] == "regression" else int(v)
                        )

        self.features = torch.tensor(feats, dtype=torch.float32)
        self.labels: dict[str, torch.Tensor] = {}
        for name in ENCODER_PRETRAIN_LABELS:
            spec = ENCODER_LABEL_HEADS[name]
            if spec["type"] == "categorical":
                self.labels[name] = torch.tensor(labels[name], dtype=torch.long)
            else:  # binary or regression — both want float targets
                self.labels[name] = torch.tensor(labels[name], dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], {k: v[idx] for k, v in self.labels.items()}


# ---------- Model ----------
class FleetEncoderPretrainModel(nn.Module):
    """``FleetEncoder`` + one linear head per pretraining label.

    Heads are intentionally tiny (linear from ``d_model``) so the
    encoder must do the work — multi-task loss pressure has to flow
    into the projection, not get absorbed by big heads.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.encoder = FleetEncoder(d_model=d_model)
        self.heads = nn.ModuleDict()
        for name in ENCODER_PRETRAIN_LABELS:
            spec = ENCODER_LABEL_HEADS[name]
            out_dim = spec["n_classes"] if spec["type"] == "categorical" else 1
            self.heads[name] = nn.Linear(d_model, out_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        # FleetEncoder expects (B, F, FLEET_RAW_DIM); we're per-fleet, so
        # add a fake F=1 dim, project, squeeze. Mask is irrelevant — every
        # row in a pretraining batch is a real fleet.
        x = features.unsqueeze(1)
        tokens = self.encoder(x)
        z = tokens.squeeze(1)
        return {name: head(z) for name, head in self.heads.items()}


# ---------- Loss / metrics ----------
def _per_head_loss(name: str, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    spec = ENCODER_LABEL_HEADS[name]
    if spec["type"] == "categorical":
        return F.cross_entropy(logit, target)
    if spec["type"] == "binary":
        return F.binary_cross_entropy_with_logits(logit.squeeze(-1), target)
    return F.mse_loss(logit.squeeze(-1), target)


def compute_total_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Equal-weighted sum of per-head losses by default.

    Categorical / binary CE are O(1); regression MSE on normalized
    targets is also O(1). Equal weights work well in practice for
    encoder pretraining; pass ``weights`` to up-weight specific heads.
    """
    losses: dict[str, float] = {}
    total: torch.Tensor | None = None
    for name, logit in preds.items():
        l = _per_head_loss(name, logit, targets[name])
        w = (weights or {}).get(name, 1.0)
        contrib = w * l
        total = contrib if total is None else total + contrib
        losses[name] = float(l.detach())
    assert total is not None
    return total, losses


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, dict[str, float]]:
    """Return per-head ``{loss, acc?}`` averaged over the loader."""
    model.eval()
    head_state = {
        n: {"loss_sum": 0.0, "n": 0, "correct": 0} for n in ENCODER_PRETRAIN_LABELS
    }
    for features, targets in loader:
        features = features.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        preds = model(features)
        for name, logit in preds.items():
            spec = ENCODER_LABEL_HEADS[name]
            tgt = targets[name]
            bs = tgt.shape[0]
            if spec["type"] == "categorical":
                l = F.cross_entropy(logit, tgt, reduction="sum")
                head_state[name]["correct"] += int((logit.argmax(-1) == tgt).sum())
            elif spec["type"] == "binary":
                l = F.binary_cross_entropy_with_logits(
                    logit.squeeze(-1), tgt, reduction="sum"
                )
                head_state[name]["correct"] += int(
                    ((logit.squeeze(-1) > 0).float() == tgt).sum()
                )
            else:
                l = F.mse_loss(logit.squeeze(-1), tgt, reduction="sum")
            head_state[name]["loss_sum"] += float(l)
            head_state[name]["n"] += bs

    summary: dict[str, dict[str, float]] = {}
    for name, state in head_state.items():
        spec = ENCODER_LABEL_HEADS[name]
        n = max(1, state["n"])
        entry: dict[str, float] = {"loss": state["loss_sum"] / n}
        if spec["type"] != "regression":
            entry["acc"] = state["correct"] / n
        summary[name] = entry
    return summary


def _format_summary(summary: dict[str, dict[str, float]], indent: int = 4) -> str:
    pad = " " * indent
    lines = []
    for name, m in summary.items():
        acc_str = f"  acc={m['acc']:.3f}" if "acc" in m else ""
        lines.append(f"{pad}{name:<28s}  loss={m['loss']:.4f}{acc_str}")
    return "\n".join(lines)


# ---------- Train loop ----------
def train(
    *,
    data_dir: Path,
    out_dir: Path,
    d_model: int = 64,
    batch_size: int = 1024,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 1,
    analyze_every: int = 5,
    device: str | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> Path:
    """End-to-end training. Returns the best-checkpoint path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    manifest = json.loads((data_dir / "manifest.json").read_text())

    def _resolve(names: list[str]) -> list[Path]:
        return [data_dir / n for n in names]

    train_ds = FleetCsvDataset(_resolve(manifest["train"]))
    val_ds = FleetCsvDataset(_resolve(manifest["val"]))
    test_ds = FleetCsvDataset(_resolve(manifest["test"]))
    print(
        f"[pretrain] device={device}  d_model={d_model}  batch={batch_size}  epochs={epochs}\n"
        f"[pretrain] rows: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    model = FleetEncoderPretrainModel(d_model=d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "fleet_encoder_best.pt"
    last_path = out_dir / "fleet_encoder_last.pt"

    config = {
        "d_model": d_model,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "feature_dim": FLEET_RAW_DIM,
        "labels": list(ENCODER_PRETRAIN_LABELS),
        "label_heads": {k: dict(v) for k, v in ENCODER_LABEL_HEADS.items()},
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
            total_loss, per_head = compute_total_loss(preds, targets)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            running_total += float(total_loss.detach())
            for k, v in per_head.items():
                running_per_head[k] += v
            n_batches += 1

        train_total = running_total / max(1, n_batches)
        train_per_head = {k: v / max(1, n_batches) for k, v in running_per_head.items()}

        entry: dict[str, Any] = {
            "epoch": epoch,
            "train_total": train_total,
            "train_per_head": train_per_head,
            "elapsed_s": round(time.time() - t0, 2),
        }

        if epoch % eval_every == 0 or epoch == epochs:
            val_summary = evaluate(model, val_loader, device)
            mean_val_loss = sum(m["loss"] for m in val_summary.values()) / len(val_summary)
            entry["val_mean_loss"] = mean_val_loss
            entry["val"] = val_summary
            print(
                f"[epoch {epoch:>3d}/{epochs}]  train_total={train_total:.4f}  "
                f"val_mean={mean_val_loss:.4f}  ({entry['elapsed_s']}s)"
            )
            if mean_val_loss < best_val:
                best_val = mean_val_loss
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "config": config},
                    best_path,
                )

        if epoch % analyze_every == 0 or epoch == epochs:
            # Periodic per-head readout — useful for catching heads that
            # haven't started learning while the mean loss looks fine.
            print(f"  per-head val (epoch {epoch}):")
            print(_format_summary(entry.get("val", {}) or evaluate(model, val_loader, device)))

        log.append(entry)
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "config": config},
            last_path,
        )
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    # Final test using the best checkpoint
    print("\n[pretrain] evaluating best checkpoint on test set...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_summary = evaluate(model, test_loader, device)
    print(_format_summary(test_summary, indent=2))
    (out_dir / "test_summary.json").write_text(json.dumps(test_summary, indent=2))
    print(f"\n[pretrain] outputs in {out_dir}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=FLEET_DATASET_DIR)
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Default: data/runs/fleet/<timestamp>",
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--analyze-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or (FLEET_RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"))
    train(
        data_dir=args.data_dir,
        out_dir=out_dir,
        d_model=args.d_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        analyze_every=args.analyze_every,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
