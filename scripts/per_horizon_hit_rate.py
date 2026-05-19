"""Per-horizon hit-rate eval for the v2 planet/comet specialists.

For each test row, runs encoder + trajectory decoder and computes the
Euclidean distance error at every horizon h=1..30 in **board units**.
Hit-rate is then the fraction of valid predictions whose error is below
each threshold.

Run from the repo root::

    python -m scripts.per_horizon_hit_rate \\
        --run-dir data/runs/planet/specialist_planet_d128_dense30_trajw5_60ep \\
        --data-dir data/datasets/planet_only_20k_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agents.transformer_v2.featurizer.planet_featurizer import (  # noqa: E402
    ANCHOR_DXY_NORM,
    EXTRAP_HORIZONS,
    EXTRAP_MASK_COLS,
    EXTRAP_TARGET_COLS,
    N_EXTRAP_HORIZONS,
    PLANET_RAW_DIM,
)
from agents.transformer_v2.pretrain.planet_encoder import (  # noqa: E402
    PlanetEncoderPretrainModel,
)


FEATURE_COLS = tuple(f"f{i:03d}" for i in range(PLANET_RAW_DIM))


def _load_test(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    feats: list[list[float]] = []
    targets: list[list[float]] = []
    masks: list[list[int]] = []
    for name in manifest["test"]:
        with (data_dir / name).open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                feats.append([float(row[c]) for c in FEATURE_COLS])
                targets.append([float(row[c]) for c in EXTRAP_TARGET_COLS])
                masks.append([int(row[c]) for c in EXTRAP_MASK_COLS])
    return (
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.float32),
    )


def _build_model(ckpt: dict) -> PlanetEncoderPretrainModel:
    cfg = ckpt["config"]
    m = PlanetEncoderPretrainModel(
        d_model=cfg["d_model"],
        decoder=cfg.get("decoder", "vanilla"),
        head_num_layers=cfg.get("head_num_layers", 1),
        head_hidden=cfg.get("head_hidden"),
        separate_traj_decoders=cfg.get("separate_traj_decoders", False),
        use_traj_branch=cfg.get("use_traj_branch", True),
    )
    m.load_state_dict(ckpt["model"], strict=True)
    m.eval()
    return m


@torch.no_grad()
def _predict(model: PlanetEncoderPretrainModel, feats: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    out_chunks: list[torch.Tensor] = []
    for i in range(0, feats.shape[0], batch):
        preds = model(feats[i : i + batch])
        out_chunks.append(preds["extrap_trajectory"])
    return torch.cat(out_chunks, dim=0)


def _hit_rate_table(
    pred_flat: torch.Tensor,
    tgt_flat: torch.Tensor,
    mask: torch.Tensor,
    thresholds: tuple[float, ...],
) -> dict[int, dict[str, float]]:
    """Returns {h: {thr: hit_rate, "rmse": …, "n": …}}.

    Errors are computed in **board units** by un-normalizing the
    decoder's output and target via ``ANCHOR_DXY_NORM``.
    """
    B = pred_flat.shape[0]
    H = N_EXTRAP_HORIZONS
    pred = pred_flat.view(B, H, 2) * ANCHOR_DXY_NORM
    tgt = tgt_flat.view(B, H, 2) * ANCHOR_DXY_NORM
    err = ((pred - tgt) ** 2).sum(-1).sqrt()      # (B, H)

    out: dict[int, dict[str, float]] = {}
    for j, h in enumerate(EXTRAP_HORIZONS):
        m = mask[:, j].bool()
        n = int(m.sum())
        if n == 0:
            continue
        e = err[m, j]
        row: dict[str, float] = {
            "n": float(n),
            "rmse": float((e**2).mean().sqrt()),
            "median": float(e.median()),
        }
        for thr in thresholds:
            row[f"hit_{thr}"] = float((e <= thr).float().mean())
        out[int(h)] = row
    return out


def _fmt(table: dict[int, dict[str, float]], thresholds: tuple[float, ...]) -> str:
    lines = []
    header = (
        f"  {'h':>3s} {'n':>6s} {'rmse':>6s} {'med':>6s}  "
        + "  ".join(f"≤{t:g}" for t in thresholds)
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for h in sorted(table):
        r = table[h]
        hits = "  ".join(f"{r[f'hit_{t}'] * 100:>5.1f}%" for t in thresholds)
        lines.append(
            f"  {h:>3d} {int(r['n']):>6d} {r['rmse']:>6.2f} {r['median']:>6.2f}  {hits}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
        help="Hit thresholds in board units (RMSE distance in raw coords).",
    )
    args = ap.parse_args()

    print(f"[hit-rate] run-dir:  {args.run_dir}")
    print(f"[hit-rate] data-dir: {args.data_dir}")

    feats, tgts, masks = _load_test(args.data_dir)
    print(f"[hit-rate] rows: {feats.shape[0]}")

    ckpt = torch.load(args.run_dir / "planet_encoder_best.pt", map_location="cpu", weights_only=False)
    model = _build_model(ckpt)
    pred = _predict(model, feats)

    table = _hit_rate_table(pred, tgts, masks, tuple(args.thresholds))
    print(_fmt(table, tuple(args.thresholds)))


if __name__ == "__main__":
    main()
