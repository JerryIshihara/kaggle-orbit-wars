"""Per-horizon hit-rate eval for the comet past/fullpath specialist.

Mirrors ``per_horizon_hit_rate.py`` but uses ``CometPastModel`` and the
35-slot path-feature CSV format.
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
)
from agents.transformer_v2.pretrain.comet_past_encoder import (  # noqa: E402
    CometPastModel, PAST_COLS, SCALAR_FEAT_COLS,
    EXTRAP_TARGET_COLS, EXTRAP_MASK_COLS,
    N_EXTRAP,
)


def _load_test(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    feats, tgts, masks = [], [], []
    for name in manifest["test"]:
        with (data_dir / name).open() as fh:
            r = csv.DictReader(fh)
            for row in r:
                # If the CSV has the scalar input columns (f000..f017),
                # prepend them so the input vector matches what the model
                # was trained with.
                scalar = (
                    [float(row[c]) for c in SCALAR_FEAT_COLS]
                    if SCALAR_FEAT_COLS[0] in row else []
                )
                past = [float(row[c]) for c in PAST_COLS]
                feats.append(scalar + past)
                tgts.append([float(row[c]) for c in EXTRAP_TARGET_COLS])
                masks.append([int(row[c]) for c in EXTRAP_MASK_COLS])
    return (torch.tensor(feats, dtype=torch.float32),
            torch.tensor(tgts, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32))


@torch.no_grad()
def _predict(model: CometPastModel, feats: torch.Tensor, batch: int = 4096):
    out = []
    for i in range(0, feats.shape[0], batch):
        preds = model(feats[i:i+batch])
        # Multi-task model returns a dict; single-task wrap path also
        # uses ``extrap_trajectory`` key now.
        if isinstance(preds, dict):
            out.append(preds["extrap_trajectory"])
        else:
            out.append(preds)
    return torch.cat(out, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0])
    args = ap.parse_args()

    feats, tgts, masks = _load_test(args.data_dir)
    print(f"[hit-rate-comet] rows: {feats.shape[0]}")
    ckpt = torch.load(args.run_dir / "comet_past_best.pt",
                      map_location="cpu", weights_only=False)
    d_model = ckpt["config"]["d_model"]
    multi_task = ckpt["config"].get("multi_task", False)
    input_dim = ckpt["config"].get("input_dim", feats.shape[1])
    model = CometPastModel(d_model=d_model, multi_task=multi_task,
                           input_dim=input_dim)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    pred = _predict(model, feats)
    B = pred.shape[0]
    p = pred.view(B, N_EXTRAP, 2) * ANCHOR_DXY_NORM
    t = tgts.view(B, N_EXTRAP, 2) * ANCHOR_DXY_NORM
    err = ((p - t) ** 2).sum(-1).sqrt()       # (B, H)

    thresholds = tuple(args.thresholds)
    header = (f"  {'h':>3s} {'n':>6s} {'rmse':>6s} {'med':>6s}  "
              + "  ".join(f"≤{x:g}" for x in thresholds))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for j, h in enumerate(EXTRAP_HORIZONS):
        m = masks[:, j].bool()
        n = int(m.sum())
        if n == 0:
            continue
        e = err[m, j]
        rmse = float((e**2).mean().sqrt())
        med = float(e.median())
        hits = "  ".join(f"{float((e <= thr).float().mean()) * 100:>5.1f}%"
                         for thr in thresholds)
        print(f"  {h:>3d} {n:>6d} {rmse:>6.2f} {med:>6.2f}  {hits}")


if __name__ == "__main__":
    main()
