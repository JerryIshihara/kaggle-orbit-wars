"""Latent probe — locate the remaining trajectory-MSE bottleneck.

Loads the trained planet encoder (3-channel + gated-fusion checkpoint
from ``data/runs/planet/20260429-225920/`` by default), FREEZES
it, and trains a fatter trajectory decoder on top using the same
multi-task CSV dataset. Compare per-horizon comet RMSE against the
original (small) decoder:

  * Probe ≪ original: the encoder already preserves enough trajectory
    info; the original decoder was the bottleneck. Recommend keeping
    the encoder, just enlarging the decoder going forward.
  * Probe ≈ original: the bottleneck is in the encoder. No amount of
    decoder capacity can recover info that's no longer in the
    embedding. Look at encoder architecture / input features instead.

Usage:
    python scripts/probe_latent_trajectory.py \\
        --run-dir data/runs/planet/20260429-225920 \\
        --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from agents.transformer_v1.encoder.planet_encoder import PlanetEncoder  # noqa: E402
from agents.transformer_v1.pretrain.planet_encoder import (  # noqa: E402
    PlanetCsvDataset,
    _masked_traj_mse,
)
from agents.transformer_v1.featurizer.planet_featurizer import (  # noqa: E402
    N_EXTRAP_HORIZONS,
)


class FatTrajectoryDecoder(nn.Module):
    """Three-layer MLP, much wider than the original 2-layer 64→64→20.

    The probe's whole point is "give the decoder more capacity than the
    original ever had, and see whether MSE drops." If yes → the
    encoder's bottleneck embedding *does* still carry trajectory info
    that the original decoder failed to extract. If MSE plateaus near
    the original — encoder is the bottleneck.
    """

    def __init__(self, d_model: int = 64, n_horizons: int = 10, hidden: int = 256):
        super().__init__()
        self.n_horizons = n_horizons
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * n_horizons),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    from agents.transformer_v1.paths import PLANET_DATASET_DIR, PLANET_RUNS_DIR
    parser.add_argument(
        "--run-dir", type=Path,
        default=PLANET_RUNS_DIR / "20260429-225920",
        help="Source encoder checkpoint dir (must contain planet_encoder_best.pt).",
    )
    parser.add_argument("--data-dir", type=Path, default=PLANET_DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or (
        PLANET_RUNS_DIR / f"probe-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load and freeze encoder ----
    ckpt_path = args.run_dir / "planet_encoder_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d_model = ckpt["config"]["d_model"]
    encoder = PlanetEncoder(d_model=d_model)
    encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in ckpt["model"].items()
         if k.startswith("encoder.")},
        strict=True,
    )
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ---- Load dataset ----
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    train_ds = PlanetCsvDataset([args.data_dir / n for n in manifest["train"]])
    val_ds = PlanetCsvDataset([args.data_dir / n for n in manifest["val"]])
    test_ds = PlanetCsvDataset([args.data_dir / n for n in manifest["test"]])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    print(f"[probe] device={device}  d_model={d_model}  hidden={args.hidden}  "
          f"epochs={args.epochs}")
    print(f"[probe] encoder frozen from {ckpt_path.relative_to(REPO)}")
    print(f"[probe] rows: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ---- Build fat decoder ----
    decoder = FatTrajectoryDecoder(
        d_model=d_model, n_horizons=N_EXTRAP_HORIZONS, hidden=args.hidden,
    ).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"[probe] decoder params: {n_params:,} (vs original ~9k)")

    opt = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- Helpers ----
    def encode(features: torch.Tensor) -> torch.Tensor:
        # Encoder expects (B, F, PLANET_RAW_DIM); we feed unbatched per-
        # entity rows so unsqueeze a fake F dim.
        with torch.no_grad():
            x = features.unsqueeze(1)
            tokens = encoder(x)
            return tokens.squeeze(1)            # (B, d_model)

    @torch.no_grad()
    def evaluate(loader: DataLoader) -> dict[str, float]:
        decoder.eval()
        traj_sse = torch.zeros(N_EXTRAP_HORIZONS)
        traj_n = torch.zeros(N_EXTRAP_HORIZONS)
        for features, targets in loader:
            features = features.to(device)
            tgt = targets["extrap_trajectory"].to(device)
            mask = targets["extrap_mask"].to(device)
            z = encode(features)
            pred = decoder(z)
            B = pred.shape[0]
            pred_h = pred.view(B, N_EXTRAP_HORIZONS, 2).cpu()
            tgt_h = tgt.view(B, N_EXTRAP_HORIZONS, 2).cpu()
            m_h = mask.cpu()
            per_step_sq = ((pred_h - tgt_h).pow(2).sum(-1) * m_h)
            traj_sse += per_step_sq.sum(dim=0)
            traj_n += m_h.sum(dim=0) * 2.0
        per_h = (traj_sse / traj_n.clamp(min=1.0)).tolist()
        decoder.train()
        return {"per_horizon_mse": [round(v, 6) for v in per_h]}

    # ---- Train ----
    log = []
    best_val_mean = float("inf")
    best_path = out_dir / "probe_decoder_best.pt"
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        running = 0.0
        n_batches = 0
        for features, targets in train_loader:
            features = features.to(device)
            tgt = targets["extrap_trajectory"].to(device)
            mask = targets["extrap_mask"].to(device)
            z = encode(features)
            pred = decoder(z)
            loss = _masked_traj_mse(pred, tgt, mask, N_EXTRAP_HORIZONS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n_batches += 1
        train_loss = running / max(1, n_batches)
        val = evaluate(val_loader)
        val_mean = sum(val["per_horizon_mse"]) / len(val["per_horizon_mse"])
        log.append({"epoch": epoch, "train_loss": train_loss,
                    "val_mean": val_mean, **val})
        print(f"  ep {epoch:>2}/{args.epochs}  train={train_loss:.5f}  "
              f"val_mean_mse={val_mean:.5f}")
        if val_mean < best_val_mean:
            best_val_mean = val_mean
            torch.save({"decoder": decoder.state_dict(),
                        "epoch": epoch,
                        "encoder_run_dir": str(args.run_dir),
                        "hidden": args.hidden}, best_path)

    # ---- Test with best ----
    ckpt2 = torch.load(best_path, map_location=device, weights_only=False)
    decoder.load_state_dict(ckpt2["decoder"])
    test = evaluate(test_loader)
    (out_dir / "probe_log.json").write_text(json.dumps(log, indent=2))
    (out_dir / "probe_test_summary.json").write_text(json.dumps(test, indent=2))

    print()
    print("==== test set per-horizon MSE ====")
    for h, mse in enumerate(test["per_horizon_mse"], 1):
        rmse_units = (mse ** 0.5) * 50.0
        print(f"  h={h:>2}  MSE={mse:.6f}  RMSE ≈ {rmse_units:.3f} board units")
    print(f"\n[probe] outputs in {out_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
