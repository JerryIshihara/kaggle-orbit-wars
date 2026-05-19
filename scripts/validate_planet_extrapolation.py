"""Validate the planet/comet trajectory decoder against a "within
planet radius" success threshold.

The decoder predicts the next 10 turns' (dx, dy) per entity. For each
predicted position we ask: did the prediction land within the
entity's own radius of the ground-truth position? That's the threshold
that matters semantically — a fleet aiming for the planet only needs
to land inside ``radius`` of the planet center to count as a hit.

We split entities into three kinds, since their motion regimes differ
sharply:

  * ``static``  — outer planets that never move
  * ``orbital`` — inner planets rotating with ``angular_velocity``
  * ``comet``   — non-orbital entities following an env-supplied path

Reports per-horizon hit rates per kind plus the overall RMSE for
comparison with ``test_summary.json``.

Run from the repo root:

    python scripts/validate_planet_extrapolation.py
    python scripts/validate_planet_extrapolation.py \\
        --run-dir data/encoder_runs_planet/20260429-194952 \\
        --split test
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agents.archive.transformer_v1.encoder.planet_encoder import PlanetEncoder  # noqa: E402
from agents.archive.transformer_v1.pretrain.planet_encoder import (  # noqa: E402
    StratifiedTrajectoryDecoder,
    TrajectoryDecoder,
)
from agents.archive.transformer_v1.featurizer import featurize_planets  # noqa: E402
from agents.archive.transformer_v1.featurizer.planet_featurizer import (  # noqa: E402
    ANCHOR_DXY_NORM,
    EXTRAP_HORIZONS,
    N_EXTRAP_HORIZONS,
    _is_orbiting,
    _planet_pos_at,
)


def _load(run_dir: Path) -> tuple[PlanetEncoder, TrajectoryDecoder, dict]:
    ckpt = torch.load(
        run_dir / "planet_encoder_best.pt", map_location="cpu", weights_only=False,
    )
    d_model = ckpt["config"]["d_model"]
    enc = PlanetEncoder(d_model=d_model)
    enc.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in ckpt["model"].items()
         if k.startswith("encoder.")},
        strict=True,
    )
    enc.eval()
    # Detect decoder kind from the state-dict prefix layout. Vanilla =
    # ``heads.extrap_trajectory.net.*``; stratified =
    # ``heads.extrap_trajectory.heads.*`` (a ModuleList of sub-MLPs).
    dec_keys = [
        k.removeprefix("heads.extrap_trajectory.")
        for k in ckpt["model"].keys()
        if k.startswith("heads.extrap_trajectory.")
    ]
    is_stratified = any(k.startswith("heads.") for k in dec_keys)
    if is_stratified:
        dec: torch.nn.Module = StratifiedTrajectoryDecoder(
            d_model=d_model, n_horizons=N_EXTRAP_HORIZONS,
        )
    else:
        dec = TrajectoryDecoder(d_model=d_model, n_horizons=N_EXTRAP_HORIZONS)
    dec.load_state_dict(
        {k.removeprefix("heads.extrap_trajectory."): v
         for k, v in ckpt["model"].items()
         if k.startswith("heads.extrap_trajectory.")},
        strict=True,
    )
    dec.eval()
    return enc, dec, ckpt


def _split_replays(manifest_path: Path, split: str, replay_root: Path) -> list[Path]:
    """Map manifest CSV names back to replay paths."""
    manifest = json.loads(manifest_path.read_text())
    out: list[Path] = []
    for csv_name in manifest[split]:
        stem = csv_name.removeprefix("planet_").removesuffix(".csv")
        for p in replay_root.rglob("*.json.gz"):
            if p.name.split(".")[0] == stem:
                out.append(p)
                break
    return out


def _collect_predictions(
    encoder: PlanetEncoder,
    decoder: TrajectoryDecoder,
    replays: list[Path],
    *,
    sample_every: int = 1,
) -> dict[str, np.ndarray]:
    """Run encoder+decoder on every (turn, entity) pair (or every
    ``sample_every``-th turn for speed), collect:

      err[i]    — Euclidean prediction error at horizon h_idx
      radius[i] — entity radius (so callers can apply the threshold)
      kind[i]   — 'static' / 'orbital' / 'comet' label

    Stacked across all replays, returned as parallel ndarrays.
    """
    err_rows: list[list[float]] = []
    valid_rows: list[list[bool]] = []
    radius_rows: list[float] = []
    kinds: list[str] = []

    for path in replays:
        with gzip.open(path, "rt") as fh:
            replay = json.load(fh)
        steps = replay.get("steps") or []
        if not steps:
            continue
        n_players = len(steps[0])

        for t in range(0, len(steps), sample_every):
            if t + max(EXTRAP_HORIZONS) >= len(steps):
                break
            step = steps[t]
            if not step:
                continue
            obs = step[0].get("observation") if isinstance(step[0], dict) else None
            if not obs:
                continue
            av = float(obs.get("angular_velocity") or 0.0)

            features, mask, records = featurize_planets(
                obs, learner_slot=0, num_players=n_players, max_entities=64,
            )
            n = int(mask.sum())
            if n == 0:
                continue
            with torch.no_grad():
                z = encoder(features[:n].unsqueeze(0)).squeeze(0)
                pred_flat = decoder(z)
            pred_dxy = (
                pred_flat.view(n, N_EXTRAP_HORIZONS, 2).numpy() * ANCHOR_DXY_NORM
            )

            for i, rec in enumerate(records[:n]):
                row_err = []
                row_valid = []
                for j, h in enumerate(EXTRAP_HORIZONS):
                    fut = _planet_pos_at(steps, t + h, rec.planet_id)
                    if fut is None:
                        row_err.append(0.0)
                        row_valid.append(False)
                    else:
                        px = rec.x + pred_dxy[i, j, 0]
                        py = rec.y + pred_dxy[i, j, 1]
                        e = float(np.hypot(px - fut[0], py - fut[1]))
                        row_err.append(e)
                        row_valid.append(True)
                err_rows.append(row_err)
                valid_rows.append(row_valid)
                radius_rows.append(float(rec.radius))
                if rec.is_comet:
                    kinds.append("comet")
                elif _is_orbiting(rec.x, rec.y, rec.radius, av):
                    kinds.append("orbital")
                else:
                    kinds.append("static")

    return {
        "err": np.asarray(err_rows, dtype=np.float64),         # (N, H)
        "valid": np.asarray(valid_rows, dtype=bool),            # (N, H)
        "radius": np.asarray(radius_rows, dtype=np.float64),   # (N,)
        "kind": np.asarray(kinds),                              # (N,)
    }


def _hit_table(data: dict[str, np.ndarray], multipliers: tuple[float, ...]) -> str:
    err = data["err"]
    valid = data["valid"]
    radius = data["radius"][:, None]    # (N, 1)
    kind = data["kind"]

    lines = []
    lines.append(
        f"\n{'horizon':>7s}  {'kind':<8s}  {'n':>6s}  "
        f"{'rmse':>6s}  "
        + "  ".join(f"hit≤{m:g}r" for m in multipliers)
    )
    lines.append("-" * (40 + 9 * len(multipliers)))

    for j, h in enumerate(EXTRAP_HORIZONS):
        for kn in ("static", "orbital", "comet", "all"):
            if kn == "all":
                sel = np.ones(len(kind), dtype=bool)
            else:
                sel = kind == kn
            v = valid[:, j] & sel
            n_valid = int(v.sum())
            if n_valid == 0:
                continue
            e = err[v, j]
            r = radius[v, 0]
            rmse = float(np.sqrt((e ** 2).mean()))
            hit_strs = []
            for m in multipliers:
                hit_rate = float((e <= m * r).mean())
                hit_strs.append(f"{hit_rate * 100:>6.1f}%")
            lines.append(
                f"  h={h:<3d}   {kn:<8s}  {n_valid:>6d}  "
                f"{rmse:>6.3f}  " + "  ".join(hit_strs)
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=sorted((REPO / "data" / "encoder_runs_planet").glob("*"))[-1],
    )
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Use every Nth turn (1 = all turns)")
    parser.add_argument(
        "--multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0],
        help="Pred is a 'hit' if error ≤ mult × radius. Pass several to see a sweep.",
    )
    args = parser.parse_args()

    print(f"[validate] run-dir: {args.run_dir.relative_to(REPO)}")
    encoder, decoder, ckpt = _load(args.run_dir)

    manifest_path = REPO / "data" / "encoders_planet" / "manifest.json"
    replays = _split_replays(manifest_path, args.split, REPO / "data" / "replays")
    print(f"[validate] split={args.split}  replays={len(replays)}  "
          f"sample_every={args.sample_every}")

    data = _collect_predictions(
        encoder, decoder, replays, sample_every=args.sample_every,
    )
    print(f"[validate] entities scored: {len(data['kind'])}  "
          f"({(data['kind'] == 'comet').sum()} comet, "
          f"{(data['kind'] == 'orbital').sum()} orbital, "
          f"{(data['kind'] == 'static').sum()} static)")

    print(_hit_table(data, tuple(args.multipliers)))

    # Compact summary: hit-rate at each multiplier averaged over all horizons.
    print("\nMean hit rate across all horizons (all entities):")
    err = data["err"]
    valid = data["valid"]
    radius = data["radius"][:, None]
    for m in args.multipliers:
        hits = (err <= m * radius) & valid
        rate = hits.sum() / max(1, valid.sum())
        print(f"  err ≤ {m:g} × radius :  {rate * 100:.2f}%")


if __name__ == "__main__":
    main()
