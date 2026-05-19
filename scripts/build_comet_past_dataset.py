"""Build a comet-only dataset whose features are the comet's FULL path
(35 slots covering path indices 0..34) plus per-slot validity, and
whose labels are the next 30 turns of future trajectory.

Why this shape: the user wants the model to always see the comet's
entire trajectory, informed (via per-slot ``valid`` flags) about which
slots are real path positions vs. padding. The model has to use the
full trajectory shape to predict where the comet is going next.

For each comet row in ``data/datasets/comet_only_40k``:

* 35 slots, each (``past_dx_tNN``, ``past_dy_tNN``, ``past_valid_tNN``)
  where the ``NN`` is the path index 1..35 (1-indexed for readability;
  internally slot 1 = path[0]). ``dx`` / ``dy`` are normalized
  displacements **from the current position** (so the current slot has
  dx=dy=0). ``valid`` is 1 if the comet's path has this index, 0
  otherwise (padding for shorter paths).
* 30 future slots, each (``extrap_dx_hNN``, ``extrap_dy_hNN``,
  ``extrap_mask_hNN``)
* meta: episode_id, turn, planet_id

Run::

    python -m scripts.build_comet_past_dataset
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.transformer_v2.featurizer.planet_featurizer import (  # noqa: E402
    ANCHOR_DXY_NORM,
    DIST_SUN_NORM,
    EXTRAP_HORIZONS,
    SCALAR_DIM,
    SPEED_BUCKET_EDGES,
    SUN_BUCKET_EDGES,
    _comet_future_xy,
    _distance_to_sun_bucket,
    _speed_bucket,
    featurize_planets,
)
from agents.transformer_v2.featurizer.fleet_featurizer import (  # noqa: E402
    BOARD, MAX_SPEED,
)
from agents.physics_utils import SUN_CX, SUN_CY  # noqa: E402
import math  # noqa: E402

# Scalar feature column names (matching planet specialist's f000..f017
# convention so the comet model can use the same 18 inputs).
SCALAR_FEAT_COLS = tuple(f"f{i:03d}" for i in range(SCALAR_DIM))

# Trajectory slots: 35 slots covering path indices 0..34. The slot
# stores the displacement from the current comet position to path[k],
# plus a valid flag (1 if the path has this index, 0 otherwise). Slot
# at the current path_index naturally has dx=dy=0.
PATH_SLOT_INDICES = tuple(range(0, 35))   # 0..34 — 35 absolute path indices
N_PAST = len(PATH_SLOT_INDICES)
PAST_CHANNELS = 3

SUBSET_IN = REPO / "data" / "datasets" / "comet_only_40k"
SUBSET_OUT = REPO / "data" / "datasets" / "comet_only_40k_past"
REPLAY_DIR = REPO / "data" / "replays"


def _uuid_to_replay() -> dict[str, Path]:
    """Build/load the cached UUID-to-replay map. Same logic as
    ``regen_planet_subset_20k.py``."""
    import pickle
    cache_path = Path("/tmp/orbit_uuid_to_replay.pkl")
    if cache_path.exists():
        try:
            data = pickle.loads(cache_path.read_bytes())
            data = {k: v for k, v in data.items() if v.exists()}
            if data:
                return data
        except Exception:
            pass
    out: dict[str, Path] = {}
    paths = list(REPLAY_DIR.rglob("*.json.gz"))
    print(f"  scanning {len(paths)} replay files to map UUIDs...", flush=True)
    for i, p in enumerate(paths):
        try:
            with gzip.open(p, "rb") as fh:
                replay = json.load(fh)
            rid = replay.get("id")
            if rid:
                out[rid] = p
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(paths)}", flush=True)
    cache_path.write_bytes(pickle.dumps(out))
    return out


def _seat_from_path(p: Path) -> int:
    parts = p.stem.removesuffix(".json").split("_")
    return int(parts[-1])


def _comet_meta_for_pid(obs: dict, target_pid: int) -> dict | None:
    """Find the comet's meta entry (paths + path_index) for the given
    planet id at this turn's obs."""
    raw_comets = obs.get("comets") or []
    for cm in raw_comets:
        pids = cm.get("planet_ids") or []
        for i, pid in enumerate(pids):
            if int(pid) == target_pid:
                out = dict(cm)
                out["_seat_index"] = i
                return out
    return None


def _comet_current_xy(obs: dict, target_pid: int) -> tuple[float, float] | None:
    """Find this comet's current (x, y) by scanning obs['planets']."""
    for p in obs.get("planets") or []:
        if int(p[0]) == target_pid:
            return float(p[2]), float(p[3])
    return None


def _build_split(
    src_csv: Path,
    out_csv: Path,
    uuid_to_replay: dict[str, Path],
    out_header: list[str],
) -> None:
    with src_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    print(f"\n{src_csv.name}: {len(rows)} rows")

    by_episode_turn: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r in rows:
        ep = r["episode_id"]
        t = int(r["turn"])
        pid = int(r["planet_id"])
        by_episode_turn[(ep, t)].append(pid)

    replay_cache: dict[str, dict] = {}
    n_written = 0
    n_skipped_no_replay = 0
    n_skipped_no_meta = 0
    t0 = time.time()
    with out_csv.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=out_header)
        writer.writeheader()
        for (ep, turn), pids in sorted(by_episode_turn.items()):
            replay_path = uuid_to_replay.get(ep)
            if replay_path is None:
                n_skipped_no_replay += 1
                continue
            if ep not in replay_cache:
                with gzip.open(replay_path, "rb") as fh:
                    replay_cache[ep] = json.load(fh)
                if len(replay_cache) > 30:
                    oldest = next(iter(replay_cache))
                    if oldest != ep:
                        replay_cache.pop(oldest)
            replay = replay_cache[ep]
            steps = replay["steps"]
            if turn >= len(steps):
                continue
            step = steps[turn]
            seat = _seat_from_path(replay_path)
            if not step or seat >= len(step):
                continue
            obs = step[seat].get("observation") if isinstance(
                step[seat], dict) else None
            if not obs:
                continue

            # Run the planet featurizer once per (ep, turn) so we can
            # read off the 18-dim scalar vector per comet (same
            # normalization the planet specialist uses).
            num_players = len(step)
            _, _, recs = featurize_planets(
                obs, learner_slot=seat, num_players=num_players,
                max_entities=64,
            )
            rec_by_pid = {r.planet_id: r for r in recs}

            for pid in pids:
                cur_xy = _comet_current_xy(obs, pid)
                meta = _comet_meta_for_pid(obs, pid)
                if cur_xy is None or meta is None:
                    n_skipped_no_meta += 1
                    continue
                cx, cy = cur_xy

                row: dict[str, str | int | float] = {
                    "episode_id": ep,
                    "turn": turn,
                    "planet_id": pid,
                }

                # 18 scalar features (same as planet specialist input).
                rec = rec_by_pid.get(pid)
                if rec is None:
                    n_skipped_no_meta += 1
                    continue
                full_vec = rec.to_vector(
                    learner_slot=seat, num_players=num_players,
                )
                for i, val in enumerate(full_vec[:SCALAR_DIM]):
                    row[SCALAR_FEAT_COLS[i]] = val

                # 35 slots covering path indices 0..34. dx/dy are
                # displacements from the current position to path[k],
                # normalized by ANCHOR_DXY_NORM. Slot at current
                # path_index has dx=dy=0. valid is 1 if k is a real
                # index in this comet's path, 0 if k >= len(path)
                # (padding for shorter paths).
                paths = meta["paths"]
                seat_idx = meta["_seat_index"]
                path = paths[seat_idx]
                path_len = len(path)
                for slot_i, k in enumerate(PATH_SLOT_INDICES, start=1):
                    col_dx = f"past_dx_t{slot_i:02d}"
                    col_dy = f"past_dy_t{slot_i:02d}"
                    col_v = f"past_valid_t{slot_i:02d}"
                    if 0 <= k < path_len:
                        px, py = path[k][0], path[k][1]
                        row[col_dx] = (px - cx) / ANCHOR_DXY_NORM
                        row[col_dy] = (py - cy) / ANCHOR_DXY_NORM
                        row[col_v] = 1
                    else:
                        row[col_dx] = 0.0
                        row[col_dy] = 0.0
                        row[col_v] = 0

                # Scalar labels (mirror planet featurizer's
                # ``to_label_dict``): use the comet's current x/y from
                # obs + velocity from the next path step. This matches
                # what the planet specialist sees.
                nxt = _comet_future_xy(meta, 1)
                if nxt is not None:
                    vx = nxt[0] - cx
                    vy = nxt[1] - cy
                else:
                    vx = vy = 0.0
                dist_sun = math.hypot(cx - SUN_CX, cy - SUN_CY)
                speed = math.hypot(vx, vy)
                row["distance_to_sun_bucket"] = _distance_to_sun_bucket(dist_sun)
                row["speed_bucket"] = _speed_bucket(speed)
                row["recon_x_norm"] = cx / BOARD
                row["recon_y_norm"] = cy / BOARD
                row["recon_vx_norm"] = vx / MAX_SPEED
                row["recon_vy_norm"] = vy / MAX_SPEED

                # Future labels: same convention as planet featurizer.
                # Walks the replay (not just the path) so it handles
                # comet death — if the entity isn't in obs[t+h]'s
                # planets list, mask is 0.
                for h in EXTRAP_HORIZONS:
                    fut_t = turn + h
                    fut_xy: tuple[float, float] | None = None
                    if 0 <= fut_t < len(steps):
                        fut_step = steps[fut_t]
                        if fut_step and seat < len(fut_step):
                            fut_obs = fut_step[seat].get("observation") if isinstance(
                                fut_step[seat], dict) else None
                            if fut_obs:
                                for p in fut_obs.get("planets") or []:
                                    if int(p[0]) == pid:
                                        fut_xy = (float(p[2]), float(p[3]))
                                        break
                    if fut_xy is None:
                        row[f"extrap_dx_h{h:02d}"] = 0.0
                        row[f"extrap_dy_h{h:02d}"] = 0.0
                        row[f"extrap_mask_h{h:02d}"] = 0
                    else:
                        row[f"extrap_dx_h{h:02d}"] = (fut_xy[0] - cx) / ANCHOR_DXY_NORM
                        row[f"extrap_dy_h{h:02d}"] = (fut_xy[1] - cy) / ANCHOR_DXY_NORM
                        row[f"extrap_mask_h{h:02d}"] = 1

                writer.writerow(row)
                n_written += 1
                if n_written % 5000 == 0:
                    print(f"  ... {n_written} rows ({time.time()-t0:.1f}s)",
                          flush=True)

    elapsed = time.time() - t0
    print(f"wrote {out_csv}: {n_written} rows in {elapsed:.1f}s "
          f"(skipped: {n_skipped_no_replay} no-replay, "
          f"{n_skipped_no_meta} no-meta)")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, default=SUBSET_IN)
    ap.add_argument("--out-dir", type=Path, default=SUBSET_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    uuid_to_replay = _uuid_to_replay()
    print(f"discovered {len(uuid_to_replay)} replay UUIDs")

    past_cols: list[str] = []
    for slot_i in range(1, N_PAST + 1):
        past_cols.append(f"past_dx_t{slot_i:02d}")
        past_cols.append(f"past_dy_t{slot_i:02d}")
        past_cols.append(f"past_valid_t{slot_i:02d}")
    extrap_target_cols: list[str] = []
    extrap_mask_cols: list[str] = []
    for h in EXTRAP_HORIZONS:
        extrap_target_cols.append(f"extrap_dx_h{h:02d}")
        extrap_target_cols.append(f"extrap_dy_h{h:02d}")
        extrap_mask_cols.append(f"extrap_mask_h{h:02d}")
    scalar_label_cols = [
        "distance_to_sun_bucket", "speed_bucket",
        "recon_x_norm", "recon_y_norm", "recon_vx_norm", "recon_vy_norm",
    ]
    out_header = (
        ["episode_id", "turn", "planet_id"]
        + list(SCALAR_FEAT_COLS)         # 18 scalar input features (f000..f017)
        + past_cols
        + scalar_label_cols
        + extrap_target_cols
        + extrap_mask_cols
    )

    for split in ("planet_train.csv", "planet_val.csv", "planet_test.csv"):
        src = args.src_dir / split
        if not src.exists():
            print(f"  skip: {src} missing")
            continue
        out = args.out_dir / split
        _build_split(src, out, uuid_to_replay, out_header)

    manifest = {
        "train": ["planet_train.csv"],
        "val":   ["planet_val.csv"],
        "test":  ["planet_test.csv"],
        "src":   str(args.src_dir),
        "n_past": N_PAST,
        "past_channels": PAST_CHANNELS,
        "n_extrap_horizons": len(EXTRAP_HORIZONS),
        "anchor_dxy_norm": ANCHOR_DXY_NORM,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
