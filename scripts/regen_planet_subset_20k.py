"""Re-featurize ONLY the rows of the existing 20k planet subset.

After bumping ``ANCHOR_DXY_NORM`` and extending ``EXTRAP_HORIZONS``,
the existing ``data/datasets/planet/`` CSVs and the
``data/datasets/planet_diverse_20k/`` subset are stale. A full source
regen takes ~25 min; this script only re-featurizes the ~20 k rows
already chosen in the diverse subset, by reading the existing subset's
``(episode_id, turn, planet_id)`` tuples and walking replays directly.

Output: ``data/datasets/planet_diverse_20k_v2/`` with the same three
splits (train/val/test) and manifest.json.

Time: ~2-3 min vs ~25 min for full source regen.
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

from agents.transformer_v2.featurizer.planet_featurizer import (
    ENCODER_LABEL_HEADS,
    ENCODER_PRETRAIN_LABELS,
    EXTRAP_HORIZONS,
    EXTRAP_MASK_COLS,
    EXTRAP_TARGET_COLS,
    PLANET_LABEL_FIELDS,
    PLANET_RAW_DIM,
    PlanetFeaturizer,
    featurize_planets,
    _comet_future_xy,
    _planet_future_xy,
    ANCHOR_DXY_NORM,
    SHIPS_LOG_MAX,
    signed_log1p,
)
import math


SUBSET_IN = REPO / "data" / "datasets" / "planet_diverse_20k"
SUBSET_OUT = REPO / "data" / "datasets" / "planet_diverse_20k_v2"
REPLAY_DIR = REPO / "data" / "replays"

LOOKAHEAD = 5  # owner_t_plus_5 horizon — matches featurizer default


def _uuid_to_replay() -> dict[str, Path]:
    """Build {replay['id']: path} for every replay. The CSV's
    ``episode_id`` column is the UUID, not the filename — we need this
    map to find the right replay file.

    Cache to a tmp pickle so we don't re-open every replay on each
    invocation (it's ~1 min to scan 1100+ files cold).
    """
    import pickle
    cache_path = Path("/tmp/orbit_uuid_to_replay.pkl")
    if cache_path.exists():
        try:
            data = pickle.loads(cache_path.read_bytes())
            # Stale-cache guard: drop entries whose path no longer exists.
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
    """Replay filename is ``<runid>_<num_players>_<seat>.json.gz``."""
    parts = p.stem.removesuffix(".json").split("_")
    return int(parts[-1])


def _featurize_turn(
    obs: dict, learner_slot: int, num_players: int,
) -> list[PlanetFeaturizer]:
    _, _, recs = featurize_planets(
        obs, learner_slot=learner_slot, num_players=num_players,
        max_entities=64,
    )
    return recs


def _replay_extrap_targets(
    replay: dict, seat: int, turn: int, rec: PlanetFeaturizer,
) -> tuple[list[float], list[int]]:
    """Compute (dx, dy, valid) at each horizon in ``EXTRAP_HORIZONS`` for
    the given entity, by walking the replay for true future positions.

    Returns flat list ``[dx_h1, dy_h1, ..., dx_hN, dy_hN]`` and per-step
    mask list ``[m_h1, ..., m_hN]``.
    """
    steps = replay["steps"]
    targets = []
    masks = []
    for h in EXTRAP_HORIZONS:
        fut_t = turn + h
        if fut_t >= len(steps):
            targets.extend([0.0, 0.0])
            masks.append(0)
            continue
        fut_step = steps[fut_t]
        if not fut_step or seat >= len(fut_step):
            targets.extend([0.0, 0.0])
            masks.append(0)
            continue
        fut_obs = fut_step[seat].get("observation") if isinstance(
            fut_step[seat], dict) else None
        if not fut_obs:
            targets.extend([0.0, 0.0])
            masks.append(0)
            continue
        # Find this planet in the future obs
        fut_planets = fut_obs.get("planets") or []
        fut_pos = None
        for p in fut_planets:
            if int(p[0]) == rec.planet_id:
                fut_pos = (float(p[2]), float(p[3]))
                break
        if fut_pos is None:
            targets.extend([0.0, 0.0])
            masks.append(0)
            continue
        dx = (fut_pos[0] - rec.x) / ANCHOR_DXY_NORM
        dy = (fut_pos[1] - rec.y) / ANCHOR_DXY_NORM
        targets.extend([dx, dy])
        masks.append(1)
    return targets, masks


def _aux_labels(
    replay: dict, seat: int, turn: int, rec: PlanetFeaturizer,
) -> dict[str, float | int]:
    """Compute the post-encoder aux labels (owner_t_plus_5,
    net_ships_t_plus_5_signed_log, valid_t_plus_5, owner_changes_in_5)."""
    steps = replay["steps"]
    cur_owner = rec.owner_id
    cur_ships = rec.ships

    fut_t = turn + LOOKAHEAD
    fut_owner = -2  # sentinel for "missing"
    fut_ships = 0
    valid = 0
    if fut_t < len(steps):
        fut_step = steps[fut_t]
        if fut_step and seat < len(fut_step):
            fut_obs = fut_step[seat].get("observation") if isinstance(
                fut_step[seat], dict) else None
            if fut_obs:
                for p in fut_obs.get("planets") or []:
                    if int(p[0]) == rec.planet_id:
                        fut_owner = int(p[1])
                        fut_ships = int(p[6])
                        valid = 1
                        break

    # Detect ownership change within [turn+1, turn+LOOKAHEAD]
    changed = 0
    if valid:
        prev_owner = cur_owner
        for t in range(turn + 1, fut_t + 1):
            if t >= len(steps):
                break
            step = steps[t]
            if not step or seat >= len(step):
                continue
            obs = step[seat].get("observation") if isinstance(step[seat], dict) else None
            if not obs:
                continue
            for p in obs.get("planets") or []:
                if int(p[0]) == rec.planet_id:
                    o = int(p[1])
                    if o != prev_owner:
                        changed = 1
                    prev_owner = o
                    break

    net_diff = (fut_ships - cur_ships) if valid else 0
    return {
        "owner_t_plus_5": fut_owner if valid else cur_owner,
        "owner_changes_in_5": changed,
        "net_ships_t_plus_5_signed_log": signed_log1p(net_diff),
        "valid_t_plus_5": valid,
    }


def _emit_row(
    writer: csv.DictWriter,
    rec: PlanetFeaturizer,
    episode_id: str,
    turn: int,
    learner_slot: int,
    num_players: int,
    replay: dict,
    seat: int,
) -> None:
    feats = rec.to_vector(learner_slot=learner_slot, num_players=num_players)
    labels = rec.to_label_dict()
    extrap_targets, extrap_masks = _replay_extrap_targets(
        replay, seat, turn, rec,
    )
    aux = _aux_labels(replay, seat, turn, rec)

    row: dict[str, str | int | float] = {
        "episode_id": episode_id,
        "turn": turn,
    }
    for i, v in enumerate(feats):
        row[f"f{i:03d}"] = v
    # Encoder pretrain + aux labels (non-trajectory).
    for name in PLANET_LABEL_FIELDS:
        if name in labels:
            row[name] = labels[name]
        elif name in aux:
            row[name] = aux[name]
    # Trajectory targets + masks
    for col, val in zip(EXTRAP_TARGET_COLS, extrap_targets):
        row[col] = val
    for col, val in zip(EXTRAP_MASK_COLS, extrap_masks):
        row[col] = val
    # Debug ints
    row["planet_id"] = rec.planet_id
    row["is_comet"] = int(rec.is_comet)
    row["owner_id"] = rec.owner_id
    row["ships"] = rec.ships
    writer.writerow(row)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, default=SUBSET_IN)
    ap.add_argument("--out-dir", type=Path, default=SUBSET_OUT)
    args = ap.parse_args()

    uuid_to_replay = _uuid_to_replay()
    print(f"discovered {len(uuid_to_replay)} replay UUIDs")

    src_in = args.src_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"regen: {src_in} -> {out_dir}")

    # Each split processed independently
    for split_name in ("planet_train.csv", "planet_val.csv", "planet_test.csv"):
        src = src_in / split_name
        if not src.exists():
            print(f"  skip: {src} missing")
            continue
        with src.open() as fh:
            rows = list(csv.DictReader(fh))
        # Group by (episode_id, turn)
        by_episode_turn: dict[tuple[str, int], list[int]] = defaultdict(list)
        for r in rows:
            ep = r["episode_id"]
            t = int(r["turn"])
            pid = int(r["planet_id"])
            by_episode_turn[(ep, t)].append(pid)
        print(f"\n{split_name}: {len(rows)} rows, "
              f"{len(by_episode_turn)} unique (episode, turn) pairs")

        # Output header: same as source minus old extrap cols, plus new
        # extrap cols (which is just whatever EXTRAP_TARGET_COLS expands
        # to under the new EXTRAP_HORIZONS).
        feat_cols = [f"f{i:03d}" for i in range(PLANET_RAW_DIM)]
        out_header = (
            ["episode_id", "turn"]
            + feat_cols
            + list(PLANET_LABEL_FIELDS)
            + list(EXTRAP_TARGET_COLS)
            + list(EXTRAP_MASK_COLS)
            + ["planet_id", "is_comet", "owner_id", "ships"]
        )

        out_path = out_dir / split_name
        replay_cache: dict[str, dict] = {}
        n_written = 0
        t0 = time.time()
        with out_path.open("w", newline="") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=out_header)
            writer.writeheader()
            for (ep, turn), pids in sorted(by_episode_turn.items()):
                # episode_id in the CSV IS the replay UUID
                replay_path = uuid_to_replay.get(ep)
                if replay_path is None:
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
                obs = step[seat].get("observation") if isinstance(step[seat], dict) else None
                if not obs:
                    continue
                num_players = len(step)
                recs = _featurize_turn(obs, learner_slot=seat, num_players=num_players)
                rec_by_pid = {r.planet_id: r for r in recs}
                for pid in pids:
                    rec = rec_by_pid.get(pid)
                    if rec is None:
                        continue
                    _emit_row(
                        writer, rec, ep, turn, seat, num_players, replay, seat,
                    )
                    n_written += 1
                if n_written % 2000 == 0 and n_written > 0:
                    print(
                        f"  ... {n_written} rows written ({time.time()-t0:.1f}s)",
                        flush=True,
                    )
        elapsed = time.time() - t0
        print(f"wrote {out_path}: {n_written} rows in {elapsed:.1f}s")

    # Write manifest mirroring the source
    manifest = {
        "train": ["planet_train.csv"],
        "val":   ["planet_val.csv"],
        "test":  ["planet_test.csv"],
        "src":   "regenerated from replays via scripts/regen_planet_subset_20k.py",
        "anchor_dxy_norm": ANCHOR_DXY_NORM,
        "n_extrap_horizons": len(EXTRAP_HORIZONS),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
