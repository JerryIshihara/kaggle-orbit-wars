"""Build per-stem comet feature CSVs aligned to the entity dataset.

For each ``entity_<stem>.csv`` in ``data/datasets/entity/`` (filtered to
stems whose ``planet_<stem>.csv`` and ``fleet_<stem>.csv`` exist too,
matching the train-time loader), opens the matching replay and writes
``data/datasets/entity_comet/comet_<stem>.csv`` containing the
123-dim comet input vector for every ``is_comet=1`` row.

Layout per row (matches the comet specialist's training CSV):
  meta:    ``episode_id``, ``turn``, ``planet_id``
  scalars: ``f000..f017``                          — 18 dims
  path:    ``past_dx_tNN``, ``past_dy_tNN``,
           ``past_valid_tNN`` for ``NN ∈ {01..35}`` — 105 dims

The 18 scalars are the first 18 components of
``featurize_planets(...).to_vector()`` (same normalization the comet
specialist saw). The 35 path slots store displacements
``(path[k] - current_xy) / ANCHOR_DXY_NORM`` with ``valid=1`` when ``k``
is in range of the comet's path, ``valid=0`` for padding past
``len(path)``.

Run::

    python scripts/build_entity_comet_features.py
    python scripts/build_entity_comet_features.py --limit-stems 5    # dry run

Outputs cover 100% of is_comet=1 rows in the viable entity stems — the
runtime fallback in ``EntitySnapshotDataset`` (scalar-only zero-path)
only triggers for stems that don't have a per-stem CSV here.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.physics_utils import SUN_CX, SUN_CY  # noqa: E402
from agents.transformer_v2.featurizer.fleet_featurizer import (  # noqa: E402
    BOARD, MAX_SPEED,
)
from agents.transformer_v2.featurizer.planet_featurizer import (  # noqa: E402
    ANCHOR_DXY_NORM,
    SCALAR_DIM,
    _comet_future_xy,
    _distance_to_sun_bucket,
    _speed_bucket,
    featurize_planets,
)
from agents.transformer_v2.paths import (  # noqa: E402
    DATASETS_ROOT, ENTITY_DATASET_DIR, FLEET_DATASET_DIR, PLANET_DATASET_DIR,
    REPLAYS_DIR,
)


N_PAST = 35
PATH_SLOT_INDICES = tuple(range(0, N_PAST))
SCALAR_FEAT_COLS = tuple(f"f{i:03d}" for i in range(SCALAR_DIM))


def _replay_path_for_stem(stem: str) -> Path | None:
    """Find ``data/replays/**/{stem}.json.gz`` (first match)."""
    matches = list(REPLAYS_DIR.rglob(f"{stem}.json.gz"))
    return matches[0] if matches else None


def _viable_stems() -> list[str]:
    """Stems whose entity/planet/fleet CSVs all exist on disk (mirrors
    the train-time filter in ``EntitySnapshotDataset``).
    """
    manifest = json.loads((ENTITY_DATASET_DIR / "manifest.json").read_text())
    stems: list[str] = []
    for split in ("train", "val", "test"):
        stems.extend(
            n.removeprefix("entity_").removesuffix(".csv") for n in manifest[split]
        )
    viable = []
    for s in sorted(set(stems)):
        if (
            (ENTITY_DATASET_DIR / f"entity_{s}.csv").exists()
            and (PLANET_DATASET_DIR / f"planet_{s}.csv").exists()
            and (FLEET_DATASET_DIR / f"fleet_{s}.csv").exists()
        ):
            viable.append(s)
    return viable


def _comet_meta_for_pid(obs: dict, target_pid: int) -> dict | None:
    """Find the comet entry (with paths + seat_index) for ``target_pid``
    in this turn's obs."""
    for cm in obs.get("comets") or []:
        pids = cm.get("planet_ids") or []
        for i, pid in enumerate(pids):
            if int(pid) == target_pid:
                out = dict(cm)
                out["_seat_index"] = i
                return out
    return None


def _comet_current_xy(obs: dict, target_pid: int) -> tuple[float, float] | None:
    for p in obs.get("planets") or []:
        if int(p[0]) == target_pid:
            return float(p[2]), float(p[3])
    return None


def _seat_from_stem(stem: str) -> int:
    return int(stem.split("_")[-1])


def _build_header() -> list[str]:
    cols = ["episode_id", "turn", "planet_id"]
    cols.extend(SCALAR_FEAT_COLS)
    for k in range(1, N_PAST + 1):
        cols.append(f"past_dx_t{k:02d}")
        cols.append(f"past_dy_t{k:02d}")
        cols.append(f"past_valid_t{k:02d}")
    return cols


def _comet_rows_by_turn(entity_csv: Path) -> dict[int, list[int]]:
    """Map ``turn → [planet_id ...]`` for ``is_comet=1`` rows."""
    out: dict[int, list[int]] = defaultdict(list)
    with entity_csv.open() as fh:
        for row in csv.DictReader(fh):
            if int(row.get("is_comet", 0)) == 1:
                out[int(row["turn"])].append(int(row["planet_id"]))
    return out


def _process_stem(
    stem: str, out_dir: Path, header: list[str], *, overwrite: bool = False,
) -> tuple[int, int]:
    """Returns ``(rows_written, missing_comets)``."""
    out_csv = out_dir / f"comet_{stem}.csv"
    if out_csv.exists() and not overwrite:
        # Count rows for the summary without re-reading the CSV body.
        n = sum(1 for _ in out_csv.open()) - 1
        return max(0, n), 0

    replay_path = _replay_path_for_stem(stem)
    if replay_path is None:
        print(f"  skip {stem}: no replay file found", flush=True)
        return 0, 0
    seat = _seat_from_stem(stem)
    entity_csv = ENTITY_DATASET_DIR / f"entity_{stem}.csv"
    by_turn = _comet_rows_by_turn(entity_csv)
    if not by_turn:
        # No comet rows in this stem; emit an empty CSV (just the header)
        # so downstream "exists?" checks behave consistently.
        with out_csv.open("w", newline="") as out_fh:
            csv.DictWriter(out_fh, fieldnames=header).writeheader()
        return 0, 0

    with gzip.open(replay_path, "rb") as fh:
        replay = json.load(fh)
    steps = replay["steps"]
    episode_id = replay.get("id", "")
    n_players = len(steps[0]) if steps else 0
    n_written = 0
    n_missing = 0

    with out_csv.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=header)
        writer.writeheader()
        for turn, pids in sorted(by_turn.items()):
            if turn >= len(steps):
                n_missing += len(pids)
                continue
            step = steps[turn]
            if not step or seat >= len(step):
                n_missing += len(pids)
                continue
            obs = step[seat].get("observation") if isinstance(
                step[seat], dict) else None
            if not obs:
                n_missing += len(pids)
                continue

            # Run featurizer once per turn; reuse across all pids that
            # appear at this turn. ``rec.to_vector(...)[:SCALAR_DIM]`` is
            # the 18-scalar block the comet specialist consumes.
            _, _, recs = featurize_planets(
                obs, learner_slot=seat, num_players=n_players, max_entities=64,
            )
            rec_by_pid = {r.planet_id: r for r in recs}

            for pid in pids:
                rec = rec_by_pid.get(pid)
                cur_xy = _comet_current_xy(obs, pid)
                meta = _comet_meta_for_pid(obs, pid)
                if rec is None or cur_xy is None or meta is None:
                    n_missing += 1
                    continue
                cx, cy = cur_xy
                row: dict[str, str | int | float] = {
                    "episode_id": episode_id,
                    "turn": turn,
                    "planet_id": pid,
                }
                full_vec = rec.to_vector(
                    learner_slot=seat, num_players=n_players,
                )
                for i, val in enumerate(full_vec[:SCALAR_DIM]):
                    row[SCALAR_FEAT_COLS[i]] = val

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
                writer.writerow(row)
                n_written += 1
    return n_written, n_missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path,
        default=DATASETS_ROOT / "entity_comet",
        help="Where to write comet_<stem>.csv files.",
    )
    ap.add_argument(
        "--limit-stems", type=int, default=None,
        help="If set, only process the first N viable stems (smoke test).",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Re-emit comet_<stem>.csv even when it already exists.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    header = _build_header()
    stems = _viable_stems()
    if args.limit_stems is not None:
        stems = stems[:args.limit_stems]
    print(f"[entity-comet] {len(stems)} viable stems → {args.out_dir}")

    t0 = time.time()
    total_written = 0
    total_missing = 0
    log_every = max(1, len(stems) // 20)
    for i, stem in enumerate(stems, 1):
        n, m = _process_stem(stem, args.out_dir, header,
                             overwrite=args.overwrite)
        total_written += n
        total_missing += m
        if i % log_every == 0 or i == len(stems):
            elapsed = time.time() - t0
            print(
                f"  [{i}/{len(stems)}] {stem}: +{n} rows  "
                f"(running total: {total_written:,} rows, "
                f"{total_missing:,} missing, {elapsed:.1f}s)",
                flush=True,
            )
    print(
        f"\n[entity-comet] done: {total_written:,} rows across {len(stems)} "
        f"stems  ({total_missing:,} comet rows had no obs/meta match)"
    )


if __name__ == "__main__":
    main()
