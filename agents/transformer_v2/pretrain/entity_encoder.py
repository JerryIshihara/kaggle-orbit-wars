"""Current transformer_v2 entity / pair pretraining.

The active training path freezes three L0 specialist encoders
(:class:`PlanetEncoder`, :class:`CometEncoder`, and
:class:`FleetEncoder`) and trains the L1→L4 perception / role stack plus
a single :class:`PairHead` on expert source→target pair-set labels.

The on-disk pair cache stores raw snapshot tensors plus current-turn
``pair_labels`` / ``pair_valid`` matrices. At train time we run the
frozen L0 encoders under ``no_grad``, hard-route planet-vs-comet slots
with ``torch.where(is_comet, comet_tok, planet_tok)``, then train:

``PlanetEntityEncoder → CrossEntityAttention → DualRoleAttention →
JointRoleAttention → PairHead``.

The current cache is history-aware: ``CachedPairDataset`` stacks the
T=6 window ``(t-5, t-4, t-3, t-2, t-1, t)`` at ``__getitem__`` time,
while labels stay current-turn-only.

Run from the repo root:

    python -m agents.transformer_v2.pretrain.entity_encoder \\
        --epochs 30 --batch-size 32

Outputs (under ``data/runs/entity/<timestamp>/``):
  * ``entity_encoder_best.pt`` / ``entity_encoder_last.pt``
  * ``log.json`` — per-epoch train + val pair metrics
  * ``test_summary.json`` — pair metrics from the best ckpt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..aggregator import (
    CrossEntityAttention,
    DualRoleAttention,
    JointRoleAttention,
    PairHead,
)
from ..encoder.entity_encoder import PlanetEntityEncoder
from ..encoder.fleet_encoder import FleetEncoder
from ..encoder.planet_encoder import PlanetEncoder
from ..featurizer import (
    ENTITY_ARRIVAL_HORIZONS,
    ENTITY_LABEL_HORIZONS,
    ENTITY_NUM_OWNER_SLOTS,
    ENTITY_N_OWNER_CLASSES,
    FLEET_RAW_DIM,
    PLANET_RAW_DIM,
)
from ..featurizer.fleet_featurizer import ETA_NORM, SHIPS_LOG_MAX
from ..paths import (
    DATASETS_ROOT,
    ENTITY_DATASET_DIR,
    ENTITY_RUNS_DIR,
    FLEET_DATASET_DIR,
    FLEET_RUNS_DIR,
    PLANET_DATASET_DIR,
    PLANET_RUNS_DIR,
    RUNS_ROOT,
)
from .comet_past_encoder import (
    CometEncoder,
    CometPastModel,
    INPUT_DIM as COMET_INPUT_DIM,
    N_PAST as COMET_N_PAST,
    PAST_COLS as COMET_PAST_COLS,
    SCALAR_FEAT_COLS as COMET_SCALAR_COLS,
    SCALAR_FEAT_DIM as COMET_SCALAR_DIM,
    _remap_legacy_comet_state_dict,
)


# ---------- Dataset ----------
PLANET_FEATURE_COLS = tuple(f"f{i:03d}" for i in range(PLANET_RAW_DIM))
FLEET_FEATURE_COLS = tuple(f"f{i:03d}" for i in range(FLEET_RAW_DIM))

# Comet feature sources, in preference order:
#
#  1. ``COMET_PER_STEM_DIR/comet_<stem>.csv`` — per-stem CSVs produced by
#     ``scripts/build_entity_comet_features.py``. Aim is 100% coverage
#     of ``is_comet=1`` rows in the entity dataset, with the 123-dim
#     layout the comet specialist saw at pretrain time.
#  2. ``COMET_LOOKUP_DIR`` — global ``(uuid, turn, pid) → 123-dim`` map
#     pulled from the comet specialist's training split. ~48% coverage
#     of the entity dataset's comet rows; only used as backfill when
#     the per-stem CSV is missing.
#  3. Runtime fallback — copy the first 18 scalars from the planet CSV
#     (cols ``f000..f017`` share the same normalization), leave the 35
#     path slots zeroed with ``valid=0``. Out-of-distribution for the
#     comet encoder but numerically safe.
COMET_PER_STEM_DIR: Path = DATASETS_ROOT / "entity_comet"
COMET_LOOKUP_DIR: Path = DATASETS_ROOT / "comet_only_40k_fullpath_scalar"


def _load_csv_grouped_by_turn(
    path: Path,
) -> dict[tuple[str, int], list[dict[str, str]]]:
    """Load a featurizer CSV, grouped by ``(episode_id, turn)``."""
    out: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["episode_id"], int(row["turn"]))
            out[key].append(row)
    return out


def load_comet_lookup(
    lookup_dir: Path = COMET_LOOKUP_DIR,
) -> dict[tuple[str, int, int], list[float]]:
    """Build ``(episode_id, turn, planet_id) → 123-dim feature vector``.

    The comet specialist's training CSVs already store the 18-scalar +
    35×(dx, dy, valid) layout per comet row, keyed by full episode UUID.
    The entity dataset uses the same UUIDs in its ``episode_id`` column,
    so a flat dict over the union of {train, val, test} is enough to
    join on (episode, turn, planet_id) at snapshot build time.
    """
    out: dict[tuple[str, int, int], list[float]] = {}
    if not lookup_dir.exists():
        return out
    for split in ("planet_train.csv", "planet_val.csv", "planet_test.csv"):
        p = lookup_dir / split
        if not p.exists():
            continue
        with p.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = (row["episode_id"], int(row["turn"]), int(row["planet_id"]))
                vec = [float(row[c]) for c in COMET_SCALAR_COLS]
                vec.extend(float(row[c]) for c in COMET_PAST_COLS)
                out[key] = vec
    return out


def merge_per_stem_comet_csvs(
    lookup: dict[tuple[str, int, int], list[float]],
    per_stem_dir: Path,
    stems: list[str],
) -> int:
    """Overlay ``comet_<stem>.csv`` rows on top of ``lookup``.

    Per-stem CSVs are emitted by ``scripts/build_entity_comet_features.py``
    and aim for 100% coverage of the entity dataset's ``is_comet=1``
    rows; entries here win over the global lookup so callers can
    transition smoothly from partial-coverage backfill to full-coverage
    per-stem inputs without changing the dataset class.

    Returns the number of new ``(episode_id, turn, planet_id)`` keys
    added to the lookup (existing keys are also overwritten but not
    counted as additions).
    """
    added = 0
    if not per_stem_dir.exists():
        return 0
    for stem in stems:
        p = per_stem_dir / f"comet_{stem}.csv"
        if not p.exists():
            continue
        with p.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = (row["episode_id"], int(row["turn"]), int(row["planet_id"]))
                vec = [float(row[c]) for c in COMET_SCALAR_COLS]
                vec.extend(float(row[c]) for c in COMET_PAST_COLS)
                if key not in lookup:
                    added += 1
                lookup[key] = vec
    return added


def _parse_stem_bundle(
    paths: dict[str, Path],
) -> dict[str, dict[tuple[str, int], list[dict[str, str]]]]:
    """Worker entry point: parse all CSVs for a single stem in one go.

    Top-level so it pickles cleanly for ``ProcessPoolExecutor``. Returns
    ``{kind: csv_grouped_by_turn}`` for each path supplied. Caller picks
    which kinds to bundle (the base class needs planet/fleet/entity;
    children may add cross_entity, etc.).
    """
    return {kind: _load_csv_grouped_by_turn(p) for kind, p in paths.items()}


#: Default dense T=6 window (current + 5 past, oldest first per the
#: step-embedding convention). Override via the ``history_offsets``
#: kwarg if you want a sparse / longer window. ``None`` (the default)
#: gives single-frame snapshots, matching the historical behavior.
HISTORY_OFFSETS_T6: tuple[int, ...] = (5, 4, 3, 2, 1, 0)


class EntitySnapshotDataset(Dataset):
    """Per-snapshot dataset. One item = one ``(episode, turn)``.

    Rows from three CSVs are joined:
      * planet CSV — raw planet features (PLANET_RAW_DIM cols), planet_id
      * fleet CSV — raw fleet features (FLEET_RAW_DIM cols), source/target
        planet ids, owner_id, ships
      * entity CSV — per-(planet) labels and per-(planet, player) labels

    Output dict per snapshot (all tensors padded to fixed sizes):
      * ``planet_features`` (max_planets, PLANET_RAW_DIM)
      * ``planet_mask`` (max_planets,) bool
      * ``comet_features`` (max_planets, COMET_INPUT_DIM)
      * ``is_comet`` (max_planets,) bool
      * ``fleet_features`` (max_fleets, FLEET_RAW_DIM)
      * ``fleet_mask`` (max_fleets,) bool
      * ``fleet_target_idx`` (max_fleets,) long, -1 = no target
      * ``fleet_source_idx`` (max_fleets,) long, -1 = unknown
      * ``fleet_owner_slot`` (max_fleets,) long
      * ``fleet_ships_log`` (max_fleets,) float
      * ``fleet_eta_norm`` (max_fleets,) float
      * label tensors (per-planet and per-(planet, player)):
          - earliest_arrival_owner_slot, is_source, is_target
          - owner_t_plus_K, log_ships_t_plus_K, valid_t_plus_K (each K)
          - ships_arriving_within_K (per player)

    When ``history_offsets`` is set, every input tensor listed in
    ``_STACK_KEYS`` gains a leading ``T`` axis (oldest first); missing
    past frames are zero-filled and their ``planet_mask`` /
    ``fleet_mask`` stays all-False so attention ignores them. Labels
    stay current-turn-only — we only supervise predictions for ``t``.
    """

    # Input tensors that get history-stacked along a new ``T`` axis
    # when ``history_offsets`` is set. Labels and routing-only-at-t
    # fields are left at the current turn.
    _STACK_KEYS: tuple[str, ...] = (
        "planet_features", "planet_mask",
        "comet_features", "is_comet",
        "fleet_features", "fleet_mask",
        "fleet_target_idx", "fleet_source_idx",
        "fleet_owner_slot", "fleet_ships_log", "fleet_eta_norm",
    )

    def __init__(
        self,
        planet_csv_paths: list[Path],
        fleet_csv_paths: list[Path],
        entity_csv_paths: list[Path],
        *,
        max_planets: int = 64,
        max_fleets: int = 1024,
        learner_slot: int = 0,
        num_players: int = 4,
        num_load_workers: int | None = None,
        comet_lookup: dict[tuple[str, int, int], list[float]] | None = None,
        history_offsets: tuple[int, ...] | None = None,
    ):
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.learner_slot = learner_slot
        self.num_players = num_players
        self.comet_lookup = comet_lookup or {}
        # When set, ``__getitem__`` stacks ``_STACK_KEYS`` along a new
        # leading ``T`` axis using these (oldest-first) offsets from the
        # current turn. Default ``None`` keeps the original single-frame
        # behavior so old callers don't change.
        self.history_offsets: tuple[int, ...] | None = (
            tuple(history_offsets) if history_offsets is not None else None
        )
        # Per-dataset coverage stats; populated as snapshots are built.
        # ``hits`` = comet rows resolved against ``comet_lookup``;
        # ``misses`` = is_comet=1 rows we had to fall back to scalar-only.
        self._comet_hits = 0
        self._comet_misses = 0

        # ---- Stream CSVs one stem at a time, with parallel prefetch. ----
        # Per-stem streaming caps peak CSV-dict memory at ~5 MB instead
        # of ~5 GB (whole corpus). A ``ProcessPoolExecutor`` prefetches
        # the upcoming stems' CSVs in the background so the main process
        # tensorizes back-to-back without idle gaps. With 8 workers the
        # full 600+ episode corpus loads in ~1-2 min instead of ~10+.
        def _index(paths: list[Path], prefix: str) -> dict[str, Path]:
            return {p.stem.removeprefix(prefix): p for p in paths}

        planet_paths = _index(planet_csv_paths, "planet_")
        fleet_paths = _index(fleet_csv_paths, "fleet_")
        entity_paths = _index(entity_csv_paths, "entity_")

        # Children may require additional CSV types per stem (cross_entity,
        # action). Hook lets them filter the stem set down + supply extra
        # paths to bundle into the worker call.
        common_stems = sorted(
            set(planet_paths) & set(fleet_paths) & set(entity_paths)
        )
        # Existence filter: the manifest references stems by entity CSV
        # name, but planet/fleet CSVs may have been generated on a
        # narrower episode set (e.g., when the planet pipeline was
        # re-run for fewer replays). Silently constructing missing
        # paths and discovering the gap at parse time would mid-train
        # crash; drop them up front with a loud summary instead.
        before = len(common_stems)
        viable = [
            s for s in common_stems
            if planet_paths[s].exists()
            and fleet_paths[s].exists()
            and entity_paths[s].exists()
        ]
        dropped = before - len(viable)
        if dropped:
            print(
                f"    [stems] dropping {dropped}/{before} stems whose "
                f"planet/fleet/entity CSV is missing on disk; "
                f"{len(viable)} viable",
                flush=True,
            )
        common_stems = self._filter_stems(viable)

        # Default to serial: in benchmarks ProcessPool was ~5% slower
        # because pickling the parsed dict-of-rows back from each worker
        # cost more than the parsing it saved. Users with very fast
        # disks can opt-in via ``--num-load-workers``.
        if num_load_workers is None:
            num_load_workers = 1
        n_stems = len(common_stems)

        # Build a list of {kind: path} bundles for each stem. Children
        # extend ``_extra_paths_for_stem(stem)`` to include their CSVs.
        bundles = [
            {
                "planet": planet_paths[stem],
                "fleet": fleet_paths[stem],
                "entity": entity_paths[stem],
                **self._extra_paths_for_stem(stem),
            }
            for stem in common_stems
        ]

        self.keys: list[tuple[str, int]] = []
        self.snapshots: list[dict[str, torch.Tensor]] = []
        log_every = max(1, n_stems // 20)
        t_start = time.time()

        def _consume(stem: str, parsed: dict[str, dict]) -> None:
            self._load_extra_csv_for_stem(stem, parsed)
            planet_rows = parsed["planet"]
            fleet_rows = parsed["fleet"]
            entity_rows = parsed["entity"]
            for key in sorted(set(planet_rows) & set(entity_rows)):
                self.snapshots.append(
                    self._build_snapshot(
                        key,
                        planet_rows[key],
                        fleet_rows.get(key, []),
                        entity_rows[key],
                    )
                )
                self.keys.append(key)

        if num_load_workers > 1 and n_stems > 1:
            print(
                f"    [stems] parsing in parallel ({num_load_workers} workers)",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=num_load_workers) as pool:
                # ``map`` preserves order; results stream back as workers
                # finish. ``chunksize=1`` keeps memory bounded since each
                # task's return is small.
                for i, parsed in enumerate(
                    pool.map(_parse_stem_bundle, bundles, chunksize=1)
                ):
                    _consume(common_stems[i], parsed)
                    del parsed
                    if (i + 1) % log_every == 0 or (i + 1) == n_stems:
                        elapsed = time.time() - t_start
                        rate = (i + 1) / max(elapsed, 1e-3)
                        print(
                            f"    [stems] {i + 1}/{n_stems} "
                            f"({100 * (i + 1) / n_stems:.0f}%, "
                            f"snapshots={len(self.snapshots)}, "
                            f"{rate:.1f} stems/s)",
                            flush=True,
                        )
        else:
            for i, (stem, paths) in enumerate(zip(common_stems, bundles)):
                _consume(stem, _parse_stem_bundle(paths))
                if (i + 1) % log_every == 0 or (i + 1) == n_stems:
                    print(
                        f"    [stems] {i + 1}/{n_stems} "
                        f"({100 * (i + 1) / n_stems:.0f}%, "
                        f"snapshots={len(self.snapshots)})",
                        flush=True,
                    )

    def _filter_stems(self, stems: list[str]) -> list[str]:
        """Subclass hook: narrow the stem set if children require
        additional CSV types beyond planet/fleet/entity. Default no-op.
        """
        return stems

    def _extra_paths_for_stem(self, stem: str) -> dict[str, Path]:
        """Subclass hook: supply additional ``{kind: path}`` entries to
        be parsed by the worker for this stem. Default empty.
        """
        return {}

    def _load_extra_csv_for_stem(
        self,
        stem: str,
        parsed: dict[str, dict[tuple[str, int], list[dict[str, str]]]],
    ) -> None:
        """Subclass hook: refresh per-stem CSV state used by
        :meth:`_build_snapshot`. Receives the worker's ``parsed`` bundle
        so children can pull their own kind out without re-reading the
        file. Default no-op.
        """
        return None

    def _build_snapshot(
        self,
        key: tuple[str, int],
        planet_rows: list[dict[str, str]],
        fleet_rows: list[dict[str, str]],
        entity_rows: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        """Materialize one ``(episode, turn)`` slice into padded tensors.

        Subclasses can override / extend to add more label tensors;
        :class:`CrossEntitySnapshotDataset` does this to mix in the
        Tier-1 / Tier-3 cross-entity labels.
        """
        P = self.max_planets
        F = self.max_fleets
        episode_id, turn = key

        # planet_id → row index. Entity / cross-entity rows look planets
        # up by id; their order in the CSV may differ from planet CSV.
        pid_to_idx: dict[int, int] = {}
        for i, row in enumerate(planet_rows[:P]):
            pid_to_idx[int(row["planet_id"])] = i

        # Planet features — build a flat list, materialize as ONE
        # tensor() call. PyTorch tensor scalar-assignment in a Python
        # loop costs ~30 us per element due to dispatch overhead; for
        # 64 planets × 13 features × 5K snapshots × 600 stems that adds
        # up to ~30 min. The list-build → tensor cast is ~30x faster.
        planet_features = torch.zeros(P, PLANET_RAW_DIM, dtype=torch.float32)
        planet_mask = torch.zeros(P, dtype=torch.bool)
        n_real_p = min(len(planet_rows), P)
        if n_real_p > 0:
            # ``row.get(col, 0.0)`` zero-pads when the CSV was generated
            # by an older featurizer that emitted fewer feature columns
            # than the current ``PLANET_RAW_DIM``. Safe for any encoder
            # that ignores the missing trailing slice (e.g. the
            # ``use_traj_branch=False`` planet specialist only consumes
            # the first ``SCALAR_DIM`` dims).
            flat = [
                float(row.get(col, 0.0) or 0.0)
                for row in planet_rows[:n_real_p]
                for col in PLANET_FEATURE_COLS
            ]
            planet_features[:n_real_p] = torch.tensor(
                flat, dtype=torch.float32,
            ).view(n_real_p, PLANET_RAW_DIM)
            planet_mask[:n_real_p] = True

        # Comet features (per row that is_comet=1): 18 scalars + 35×3
        # path slots. is_comet=0 slots get zeros (will be masked out by
        # the where-scatter in :class:`EntityPretrainModel`).
        comet_features = torch.zeros(P, COMET_INPUT_DIM, dtype=torch.float32)
        is_comet = torch.zeros(P, dtype=torch.bool)
        if n_real_p > 0:
            for i, row in enumerate(planet_rows[:n_real_p]):
                if int(row.get("is_comet", 0)) == 0:
                    continue
                is_comet[i] = True
                pid = int(row["planet_id"])
                vec = self.comet_lookup.get((episode_id, turn, pid))
                if vec is not None:
                    comet_features[i] = torch.tensor(vec, dtype=torch.float32)
                    self._comet_hits += 1
                else:
                    # Fall back to the 18 scalars carried by the planet
                    # CSV (cols f000..f017 share the same normalization
                    # as the comet specialist's scalar block); leave the
                    # 35 path slots zeroed with valid=0.
                    comet_features[i, :COMET_SCALAR_DIM] = planet_features[
                        i, :COMET_SCALAR_DIM
                    ]
                    self._comet_misses += 1

        # Fleet features + routing — same vectorization.
        fleet_features = torch.zeros(F, FLEET_RAW_DIM, dtype=torch.float32)
        fleet_mask = torch.zeros(F, dtype=torch.bool)
        fleet_target_idx = torch.full((F,), -1, dtype=torch.long)
        fleet_source_idx = torch.full((F,), -1, dtype=torch.long)
        fleet_owner_slot = torch.zeros(F, dtype=torch.long)
        fleet_ships_log = torch.zeros(F, dtype=torch.float32)
        fleet_eta_norm = torch.ones(F, dtype=torch.float32)
        n_real_f = min(len(fleet_rows), F)
        if n_real_f > 0:
            flat_f = [
                float(row[col])
                for row in fleet_rows[:n_real_f]
                for col in FLEET_FEATURE_COLS
            ]
            fleet_features[:n_real_f] = torch.tensor(
                flat_f, dtype=torch.float32,
            ).view(n_real_f, FLEET_RAW_DIM)
            fleet_mask[:n_real_f] = True

            target_indices = []
            source_indices = []
            owner_slots = []
            ships_logs = []
            eta_norms = []
            for row in fleet_rows[:n_real_f]:
                tgt_pid = int(row["target_planet_id"])
                src_pid = int(row["source_planet_id"])
                target_indices.append(pid_to_idx.get(tgt_pid, -1))
                source_indices.append(pid_to_idx.get(src_pid, -1))
                owner = int(row["owner_id"])
                if 0 <= owner < self.num_players:
                    owner_slots.append((owner - self.learner_slot) % self.num_players)
                else:
                    owner_slots.append(0)
                ships = int(row["ships"])
                ships_logs.append(math.log1p(max(0, ships)) / SHIPS_LOG_MAX)
                # Slot 10 of the fleet feature vector carries eta_norm.
                eta_norms.append(float(row[FLEET_FEATURE_COLS[10]]))
            fleet_target_idx[:n_real_f] = torch.tensor(target_indices, dtype=torch.long)
            fleet_source_idx[:n_real_f] = torch.tensor(source_indices, dtype=torch.long)
            fleet_owner_slot[:n_real_f] = torch.tensor(owner_slots, dtype=torch.long)
            fleet_ships_log[:n_real_f] = torch.tensor(ships_logs, dtype=torch.float32)
            fleet_eta_norm[:n_real_f] = torch.tensor(eta_norms, dtype=torch.float32)

        # Labels (per-planet and per-(planet, player)).
        labels: dict[str, torch.Tensor] = {
            "earliest_arrival_owner_slot": torch.zeros(P, dtype=torch.long),
            "is_source_this_turn": torch.zeros(P, dtype=torch.float32),
            "is_target_this_turn": torch.zeros(P, dtype=torch.float32),
        }
        for k in ENTITY_LABEL_HORIZONS:
            labels[f"owner_t_plus_{k}"] = torch.zeros(P, dtype=torch.long)
            labels[f"log_ships_t_plus_{k}"] = torch.zeros(P, dtype=torch.float32)
            labels[f"valid_t_plus_{k}"] = torch.zeros(P, dtype=torch.float32)
        for h in ENTITY_ARRIVAL_HORIZONS:
            labels[f"ships_arriving_within_{h}"] = torch.zeros(
                P, ENTITY_NUM_OWNER_SLOTS, dtype=torch.float32,
            )

        # Vectorize entity-row → label tensor copy. Same trick: gather
        # a Python list per label, do one tensor() call, scatter.
        entity_idxs: list[int] = []
        earliest_vals: list[int] = []
        is_source_vals: list[float] = []
        is_target_vals: list[float] = []
        owner_k_vals: dict[int, list[int]] = {k: [] for k in ENTITY_LABEL_HORIZONS}
        log_ships_k_vals: dict[int, list[float]] = {k: [] for k in ENTITY_LABEL_HORIZONS}
        valid_k_vals: dict[int, list[float]] = {k: [] for k in ENTITY_LABEL_HORIZONS}
        # ships_arriving has shape (P, NUM_OWNER_SLOTS); flatten per row.
        arriving_flat: dict[int, list[float]] = {h: [] for h in ENTITY_ARRIVAL_HORIZONS}
        for row in entity_rows:
            pid = int(row["planet_id"])
            if pid not in pid_to_idx:
                continue
            entity_idxs.append(pid_to_idx[pid])
            earliest_vals.append(int(row["earliest_arrival_owner_slot"]))
            is_source_vals.append(float(row["is_source_this_turn"]))
            is_target_vals.append(float(row["is_target_this_turn"]))
            for k in ENTITY_LABEL_HORIZONS:
                owner_k_vals[k].append(int(row[f"owner_t_plus_{k}"]))
                log_ships_k_vals[k].append(float(row[f"log_ships_t_plus_{k}"]))
                valid_k_vals[k].append(float(row[f"valid_t_plus_{k}"]))
            for h in ENTITY_ARRIVAL_HORIZONS:
                for slot in range(ENTITY_NUM_OWNER_SLOTS):
                    arriving_flat[h].append(
                        float(row[f"p{slot}_ships_arriving_within_{h}"])
                    )

        if entity_idxs:
            idxs_t = torch.tensor(entity_idxs, dtype=torch.long)
            labels["earliest_arrival_owner_slot"][idxs_t] = torch.tensor(
                earliest_vals, dtype=torch.long,
            )
            labels["is_source_this_turn"][idxs_t] = torch.tensor(
                is_source_vals, dtype=torch.float32,
            )
            labels["is_target_this_turn"][idxs_t] = torch.tensor(
                is_target_vals, dtype=torch.float32,
            )
            for k in ENTITY_LABEL_HORIZONS:
                labels[f"owner_t_plus_{k}"][idxs_t] = torch.tensor(
                    owner_k_vals[k], dtype=torch.long,
                )
                labels[f"log_ships_t_plus_{k}"][idxs_t] = torch.tensor(
                    log_ships_k_vals[k], dtype=torch.float32,
                )
                labels[f"valid_t_plus_{k}"][idxs_t] = torch.tensor(
                    valid_k_vals[k], dtype=torch.float32,
                )
            for h in ENTITY_ARRIVAL_HORIZONS:
                labels[f"ships_arriving_within_{h}"][idxs_t] = torch.tensor(
                    arriving_flat[h], dtype=torch.float32,
                ).view(len(entity_idxs), ENTITY_NUM_OWNER_SLOTS)

        snapshot: dict[str, torch.Tensor] = {
            "planet_features": planet_features,
            "planet_mask": planet_mask,
            "comet_features": comet_features,
            "is_comet": is_comet,
            "fleet_features": fleet_features,
            "fleet_mask": fleet_mask,
            "fleet_target_idx": fleet_target_idx,
            "fleet_source_idx": fleet_source_idx,
            "fleet_owner_slot": fleet_owner_slot,
            "fleet_ships_log": fleet_ships_log,
            "fleet_eta_norm": fleet_eta_norm,
        }
        snapshot.update(labels)
        # Stash the pid→idx map on the snapshot so subclasses can re-use
        # it cheaply when adding extra labels (e.g., cross_entity).
        # Stored as a non-tensor field; PyTorch DataLoader won't try to
        # batch it because the standard collate skips dict items keyed
        # by ``_`` prefix… actually DataLoader DOES try to collate
        # everything in the dict, so don't return it. Subclasses must
        # rebuild the map if they need it (cheap, O(P) per snapshot
        # at __init__ time only).
        return snapshot

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def _key_to_idx(self) -> dict[tuple[str, int], int]:
        """Lazy ``(episode_id, turn) → snapshot index`` map for cheap
        history lookups in :meth:`__getitem__`. Built once on first
        access; cached as ``self.__key_to_idx_cache``."""
        cache = getattr(self, "_EntitySnapshotDataset__key_to_idx_cache", None)
        if cache is None:
            cache = {k: i for i, k in enumerate(self.keys)}
            self._EntitySnapshotDataset__key_to_idx_cache = cache
        return cache

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cur = self.snapshots[idx]
        if self.history_offsets is None:
            return cur
        ep, t = self.keys[idx]
        # Gather past snapshots; missing frames become None and get
        # zero-filled below. Offsets are oldest-first per the L2
        # ``step_embed`` convention (slot 0 = oldest, slot T-1 = now).
        history: list[dict[str, torch.Tensor] | None] = []
        for off in self.history_offsets:
            prev_idx = self._key_to_idx.get((ep, t - off))
            history.append(
                self.snapshots[prev_idx] if prev_idx is not None else None
            )
        out: dict[str, torch.Tensor] = {}
        for key, val in cur.items():
            if key in self._STACK_KEYS:
                stacked: list[torch.Tensor] = []
                for snap in history:
                    if snap is None:
                        # Zero frame + all-False mask → ignored by L2
                        # attention (key_padding_mask flips it to True
                        # = "mask"). Per-fleet routing tensors get -1
                        # for indices, 0.0 for ships/eta — none of
                        # which matter because the mask is False.
                        if val.dtype == torch.long:
                            zeros = torch.full_like(val, -1) if "_idx" in key else torch.zeros_like(val)
                        else:
                            zeros = torch.zeros_like(val)
                        stacked.append(zeros)
                    else:
                        stacked.append(snap[key])
                out[key] = torch.stack(stacked, dim=0)               # (T, ...)
            else:
                # Labels + scalar CLS targets stay at current step only.
                out[key] = val
        return out


# ---------- Model ----------
class EntityPretrainModel(nn.Module):
    """Per-planet supervised head over a 4-layer trainable stack.

    Stack (L0 specialists held externally, frozen):

      L1  PlanetEntityEncoder
          cross-attention from each planet over relation-aware
          fleet representations → ``entity_tokens (B, P, d)``.
      L2  CrossEntityAttention
          self-attention across planets + learned CLS → per-planet
          ``ctx_now (B, P, d)`` and snapshot ``glob (B, d)``.
      L3  DualRoleAttention
          two parallel role-conditioned cross-attention branches over
          ``ctx_now``:
            * source-to-target → ``source_aware (B, P, d)``
            * target-to-source → ``target_aware (B, P, d)``
      L4  JointRoleAttention
          concatenates ``source_aware`` and ``target_aware`` into one
          ``(B, 2P, d)`` sequence (with fresh role embeddings),
          runs a 1-layer Pre-LN TransformerEncoder so every (slot,
          role) attends to every other (slot, role), then splits
          back into ``source_joint (B, P, d)`` and
          ``target_joint (B, P, d)``.

      Head (pair score, replaces the prior binary decoders):
        * ``pair_head(source_joint, target_joint, ctx_now) →
          pair_logits (B, P, P)``. Per (source, target) cell, the
          head builds a feature vector from role tokens, contextual
          tokens, and their Hadamard products, then a 3-Linear MLP
          produces one logit. Loss is BCE-with-logits per cell,
          masked by ``pair_valid`` (many-to-one supported naturally:
          a target can have multiple True source cells in one snap).

      The 7 future-state heads and the two binary "this turn" decoders
      were removed; the entity-pretrain task is now a pair-score head
      directly, supervised by the expert pair-set labels in the
      ``CachedPairDataset`` cache (built by
      ``scripts/build_pair_dataset_orbital_occle.py``).

    L4 is the explicit mix step after the parallel source/target
    branches. It does not collapse the roles into one vector; it lets
    source-mode and target-mode tokens exchange information, then keeps
    separate ``source_joint`` and ``target_joint`` streams for the pair
    scorer.
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        entity_n_heads: int = 8,
        cross_n_heads: int = 8,
        cross_n_layers: int = 2,
        cross_ff_mult: int = 2,
        dual_n_heads: int = 8,
        dropout: float = 0.0,
        n_steps: int = 1,
        d_pair: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_steps = int(n_steps)
        # All MHA blocks default to 8 heads (head_dim = d_model // 8 = 32 at
        # d_model=256). ``entity_n_heads`` controls L1's planet←fleet cross-attn;
        # ``cross_n_heads`` controls L2's self-attn; ``dual_n_heads`` controls
        # BOTH L3 (DualRoleAttention) and L4 (JointRoleAttention).
        self.entity_n_heads = int(entity_n_heads)
        self.cross_n_heads = int(cross_n_heads)
        self.cross_n_layers = int(cross_n_layers)
        self.dual_n_heads = int(dual_n_heads)
        # ``d_pair`` controls PairHead's projection width. Default = d_model
        # (no down-projection). Pass an explicit smaller value to reproduce
        # the legacy 128-wide layout for ablation.
        self.d_pair = int(d_pair) if d_pair is not None else int(d_model)
        self.entity = PlanetEntityEncoder(
            d_model=d_model, n_heads=self.entity_n_heads,
        )
        self.cross = CrossEntityAttention(
            d_model=d_model,
            n_heads=cross_n_heads,
            n_layers=cross_n_layers,
            ff_mult=cross_ff_mult,
            dropout=dropout,
            # Size the step-embedding table to the configured history
            # window. Single-frame callers (``n_steps=1``) still work —
            # ``CrossEntityAttention.forward`` slices ``step_embed[-T:]``
            # so callers can pass T <= n_steps freely.
            n_steps=self.n_steps,
        )
        self.dual_role = DualRoleAttention(
            d_model=d_model, n_heads=dual_n_heads, dropout=dropout,
        )
        # L4: joint self-attention over the 2P concatenated role tokens.
        # 1 layer is enough — most relational structure was already
        # established by L1 (planet ↔ fleet) and L2 (planet ↔ planet);
        # this layer's job is just to let same-role tokens see each
        # other directly under the new role tagging.
        self.joint_role = JointRoleAttention(
            d_model=d_model,
            n_heads=dual_n_heads,
            n_layers=1,
            ff_mult=cross_ff_mult,
            dropout=dropout,
        )

        # Pair-score head. Consumes the L4 joint role tokens plus L2
        # ctx_now and emits a dict of 5 logits/scores from a shared
        # trunk. With d_pair = d_model = 256 (no down-projection), the
        # trunk is: Linear(6·d_model = 1536 → trunk_hidden = d_model),
        # GELU, Linear(d_model → d_model), GELU. Five single-Linear
        # heads consume the trunk:
        #   * pair_logits  (B, P, P)  per-cell source→target launch
        #   * pair_frac    (B, P, P)  fraction-of-source ships raw
        #   * source_act   (B, P)     "this planet launches"
        #   * target_aim   (B, P)     "this planet is targeted"
        #   * glob_act     (B,)       snapshot-level "any action"
        # Per-head loss masking lives in :func:`compute_multi_loss`.
        self.pair_head = PairHead(
            d_model=d_model,
            d_pair=self.d_pair,       # default = d_model (full-width, no down-projection)
            trunk_hidden=d_model,
            dropout=dropout,
        )

    def forward(
        self,
        planet_tokens: torch.Tensor,
        fleet_tokens: torch.Tensor,
        routing: dict[str, torch.Tensor],
        planet_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Detect history-stacked input (B, T, P, d) vs single-frame
        # (B, P, d). The history-stacking dataset emits a leading T
        # axis on every input tensor; supervision labels stay
        # current-turn-only, so heads always read off the last step.
        is_temporal = planet_tokens.dim() == 4
        if is_temporal:
            B, T, P, d = planet_tokens.shape
            F = fleet_tokens.shape[2]
            # Flatten time into batch so L1 sees a per-frame slice:
            # each planet attends only to fleets at the SAME timestep.
            entity_tokens_flat = self.entity(
                planet_tokens.reshape(B * T, P, d),
                fleet_tokens.reshape(B * T, F, d),
                routing["fleet_target_idx"].reshape(B * T, F),
                routing["fleet_source_idx"].reshape(B * T, F),
                routing["fleet_owner_slot"].reshape(B * T, F),
                routing["fleet_ships_log"].reshape(B * T, F),
                routing["fleet_eta_norm"].reshape(B * T, F),
                routing["fleet_mask"].reshape(B * T, F),
                planet_mask=planet_mask.reshape(B * T, P),
            )                                                # (B*T, P, d)
            entity_tokens = entity_tokens_flat.reshape(B, T, P, d)
        else:
            # L1: per-planet entity tokens from fleet cross-attention.
            entity_tokens = self.entity(
                planet_tokens,
                fleet_tokens,
                routing["fleet_target_idx"],
                routing["fleet_source_idx"],
                routing["fleet_owner_slot"],
                routing["fleet_ships_log"],
                routing["fleet_eta_norm"],
                routing["fleet_mask"],
                planet_mask=planet_mask,
            )                                                # (B, P, d)

        # L2: planet-to-planet contextualization. CrossEntityAttention
        # accepts both rank-3 (single-frame) and rank-4 (multi-step);
        # in the temporal case it returns ``(B, T, P, d)`` and we take
        # the last (current) step for L3 and the heads.
        ctx_full, _glob = self.cross(
            entity_tokens,
            planet_mask if not is_temporal else planet_mask,
        )
        if is_temporal:
            ctx_now = ctx_full[:, -1]                        # (B, P, d)
            planet_mask_now = planet_mask[:, -1]             # (B, P)
        else:
            ctx_now = ctx_full                                # (B, P, d)
            planet_mask_now = planet_mask                     # (B, P)

        # L3: parallel source/target role-conditioned attention.
        source_aware, target_aware = self.dual_role(ctx_now, planet_mask_now)

        # L4: joint self-attention over the 2P concatenated role tokens.
        source_joint, target_joint = self.joint_role(
            source_aware, target_aware, planet_mask_now,
        )                                                    # (B, P, d) each

        # Derive pair_valid from the current-step planet_mask: a pair
        # cell (s, t) is valid iff both endpoints are real planets and
        # s != t. Lets the head's per-planet/snapshot pools mean over
        # only the valid cells instead of polluted-by-padding rows.
        B_now, P_now = planet_mask_now.shape
        pair_valid = (
            planet_mask_now.unsqueeze(2)
            & planet_mask_now.unsqueeze(1)
        )
        eye = torch.eye(P_now, dtype=torch.bool, device=pair_valid.device)
        pair_valid = pair_valid & ~eye.unsqueeze(0)

        # Pair-score head returns a 5-key dict of raw logits/scores.
        # Loss masking lives in :func:`compute_multi_loss`; the head
        # emits raw outputs so inference (argmax / top-k / sigmoid)
        # stays callable downstream.
        heads = self.pair_head(
            source_joint, target_joint, ctx_now, pair_valid=pair_valid,
        )
        return heads


# ---------- Loss ----------
# Canonical 5-head ordering. The diagonal (s == s) stays masked out of the
# loss — every off-diagonal cell is an independent binary "should source s
# launch to target t?" prediction. Multi-target rows (coalition launches)
# fall out of the per-cell BCE naturally.
_HEAD_NAMES: tuple[str, ...] = (
    "pair_logits", "pair_frac", "source_act", "target_aim", "glob_act",
)


def _masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Masked BCE-with-logits, summed-over-mask / sum(mask)."""
    mask_f = mask.to(logits.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    if pos_weight == 1.0:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
    else:
        pw = torch.tensor(
            pos_weight, device=logits.device, dtype=logits.dtype,
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none",
        )
    return (bce * mask_f).sum() / denom


def compute_multi_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    pair_pos_weight: float = 600.0,
    source_act_pos_weight: float = 100.0,
    target_aim_pos_weight: float = 100.0,
    glob_act_pos_weight: float = 1.0,
    enabled_heads: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """5-head loss over the joint PairHead output.

    Heads (mask & objective):

      * ``pair_logits``   BCE vs ``batch['pair_labels']`` masked by
        ``batch['pair_valid']``; ``pos_weight=pair_pos_weight``.
        Diagonal stays masked out (cache invariant). Multi-target rows
        (coalition launches) are handled naturally: each off-diagonal
        cell is independent, so the loss penalizes EVERY True cell
        regardless of how many other True cells share the row.
      * ``pair_frac``     MSE on sigmoid(pair_frac) vs the row-normalized
        ``pair_ships`` (fraction of source's ships sent to target).
        Masked to positive cells (pair_labels & pair_valid). Skipped
        entirely when ``pair_ships`` is missing from the batch.
      * ``source_act``    BCE vs ``pair_labels.any(dim=-1)`` masked by
        the current-step planet_mask (planet_mask[:, -1] when temporal).
        ``pos_weight=source_act_pos_weight``.
      * ``target_aim``    BCE vs ``pair_labels.any(dim=-2)``, same mask
        and ``pos_weight=target_aim_pos_weight``.
      * ``glob_act``      BCE vs ``pair_labels.any(dim=(-1, -2))``
        (snapshot-level); no per-cell mask. ``pos_weight=glob_act_pos_weight``.

    Returns ``(total_loss, per_head_dict)`` where ``per_head_dict``
    maps head name → unweighted scalar loss (for logging). The total
    sums all enabled heads' losses (with pos_weight applied where it
    applies). ``enabled_heads`` lets the legacy ``compute_pair_loss``
    shim restrict the sum to ``("pair_logits",)``.
    """
    if enabled_heads is None:
        enabled_heads = _HEAD_NAMES

    per_head: dict[str, float] = {}
    total = torch.zeros((), device=preds["pair_logits"].device,
                        dtype=preds["pair_logits"].dtype)

    pair_logits = preds["pair_logits"]
    pair_labels = batch["pair_labels"].bool()
    pair_valid = batch["pair_valid"].bool()

    # 1) pair_logits — masked BCE over (B, P, P).
    if "pair_logits" in enabled_heads:
        loss_pair = _masked_bce(
            pair_logits, pair_labels.float(), pair_valid,
            pos_weight=pair_pos_weight,
        )
        per_head["pair_logits"] = float(loss_pair.detach())
        total = total + loss_pair

    # planet_mask at the current step. The CachedPairDataset emits
    # planet_mask shaped (T, P) when history is on, else (P,); the
    # train loop collates to (B, T, P) or (B, P).
    planet_mask = batch["planet_mask"]
    if planet_mask.dim() == 3:
        planet_mask_now = planet_mask[:, -1].bool()
    else:
        planet_mask_now = planet_mask.bool()

    # 2) pair_frac — masked MSE-on-sigmoid vs row-normalized ship counts.
    # Skip gracefully when pair_ships is missing (older caches).
    if "pair_frac" in enabled_heads and "pair_ships" in batch:
        ships = batch["pair_ships"].float()              # (B, P, P)
        row_sum = ships.sum(dim=-1, keepdim=True).clamp(min=1.0)
        target_frac = ships / row_sum                     # (B, P, P)
        pos_mask = (pair_labels & pair_valid).to(pair_logits.dtype)
        frac_sigmoid = torch.sigmoid(preds["pair_frac"])
        sq_err = (frac_sigmoid - target_frac).pow(2)
        loss_frac = (sq_err * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)
        per_head["pair_frac"] = float(loss_frac.detach())
        total = total + loss_frac
    elif "pair_frac" in enabled_heads:
        per_head["pair_frac"] = float("nan")

    # 3) source_act — per-planet "this planet launches?" (any target).
    if "source_act" in enabled_heads:
        src_labels = pair_labels.any(dim=-1).float()      # (B, P)
        loss_src = _masked_bce(
            preds["source_act"], src_labels, planet_mask_now,
            pos_weight=source_act_pos_weight,
        )
        per_head["source_act"] = float(loss_src.detach())
        total = total + loss_src

    # 4) target_aim — per-planet "is this planet hit by anyone?".
    if "target_aim" in enabled_heads:
        tgt_labels = pair_labels.any(dim=-2).float()      # (B, P)
        loss_tgt = _masked_bce(
            preds["target_aim"], tgt_labels, planet_mask_now,
            pos_weight=target_aim_pos_weight,
        )
        per_head["target_aim"] = float(loss_tgt.detach())
        total = total + loss_tgt

    # 5) glob_act — snapshot-level "any action this turn?".
    if "glob_act" in enabled_heads:
        glob_labels = pair_labels.any(dim=-1).any(dim=-1).float()  # (B,)
        ones = torch.ones_like(glob_labels)
        loss_glob = _masked_bce(
            preds["glob_act"], glob_labels, ones,
            pos_weight=glob_act_pos_weight,
        )
        per_head["glob_act"] = float(loss_glob.detach())
        total = total + loss_glob

    return total, per_head


def compute_pair_loss(
    pair_logits: torch.Tensor,        # (B, P, P)
    pair_labels: torch.Tensor,        # (B, P, P) bool
    pair_valid: torch.Tensor,         # (B, P, P) bool
    *,
    pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Backward-compat shim: single-head pair-BCE only.

    Forwards to :func:`compute_multi_loss` with only ``pair_logits``
    enabled, so external callers (tests, scripts) keep working without
    seeing the other 4 heads.
    """
    preds = {"pair_logits": pair_logits}
    batch = {"pair_labels": pair_labels, "pair_valid": pair_valid}
    return compute_multi_loss(
        preds, batch,
        pair_pos_weight=pos_weight,
        enabled_heads=("pair_logits",),
    )


# Kept under the old name so external callers (tests, scripts) don't break.
compute_loss = compute_pair_loss


# ---------- Eval ----------
def _build_entity_self_tokens(
    planet_tok: torch.Tensor,    # (B, P, d)
    comet_tok: torch.Tensor,     # (B, P, d)
    is_comet: torch.Tensor,      # (B, P) bool
) -> torch.Tensor:
    """Route per-slot self-tokens by entity class.

    Comet rows receive the comet-specialist token; planet rows receive
    the planet-specialist token. Both specialists emit ``d_model``-wide
    tokens, so the downstream :class:`PlanetEntityEncoder` consumes a
    uniform-width stream without a projection bridge. The route is a
    hard per-slot selection, not a learned mixture.
    """
    return torch.where(is_comet.unsqueeze(-1), comet_tok, planet_tok)


@torch.no_grad()
def evaluate(
    model: EntityPretrainModel,
    fleet_enc: FleetEncoder,
    planet_enc: PlanetEncoder,
    comet_enc: CometEncoder,
    loader: DataLoader,
    device: str,
) -> dict[str, dict[str, float]]:
    """Per-head metrics over the validation set.

    For each of the 5 heads we accumulate ``tp/fp/tn/fn`` (with the
    head's natural mask) and an unweighted BCE loss. ``pair_logits``
    additionally keeps the legacy ``recall_at_{1,5,10}`` / row-softmax
    variants. ``pair_frac`` is regression: only its MSE is reported.
    """
    model.eval()

    # Per-head confusion counters. ``pair_frac`` is regression — we
    # only track MSE for it.
    bce_stats: dict[str, dict[str, int]] = {
        name: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for name in ("pair_logits", "source_act", "target_aim", "glob_act")
    }
    bce_loss_sum: dict[str, float] = {
        name: 0.0 for name in (
            "pair_logits", "source_act", "target_aim", "glob_act",
        )
    }
    bce_denom: dict[str, int] = {
        name: 0 for name in (
            "pair_logits", "source_act", "target_aim", "glob_act",
        )
    }

    # pair_frac (regression): MSE on positive cells.
    frac_se_sum = 0.0
    frac_n = 0
    frac_n_pos = 0

    # pair_logits-specific top-k accounting (kept from prior eval).
    rk_hits: dict[int, int] = {1: 0, 5: 0, 10: 0}
    rk_total = 0
    pair_rk_hits: dict[int, int] = {1: 0, 5: 0, 10: 0}
    row_rk_hits: dict[int, int] = {1: 0, 5: 0, 10: 0}
    pos_total = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        planet_tok = planet_enc(batch["planet_features"])
        comet_tok = comet_enc(batch["comet_features"])
        entity_self = _build_entity_self_tokens(
            planet_tok, comet_tok, batch["is_comet"],
        )
        fleet_tok = fleet_enc(batch["fleet_features"])
        routing = {
            "fleet_target_idx": batch["fleet_target_idx"],
            "fleet_source_idx": batch["fleet_source_idx"],
            "fleet_owner_slot": batch["fleet_owner_slot"],
            "fleet_ships_log": batch["fleet_ships_log"],
            "fleet_eta_norm": batch["fleet_eta_norm"],
            "fleet_mask": batch["fleet_mask"],
        }
        preds = model(entity_self, fleet_tok, routing, batch["planet_mask"])
        pair_logits = preds["pair_logits"]               # (B, P, P)
        pair_labels = batch["pair_labels"].bool()        # (B, P, P)
        pair_valid = batch["pair_valid"].bool()          # (B, P, P)

        planet_mask = batch["planet_mask"]
        if planet_mask.dim() == 3:
            planet_mask_now = planet_mask[:, -1].bool()
        else:
            planet_mask_now = planet_mask.bool()

        # ---- pair_logits ----
        bce = F.binary_cross_entropy_with_logits(
            pair_logits, pair_labels.float(), reduction="none",
        )
        valid_f = pair_valid.float()
        bce_loss_sum["pair_logits"] += float((bce * valid_f).sum())
        bce_denom["pair_logits"] += int(valid_f.sum())

        pos_pred = pair_logits > 0
        s = bce_stats["pair_logits"]
        s["tp"] += int(((pos_pred & pair_labels) & pair_valid).sum())
        s["fp"] += int(((pos_pred & ~pair_labels) & pair_valid).sum())
        s["tn"] += int(((~pos_pred & ~pair_labels) & pair_valid).sum())
        s["fn"] += int(((~pos_pred & pair_labels) & pair_valid).sum())

        # Recall@k over flattened P² grid (one ranking per snapshot).
        B, P, _ = pair_logits.shape
        flat_logits = pair_logits.masked_fill(~pair_valid, float("-inf"))
        flat_logits = flat_logits.reshape(B, P * P)
        flat_labels = (pair_labels & pair_valid).reshape(B, P * P)
        for k in rk_hits:
            topk = flat_logits.topk(min(k, P * P), dim=-1).indices
            top_mask = torch.zeros_like(flat_labels)
            top_mask.scatter_(1, topk, True)
            per_snap = (top_mask & flat_labels).any(dim=-1).int()
            rk_hits[k] += int(per_snap.sum())
            pair_rk_hits[k] += int((top_mask & flat_labels).sum())
        rk_total += int((pair_labels & pair_valid).any(dim=(-1, -2)).sum())
        pos_total += int((pair_labels & pair_valid).sum())

        row_logits = pair_logits.masked_fill(~pair_valid, float("-inf"))
        row_labels = pair_labels & pair_valid
        for k in row_rk_hits:
            topk_t = row_logits.topk(min(k, P), dim=-1).indices
            row_top_mask = torch.zeros_like(row_labels)
            row_top_mask.scatter_(2, topk_t, True)
            row_rk_hits[k] += int((row_top_mask & row_labels).sum())

        # ---- pair_frac (regression — MSE on positive cells) ----
        if "pair_ships" in batch:
            ships = batch["pair_ships"].float()
            row_sum = ships.sum(dim=-1, keepdim=True).clamp(min=1.0)
            target_frac = ships / row_sum
            pos_mask = (pair_labels & pair_valid).float()
            frac_sig = torch.sigmoid(preds["pair_frac"])
            sq = (frac_sig - target_frac).pow(2)
            frac_se_sum += float((sq * pos_mask).sum())
            frac_n += int(pos_mask.sum())
            frac_n_pos += int(pos_mask.sum())

        # ---- source_act ----
        src_labels = pair_labels.any(dim=-1)             # (B, P) bool
        src_logits = preds["source_act"]                 # (B, P)
        bce_src = F.binary_cross_entropy_with_logits(
            src_logits, src_labels.float(), reduction="none",
        )
        mask_f = planet_mask_now.float()
        bce_loss_sum["source_act"] += float((bce_src * mask_f).sum())
        bce_denom["source_act"] += int(mask_f.sum())

        pred_src = src_logits > 0
        s = bce_stats["source_act"]
        s["tp"] += int(((pred_src & src_labels) & planet_mask_now).sum())
        s["fp"] += int(((pred_src & ~src_labels) & planet_mask_now).sum())
        s["tn"] += int(((~pred_src & ~src_labels) & planet_mask_now).sum())
        s["fn"] += int(((~pred_src & src_labels) & planet_mask_now).sum())

        # ---- target_aim ----
        tgt_labels = pair_labels.any(dim=-2)             # (B, P) bool
        tgt_logits = preds["target_aim"]                 # (B, P)
        bce_tgt = F.binary_cross_entropy_with_logits(
            tgt_logits, tgt_labels.float(), reduction="none",
        )
        bce_loss_sum["target_aim"] += float((bce_tgt * mask_f).sum())
        bce_denom["target_aim"] += int(mask_f.sum())

        pred_tgt = tgt_logits > 0
        s = bce_stats["target_aim"]
        s["tp"] += int(((pred_tgt & tgt_labels) & planet_mask_now).sum())
        s["fp"] += int(((pred_tgt & ~tgt_labels) & planet_mask_now).sum())
        s["tn"] += int(((~pred_tgt & ~tgt_labels) & planet_mask_now).sum())
        s["fn"] += int(((~pred_tgt & tgt_labels) & planet_mask_now).sum())

        # ---- glob_act ----
        glob_labels = pair_labels.any(dim=-1).any(dim=-1)  # (B,) bool
        glob_logits = preds["glob_act"]                    # (B,)
        bce_glob = F.binary_cross_entropy_with_logits(
            glob_logits, glob_labels.float(), reduction="none",
        )
        bce_loss_sum["glob_act"] += float(bce_glob.sum())
        bce_denom["glob_act"] += int(glob_labels.numel())

        pred_glob = glob_logits > 0
        s = bce_stats["glob_act"]
        s["tp"] += int((pred_glob & glob_labels).sum())
        s["fp"] += int((pred_glob & ~glob_labels).sum())
        s["tn"] += int((~pred_glob & ~glob_labels).sum())
        s["fn"] += int((~pred_glob & glob_labels).sum())

    summary: dict[str, dict[str, float]] = {}

    # 5-head rows. Build pair_logits first so the table order matches
    # ``_HEAD_NAMES``.
    for name in ("pair_logits", "source_act", "target_aim", "glob_act"):
        s = bce_stats[name]
        pos = s["tp"] + s["fn"]
        neg = s["tn"] + s["fp"]
        total_cells = max(1, pos + neg)
        entry: dict[str, float] = {
            "loss": bce_loss_sum[name] / max(1, bce_denom[name]),
            "recall_true":  s["tp"] / max(1, pos),
            "recall_false": s["tn"] / max(1, neg),
            "n_pos": float(pos),
            "n_neg": float(neg),
            "pos_frac": pos / total_cells,
        }
        if name == "pair_logits":
            for k, hits in rk_hits.items():
                entry[f"recall_at_{k}"] = hits / max(1, rk_total)
                entry[f"pair_recall_at_{k}"] = (
                    pair_rk_hits[k] / max(1, pos_total)
                )
                entry[f"row_recall_at_{k}"] = (
                    row_rk_hits[k] / max(1, pos_total)
                )
        summary[name] = entry

    # pair_frac row — regression, MSE only. Placed between pair_logits
    # and source_act so the print order matches ``_HEAD_NAMES``.
    if frac_n > 0:
        frac_entry = {
            "loss": frac_se_sum / max(1, frac_n),
            "n_pos": float(frac_n_pos),
        }
    else:
        # No pair_ships in batch — emit a row so the table is uniform,
        # but mark loss NaN so the printer shows "—".
        frac_entry = {"loss": float("nan"), "n_pos": 0.0}
    summary["pair_frac"] = frac_entry

    # Reorder summary to match _HEAD_NAMES so callers iterating
    # ``summary.items()`` get the canonical row sequence.
    ordered = {name: summary[name] for name in _HEAD_NAMES if name in summary}

    model.train()
    return ordered


def _format_summary(summary: dict[str, dict[str, float]]) -> str:
    lines = []
    for name, m in summary.items():
        acc = f"  acc={m['acc']:.3f}" if "acc" in m else ""
        lines.append(f"    {name:<32s}  loss={m['loss']:.4f}{acc}")
    return "\n".join(lines)


# Canonical print order matches ``_HEAD_NAMES``.
_HEAD_PRINT_ORDER: tuple[str, ...] = _HEAD_NAMES


def _format_heads_table(
    train_per_head: dict[str, float] | None,
    val_summary: dict[str, dict[str, float]] | None,
) -> str:
    """One row per head with train_loss / val_loss / recall / R@k.

    Cells that don't apply to a head (e.g. recall on ``pair_frac``,
    R@k on the per-planet heads) show as ``—``. Either side may be
    ``None`` (e.g. on a no-eval epoch); missing entries fall back to
    ``—`` too.
    """
    em = "\u2014"  # em dash for "not applicable"

    def _fnum(x: float | None, w: int, prec: int) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return f"{em:>{w}s}"
        return f"{x:>{w}.{prec}f}"

    def _fint(x: float | None, w: int) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return f"{em:>{w}s}"
        return f"{int(x):>{w}d}"

    headers = [
        ("head",         14, "s"),
        ("train_loss",   10, "f4"),
        ("val_loss",      9, "f4"),
        ("rec_pos",       8, "f3"),
        ("rec_neg",       8, "f3"),
        ("pos_frac",      9, "f5"),
        ("R@1",           5, "f3"),
        ("R@5",           5, "f3"),
        ("R@10",          5, "f3"),
        ("n_pos",        10, "d"),
        ("n_neg",        12, "d"),
    ]
    head_line = "    " + "  ".join(f"{name:>{w}s}" for name, w, _ in headers)
    # Left-justify the head column for readability.
    head_line = (
        f"    {'head':<14s}  "
        + "  ".join(f"{name:>{w}s}" for name, w, _ in headers[1:])
    )
    sep_line = (
        f"    {'-' * 14}  "
        + "  ".join("-" * w for _, w, _ in headers[1:])
    )
    lines = [head_line, sep_line]

    # Heads where recall/pos_frac make sense (BCE classification).
    bce_heads = {"pair_logits", "source_act", "target_aim", "glob_act"}
    # Head where R@k makes sense.
    rk_heads = {"pair_logits"}

    for name in _HEAD_PRINT_ORDER:
        t = train_per_head.get(name) if train_per_head else None
        v = val_summary.get(name) if val_summary else None

        cells = [_fnum(t, 10, 4)]
        if v is None:
            cells.extend([
                _fnum(None, 9, 4),
                _fnum(None, 8, 3),
                _fnum(None, 8, 3),
                _fnum(None, 9, 5),
                _fnum(None, 5, 3),
                _fnum(None, 5, 3),
                _fnum(None, 5, 3),
                _fint(None, 10),
                _fint(None, 12),
            ])
        else:
            cells.append(_fnum(v.get("loss"), 9, 4))
            if name in bce_heads:
                cells.append(_fnum(v.get("recall_true"), 8, 3))
                cells.append(_fnum(v.get("recall_false"), 8, 3))
                cells.append(_fnum(v.get("pos_frac"), 9, 5))
            else:
                cells.extend([
                    _fnum(None, 8, 3),
                    _fnum(None, 8, 3),
                    _fnum(None, 9, 5),
                ])
            if name in rk_heads:
                cells.append(_fnum(v.get("recall_at_1"), 5, 3))
                cells.append(_fnum(v.get("recall_at_5"), 5, 3))
                cells.append(_fnum(v.get("recall_at_10"), 5, 3))
            else:
                cells.extend([
                    _fnum(None, 5, 3),
                    _fnum(None, 5, 3),
                    _fnum(None, 5, 3),
                ])
            cells.append(_fint(v.get("n_pos"), 10))
            if name in bce_heads:
                cells.append(_fint(v.get("n_neg"), 12))
            else:
                cells.append(_fint(None, 12))

        lines.append(f"    {name:<14s}  " + "  ".join(cells))
    return "\n".join(lines)


# Alias to preserve the legacy symbol for older test scripts.
_format_per_head_table = _format_heads_table


# ---------- Train loop ----------
def _load_encoders(
    fleet_run_dir: Path,
    planet_run_dir: Path,
    comet_run_dir: Path,
    *,
    device: str,
    expected_d_model: int | None = None,
) -> tuple[FleetEncoder, PlanetEncoder, CometEncoder]:
    fc = torch.load(
        fleet_run_dir / "fleet_encoder_best.pt",
        map_location=device, weights_only=False,
    )
    pc = torch.load(
        planet_run_dir / "planet_encoder_best.pt",
        map_location=device, weights_only=False,
    )
    cc = torch.load(
        comet_run_dir / "comet_past_best.pt",
        map_location=device, weights_only=False,
    )
    dims = {
        "fleet": int(fc["config"]["d_model"]),
        "planet": int(pc["config"]["d_model"]),
        "comet": int(cc["config"]["d_model"]),
    }
    if len(set(dims.values())) != 1:
        raise ValueError(
            "L0 specialist width mismatch: "
            + ", ".join(f"{name}={dim}" for name, dim in dims.items())
        )
    if expected_d_model is not None and next(iter(dims.values())) != expected_d_model:
        raise ValueError(
            f"L0 specialists emit d_model={next(iter(dims.values()))}, "
            f"but entity stack was requested at d_model={expected_d_model}. "
            "Use matching ckpts or add an explicit projection bridge."
        )
    fenc = FleetEncoder(d_model=fc["config"]["d_model"])
    fenc.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in fc["model"].items()
         if k.startswith("encoder.")}
    )
    # Honor encoder-architecture flags written into the ckpt config so
    # the constructor builds the same submodules the state-dict carries
    # (e.g., ``use_traj_branch=False`` skips the ``traj`` and ``trunk``
    # branches and resizes ``scalar`` to ``d_model`` instead of
    # ``branch_dim``).
    pcfg = pc["config"]
    penc_kwargs: dict = {"d_model": pcfg["d_model"]}
    if "use_traj_branch" in pcfg:
        penc_kwargs["use_traj_branch"] = pcfg["use_traj_branch"]
    penc = PlanetEncoder(**penc_kwargs)
    penc.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in pc["model"].items()
         if k.startswith("encoder.")}
    )
    ccfg = cc["config"]
    cenc = CometEncoder(
        d_model=ccfg["d_model"],
        input_dim=ccfg.get("input_dim", COMET_INPUT_DIM),
    )
    # The ckpt may carry either the new composed layout
    # (``encoder.encoder.*`` / ``encoder.norm.*`` / ``decoder.*``) or the
    # legacy flat one (``encoder.*`` / ``norm.*`` / ``decoder.*`` /
    # ``scalar_heads.*``). Extract the encoder's slice for either case.
    cstate = cc["model"]
    if "encoder.encoder.0.weight" in cstate:
        # New layout — keys under the top-level ``encoder.`` submodule are
        # the encoder; strip one prefix to land on CometEncoder's keys.
        enc_state = {
            k[len("encoder."):]: v
            for k, v in cstate.items() if k.startswith("encoder.")
        }
    else:
        # Legacy layout — ``encoder.*`` and ``norm.*`` keys already match
        # CometEncoder's state_dict; drop the decoder + scalar_heads keys.
        enc_state = {
            k: v for k, v in cstate.items()
            if k.startswith("encoder.") or k.startswith("norm.")
        }
    cenc.load_state_dict(enc_state)
    fenc.to(device).eval()
    penc.to(device).eval()
    cenc.to(device).eval()
    for p in fenc.parameters(): p.requires_grad_(False)
    for p in penc.parameters(): p.requires_grad_(False)
    for p in cenc.parameters(): p.requires_grad_(False)
    return fenc, penc, cenc


def train(
    *,
    out_dir: Path,
    fleet_run_dir: Path,
    planet_run_dir: Path,
    comet_run_dir: Path,
    d_model: int = 128,
    d_pair: int | None = None,
    entity_n_heads: int = 8,
    cross_n_heads: int = 8,
    cross_n_layers: int = 2,
    dual_n_heads: int = 8,
    batch_size: int = 128,
    epochs: int = 30,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    eval_every: int = 1,
    max_planets: int = 64,
    max_fleets: int = 1024,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 0,
    pair_cache_path: Path = DATASETS_ROOT / "_pair_cache" / "bowwowforeach_Ebi_T6" / "bowwowforeach_Ebi_T6_p64_f1024_all.pt",
    pair_pos_weight: float = 600.0,
    source_act_pos_weight: float = 100.0,
    target_aim_pos_weight: float = 100.0,
    glob_act_pos_weight: float = 1.0,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # ---- Load the pair-set cache ----
    # CachedPairDataset reads the .pt file lazily and lazily stacks the
    # T-history at __getitem__ time using ``history_offsets`` baked into
    # the cache. The pair labels and pair_valid mask come ready to use.
    from scripts.build_pair_dataset_orbital_occle import CachedPairDataset

    if not pair_cache_path.exists():
        candidates = sorted(
            DATASETS_ROOT.glob("_pair_cache/*/*_T6_p*_f*_acted.pt")
        )
        avail = "\n".join(f"    {p}" for p in candidates[:20]) or "    <none>"
        raise FileNotFoundError(
            f"pair cache not found: {pair_cache_path}\n"
            f"Available acted T6 caches under {DATASETS_ROOT}/_pair_cache:\n"
            f"{avail}\n"
            "Pass --pair-cache-path explicitly, or rebuild the expected cache."
        )

    print(f"[entity-pretrain] loading pair cache from {pair_cache_path} ...")
    full_ds = CachedPairDataset(pair_cache_path)
    print(
        f"  cache config: player={full_ds.config.get('player')!r}  "
        f"keep_non_acted={full_ds.config.get('keep_non_acted')}  "
        f"history_offsets={full_ds.config.get('history_offsets')}  "
        f"max_planets={full_ds.config.get('max_planets')}  "
        f"max_fleets={full_ds.config.get('max_fleets')}"
    )
    acted_indices = list(getattr(full_ds, "acted_indices", []))
    if not acted_indices:
        acted_indices = [
            i for i, snap in enumerate(full_ds.snapshots)
            if bool(snap["pair_labels"].any())
        ]
    train_row_indices = (
        list(range(len(full_ds)))
        if bool(full_ds.config.get("keep_non_acted", False))
        else acted_indices
    )
    print(
        f"  snapshots: {len(full_ds):,} context rows; "
        f"{len(acted_indices):,} acted rows; "
        f"{len(train_row_indices):,} supervised rows"
    )
    if full_ds.history_offsets:
        keyset = set(full_ds.keys)
        n_required = len(full_ds.history_offsets)
        complete = 0
        any_history = 0
        have_hist_counts: dict[int, int] = defaultdict(int)
        for i in train_row_indices:
            ep, t = full_ds.keys[i]
            have = sum(1 for off in full_ds.history_offsets if (ep, t - off) in keyset)
            have_hist_counts[have] += 1
            if have == n_required:
                complete += 1
            if have >= 1:
                any_history += 1
        complete_frac = complete / max(1, len(train_row_indices))
        any_frac = any_history / max(1, len(train_row_indices))
        print(
            f"  T-history availability: complete={complete_frac:.3f}  "
            f"any={any_frac:.3f}  counts={dict(sorted(have_hist_counts.items()))}"
        )
        # Missing past frames get zero-filled with all-False planet_mask
        # so L2 attention's key_padding_mask cleanly ignores them — the
        # model handles variable effective T natively. Only refuse to
        # train when most training rows are total orphans (no history
        # at all).
        if any_frac < 0.10:
            raise ValueError(
                f"pair cache has <10% T-history coverage on training "
                f"rows (any_frac={any_frac:.3f}) — rebuild with all "
                f"non-acted frames retained, or disable history."
            )
        if complete_frac < 0.50:
            import warnings
            warnings.warn(
                f"only {complete_frac:.1%} of training rows have the "
                f"full {n_required}-step history; the rest have "
                f"{n_required - 1} or fewer real frames and the "
                f"missing ones get zero-padded + mask-ignored at L2. "
                f"This is fine for training but rebuilding the cache "
                f"with all non-acted context retained would give the "
                f"L2 step embedding cleaner gradients.",
                stacklevel=2,
            )
    cache_max_planets = int(full_ds.config.get("max_planets", max_planets))
    cache_max_fleets = int(full_ds.config.get("max_fleets", max_fleets))
    if cache_max_planets != max_planets or cache_max_fleets != max_fleets:
        raise ValueError(
            "CLI max sizes must match the pair cache: "
            f"cache has P={cache_max_planets}, F={cache_max_fleets}; "
            f"got --max-planets={max_planets}, --max-fleets={max_fleets}. "
            "Rebuild/select a matching cache or pass matching flags."
        )

    # Episode-level split so train/val/test snapshots come from disjoint
    # replays (preserves the i.i.d. assumption against same-episode
    # turn correlation).
    keys = full_ds.keys
    supervised_keys = [keys[i] for i in train_row_indices]
    ep_ids = sorted({k[0] for k in supervised_keys})
    rng = torch.Generator().manual_seed(seed)
    ep_order = [ep_ids[i] for i in torch.randperm(len(ep_ids), generator=rng).tolist()]
    n_ep = len(ep_order)
    n_test = max(1, int(round(n_ep * test_frac)))
    n_val = max(1, int(round(n_ep * val_frac)))
    test_eps = set(ep_order[:n_test])
    val_eps = set(ep_order[n_test : n_test + n_val])
    train_eps = set(ep_order[n_test + n_val :])

    from torch.utils.data import Subset
    split_idxs: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i in train_row_indices:
        ep, _t = keys[i]
        if ep in train_eps:
            split_idxs["train"].append(i)
        elif ep in val_eps:
            split_idxs["val"].append(i)
        elif ep in test_eps:
            split_idxs["test"].append(i)
    splits = {name: Subset(full_ds, idxs) for name, idxs in split_idxs.items()}
    for split in ("train", "val", "test"):
        print(
            f"  {split}: {len(splits[split]):,} snapshots  "
            f"({len(split_idxs[split])} from {len({keys[i][0] for i in split_idxs[split]})} episodes)"
        )

    # History dims are baked into the cache; surface for downstream use.
    history_offsets = tuple(full_ds.history_offsets) if full_ds.history_offsets else None
    n_steps = len(history_offsets) if history_offsets else 1

    train_loader = DataLoader(
        splits["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        splits["val"], batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        splits["test"], batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---- Load frozen encoders ----
    print(f"[entity-pretrain] loading encoders frozen ...")
    fleet_enc, planet_enc, comet_enc = _load_encoders(
        fleet_run_dir, planet_run_dir, comet_run_dir,
        device=device, expected_d_model=d_model,
    )
    print(f"  fleet enc params:  {sum(p.numel() for p in fleet_enc.parameters()):,} (frozen)")
    print(f"  planet enc params: {sum(p.numel() for p in planet_enc.parameters()):,} (frozen)")
    print(f"  comet enc params:  {sum(p.numel() for p in comet_enc.parameters()):,} (frozen)")

    # ---- Build entity encoder + heads ----
    n_steps = len(history_offsets) if history_offsets is not None else 1
    effective_d_pair = int(d_pair) if d_pair is not None else int(d_model)
    model = EntityPretrainModel(
        d_model=d_model, n_steps=n_steps, d_pair=effective_d_pair,
        entity_n_heads=entity_n_heads,
        cross_n_heads=cross_n_heads,
        cross_n_layers=cross_n_layers,
        dual_n_heads=dual_n_heads,
    ).to(device)
    print(
        f"  entity model params: "
        f"{sum(p.numel() for p in model.parameters()):,}  "
        f"(n_steps={n_steps}, d_pair={effective_d_pair}, "
        f"heads: L1={entity_n_heads} L2={cross_n_heads}×{cross_n_layers}L "
        f"L3=L4={dual_n_heads})"
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    config = {
        "d_model": d_model, "d_pair": effective_d_pair,
        "entity_n_heads": entity_n_heads,
        "cross_n_heads": cross_n_heads,
        "cross_n_layers": cross_n_layers,
        "dual_n_heads": dual_n_heads,
        "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": epochs,
        "fleet_run_dir": str(fleet_run_dir),
        "planet_run_dir": str(planet_run_dir),
        "comet_run_dir": str(comet_run_dir),
        "pair_cache_path": str(pair_cache_path),
        "pair_pos_weight": pair_pos_weight,
        "source_act_pos_weight": source_act_pos_weight,
        "target_aim_pos_weight": target_aim_pos_weight,
        "glob_act_pos_weight": glob_act_pos_weight,
        "history_offsets": list(history_offsets) if history_offsets is not None else None,
        "n_steps": n_steps,
    }
    if history_offsets is not None:
        print(
            f"[entity-pretrain] history-stacked input: T={n_steps} "
            f"offsets={history_offsets} (oldest first; labels stay at t)"
        )
    if pair_pos_weight != 1.0:
        print(
            f"[entity-pretrain] pair-BCE pos_weight = {pair_pos_weight} "
            f"(applied to train loss; val/test loss stays unweighted)"
        )
    print(
        f"[entity-pretrain] head pos_weights: "
        f"source_act={source_act_pos_weight}, "
        f"target_aim={target_aim_pos_weight}, "
        f"glob_act={glob_act_pos_weight}"
    )

    log: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "entity_encoder_best.pt"
    last_path = out_dir / "entity_encoder_last.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        n_batches = 0
        running_per_head: dict[str, float] = defaultdict(float)
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                planet_tok = planet_enc(batch["planet_features"])
                comet_tok = comet_enc(batch["comet_features"])
                fleet_tok = fleet_enc(batch["fleet_features"])
            entity_self = _build_entity_self_tokens(
                planet_tok, comet_tok, batch["is_comet"],
            )
            routing = {
                "fleet_target_idx": batch["fleet_target_idx"],
                "fleet_source_idx": batch["fleet_source_idx"],
                "fleet_owner_slot": batch["fleet_owner_slot"],
                "fleet_ships_log": batch["fleet_ships_log"],
                "fleet_eta_norm": batch["fleet_eta_norm"],
                "fleet_mask": batch["fleet_mask"],
            }
            preds = model(entity_self, fleet_tok, routing, batch["planet_mask"])
            total_loss, per_head = compute_multi_loss(
                preds, batch,
                pair_pos_weight=pair_pos_weight,
                source_act_pos_weight=source_act_pos_weight,
                target_aim_pos_weight=target_aim_pos_weight,
                glob_act_pos_weight=glob_act_pos_weight,
            )
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            running_total += float(total_loss.detach())
            for k, v in per_head.items():
                # Skip NaN sentinels from heads that are inactive this
                # batch (e.g., pair_frac when pair_ships is absent).
                if isinstance(v, float) and math.isnan(v):
                    continue
                running_per_head[k] += v
                running_per_head[f"_n_{k}"] = running_per_head.get(f"_n_{k}", 0) + 1
            n_batches += 1

        train_total = running_total / max(1, n_batches)
        train_per_head: dict[str, float] = {}
        for k in _HEAD_NAMES:
            n = running_per_head.get(f"_n_{k}", 0)
            if n > 0:
                train_per_head[k] = running_per_head[k] / n
        elapsed = round(time.time() - t0, 2)
        entry: dict[str, Any] = {
            "epoch": epoch, "train_total": train_total,
            "train_per_head": train_per_head, "elapsed_s": elapsed,
        }

        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate(model, fleet_enc, planet_enc, comet_enc, val_loader, device)
            # Average across heads whose loss is finite. ``pair_frac``
            # emits NaN on caches without ``pair_ships``; skip those so
            # the mean isn't poisoned.
            losses = [m["loss"] for m in val.values()
                      if not (isinstance(m["loss"], float) and math.isnan(m["loss"]))]
            mean = sum(losses) / max(1, len(losses))
            entry["val_mean_loss"] = mean
            entry["val"] = val
            print(
                f"[ep {epoch:>2}/{epochs}]  train_total={train_total:.4f}  "
                f"val_mean={mean:.4f}  ({elapsed}s)",
                flush=True,
            )
            print(_format_per_head_table(train_per_head, val), flush=True)
            if mean < best_val:
                best_val = mean
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, best_path)
        else:
            # Train-only epoch (eval_every > 1): still surface the
            # per-head breakdown so a regression in a single criterion
            # doesn't hide inside the total.
            print(
                f"[ep {epoch:>2}/{epochs}]  train_total={train_total:.4f}  "
                f"(no val; {elapsed}s)",
                flush=True,
            )
            print(_format_per_head_table(train_per_head, None), flush=True)

        log.append(entry)
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, last_path)
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    # ---- Test with best ckpt ----
    print("\n[entity-pretrain] evaluating best on test ...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, fleet_enc, planet_enc, comet_enc, test_loader, device)
    print(_format_summary(test))
    (out_dir / "test_summary.json").write_text(json.dumps(test, indent=2))
    print(f"\n[entity-pretrain] outputs in {out_dir}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fleet-run-dir", type=Path,
        default=FLEET_RUNS_DIR / "specialist_fleet_d256_40k_lr1e4_120ep",
        help="Source FleetEncoder checkpoint dir.",
    )
    parser.add_argument(
        "--planet-run-dir", type=Path,
        default=PLANET_RUNS_DIR / "specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep",
        help="Source PlanetEncoder checkpoint dir.",
    )
    parser.add_argument(
        "--comet-run-dir", type=Path,
        default=RUNS_ROOT / "comet" / "fullpath_scalar_multitask_d256_40k_lr1e4_120ep",
        help="Source CometPastModel checkpoint dir (d_model must equal "
             "the entity encoder's d_model so the where-scatter doesn't "
             "need a projection).",
    )
    # Comet lookup args removed — the pair cache now carries the
    # 123-dim comet features baked in per snapshot. Re-add these if you
    # ever switch back to live EntitySnapshotDataset construction.
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument(
        "--d-pair", type=int, default=None,
        help="PairHead projection width. None (default) keeps the full "
             "d_model width into the trunk's 6-way concat — no "
             "down-projection. Pass 128 to reproduce the legacy "
             "narrowed-trunk layout for ablation.",
    )
    parser.add_argument(
        "--entity-n-heads", type=int, default=8,
        help="L1 PlanetEntityEncoder cross-attn heads (default 8 → "
             "32-dim per head at d_model=256).",
    )
    parser.add_argument(
        "--cross-n-heads", type=int, default=8,
        help="L2 CrossEntityAttention self-attn heads per encoder layer.",
    )
    parser.add_argument(
        "--cross-n-layers", type=int, default=2,
        help="L2 CrossEntityAttention number of Pre-LN encoder layers.",
    )
    parser.add_argument(
        "--dual-n-heads", type=int, default=8,
        help="Shared head count for L3 DualRoleAttention and L4 "
             "JointRoleAttention (default 8).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=1024,
                        help="Per-snapshot fleet cap. Real max observed = 813 "
                             "across all replays; 1024 is the next pow-of-2 "
                             "above it. Lower values (256/512) save memory "
                             "but truncate ~3k snapshots and corrupt the "
                             "inbound labels for them.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pair-cache-path", type=Path,
        default=DATASETS_ROOT / "_pair_cache" / "bowwowforeach_Ebi_T6" / "bowwowforeach_Ebi_T6_p64_f1024_all.pt",
        help="Path to the pair-set cache .pt file built by "
             "scripts/build_pair_dataset_orbital_occle.py. The cache "
             "carries the history_offsets / max_planets / max_fleets "
             "config; we read those at startup.",
    )
    parser.add_argument(
        "--pair-pos-weight", type=float, default=600.0,
        help="pos_weight multiplier for the masked pair-BCE loss. "
             "Observed positive-cell fraction is ~0.16%% (29,820 / "
             "~18M valid cells), so n_neg/n_pos ≈ 600. Default 600 "
             "trades precision for recall on the rare positives. "
             "Set to 1.0 to disable.",
    )
    parser.add_argument(
        "--source-act-pos-weight", type=float, default=100.0,
        help="pos_weight for the per-source 'launches?' head (matches "
             "the prior src_pos_weight default of 100).",
    )
    parser.add_argument(
        "--target-aim-pos-weight", type=float, default=100.0,
        help="pos_weight for the per-target 'is targeted?' head.",
    )
    parser.add_argument(
        "--glob-act-pos-weight", type=float, default=1.0,
        help="pos_weight for the snapshot-level 'any action?' head. "
             "Default 1.0 since the mixed-acted cache balances the "
             "class. Bump if you train on acted-only.",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.10,
        help="Fraction of episodes held out for val (episode-level "
             "split so train/val/test come from disjoint replays).",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.10,
        help="Fraction of episodes held out for test.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (ENTITY_RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"))
    train(
        out_dir=out_dir,
        fleet_run_dir=args.fleet_run_dir,
        planet_run_dir=args.planet_run_dir,
        comet_run_dir=args.comet_run_dir,
        d_model=args.d_model,
        d_pair=args.d_pair,
        entity_n_heads=args.entity_n_heads,
        cross_n_heads=args.cross_n_heads,
        cross_n_layers=args.cross_n_layers,
        dual_n_heads=args.dual_n_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        max_planets=args.max_planets,
        max_fleets=args.max_fleets,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        pair_cache_path=args.pair_cache_path,
        pair_pos_weight=args.pair_pos_weight,
        source_act_pos_weight=args.source_act_pos_weight,
        target_aim_pos_weight=args.target_aim_pos_weight,
        glob_act_pos_weight=args.glob_act_pos_weight,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )


if __name__ == "__main__":
    main()
