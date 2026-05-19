"""Per-snapshot entity encoder pretraining.

The :class:`PlanetEntityEncoder` consumes *embeddings* from
:class:`FleetEncoder` and :class:`PlanetEncoder`, not raw features. To
keep the dataset on disk small and replayable, the CSVs store **raw**
inputs and labels; this script loads them, runs the two upstream
encoders **frozen / no_grad** to produce embeddings, then trains the
entity encoder + multi-task heads on top.

Per-snapshot batching: each training example is one ``(episode, turn)``
slice of the game. We pad to fixed ``max_planets``/``max_fleets`` so
the snapshot tensors stack cleanly into a batch. Labels and routing
tensors are extracted from the corresponding CSV rows joined by
``(episode_id, turn[, planet_id])``.

Run from the repo root:

    python -m agents.transformer_v1.pretrain.entity_encoder \\
        --epochs 30 --batch-size 32

Outputs (under ``data/runs/entity/<timestamp>/``):
  * ``entity_encoder_best.pt`` / ``entity_encoder_last.pt``
  * ``log.json`` — per-epoch train + val per-head losses & metrics
  * ``test_summary.json`` — per-head test metrics from the best ckpt
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
    ENTITY_DATASET_DIR,
    ENTITY_RUNS_DIR,
    FLEET_DATASET_DIR,
    FLEET_RUNS_DIR,
    PLANET_DATASET_DIR,
    PLANET_RUNS_DIR,
)


# ---------- Dataset ----------
PLANET_FEATURE_COLS = tuple(f"f{i:03d}" for i in range(PLANET_RAW_DIM))
FLEET_FEATURE_COLS = tuple(f"f{i:03d}" for i in range(FLEET_RAW_DIM))


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
    """

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
    ):
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.learner_slot = learner_slot
        self.num_players = num_players

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
        common_stems = self._filter_stems(common_stems)

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
            flat = [
                float(row[col])
                for row in planet_rows[:n_real_p]
                for col in PLANET_FEATURE_COLS
            ]
            planet_features[:n_real_p] = torch.tensor(
                flat, dtype=torch.float32,
            ).view(n_real_p, PLANET_RAW_DIM)
            planet_mask[:n_real_p] = True

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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.snapshots[idx]


# ---------- Model ----------
class EntityPretrainModel(nn.Module):
    """``PlanetEntityEncoder`` + per-label heads.

    The fleet/planet encoders are held externally (frozen, ``no_grad``)
    so this module's optimizer only steps the entity encoder + heads.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.entity = PlanetEntityEncoder(
            d_model=d_model, num_owner_slots=ENTITY_NUM_OWNER_SLOTS,
        )
        K = ENTITY_NUM_OWNER_SLOTS
        # Classification heads
        self.head_earliest = nn.Linear(d_model, ENTITY_N_OWNER_CLASSES)
        self.head_is_source = nn.Linear(d_model, 1)
        self.head_is_target = nn.Linear(d_model, 1)
        self.head_owner_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, ENTITY_N_OWNER_CLASSES)
            for k in ENTITY_LABEL_HORIZONS
        })
        self.head_log_ships_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1) for k in ENTITY_LABEL_HORIZONS
        })
        self.head_arriving_h = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, K) for h in ENTITY_ARRIVAL_HORIZONS
        })

    def forward(
        self,
        planet_tokens: torch.Tensor,
        fleet_tokens: torch.Tensor,
        routing: dict[str, torch.Tensor],
        planet_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z = self.entity(
            planet_tokens,
            fleet_tokens,
            routing["fleet_target_idx"],
            routing["fleet_source_idx"],
            routing["fleet_owner_slot"],
            routing["fleet_ships_log"],
            routing["fleet_eta_norm"],
            routing["fleet_mask"],
            planet_mask=planet_mask,
        )                                                    # (B, P, d)
        out: dict[str, torch.Tensor] = {
            "earliest_arrival_owner_slot": self.head_earliest(z),
            "is_source_this_turn": self.head_is_source(z).squeeze(-1),
            "is_target_this_turn": self.head_is_target(z).squeeze(-1),
        }
        for k in ENTITY_LABEL_HORIZONS:
            out[f"owner_t_plus_{k}"] = self.head_owner_k[f"k{k}"](z)
            out[f"log_ships_t_plus_{k}"] = self.head_log_ships_k[f"k{k}"](z).squeeze(-1)
        for h in ENTITY_ARRIVAL_HORIZONS:
            out[f"ships_arriving_within_{h}"] = self.head_arriving_h[f"h{h}"](z)
        return out


# ---------- Loss ----------
def compute_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    planet_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-task loss; per-planet padding masked out for every term.

    Categorical heads use cross-entropy (mean over real planets), binary
    heads use BCE-with-logits, regression heads use MSE — all gated by
    ``planet_mask`` AND, where applicable, ``valid_t_plus_K`` so
    "no future" rows don't leak into the regression / owner-class
    losses.
    """
    losses: dict[str, float] = {}
    total: torch.Tensor | None = None
    pm = planet_mask.float()                                # (B, P)
    n_real = pm.sum().clamp(min=1.0)

    def add(name: str, term: torch.Tensor) -> None:
        nonlocal total
        total = term if total is None else total + term
        losses[name] = float(term.detach())

    # Earliest arrival owner slot (categorical, K+1 classes)
    logits = preds["earliest_arrival_owner_slot"]            # (B, P, K+1)
    tgt = targets["earliest_arrival_owner_slot"]             # (B, P)
    ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")  # (B, P)
    add("earliest_arrival_owner_slot", (ce * pm).sum() / n_real)

    # is_source / is_target (binary)
    for name in ("is_source_this_turn", "is_target_this_turn"):
        bce = F.binary_cross_entropy_with_logits(
            preds[name], targets[name], reduction="none",
        )
        add(name, (bce * pm).sum() / n_real)

    # Per-K classification + regression
    for k in ENTITY_LABEL_HORIZONS:
        valid = targets[f"valid_t_plus_{k}"] * pm           # (B, P)
        n_valid = valid.sum().clamp(min=1.0)

        # owner classification
        logits_k = preds[f"owner_t_plus_{k}"]                # (B, P, K+1)
        tgt_k = targets[f"owner_t_plus_{k}"]
        ce_k = F.cross_entropy(logits_k.transpose(1, 2), tgt_k, reduction="none")
        add(f"owner_t_plus_{k}", (ce_k * valid).sum() / n_valid)

        # log-ships regression
        pred_s = preds[f"log_ships_t_plus_{k}"]
        tgt_s = targets[f"log_ships_t_plus_{k}"]
        mse = (pred_s - tgt_s).pow(2)
        add(f"log_ships_t_plus_{k}", (mse * valid).sum() / n_valid)

    # Per-(player) arrival regression: heads emit (B, P, K), targets
    # are (B, P, K). Mask-broadcast over the K axis.
    for h in ENTITY_ARRIVAL_HORIZONS:
        pred_a = preds[f"ships_arriving_within_{h}"]
        tgt_a = targets[f"ships_arriving_within_{h}"]
        mse = (pred_a - tgt_a).pow(2)                       # (B, P, K)
        per_planet = mse.sum(-1)                             # (B, P)
        add(f"ships_arriving_within_{h}", (per_planet * pm).sum() / (n_real * ENTITY_NUM_OWNER_SLOTS))

    assert total is not None
    return total, losses


# ---------- Eval ----------
@torch.no_grad()
def evaluate(
    model: EntityPretrainModel,
    fleet_enc: FleetEncoder,
    planet_enc: PlanetEncoder,
    loader: DataLoader,
    device: str,
) -> dict[str, dict[str, float]]:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    correct: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        planet_tok = planet_enc(batch["planet_features"])
        fleet_tok = fleet_enc(batch["fleet_features"])
        routing = {
            "fleet_target_idx": batch["fleet_target_idx"],
            "fleet_source_idx": batch["fleet_source_idx"],
            "fleet_owner_slot": batch["fleet_owner_slot"],
            "fleet_ships_log": batch["fleet_ships_log"],
            "fleet_eta_norm": batch["fleet_eta_norm"],
            "fleet_mask": batch["fleet_mask"],
        }
        preds = model(planet_tok, fleet_tok, routing, batch["planet_mask"])
        pm = batch["planet_mask"].float()
        n_real = int(pm.sum())

        # Categorical accuracy
        for name, n_classes in [
            ("earliest_arrival_owner_slot", ENTITY_N_OWNER_CLASSES),
            *[(f"owner_t_plus_{k}", ENTITY_N_OWNER_CLASSES) for k in ENTITY_LABEL_HORIZONS],
        ]:
            logits = preds[name]                              # (B, P, C)
            tgt = batch[name]
            valid_mask = pm.bool()
            if name.startswith("owner_t_plus_"):
                k = int(name.split("_")[-1])
                valid_mask &= batch[f"valid_t_plus_{k}"].bool()
            argmax = logits.argmax(-1)
            hits = ((argmax == tgt) & valid_mask).sum().item()
            n = valid_mask.sum().item()
            correct[name] += hits
            counts[name] += n
            ce = F.cross_entropy(
                logits.transpose(1, 2), tgt, reduction="none",
            )
            sums[name] += (ce * valid_mask.float()).sum().item()

        # Binary accuracy (>0 ⇒ 1)
        for name in ("is_source_this_turn", "is_target_this_turn"):
            logit = preds[name]
            tgt = batch[name]
            pred_bin = (logit > 0).float()
            hits = ((pred_bin == tgt) * pm).sum().item()
            correct[name] += int(hits)
            counts[name] += n_real
            bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
            sums[name] += (bce * pm).sum().item()

        # Regression MSE
        for k in ENTITY_LABEL_HORIZONS:
            valid = batch[f"valid_t_plus_{k}"] * pm
            mse = (preds[f"log_ships_t_plus_{k}"] - batch[f"log_ships_t_plus_{k}"]).pow(2)
            sums[f"log_ships_t_plus_{k}"] += (mse * valid).sum().item()
            counts[f"log_ships_t_plus_{k}"] += int(valid.sum())

        for h in ENTITY_ARRIVAL_HORIZONS:
            pred_a = preds[f"ships_arriving_within_{h}"]
            tgt_a = batch[f"ships_arriving_within_{h}"]
            mse = (pred_a - tgt_a).pow(2).sum(-1)
            sums[f"ships_arriving_within_{h}"] += (mse * pm).sum().item()
            counts[f"ships_arriving_within_{h}"] += n_real * ENTITY_NUM_OWNER_SLOTS

    summary: dict[str, dict[str, float]] = {}
    for name, total in sums.items():
        n = max(1, counts[name])
        entry: dict[str, float] = {"loss": total / n}
        if name in correct and counts[name] > 0 and name.endswith(("_slot", "_5", "_10", "_turn")) is False:
            pass  # filled below for proper keys
        if name in (
            "earliest_arrival_owner_slot",
            "is_source_this_turn", "is_target_this_turn",
            *[f"owner_t_plus_{k}" for k in ENTITY_LABEL_HORIZONS],
        ):
            entry["acc"] = correct[name] / max(1, counts[name])
        summary[name] = entry
    model.train()
    return summary


def _format_summary(summary: dict[str, dict[str, float]]) -> str:
    lines = []
    for name, m in summary.items():
        acc = f"  acc={m['acc']:.3f}" if "acc" in m else ""
        lines.append(f"    {name:<32s}  loss={m['loss']:.4f}{acc}")
    return "\n".join(lines)


# ---------- Train loop ----------
def _load_encoders(
    fleet_run_dir: Path, planet_run_dir: Path, *, device: str,
) -> tuple[FleetEncoder, PlanetEncoder]:
    fc = torch.load(
        fleet_run_dir / "fleet_encoder_best.pt",
        map_location=device, weights_only=False,
    )
    pc = torch.load(
        planet_run_dir / "planet_encoder_best.pt",
        map_location=device, weights_only=False,
    )
    fenc = FleetEncoder(d_model=fc["config"]["d_model"])
    fenc.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in fc["model"].items()
         if k.startswith("encoder.")}
    )
    penc = PlanetEncoder(d_model=pc["config"]["d_model"])
    penc.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in pc["model"].items()
         if k.startswith("encoder.")}
    )
    fenc.to(device).eval()
    penc.to(device).eval()
    for p in fenc.parameters(): p.requires_grad_(False)
    for p in penc.parameters(): p.requires_grad_(False)
    return fenc, penc


def train(
    *,
    out_dir: Path,
    fleet_run_dir: Path,
    planet_run_dir: Path,
    d_model: int = 64,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 1,
    max_planets: int = 64,
    max_fleets: int = 1024,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # ---- Load datasets ----
    manifest = json.loads((ENTITY_DATASET_DIR / "manifest.json").read_text())

    def stems_of(split: str) -> list[str]:
        return [n.removeprefix("entity_").removesuffix(".csv") for n in manifest[split]]

    def csvs_for(stems: list[str], dataset_dir: Path, prefix: str) -> list[Path]:
        return [dataset_dir / f"{prefix}_{stem}.csv" for stem in stems]

    print("[entity-pretrain] loading CSVs ...")
    splits = {}
    for split in ("train", "val", "test"):
        stems = stems_of(split)
        splits[split] = EntitySnapshotDataset(
            planet_csv_paths=csvs_for(stems, PLANET_DATASET_DIR, "planet"),
            fleet_csv_paths=csvs_for(stems, FLEET_DATASET_DIR, "fleet"),
            entity_csv_paths=csvs_for(stems, ENTITY_DATASET_DIR, "entity"),
            max_planets=max_planets,
            max_fleets=max_fleets,
        )
        print(f"  {split}: {len(splits[split])} snapshots")

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
    fleet_enc, planet_enc = _load_encoders(fleet_run_dir, planet_run_dir, device=device)
    print(f"  fleet enc params:  {sum(p.numel() for p in fleet_enc.parameters()):,} (frozen)")
    print(f"  planet enc params: {sum(p.numel() for p in planet_enc.parameters()):,} (frozen)")

    # ---- Build entity encoder + heads ----
    model = EntityPretrainModel(d_model=d_model).to(device)
    print(f"  entity model params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    config = {
        "d_model": d_model, "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": epochs,
        "fleet_run_dir": str(fleet_run_dir),
        "planet_run_dir": str(planet_run_dir),
    }

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
                fleet_tok = fleet_enc(batch["fleet_features"])
            routing = {
                "fleet_target_idx": batch["fleet_target_idx"],
                "fleet_source_idx": batch["fleet_source_idx"],
                "fleet_owner_slot": batch["fleet_owner_slot"],
                "fleet_ships_log": batch["fleet_ships_log"],
                "fleet_eta_norm": batch["fleet_eta_norm"],
                "fleet_mask": batch["fleet_mask"],
            }
            preds = model(planet_tok, fleet_tok, routing, batch["planet_mask"])
            total_loss, per_head = compute_loss(preds, batch, batch["planet_mask"])
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            running_total += float(total_loss.detach())
            for k, v in per_head.items():
                running_per_head[k] += v
            n_batches += 1

        train_total = running_total / max(1, n_batches)
        train_per_head = {k: v / max(1, n_batches) for k, v in running_per_head.items()}
        elapsed = round(time.time() - t0, 2)
        entry: dict[str, Any] = {
            "epoch": epoch, "train_total": train_total,
            "train_per_head": train_per_head, "elapsed_s": elapsed,
        }

        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate(model, fleet_enc, planet_enc, val_loader, device)
            mean = sum(m["loss"] for m in val.values()) / max(1, len(val))
            entry["val_mean_loss"] = mean
            entry["val"] = val
            print(
                f"[ep {epoch:>2}/{epochs}]  train_total={train_total:.4f}  "
                f"val_mean={mean:.4f}  ({elapsed}s)"
            )
            if mean < best_val:
                best_val = mean
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, best_path)

        log.append(entry)
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, last_path)
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    # ---- Test with best ckpt ----
    print("\n[entity-pretrain] evaluating best on test ...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, fleet_enc, planet_enc, test_loader, device)
    print(_format_summary(test))
    (out_dir / "test_summary.json").write_text(json.dumps(test, indent=2))
    print(f"\n[entity-pretrain] outputs in {out_dir}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fleet-run-dir", type=Path,
        default=FLEET_RUNS_DIR / "20260429-194952",
        help="Source FleetEncoder checkpoint dir.",
    )
    parser.add_argument(
        "--planet-run-dir", type=Path,
        default=PLANET_RUNS_DIR / "20260429-225920",
        help="Source PlanetEncoder checkpoint dir.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    args = parser.parse_args()

    out_dir = args.out_dir or (ENTITY_RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"))
    train(
        out_dir=out_dir,
        fleet_run_dir=args.fleet_run_dir,
        planet_run_dir=args.planet_run_dir,
        d_model=args.d_model,
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
    )


if __name__ == "__main__":
    main()
