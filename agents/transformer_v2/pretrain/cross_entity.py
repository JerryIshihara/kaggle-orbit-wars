"""Cross-entity attention pretraining.

Loads the three frozen upstream encoders (Fleet, Planet, PlanetEntity)
and trains :class:`CrossEntityAttention` + supervision heads on the
Tier-1/2/3/4 cross-entity labels plus selected entity-label reuse
targets stored in
``data/datasets/cross_entity/``. See
``agents/transformer_v2/aggregator/README.md`` for the design.

Run from the repo root:

    python -m agents.transformer_v2.pretrain.cross_entity \\
        --epochs 30 --batch-size 64 --device cuda

Outputs (under ``data/runs/cross_entity/<timestamp>/``):
  * ``cross_entity_best.pt`` / ``cross_entity_last.pt``
  * ``log.json`` — per-epoch train + val per-head metrics
  * ``test_summary.json`` — per-head test metrics from the best ckpt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..aggregator import (
    CrossEntityAttention,
    PlayerConsolidator,
    TemporalCrossEntityAttention,
)
from ..encoder.entity_encoder import PlanetEntityEncoder
from ..encoder.fleet_encoder import FleetEncoder
from ..encoder.planet_encoder import PlanetEncoder
from ..history import HISTORY_OFFSETS, N_HISTORY
from ..featurizer import (
    CROSS_ENTITY_LABEL_COLS,
    CROSS_ENTITY_TACTICAL_HORIZONS,
    CROSS_ENTITY_VALUE_HORIZONS,
    ENTITY_ARRIVAL_HORIZONS,
    ENTITY_LABEL_HORIZONS,
    ENTITY_N_FRONTIER_CLASSES,
    ENTITY_N_OWNER_CLASSES,
    ENTITY_NUM_OWNER_SLOTS,
    FLEET_RAW_DIM,
    PLANET_RAW_DIM,
)
from ..paths import (
    CROSS_ENTITY_DATASET_DIR,
    CROSS_ENTITY_RUNS_DIR,
    ENTITY_DATASET_DIR,
    ENTITY_RUNS_DIR,
    FLEET_DATASET_DIR,
    FLEET_RUNS_DIR,
    PLANET_DATASET_DIR,
    PLANET_RUNS_DIR,
)
from .entity_encoder import (
    EntitySnapshotDataset,
    _load_csv_grouped_by_turn,
)
from .consolidator_heads import _build_mlp, compute_current_state_labels


SPATIAL_REGRESSION_HEADS: tuple[str, ...] = (
    "n_friendly_within_R_norm",
    "n_enemy_within_R_norm",
    "nearest_friendly_dist_norm",
    "nearest_enemy_dist_norm",
    "sector_advantage_log",
)

CURRENT_GLOBAL_REGRESSION_HEADS: tuple[str, ...] = (
    "score_advantage_at_end_log",
    "turns_until_episode_end",
)


# ---------- Dataset ----------
class CrossEntitySnapshotDataset(EntitySnapshotDataset):
    """Same per-snapshot tensors as ``EntitySnapshotDataset`` plus
    cross-entity labels keyed by ``(episode_id, turn, planet_id)``.

    Cross CSV labels may be per-planet (Tier-1 / Tier-4) or per-snapshot
    scalars/classes (Tier-3 short/long horizon value). Entity-label reuse
    targets come from the inherited entity snapshot fields.
    """

    # Feature tensors that get stacked along a new ``T`` axis so the
    # cross-entity attention sees ``n_history`` past turns + the
    # current one. Everything else (per-planet labels, CLS scalars) is
    # current-turn-only — we don't supervise predictions for past
    # frames.
    _STACK_KEYS: tuple[str, ...] = (
        "planet_features", "planet_mask",
        "fleet_features", "fleet_mask",
        "fleet_target_idx", "fleet_source_idx",
        "fleet_owner_slot", "fleet_ships_log", "fleet_eta_norm",
    )

    def __init__(
        self,
        planet_csv_paths: list[Path],
        fleet_csv_paths: list[Path],
        entity_csv_paths: list[Path],
        cross_entity_csv_paths: list[Path],
        *,
        max_planets: int = 64,
        max_fleets: int = 1024,
        learner_slot: int = 0,
        num_players: int = 4,
        n_history: int = N_HISTORY,
        num_load_workers: int | None = None,
    ):
        # Index cross-entity CSV paths by stem so the base class can
        # stream-load one stem's CSV at a time via the
        # ``_load_extra_csv_for_stem`` hook. Loading all 600+ cross-
        # entity CSVs up-front into Python dicts contributes the bulk
        # of the OOM during dataset construction; per-stem loading
        # caps peak memory at one episode's worth (~5 MB).
        self._cross_entity_paths: dict[str, Path] = {}
        for p in cross_entity_csv_paths:
            stem = p.stem.removeprefix("cross_entity_")
            self._cross_entity_paths[stem] = p
        # Holds the CURRENT stem's cross-entity rows; refreshed by the
        # ``_load_extra_csv_for_stem`` hook before each stem's snapshots
        # are built. ``_build_snapshot`` reads from this.
        self._cross_entity_by_key: dict[tuple[str, int], list[dict]] = {}
        super().__init__(
            planet_csv_paths=planet_csv_paths,
            fleet_csv_paths=fleet_csv_paths,
            entity_csv_paths=entity_csv_paths,
            max_planets=max_planets,
            max_fleets=max_fleets,
            learner_slot=learner_slot,
            num_players=num_players,
            num_load_workers=num_load_workers,
        )
        self._cross_entity_by_key.clear()

        # v2 always stacks with ``HISTORY_OFFSETS``; ``n_history`` here
        # is just the slot count we expose to callers (cache keys, ckpt
        # config). Coerce a stale v1-style ``n_history=3`` up to
        # ``N_HISTORY`` with a warning so an old notebook cell doesn't
        # crash mid-run.
        if n_history != N_HISTORY:
            import warnings
            warnings.warn(
                f"v2 uses the fixed sparse window HISTORY_OFFSETS "
                f"(len={N_HISTORY}); got n_history={n_history}. "
                f"Coercing to {N_HISTORY}. Update the caller to pass "
                f"n_history=agents.transformer_v2.history.N_HISTORY or "
                f"omit it.",
                stacklevel=2,
            )
        self.n_history = N_HISTORY

        # NOTE: ``self._key_to_idx`` is provided by the parent class as a
        # lazy ``@property`` (built once on first access, cached on the
        # instance). Don't assign to it here — that conflicts with the
        # property descriptor and raises AttributeError.

    def _filter_stems(self, stems: list[str]) -> list[str]:
        return [s for s in stems if s in self._cross_entity_paths]

    def _extra_paths_for_stem(self, stem: str) -> dict[str, Path]:
        # Bundle cross_entity into the per-stem worker call so it gets
        # parsed in parallel along with planet/fleet/entity.
        path = self._cross_entity_paths.get(stem)
        return {"cross_entity": path} if path is not None else {}

    def _load_extra_csv_for_stem(
        self,
        stem: str,
        parsed: dict[str, dict[tuple[str, int], list[dict[str, str]]]],
    ) -> None:
        # Refresh the per-stem cross-entity rows from the parsed bundle
        # (worker already parsed it in parallel; just take the dict).
        self._cross_entity_by_key.clear()
        cross_rows = parsed.get("cross_entity")
        if cross_rows is not None:
            self._cross_entity_by_key.update(cross_rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return the current snapshot stacked with the past frames at
        ``HISTORY_OFFSETS`` (oldest first). Missing past turns (start of
        episode or sparse-window gaps before the episode begins) are
        zero-filled; their masks stay all-False so the attention layer
        ignores them.
        """
        cur = self.snapshots[idx]
        ep, t = self.keys[idx]

        # Oldest-first per the step embedding convention. v2 uses the
        # sparse window from ``agents.transformer_v2.history`` rather
        # than a dense range; ``self.n_history`` is kept in sync with
        # ``len(HISTORY_OFFSETS)`` so external callers can still read
        # the slot count.
        offsets = list(HISTORY_OFFSETS)
        history_snaps: list[dict[str, torch.Tensor] | None] = []
        for off in offsets:
            prev_idx = self._key_to_idx.get((ep, t - off))
            history_snaps.append(
                self.snapshots[prev_idx] if prev_idx is not None else None
            )

        out: dict[str, torch.Tensor] = {}
        for key, val in cur.items():
            if key in self._STACK_KEYS:
                # Stack history along a new T-dim. Missing frames are
                # zero-filled; masks therefore stay all-False, which
                # the attention's key_padding_mask treats as "ignore".
                stack: list[torch.Tensor] = []
                for snap in history_snaps:
                    if snap is None:
                        stack.append(torch.zeros_like(val))
                    else:
                        stack.append(snap[key])
                out[key] = torch.stack(stack, dim=0)               # (T, ...)
            else:
                # Labels + scalar CLS targets stay at current step only.
                out[key] = val
        return out

    def _build_snapshot(
        self,
        key: tuple[str, int],
        planet_rows: list[dict[str, str]],
        fleet_rows: list[dict[str, str]],
        entity_rows: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        """Same as the base snapshot, plus cross-entity
        label tensors. Uses the same planet ordering (and so the same
        ``pid_to_idx`` map) as the base class."""
        snapshot = super()._build_snapshot(key, planet_rows, fleet_rows, entity_rows)
        P = self.max_planets

        # Recreate pid_to_idx from planet_rows (cheap, O(P)).
        pid_to_idx: dict[int, int] = {
            int(r["planet_id"]): i for i, r in enumerate(planet_rows[:P])
        }

        rows = self._cross_entity_by_key.get(key, [])
        by_pid = {int(r["planet_id"]): r for r in rows}

        frontier_class = torch.zeros(P, dtype=torch.long)
        n_friendly = torch.zeros(P, dtype=torch.float32)
        n_enemy = torch.zeros(P, dtype=torch.float32)
        nearest_friendly = torch.ones(P, dtype=torch.float32)
        nearest_enemy = torch.ones(P, dtype=torch.float32)
        sector_advantage = torch.zeros(P, dtype=torch.float32)
        tactical: dict[int, dict[str, torch.Tensor]] = {
            h: {
                "can_friendly_reinforce_within": torch.zeros(P, dtype=torch.float32),
                "enemy_can_capture_within": torch.zeros(P, dtype=torch.float32),
                "best_local_support_margin_within_log": torch.zeros(P, dtype=torch.float32),
            }
            for h in CROSS_ENTITY_TACTICAL_HORIZONS
        }

        # Per-snapshot scalars (broadcast — read once from any row).
        winner = 0
        score_adv = 0.0
        turns_left = 0.0
        leader_future = {
            h: torch.tensor(0, dtype=torch.long)
            for h in CROSS_ENTITY_VALUE_HORIZONS
        }
        score_adv_future = {
            h: torch.tensor(0.0, dtype=torch.float32)
            for h in CROSS_ENTITY_VALUE_HORIZONS
        }
        is_ahead_future = {
            h: torch.tensor(0.0, dtype=torch.float32)
            for h in CROSS_ENTITY_VALUE_HORIZONS
        }
        valid_global_future = {
            h: torch.tensor(0.0, dtype=torch.float32)
            for h in CROSS_ENTITY_VALUE_HORIZONS
        }
        # Neutral (player-perspective-independent) global labels — default to
        # zeros for the no-rows case. These are scalars per snapshot.
        total_ships_log = 0.0
        ship_dist_entropy = 0.0
        n_neutral_planets = 0.0
        game_phase = 0
        # Per-player FINAL standing (learner-relative, 0 = best) + validity
        # mask. Defaults: worst rank + all-invalid, so a cache built before
        # the featurizer added these columns loads but the critic loss finds
        # no valid rows (it asserts at least one valid slot — see compute_loss_critic).
        final_rank = torch.full(
            (ENTITY_NUM_OWNER_SLOTS,), ENTITY_NUM_OWNER_SLOTS - 1, dtype=torch.long,
        )
        player_valid = torch.zeros(ENTITY_NUM_OWNER_SLOTS, dtype=torch.float32)
        # Per-turn temporal momentum (anti-shortcut anchors). Defaults: zeros +
        # valid_fwd=0 so old caches load without delta supervision.
        dships_back = torch.zeros(ENTITY_NUM_OWNER_SLOTS, dtype=torch.float32)
        dships_fwd = torch.zeros(ENTITY_NUM_OWNER_SLOTS, dtype=torch.float32)
        survives_fwd = torch.zeros(ENTITY_NUM_OWNER_SLOTS, dtype=torch.float32)
        score_adv_slope_back = torch.zeros((), dtype=torch.float32)
        valid_fwd = torch.zeros((), dtype=torch.float32)
        if rows:
            head = rows[0]
            winner = int(head["winner_seat"])
            score_adv = float(head["score_advantage_at_end_log"])
            turns_left = float(head["turns_until_episode_end"])
            for h in CROSS_ENTITY_VALUE_HORIZONS:
                leader_future[h] = torch.tensor(
                    int(head[f"leader_seat_t_plus_{h}"]),
                    dtype=torch.long,
                )
                score_adv_future[h] = torch.tensor(
                    float(head[f"score_advantage_t_plus_{h}_log"]),
                    dtype=torch.float32,
                )
                is_ahead_future[h] = torch.tensor(
                    float(head[f"is_ahead_t_plus_{h}"]),
                    dtype=torch.float32,
                )
                valid_global_future[h] = torch.tensor(
                    float(head[f"valid_global_t_plus_{h}"]),
                    dtype=torch.float32,
                )
            # New neutral global labels — guard with .get() so a freshly-cloned
            # but not-yet-regenerated CSV (without these columns) still loads.
            total_ships_log = float(head.get("total_ships_in_play_log", 0.0) or 0.0)
            ship_dist_entropy = float(head.get("ship_distribution_entropy", 0.0) or 0.0)
            n_neutral_planets = float(head.get("n_neutral_planets", 0.0) or 0.0)
            game_phase = int(float(head.get("game_phase", 0.0) or 0.0))
            # Per-player final standing + validity (guarded for old CSVs).
            if f"final_rank_0" in head:
                final_rank = torch.tensor(
                    [int(float(head[f"final_rank_{s}"])) for s in range(ENTITY_NUM_OWNER_SLOTS)],
                    dtype=torch.long,
                )
                player_valid = torch.tensor(
                    [float(head[f"player_valid_{s}"]) for s in range(ENTITY_NUM_OWNER_SLOTS)],
                    dtype=torch.float32,
                )
            if "dships_back_0" in head:
                dships_back = torch.tensor(
                    [float(head[f"dships_back_{s}"]) for s in range(ENTITY_NUM_OWNER_SLOTS)],
                    dtype=torch.float32,
                )
                dships_fwd = torch.tensor(
                    [float(head[f"dships_fwd_{s}"]) for s in range(ENTITY_NUM_OWNER_SLOTS)],
                    dtype=torch.float32,
                )
                survives_fwd = torch.tensor(
                    [float(head[f"survives_fwd_{s}"]) for s in range(ENTITY_NUM_OWNER_SLOTS)],
                    dtype=torch.float32,
                )
                score_adv_slope_back = torch.tensor(
                    float(head["score_adv_slope_back"]), dtype=torch.float32,
                )
                valid_fwd = torch.tensor(float(head["valid_fwd"]), dtype=torch.float32)

        # Vectorize per-planet cross-entity label scatter (same trick as
        # EntitySnapshotDataset._build_snapshot — gather lists first,
        # one tensor() call, scatter).
        ce_idxs: list[int] = []
        ce_frontier: list[int] = []
        ce_n_friendly: list[float] = []
        ce_n_enemy: list[float] = []
        ce_nearest_friendly: list[float] = []
        ce_nearest_enemy: list[float] = []
        ce_sector_adv: list[float] = []
        ce_can_reinforce: dict[int, list[float]] = {h: [] for h in CROSS_ENTITY_TACTICAL_HORIZONS}
        ce_enemy_capture: dict[int, list[float]] = {h: [] for h in CROSS_ENTITY_TACTICAL_HORIZONS}
        ce_support_margin: dict[int, list[float]] = {h: [] for h in CROSS_ENTITY_TACTICAL_HORIZONS}
        for prow in planet_rows[:P]:
            pid = int(prow["planet_id"])
            crow = by_pid.get(pid)
            if crow is None:
                continue
            ce_idxs.append(pid_to_idx[pid])
            ce_frontier.append(int(crow["frontier_class"]))
            ce_n_friendly.append(float(crow["n_friendly_within_R_norm"]))
            ce_n_enemy.append(float(crow["n_enemy_within_R_norm"]))
            ce_nearest_friendly.append(float(crow["nearest_friendly_dist_norm"]))
            ce_nearest_enemy.append(float(crow["nearest_enemy_dist_norm"]))
            ce_sector_adv.append(float(crow["sector_advantage_log"]))
            for h in CROSS_ENTITY_TACTICAL_HORIZONS:
                ce_can_reinforce[h].append(float(crow[f"can_friendly_reinforce_within_{h}"]))
                ce_enemy_capture[h].append(float(crow[f"enemy_can_capture_within_{h}"]))
                ce_support_margin[h].append(float(crow[f"best_local_support_margin_within_{h}_log"]))

        if ce_idxs:
            idx_t = torch.tensor(ce_idxs, dtype=torch.long)
            frontier_class[idx_t] = torch.tensor(ce_frontier, dtype=torch.long)
            n_friendly[idx_t] = torch.tensor(ce_n_friendly, dtype=torch.float32)
            n_enemy[idx_t] = torch.tensor(ce_n_enemy, dtype=torch.float32)
            nearest_friendly[idx_t] = torch.tensor(ce_nearest_friendly, dtype=torch.float32)
            nearest_enemy[idx_t] = torch.tensor(ce_nearest_enemy, dtype=torch.float32)
            sector_advantage[idx_t] = torch.tensor(ce_sector_adv, dtype=torch.float32)
            for h in CROSS_ENTITY_TACTICAL_HORIZONS:
                tactical[h]["can_friendly_reinforce_within"][idx_t] = torch.tensor(
                    ce_can_reinforce[h], dtype=torch.float32,
                )
                tactical[h]["enemy_can_capture_within"][idx_t] = torch.tensor(
                    ce_enemy_capture[h], dtype=torch.float32,
                )
                tactical[h]["best_local_support_margin_within_log"][idx_t] = torch.tensor(
                    ce_support_margin[h], dtype=torch.float32,
                )

        snapshot.update({
            "frontier_class": frontier_class,
            "n_friendly_within_R_norm": n_friendly,
            "n_enemy_within_R_norm": n_enemy,
            "nearest_friendly_dist_norm": nearest_friendly,
            "nearest_enemy_dist_norm": nearest_enemy,
            "sector_advantage_log": sector_advantage,
            "winner_seat": torch.tensor(winner, dtype=torch.long),
            "final_rank": final_rank,
            "player_valid": player_valid,
            "dships_back": dships_back,
            "dships_fwd": dships_fwd,
            "survives_fwd": survives_fwd,
            "score_adv_slope_back": score_adv_slope_back,
            "valid_fwd": valid_fwd,
            "score_advantage_at_end_log": torch.tensor(score_adv, dtype=torch.float32),
            "turns_until_episode_end": torch.tensor(turns_left, dtype=torch.float32),
            "expert_acted_this_turn": torch.tensor(
                float(snapshot["is_source_this_turn"].max().item() > 0.0),
                dtype=torch.float32,
            ),
        })
        for h in CROSS_ENTITY_VALUE_HORIZONS:
            snapshot[f"leader_seat_t_plus_{h}"] = leader_future[h]
            snapshot[f"score_advantage_t_plus_{h}_log"] = score_adv_future[h]
            snapshot[f"is_ahead_t_plus_{h}"] = is_ahead_future[h]
            snapshot[f"valid_global_t_plus_{h}"] = valid_global_future[h]
        for h in CROSS_ENTITY_TACTICAL_HORIZONS:
            snapshot[f"can_friendly_reinforce_within_{h}"] = tactical[h]["can_friendly_reinforce_within"]
            snapshot[f"enemy_can_capture_within_{h}"] = tactical[h]["enemy_can_capture_within"]
            snapshot[f"best_local_support_margin_within_{h}_log"] = tactical[h]["best_local_support_margin_within_log"]
        # Neutral (player-perspective-independent) global labels.
        snapshot["total_ships_in_play_log"] = torch.tensor(total_ships_log, dtype=torch.float32)
        snapshot["ship_distribution_entropy"] = torch.tensor(ship_dist_entropy, dtype=torch.float32)
        snapshot["n_neutral_planets"] = torch.tensor(n_neutral_planets, dtype=torch.float32)
        snapshot["game_phase"] = torch.tensor(game_phase, dtype=torch.long)
        return snapshot


# ---------- Cached dataset (skip CSV walk on Colab) ----------
class CachedCrossEntitySnapshotDataset(torch.utils.data.Dataset):
    """Drop-in replacement for :class:`CrossEntitySnapshotDataset` that
    loads pre-built single-frame snapshots from a ``.pt`` cache file.

    Skips the per-CSV walk (5-10 min on the 180k-snapshot corpus).
    Current caches store tensorized ``columns`` + ``keys`` + a small
    ``config`` blob so ``torch.load(..., mmap=True)`` can avoid
    unpickling one Python dict per snapshot. Legacy caches that stored
    ``snapshots`` as ``list[dict[str, Tensor]]`` are still accepted.
    T-stacking at ``__getitem__`` time still consults ``HISTORY_OFFSETS``
    from ``agents.transformer_v2.history`` (the single source of truth),
    so changing the offsets does NOT require rebuilding the cache.

    Build one with :func:`build_cross_entity_cache`.
    """

    # Lifted from CrossEntitySnapshotDataset so the lazy stack matches.
    _STACK_KEYS: tuple[str, ...] = (
        "planet_features", "planet_mask",
        "comet_features", "is_comet",
        "fleet_features", "fleet_mask",
        "fleet_target_idx", "fleet_source_idx",
        "fleet_owner_slot", "fleet_ships_log", "fleet_eta_norm",
    )

    def __init__(self, payload_or_path, per_frame_label_keys=None, with_posterior=False):
        # ``per_frame_label_keys``: when set, ``__getitem__`` also stacks
        # each named label across the T history frames (yielding
        # ``{key}_perT`` with shape ``(T, *label_shape)``) and emits a
        # ``frame_valid`` mask ``(T,)`` (True where a real history frame
        # exists). Used by the V3 temporal pretrain, which supervises the
        # value/outcome heads at every frame. Default None keeps the
        # current single-frame label behavior for V1/V2.
        self.per_frame_label_keys: tuple[str, ...] = tuple(per_frame_label_keys or ())
        # ``with_posterior``: when True, also emit a FUTURE-window stack of
        # every feature key (``{key}_post``) plus a ``post_valid`` mask. The
        # posterior window is the mirror of the prior: slot j holds turn
        # ``t + HISTORY_OFFSETS[j]`` so the anchor frame ``t`` lands LAST
        # (same slot as the prior's current frame), with future frames in
        # the leading slots. Used by the critic's prior/posterior consistency
        # aux — the future-informed teacher view. Frames past the episode end
        # are zero-padded + masked.
        self.with_posterior: bool = bool(with_posterior)
        if isinstance(payload_or_path, (str, Path)):
            payload = _load_cross_entity_cache_payload(Path(payload_or_path))
        else:
            payload = payload_or_path
        self.config = payload.get("config", {})
        self.keys: list[tuple[str, int]] = payload["keys"]
        self.columns: dict[str, torch.Tensor] | None = payload.get("columns")
        self.snapshots: list[dict[str, torch.Tensor]] | None = payload.get("snapshots")
        if self.columns is None and self.snapshots is None:
            raise ValueError("cache payload must contain either columns or snapshots")
        self.max_planets = int(self.config.get("max_planets", 64))
        self.max_fleets = int(self.config.get("max_fleets", 1024))
        # ``(episode_id, turn) -> idx`` is small enough to build once. Also
        # precompute each sample's history indices so dataloader workers do
        # not redo T dictionary lookups for every sample.
        self._key_to_idx: dict[tuple[str, int], int] = {
            k: i for i, k in enumerate(self.keys)
        }
        self._history_indices = torch.full(
            (len(self.keys), len(HISTORY_OFFSETS)),
            -1,
            dtype=torch.long,
        )
        for i, (ep, t) in enumerate(self.keys):
            for j, off in enumerate(HISTORY_OFFSETS):
                self._history_indices[i, j] = self._key_to_idx.get((ep, t - off), -1)
        # Posterior (future) window indices: slot j -> turn t + HISTORY_OFFSETS[j],
        # so the anchor (offset 0) lands in the LAST slot, mirroring the prior.
        if self.with_posterior:
            self._future_indices = torch.full(
                (len(self.keys), len(HISTORY_OFFSETS)), -1, dtype=torch.long,
            )
            for i, (ep, t) in enumerate(self.keys):
                for j, off in enumerate(HISTORY_OFFSETS):
                    self._future_indices[i, j] = self._key_to_idx.get((ep, t + off), -1)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Same lazy T-stack as the live ``CrossEntitySnapshotDataset``.

        ``HISTORY_OFFSETS`` is read from the module constant — change
        ``agents.transformer_v2.history.HISTORY_OFFSETS`` to retune the
        temporal window without rebuilding the cache.
        """
        cur = self._frame(idx)
        hist_idxs = self._history_indices[idx]
        out: dict[str, torch.Tensor] = {}
        for key, val in cur.items():
            if key in self._STACK_KEYS:
                out[key] = self._stack_history(key, val, hist_idxs)
            else:
                out[key] = val
        # V3 temporal path: stack the requested label keys across frames
        # and emit a per-frame validity mask. ``_stack_history`` zero-fills
        # missing frames; ``frame_valid`` lets the loss ignore them.
        if self.per_frame_label_keys:
            for key in self.per_frame_label_keys:
                if key in cur:
                    out[f"{key}_perT"] = self._stack_history(key, cur[key], hist_idxs)
            out["frame_valid"] = (hist_idxs >= 0)
        # Posterior (future) feature window for the critic consistency aux.
        if self.with_posterior:
            fut_idxs = self._future_indices[idx]
            for key in self._STACK_KEYS:
                if key in cur:
                    out[f"{key}_post"] = self._stack_history(key, cur[key], fut_idxs)
            out["post_valid"] = (fut_idxs >= 0)
        return out

    def _frame(self, idx: int) -> dict[str, torch.Tensor]:
        if self.columns is not None:
            return {k: v[idx] for k, v in self.columns.items()}
        assert self.snapshots is not None
        return self.snapshots[idx]

    def _stack_history(
        self,
        key: str,
        cur_val: torch.Tensor,
        hist_idxs: torch.Tensor,
    ) -> torch.Tensor:
        if self.columns is not None:
            col = self.columns[key]
            valid = hist_idxs >= 0
            if bool(valid.all()):
                return col[hist_idxs]
            out = cur_val.new_zeros((len(hist_idxs),) + tuple(cur_val.shape))
            if bool(valid.any()):
                out[valid] = col[hist_idxs[valid]]
            return out

        assert self.snapshots is not None
        stack: list[torch.Tensor] = []
        for hist_idx in hist_idxs.tolist():
            if hist_idx < 0:
                stack.append(torch.zeros_like(cur_val))
            else:
                stack.append(self.snapshots[hist_idx][key])
        return torch.stack(stack, dim=0)


def _mem_available_gb() -> float | None:
    """Best-effort available host RAM on Linux/Colab."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024**2
    return None


def _load_cross_entity_cache_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"cross-entity cache not found: {path}. Run the notebook pull "
            "cell, or check CACHE_OBJECT/BUCKET."
        )
    size_gb = path.stat().st_size / 1024**3
    ram_gb = _mem_available_gb()
    ram_msg = f"; MemAvailable={ram_gb:.1f} GB" if ram_gb is not None else ""

    try:
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            # Older PyTorch builds do not expose ``mmap``.
            return torch.load(path, map_location="cpu", weights_only=False)
        except RuntimeError as exc:
            if "mmap" not in str(exc).lower():
                raise
            return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            "failed to load cross-entity cache "
            f"{path} (size={size_gb:.2f} GB{ram_msg}). "
            "If this is the full cache on Colab, it may exceed host RAM; "
            "start with the tiny cache or rebuild a capped cache via "
            "`python -m scripts.build_cross_entity_cache --out ... "
            "--per-split-cap N`. If size is near 0 GB, the GCS copy is "
            "missing or truncated."
        ) from exc


def _tensorize_snapshots(
    snapshots: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Convert ``list[dict]`` snapshots into columnar tensors.

    The legacy cache layout made Colab spend most startup time unpickling
    Python containers. One tensor per field loads much faster and supports
    mmap-backed storage in recent PyTorch versions.
    """
    if not snapshots:
        return {}
    columns: dict[str, torch.Tensor] = {}
    keys = list(snapshots[0].keys())
    for key in keys:
        vals = [snap[key] for snap in snapshots]
        if not all(torch.is_tensor(v) for v in vals):
            raise TypeError(f"snapshot field {key!r} is not tensor-valued")
        columns[key] = torch.stack(vals, dim=0)
    return columns


def build_cross_entity_cache(
    *,
    out_path: Path,
    stems: list[str] | None = None,
    split_stems: dict[str, list[str]] | None = None,
    max_planets: int = 64,
    max_fleets: int = 1024,
    cache_format: str = "tensorized",
) -> Path:
    """Walk the cross-entity CSVs once and serialize single-frame
    snapshots to ``out_path``.

    ``stems`` defaults to **all** common stems present in the four
    CSV dataset dirs. Pass a subset (e.g. 5 episode stems) to build a
    tiny smoke-test cache.

    The resulting ``.pt`` payload has shape::

        {
          "columns":   {field: Tensor[N, ...], ...},
          "keys":      [(episode_id, turn), ...],
          "config":    {
              cache_format,
              max_planets,
              max_fleets,
              n_stems,
              n_snapshots,
              history_offsets,
              stems,
              split_stems,
              stem_to_episode,
              episode_to_stem,
          },
        }

    History offsets live in ``agents.transformer_v2.history`` and are
    NOT baked into the cache — change the constant to retune T without
    rebuilding.
    """
    if cache_format not in {"tensorized", "legacy"}:
        raise ValueError("cache_format must be 'tensorized' or 'legacy'")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if stems is None:
        # All stems present in every CSV dir.
        def _stems_under(d: Path, prefix: str) -> set[str]:
            return {
                p.stem.removeprefix(prefix)
                for p in d.glob(f"{prefix}*.csv")
            }
        planet_stems = _stems_under(PLANET_DATASET_DIR, "planet_")
        fleet_stems = _stems_under(FLEET_DATASET_DIR, "fleet_")
        entity_stems = _stems_under(ENTITY_DATASET_DIR, "entity_")
        cross_stems = _stems_under(CROSS_ENTITY_DATASET_DIR, "cross_entity_")
        stems = sorted(planet_stems & fleet_stems & entity_stems & cross_stems)
    print(f"[cache-build] walking {len(stems)} stems ...", flush=True)
    ds = CrossEntitySnapshotDataset(
        planet_csv_paths=[PLANET_DATASET_DIR / f"planet_{s}.csv" for s in stems],
        fleet_csv_paths=[FLEET_DATASET_DIR / f"fleet_{s}.csv" for s in stems],
        entity_csv_paths=[ENTITY_DATASET_DIR / f"entity_{s}.csv" for s in stems],
        cross_entity_csv_paths=[
            CROSS_ENTITY_DATASET_DIR / f"cross_entity_{s}.csv" for s in stems
        ],
        max_planets=max_planets, max_fleets=max_fleets,
    )
    n = len(ds)
    episode_to_stem: dict[str, str] = {}
    stem_to_episode: dict[str, str] = {}
    for stem in stems:
        csv_path = CROSS_ENTITY_DATASET_DIR / f"cross_entity_{stem}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            row = next(reader, None)
        if row is None:
            continue
        episode_id = str(row["episode_id"])
        stem_to_episode[stem] = episode_id
        episode_to_stem[episode_id] = stem
    print(
        f"[cache-build] saving {n:,} snapshots to {out_path} "
        f"(format={cache_format}) ...",
        flush=True,
    )
    config = {
        "cache_format": cache_format,
        "max_planets": max_planets,
        "max_fleets": max_fleets,
        "n_stems": len(stems),
        "n_snapshots": n,
        "history_offsets": list(HISTORY_OFFSETS),
        "stems": list(stems),
        "split_stems": split_stems or {},
        "stem_to_episode": stem_to_episode,
        "episode_to_stem": episode_to_stem,
    }
    payload = {
        "keys": ds.keys,
        "config": config,
    }
    if cache_format == "tensorized":
        payload["columns"] = _tensorize_snapshots(ds.snapshots)
    else:
        payload["snapshots"] = ds.snapshots
    torch.save(payload, out_path)
    print(
        f"[cache-build] done. cache size: "
        f"{out_path.stat().st_size / 1024**3:.2f} GB",
        flush=True,
    )
    return out_path


# ---------- Model ----------
class CrossEntityPretrainModel(nn.Module):
    """``CrossEntityAttention`` + label heads. The three upstream
    encoders + the entity encoder are held externally (frozen, no_grad)
    so this module's optimizer steps cross-attention + heads only.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.cross = CrossEntityAttention(
            d_model=d_model, n_heads=8, n_layers=2,
            n_steps=N_HISTORY,
        )
        # Per-planet heads (operate on contextual_tokens, shape (B,P,d))
        self.head_frontier = nn.Linear(d_model, ENTITY_N_FRONTIER_CLASSES)
        self.head_n_friendly = nn.Linear(d_model, 1)
        self.head_n_enemy = nn.Linear(d_model, 1)
        self.head_nearest_friendly = nn.Linear(d_model, 1)
        self.head_nearest_enemy = nn.Linear(d_model, 1)
        self.head_sector_adv = nn.Linear(d_model, 1)
        # Category 1: existing entity labels at the cross layer.
        self.head_earliest = nn.Linear(d_model, ENTITY_N_OWNER_CLASSES)
        self.head_owner_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, ENTITY_N_OWNER_CLASSES)
            for k in ENTITY_LABEL_HORIZONS
        })
        self.head_log_ships_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1)
            for k in ENTITY_LABEL_HORIZONS
        })
        self.head_arriving_h = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, ENTITY_NUM_OWNER_SLOTS)
            for h in ENTITY_ARRIVAL_HORIZONS
        })
        # Category 2: imitation / expert action.
        self.head_expert_source = nn.Linear(d_model, 1)
        self.head_expert_target = nn.Linear(d_model, 1)
        # Category 4: tactical local labels.
        self.head_can_reinforce_h = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, 1)
            for h in CROSS_ENTITY_TACTICAL_HORIZONS
        })
        self.head_enemy_capture_h = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, 1)
            for h in CROSS_ENTITY_TACTICAL_HORIZONS
        })
        self.head_support_margin_h = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, 1)
            for h in CROSS_ENTITY_TACTICAL_HORIZONS
        })
        # CLS-level heads (operate on global_token, shape (B, d))
        self.head_winner = nn.Linear(d_model, ENTITY_N_OWNER_CLASSES)
        self.head_score_adv = nn.Linear(d_model, 1)
        self.head_turns_left = nn.Linear(d_model, 1)
        self.head_expert_acted = nn.Linear(d_model, 1)
        # Category 3: short-horizon global value.
        self.head_leader_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, ENTITY_N_OWNER_CLASSES)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        self.head_score_adv_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        self.head_is_ahead_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })

    def forward(
        self,
        entity_tokens: torch.Tensor,    # (B, T, P, d) or (B, P, d)
        entity_mask: torch.Tensor,       # (B, T, P) or (B, P) bool
    ) -> dict[str, torch.Tensor]:
        ctx, glob = self.cross(entity_tokens, entity_mask)
        # Per-planet heads supervise the CURRENT step only — past steps
        # are context for attention but their per-token outputs are
        # unsupervised. ``ctx`` is (B, T, P, d) for multi-step or
        # (B, P, d) for single-step (cross attention auto-squeezes).
        ctx_now = ctx[:, -1] if ctx.dim() == 4 else ctx           # (B, P, d)
        out: dict[str, torch.Tensor] = {
            "frontier_class": self.head_frontier(ctx_now),
            "n_friendly_within_R_norm": self.head_n_friendly(ctx_now).squeeze(-1),
            "n_enemy_within_R_norm": self.head_n_enemy(ctx_now).squeeze(-1),
            "nearest_friendly_dist_norm": self.head_nearest_friendly(ctx_now).squeeze(-1),
            "nearest_enemy_dist_norm": self.head_nearest_enemy(ctx_now).squeeze(-1),
            "sector_advantage_log": self.head_sector_adv(ctx_now).squeeze(-1),
            "earliest_arrival_owner_slot": self.head_earliest(ctx_now),
            "expert_source_logit": self.head_expert_source(ctx_now).squeeze(-1),
            "expert_target_logit": self.head_expert_target(ctx_now).squeeze(-1),
            # CLS / global
            "winner_seat": self.head_winner(glob),
            "score_advantage_at_end_log": self.head_score_adv(glob).squeeze(-1),
            "turns_until_episode_end": torch.sigmoid(
                self.head_turns_left(glob).squeeze(-1)
            ),
            "expert_acted_this_turn": self.head_expert_acted(glob).squeeze(-1),
        }
        for k in ENTITY_LABEL_HORIZONS:
            out[f"owner_t_plus_{k}"] = self.head_owner_k[f"k{k}"](ctx_now)
            out[f"log_ships_t_plus_{k}"] = self.head_log_ships_k[f"k{k}"](ctx_now).squeeze(-1)
        for h in ENTITY_ARRIVAL_HORIZONS:
            out[f"ships_arriving_within_{h}"] = self.head_arriving_h[f"h{h}"](ctx_now)
        for h in CROSS_ENTITY_TACTICAL_HORIZONS:
            out[f"can_friendly_reinforce_within_{h}"] = self.head_can_reinforce_h[f"h{h}"](ctx_now).squeeze(-1)
            out[f"enemy_can_capture_within_{h}"] = self.head_enemy_capture_h[f"h{h}"](ctx_now).squeeze(-1)
            out[f"best_local_support_margin_within_{h}_log"] = self.head_support_margin_h[f"h{h}"](ctx_now).squeeze(-1)
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"leader_seat_t_plus_{k}"] = self.head_leader_k[f"k{k}"](glob)
            out[f"score_advantage_t_plus_{k}_log"] = self.head_score_adv_k[f"k{k}"](glob).squeeze(-1)
            out[f"is_ahead_t_plus_{k}"] = self.head_is_ahead_k[f"k{k}"](glob).squeeze(-1)
        return out


# ============================================================================ #
# V2: PlayerConsolidator-aware L2 pretrain                                       #
# ============================================================================ #
# Augments the L2 stack with the PlayerConsolidator (4 learned per-player CLS
# tokens reading from L2's ``ctx_now``). The head menu is slimmed to advantage +
# future-state tasks only — the per-planet spatial/frontier/expert/tactical heads
# are dropped. Per-player heads (winner, leader_k) read from all 4 ``player_state``
# slots; learner-relative heads (is_ahead_k, score_adv_k, score_adv_end,
# turns_left) read from ``player_state[:, 0]`` (the learner's slot under the
# featurizer's learner-relative ownership convention).

_PLANET_OWNER_START_IDX: int = 1
_PLANET_OWNER_END_IDX: int = 1 + ENTITY_N_OWNER_CLASSES  # 5-dim one-hot (slice [1:6])


def _owner_oh_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Slice the 5-dim learner-relative owner one-hot from raw planet features.

    Returns a tensor matching ``batch["planet_features"]`` leading shape
    minus the feature dim, plus a trailing ``ENTITY_N_OWNER_CLASSES`` axis.
    Both ``(B, T, P, F)`` and ``(B, P, F)`` are accepted; the consolidator
    consumes the current-step slice (``[:, -1]``) when given a temporal stack.
    """
    pf = batch["planet_features"]
    return pf[..., _PLANET_OWNER_START_IDX:_PLANET_OWNER_END_IDX]


class CrossEntityPretrainModelV2(nn.Module):
    """L2 + PlayerConsolidator + advantage/future heads (per-player + learner).

    Head menu (12 heads, 24 scalar outputs total):

      Per-player (read each of 4 ``player_state`` slots, BCE):
        winner_p                     binary "this player wins"
        leader_k{10,20,50}_p         binary "this player leads at t+K"

      Pairwise (read ``glob`` + ordered pairs of ``player_state`` slots):
        current_pair_ahead           binary "player i currently ranks above j"
        winner_pair                  binary "winner beats this other player"
        leader_k{10,20,50}_pair      binary "future leader beats this other player"

      Learner-relative (read ``player_state[:, 0]``):
        is_ahead_k{10,20,50}         BCE
        score_advantage_t_plus_K_log MSE
        score_advantage_at_end_log   MSE
        turns_until_episode_end      MSE on sigmoid

    All upstream encoders (L0 fleet + L0 planet + L1 entity) are held
    externally and frozen during pretrain. This module's optimizer steps
    cross-attention + Consolidator + the head menu.
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        consolidator_n_heads: int = 8,
        consolidator_ff_mult: int = 4,
    ):
        super().__init__()
        self.cross = CrossEntityAttention(
            d_model=d_model, n_heads=8, n_layers=2, n_steps=N_HISTORY,
        )
        self.consolidator = PlayerConsolidator(
            d_model=d_model,
            n_heads=consolidator_n_heads,
            ff_mult=consolidator_ff_mult,
        )
        # Per-player heads — applied to each of the 4 ``player_state`` slots.
        self.head_winner_p = nn.Linear(d_model, 1)
        self.head_leader_k_p = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        # Pairwise result heads. Features are:
        # [glob, state_i, state_j, state_i - state_j, state_i * state_j].
        pair_in = d_model * 5
        self.head_current_pair = nn.Sequential(
            nn.LayerNorm(pair_in),
            nn.Linear(pair_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.head_winner_pair = nn.Sequential(
            nn.LayerNorm(pair_in),
            nn.Linear(pair_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.head_leader_k_pair = nn.ModuleDict({
            f"k{k}": nn.Sequential(
                nn.LayerNorm(pair_in),
                nn.Linear(pair_in, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        # Learner-relative heads — read ``player_state[:, 0]`` only.
        self.head_is_ahead_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        self.head_score_adv_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        self.head_score_adv_end = nn.Linear(d_model, 1)
        self.head_turns_left = nn.Linear(d_model, 1)

    def _pair_features(
        self,
        player_state: torch.Tensor,      # (B, 4, d)
        glob: torch.Tensor,              # (B, d)
    ) -> torch.Tensor:
        """Build ordered player-pair features with global context.

        Output shape is ``(B, 4, 4, 5*d)``. Diagonal cells are still
        emitted; the loss masks them out. Keeping the full square makes
        the feature layout easy to inspect and aligns with the target
        tensors.
        """
        B, S, d = player_state.shape
        left = player_state.unsqueeze(2).expand(B, S, S, d)
        right = player_state.unsqueeze(1).expand(B, S, S, d)
        glob_pair = glob.view(B, 1, 1, d).expand(B, S, S, d)
        return torch.cat(
            [glob_pair, left, right, left - right, left * right],
            dim=-1,
        )

    def forward(
        self,
        entity_tokens: torch.Tensor,        # (B, T, P, d) or (B, P, d)
        entity_mask: torch.Tensor,          # (B, T, P) or (B, P) bool
        planet_owner_oh: torch.Tensor,      # (B, T, P, 5) or (B, P, 5)
    ) -> dict[str, torch.Tensor]:
        ctx, glob = self.cross(entity_tokens, entity_mask)
        # Current-step slice for the Consolidator (it consumes (B, P, d)).
        ctx_now = ctx[:, -1] if ctx.dim() == 4 else ctx
        mask_now = entity_mask[:, -1] if entity_mask.dim() == 3 else entity_mask
        owner_now = (
            planet_owner_oh[:, -1] if planet_owner_oh.dim() == 4 else planet_owner_oh
        ).to(ctx_now.dtype)
        player_state = self.consolidator(ctx_now, mask_now, owner_now)  # (B, 4, d)

        out: dict[str, torch.Tensor] = {}
        # Per-player BCE heads — squeeze the last dim so shape is (B, 4).
        out["winner_p"] = self.head_winner_p(player_state).squeeze(-1)
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"leader_k{k}_p"] = self.head_leader_k_p[f"k{k}"](player_state).squeeze(-1)

        # Ordered player-pair result heads. These consume both the global
        # CLS and the per-player states.
        pair_feat = self._pair_features(player_state, glob)
        out["current_pair_ahead"] = self.head_current_pair(pair_feat).squeeze(-1)
        out["winner_pair"] = self.head_winner_pair(pair_feat).squeeze(-1)
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"leader_k{k}_pair"] = self.head_leader_k_pair[f"k{k}"](pair_feat).squeeze(-1)

        # Learner-relative heads — read slot 0 only.
        learner = player_state[:, 0]
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"is_ahead_t_plus_{k}"] = self.head_is_ahead_k[f"k{k}"](learner).squeeze(-1)
            out[f"score_advantage_t_plus_{k}_log"] = self.head_score_adv_k[f"k{k}"](learner).squeeze(-1)
        out["score_advantage_at_end_log"] = self.head_score_adv_end(learner).squeeze(-1)
        out["turns_until_episode_end"] = torch.sigmoid(
            self.head_turns_left(learner).squeeze(-1)
        )

        # Expose the intermediate so downstream code can introspect or
        # branch (e.g., for ablations that probe the per-player CLS).
        out["player_state"] = player_state
        return out


# ---- V2 loss ----------------------------------------------------------------
def _per_player_targets_from_class(
    cls_target: torch.Tensor,                     # (B,) long, values in [0, 4]
    *,
    n_players: int = ENTITY_NUM_OWNER_SLOTS,
) -> torch.Tensor:
    """Build ``(B, n_players)`` one-hot binary targets from a class index.

    The class indexing is learner-relative: ``0`` = learner, ``1..3`` = the
    three opponents in fixed order, ``4`` (if present) = neutral / tied /
    unresolved. Per-player binary BCE targets ignore class ``4`` by
    producing all-zeros for that row (the loss treats it as "no player
    won/leads", which is the correct semantic).
    """
    one_hot = torch.zeros(
        cls_target.shape[0], n_players,
        dtype=torch.float32, device=cls_target.device,
    )
    in_range = (cls_target >= 0) & (cls_target < n_players)
    rows = torch.where(in_range)[0]
    cols = cls_target[in_range].long()
    one_hot[rows, cols] = 1.0
    return one_hot


def _pair_targets_from_winner_class(
    cls_target: torch.Tensor,                     # (B,) long, values in [0, 4]
    *,
    valid: torch.Tensor | None = None,            # (B,) float/bool
    n_players: int = ENTITY_NUM_OWNER_SLOTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ordered pair labels from a winner/leader class.

    The existing dataset gives one class id for winner/future leader, not
    a full final ranking. To avoid inventing labels among non-winners, the
    pair task supervises only pairs involving the winning/leading slot:

      * target[i, j] = 1 when i is the winner/leader and j is another slot.
      * target[i, j] = 0 when j is the winner/leader and i is another slot.
      * pairs among non-winners and all diagonal cells are masked out.

    Class ``4`` means neutral/tie/unresolved and produces no valid pairs.
    """
    B = cls_target.shape[0]
    device = cls_target.device
    target = torch.zeros(B, n_players, n_players, dtype=torch.float32, device=device)
    mask = torch.zeros_like(target)
    row_valid = (cls_target >= 0) & (cls_target < n_players)
    if valid is not None:
        row_valid = row_valid & (valid.to(device=device).float() > 0.5)
    rows = torch.where(row_valid)[0]
    if rows.numel() == 0:
        return target, mask

    winners = cls_target[row_valid].long()
    slots = torch.arange(n_players, device=device)
    for row, winner in zip(rows.tolist(), winners.tolist(), strict=False):
        others = slots[slots != winner]
        target[row, winner, others] = 1.0
        mask[row, winner, others] = 1.0
        mask[row, others, winner] = 1.0
    return target, mask


def _current_pair_targets_from_batch(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Current ordered pair labels from raw planet/fleet snapshot tensors."""
    labels = compute_current_state_labels(
        planet_features=batch["planet_features"],
        planet_mask=batch["planet_mask"],
        fleet_features=batch["fleet_features"],
        fleet_mask=batch["fleet_mask"],
    )
    # Same strength convention as CurrentStateHead labels: combined ships
    # with a tiny planet-count tie-breaker. Tied/dead pairs are masked.
    strength = labels.total_ships_log + 1e-3 * labels.planet_count
    diff = strength.unsqueeze(2) - strength.unsqueeze(1)          # (B, 4, 4)
    alive_pair = labels.alive.bool().unsqueeze(2) & labels.alive.bool().unsqueeze(1)
    S = strength.shape[-1]
    eye = torch.eye(S, dtype=torch.bool, device=strength.device).unsqueeze(0)
    mask = alive_pair & ~eye & (diff.abs() > 1e-8)
    target = (diff > 0).to(torch.float32)
    return target, mask.to(torch.float32)


def _masked_pair_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denom = mask.float().sum().clamp(min=1.0)
    return (bce * mask.float()).sum() / denom


def compute_loss_v2(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the V2 advantage/future loss.

    Per-player heads are BCE'd against derived one-hot targets from the
    existing ``winner_seat`` / ``leader_seat_t_plus_K`` labels. Future-K
    heads are masked by ``valid_global_t_plus_K`` (the dataset builder
    sets this to 0 when t+K is past the terminal step).
    """
    losses: dict[str, float] = {}
    total: torch.Tensor | None = None

    def add(name: str, term: torch.Tensor) -> None:
        nonlocal total
        total = term if total is None else total + term
        losses[name] = float(term.detach())

    # Per-player BCE: winner_p
    winner_t = _per_player_targets_from_class(targets["winner_seat"])
    add(
        "winner_p",
        F.binary_cross_entropy_with_logits(preds["winner_p"], winner_t),
    )
    winner_pair_t, winner_pair_m = _pair_targets_from_winner_class(
        targets["winner_seat"],
    )
    add(
        "winner_pair",
        _masked_pair_bce(preds["winner_pair"], winner_pair_t, winner_pair_m),
    )

    current_pair_t, current_pair_m = _current_pair_targets_from_batch(targets)
    add(
        "current_pair_ahead",
        _masked_pair_bce(
            preds["current_pair_ahead"], current_pair_t, current_pair_m,
        ),
    )

    # Per-player BCE: leader_k_p (masked by valid_global_t_plus_K)
    for k in CROSS_ENTITY_VALUE_HORIZONS:
        leader_t = _per_player_targets_from_class(targets[f"leader_seat_t_plus_{k}"])
        valid = targets.get(f"valid_global_t_plus_{k}")
        bce = F.binary_cross_entropy_with_logits(
            preds[f"leader_k{k}_p"], leader_t, reduction="none",
        )                                                 # (B, 4)
        if valid is not None:
            valid_b = valid.float().unsqueeze(-1)         # (B, 1)
            num = (bce * valid_b).sum()
            den = valid_b.sum().clamp(min=1.0) * bce.shape[-1]
            add(f"leader_k{k}_p", num / den)
        else:
            add(f"leader_k{k}_p", bce.mean())

        leader_pair_t, leader_pair_m = _pair_targets_from_winner_class(
            targets[f"leader_seat_t_plus_{k}"],
            valid=valid,
        )
        add(
            f"leader_k{k}_pair",
            _masked_pair_bce(
                preds[f"leader_k{k}_pair"], leader_pair_t, leader_pair_m,
            ),
        )

    # Learner-relative BCE: is_ahead_k (masked)
    for k in CROSS_ENTITY_VALUE_HORIZONS:
        valid = targets.get(f"valid_global_t_plus_{k}")
        target = targets[f"is_ahead_t_plus_{k}"].float()
        bce = F.binary_cross_entropy_with_logits(
            preds[f"is_ahead_t_plus_{k}"], target, reduction="none",
        )                                                 # (B,)
        if valid is not None:
            num = (bce * valid.float()).sum()
            den = valid.float().sum().clamp(min=1.0)
            add(f"is_ahead_t_plus_{k}", num / den)
        else:
            add(f"is_ahead_t_plus_{k}", bce.mean())

    # Learner-relative MSE: score_adv_k (masked)
    for k in CROSS_ENTITY_VALUE_HORIZONS:
        valid = targets.get(f"valid_global_t_plus_{k}")
        target = targets[f"score_advantage_t_plus_{k}_log"]
        mse = (preds[f"score_advantage_t_plus_{k}_log"] - target).pow(2)
        if valid is not None:
            num = (mse * valid.float()).sum()
            den = valid.float().sum().clamp(min=1.0)
            add(f"score_advantage_t_plus_{k}_log", num / den)
        else:
            add(f"score_advantage_t_plus_{k}_log", mse.mean())

    # Terminal heads (always valid for a completed episode).
    add(
        "score_advantage_at_end_log",
        F.mse_loss(preds["score_advantage_at_end_log"], targets["score_advantage_at_end_log"]),
    )
    add(
        "turns_until_episode_end",
        F.mse_loss(preds["turns_until_episode_end"], targets["turns_until_episode_end"]),
    )

    assert total is not None
    return total, losses


# ============================================================================ #
# V3: temporal (per-frame) player/glob L2 — block-causal, deep supervision      #
# ============================================================================ #
# The summary tokens (CLS + 4 player + neutral) live INSIDE L2 and are emitted
# per frame under block-causal masking (frame t sees only frames <= t). Every
# frame's value/outcome heads are supervised against that frame's own labels —
# ~T× denser supervision than V2's single anchor frame, which directly attacks
# the Cluster-C value-head overfit. The deployed critic reads the LAST frame.
#
# Head set mirrors V2 (per-player + pairwise + learner-relative); the only
# structural change is the temporal trunk and per-frame supervision.

_V3_N_SLOT_TOKENS: int = ENTITY_N_OWNER_CLASSES          # 4 players + neutral
_V3_REAL_PLAYER_SLICE = slice(0, ENTITY_NUM_OWNER_SLOTS) # first 4 slots only
_V3_DIAGNOSTIC_KEYS: frozenset[str] = frozenset({
    "glob",
    "player_state",
    "slot_state",
    "neutral_state",
})

# Label keys the V3 dataset must stack per-frame (consumed by compute_loss_v3).
V3_PER_FRAME_LABEL_KEYS: tuple[str, ...] = (
    "winner_seat",
    "score_advantage_at_end_log",
    "turns_until_episode_end",
    *[f"leader_seat_t_plus_{k}" for k in CROSS_ENTITY_VALUE_HORIZONS],
    *[f"is_ahead_t_plus_{k}" for k in CROSS_ENTITY_VALUE_HORIZONS],
    *[f"score_advantage_t_plus_{k}_log" for k in CROSS_ENTITY_VALUE_HORIZONS],
    *[f"valid_global_t_plus_{k}" for k in CROSS_ENTITY_VALUE_HORIZONS],
)


def _pair_features_temporal(
    player_state: torch.Tensor,      # (B, T, 4, d)
    glob: torch.Tensor,              # (B, T, d)
) -> torch.Tensor:
    """Ordered player-pair features with global context, per frame.

    Output ``(B, T, 4, 4, 5*d)`` = ``[glob, sᵢ, sⱼ, sᵢ-sⱼ, sᵢ*sⱼ]``.
    """
    B, T, S, d = player_state.shape
    left = player_state.unsqueeze(3).expand(B, T, S, S, d)
    right = player_state.unsqueeze(2).expand(B, T, S, S, d)
    glob_pair = glob.view(B, T, 1, 1, d).expand(B, T, S, S, d)
    return torch.cat([glob_pair, left, right, left - right, left * right], dim=-1)


class CrossEntityPretrainModelV3(nn.Module):
    """Temporal L2 (per-frame CLS + player/neutral slots) + V2 head set.

    Forward returns per-frame predictions ``(B, T, ...)``. The same head
    modules as V2 are applied at every frame; the loss supervises all
    valid frames. The temporal trunk carries 5 learner-relative slot tokens:
    4 real players plus neutral. The supervised player-result heads consume
    only the first 4 real-player slots; the neutral slot is exposed as
    diagnostics/context. Inference / the PPO critic read ``[:, -1]``.
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_mult: int = 2,
    ):
        super().__init__()
        self.cross = TemporalCrossEntityAttention(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            ff_mult=ff_mult, n_steps=N_HISTORY, n_players=_V3_N_SLOT_TOKENS,
        )
        # Per-player heads (read each of 4 player slots).
        self.head_winner_p = nn.Linear(d_model, 1)
        self.head_leader_k_p = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1) for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        # Pairwise heads (read [glob, sᵢ, sⱼ, sᵢ-sⱼ, sᵢ*sⱼ]).
        pair_in = d_model * 5
        def _pair_head() -> nn.Module:
            return nn.Sequential(
                nn.LayerNorm(pair_in), nn.Linear(pair_in, d_model),
                nn.GELU(), nn.Linear(d_model, 1),
            )
        self.head_current_pair = _pair_head()
        self.head_winner_pair = _pair_head()
        self.head_leader_k_pair = nn.ModuleDict({
            f"k{k}": _pair_head() for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        # Learner-relative heads (read player_state[:, :, 0]).
        self.head_is_ahead_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1) for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        self.head_score_adv_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1) for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        self.head_score_adv_end = nn.Linear(d_model, 1)
        self.head_turns_left = nn.Linear(d_model, 1)

    def forward(
        self,
        entity_tokens: torch.Tensor,        # (B, T, P, d)
        entity_mask: torch.Tensor,          # (B, T, P) bool
        planet_owner_oh: torch.Tensor,      # (B, T, P, 5)
    ) -> dict[str, torch.Tensor]:
        if entity_tokens.dim() != 4:
            raise ValueError("V3 requires (B, T, P, d) entity tokens")
        owner = planet_owner_oh.to(entity_tokens.dtype)
        _ctx, glob, slot_state = self.cross(entity_tokens, entity_mask, owner)
        # glob (B,T,d), slot_state (B,T,5,d). First four slots are real
        # player states; the fifth is the neutral-summary slot.
        player_state = slot_state[:, :, _V3_REAL_PLAYER_SLICE]

        out: dict[str, torch.Tensor] = {}
        out["winner_p"] = self.head_winner_p(player_state).squeeze(-1)   # (B,T,4)
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"leader_k{k}_p"] = self.head_leader_k_p[f"k{k}"](player_state).squeeze(-1)

        pair_feat = _pair_features_temporal(player_state, glob)          # (B,T,4,4,5d)
        out["current_pair_ahead"] = self.head_current_pair(pair_feat).squeeze(-1)  # (B,T,4,4)
        out["winner_pair"] = self.head_winner_pair(pair_feat).squeeze(-1)
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"leader_k{k}_pair"] = self.head_leader_k_pair[f"k{k}"](pair_feat).squeeze(-1)

        learner = player_state[:, :, 0]                                  # (B,T,d)
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"is_ahead_t_plus_{k}"] = self.head_is_ahead_k[f"k{k}"](learner).squeeze(-1)
            out[f"score_advantage_t_plus_{k}_log"] = self.head_score_adv_k[f"k{k}"](learner).squeeze(-1)
        out["score_advantage_at_end_log"] = self.head_score_adv_end(learner).squeeze(-1)
        out["turns_until_episode_end"] = torch.sigmoid(
            self.head_turns_left(learner).squeeze(-1)
        )
        out["player_state"] = player_state
        out["slot_state"] = slot_state
        out["neutral_state"] = slot_state[:, :, ENTITY_NUM_OWNER_SLOTS]
        out["glob"] = glob
        return out


def _flatten_valid_frames(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    frame_valid: torch.Tensor,           # (B, T) bool
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """Flatten (B, T) → N and select only valid frames.

    Returns (preds_flat_valid, targets_flat_valid, n_valid) where the
    target dict uses the V2 key names (``_perT`` suffix stripped, and the
    stacked planet/fleet features renamed for ``current_pair`` labels).
    """
    B, T = frame_valid.shape
    vsel = frame_valid.reshape(B * T).bool()
    n_valid = int(vsel.sum())
    if n_valid == 0:
        return {}, {}, 0

    def fv(x: torch.Tensor) -> torch.Tensor:
        # collapse leading (B, T) → N, then select valid rows.
        return x.reshape(B * T, *x.shape[2:])[vsel]

    preds_v = {k: fv(v) for k, v in preds.items() if k not in _V3_DIAGNOSTIC_KEYS}

    targets_v: dict[str, torch.Tensor] = {}
    # per-frame labels carried as ``{key}_perT``.
    for key in V3_PER_FRAME_LABEL_KEYS:
        pt = batch.get(f"{key}_perT")
        if pt is not None:
            targets_v[key] = fv(pt)
    # current-state pair labels are computed from per-frame raw features
    # (these are T-stacked feature keys, present at (B, T, ...)).
    for key in ("planet_features", "planet_mask", "fleet_features", "fleet_mask"):
        if key in batch:
            targets_v[key] = fv(batch[key])
    return preds_v, targets_v, n_valid


def compute_loss_v3(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-frame V2 loss over all valid frames (block-causal deep supervision).

    Flattens (B, T) → N, drops frames with no real history, and reuses
    :func:`compute_loss_v2` on the flattened valid batch.
    """
    frame_valid = batch["frame_valid"]
    preds_v, targets_v, n_valid = _flatten_valid_frames(preds, batch, frame_valid)
    if n_valid == 0:
        # Degenerate batch (shouldn't happen — current frame is always
        # valid). Return a zero connected to the graph.
        z = preds["winner_p"].sum() * 0.0
        return z, {"empty": 0.0}
    return compute_loss_v2(preds_v, targets_v)


# ============================================================================ #
# CRITIC: pairwise value model on V2 player_state + per-player is_ahead          #
# ============================================================================ #
# The deployable critic is relative: a shared pairwise head reads
# ``[player_i || player_j || glob]`` and predicts whether player i is ahead of
# player j. PPO maps the learner slot's mean pairwise win probability against
# valid opponents to the terminal-reward scale. A companion per-player
# ``is_ahead`` head predicts the future leader class at each cached value
# horizon, derived from existing ``leader_seat_t_plus_K`` labels.


class ValueDecoder(nn.Module):
    """Per-player value head over a ``[glob ‖ player_state]`` feature -> (B, 4).

    Each player's value sees BOTH the global board summary (``glob`` CLS,
    broadcast) and its own ``player_state`` slot — so the head has board
    context plus per-player detail. Shared trunk applied batched over the
    slot axis; the final Linear is zero-initialized so V(s) starts at
    sigmoid(0)=0.5 (no GAE bootstrap shock).
    """

    def __init__(
        self,
        in_dim: int,
        *,
        hidden: int,
        trunk_n_layers: int = 2,
        head_n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        h = hidden
        trunk: list[nn.Module] = []
        for i in range(trunk_n_layers):
            trunk.append(nn.Linear(in_dim if i == 0 else h, h))
            trunk.append(nn.GELU())
            if dropout > 0.0:
                trunk.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*trunk)
        self.win_head = _build_mlp(
            in_dim=h, hidden=h, out_dim=1, n_layers=head_n_layers, dropout=dropout,
        )
        last = self.win_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.win_head(self.trunk(feat)).squeeze(-1)   # (B, 4)


class PairCompareHead(nn.Module):
    """P(player i outperforms player j) from ``[player_i || player_j || glob]``.

    The final layer is zero-initialized so the initial pair probability is
    ``sigmoid(0)=0.5`` for every ordered pair.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp = _build_mlp(
            in_dim=3 * d_model,
            hidden=hidden,
            out_dim=1,
            n_layers=n_layers,
            dropout=dropout,
        )
        last = self.mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, player_state: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        B, S, d = player_state.shape
        left = player_state.unsqueeze(2).expand(B, S, S, d)
        right = player_state.unsqueeze(1).expand(B, S, S, d)
        glob_pair = glob.view(B, 1, 1, d).expand(B, S, S, d)
        feat = torch.cat([left, right, glob_pair], dim=-1)
        return self.mlp(feat).squeeze(-1)                    # (B, S, S)


class CrossEntityCriticModel(nn.Module):
    """V2 backbone (L2 + PlayerConsolidator) + pairwise value & momentum heads.

    Backbone ctor args are IDENTICAL to ``CrossEntityPretrainModelV2`` so a V2
    ckpt's ``cross.*`` / ``consolidator.*`` keys load cleanly (strict=False).
    The value head is ``PairCompareHead`` over ordered player pairs, plus a
    per-player ``is_ahead`` horizon head. Momentum heads
    (``dships_back``/``slope_back`` for the prior view;
    ``dships_fwd``/``survives_fwd`` for the posterior) are the anti-shortcut
    temporal anchors.
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        consolidator_n_heads: int = 8,
        consolidator_ff_mult: int = 4,
        value_trunk_n_layers: int = 2,
        value_head_n_layers: int = 2,
    ):
        super().__init__()
        self.cross = CrossEntityAttention(
            d_model=d_model, n_heads=8, n_layers=2, n_steps=N_HISTORY,
        )
        self.consolidator = PlayerConsolidator(
            d_model=d_model,
            n_heads=consolidator_n_heads,
            ff_mult=consolidator_ff_mult,
        )
        feat_dim = 2 * d_model                                # [glob ‖ player_state]
        self.pair_compare = PairCompareHead(
            d_model,
            hidden=d_model,
            n_layers=max(1, value_trunk_n_layers + value_head_n_layers - 1),
        )
        self.is_ahead_head = _build_mlp(
            in_dim=feat_dim,
            hidden=d_model,
            out_dim=len(CROSS_ENTITY_VALUE_HORIZONS),
            n_layers=value_head_n_layers,
        )
        last = self.is_ahead_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        # Temporal momentum heads (per-player) on the same feature.
        def _mlp() -> nn.Module:
            return _build_mlp(in_dim=feat_dim, hidden=d_model, out_dim=1, n_layers=2)
        self.head_dships_back = _mlp()
        self.head_dships_fwd = _mlp()
        self.head_survives_fwd = _mlp()
        self.head_slope_back = _mlp()                         # read learner slot only

    def forward(
        self,
        entity_tokens: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_owner_oh: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ctx, glob = self.cross(entity_tokens, entity_mask)   # glob (B, d)
        ctx_now = ctx[:, -1] if ctx.dim() == 4 else ctx
        mask_now = entity_mask[:, -1] if entity_mask.dim() == 3 else entity_mask
        owner_now = (
            planet_owner_oh[:, -1] if planet_owner_oh.dim() == 4 else planet_owner_oh
        ).to(ctx_now.dtype)
        player_state = self.consolidator(ctx_now, mask_now, owner_now)   # (B, 4, d)
        glob_b = glob.unsqueeze(1).expand(-1, player_state.size(1), -1)  # (B, 4, d)
        feat = torch.cat([glob_b, player_state], dim=-1)                 # (B, 4, 2d)
        return {
            "pair_logits": self.pair_compare(player_state, glob),     # (B, 4, 4)
            "is_ahead_logits": self.is_ahead_head(feat),              # (B, 4, H)
            "dships_back": self.head_dships_back(feat).squeeze(-1),    # (B, 4)
            "dships_fwd": self.head_dships_fwd(feat).squeeze(-1),      # (B, 4)
            "survives_fwd": self.head_survives_fwd(feat).squeeze(-1),  # (B, 4)
            "slope_back": self.head_slope_back(feat[:, 0]).squeeze(-1),  # (B,)
            "player_state": player_state,
            "glob": glob,
        }


def _listmle_loss(
    scores: torch.Tensor,        # (B, S) per-player value logits
    rank: torch.Tensor,          # (B, S) long, 0 = best
    valid: torch.Tensor,         # (B, S) {0,1} float — real player slots
) -> torch.Tensor:
    """Plackett-Luce (ListMLE) negative log-likelihood of the observed order.

    For each row, sort slots best->worst by ``rank`` (invalid slots sort to
    the tail with score -inf), then sum log P(pick o_k | remaining tail) =
    ``s[o_k] - logsumexp(s[o_k..o_{K-1}])``. Rows with < 2 valid players
    contribute nothing (no order to learn).
    """
    neg_inf = torch.finfo(scores.dtype).min
    is_valid = valid > 0.5
    s = scores.masked_fill(~is_valid, neg_inf)
    # Order best->worst; invalid slots (rank set to +inf) sort last.
    sort_key = rank.float().masked_fill(~is_valid, float("inf"))
    order = torch.argsort(sort_key, dim=-1)
    s_sorted = torch.gather(s, 1, order)                       # (B, S), invalid = -inf tail
    # Suffix logsumexp: cumlse[k] = logsumexp(s_sorted[k:]) via flip+cumsum+flip.
    rev_cumlse = torch.logcumsumexp(torch.flip(s_sorted, dims=[-1]), dim=-1)
    cumlse = torch.flip(rev_cumlse, dims=[-1])
    valid_pos = s_sorted > (neg_inf / 2)
    per_pos = (s_sorted - cumlse).masked_fill(~valid_pos, 0.0)  # last valid pos -> ~0
    row_loss = -per_pos.sum(dim=-1)                            # (B,)
    has_order = is_valid.sum(dim=-1) >= 2
    if has_order.any():
        return row_loss[has_order].mean()
    return scores.sum() * 0.0


# Pairwise/is_ahead value loss weights.
CRITIC_LAMBDA_IS_AHEAD: float = 0.5
# Prior/posterior consistency aux (only when --aux-posterior). λ_post weights the
# posterior view's own outcome loss; λ_cons pulls the deployable prior value
# toward the stop-grad future-informed posterior value (on logits).
CRITIC_LAMBDA_POST: float = 0.5
CRITIC_LAMBDA_CONS: float = 0.2
# Temporal momentum anchors (anti-shortcut): backward on the prior view,
# forward on the posterior view. These force each view to read its own window.
CRITIC_LAMBDA_DELTA: float = 0.3


def _critic_backward_delta_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    valid: torch.Tensor,          # (B,4) player_valid
) -> tuple[torch.Tensor, dict[str, float]]:
    """Backward momentum: ``dships_back`` (per-player) + ``slope_back`` (learner).

    Only the PRIOR window contains ``t-K`` — so a center-frame-only model fails
    these. No loss if the cache lacks the columns.
    """
    if "dships_back" not in batch:
        return preds["pair_logits"].sum() * 0.0, {}
    total = torch.zeros((), device=valid.device)
    terms: dict[str, float] = {}
    hub = F.smooth_l1_loss(preds["dships_back"], batch["dships_back"], reduction="none")
    L_d = (hub * valid).sum() / valid.sum().clamp(min=1.0)
    total = total + L_d
    terms["dships_back"] = float(L_d.detach())
    L_s = F.smooth_l1_loss(preds["slope_back"], batch["score_adv_slope_back"])
    total = total + L_s
    terms["slope_back"] = float(L_s.detach())
    return total, terms


def _critic_forward_delta_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    valid: torch.Tensor,          # (B,4) player_valid
) -> tuple[torch.Tensor, dict[str, float]]:
    """Forward momentum: ``dships_fwd`` (Huber) + ``survives_fwd`` (BCE), masked
    by ``valid_fwd`` (t+K within episode). Only the POSTERIOR window contains
    ``t+K``."""
    vf = batch.get("valid_fwd")
    if "dships_fwd" not in batch or vf is None:
        return preds["pair_logits"].sum() * 0.0, {}
    m = valid * vf.float().unsqueeze(-1)                       # (B,4)
    den = m.sum().clamp(min=1.0)
    hub = F.smooth_l1_loss(preds["dships_fwd"], batch["dships_fwd"], reduction="none")
    L_d = (hub * m).sum() / den
    bce = F.binary_cross_entropy_with_logits(
        preds["survives_fwd"], batch["survives_fwd"], reduction="none",
    )
    L_s = (bce * m).sum() / den
    return L_d + L_s, {"dships_fwd": float(L_d.detach()), "survives_fwd": float(L_s.detach())}


def _critic_is_ahead_targets(
    batch: dict[str, torch.Tensor],
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-player future-leader labels from existing cache columns.

    The cache's historical ``is_ahead_t_plus_K`` column is learner-only.
    For the redesigned critic we derive the per-player version from
    ``leader_seat_t_plus_K``: the leading player at horizon K gets target 1,
    the other real player slots get target 0, and rows where t+K is outside
    the episode are masked by ``valid_global_t_plus_K``.
    """
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for k in CROSS_ENTITY_VALUE_HORIZONS:
        leader = batch[f"leader_seat_t_plus_{k}"].long()
        target = _per_player_targets_from_class(leader)              # (B,4)
        global_valid = batch.get(f"valid_global_t_plus_{k}")
        if global_valid is None:
            mask = valid
        else:
            mask = valid * global_valid.float().unsqueeze(-1)
        targets.append(target)
        masks.append(mask)
    return torch.stack(targets, dim=-1), torch.stack(masks, dim=-1)   # (B,4,H)


def _critic_pair_targets_from_ahead(
    ahead: torch.Tensor,          # (B,4) binary is_ahead/leader target
    valid: torch.Tensor,          # (B,4) player_valid
    row_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ordered-pair labels from per-player ahead flags.

    ``target[i,j] = ahead_i``. A pair is supervised only when both slots are
    real, ``i != j``, and exactly one side is ahead; non-leader-vs-non-leader
    pairs are intentionally masked because the cache label does not order them.
    """
    B, S = ahead.shape
    target = ahead.unsqueeze(2).expand(B, S, S).float()
    pair_valid = valid.unsqueeze(2) * valid.unsqueeze(1)
    diff = (ahead.unsqueeze(2) != ahead.unsqueeze(1)).float()
    eye = torch.eye(S, dtype=pair_valid.dtype, device=pair_valid.device).unsqueeze(0)
    mask = pair_valid * diff * (1.0 - eye)
    if row_valid is not None:
        mask = mask * row_valid.float().view(B, 1, 1)
    return target, mask


def _critic_pair_loss(
    pair_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Terminal-horizon pair BCE for one critic view."""
    terminal_h = max(CROSS_ENTITY_VALUE_HORIZONS)
    leader = batch[f"leader_seat_t_plus_{terminal_h}"].long()
    ahead = _per_player_targets_from_class(leader)
    row_valid = batch.get(f"valid_global_t_plus_{terminal_h}")
    target, mask = _critic_pair_targets_from_ahead(ahead, valid, row_valid=row_valid)
    bce = F.binary_cross_entropy_with_logits(pair_logits, target, reduction="none")
    den = mask.sum().clamp(min=1.0)
    loss = (bce * mask).sum() / den
    pred = (pair_logits >= 0.0).float()
    acc = (((pred == target).float() * mask).sum() / den).detach()
    terms = {
        "pair_bce": float(loss.detach()),
        "pair_acc": float(acc),
        "pair_count": float(mask.sum().detach()),
        "pair_terminal_h": float(terminal_h),
    }
    return loss, terms, mask


def _critic_is_ahead_loss(
    is_ahead_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    target, mask = _critic_is_ahead_targets(batch, valid)
    bce = F.binary_cross_entropy_with_logits(is_ahead_logits, target, reduction="none")
    loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)
    pred = (is_ahead_logits >= 0.0).float()
    den = mask.sum().clamp(min=1.0)
    acc = (((pred == target).float() * mask).sum() / den).detach()
    terms: dict[str, float] = {
        "is_ahead_bce": float(loss.detach()),
        "is_ahead_acc": float(acc),
    }
    for idx, k in enumerate(CROSS_ENTITY_VALUE_HORIZONS):
        mk = mask[..., idx]
        bk = bce[..., idx]
        dk = mk.sum().clamp(min=1.0)
        terms[f"is_ahead_t_plus_{k}"] = float(((bk * mk).sum() / dk).detach())
    return loss, terms


def _critic_outcome_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    L_pair, pair_terms, pair_mask = _critic_pair_loss(
        preds["pair_logits"], batch, valid,
    )
    L_ahead, ahead_terms = _critic_is_ahead_loss(
        preds["is_ahead_logits"], batch, valid,
    )
    total = L_pair + CRITIC_LAMBDA_IS_AHEAD * L_ahead
    terms = dict(pair_terms)
    terms.update(ahead_terms)
    return total, terms, pair_mask


def compute_loss_critic(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    post_preds: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pairwise comparison + per-player is_ahead on the prior view.

    When ``post_preds`` is given (the future-informed posterior view), also
    fit it to the same labels and pull the prior pair logits toward the
    stop-grad posterior pair logits: the deployable critic learns a
    future-informed-but-not-future-leaking value.
    """
    valid = batch["player_valid"].float()                      # (B, 4)
    if float(valid.sum()) == 0.0:
        raise ValueError(
            "player_valid is all-zero — the cache predates the final_rank/"
            "player_valid columns. Rebuild the cross_entity cache with the "
            "updated featurizer before training head_set='critic'."
        )

    total, per_head, pair_mask = _critic_outcome_loss(preds, batch, valid)

    # Backward momentum anchor on the PRIOR (deployable) view — forces it to
    # read its history (anti-shortcut). Active even without the posterior aux.
    L_back, back_terms = _critic_backward_delta_loss(preds, batch, valid)
    total = total + CRITIC_LAMBDA_DELTA * L_back
    per_head.update(back_terms)

    if post_preds is not None:
        L_post, post_terms, _ = _critic_outcome_loss(post_preds, batch, valid)
        # Consistency on pair logits; posterior is the stop-grad teacher.
        # Use the supervised terminal-pair mask so consistency follows the
        # same label support as the deployed value head.
        cons = (
            ((preds["pair_logits"] - post_preds["pair_logits"].detach()) ** 2)
            * pair_mask
        ).sum() / pair_mask.sum().clamp(min=1.0)
        # Forward momentum anchor on the POSTERIOR view — forces it to read the
        # future (the other half of the asymmetric anti-shortcut).
        L_fwd, fwd_terms = _critic_forward_delta_loss(post_preds, batch, valid)
        total = (
            total
            + CRITIC_LAMBDA_POST * L_post
            + CRITIC_LAMBDA_CONS * cons
            + CRITIC_LAMBDA_DELTA * L_fwd
        )
        per_head.update(fwd_terms)
        per_head.update({f"post_{k}": v for k, v in post_terms.items()})
        per_head["consistency"] = float(cons.detach())
    return total, per_head


class CrossEntityTrainStack(nn.Module):
    """Bundle the full cross-entity stack for gradual unfreezing.

    The frozen pretrain keeps the upstream encoders external and trains
    only ``CrossEntityPretrainModel``. The gradual-unfreeze path needs a
    single module tree so we can toggle ``requires_grad`` and build
    discriminative LR param groups across the whole stack.
    """

    def __init__(
        self,
        *,
        fleet_encoder: FleetEncoder,
        planet_encoder: PlanetEncoder,
        entity_encoder: PlanetEntityEncoder,
        cross_model: CrossEntityPretrainModel,
    ):
        super().__init__()
        self.fleet_encoder = fleet_encoder
        self.planet_encoder = planet_encoder
        self.entity_encoder = entity_encoder
        self.cross_model = cross_model

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch,
            self.fleet_encoder,
            self.planet_encoder,
            self.entity_encoder,
        )
        return self.cross_model(entity_tokens, entity_mask)


# ---------- Loss ----------
def compute_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    planet_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: dict[str, float] = {}
    total: torch.Tensor | None = None
    pm = planet_mask.float()
    n_real = pm.sum().clamp(min=1.0)

    def add(name: str, term: torch.Tensor) -> None:
        nonlocal total
        total = term if total is None else total + term
        losses[name] = float(term.detach())

    # Tier-1: per-planet
    ce = F.cross_entropy(
        preds["frontier_class"].transpose(1, 2),
        targets["frontier_class"],
        reduction="none",
    )
    add("frontier_class", (ce * pm).sum() / n_real)

    for name in SPATIAL_REGRESSION_HEADS:
        mse = (preds[name] - targets[name]).pow(2)
        add(name, (mse * pm).sum() / n_real)

    # Category 1: existing entity labels at the cross layer.
    ce = F.cross_entropy(
        preds["earliest_arrival_owner_slot"].transpose(1, 2),
        targets["earliest_arrival_owner_slot"],
        reduction="none",
    )
    add("earliest_arrival_owner_slot", (ce * pm).sum() / n_real)

    for k in ENTITY_LABEL_HORIZONS:
        valid = targets[f"valid_t_plus_{k}"] * pm
        n_valid = valid.sum().clamp(min=1.0)
        ce_k = F.cross_entropy(
            preds[f"owner_t_plus_{k}"].transpose(1, 2),
            targets[f"owner_t_plus_{k}"],
            reduction="none",
        )
        add(f"owner_t_plus_{k}", (ce_k * valid).sum() / n_valid)
        mse = (preds[f"log_ships_t_plus_{k}"] - targets[f"log_ships_t_plus_{k}"]).pow(2)
        add(f"log_ships_t_plus_{k}", (mse * valid).sum() / n_valid)

    for h in ENTITY_ARRIVAL_HORIZONS:
        mse = (preds[f"ships_arriving_within_{h}"] - targets[f"ships_arriving_within_{h}"]).pow(2)
        per_planet = mse.sum(-1)
        add(
            f"ships_arriving_within_{h}",
            (per_planet * pm).sum() / (n_real * ENTITY_NUM_OWNER_SLOTS),
        )

    # Category 2: imitation / expert action.
    for pred_name, target_name in (
        ("expert_source_logit", "is_source_this_turn"),
        ("expert_target_logit", "is_target_this_turn"),
    ):
        bce = F.binary_cross_entropy_with_logits(
            preds[pred_name],
            targets[target_name],
            reduction="none",
        )
        add(pred_name, (bce * pm).sum() / n_real)
    add(
        "expert_acted_this_turn",
        F.binary_cross_entropy_with_logits(
            preds["expert_acted_this_turn"],
            targets["expert_acted_this_turn"],
        ),
    )

    # Tier-3: CLS / global (one scalar per batch element, no mask)
    add("winner_seat", F.cross_entropy(preds["winner_seat"], targets["winner_seat"]))
    add(
        "score_advantage_at_end_log",
        F.mse_loss(preds["score_advantage_at_end_log"], targets["score_advantage_at_end_log"]),
    )
    add(
        "turns_until_episode_end",
        F.mse_loss(preds["turns_until_episode_end"], targets["turns_until_episode_end"]),
    )

    # Category 3: short-horizon global value labels.
    for h in CROSS_ENTITY_VALUE_HORIZONS:
        valid = targets[f"valid_global_t_plus_{h}"]
        n_valid = valid.sum().clamp(min=1.0)
        ce_k = F.cross_entropy(
            preds[f"leader_seat_t_plus_{h}"],
            targets[f"leader_seat_t_plus_{h}"],
            reduction="none",
        )
        add(f"leader_seat_t_plus_{h}", (ce_k * valid).sum() / n_valid)
        mse = (
            preds[f"score_advantage_t_plus_{h}_log"]
            - targets[f"score_advantage_t_plus_{h}_log"]
        ).pow(2)
        add(f"score_advantage_t_plus_{h}_log", (mse * valid).sum() / n_valid)
        bce = F.binary_cross_entropy_with_logits(
            preds[f"is_ahead_t_plus_{h}"],
            targets[f"is_ahead_t_plus_{h}"],
            reduction="none",
        )
        add(f"is_ahead_t_plus_{h}", (bce * valid).sum() / n_valid)

    # Category 4: tactical neighborhood labels.
    for h in CROSS_ENTITY_TACTICAL_HORIZONS:
        for name in (
            f"can_friendly_reinforce_within_{h}",
            f"enemy_can_capture_within_{h}",
        ):
            bce = F.binary_cross_entropy_with_logits(
                preds[name],
                targets[name],
                reduction="none",
            )
            add(name, (bce * pm).sum() / n_real)
        mse = (
            preds[f"best_local_support_margin_within_{h}_log"]
            - targets[f"best_local_support_margin_within_{h}_log"]
        ).pow(2)
        add(
            f"best_local_support_margin_within_{h}_log",
            (mse * pm).sum() / n_real,
        )

    assert total is not None
    return total, losses


# ---------- Encoder pass (flatten T into batch, run, restack) ----------
def _entity_tokens_per_step(
    batch: dict[str, torch.Tensor],
    fleet_enc: FleetEncoder,
    planet_enc: PlanetEncoder,
    entity_enc: PlanetEntityEncoder,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frozen encoders for every time step in the batch.

    Snapshot tensors come in shape ``(B, T, P, ...)`` /
    ``(B, T, F, ...)``. We flatten ``(B, T)`` into one batch dim, run
    each encoder once on the flattened batch, then reshape entity
    tokens back to ``(B, T, P, d_model)`` for the cross-entity layer.
    Single-step ``(B, P, ...)`` inputs work without changes — the
    flatten is a no-op when ``T`` is absent.
    """
    pf = batch["planet_features"]
    if pf.dim() == 4:                                     # (B, T, P, D_p)
        B, T, P, D_p = pf.shape
        flat_B = B * T
        # Flatten the T axis into the batch dim for each input tensor.
        flat_planet_features = pf.reshape(flat_B, P, D_p)
        flat_fleet_features = batch["fleet_features"].reshape(
            flat_B, batch["fleet_features"].shape[2], FLEET_RAW_DIM,
        )
        flat_planet_mask = batch["planet_mask"].reshape(flat_B, P).bool()
        flat_fleet_mask = batch["fleet_mask"].reshape(
            flat_B, batch["fleet_mask"].shape[2],
        ).bool()
        flat_target = batch["fleet_target_idx"].reshape(flat_B, -1)
        flat_source = batch["fleet_source_idx"].reshape(flat_B, -1)
        flat_owner = batch["fleet_owner_slot"].reshape(flat_B, -1)
        flat_ships = batch["fleet_ships_log"].reshape(flat_B, -1)
        flat_eta = batch["fleet_eta_norm"].reshape(flat_B, -1)
    else:                                                 # (B, P, D_p)
        B = pf.shape[0]
        T = 1
        flat_planet_features = pf
        flat_fleet_features = batch["fleet_features"]
        flat_planet_mask = batch["planet_mask"].bool()
        flat_fleet_mask = batch["fleet_mask"].bool()
        flat_target = batch["fleet_target_idx"]
        flat_source = batch["fleet_source_idx"]
        flat_owner = batch["fleet_owner_slot"]
        flat_ships = batch["fleet_ships_log"]
        flat_eta = batch["fleet_eta_norm"]

    planet_trainable = any(p.requires_grad for p in planet_enc.parameters())
    fleet_trainable = any(p.requires_grad for p in fleet_enc.parameters())
    entity_trainable = any(p.requires_grad for p in entity_enc.parameters())

    planet_ctx = nullcontext() if planet_trainable else torch.no_grad()
    fleet_ctx = nullcontext() if fleet_trainable else torch.no_grad()
    # Even when the entity encoder itself is frozen, it must stay in the
    # autograd graph if a lower encoder is trainable so gradients can
    # flow through it back to that lower layer.
    entity_ctx = (
        nullcontext()
        if (entity_trainable or planet_trainable or fleet_trainable)
        else torch.no_grad()
    )

    with planet_ctx:
        planet_tok = planet_enc(flat_planet_features)
    with fleet_ctx:
        fleet_tok = fleet_enc(flat_fleet_features)
    with entity_ctx:
        entity_tokens = entity_enc(
            planet_tok, fleet_tok,
            flat_target, flat_source, flat_owner, flat_ships, flat_eta,
            flat_fleet_mask,
            planet_mask=flat_planet_mask,
        )                                                 # (flat_B, P, d)
    if T > 1:
        d = entity_tokens.shape[-1]
        entity_tokens = entity_tokens.reshape(B, T, -1, d)
        # Mask passed to cross-attention is (B, T, P).
        entity_mask = batch["planet_mask"].bool()
    else:
        entity_mask = batch["planet_mask"].bool()
    return entity_tokens, entity_mask


# ---------- Eval ----------
@torch.no_grad()
def evaluate(
    model: CrossEntityPretrainModel,
    fleet_enc: FleetEncoder,
    planet_enc: PlanetEncoder,
    entity_enc: PlanetEntityEncoder,
    loader: DataLoader,
    device: str,
) -> dict[str, dict[str, float]]:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    correct: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch, fleet_enc, planet_enc, entity_enc,
        )
        preds = model(entity_tokens, entity_mask)
        # Per-planet supervision is on the CURRENT step only — that's
        # the last index along T. ``planet_mask`` here is (B, T, P)
        # for multi-step; pull (B, P) for the loss / metrics.
        if batch["planet_mask"].dim() == 3:
            cur_planet_mask = batch["planet_mask"][:, -1]
        else:
            cur_planet_mask = batch["planet_mask"]
        # Rebind in batch so the existing loss/eval code reads it.
        batch["planet_mask"] = cur_planet_mask
        pm = batch["planet_mask"].float()
        n_real = int(pm.sum())
        valid_planets = pm.bool()

        # Per-planet categorical heads.
        for name in ("frontier_class", "earliest_arrival_owner_slot"):
            logits = preds[name]
            argmax = logits.argmax(-1)
            correct[name] += int(((argmax == batch[name]) & valid_planets).sum())
            counts[name] += n_real
            ce = F.cross_entropy(
                logits.transpose(1, 2),
                batch[name],
                reduction="none",
            )
            sums[name] += float((ce * pm).sum())

        # Per-planet regression heads.
        for name in SPATIAL_REGRESSION_HEADS:
            mse = (preds[name] - batch[name]).pow(2)
            sums[name] += float((mse * pm).sum())
            counts[name] += n_real

        # Category 1: per-planet future labels with valid masks.
        for k in ENTITY_LABEL_HORIZONS:
            valid = batch[f"valid_t_plus_{k}"] * pm
            n_valid = int(valid.sum())
            logits = preds[f"owner_t_plus_{k}"]
            argmax = logits.argmax(-1)
            valid_mask = valid > 0.5
            correct[f"owner_t_plus_{k}"] += int(((argmax == batch[f"owner_t_plus_{k}"]) & valid_mask).sum())
            counts[f"owner_t_plus_{k}"] += n_valid
            ce = F.cross_entropy(
                logits.transpose(1, 2),
                batch[f"owner_t_plus_{k}"],
                reduction="none",
            )
            sums[f"owner_t_plus_{k}"] += float((ce * valid).sum())

            mse = (preds[f"log_ships_t_plus_{k}"] - batch[f"log_ships_t_plus_{k}"]).pow(2)
            sums[f"log_ships_t_plus_{k}"] += float((mse * valid).sum())
            counts[f"log_ships_t_plus_{k}"] += n_valid

        for h in ENTITY_ARRIVAL_HORIZONS:
            mse = (preds[f"ships_arriving_within_{h}"] - batch[f"ships_arriving_within_{h}"]).pow(2)
            sums[f"ships_arriving_within_{h}"] += float((mse * pm.unsqueeze(-1)).sum())
            counts[f"ships_arriving_within_{h}"] += n_real * ENTITY_NUM_OWNER_SLOTS

        # Category 2 + 4 per-planet binary heads.
        for name, target_name in (
            ("expert_source_logit", "is_source_this_turn"),
            ("expert_target_logit", "is_target_this_turn"),
            *[
                (f"can_friendly_reinforce_within_{h}", f"can_friendly_reinforce_within_{h}")
                for h in CROSS_ENTITY_TACTICAL_HORIZONS
            ],
            *[
                (f"enemy_can_capture_within_{h}", f"enemy_can_capture_within_{h}")
                for h in CROSS_ENTITY_TACTICAL_HORIZONS
            ],
        ):
            logits = preds[name]
            target = batch[target_name]
            pred_bin = logits >= 0.0
            tgt_bin = target > 0.5
            correct[name] += int(((pred_bin == tgt_bin) & valid_planets).sum())
            counts[name] += n_real
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            sums[name] += float((bce * pm).sum())

        for h in CROSS_ENTITY_TACTICAL_HORIZONS:
            name = f"best_local_support_margin_within_{h}_log"
            mse = (preds[name] - batch[name]).pow(2)
            sums[name] += float((mse * pm).sum())
            counts[name] += n_real

        # CLS categorical / binary / regression heads.
        logits = preds["winner_seat"]
        winner_argmax = logits.argmax(-1)
        correct["winner_seat"] += int((winner_argmax == batch["winner_seat"]).sum())
        counts["winner_seat"] += int(batch["winner_seat"].shape[0])
        sums["winner_seat"] += float(
            F.cross_entropy(logits, batch["winner_seat"], reduction="sum")
        )

        for name in CURRENT_GLOBAL_REGRESSION_HEADS:
            mse = (preds[name] - batch[name]).pow(2)
            sums[name] += float(mse.sum())
            counts[name] += int(batch[name].shape[0])

        acted_logits = preds["expert_acted_this_turn"]
        acted_target = batch["expert_acted_this_turn"]
        correct["expert_acted_this_turn"] += int(((acted_logits >= 0.0) == (acted_target > 0.5)).sum())
        counts["expert_acted_this_turn"] += int(acted_target.shape[0])
        sums["expert_acted_this_turn"] += float(
            F.binary_cross_entropy_with_logits(
                acted_logits,
                acted_target,
                reduction="sum",
            )
        )

        for h in CROSS_ENTITY_VALUE_HORIZONS:
            valid = batch[f"valid_global_t_plus_{h}"]
            n_valid = int(valid.sum())

            logits = preds[f"leader_seat_t_plus_{h}"]
            argmax = logits.argmax(-1)
            valid_mask = valid > 0.5
            correct[f"leader_seat_t_plus_{h}"] += int(((argmax == batch[f"leader_seat_t_plus_{h}"]) & valid_mask).sum())
            counts[f"leader_seat_t_plus_{h}"] += n_valid
            ce = F.cross_entropy(
                logits,
                batch[f"leader_seat_t_plus_{h}"],
                reduction="none",
            )
            sums[f"leader_seat_t_plus_{h}"] += float((ce * valid).sum())

            score_name = f"score_advantage_t_plus_{h}_log"
            mse = (preds[score_name] - batch[score_name]).pow(2)
            sums[score_name] += float((mse * valid).sum())
            counts[score_name] += n_valid

            ahead_name = f"is_ahead_t_plus_{h}"
            logits = preds[ahead_name]
            target = batch[ahead_name]
            pred_bin = logits >= 0.0
            tgt_bin = target > 0.5
            correct[ahead_name] += int(((pred_bin == tgt_bin) & valid_mask).sum())
            counts[ahead_name] += n_valid
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            sums[ahead_name] += float((bce * valid).sum())

    summary: dict[str, dict[str, float]] = {}
    for name, total in sums.items():
        n = max(1, counts[name])
        entry: dict[str, float] = {"loss": total / n}
        if name in correct:
            entry["acc"] = correct[name] / n
        summary[name] = entry
    model.train()
    return summary


@torch.no_grad()
def evaluate_v2(
    model: CrossEntityPretrainModelV2,
    fleet_enc: FleetEncoder,
    planet_enc: PlanetEncoder,
    entity_enc: PlanetEntityEncoder,
    loader: DataLoader,
    device: str,
) -> dict[str, dict[str, float]]:
    """Evaluation pass scoped to the V2 advantage/future head set.

    Returns ``{head_name: {"loss": ..., "acc"?: ...}}``. Per-player heads
    report mean BCE across the 4 slots and slot-level accuracy.
    """
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    correct: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch, fleet_enc, planet_enc, entity_enc,
        )
        owner_oh = _owner_oh_from_batch(batch)
        preds = model(entity_tokens, entity_mask, owner_oh)
        B = batch["winner_seat"].shape[0]

        # --- per-player BCE: winner_p (always valid) ---
        winner_t = _per_player_targets_from_class(batch["winner_seat"])
        bce = F.binary_cross_entropy_with_logits(
            preds["winner_p"], winner_t, reduction="none",
        )                                                  # (B, 4)
        sums["winner_p"] += float(bce.sum())
        counts["winner_p"] += B * ENTITY_NUM_OWNER_SLOTS
        pred_bin = (preds["winner_p"] >= 0.0).float()
        correct["winner_p"] += int((pred_bin == winner_t).sum())

        # --- ordered pair BCE: final winner beats other players ---
        winner_pair_t, winner_pair_m = _pair_targets_from_winner_class(
            batch["winner_seat"],
        )
        bce = F.binary_cross_entropy_with_logits(
            preds["winner_pair"], winner_pair_t, reduction="none",
        )
        sums["winner_pair"] += float((bce * winner_pair_m).sum())
        counts["winner_pair"] += int(winner_pair_m.sum())
        pred_bin = (preds["winner_pair"] >= 0.0).float()
        correct["winner_pair"] += int(
            ((pred_bin == winner_pair_t).float() * winner_pair_m).sum()
        )

        # --- ordered pair BCE: current player strength ordering ---
        current_pair_t, current_pair_m = _current_pair_targets_from_batch(batch)
        bce = F.binary_cross_entropy_with_logits(
            preds["current_pair_ahead"], current_pair_t, reduction="none",
        )
        sums["current_pair_ahead"] += float((bce * current_pair_m).sum())
        counts["current_pair_ahead"] += int(current_pair_m.sum())
        pred_bin = (preds["current_pair_ahead"] >= 0.0).float()
        correct["current_pair_ahead"] += int(
            ((pred_bin == current_pair_t).float() * current_pair_m).sum()
        )

        # --- per-player BCE: leader_k_p (masked by valid_global) ---
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            leader_t = _per_player_targets_from_class(batch[f"leader_seat_t_plus_{k}"])
            valid = batch[f"valid_global_t_plus_{k}"].float()        # (B,)
            v_exp = valid.unsqueeze(-1)                              # (B, 1)
            bce = F.binary_cross_entropy_with_logits(
                preds[f"leader_k{k}_p"], leader_t, reduction="none",
            )                                                        # (B, 4)
            sums[f"leader_k{k}_p"] += float((bce * v_exp).sum())
            counts[f"leader_k{k}_p"] += int(valid.sum()) * ENTITY_NUM_OWNER_SLOTS
            pred_bin = (preds[f"leader_k{k}_p"] >= 0.0).float()
            correct[f"leader_k{k}_p"] += int(((pred_bin == leader_t).float() * v_exp).sum())

            leader_pair_t, leader_pair_m = _pair_targets_from_winner_class(
                batch[f"leader_seat_t_plus_{k}"],
                valid=valid,
            )
            bce = F.binary_cross_entropy_with_logits(
                preds[f"leader_k{k}_pair"], leader_pair_t, reduction="none",
            )
            sums[f"leader_k{k}_pair"] += float((bce * leader_pair_m).sum())
            counts[f"leader_k{k}_pair"] += int(leader_pair_m.sum())
            pred_bin = (preds[f"leader_k{k}_pair"] >= 0.0).float()
            correct[f"leader_k{k}_pair"] += int(
                ((pred_bin == leader_pair_t).float() * leader_pair_m).sum()
            )

        # --- learner-relative BCE: is_ahead_k (masked) ---
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            valid = batch[f"valid_global_t_plus_{k}"].float()
            target = batch[f"is_ahead_t_plus_{k}"].float()
            bce = F.binary_cross_entropy_with_logits(
                preds[f"is_ahead_t_plus_{k}"], target, reduction="none",
            )
            sums[f"is_ahead_t_plus_{k}"] += float((bce * valid).sum())
            counts[f"is_ahead_t_plus_{k}"] += int(valid.sum())
            pred_bin = (preds[f"is_ahead_t_plus_{k}"] >= 0.0).float()
            tgt_bin = (target > 0.5).float()
            correct[f"is_ahead_t_plus_{k}"] += int(((pred_bin == tgt_bin).float() * valid).sum())

        # --- learner-relative MSE: score_adv_k (masked) ---
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            valid = batch[f"valid_global_t_plus_{k}"].float()
            mse = (preds[f"score_advantage_t_plus_{k}_log"]
                   - batch[f"score_advantage_t_plus_{k}_log"]).pow(2)
            sums[f"score_advantage_t_plus_{k}_log"] += float((mse * valid).sum())
            counts[f"score_advantage_t_plus_{k}_log"] += int(valid.sum())

        # --- terminal regression heads ---
        for name in ("score_advantage_at_end_log", "turns_until_episode_end"):
            mse = (preds[name] - batch[name]).pow(2)
            sums[name] += float(mse.sum())
            counts[name] += B

    summary: dict[str, dict[str, float]] = {}
    for name, total in sums.items():
        n = max(1, counts[name])
        entry: dict[str, float] = {"loss": total / n}
        if name in correct:
            entry["acc"] = correct[name] / n
        summary[name] = entry
    model.train()
    return summary


@torch.no_grad()
def evaluate_v3(
    model: CrossEntityPretrainModelV3,
    fleet_enc: FleetEncoder,
    planet_enc: PlanetEncoder,
    entity_enc: PlanetEntityEncoder,
    loader: DataLoader,
    device: str,
) -> dict[str, dict[str, float]]:
    """V3 eval scoped to the **last (current) frame** — the deployed readout.

    Slices the per-frame predictions to ``[:, -1]`` and scores them against
    the anchor labels (the current-frame, non-``_perT`` keys), reusing the
    same per-head metric definitions as :func:`evaluate_v2`. This makes V3's
    headline numbers directly comparable to V2.
    """
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    correct: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch, fleet_enc, planet_enc, entity_enc,
        )
        owner_oh = _owner_oh_from_batch(batch)
        full = model(entity_tokens, entity_mask, owner_oh)
        # Slice to last frame → (B, ...). 4x4 pair heads → (B,4,4).
        preds = {
            k: v[:, -1]
            for k, v in full.items()
            if k not in _V3_DIAGNOSTIC_KEYS
        }
        B = batch["winner_seat"].shape[0]

        winner_t = _per_player_targets_from_class(batch["winner_seat"])
        bce = F.binary_cross_entropy_with_logits(preds["winner_p"], winner_t, reduction="none")
        sums["winner_p"] += float(bce.sum()); counts["winner_p"] += B * ENTITY_NUM_OWNER_SLOTS
        correct["winner_p"] += int(((preds["winner_p"] >= 0.0).float() == winner_t).sum())

        wp_t, wp_m = _pair_targets_from_winner_class(batch["winner_seat"])
        sums["winner_pair"] += float(
            (F.binary_cross_entropy_with_logits(preds["winner_pair"], wp_t, reduction="none") * wp_m).sum()
        )
        counts["winner_pair"] += int(wp_m.sum())
        correct["winner_pair"] += int((((preds["winner_pair"] >= 0.0).float() == wp_t).float() * wp_m).sum())

        for k in CROSS_ENTITY_VALUE_HORIZONS:
            valid = batch[f"valid_global_t_plus_{k}"].float()
            # per-player leader
            lt = _per_player_targets_from_class(batch[f"leader_seat_t_plus_{k}"])
            ve = valid.unsqueeze(-1)
            bce = F.binary_cross_entropy_with_logits(preds[f"leader_k{k}_p"], lt, reduction="none")
            sums[f"leader_k{k}_p"] += float((bce * ve).sum())
            counts[f"leader_k{k}_p"] += int(valid.sum()) * ENTITY_NUM_OWNER_SLOTS
            correct[f"leader_k{k}_p"] += int((((preds[f"leader_k{k}_p"] >= 0.0).float() == lt).float() * ve).sum())
            # is_ahead
            ta = batch[f"is_ahead_t_plus_{k}"].float()
            bce = F.binary_cross_entropy_with_logits(preds[f"is_ahead_t_plus_{k}"], ta, reduction="none")
            sums[f"is_ahead_t_plus_{k}"] += float((bce * valid).sum())
            counts[f"is_ahead_t_plus_{k}"] += int(valid.sum())
            correct[f"is_ahead_t_plus_{k}"] += int((((preds[f"is_ahead_t_plus_{k}"] >= 0.0).float() == (ta > 0.5).float()).float() * valid).sum())
            # score_adv
            mse = (preds[f"score_advantage_t_plus_{k}_log"] - batch[f"score_advantage_t_plus_{k}_log"]).pow(2)
            sums[f"score_advantage_t_plus_{k}_log"] += float((mse * valid).sum())
            counts[f"score_advantage_t_plus_{k}_log"] += int(valid.sum())

        for name in ("score_advantage_at_end_log", "turns_until_episode_end"):
            mse = (preds[name] - batch[name]).pow(2)
            sums[name] += float(mse.sum()); counts[name] += B

    summary: dict[str, dict[str, float]] = {}
    for name, total in sums.items():
        n = max(1, counts[name])
        entry: dict[str, float] = {"loss": total / n}
        if name in correct:
            entry["acc"] = correct[name] / n
        summary[name] = entry
    model.train()
    return summary


@torch.no_grad()
def evaluate_critic(
    model: "CrossEntityCriticModel",
    fleet_enc: FleetEncoder,
    planet_enc: PlanetEncoder,
    entity_enc: PlanetEntityEncoder,
    loader: DataLoader,
    device: str,
) -> dict[str, dict[str, float]]:
    """Critic eval for the redesigned pairwise/is_ahead objective."""
    model.eval()
    agg = defaultdict(float)
    cnt = defaultdict(float)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        et, em = _entity_tokens_per_step(batch, fleet_enc, planet_enc, entity_enc)
        oo = _owner_oh_from_batch(batch)
        preds = model(et, em, oo)
        valid = batch["player_valid"].float()

        L_pair, _pair_terms, pair_mask = _critic_pair_loss(
            preds["pair_logits"], batch, valid,
        )
        target_h = max(CROSS_ENTITY_VALUE_HORIZONS)
        leader = batch[f"leader_seat_t_plus_{target_h}"].long()
        ahead = _per_player_targets_from_class(leader)
        row_valid = batch.get(f"valid_global_t_plus_{target_h}")
        pair_t, _ = _critic_pair_targets_from_ahead(ahead, valid, row_valid=row_valid)
        pair_bce = F.binary_cross_entropy_with_logits(
            preds["pair_logits"], pair_t, reduction="none",
        )
        pair_pred = (preds["pair_logits"] >= 0.0).float()
        is_4p = valid.sum(-1) >= 3.5
        m4 = pair_mask * is_4p.float().view(-1, 1, 1)
        m2 = pair_mask * (~is_4p).float().view(-1, 1, 1)

        agg["pair_bce"] += float((pair_bce * pair_mask).sum())
        cnt["pair_bce"] += float(pair_mask.sum())
        agg["pair_acc"] += float(((pair_pred == pair_t).float() * pair_mask).sum())
        cnt["pair_acc"] += float(pair_mask.sum())
        agg["pair_acc_2p"] += float(((pair_pred == pair_t).float() * m2).sum())
        cnt["pair_acc_2p"] += float(m2.sum())
        agg["pair_acc_4p"] += float(((pair_pred == pair_t).float() * m4).sum())
        cnt["pair_acc_4p"] += float(m4.sum())

        L_ahead, _ahead_terms = _critic_is_ahead_loss(
            preds["is_ahead_logits"], batch, valid,
        )
        ahead_t, ahead_m = _critic_is_ahead_targets(batch, valid)
        ahead_bce = F.binary_cross_entropy_with_logits(
            preds["is_ahead_logits"], ahead_t, reduction="none",
        )
        ahead_pred = (preds["is_ahead_logits"] >= 0.0).float()
        agg["is_ahead_bce"] += float((ahead_bce * ahead_m).sum())
        cnt["is_ahead_bce"] += float(ahead_m.sum())
        agg["is_ahead_acc"] += float(((ahead_pred == ahead_t).float() * ahead_m).sum())
        cnt["is_ahead_acc"] += float(ahead_m.sum())
        for idx, k in enumerate(CROSS_ENTITY_VALUE_HORIZONS):
            mk = ahead_m[..., idx]
            agg[f"is_ahead_t_plus_{k}"] += float((ahead_bce[..., idx] * mk).sum())
            cnt[f"is_ahead_t_plus_{k}"] += float(mk.sum())

        # Keep these connected loss values visible when val batches are tiny.
        agg["_pair_loss_batches"] += float(L_pair) * preds["pair_logits"].shape[0]
        cnt["_pair_loss_batches"] += preds["pair_logits"].shape[0]
        agg["_ahead_loss_batches"] += float(L_ahead) * preds["pair_logits"].shape[0]
        cnt["_ahead_loss_batches"] += preds["pair_logits"].shape[0]

    def rate(k):
        return agg[k] / cnt[k] if cnt[k] > 0 else 0.0
    summary = {
        "pair_bce": {
            "loss": rate("pair_bce"),
            "acc": rate("pair_acc"),
            "acc_2p": rate("pair_acc_2p"),
            "acc_4p": rate("pair_acc_4p"),
        },
        "is_ahead_bce": {
            "loss": rate("is_ahead_bce"),
            "acc": rate("is_ahead_acc"),
        },
    }
    for k in CROSS_ENTITY_VALUE_HORIZONS:
        summary[f"is_ahead_t_plus_{k}"] = {"loss": rate(f"is_ahead_t_plus_{k}")}
    model.train()
    return summary


# ---------- Helpers ----------
def _load_frozen_encoders(
    fleet_run_dir: Path,
    planet_run_dir: Path,
    entity_run_dir: Path,
    *,
    device: str,
) -> tuple[FleetEncoder, PlanetEncoder, PlanetEntityEncoder]:
    fc = torch.load(
        fleet_run_dir / "fleet_encoder_best.pt",
        map_location=device, weights_only=False,
    )
    pc = torch.load(
        planet_run_dir / "planet_encoder_best.pt",
        map_location=device, weights_only=False,
    )
    ec = torch.load(
        entity_run_dir / "entity_encoder_best.pt",
        map_location=device, weights_only=False,
    )

    fenc = FleetEncoder(d_model=fc["config"]["d_model"])
    fenc.load_state_dict({
        k.removeprefix("encoder."): v for k, v in fc["model"].items()
        if k.startswith("encoder.")
    })

    # Honor encoder-architecture flags from the ckpt config so the
    # constructor builds the same submodules the state-dict carries
    # (``use_traj_branch=False`` skips the ``traj``/``trunk`` branches
    # and resizes ``scalar`` to ``d_model`` instead of ``branch_dim``).
    pcfg = pc["config"]
    penc_kwargs: dict = {"d_model": pcfg["d_model"]}
    if "use_traj_branch" in pcfg:
        penc_kwargs["use_traj_branch"] = pcfg["use_traj_branch"]
    penc = PlanetEncoder(**penc_kwargs)
    penc.load_state_dict({
        k.removeprefix("encoder."): v for k, v in pc["model"].items()
        if k.startswith("encoder.")
    })

    eenc = PlanetEntityEncoder(d_model=ec["config"]["d_model"])
    eenc.load_state_dict({
        k.removeprefix("entity."): v for k, v in ec["model"].items()
        if k.startswith("entity.")
    })

    for m in (fenc, penc, eenc):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return fenc, penc, eenc


def _format_summary(summary: dict[str, dict[str, float]]) -> str:
    lines = []
    for name, m in summary.items():
        acc = f"  acc={m['acc']:.3f}" if "acc" in m else ""
        lines.append(f"    {name:<32s}  loss={m['loss']:.4f}{acc}")
    return "\n".join(lines)


def _make_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _current_planet_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pm = batch["planet_mask"]
    return pm[:, -1] if pm.dim() == 3 else pm


def _latest_run_dir(parent: Path, ckpt_name: str) -> Path:
    """Latest sorted-by-name dir under ``parent`` containing ``ckpt_name``."""
    candidates = sorted(
        p for p in parent.iterdir()
        if p.is_dir() and (p / ckpt_name).exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no run dir under {parent} contains {ckpt_name}"
        )
    return candidates[-1]


def _split_cached_cross_dataset(
    full: CachedCrossEntitySnapshotDataset,
    manifest: dict[str, list[str]] | None,
    *,
    seed: int,
) -> dict[str, torch.utils.data.Dataset]:
    """Split a cached cross-entity dataset into train/val/test subsets.

    Cache keys are replay UUIDs from the CSV ``episode_id`` column. The
    manifest stores CSV filename stems. Older cached training code compared
    those directly and could create empty splits. Prefer split metadata from
    the cache, then mapped manifest stems, then direct key matching for old
    layouts, and finally a deterministic episode split over the cache itself.
    """
    ep_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, (ep, _t) in enumerate(full.keys):
        ep_to_idxs[str(ep)].append(i)
    cache_eps = set(ep_to_idxs)

    def _manifest_stems(split: str) -> list[str]:
        if not manifest:
            return []
        return [
            n.removeprefix("cross_entity_").removesuffix(".csv")
            for n in manifest.get(split, [])
        ]

    split_eps: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    cfg_split_stems = full.config.get("split_stems") or {}
    stem_to_episode = {
        str(k): str(v)
        for k, v in (full.config.get("stem_to_episode") or {}).items()
    }

    for split in ("train", "val", "test"):
        stems = list(cfg_split_stems.get(split) or _manifest_stems(split))
        for stem in stems:
            ep = stem_to_episode.get(stem)
            if ep is not None and ep in cache_eps:
                split_eps[split].add(ep)
            # Legacy direct-match fallback for caches keyed by filename stem.
            if stem in cache_eps:
                split_eps[split].add(stem)

    if not all(split_eps[s] for s in ("train", "val", "test")):
        import warnings

        eps = sorted(cache_eps)
        g = torch.Generator().manual_seed(seed)
        order = [eps[i] for i in torch.randperm(len(eps), generator=g).tolist()]
        n = len(order)
        n_test = max(1, int(round(n * 0.10))) if n >= 3 else max(0, n - 1)
        n_val = max(1, int(round(n * 0.10))) if n >= 3 else 0
        if n_test + n_val >= n and n > 1:
            n_test = 1
            n_val = 0
        split_eps = {
            "test": set(order[:n_test]),
            "val": set(order[n_test:n_test + n_val]),
            "train": set(order[n_test + n_val:]),
        }
        if not split_eps["val"] and len(split_eps["train"]) > 1:
            moved = sorted(split_eps["train"])[0]
            split_eps["train"].remove(moved)
            split_eps["val"].add(moved)
        warnings.warn(
            "cache split metadata did not match manifest stems; falling back "
            "to a deterministic episode split over cache keys. Rebuild the "
            "cache with current scripts/build_cross_entity_cache.py to "
            "preserve manifest splits.",
            stacklevel=2,
        )

    splits: dict[str, torch.utils.data.Dataset] = {}
    for split in ("train", "val", "test"):
        idxs = [
            i
            for ep in sorted(split_eps[split])
            for i in ep_to_idxs.get(ep, [])
        ]
        if not idxs:
            raise ValueError(
                f"cached cross-entity {split} split is empty. "
                f"cache episodes={len(cache_eps)}; "
                f"manifest present={manifest is not None}; "
                "rebuild the cache with split metadata or provide a matching "
                "manifest."
            )
        splits[split] = torch.utils.data.Subset(full, idxs)
    return splits


def _resolve_attr(root: Any, path: str) -> Any:
    obj = root
    if not path:
        return obj
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_trainable_modules(
    stack: CrossEntityTrainStack,
    path: str,
) -> list[nn.Module]:
    """Resolve a trainable-path alias used by the gradual-unfreeze schedule."""
    if path == "heads":
        return [
            module
            for name, module in stack.cross_model.named_children()
            if name.startswith("head_")
        ]
    if path.startswith("heads."):
        return [getattr(stack.cross_model, path.split(".", 1)[1])]

    aliases = {
        "fleet_encoder": stack.fleet_encoder,
        "planet_encoder": stack.planet_encoder,
        "entity_encoder": stack.entity_encoder,
        "cross_model": stack.cross_model,
        "cross": stack.cross_model.cross,
    }
    if path in aliases:
        return [aliases[path]]

    root_name, dot, tail = path.partition(".")
    if root_name in aliases:
        root = aliases[root_name]
        return [_resolve_attr(root, tail)] if dot else [root]

    raise KeyError(f"unknown trainable path: {path}")


def _freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def set_trainable(
    stack: CrossEntityTrainStack,
    *,
    freeze: list[str],
    unfreeze: list[str],
) -> None:
    """Toggle ``requires_grad`` for the given path aliases."""
    for path in freeze:
        for module in _resolve_trainable_modules(stack, path):
            for p in module.parameters():
                p.requires_grad_(False)
    for path in unfreeze:
        for module in _resolve_trainable_modules(stack, path):
            for p in module.parameters():
                p.requires_grad_(True)


def build_param_groups(
    stack: CrossEntityTrainStack,
    lr_table: dict[str, float],
) -> list[dict[str, Any]]:
    """Build optimizer param groups from path aliases with discriminative LRs."""
    # Longest-first so specific leaves (e.g. ``fleet_encoder.fc2``)
    # override broader matches (e.g. ``fleet_encoder``).
    resolved: list[tuple[str, float, set[int]]] = []
    for path, lr in sorted(lr_table.items(), key=lambda kv: len(kv[0]), reverse=True):
        params: set[int] = set()
        for module in _resolve_trainable_modules(stack, path):
            for p in module.parameters():
                params.add(id(p))
        resolved.append((path, lr, params))

    grouped: dict[str, dict[str, Any]] = {}
    assigned: set[int] = set()
    for name, param in stack.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in assigned:
            continue
        match: tuple[str, float, set[int]] | None = None
        for item in resolved:
            if pid in item[2]:
                match = item
                break
        if match is None:
            raise ValueError(
                f"trainable param {name!r} is missing from the LR table"
            )
        path, lr, _ = match
        group = grouped.setdefault(path, {"params": [], "lr": lr, "name": path})
        group["params"].append(param)
        assigned.add(pid)

    return [group for group in grouped.values() if group["params"]]


def _build_gradual_unfreeze_schedule(stage_epochs: list[int]) -> list[dict[str, Any]]:
    """Stage schedule that starts from a pretrained frozen cross-entity ckpt.

    The frozen run already trained the cross-attention layer + heads, so
    this resume path starts at Stage 1 from ``docs/GRADUAL_UNFREEZE.md``.
    """
    templates = [
        {
            "index": 1,
            "name": "entity-unfreeze",
            "trainable_paths": ["cross", "heads", "entity_encoder"],
            "lr_table": {
                "cross": 1e-3,
                "heads": 1e-3,
                "entity_encoder": 1e-4,
            },
        },
        {
            "index": 2,
            "name": "top-half-unfreeze",
            "trainable_paths": [
                "cross",
                "heads",
                "entity_encoder",
                "fleet_encoder.fc2",
                "fleet_encoder.norm",
                "planet_encoder.scalar.fc2",
                "planet_encoder.traj.proj",
                "planet_encoder.gate",
                "planet_encoder.norm",
            ],
            "lr_table": {
                "cross": 1e-3,
                "heads": 1e-3,
                "entity_encoder": 1e-4,
                "fleet_encoder.fc2": 1e-5,
                "fleet_encoder.norm": 1e-5,
                "planet_encoder.scalar.fc2": 1e-5,
                "planet_encoder.traj.proj": 1e-5,
                "planet_encoder.gate": 1e-5,
                "planet_encoder.norm": 1e-5,
            },
        },
        {
            "index": 3,
            "name": "full-unfreeze",
            "trainable_paths": [
                "cross",
                "heads",
                "entity_encoder",
                "fleet_encoder",
                "planet_encoder",
            ],
            "lr_table": {
                "cross": 1e-3,
                "heads": 1e-3,
                "entity_encoder": 1e-4,
                "fleet_encoder.fc2": 1e-5,
                "fleet_encoder.norm": 1e-5,
                "fleet_encoder.fc1": 1e-6,
                "planet_encoder.scalar.fc2": 1e-5,
                "planet_encoder.traj.proj": 1e-5,
                "planet_encoder.gate": 1e-5,
                "planet_encoder.norm": 1e-5,
                "planet_encoder.scalar.fc1": 1e-6,
                "planet_encoder.traj.conv1": 1e-6,
                "planet_encoder.traj.conv2": 1e-6,
            },
        },
        {
            "index": 4,
            "name": "low-lr-settle",
            "trainable_paths": [
                "cross",
                "heads",
                "entity_encoder",
                "fleet_encoder",
                "planet_encoder",
            ],
            "lr_table": {
                "cross": 1e-5,
                "heads": 1e-5,
                "entity_encoder": 1e-5,
                "fleet_encoder": 1e-5,
                "planet_encoder": 1e-5,
            },
        },
    ]

    if not stage_epochs:
        raise ValueError("stage_epochs must contain at least one entry")
    if len(stage_epochs) > len(templates):
        raise ValueError(
            f"got {len(stage_epochs)} stage epochs, but only {len(templates)} "
            "gradual-unfreeze stages exist"
        )

    schedule: list[dict[str, Any]] = []
    start_epoch = 1
    for template, n_epochs in zip(templates, stage_epochs):
        if n_epochs <= 0:
            continue
        stage = dict(template)
        stage["epochs"] = n_epochs
        stage["start_epoch"] = start_epoch
        stage["end_epoch"] = start_epoch + n_epochs - 1
        schedule.append(stage)
        start_epoch += n_epochs
    return schedule


def _stage_for_epoch(
    schedule: list[dict[str, Any]],
    epoch: int,
) -> dict[str, Any]:
    for stage in schedule:
        if stage["start_epoch"] <= epoch <= stage["end_epoch"]:
            return stage
    raise ValueError(f"epoch {epoch} is outside the gradual-unfreeze schedule")


def _load_cross_checkpoint(
    ckpt_path: Path,
    *,
    d_model: int,
    device: str,
) -> tuple[CrossEntityPretrainModel, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_d_model = ckpt.get("config", {}).get("d_model", d_model)
    if ckpt_d_model != d_model:
        raise ValueError(
            f"checkpoint d_model={ckpt_d_model} != requested d_model={d_model}"
        )
    model = CrossEntityPretrainModel(d_model=ckpt_d_model)
    state = ckpt.get("cross_model") or ckpt.get("model")
    if state is None:
        raise KeyError(f"{ckpt_path} does not contain a cross-entity model state")
    model.load_state_dict(state)
    return model, ckpt


def _save_gradual_checkpoint(
    path: Path,
    *,
    stack: CrossEntityTrainStack,
    epoch: int,
    stage: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            # Backward-compatible key for tools that only care about the
            # cross-entity transformer + heads.
            "model": stack.cross_model.state_dict(),
            "cross_model": stack.cross_model.state_dict(),
            "fleet_encoder": stack.fleet_encoder.state_dict(),
            "planet_encoder": stack.planet_encoder.state_dict(),
            "entity_encoder": stack.entity_encoder.state_dict(),
            "epoch": epoch,
            "stage": stage["index"],
            "stage_name": stage["name"],
            "config": config,
        },
        path,
    )


@torch.no_grad()
def evaluate_stack(
    stack: CrossEntityTrainStack,
    loader: DataLoader,
    device: str,
) -> dict[str, dict[str, float]]:
    stack.eval()
    summary = evaluate(
        stack.cross_model,
        stack.fleet_encoder,
        stack.planet_encoder,
        stack.entity_encoder,
        loader,
        device,
    )
    stack.train()
    return summary


# ---------- Train loop ----------
def train(
    *,
    out_dir: Path,
    fleet_run_dir: Path,
    planet_run_dir: Path,
    entity_run_dir: Path,
    d_model: int = 128,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 1,
    max_planets: int = 64,
    max_fleets: int = 1024,
    num_workers: int = 0,
    num_load_workers: int | None = None,
    device: str | None = None,
    seed: int = 0,
    cross_cache_path: Path | None = None,
    head_set: str = "v1",
    init_from: Path | None = None,
    freeze_backbone: bool = True,
    unfreeze: str = "none",
    backbone_lr_mult: float = 0.1,
    aux_posterior: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    manifest_path = CROSS_ENTITY_DATASET_DIR / "manifest.json"
    manifest: dict[str, list[str]] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    elif cross_cache_path is None:
        raise FileNotFoundError(
            f"missing cross-entity manifest: {manifest_path}. "
            "CSV training needs the manifest; cached training can run "
            "without it."
        )

    def stems_of(split: str) -> list[str]:
        if manifest is None:
            raise RuntimeError("manifest is required for CSV split loading")
        return [
            n.removeprefix("cross_entity_").removesuffix(".csv")
            for n in manifest[split]
        ]

    splits: dict[str, torch.utils.data.Dataset] = {}
    if cross_cache_path is not None:
        # Fast path: load one .pt cache covering all stems and slice by
        # the manifest's per-split stem lists. Skips the per-CSV walk.
        print(
            f"[cross-entity-pretrain] loading cache: {cross_cache_path} ...",
            flush=True,
        )
        full = CachedCrossEntitySnapshotDataset(
            cross_cache_path,
            per_frame_label_keys=(
                V3_PER_FRAME_LABEL_KEYS if head_set == "v3" else None
            ),
            with_posterior=(head_set == "critic" and aux_posterior),
        )
        print(
            f"  cache size: {len(full):,} snapshots; "
            f"keys span {len({k[0] for k in full.keys})} episodes",
            flush=True,
        )
        splits = _split_cached_cross_dataset(full, manifest, seed=seed)
        for split in ("train", "val", "test"):
            subset = splits[split]
            idxs = list(subset.indices) if isinstance(subset, torch.utils.data.Subset) else []
            eps = {full.keys[i][0] for i in idxs}
            print(
                f"  {split}: {len(subset):,} snapshots from "
                f"{len(eps)} episodes",
                flush=True,
            )
    else:
        if num_load_workers is None:
            num_load_workers = max(1, num_workers)
        print(
            f"[cross-entity-pretrain] loading CSVs "
            f"(num_load_workers={num_load_workers}) ...",
            flush=True,
        )
        for split in ("train", "val", "test"):
            stems = stems_of(split)
            splits[split] = CrossEntitySnapshotDataset(
                planet_csv_paths=[PLANET_DATASET_DIR / f"planet_{s}.csv" for s in stems],
                fleet_csv_paths=[FLEET_DATASET_DIR / f"fleet_{s}.csv" for s in stems],
                entity_csv_paths=[ENTITY_DATASET_DIR / f"entity_{s}.csv" for s in stems],
                cross_entity_csv_paths=[
                    CROSS_ENTITY_DATASET_DIR / f"cross_entity_{s}.csv" for s in stems
                ],
                max_planets=max_planets, max_fleets=max_fleets,
                num_load_workers=num_load_workers,
            )
            print(f"  {split}: {len(splits[split])} snapshots")

    train_loader = _make_loader(
        splits["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = _make_loader(
        splits["val"], batch_size=batch_size, num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = _make_loader(
        splits["test"], batch_size=batch_size, num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"[cross-entity-pretrain] loading frozen encoders ({device}) ...")
    fenc, penc, eenc = _load_frozen_encoders(
        fleet_run_dir, planet_run_dir, entity_run_dir, device=device,
    )

    if head_set not in ("v1", "v2", "v3", "critic"):
        raise ValueError(f"unknown head_set={head_set!r}; expected 'v1', 'v2', 'v3', or 'critic'")
    if head_set == "v3" and cross_cache_path is None:
        raise ValueError(
            "head_set='v3' requires --cross-cache-path: per-frame supervision "
            "stacks history-frame labels from the cache (the CSV path only "
            "carries current-frame labels)."
        )
    if aux_posterior and (head_set != "critic" or cross_cache_path is None):
        raise ValueError(
            "--aux-posterior requires head_set='critic' AND --cross-cache-path "
            "(the future-window stacking is only built on the cached path)."
        )
    if head_set == "v3":
        model = CrossEntityPretrainModelV3(d_model=d_model).to(device)
        print(f"  V3 temporal cross+per-frame heads params: {sum(p.numel() for p in model.parameters()):,}")
    elif head_set == "v2":
        model = CrossEntityPretrainModelV2(d_model=d_model).to(device)
        print(f"  V2 cross+consolidator+heads params: {sum(p.numel() for p in model.parameters()):,}")
    elif head_set == "critic":
        model = CrossEntityCriticModel(d_model=d_model).to(device)
        print(f"  critic backbone+pair_compare params: {sum(p.numel() for p in model.parameters()):,}")
    else:
        model = CrossEntityPretrainModel(d_model=d_model).to(device)
        print(f"  V1 cross+heads params: {sum(p.numel() for p in model.parameters()):,}")

    # Warm-start (+ optional freeze) — used by the critic to load a V2
    # backbone and train pair_compare/is_ahead heads plus optional backbone
    # blocks.
    if init_from is not None:
        sd = torch.load(Path(init_from), map_location=device, weights_only=False)
        state = sd.get("model") or sd.get("cross_model")
        if state is None:
            raise KeyError(f"--init-from ckpt {init_from} has no 'model'/'cross_model' state")
        res = model.load_state_dict(state, strict=False)
        miss = list(res.missing_keys)
        # For the critic, the backbone (cross.* / consolidator.*) MUST load;
        # the new heads (pair_compare/is_ahead + momentum heads) are expected to be at
        # init. Flag only a missing backbone key as a real problem.
        bad = [k for k in miss if k.startswith("cross.") or k.startswith("consolidator.")]
        n_heads = len(miss) - len(bad)
        print(
            f"[critic] warm-start from {init_from}: missing={len(miss)} "
            f"(new heads: {n_heads}, backbone: {len(bad)}), "
            f"unexpected={len(res.unexpected_keys)}"
        )
        if head_set == "critic" and bad:
            raise RuntimeError(
                f"warm-start left BACKBONE keys uninitialized: {bad[:6]} "
                "— backbone ctor args must match the source V2 ckpt."
            )
    if head_set == "critic":
        # ``unfreeze`` controls which backbone blocks fine-tune alongside the
        # (always-trainable) critic heads. ``--no-freeze-backbone`` maps to
        # "all" for back-compat. Unfrozen pretrained blocks get a SMALLER LR
        # (lr * backbone_lr_mult) via discriminative param groups, so the
        # fresh pair/is_ahead heads keep learning at full lr while the warm-started L2 /
        # consolidator drift slowly — the fix for the frozen-backbone overfit
        # without starving the heads.
        unfreeze_eff = "all" if not freeze_backbone else unfreeze
        if unfreeze_eff == "all":
            unfrozen_mods = [model.cross, model.consolidator]
        elif unfreeze_eff == "consolidator":
            unfrozen_mods = [model.consolidator]
        elif unfreeze_eff == "l2":
            unfrozen_mods = [model.cross]
        else:  # "none"
            unfrozen_mods = []
        unfrozen_ids = {id(p) for m in unfrozen_mods for p in m.parameters()}
        # Freeze the backbone modules NOT in the unfrozen set.
        n_frozen = 0
        for m in (model.cross, model.consolidator):
            for p in m.parameters():
                if id(p) not in unfrozen_ids:
                    p.requires_grad_(False)
                    n_frozen += p.numel()
        decoder_params = (
            list(model.pair_compare.parameters())
            + list(model.is_ahead_head.parameters())
            + list(model.head_dships_back.parameters())
            + list(model.head_dships_fwd.parameters())
            + list(model.head_survives_fwd.parameters())
            + list(model.head_slope_back.parameters())
        )
        backbone_params = [p for m in unfrozen_mods for p in m.parameters()]
        groups = [{"params": decoder_params, "lr": lr}]
        bb_lr = lr * backbone_lr_mult
        if backbone_params:
            groups.append({"params": backbone_params, "lr": bb_lr})
        n_dec = sum(p.numel() for p in decoder_params)
        n_bb = sum(p.numel() for p in backbone_params)
        print(
            f"[critic] unfreeze={unfreeze_eff}  frozen={n_frozen:,}  "
            f"critic_heads={n_dec:,}@lr={lr:g}  backbone={n_bb:,}@lr={bb_lr:g}"
        )
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    config = {
        "d_model": d_model, "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": epochs,
        "fleet_run_dir": str(fleet_run_dir),
        "planet_run_dir": str(planet_run_dir),
        "entity_run_dir": str(entity_run_dir),
        "head_set": head_set,
        "history_offsets": list(HISTORY_OFFSETS),
        "init_from": str(init_from) if init_from is not None else None,
        "freeze_backbone": freeze_backbone,
        "unfreeze": unfreeze,
        "backbone_lr_mult": backbone_lr_mult,
        "aux_posterior": aux_posterior,
        "critic_lambda_is_ahead": CRITIC_LAMBDA_IS_AHEAD,
        "critic_lambda_post": CRITIC_LAMBDA_POST,
        "critic_lambda_cons": CRITIC_LAMBDA_CONS,
    }

    log: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "cross_entity_best.pt"
    last_path = out_dir / "cross_entity_last.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        n_batches = 0
        running_per_head: dict[str, float] = defaultdict(float)
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                entity_tokens, entity_mask = _entity_tokens_per_step(
                    batch, fenc, penc, eenc,
                )
            if head_set == "v3":
                owner_oh = _owner_oh_from_batch(batch)
                preds = model(entity_tokens, entity_mask, owner_oh)
                total_loss, per_head = compute_loss_v3(preds, batch)
            elif head_set == "critic":
                owner_oh = _owner_oh_from_batch(batch)
                preds = model(entity_tokens, entity_mask, owner_oh)
                post_preds = None
                if aux_posterior:
                    # Build the posterior (future-window) sub-batch by mapping
                    # the {key}_post features back to base keys, then run the
                    # SHARED model on it (L0/L1 frozen → no_grad encode).
                    pk = CachedCrossEntitySnapshotDataset._STACK_KEYS
                    post_batch = {k: batch[f"{k}_post"] for k in pk if f"{k}_post" in batch}
                    with torch.no_grad():
                        post_et, post_em = _entity_tokens_per_step(
                            post_batch, fenc, penc, eenc,
                        )
                    post_owner = _owner_oh_from_batch(post_batch)
                    post_preds = model(post_et, post_em, post_owner)
                total_loss, per_head = compute_loss_critic(preds, batch, post_preds=post_preds)
            elif head_set == "v2":
                owner_oh = _owner_oh_from_batch(batch)
                preds = model(entity_tokens, entity_mask, owner_oh)
                total_loss, per_head = compute_loss_v2(preds, batch)
            else:
                preds = model(entity_tokens, entity_mask)
                # Loss is over the CURRENT step's planet mask (B, P).
                cur_planet_mask = (
                    batch["planet_mask"][:, -1]
                    if batch["planet_mask"].dim() == 3
                    else batch["planet_mask"]
                )
                cur_batch = dict(batch)
                cur_batch["planet_mask"] = cur_planet_mask
                total_loss, per_head = compute_loss(preds, cur_batch, cur_planet_mask)
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

        _eval_fn = {"v3": evaluate_v3, "v2": evaluate_v2, "critic": evaluate_critic}.get(head_set, evaluate)
        if epoch % eval_every == 0 or epoch == epochs:
            val = _eval_fn(model, fenc, penc, eenc, val_loader, device)
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

    print("\n[cross-entity-pretrain] evaluating best on test ...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = _eval_fn(model, fenc, penc, eenc, test_loader, device)
    print(_format_summary(test))
    (out_dir / "test_summary.json").write_text(json.dumps(test, indent=2))
    print(f"\n[cross-entity-pretrain] outputs in {out_dir}")
    return best_path


def train_gradual_unfreeze(
    *,
    out_dir: Path,
    fleet_run_dir: Path,
    planet_run_dir: Path,
    entity_run_dir: Path,
    resume_cross_pt: Path,
    d_model: int = 128,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    eval_every: int = 1,
    max_planets: int = 64,
    max_fleets: int = 1024,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 0,
    stage_epochs: list[int] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    stage_epochs = stage_epochs or [5]
    schedule = _build_gradual_unfreeze_schedule(stage_epochs)
    total_epochs = schedule[-1]["end_epoch"]

    manifest = json.loads((CROSS_ENTITY_DATASET_DIR / "manifest.json").read_text())

    def stems_of(split: str) -> list[str]:
        return [
            n.removeprefix("cross_entity_").removesuffix(".csv")
            for n in manifest[split]
        ]

    print("[cross-entity-gradual] loading CSVs ...")
    splits = {}
    for split in ("train", "val", "test"):
        stems = stems_of(split)
        splits[split] = CrossEntitySnapshotDataset(
            planet_csv_paths=[PLANET_DATASET_DIR / f"planet_{s}.csv" for s in stems],
            fleet_csv_paths=[FLEET_DATASET_DIR / f"fleet_{s}.csv" for s in stems],
            entity_csv_paths=[ENTITY_DATASET_DIR / f"entity_{s}.csv" for s in stems],
            cross_entity_csv_paths=[
                CROSS_ENTITY_DATASET_DIR / f"cross_entity_{s}.csv" for s in stems
            ],
            max_planets=max_planets,
            max_fleets=max_fleets,
        )
        print(f"  {split}: {len(splits[split])} snapshots")

    train_loader = _make_loader(
        splits["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = _make_loader(
        splits["val"], batch_size=batch_size, num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = _make_loader(
        splits["test"], batch_size=batch_size, num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"[cross-entity-gradual] loading upstream encoders ({device}) ...")
    fenc, penc, eenc = _load_frozen_encoders(
        fleet_run_dir, planet_run_dir, entity_run_dir, device=device,
    )
    cross_model, resume_ckpt = _load_cross_checkpoint(
        resume_cross_pt, d_model=d_model, device=device,
    )
    cross_model.to(device)

    # If resuming from a prior gradual-unfreeze checkpoint, carry over the
    # fine-tuned encoder weights too. If the checkpoint came from the
    # original frozen run, these keys are absent and we keep the upstream
    # encoder checkpoints as-is.
    if "fleet_encoder" in resume_ckpt:
        fenc.load_state_dict(resume_ckpt["fleet_encoder"])
    if "planet_encoder" in resume_ckpt:
        penc.load_state_dict(resume_ckpt["planet_encoder"])
    if "entity_encoder" in resume_ckpt:
        eenc.load_state_dict(resume_ckpt["entity_encoder"])

    stack = CrossEntityTrainStack(
        fleet_encoder=fenc,
        planet_encoder=penc,
        entity_encoder=eenc,
        cross_model=cross_model,
    ).to(device)

    config = {
        "train_mode": "gradual-unfreeze",
        "d_model": d_model,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "fleet_run_dir": str(fleet_run_dir),
        "planet_run_dir": str(planet_run_dir),
        "entity_run_dir": str(entity_run_dir),
        "resume_cross_pt": str(resume_cross_pt),
        "stage_epochs": list(stage_epochs),
        "schedule": [
            {
                "index": stage["index"],
                "name": stage["name"],
                "start_epoch": stage["start_epoch"],
                "end_epoch": stage["end_epoch"],
                "epochs": stage["epochs"],
                "lr_table": dict(stage["lr_table"]),
            }
            for stage in schedule
        ],
    }

    print("[cross-entity-gradual] schedule:")
    for stage in schedule:
        print(
            f"  stage {stage['index']}: {stage['name']}  "
            f"epochs={stage['start_epoch']}..{stage['end_epoch']}"
        )

    log: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "cross_entity_best.pt"
    last_path = out_dir / "cross_entity_last.pt"
    current_stage_idx: int | None = None
    opt: torch.optim.Optimizer | None = None

    t0 = time.time()
    for epoch in range(1, total_epochs + 1):
        stage = _stage_for_epoch(schedule, epoch)
        if stage["index"] != current_stage_idx:
            _freeze_all(stack)
            set_trainable(
                stack,
                freeze=[],
                unfreeze=stage["trainable_paths"],
            )
            opt = torch.optim.AdamW(
                build_param_groups(stack, stage["lr_table"]),
                weight_decay=weight_decay,
            )
            current_stage_idx = stage["index"]
            print(
                f"[stage] {stage['index']} {stage['name']}  "
                f"epochs={stage['start_epoch']}..{stage['end_epoch']}  "
                f"param_groups={len(opt.param_groups)}"
            )

        assert opt is not None
        stack.train()
        running_total = 0.0
        n_batches = 0
        running_per_head: dict[str, float] = defaultdict(float)
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = stack(batch)
            cur_planet_mask = _current_planet_mask(batch)
            cur_batch = dict(batch)
            cur_batch["planet_mask"] = cur_planet_mask
            total_loss, per_head = compute_loss(preds, cur_batch, cur_planet_mask)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            running_total += float(total_loss.detach())
            for k, v in per_head.items():
                running_per_head[k] += v
            n_batches += 1

        train_total = running_total / max(1, n_batches)
        train_per_head = {
            k: v / max(1, n_batches)
            for k, v in running_per_head.items()
        }
        elapsed = round(time.time() - t0, 2)
        entry: dict[str, Any] = {
            "epoch": epoch,
            "stage": stage["index"],
            "stage_name": stage["name"],
            "train_total": train_total,
            "train_per_head": train_per_head,
            "elapsed_s": elapsed,
        }

        if epoch % eval_every == 0 or epoch == total_epochs:
            val = evaluate_stack(stack, val_loader, device)
            mean = sum(m["loss"] for m in val.values()) / max(1, len(val))
            entry["val_mean_loss"] = mean
            entry["val"] = val
            print(
                f"[ep {epoch:>2}/{total_epochs}]  stage={stage['index']} "
                f"{stage['name']}  train_total={train_total:.4f}  "
                f"val_mean={mean:.4f}  ({elapsed}s)"
            )
            if mean < best_val:
                best_val = mean
                _save_gradual_checkpoint(
                    best_path,
                    stack=stack,
                    epoch=epoch,
                    stage=stage,
                    config=config,
                )

        if epoch == stage["end_epoch"]:
            stage_path = out_dir / f"cross_entity_stage{stage['index']}_epoch{epoch}.pt"
            _save_gradual_checkpoint(
                stage_path,
                stack=stack,
                epoch=epoch,
                stage=stage,
                config=config,
            )

        log.append(entry)
        _save_gradual_checkpoint(
            last_path,
            stack=stack,
            epoch=epoch,
            stage=stage,
            config=config,
        )
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    print("\n[cross-entity-gradual] evaluating best on test ...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    stack.cross_model.load_state_dict(ckpt["cross_model"])
    if "fleet_encoder" in ckpt:
        stack.fleet_encoder.load_state_dict(ckpt["fleet_encoder"])
    if "planet_encoder" in ckpt:
        stack.planet_encoder.load_state_dict(ckpt["planet_encoder"])
    if "entity_encoder" in ckpt:
        stack.entity_encoder.load_state_dict(ckpt["entity_encoder"])
    test = evaluate_stack(stack, test_loader, device)
    print(_format_summary(test))
    (out_dir / "test_summary.json").write_text(json.dumps(test, indent=2))
    print(f"\n[cross-entity-gradual] outputs in {out_dir}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-mode",
        choices=("frozen", "gradual-unfreeze"),
        default="frozen",
        help="`frozen` trains only cross+heads; `gradual-unfreeze` resumes "
             "from a cross-entity checkpoint and thaws lower encoders in stages.",
    )
    parser.add_argument("--fleet-run-dir", type=Path, default=None,
                        help="Default: latest under data/runs/fleet/")
    parser.add_argument("--planet-run-dir", type=Path, default=None)
    parser.add_argument("--entity-run-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-cross-pt",
        type=Path,
        default=None,
        help="Cross-entity checkpoint (.pt) to resume from for "
             "`--train-mode gradual-unfreeze`. Default: latest "
             "`cross_entity_best.pt` under data/runs/cross_entity/.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--stage-epochs",
        type=str,
        default="5",
        help="Comma-separated epoch counts for gradual-unfreeze stages "
             "(starts at Stage 1 because the frozen cross run is assumed "
             "to already exist). Examples: `5`, `5,5`, `5,5,5`, `5,5,5,5`.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--num-load-workers",
        type=int,
        default=None,
        help="CSV stem parsing workers used during dataset construction. "
             "Defaults to max(1, --num-workers). Only affects the CSV path; "
             "cached training does not use it.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cross-cache-path", type=Path, default=None,
        help="Path to a pre-built .pt cache (see "
             "agents.transformer_v2.pretrain.cross_entity."
             "build_cross_entity_cache). When provided, the CSV walk is "
             "skipped and the three splits are formed by slicing the "
             "cache via the manifest's per-split stem lists.",
    )
    parser.add_argument(
        "--head-set", choices=("v1", "v2", "v3", "critic"), default="v1",
        help="v1 (default): legacy 25-head Tier-1/2/3/4 menu. "
             "v2: PlayerConsolidator + slimmed advantage/future heads "
             "(per-player winner/leader_k via BCE, learner is_ahead_k / "
             "score_adv_k / score_adv_end / turns_left). "
             "v3: TEMPORAL — per-frame CLS+player tokens inside a "
             "block-causal L2, deep supervision. Requires --cross-cache-path. "
             "critic: PairCompareHead on V2 player_state plus per-player "
             "is_ahead BCE derived from leader_seat_t_plus_K; pair with "
             "--init-from a V2 ckpt, then choose --unfreeze none/l2/all. "
             "The ckpts are NOT cross-compatible.",
    )
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="Warm-start the backbone (cross.* / consolidator.*) from a prior "
             "ckpt (strict=False). For head_set=critic, load a V2 playerpair "
             "ckpt; pair_compare.* and is_ahead_head.* start fresh.",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true", default=True,
        help="(critic) freeze L2 + PlayerConsolidator; train only critic "
             "heads. On by default; override granularly with --unfreeze.",
    )
    parser.add_argument(
        "--no-freeze-backbone", dest="freeze_backbone", action="store_false",
        help="(critic) fine-tune the whole stack (equivalent to --unfreeze all).",
    )
    parser.add_argument(
        "--unfreeze", choices=("none", "l2", "consolidator", "all"), default="none",
        help="(critic) which backbone blocks fine-tune with the critic heads: "
             "none (freeze L2 + consolidator), l2 (unfreeze L2/CrossEntityAttention, "
             "keep consolidator frozen), consolidator (train PlayerConsolidator "
             "while preserving the actor's L2), all (L2 + consolidator). "
             "Unfrozen blocks use a smaller LR (--backbone-lr-mult x --lr) so "
             "the warm-started representation drifts slowly.",
    )
    parser.add_argument(
        "--backbone-lr-mult", type=float, default=0.1,
        help="(critic) LR multiplier for unfrozen backbone blocks relative to "
             "--lr (the fresh critic heads always train at full --lr). "
             "Default 0.1 (10x smaller) for gentle pretrained-layer fine-tuning.",
    )
    parser.add_argument(
        "--aux-posterior", action="store_true",
        help="(critic) prior/posterior consistency aux: a future mirror of "
             "HISTORY_OFFSETS (for T=10, frames t+45,t+40,...,t) is fit to "
             "the same outcome and the deployable prior (t-45,...,t) logits "
             "are pulled toward the stop-grad posterior logits. Trains a "
             "future-informed-but-not-future-leaking critic; deployment uses "
             "only the prior view. Requires --cross-cache-path.",
    )
    parser.add_argument(
        "--lambda-cons", type=float, default=None,
        help="(critic, aux) weight on the prior->posterior consistency MSE "
             "(default 0.2).",
    )
    parser.add_argument(
        "--lambda-post", type=float, default=None,
        help="(critic, aux) weight on the posterior view's own outcome loss "
             "(default 0.5).",
    )
    args = parser.parse_args()

    fleet_run = args.fleet_run_dir or _latest_run_dir(FLEET_RUNS_DIR, "fleet_encoder_best.pt")
    planet_run = args.planet_run_dir or _latest_run_dir(PLANET_RUNS_DIR, "planet_encoder_best.pt")
    entity_run = args.entity_run_dir or _latest_run_dir(ENTITY_RUNS_DIR, "entity_encoder_best.pt")
    out_dir = args.out_dir or (CROSS_ENTITY_RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"))
    cross_cache_path = args.cross_cache_path
    # Optional λ overrides for the critic aux (mutate module globals read by
    # compute_loss_critic).
    if args.lambda_cons is not None:
        globals()["CRITIC_LAMBDA_CONS"] = float(args.lambda_cons)
    if args.lambda_post is not None:
        globals()["CRITIC_LAMBDA_POST"] = float(args.lambda_post)
    if args.train_mode == "frozen":
        train(
            out_dir=out_dir,
            fleet_run_dir=fleet_run,
            planet_run_dir=planet_run,
            entity_run_dir=entity_run,
            d_model=args.d_model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            max_planets=args.max_planets,
            max_fleets=args.max_fleets,
            num_workers=args.num_workers,
            num_load_workers=args.num_load_workers,
            device=args.device,
            seed=args.seed,
            cross_cache_path=cross_cache_path,
            head_set=args.head_set,
            init_from=args.init_from,
            freeze_backbone=args.freeze_backbone,
            unfreeze=args.unfreeze,
            backbone_lr_mult=args.backbone_lr_mult,
            aux_posterior=args.aux_posterior,
        )
        return

    stage_epochs = [
        int(part.strip())
        for part in args.stage_epochs.split(",")
        if part.strip()
    ]
    resume_cross_pt = args.resume_cross_pt
    if resume_cross_pt is None:
        resume_cross_pt = (
            _latest_run_dir(CROSS_ENTITY_RUNS_DIR, "cross_entity_best.pt")
            / "cross_entity_best.pt"
        )
    train_gradual_unfreeze(
        out_dir=out_dir,
        fleet_run_dir=fleet_run,
        planet_run_dir=planet_run,
        entity_run_dir=entity_run,
        resume_cross_pt=resume_cross_pt,
        d_model=args.d_model,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        max_planets=args.max_planets,
        max_fleets=args.max_fleets,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        stage_epochs=stage_epochs,
    )


if __name__ == "__main__":
    main()
