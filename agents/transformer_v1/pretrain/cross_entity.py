"""Cross-entity attention pretraining.

Loads the three frozen upstream encoders (Fleet, Planet, PlanetEntity)
and trains :class:`CrossEntityAttention` + supervision heads on the
Tier-1/2/3/4 cross-entity labels plus selected entity-label reuse
targets stored in
``data/datasets/cross_entity/``. See
``agents/transformer_v1/aggregator/README.md`` for the design.

Run from the repo root:

    python -m agents.transformer_v1.pretrain.cross_entity \\
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

from ..aggregator import CrossEntityAttention
from ..encoder.entity_encoder import PlanetEntityEncoder
from ..encoder.fleet_encoder import FleetEncoder
from ..encoder.planet_encoder import PlanetEncoder
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
        n_history: int = 3,
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

        # Build (episode_id, turn) → snapshot index map for cheap
        # history lookup at __getitem__ time. ``self.keys`` is sorted,
        # so within an episode turns are contiguous.
        self.n_history = n_history
        self._key_to_idx: dict[tuple[str, int], int] = {
            k: i for i, k in enumerate(self.keys)
        }

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
        """Return the current snapshot stacked with up to ``n_history-1``
        past turns from the same episode. Missing past turns (start of
        episode) are zero-filled; their masks stay all-False so the
        attention layer ignores them.
        """
        cur = self.snapshots[idx]
        ep, t = self.keys[idx]

        # Walk T steps back, oldest first. ``offset = n_history - 1``
        # corresponds to the oldest frame; ``offset = 0`` is the
        # current turn. Same convention as ``CrossEntityAttention``'s
        # step embedding indexing (oldest first).
        offsets = list(range(self.n_history - 1, -1, -1))
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


# ---------- Model ----------
class CrossEntityPretrainModel(nn.Module):
    """``CrossEntityAttention`` + label heads. The three upstream
    encoders + the entity encoder are held externally (frozen, no_grad)
    so this module's optimizer steps cross-attention + heads only.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.cross = CrossEntityAttention(
            d_model=d_model, n_heads=4, n_layers=3,
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

    penc = PlanetEncoder(d_model=pc["config"]["d_model"])
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
    d_model: int = 64,
    batch_size: int = 64,
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

    manifest = json.loads((CROSS_ENTITY_DATASET_DIR / "manifest.json").read_text())

    def stems_of(split: str) -> list[str]:
        return [
            n.removeprefix("cross_entity_").removesuffix(".csv")
            for n in manifest[split]
        ]

    print("[cross-entity-pretrain] loading CSVs ...")
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
            max_planets=max_planets, max_fleets=max_fleets,
        )
        print(f"  {split}: {len(splits[split])} snapshots")

    train_loader = DataLoader(
        splits["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(splits["val"], batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(splits["test"], batch_size=batch_size, num_workers=num_workers)

    print(f"[cross-entity-pretrain] loading frozen encoders ({device}) ...")
    fenc, penc, eenc = _load_frozen_encoders(
        fleet_run_dir, planet_run_dir, entity_run_dir, device=device,
    )

    model = CrossEntityPretrainModel(d_model=d_model).to(device)
    print(f"  cross+heads params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    config = {
        "d_model": d_model, "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": epochs,
        "fleet_run_dir": str(fleet_run_dir),
        "planet_run_dir": str(planet_run_dir),
        "entity_run_dir": str(entity_run_dir),
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

        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate(model, fenc, penc, eenc, val_loader, device)
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
    test = evaluate(model, fenc, penc, eenc, test_loader, device)
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
    d_model: int = 64,
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

    train_loader = DataLoader(
        splits["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        splits["val"], batch_size=batch_size, num_workers=num_workers,
    )
    test_loader = DataLoader(
        splits["test"], batch_size=batch_size, num_workers=num_workers,
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
    parser.add_argument("--d-model", type=int, default=64)
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
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fleet_run = args.fleet_run_dir or _latest_run_dir(FLEET_RUNS_DIR, "fleet_encoder_best.pt")
    planet_run = args.planet_run_dir or _latest_run_dir(PLANET_RUNS_DIR, "planet_encoder_best.pt")
    entity_run = args.entity_run_dir or _latest_run_dir(ENTITY_RUNS_DIR, "entity_encoder_best.pt")
    out_dir = args.out_dir or (CROSS_ENTITY_RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"))
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
            device=args.device,
            seed=args.seed,
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
