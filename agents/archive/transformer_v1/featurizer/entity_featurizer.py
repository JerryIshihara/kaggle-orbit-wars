"""Per-(step, planet, player) inbound-fleet summary featurizer.

Sits one level above the fleet/planet featurizers. For each turn, walks
every fleet, resolves its first-hit planet via
``fleet_featurizer._resolve_target``, and groups by
``(target_planet, owner_relative_slot)``. Each group contributes seven
summary stats — *exactly* what
``encoder.entity_encoder.PlanetEntityEncoder`` (Plan B) computes
internally per ``(planet, player_slot)`` cell. Aligning the two means a
single row of this CSV is a reproducible offline reference for what
the encoder sees at training/inference time, without re-running the
attention pool.

Per cell:

  * ``count``           — number of inbound fleets from this player
  * ``ships_sum_log``   — ``log1p(Σ ships) / SHIPS_LOG_MAX``
  * ``ships_max_log``   — ``log1p(max ships) / SHIPS_LOG_MAX``
  * ``eta_min_norm``    — ``min(eta) / ETA_NORM`` (1.0 if cell empty —
                           saturated "no fleet" marker)
  * ``eta_mean_norm``   — average ETA, same normalization (0.0 empty)
  * ``eta_max_norm``    — ``max(eta) / ETA_NORM`` (0.0 if empty,
                           matching the encoder's "no max" convention)
  * ``eta_std_norm``    — std of normalized ETAs (0.0 if empty)

Output schema (wide form, one row per (turn, planet)):

    episode_id, turn,
    planet_id, is_comet, owner_id,
    p{0..NUM_OWNER_SLOTS-1}_count,
    p{0..NUM_OWNER_SLOTS-1}_ships_sum_log,
    p{0..NUM_OWNER_SLOTS-1}_ships_max_log,
    p{0..NUM_OWNER_SLOTS-1}_eta_min_norm,
    p{0..NUM_OWNER_SLOTS-1}_eta_mean_norm,
    p{0..NUM_OWNER_SLOTS-1}_eta_max_norm,
    p{0..NUM_OWNER_SLOTS-1}_eta_std_norm

That's ``5 + 7 * NUM_OWNER_SLOTS = 33`` columns. Wide form rather than
long because it joins one-to-one with rows from
``save_episode_planet_csv`` (also one row per (turn, planet)).

Labels are intentionally omitted — the user will add downstream
supervised targets later (e.g., owner-flip-by-arrival, capture margin,
fleet survival rate).

Player slot encoding mirrors fleet/planet featurizers: the *learner's*
slot is 0; other players' slots are ``(owner - learner) % num_players``.
This keeps the same-row representation valid regardless of seat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....physics_utils import (
    F_FROM,
    F_ID,
    F_OWNER,
    F_SHIPS,
    MAX_SPEED,
    P_ID,
    P_OWNER,
    P_PRODUCTION,
    P_SHIPS,
    P_X,
    P_Y,
)
from ..paths import ENTITY_DATASET_DIR
from .fleet_featurizer import (
    ETA_NORM,
    SHIPS_LOG_MAX,
    _resolve_target,
    signed_log1p,
)

NUM_OWNER_SLOTS = 4               # 1-, 2-, 3-, 4-player; relative slots
N_OWNER_CLASSES = NUM_OWNER_SLOTS + 1   # 4 player slots + neutral
# Per-cell stats — mirrors PlanetEntityEncoder's internal stats stack.
# Order matters; the CSV column writer follows this same sequence.
STAT_NAMES: tuple[str, ...] = (
    "count",
    "ships_sum_log",
    "ships_max_log",
    "eta_min_norm",
    "eta_mean_norm",
    "eta_max_norm",
    "eta_std_norm",
)
N_STATS_PER_PLAYER = len(STAT_NAMES)

# Lookahead horizons for the per-planet supervised labels. K=5 / K=10
# match what we already write to the planet CSV for ``owner_t+5`` and
# extend it with a 10-turn variant for longer-horizon supervision.
LABEL_HORIZONS: tuple[int, ...] = (5, 10)

# Per-arrival-counting horizons for ``ships_arriving_within_K``.
ARRIVAL_HORIZONS: tuple[int, ...] = (5, 10)

# ---- Tier-1 spatial-context constants ----
# Radius (board units) used to define "neighborhood" for the
# frontier / sector-balance labels. ~30% of the board diagonal so
# every planet has a nontrivial neighbor set without trivially
# covering the whole map.
SECTOR_RADIUS = 30.0
# Distances are normalized by board diagonal (~141) so the regression
# targets stay in roughly [0, 1].
DIST_NORM = 141.0
# Counts are normalized by an empirically-loose upper bound on
# planets within ``SECTOR_RADIUS`` (typical board has ~36 total
# planets, so 16 in a sector is a reasonable saturating denominator).
COUNT_NORM = 16.0

# Frontier classes (relative to the learner):
#   0 = interior_self  — owned by learner, no enemy within R
#   1 = border_self    — owned by learner, ≥1 enemy within R
#   2 = enemy          — owned by another (non-neutral) player
#   3 = neutral        — owner == -1
N_FRONTIER_CLASSES = 4

# ---- Tier-3 game-level constants ----
EPISODE_STEPS = 500   # nominal episode horizon; matches kaggle env config

# Short-horizon global value targets for cross-entity pretraining.
CROSS_ENTITY_VALUE_HORIZONS: tuple[int, ...] = (10, 20, 50)

# Tactical neighborhood labels. Kept shorter-range than the global value
# heads so they actually mean "local support / local threat".
CROSS_ENTITY_TACTICAL_HORIZONS: tuple[int, ...] = (5, 10)


# ---------- Records ----------
@dataclass
class EntityFeaturizer:
    """One per-planet summary row.

    ``per_player[k]`` is a 7-tuple ``(count, ships_sum_log,
    ships_max_log, eta_min_norm, eta_mean_norm, eta_max_norm,
    eta_std_norm)`` for player slot ``k`` (relative to the learner).
    Slot ``k`` with no inbound fleets uses the same defaults
    ``PlanetEntityEncoder`` produces internally for empty cells:
    counts/sums/maxes are 0, ``eta_min = 1.0`` (saturated "no fleet"
    marker), ``eta_mean/max/std = 0.0``.

    ``earliest_arrival_owner_slot`` is the player slot whose fleet
    arrives soonest at this planet, or ``NUM_OWNER_SLOTS`` (= 4) when
    no inbound fleet is present (treated as the "neutral / none"
    class). It complements the per-cell stats by collapsing the
    "who-strikes-first" question into a single categorical label.

    ``ships_arriving_within`` is a (player_slot, horizon) → log-norm
    sum-of-ships table used for the per-(planet, player) regression
    target ``ships_arriving_within_K``. Horizons live in
    :data:`ARRIVAL_HORIZONS`.
    """

    planet_id: int
    is_comet: bool
    owner_id: int
    current_tick: int
    per_player: list[tuple[int, float, float, float, float, float, float]]
    earliest_arrival_owner_slot: int
    ships_arriving_within: dict[int, list[float]]   # horizon → list[K] (per player_slot)


def _player_slot(owner_id: int, learner_slot: int, num_players: int) -> int | None:
    if not (0 <= owner_id < num_players):
        return None
    return (owner_id - learner_slot) % num_players


def _infer_learner_slot_from_episode_path(
    episode_path: Path,
    *,
    fallback: int = 0,
) -> int:
    """Infer learner slot from Kaggle-style replay stems ``<eid>_<n>_<slot>``.

    Local self-play replays use ``local<seed>_<n>_<winner>``; for those we
    intentionally keep the fallback because the last field is not a seat id.
    """
    stem = episode_path.stem.split(".")[0]
    parts = stem.split("_")
    if stem.startswith("local") or len(parts) < 3:
        return fallback
    if not (parts[-1].isdigit() and parts[-2].isdigit()):
        return fallback
    slot = int(parts[-1])
    num_players = int(parts[-2])
    if 0 <= slot < max(1, num_players):
        return slot
    return fallback


# ---------- Featurization ----------
def featurize_entity_inbound(
    obs: Any,
    *,
    learner_slot: int = 0,
    num_players: int = 4,
    max_entities: int = 64,
) -> list[EntityFeaturizer]:
    """Walk inbound fleets per planet, group by player slot, compute
    bag-of-stats. Returns one record per planet (truncated to
    ``max_entities`` if the env is unusually large).

    Stats per (planet, player slot) — kept aligned with
    :class:`PlanetEntityEncoder`'s internal stats stack:

      * ``count`` — number of inbound fleets from this player.
      * ``ships_sum_log`` — ``log1p(Σ ships) / SHIPS_LOG_MAX``.
      * ``ships_max_log`` — ``log1p(max ships) / SHIPS_LOG_MAX``.
      * ``eta_min_norm`` — ``min(eta) / ETA_NORM`` clipped to [0, 1].
        Indicates how soon the soonest fleet will land.
      * ``eta_mean_norm`` — average ETA, same normalization.
      * ``eta_max_norm`` — ``max(eta) / ETA_NORM`` (latest arrival).
      * ``eta_std_norm`` — std of normalized ETAs across the cell;
        a tight spread (low std) means the fleets land in a coordinated
        burst, a wide spread spreads pressure across the planet's life.

    Empty cells are encoded as
    ``(0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)`` — matches the encoder's
    runtime behavior: ``eta_min = 1.0`` (saturated "no fleet" marker),
    ``eta_mean/max/std = 0.0`` (no average / max / spread).
    """
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    raw_planets = list(get("planets") or [])
    raw_fleets = list(get("fleets") or [])
    comet_ids = set(get("comet_planet_ids") or [])
    current_tick = int(get("step", 0) or 0)

    # Group by (target_planet_id, player_slot) → list of (ships, eta).
    # Targets resolved by the fleet's first-hit planet (same as the
    # fleet featurizer); fleets that miss every disc contribute to
    # nothing here.
    grouped: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for f in raw_fleets:
        target_pid, eta, _target_planet = _resolve_target(f, raw_planets)
        if target_pid is None or eta is None:
            continue
        slot = _player_slot(int(f[F_OWNER]), learner_slot, num_players)
        if slot is None:
            continue
        grouped.setdefault((int(target_pid), slot), []).append(
            (int(f[F_SHIPS]), float(eta))
        )

    records: list[EntityFeaturizer] = []
    # (count, ships_sum_log, ships_max_log,
    #  eta_min, eta_mean, eta_max, eta_std)
    # Empty cell defaults match what PlanetEntityEncoder computes
    # internally for a fully-masked group (see encoder/entity_encoder.py
    # _STATS branch): eta_min defaults to 1.0 (saturated marker), every
    # other stat is 0.0.
    empty_cell = (0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    for p in raw_planets[:max_entities]:
        pid = int(p[P_ID])
        per_player: list[tuple[int, float, float, float, float, float, float]] = []
        # Track earliest arriver per planet across player slots. Default
        # to NUM_OWNER_SLOTS (= "neutral / no inbound").
        earliest_eta = float("inf")
        earliest_slot = NUM_OWNER_SLOTS
        # Per-(player, horizon) sum of inbound ships counted only if
        # fleet ETA <= horizon. Materialized into log-norm scalars at
        # write time.
        arriving: dict[int, list[float]] = {h: [0.0] * NUM_OWNER_SLOTS for h in ARRIVAL_HORIZONS}

        for slot in range(NUM_OWNER_SLOTS):
            entries = grouped.get((pid, slot))
            if not entries:
                per_player.append(empty_cell)
                continue
            # ``ships_sum_log`` and ``ships_max_log`` aggregate the
            # *per-fleet* log-normed ships, matching what the encoder
            # computes at runtime: ``sum_i log1p(ships_i) / NORM`` (NOT
            # ``log1p(sum_i ships_i) / NORM``). The two are different
            # for >1 fleet per cell — keeping them aligned means a CSV
            # row is a faithful offline mirror of what the encoder sees.
            per_fleet_log = [math.log1p(max(0, e[0])) / SHIPS_LOG_MAX for e in entries]
            etas_norm = [min(1.0, e[1] / ETA_NORM) for e in entries]
            n = len(etas_norm)
            mean = sum(etas_norm) / n
            var = sum((x - mean) ** 2 for x in etas_norm) / n
            per_player.append((
                n,
                sum(per_fleet_log),
                max(per_fleet_log),
                min(etas_norm),
                mean,
                max(etas_norm),
                math.sqrt(max(0.0, var)),
            ))
            # Track earliest arriver across slots (raw eta, not norm,
            # to break ties cleanly).
            local_eta = min(e[1] for e in entries)
            if local_eta < earliest_eta:
                earliest_eta = local_eta
                earliest_slot = slot
            # Sum ships per arrival horizon
            for h in ARRIVAL_HORIZONS:
                s = sum(ships for ships, eta in entries if eta <= h)
                arriving[h][slot] = math.log1p(max(0, s)) / SHIPS_LOG_MAX

        records.append(EntityFeaturizer(
            planet_id=pid,
            is_comet=pid in comet_ids,
            owner_id=int(p[P_OWNER]),
            current_tick=current_tick,
            per_player=per_player,
            earliest_arrival_owner_slot=earliest_slot,
            ships_arriving_within=arriving,
        ))
    return records


# ---------- CSV columns ----------
def _entity_feature_columns() -> list[str]:
    cols: list[str] = []
    for slot in range(NUM_OWNER_SLOTS):
        for s in STAT_NAMES:
            cols.append(f"p{slot}_{s}")
    return cols


def _entity_label_columns() -> list[str]:
    """Per-planet labels for the entity-encoder pretrain. Single
    source of truth for the entity CSV header.

    Cross-attention-specific labels (Tier-1 spatial + Tier-3
    game-level) live in a separate ``cross_entity`` CSV — see
    :data:`CROSS_ENTITY_LABEL_COLS` and
    :func:`save_episode_cross_entity_csv`. Keeping them separate
    means the entity-encoder pretrain dataset stays stable while
    cross-attention adds new supervision additively.
    """
    cols: list[str] = []
    cols.append("earliest_arrival_owner_slot")
    cols.append("is_source_this_turn")
    cols.append("is_target_this_turn")
    for k in LABEL_HORIZONS:
        cols.append(f"owner_t_plus_{k}")
        cols.append(f"log_ships_t_plus_{k}")
        cols.append(f"valid_t_plus_{k}")
    for h in ARRIVAL_HORIZONS:
        for slot in range(NUM_OWNER_SLOTS):
            cols.append(f"p{slot}_ships_arriving_within_{h}")
    return cols


def _cross_entity_label_columns() -> list[str]:
    """Labels for the cross-entity pretrain dataset.

    Tier-1 (spatial, per-planet): frontier_class, n_friendly/enemy_within_R_norm,
    nearest_friendly/enemy_dist_norm, sector_advantage_log.

    Tier-3 (game-level, broadcast per snapshot): winner_seat,
    score_advantage_at_end_log, turns_until_episode_end, plus
    short-horizon global value labels.

    Tier-4 (tactical, per-planet): local same-owner support / hostile
    pressure labels inside max-speed reachability horizons.
    """
    cols: list[str] = []
    # Tier-1 spatial
    cols.append("frontier_class")
    cols.append("n_friendly_within_R_norm")
    cols.append("n_enemy_within_R_norm")
    cols.append("nearest_friendly_dist_norm")
    cols.append("nearest_enemy_dist_norm")
    cols.append("sector_advantage_log")
    # Tier-3 game-level
    cols.append("winner_seat")
    cols.append("score_advantage_at_end_log")
    cols.append("turns_until_episode_end")
    for h in CROSS_ENTITY_VALUE_HORIZONS:
        cols.append(f"leader_seat_t_plus_{h}")
        cols.append(f"score_advantage_t_plus_{h}_log")
        cols.append(f"is_ahead_t_plus_{h}")
        cols.append(f"valid_global_t_plus_{h}")
    # Tier-3b neutral game-level (player-perspective-independent)
    cols.append("total_ships_in_play_log")
    cols.append("ship_distribution_entropy")
    cols.append("n_neutral_planets")
    cols.append("game_phase")
    for h in CROSS_ENTITY_TACTICAL_HORIZONS:
        cols.append(f"can_friendly_reinforce_within_{h}")
        cols.append(f"enemy_can_capture_within_{h}")
        cols.append(f"best_local_support_margin_within_{h}_log")
    return cols


ENTITY_FEATURE_COLS: tuple[str, ...] = tuple(_entity_feature_columns())
ENTITY_LABEL_COLS: tuple[str, ...] = tuple(_entity_label_columns())
CROSS_ENTITY_LABEL_COLS: tuple[str, ...] = tuple(_cross_entity_label_columns())


# ---------- Lookahead helpers ----------
def _planet_state_at(steps: list, t: int, target_pid: int) -> tuple[int, int] | None:
    """Return ``(owner, ships)`` for ``target_pid`` at step ``t``, or
    ``None`` if the entity isn't in the planets list at that step
    (planets persist; this almost never returns None)."""
    if t >= len(steps) or not steps[t]:
        return None
    obs = steps[t][0].get("observation") if isinstance(steps[t][0], dict) else None
    if not obs:
        return None
    for p in obs.get("planets") or []:
        if int(p[P_ID]) == target_pid:
            return int(p[P_OWNER]), int(p[P_SHIPS])
    return None


def _step_obs(
    steps: list,
    t: int,
    *,
    learner_slot: int,
) -> dict[str, Any] | None:
    """Observation for ``learner_slot`` at step ``t``, or ``None``."""
    if t >= len(steps) or not steps[t]:
        return None
    seat = steps[t][learner_slot] if len(steps[t]) > learner_slot else steps[t][0]
    return seat.get("observation") if isinstance(seat, dict) else None


def _snapshot_totals(obs: dict[str, Any] | None, *, num_players: int) -> list[int]:
    """Total ships per player from one observation."""
    totals = [0] * num_players
    if not obs:
        return totals
    for p in obs.get("planets") or []:
        owner = int(p[P_OWNER])
        if 0 <= owner < num_players:
            totals[owner] += int(p[P_SHIPS])
    for f in obs.get("fleets") or []:
        owner = int(f[F_OWNER])
        if 0 <= owner < num_players:
            totals[owner] += int(f[F_SHIPS])
    return totals


# ---------- Player-perspective-independent (neutral) global labels ----------
# These quantities are computed without reference to ``learner_slot`` so the
# CLS token can learn a perspective-invariant game-state summary alongside
# the learner-relative labels.
GAME_PHASE_N_CLASSES: int = 3   # early / mid / late


def _compute_neutral_global_labels(
    obs: dict[str, Any] | None,
    *,
    num_players: int,
    turn: int,
) -> dict[str, float]:
    """Per-snapshot, perspective-invariant summary of the board.

    Quantities here are the same regardless of which seat is the
    ``learner_slot``: total ships in play, how concentrated those ships
    are across players, free-real-estate count, and a coarse game-phase
    bucket. Use them as auxiliary supervision for the CLS token to keep
    the global representation from collapsing into "am I winning."

    All values are floats (game_phase encoded as float for CSV uniformity;
    consumer casts back to long for the categorical head).
    """
    if not obs:
        return {
            "total_ships_in_play_log": 0.0,
            "ship_distribution_entropy": 0.0,
            "n_neutral_planets": 0.0,
            "game_phase": 0.0,
        }
    planets = obs.get("planets") or []

    player_ships = [0] * num_players
    n_neutral = 0
    for p in planets:
        owner = int(p[P_OWNER])
        ships = int(p[P_SHIPS])
        if 0 <= owner < num_players:
            player_ships[owner] += ships
        else:
            n_neutral += 1
    # Add inflight fleets to the per-player ship totals.
    for f in obs.get("fleets") or []:
        owner = int(f[F_OWNER])
        if 0 <= owner < num_players:
            player_ships[owner] += int(f[F_SHIPS])

    total = sum(player_ships)
    total_log = math.log1p(total)

    # Shannon entropy over player ship distribution (nats). Uniform 4-way
    # split → log(4) ≈ 1.386; one player owns everything → 0.
    if total > 0:
        entropy = 0.0
        for s in player_ships:
            if s > 0:
                p = s / total
                entropy -= p * math.log(p)
    else:
        entropy = 0.0

    # Coarse game-phase bucket from current turn (independent of t_until_end
    # so it stays defined even on truncated episodes).
    progress = turn / max(1, EPISODE_STEPS - 1)
    if progress < 1.0 / 3.0:
        phase = 0
    elif progress < 2.0 / 3.0:
        phase = 1
    else:
        phase = 2

    return {
        "total_ships_in_play_log": float(total_log),
        "ship_distribution_entropy": float(entropy),
        "n_neutral_planets": float(n_neutral),
        "game_phase": float(phase),
    }


def _winner_class_from_totals(
    totals: list[int],
    *,
    learner_slot: int,
    num_players: int,
) -> int:
    """Relative winner class, with ties collapsing to neutral/unknown."""
    if not totals or max(totals) == 0:
        return NUM_OWNER_SLOTS
    sorted_totals = sorted(totals, reverse=True)
    if len(sorted_totals) >= 2 and sorted_totals[0] == sorted_totals[1]:
        return NUM_OWNER_SLOTS
    winner_seat = max(range(num_players), key=lambda i: totals[i])
    return (winner_seat - learner_slot) % num_players


def _score_advantage_from_totals(
    totals: list[int],
    *,
    learner_slot: int,
    num_players: int,
) -> float:
    learner_total = totals[learner_slot] if 0 <= learner_slot < num_players else 0
    others_max = max(
        (totals[i] for i in range(num_players) if i != learner_slot),
        default=0,
    )
    return signed_log1p(learner_total - others_max) / SHIPS_LOG_MAX


def _compute_future_global_labels(
    steps: list,
    t: int,
    *,
    learner_slot: int,
    num_players: int,
) -> dict[str, float]:
    """Short-horizon CLS targets at ``t+K`` with validity masks."""
    out: dict[str, float] = {}
    for h in CROSS_ENTITY_VALUE_HORIZONS:
        fut_obs = _step_obs(steps, t + h, learner_slot=learner_slot)
        valid = 1.0 if fut_obs else 0.0
        totals = _snapshot_totals(fut_obs, num_players=num_players)
        leader = (
            _winner_class_from_totals(
                totals,
                learner_slot=learner_slot,
                num_players=num_players,
            )
            if valid > 0.0 else NUM_OWNER_SLOTS
        )
        score_adv = (
            _score_advantage_from_totals(
                totals,
                learner_slot=learner_slot,
                num_players=num_players,
            )
            if valid > 0.0 else 0.0
        )
        learner_total = totals[learner_slot] if 0 <= learner_slot < num_players else 0
        others_max = max(
            (totals[i] for i in range(num_players) if i != learner_slot),
            default=0,
        )
        out[f"leader_seat_t_plus_{h}"] = int(leader)
        out[f"score_advantage_t_plus_{h}_log"] = float(score_adv)
        out[f"is_ahead_t_plus_{h}"] = float(valid > 0.0 and learner_total > others_max)
        out[f"valid_global_t_plus_{h}"] = float(valid)
    return out


def _newly_launched_fleet_ids(prev_obs, next_obs) -> set[int]:
    """IDs of fleets that exist in ``next_obs`` but not in ``prev_obs``
    — i.e., fleets created by actions taken between t and t+1.

    The env's monotonic ``next_fleet_id`` makes this clean: any fleet at
    t+1 with id ≥ ``prev_obs['next_fleet_id']`` was just launched.
    """
    if not prev_obs or not next_obs:
        return set()
    cutoff = int(prev_obs.get("next_fleet_id", 0) or 0)
    next_fleets = next_obs.get("fleets") or []
    return {int(f[F_ID]) for f in next_fleets if int(f[F_ID]) >= cutoff}


def _action_planet_flags(
    prev_obs,
    next_obs,
    *,
    learner_slot: int,
    num_players: int,
) -> tuple[set[int], set[int]]:
    """Return ``(source_pids, target_pids)`` — planets that the
    **expert (= the learner seat in this episode)** *launched from* or
    *aimed at* on this turn.

    For imitation learning we want "did the expert act here", not "did
    *any* player act here". Owner filter checks the new fleet's
    ``F_OWNER`` against ``learner_slot``; every other player's
    launches are intentionally dropped.

    ``source_pid`` comes straight from the new fleet's ``F_FROM``.
    ``target_pid`` is computed via :func:`_resolve_target` against the
    next-turn obs (where the new fleet is fully visible).
    """
    new_ids = _newly_launched_fleet_ids(prev_obs, next_obs)
    if not new_ids:
        return set(), set()
    sources: set[int] = set()
    targets: set[int] = set()
    next_planets = next_obs.get("planets") or []
    for f in next_obs.get("fleets") or []:
        if int(f[F_ID]) not in new_ids:
            continue
        # Expert-only filter: imitation labels need to mean "the
        # learner launched this", not "anyone launched this".
        if int(f[F_OWNER]) != learner_slot:
            continue
        sources.add(int(f[F_FROM]))
        target_pid, _eta, _planet = _resolve_target(f, next_planets)
        if target_pid is not None:
            targets.add(int(target_pid))
    return sources, targets


def _owner_to_class(owner_id: int, learner_slot: int, num_players: int) -> int:
    """Map raw env owner id → relative class index in ``[0, N_OWNER_CLASSES)``.

    Class ``NUM_OWNER_SLOTS`` (= last) is "neutral / none". Out-of-range
    owners (shouldn't happen in real env) collapse to neutral.
    """
    if owner_id == -1:
        return NUM_OWNER_SLOTS
    if 0 <= owner_id < num_players:
        return (owner_id - learner_slot) % num_players
    return NUM_OWNER_SLOTS


# ---------- Tier-1 spatial helpers ----------
def _compute_spatial_labels(
    planet: tuple,
    all_planets: list,
    *,
    learner_slot: int,
) -> dict[str, float]:
    """Per-planet spatial-context labels. Walks every other planet
    once (O(P) per planet, O(P²) per snapshot — fine for ≤64 planets)
    and accumulates the friendly / enemy aggregates within
    ``SECTOR_RADIUS``.

    Returns a flat dict matching the Tier-1 column names below.
    """
    pid = int(planet[P_ID])
    own_owner = int(planet[P_OWNER])
    px = float(planet[P_X])
    py = float(planet[P_Y])

    is_self = own_owner == learner_slot
    is_neutral = own_owner == -1

    n_friendly = 0
    n_enemy = 0
    ships_friendly = 0
    ships_enemy = 0
    nearest_friendly = float("inf")
    nearest_enemy = float("inf")

    for other in all_planets:
        if int(other[P_ID]) == pid:
            continue
        ox = float(other[P_X])
        oy = float(other[P_Y])
        d = math.hypot(ox - px, oy - py)
        if d > SECTOR_RADIUS:
            continue
        oowner = int(other[P_OWNER])
        oships = int(other[P_SHIPS])
        if oowner == learner_slot:
            n_friendly += 1
            ships_friendly += oships
            if d < nearest_friendly:
                nearest_friendly = d
        elif oowner != -1:
            n_enemy += 1
            ships_enemy += oships
            if d < nearest_enemy:
                nearest_enemy = d
        # Neutrals contribute to neither count — they're just terrain.

    # Frontier classification (learner-relative).
    if is_self:
        frontier = 1 if n_enemy > 0 else 0
    elif is_neutral:
        frontier = 3
    else:
        frontier = 2

    # Saturated "no neighbor" → 1.0 normalized distance (max-far marker).
    nearest_friendly_norm = (
        min(1.0, nearest_friendly / DIST_NORM)
        if nearest_friendly != float("inf") else 1.0
    )
    nearest_enemy_norm = (
        min(1.0, nearest_enemy / DIST_NORM)
        if nearest_enemy != float("inf") else 1.0
    )

    return {
        "frontier_class": frontier,
        "n_friendly_within_R_norm": min(1.0, n_friendly / COUNT_NORM),
        "n_enemy_within_R_norm": min(1.0, n_enemy / COUNT_NORM),
        "nearest_friendly_dist_norm": nearest_friendly_norm,
        "nearest_enemy_dist_norm": nearest_enemy_norm,
        "sector_advantage_log": signed_log1p(ships_friendly - ships_enemy)
                                 / SHIPS_LOG_MAX,
    }


# ---------- Tier-3 game-level helpers ----------
def _compute_episode_summary(
    steps: list,
    *,
    learner_slot: int,
    num_players: int,
) -> dict[str, float]:
    """Compute the per-episode constants used as broadcast labels:

      * ``winner_seat`` — relative class id of the winning player (or
        ``NUM_OWNER_SLOTS`` for draws / unfinished games).
      * ``score_advantage_at_end_log`` — signed-log of
        ``my_total_ships - max_other_total_ships`` at the final step.

    Returns a dict with both keys; same value for every snapshot in
    the episode (broadcast at row-write time).
    """
    last = len(steps) - 1
    while last >= 0 and not steps[last]:
        last -= 1
    if last < 0:
        return {
            "winner_seat": NUM_OWNER_SLOTS,
            "score_advantage_at_end_log": 0.0,
        }
    obs = _step_obs(steps, last, learner_slot=learner_slot)
    if not obs:
        return {
            "winner_seat": NUM_OWNER_SLOTS,
            "score_advantage_at_end_log": 0.0,
        }
    totals = _snapshot_totals(obs, num_players=num_players)
    winner_class = _winner_class_from_totals(
        totals,
        learner_slot=learner_slot,
        num_players=num_players,
    )
    advantage = _score_advantage_from_totals(
        totals,
        learner_slot=learner_slot,
        num_players=num_players,
    )

    return {
        "winner_seat": int(winner_class),
        "score_advantage_at_end_log": float(advantage),
    }


def _compute_tactical_labels(
    planet: tuple,
    all_planets: list,
) -> dict[str, float]:
    """Local support / threat labels within max-speed reachability horizons.

    The semantics are owner-relative:
      * friendly = same owner as ``planet`` (excluding the planet itself)
      * enemy    = any other non-neutral owner

    We deliberately keep these labels cheap and deterministic. Reachability
    uses the theoretical max fleet speed (`MAX_SPEED`) as a radius proxy.
    """
    pid = int(planet[P_ID])
    owner = int(planet[P_OWNER])
    px = float(planet[P_X])
    py = float(planet[P_Y])
    ships = int(planet[P_SHIPS])
    prod = int(planet[P_PRODUCTION])

    out: dict[str, float] = {}
    for h in CROSS_ENTITY_TACTICAL_HORIZONS:
        friendly_support = 0
        enemy_support = 0
        best_enemy_capture_margin = float("-inf")
        for other in all_planets:
            if int(other[P_ID]) == pid:
                continue
            d = math.hypot(float(other[P_X]) - px, float(other[P_Y]) - py)
            eta = d / MAX_SPEED
            if eta > h:
                continue
            other_owner = int(other[P_OWNER])
            other_ships = int(other[P_SHIPS])
            if owner != -1 and other_owner == owner:
                friendly_support = max(friendly_support, other_ships)
                continue
            if other_owner == -1 or other_owner == owner:
                continue
            enemy_support = max(enemy_support, other_ships)
            defender = ships if owner == -1 else ships + int(prod * eta)
            best_enemy_capture_margin = max(
                best_enemy_capture_margin,
                float(other_ships - defender),
            )

        out[f"can_friendly_reinforce_within_{h}"] = float(friendly_support > 0)
        out[f"enemy_can_capture_within_{h}"] = float(best_enemy_capture_margin > 0.0)
        out[f"best_local_support_margin_within_{h}_log"] = (
            signed_log1p(friendly_support - enemy_support) / SHIPS_LOG_MAX
        )
    return out


# ---------- CSV export ----------
def save_episode_entity_csv(
    episode_path: str | Path,
    out_path: str | Path | None = None,
    *,
    learner_slot: int | None = None,
    num_players: int | None = None,
    max_entities: int = 64,
) -> Path:
    """Featurize every (turn, planet) in one episode and write a CSV
    suitable for entity-encoder pretraining.

    Each row carries:

      * meta: ``episode_id, turn, planet_id, is_comet, owner_id``
      * features: ``ENTITY_FEATURE_COLS`` — the 4×7 per-(planet, player)
        bag-of-stats described in the module docstring (matches the
        encoder's internal stats)
      * labels: ``ENTITY_LABEL_COLS`` — per-planet & per-(planet, player)
        supervised targets:

          - ``earliest_arrival_owner_slot`` (categorical, 0..NUM_OWNER_SLOTS)
          - ``is_source_this_turn`` / ``is_target_this_turn`` (binary,
            from action labels — replaces P-way pair classification)
          - For each ``K ∈ {5, 10}``:
              * ``owner_t_plus_K`` (categorical, 0..NUM_OWNER_SLOTS)
              * ``log_ships_t_plus_K`` (regression, log-norm)
              * ``valid_t_plus_K`` (mask, 0 if t+K runs off the
                episode end)
          - For each player_slot × horizon:
              * ``p{k}_ships_arriving_within_{H}`` (regression,
                log-norm of inbound ships from player k whose
                ETA ≤ H)

    The pretrain pipeline reads this CSV alongside the existing planet
    and fleet CSVs, joins on ``(episode_id, turn, planet_id)`` /
    ``(episode_id, turn)``, runs **frozen** ``FleetEncoder`` and
    ``PlanetEncoder`` to produce embeddings, and trains the entity
    encoder + label heads on top.

    Output defaults to ``data/datasets/entity/``.
    """
    import csv
    import gzip
    import json

    episode_path = Path(episode_path)
    if out_path is None:
        out_path = ENTITY_DATASET_DIR / f"entity_{episode_path.stem.split('.')[0]}.csv"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(episode_path, "rb") as fh:
        replay = json.load(fh)
    steps = replay.get("steps") or []
    if num_players is None:
        num_players = len(steps[0]) if steps else 4
    if learner_slot is None:
        learner_slot = _infer_learner_slot_from_episode_path(episode_path)
    episode_id = replay.get("id") or episode_path.stem

    feature_cols = list(ENTITY_FEATURE_COLS)
    label_cols = list(ENTITY_LABEL_COLS)
    meta_cols = ["planet_id", "is_comet", "owner_id"]
    header = ["episode_id", "turn", *meta_cols, *feature_cols, *label_cols]

    rows_written = 0
    with out_path.open("w", newline="") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(header)
        for t, step in enumerate(steps):
            if not step:
                continue
            seat = step[learner_slot] if len(step) > learner_slot else step[0]
            obs = seat.get("observation") if isinstance(seat, dict) else None
            if not obs:
                continue

            recs = featurize_entity_inbound(
                obs,
                learner_slot=learner_slot,
                num_players=num_players,
                max_entities=max_entities,
            )

            # Action labels: scan the t→t+1 transition for new fleets.
            # Pull next_obs from the same seat as ``obs`` so any
            # seat-locality in the env never leaks across the
            # transition.
            next_obs = None
            if t + 1 < len(steps) and steps[t + 1]:
                next_step = steps[t + 1]
                next_seat = (
                    next_step[learner_slot]
                    if len(next_step) > learner_slot
                    else next_step[0]
                )
                next_obs = (
                    next_seat.get("observation")
                    if isinstance(next_seat, dict) else None
                )
            source_pids, target_pids = _action_planet_flags(
                obs, next_obs, learner_slot=learner_slot, num_players=num_players,
            )

            for rec in recs:
                row: list = [
                    episode_id,
                    t,
                    rec.planet_id,
                    int(rec.is_comet),
                    rec.owner_id,
                ]
                for cell in rec.per_player:
                    row.extend(cell)

                # ---- labels ----
                row.append(rec.earliest_arrival_owner_slot)
                row.append(int(rec.planet_id in source_pids))
                row.append(int(rec.planet_id in target_pids))

                # planet-level lookahead (owner & ships at t+K)
                for k in LABEL_HORIZONS:
                    fut = _planet_state_at(steps, t + k, rec.planet_id)
                    if fut is None:
                        valid_k = 0
                        owner_k_class = NUM_OWNER_SLOTS
                        log_ships_k = 0.0
                    else:
                        valid_k = 1
                        fut_owner, fut_ships = fut
                        owner_k_class = _owner_to_class(
                            fut_owner, learner_slot, num_players,
                        )
                        log_ships_k = math.log1p(max(0, fut_ships)) / SHIPS_LOG_MAX
                    row.append(owner_k_class)
                    row.append(log_ships_k)
                    row.append(valid_k)

                # per-(player_slot) × horizon arrivals
                for h in ARRIVAL_HORIZONS:
                    row.extend(rec.ships_arriving_within[h])

                writer.writerow(row)
                rows_written += 1

    print(f"[entity_featurizer] wrote {rows_written} rows → {out_path}")
    return out_path


# ---------- Cross-entity CSV export ----------
def save_episode_cross_entity_csv(
    episode_path: str | Path,
    out_path: str | Path | None = None,
    *,
    learner_slot: int | None = None,
    num_players: int | None = None,
    max_entities: int = 64,
) -> Path:
    """Write the cross-attention pretrain dataset for one episode.

    A thin label-only file keyed by ``(episode_id, turn, planet_id)``
    with cross-attention-specific labels:

      Tier-1 spatial (per-planet):
        * ``frontier_class``                         (categorical, 4)
        * ``n_friendly_within_R_norm``               (regression)
        * ``n_enemy_within_R_norm``                  (regression)
        * ``nearest_friendly_dist_norm``             (regression)
        * ``nearest_enemy_dist_norm``                (regression)
        * ``sector_advantage_log``                   (regression)

      Tier-3 game-level (broadcast across the whole episode / snapshot):
        * ``winner_seat``                            (categorical, K+1)
        * ``score_advantage_at_end_log``             (regression)
        * ``turns_until_episode_end``                (regression)
        * ``leader_seat_t_plus_{10,20,50}``          (categorical, K+1)
        * ``score_advantage_t_plus_{10,20,50}_log``  (regression)
        * ``is_ahead_t_plus_{10,20,50}``             (binary)

      Tier-4 tactical (per-planet):
        * ``can_friendly_reinforce_within_{5,10}``         (binary)
        * ``enemy_can_capture_within_{5,10}``              (binary)
        * ``best_local_support_margin_within_{5,10}_log``  (regression)

    Joins one-to-one with the entity CSV by
    ``(episode_id, turn, planet_id)`` so the cross-attention pretrain
    pipeline can pull both raw features (planet/fleet CSV), entity-
    encoder labels (entity CSV), and the new labels (this CSV)
    without duplicating either.

    Output defaults to ``data/datasets/cross_entity/``.
    """
    import csv
    import gzip
    import json

    from ..paths import CROSS_ENTITY_DATASET_DIR

    episode_path = Path(episode_path)
    if out_path is None:
        out_path = (
            CROSS_ENTITY_DATASET_DIR
            / f"cross_entity_{episode_path.stem.split('.')[0]}.csv"
        )
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(episode_path, "rb") as fh:
        replay = json.load(fh)
    steps = replay.get("steps") or []
    if num_players is None:
        num_players = len(steps[0]) if steps else 4
    if learner_slot is None:
        learner_slot = _infer_learner_slot_from_episode_path(episode_path)
    episode_id = replay.get("id") or episode_path.stem
    n_steps = len(steps)

    episode_summary = _compute_episode_summary(
        steps, learner_slot=learner_slot, num_players=num_players,
    )

    label_cols = list(CROSS_ENTITY_LABEL_COLS)
    header = ["episode_id", "turn", "planet_id", *label_cols]

    rows_written = 0
    with out_path.open("w", newline="") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(header)
        for t, step in enumerate(steps):
            if not step:
                continue
            seat = step[learner_slot] if len(step) > learner_slot else step[0]
            obs = seat.get("observation") if isinstance(seat, dict) else None
            if not obs:
                continue
            raw_planets = obs.get("planets") or []

            # Same per-turn scalar for every planet in this snapshot.
            turns_until_end = max(
                0.0, min(1.0, (n_steps - 1 - t) / EPISODE_STEPS),
            )
            future_global = _compute_future_global_labels(
                steps,
                t,
                learner_slot=learner_slot,
                num_players=num_players,
            )
            neutral_global = _compute_neutral_global_labels(
                obs, num_players=num_players, turn=t,
            )

            # Iterate the planet list in env order, capped at
            # max_entities to mirror the entity CSV's truncation.
            for p in raw_planets[:max_entities]:
                pid = int(p[P_ID])
                spatial = _compute_spatial_labels(
                    p, raw_planets, learner_slot=learner_slot,
                )
                tactical = _compute_tactical_labels(p, raw_planets)
                writer.writerow([
                    episode_id,
                    t,
                    pid,
                    spatial["frontier_class"],
                    spatial["n_friendly_within_R_norm"],
                    spatial["n_enemy_within_R_norm"],
                    spatial["nearest_friendly_dist_norm"],
                    spatial["nearest_enemy_dist_norm"],
                    spatial["sector_advantage_log"],
                    episode_summary["winner_seat"],
                    episode_summary["score_advantage_at_end_log"],
                    turns_until_end,
                    *[
                        future_global[name]
                        for h in CROSS_ENTITY_VALUE_HORIZONS
                        for name in (
                            f"leader_seat_t_plus_{h}",
                            f"score_advantage_t_plus_{h}_log",
                            f"is_ahead_t_plus_{h}",
                            f"valid_global_t_plus_{h}",
                        )
                    ],
                    neutral_global["total_ships_in_play_log"],
                    neutral_global["ship_distribution_entropy"],
                    neutral_global["n_neutral_planets"],
                    neutral_global["game_phase"],
                    *[
                        tactical[name]
                        for h in CROSS_ENTITY_TACTICAL_HORIZONS
                        for name in (
                            f"can_friendly_reinforce_within_{h}",
                            f"enemy_can_capture_within_{h}",
                            f"best_local_support_margin_within_{h}_log",
                        )
                    ],
                ])
                rows_written += 1

    print(f"[cross_entity_featurizer] wrote {rows_written} rows → {out_path}")
    return out_path
