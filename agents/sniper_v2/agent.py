"""Sniper v2 — every launch is gated by ``physics_utils.sniper``.

Strategy
--------

Per turn, for each owned planet ``s``:

1. Compute a *safe surplus* — the ships ``s`` can spend without leaving
   itself defenseless against incoming hostile fleets.
2. Score every other planet from ``s``'s perspective via
   :func:`evaluate_planet`. The score blends production gain, capture
   cost, distance, neutral discount, comet penalty.
3. Iterate candidates by descending score; for each one, size a fleet
   to the predicted garrison + safety buffer and call
   :func:`physics_utils.sniper` to validate. ``sniper`` returns
   ``(ships, angle, eta)``; if ``ships > 0`` the launch is sound by
   construction (trajectory hits target, garrison can be beaten, comet
   path long enough, game won't end first).
4. Commit the launch, debit the surplus, continue with the next
   candidate until surplus is exhausted or no targets remain.

Because every move passes ``sniper``, the agent's expected miss rate
(launches that don't reach their intended target) is **zero by
design** — the trajectory check inside ``sniper`` is the same physics
the env later resolves the fleet against.

A module-level ``LAUNCH_LOG`` mirrors every committed move with its
``(turn, src_pid, intended_target_pid, ships, angle, eta)`` so an
external runner can post-hoc verify the trajectory landed on the
intended target.
"""

from __future__ import annotations

import math

from ..physics_utils import (
    F_OWNER,
    F_SHIPS,
    P_ID,
    P_OWNER,
    P_PRODUCTION,
    P_RADIUS,
    P_SHIPS,
    P_X,
    P_Y,
    SUN_CX,
    SUN_CY,
    _fleet_eta_to_planet,
    _is_orbiting_xy,
    sniper,
)
from ..registry import register


# ---------- Strategy parameters ----------
SAFETY_FLOOR = 3                 # absolute minimum garrison to keep at source
SAFETY_FACTOR = 1.0              # multiplier on incoming hostile ships
NEUTRAL_PRICE_FACTOR = 0.6       # neutrals look cheaper than enemies
ENEMY_PRICE_FACTOR = 1.0
PRODUCTION_WEIGHT = 8.0
DISTANCE_WEIGHT = 0.4
SAFETY_BUFFER = 3                # extra ships beyond predicted garrison
MIN_LAUNCH = 5                   # don't launch tiny dribbles
DANGER_HORIZON = 60              # ignore enemy fleets > this far away (turns)

# Hard priority by entity type. Comet bonus is large enough to dominate
# any score gap between comets and orbital planets, ensuring sniper_v2
# clears comets before orbital targets. Static planets are excluded
# (returned as -inf), so they're never shot — even if they're the only
# targets left.
ENTITY_BONUS_COMET = 1_000_000.0
ENTITY_BONUS_ORBITAL = 1_000.0
SHOOT_STATIC = False             # skip static (outer-ring) planets


# ---------- Tracking (for missing-rate post-hoc analysis) ----------
# Each entry: (turn, src_pid, intended_target_pid, ships, angle, eta).
# Cleared at step 0 on the first call from any seat; the runner can
# read this between matches.
LAUNCH_LOG: list[tuple[int, int, int, int, float, float]] = []
_LAST_STEP_SEEN = {"step": -1}


# ---------- Helpers ----------
def _incoming_hostile(planet: tuple, fleets: list, player: int) -> int:
    """Sum of enemy ship counts inbound to ``planet`` within
    ``DANGER_HORIZON`` turns. Used to decide how many ships must stay
    home as defensive reserve.
    """
    total = 0
    for f in fleets:
        if int(f[F_OWNER]) == player:
            continue
        eta = _fleet_eta_to_planet(f, planet)
        if eta is None or eta > DANGER_HORIZON:
            continue
        total += int(f[F_SHIPS])
    return total


def _safe_surplus(source: tuple, fleets: list, player: int) -> int:
    """Ships available to ``source`` after holding back enough to
    survive predicted incoming attacks."""
    danger = _incoming_hostile(source, fleets, player)
    reserve = max(SAFETY_FLOOR, int(math.ceil(danger * SAFETY_FACTOR)))
    return max(0, int(source[P_SHIPS]) - reserve)


# ---------- Scoring ----------
def evaluate_planet(
    target: tuple,
    source: tuple,
    obs,
    *,
    player: int,
) -> float:
    """Score the desirability of attacking/capturing ``target`` from
    ``source`` *this turn*. Higher = better. ``-inf`` to reject.

    Hard priority by entity type:

      1. **Comets** — always preferred over any other target. Their
         finite lifespan means we want to grab their production windfall
         while it's available.
      2. **Orbital planets** — second priority, ranked by the usual
         production / cost / distance score.
      3. **Static planets** — *never shot* (``SHOOT_STATIC=False``).
         Static planets sit in the outer ring with high garrison; they
         tend to be expensive and slow to flip, so this version of the
         sniper deliberately ignores them.

    Skipped (returns ``-inf``):
      * self-target
      * already friendly (no reinforcement; defensive play is upstream)
      * static, unless ``SHOOT_STATIC`` is enabled
    """
    if target[P_ID] == source[P_ID]:
        return float("-inf")
    if int(target[P_OWNER]) == player:
        return float("-inf")

    angular_velocity = float(
        (obs.get("angular_velocity") if isinstance(obs, dict)
         else getattr(obs, "angular_velocity", 0.0)) or 0.0
    )
    comet_ids = (
        obs.get("comet_planet_ids") if isinstance(obs, dict)
        else getattr(obs, "comet_planet_ids", None)
    ) or []
    is_comet = int(target[P_ID]) in set(comet_ids)
    is_orbital = (
        not is_comet
        and _is_orbiting_xy(
            float(target[P_X]), float(target[P_Y]),
            float(target[P_RADIUS]), angular_velocity,
        )
    )
    if not is_comet and not is_orbital and not SHOOT_STATIC:
        return float("-inf")

    is_neutral = int(target[P_OWNER]) == -1
    dist = math.hypot(
        float(target[P_X]) - float(source[P_X]),
        float(target[P_Y]) - float(source[P_Y]),
    )
    price = NEUTRAL_PRICE_FACTOR if is_neutral else ENEMY_PRICE_FACTOR

    score = (
        PRODUCTION_WEIGHT * float(target[P_PRODUCTION])
        - price * float(target[P_SHIPS])
        - DISTANCE_WEIGHT * dist
    )

    # Type-priority bonuses. Comet bonus dwarfs orbital so that any
    # comet beats any orbital planet regardless of relative production
    # / cost / distance, matching the "shoot comets first" rule.
    if is_comet:
        score += ENTITY_BONUS_COMET
    elif is_orbital:
        score += ENTITY_BONUS_ORBITAL
    return score


# ---------- Agent ----------
@register(
    "sniper_v2",
    "Sniper agent that gates every launch through physics_utils.sniper. "
    "Per source planet, scores targets via evaluate_planet, holds back "
    "ships to cover incoming hostile fleets, and only commits validated "
    "launches — designed for zero miss rate by construction.",
)
def sniper_v2(obs):
    moves: list[list] = []

    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    player = int(get("player", 0) or 0)
    raw_planets = list(get("planets") or [])
    raw_fleets = list(get("fleets") or [])
    step = int(get("step", 0) or 0)

    # Reset launch log when a new game starts. Detect via step==0 on the
    # first call we see at this step (multi-seat games would otherwise
    # clear after each seat's first call).
    if step == 0 and _LAST_STEP_SEEN.get("step", -1) != 0:
        LAUNCH_LOG.clear()
    _LAST_STEP_SEEN["step"] = step

    my_planets = [p for p in raw_planets if int(p[P_OWNER]) == player]
    if not my_planets:
        return moves

    # Pre-compute safe surplus per source so the per-target loop below
    # can debit it as launches commit.
    surplus_by_pid: dict[int, int] = {
        int(s[P_ID]): _safe_surplus(s, raw_fleets, player)
        for s in my_planets
    }

    for source in my_planets:
        sid = int(source[P_ID])
        surplus = surplus_by_pid[sid]
        if surplus < MIN_LAUNCH:
            continue

        # Score every candidate from this source's perspective.
        scored: list[tuple[float, tuple]] = []
        for tgt in raw_planets:
            s = evaluate_planet(tgt, source, obs, player=player)
            if s == float("-inf"):
                continue
            scored.append((s, tgt))
        scored.sort(reverse=True, key=lambda kv: kv[0])

        for _score, tgt in scored:
            # Size against current garrison + a small safety buffer.
            # ``sniper`` will reject if the predicted garrison-at-
            # arrival is higher (production growth + sibling fleets).
            ships_needed = max(MIN_LAUNCH, int(tgt[P_SHIPS]) + SAFETY_BUFFER)
            ships_to_send = min(ships_needed, surplus)
            if ships_to_send < MIN_LAUNCH:
                break

            ships, angle, eta = sniper(
                source, tgt, ships_to_send, obs,
                player=player, safety_buffer=SAFETY_BUFFER,
            )
            if ships <= 0:
                continue

            # Commit and debit.
            moves.append([sid, float(angle), int(ships)])
            surplus -= ships
            LAUNCH_LOG.append(
                (step, sid, int(tgt[P_ID]), int(ships), float(angle), float(eta))
            )
            if surplus < MIN_LAUNCH:
                break

        surplus_by_pid[sid] = surplus

    return moves
