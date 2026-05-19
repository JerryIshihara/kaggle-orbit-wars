"""Physical static v1 — only shoots STATIC (outer-ring) planets.

One of three single-motion-class siblings (``physical_static_v1`` /
``physical_orbit_v1`` / ``physical_comet_v1``). Each gates every launch
through the matching public shooter (``shoot_static`` / ``shoot_orbit``
/ ``shoot_comet``) so source-to-target trajectories are exact.

Per turn, for each owned source planet:
1. Compute safe surplus = ships − reserve_against_incoming_enemies.
2. Build candidate targets that are *static* (outside the orbital ring
   or with zero ``angular_velocity``) and not already ours.
3. Rank candidates by a (production, cost, distance) score.
4. Iterate top-down: size a fleet to ``ships_needed = ships + buffer``,
   capped at surplus; call :func:`shoot_static`; if it returns
   ``ok=True`` commit ``[src, angle, ships]`` and debit surplus.

Every committed move is appended to ``LAUNCH_LOG`` in the same shape
``measure_physical_v4_miss_rate.py`` reads, so miss-rate verification
works out of the box.
"""

from __future__ import annotations

import math

from ...physics_utils import (
    F_OWNER,
    F_SHIPS,
    P_ID,
    P_OWNER,
    P_PRODUCTION,
    P_RADIUS,
    P_SHIPS,
    P_X,
    P_Y,
    _fleet_eta_to_planet,
    _is_orbiting_xy,
    find_collision,
    predict_garrison_at_arrival,
    shoot_static,
)
from ...registry import register


# ---------- Strategy parameters ----------
SAFETY_FLOOR = 3
SAFETY_FACTOR = 1.0
NEUTRAL_PRICE_FACTOR = 0.6
ENEMY_PRICE_FACTOR = 1.0
PRODUCTION_WEIGHT = 8.0
DISTANCE_WEIGHT = 0.4
SAFETY_BUFFER = 3
MIN_LAUNCH = 5
DANGER_HORIZON = 60


# ---------- Tracking (for miss-rate post-hoc analysis) ----------
LAUNCH_LOG: list[tuple[int, int, int, int, float, float]] = []
# (src_pid, tgt_pid) pairs already shot this game. A pair is one-shot:
# once source S has launched at target T, S never launches at T again
# even if T flips owner. Reset on step==0.
_LAUNCHED_PAIRS: set[tuple[int, int]] = set()
_LAST_STEP_SEEN = {"step": -1}


def _incoming_hostile(planet: tuple, fleets: list, player: int) -> int:
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
    danger = _incoming_hostile(source, fleets, player)
    reserve = max(SAFETY_FLOOR, int(math.ceil(danger * SAFETY_FACTOR)))
    return max(0, int(source[P_SHIPS]) - reserve)


def _is_static(target: tuple, angular_velocity: float, comet_ids: set[int]) -> bool:
    if int(target[P_ID]) in comet_ids:
        return False
    return not _is_orbiting_xy(
        float(target[P_X]),
        float(target[P_Y]),
        float(target[P_RADIUS]),
        angular_velocity,
    )


def _score(target: tuple, source: tuple, *, neutral: bool) -> float:
    dist = math.hypot(
        float(target[P_X]) - float(source[P_X]),
        float(target[P_Y]) - float(source[P_Y]),
    )
    price = NEUTRAL_PRICE_FACTOR if neutral else ENEMY_PRICE_FACTOR
    return (
        PRODUCTION_WEIGHT * float(target[P_PRODUCTION])
        - price * float(target[P_SHIPS])
        - DISTANCE_WEIGHT * dist
    )


@register(
    "physical_static_v1",
    "Heuristic agent that only shoots STATIC targets via shoot_static. "
    "Source-to-target trajectories are exact by construction.",
)
def physical_static_v1(obs):
    moves: list[list] = []

    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    player = int(get("player", 0) or 0)
    raw_planets = list(get("planets") or [])
    raw_fleets = list(get("fleets") or [])
    angular_velocity = abs(float(get("angular_velocity") or 0.0))
    comet_ids = set(get("comet_planet_ids") or [])
    step = int(get("step", 0) or 0)

    if step == 0 and _LAST_STEP_SEEN.get("step", -1) != 0:
        LAUNCH_LOG.clear()
        _LAUNCHED_PAIRS.clear()
    _LAST_STEP_SEEN["step"] = step

    my_planets = [p for p in raw_planets if int(p[P_OWNER]) == player]
    if not my_planets:
        return moves

    targets = [
        p for p in raw_planets
        if int(p[P_OWNER]) != player
        and int(p[P_ID]) not in {int(s[P_ID]) for s in my_planets}
        and _is_static(p, angular_velocity, comet_ids)
    ]
    if not targets:
        return moves

    surplus_by_pid: dict[int, int] = {
        int(s[P_ID]): _safe_surplus(s, raw_fleets, player) for s in my_planets
    }

    for source in my_planets:
        sid = int(source[P_ID])
        surplus = surplus_by_pid[sid]
        if surplus < MIN_LAUNCH:
            continue

        scored: list[tuple[float, tuple]] = []
        for tgt in targets:
            scored.append((
                _score(tgt, source, neutral=int(tgt[P_OWNER]) == -1),
                tgt,
            ))
        scored.sort(reverse=True, key=lambda kv: kv[0])

        for _, tgt in scored:
            tid = int(tgt[P_ID])
            if (sid, tid) in _LAUNCHED_PAIRS:
                continue
            ships_needed = max(MIN_LAUNCH, int(tgt[P_SHIPS]) + SAFETY_BUFFER)
            ships_to_send = min(ships_needed, surplus)
            if ships_to_send < MIN_LAUNCH:
                break
            angle, eta, ships = shoot_static(source, tgt, ships_to_send)
            # Strategy: trajectory must clear sun + other planets and
            # hit exactly the intended target.
            hit = find_collision(source, angle, ships, obs)
            if hit is None or hit["kind"] != "planet":
                continue
            if int(hit["planet"][P_ID]) != tid:
                continue
            # Strategy: fleet must beat predicted defenders at arrival.
            garrison = predict_garrison_at_arrival(tgt, eta, player, raw_fleets)
            if ships <= garrison:
                continue
            moves.append([sid, float(angle), int(ships)])
            surplus -= int(ships)
            _LAUNCHED_PAIRS.add((sid, tid))
            LAUNCH_LOG.append(
                (step, sid, tid, int(ships), float(angle), float(eta))
            )
            if surplus < MIN_LAUNCH:
                break
        surplus_by_pid[sid] = surplus

    return moves
