"""Physical comet v1 — only shoots COMET planets.

Sibling of ``physical_static_v1`` / ``physical_orbit_v1``. Every launch
is gated by :func:`shoot_comet`, which lead-aims along the comet's
pre-computed path and rejects launches whose path would expire before
arrival. Source-to-target trajectories are exact.

Coverage strategy: iterate per comet (most-urgent first) and pick the
closest source that can validate a launch. Each comet gets at most one
fleet committed; once covered, the agent moves to the next comet.
This guarantees every reachable comet gets attacked rather than letting
a single source fixate on its highest-scoring pick.

Comet urgency = remaining path length (smallest first). A comet whose
path expires in 8 turns is shot before a longer-lived one even if the
longer one is bigger, because the dying comet is about to vanish.
"""

from __future__ import annotations

import math

from ...physics_utils import (
    F_OWNER,
    F_SHIPS,
    P_ID,
    P_OWNER,
    P_PRODUCTION,
    P_SHIPS,
    _build_comet_lookup,
    _fleet_eta_to_planet,
    find_collision,
    predict_garrison_at_arrival,
    shoot_comet,
)
from ...registry import register


SAFETY_FLOOR = 3
SAFETY_FACTOR = 1.0
SAFETY_BUFFER = 3
MIN_LAUNCH = 5
DANGER_HORIZON = 60


LAUNCH_LOG: list[tuple[int, int, int, int, float, float]] = []
# One-shot per (src_pid, tgt_pid). Comets are short-lived so this rule
# rarely matters within a single comet's lifespan, but it's enforced for
# parity with the static / orbit siblings. Reset on step==0.
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


def _comet_remaining_steps(
    pid: int,
    comet_lookup: dict[int, tuple[list, int]],
) -> int:
    entry = comet_lookup.get(pid)
    if entry is None:
        return 0
    path, path_index = entry
    return max(0, len(path) - int(path_index))


def _friendly_inbound_ships(target: tuple, fleets: list, player: int) -> int:
    """Total of our own ships already inbound to ``target``. Used to skip
    comets we've already committed a fleet to, so we don't double-send
    on subsequent turns while the original fleet is still in flight."""
    total = 0
    for f in fleets:
        if int(f[F_OWNER]) != player:
            continue
        if _fleet_eta_to_planet(f, target) is None:
            continue
        total += int(f[F_SHIPS])
    return total


@register(
    "physical_comet_v1",
    "Heuristic agent that only shoots COMET targets via shoot_comet. "
    "Iterates per comet (urgent-first) and assigns each to its closest "
    "valid source — every reachable comet gets exactly one fleet.",
)
def physical_comet_v1(obs):
    moves: list[list] = []

    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    player = int(get("player", 0) or 0)
    raw_planets = list(get("planets") or [])
    raw_fleets = list(get("fleets") or [])
    comet_ids = set(get("comet_planet_ids") or [])
    comets_meta = list(get("comets") or [])
    step = int(get("step", 0) or 0)

    if step == 0 and _LAST_STEP_SEEN.get("step", -1) != 0:
        LAUNCH_LOG.clear()
        _LAUNCHED_PAIRS.clear()
    _LAST_STEP_SEEN["step"] = step

    if not comet_ids:
        return moves

    my_planets = [p for p in raw_planets if int(p[P_OWNER]) == player]
    if not my_planets:
        return moves

    targets = [
        p for p in raw_planets
        if int(p[P_OWNER]) != player
        and int(p[P_ID]) in comet_ids
    ]
    if not targets:
        return moves

    comet_lookup = _build_comet_lookup(comets_meta)

    # Urgency: shortest-remaining-path first. Production breaks ties so a
    # high-yield comet beats a tiny one with the same lifespan left.
    targets.sort(
        key=lambda t: (
            _comet_remaining_steps(int(t[P_ID]), comet_lookup),
            -float(t[P_PRODUCTION]),
        )
    )

    surplus_by_pid: dict[int, int] = {
        int(s[P_ID]): _safe_surplus(s, raw_fleets, player) for s in my_planets
    }

    for tgt in targets:
        ships_needed = max(MIN_LAUNCH, int(tgt[P_SHIPS]) + SAFETY_BUFFER)
        # Skip if our existing inbound fleets already overpower the
        # comet's defenders — sending more would be ship-burn.
        if _friendly_inbound_ships(tgt, raw_fleets, player) >= ships_needed:
            continue
        tid = int(tgt[P_ID])

        # Earliest-arrival policy: aim from every eligible source via
        # shoot_comet, run the agent's strategy gates (sun / wrong
        # planet / garrison), and commit the source whose ETA is
        # smallest among the survivors.
        best: tuple[float, int, float, int] | None = None
        for source in my_planets:
            sid = int(source[P_ID])
            if (sid, tid) in _LAUNCHED_PAIRS:
                continue
            surplus = surplus_by_pid[sid]
            if surplus < MIN_LAUNCH:
                continue
            ships_to_send = min(ships_needed, surplus)
            if ships_to_send < MIN_LAUNCH:
                continue
            angle, eta, ships = shoot_comet(source, tgt, ships_to_send, obs)
            hit = find_collision(source, angle, ships, obs)
            if hit is None or hit["kind"] != "planet":
                continue
            if int(hit["planet"][P_ID]) != tid:
                continue
            garrison = predict_garrison_at_arrival(tgt, eta, player, raw_fleets)
            if ships <= garrison:
                continue
            if best is None or eta < best[2]:
                best = (sid, int(ships), eta, float(angle))

        if best is None:
            continue
        sid, ships, eta, angle = best[0], best[1], best[2], best[3]
        moves.append([sid, float(angle), int(ships)])
        surplus_by_pid[sid] = surplus_by_pid[sid] - int(ships)
        _LAUNCHED_PAIRS.add((sid, tid))
        LAUNCH_LOG.append(
            (step, sid, tid, int(ships), float(angle), float(eta))
        )

    return moves
