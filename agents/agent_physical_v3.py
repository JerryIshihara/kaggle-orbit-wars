"""Heuristic physical agent v3 — v2 + multi-target launches per source per turn.

Single improvement over v2: instead of picking the single best target from a
source planet and emitting one launch, v3 collects ALL viable candidates,
sorts them by score ascending, then greedily allocates the defensive surplus
budget across multiple launches until the remaining surplus falls below
MIN_LAUNCH_SHIPS.  Duplicate targets from the same source in the same turn
are skipped.

Everything else (lead-aim, sun-dodge, rotation sign inference,
travel_time/production scoring, production-during-travel allocation,
fleet_eta_to_planet, compute_surplus) is copied verbatim from v2 so the
file stays self-contained for packing.
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from .registry import register

SUN_CX = 50.0
SUN_CY = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0
MAX_SPEED = 6.0
SPEED_LOG_DENOM = math.log(1000.0)
ROTATION_RADIUS_LIMIT = 50.0
LEAD_AIM_ITERS = 6
SAFETY_BUFFER = 3
MIN_LAUNCH_SHIPS = 5
NEUTRAL_BONUS = 0.8
DEFENSE_BUFFER = 3


def fleet_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / SPEED_LOG_DENOM) ** 1.5


def _dist_from_sun(x: float, y: float) -> float:
    return math.hypot(x - SUN_CX, y - SUN_CY)


def is_orbiting(p: Planet, angular_velocity: float) -> bool:
    if angular_velocity == 0:
        return False
    return (_dist_from_sun(p.x, p.y) + p.radius) < ROTATION_RADIUS_LIMIT


def crosses_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return _dist_from_sun(x1, y1) <= SUN_RADIUS + SUN_MARGIN
    t = max(0.0, min(1.0, ((SUN_CX - x1) * dx + (SUN_CY - y1) * dy) / len_sq))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(cx - SUN_CX, cy - SUN_CY) <= SUN_RADIUS + SUN_MARGIN


def infer_rotation_sign(planets: list[Planet], initial_planets: list) -> int:
    init = {row[0]: row for row in initial_planets}
    for p in planets:
        if p.id not in init:
            continue
        ip = init[p.id]
        ix, iy = ip[2], ip[3]
        ir = math.hypot(ix - SUN_CX, iy - SUN_CY)
        cr = _dist_from_sun(p.x, p.y)
        if abs(ir - cr) > 0.5:
            continue
        ia = math.atan2(iy - SUN_CY, ix - SUN_CX)
        ca = math.atan2(p.y - SUN_CY, p.x - SUN_CX)
        delta = (ca - ia + math.pi) % (2 * math.pi) - math.pi
        if abs(delta) > 1e-3:
            return 1 if delta > 0 else -1
    return 1


def predict_position(p: Planet, av_signed: float, turns: float) -> tuple[float, float]:
    dx, dy = p.x - SUN_CX, p.y - SUN_CY
    r = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) + av_signed * turns
    return SUN_CX + r * math.cos(angle), SUN_CY + r * math.sin(angle)


def lead_aim(
    source: Planet,
    target: Planet,
    fleet_ships: int,
    av_signed: float,
    orbiting: bool,
) -> tuple[float, float, float]:
    speed = fleet_speed(fleet_ships)
    if not orbiting:
        dist = math.hypot(target.x - source.x, target.y - source.y)
        return target.x, target.y, dist / speed
    px, py = target.x, target.y
    turns = math.hypot(px - source.x, py - source.y) / speed
    for _ in range(LEAD_AIM_ITERS):
        px, py = predict_position(target, av_signed, turns)
        turns = math.hypot(px - source.x, py - source.y) / speed
    return px, py, turns


def fleet_eta_to_planet(fleet: Fleet, planet: Planet) -> float | None:
    """Time until ``fleet``'s trajectory comes within ``planet.radius``.

    Returns None if the fleet misses the planet or is moving away.
    Treats the planet as static — good enough for a v2 threat heuristic;
    orbiting planets that move away can only over-count threat, which
    the DEFENSE_BUFFER already absorbs.
    """
    speed = fleet_speed(fleet.ships)
    if speed <= 0:
        return None
    ch = math.cos(fleet.angle)
    sh = math.sin(fleet.angle)
    dx = planet.x - fleet.x
    dy = planet.y - fleet.y
    t_closest = (dx * ch + dy * sh) / speed
    if t_closest < 0:
        return None
    cx = fleet.x + speed * t_closest * ch
    cy = fleet.y + speed * t_closest * sh
    if math.hypot(cx - planet.x, cy - planet.y) > planet.radius:
        return None
    return t_closest


def compute_surplus(source: Planet, enemy_fleets: list[Fleet]) -> int:
    """Min garrison over the threat timeline, minus DEFENSE_BUFFER."""
    threats = []
    for f in enemy_fleets:
        eta = fleet_eta_to_planet(f, source)
        if eta is not None:
            threats.append((eta, f.ships))
    if not threats:
        return max(0, source.ships - DEFENSE_BUFFER)
    threats.sort(key=lambda x: x[0])
    garrison = float(source.ships)
    min_garrison = garrison
    last_t = 0.0
    for t, ships in threats:
        garrison += source.production * (t - last_t)
        garrison -= ships
        if garrison < min_garrison:
            min_garrison = garrison
        last_t = t
    return int(max(0, math.floor(min_garrison) - DEFENSE_BUFFER))


def _score(turns: float, target: Planet) -> float:
    base = turns / max(1, target.production)
    return base * (NEUTRAL_BONUS if target.owner == -1 else 1.0)


@register(
    "physical_v3",
    "physical_v2 + multi-target per source per turn. Greedy allocation "
    "under the defensive surplus budget — sorted by score ascending.",
)
def physical_v3_agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    raw_planets = get("planets") or []
    raw_fleets = get("fleets") or []
    angular_velocity = abs(float(get("angular_velocity") or 0.0))
    initial_planets = get("initial_planets") or []

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    av_sign = infer_rotation_sign(planets, initial_planets)
    av_signed = angular_velocity * av_sign

    my_planets = [p for p in planets if p.owner == player and p.ships >= MIN_LAUNCH_SHIPS]
    targets = [p for p in planets if p.owner != player]
    enemy_fleets = [f for f in fleets if f.owner != player and f.owner >= 0]
    if not my_planets or not targets:
        return []

    moves = []
    for source in my_planets:
        surplus = compute_surplus(source, enemy_fleets)
        if surplus < MIN_LAUNCH_SHIPS:
            continue

        # Collect all viable candidates for this source.
        candidates = []  # (score, target_id, angle, ships_needed)
        for target in targets:
            orbiting = is_orbiting(target, angular_velocity)
            fleet_guess = min(max(target.ships + 10, 20), surplus)
            if fleet_guess < 1:
                continue

            px, py, turns = lead_aim(source, target, fleet_guess, av_signed, orbiting)
            if crosses_sun(source.x, source.y, px, py):
                continue

            if target.owner == -1:
                ships_on_arrival = target.ships
            else:
                ships_on_arrival = target.ships + int(target.production * turns)
            ships_needed = ships_on_arrival + SAFETY_BUFFER

            if ships_needed > surplus:
                continue

            angle = math.atan2(py - source.y, px - source.x)
            sc = _score(turns, target)
            candidates.append((sc, target.id, angle, ships_needed))

        # Sort by score ascending (best first).
        candidates.sort(key=lambda c: c[0])

        # Greedy allocation: emit launches while budget allows, no duplicate targets.
        remaining_surplus = surplus
        launched_targets: set[int] = set()
        for sc, target_id, angle, ships_needed in candidates:
            if remaining_surplus < MIN_LAUNCH_SHIPS:
                break
            if target_id in launched_targets:
                continue
            if ships_needed > remaining_surplus:
                continue
            moves.append([source.id, angle, ships_needed])
            remaining_surplus -= ships_needed
            launched_targets.add(target_id)

    return moves
