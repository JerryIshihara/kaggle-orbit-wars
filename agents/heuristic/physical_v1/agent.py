"""Heuristic physical agent v1.

Based on the patterns that dominate the public leaderboard (see
logs/research/methodologies.md). Three ingredients:

1. **Accurate physics** — fleet speed = 1 + (maxSpeed − 1) × (log(ships)/log(1000))^1.5.
   Planet rotation simulated via `angular_velocity`.
2. **Lead-aim prediction** — for orbiting targets, iterate 6 times:
   estimate travel time → predict target position → recompute. The
   iteration converges quickly because arrival time is a smooth function
   of target position.
3. **Sun-dodge** — skip targets whose straight-line path passes within
   the sun's radius (point-to-segment distance to the sun).

Per turn, each of my planets picks its best target by score
`travel_time / production` (lower is better), with a small bonus for
neutrals (no enemy reinforcements in flight). Ships sent =
`target.ships + target.production * travel_time + safety_buffer`.

Not yet included (deferred to v2+):
  - Multi-turn lookahead / candidate-strategy search
  - Defensive moves (dodge incoming fleets, reinforce threatened planets)
  - Inner-planet dominance / comet prioritisation
  - Game-phase strategy switching (early-expand / mid-attack / late-finish)
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from ...registry import register

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
    """Does segment (x1,y1)→(x2,y2) pass within SUN_RADIUS + margin of the sun?"""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return _dist_from_sun(x1, y1) <= SUN_RADIUS + SUN_MARGIN
    t = max(0.0, min(1.0, ((SUN_CX - x1) * dx + (SUN_CY - y1) * dy) / len_sq))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(cx - SUN_CX, cy - SUN_CY) <= SUN_RADIUS + SUN_MARGIN


def infer_rotation_sign(planets: list[Planet], initial_planets: list) -> int:
    """angular_velocity magnitude is given; infer the sign from one orbiting planet."""
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


def _score(turns: float, target: Planet) -> float:
    base = turns / max(1, target.production)
    return base * (NEUTRAL_BONUS if target.owner == -1 else 1.0)


@register(
    "physical_v1",
    "Heuristic physical agent — 6-iter lead-aim on rotating targets, sun-dodge, "
    "scores targets by travel_time / production.",
)
def physical_v1_agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    raw_planets = get("planets") or []
    angular_velocity = abs(float(get("angular_velocity") or 0.0))
    initial_planets = get("initial_planets") or []

    planets = [Planet(*p) for p in raw_planets]
    av_sign = infer_rotation_sign(planets, initial_planets)
    av_signed = angular_velocity * av_sign

    my_planets = [p for p in planets if p.owner == player and p.ships >= MIN_LAUNCH_SHIPS]
    targets = [p for p in planets if p.owner != player]
    if not my_planets or not targets:
        return []

    moves = []
    for source in my_planets:
        best = None
        best_score = float("inf")
        best_angle = 0.0
        best_ships = 0

        for target in targets:
            orbiting = is_orbiting(target, angular_velocity)
            fleet_guess = min(max(target.ships + 10, 20), source.ships)
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

            if source.ships < ships_needed:
                continue

            sc = _score(turns, target)
            if sc < best_score:
                best_score = sc
                best = target
                best_angle = math.atan2(py - source.y, px - source.x)
                best_ships = ships_needed

        if best is not None and best_ships > 0:
            moves.append([source.id, best_angle, best_ships])

    return moves
