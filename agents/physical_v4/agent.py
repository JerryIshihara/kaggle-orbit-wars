"""Heuristic physical agent v4 — v2 + phase tuning + frontier scoring + multi-source swarm.

Three improvements over v2 (in priority order by expected leverage):

1. Multi-source coordinated swarm with ETA tolerance (Pascal's orbitwork-v14 style).
   After per-source candidates are picked, a coordination pass finds targets where
   2–4 of my planets can launch this turn with arrival times within ETA_TOLERANCE
   turns. Their combined fleet overwhelms defenders that could repel any single
   fleet alone.

2. Frontier-distance target scoring (ykhnkf/distance-prioritized-agent style).
   Targets are scored by their proximity to MY BORDER (min distance from any of
   my planets), not just from the launching source. Pushes outward expansion
   from the frontier rather than stranding fleets across the map.

3. Phase-aware constants (debugendless/sun-dodging-baseline style).
   Game splits into EARLY / OPENING / MID / LATE / VERY_LATE phases derived
   from step + remaining turns. Each phase has its own neutral_bonus,
   defense_buffer, min_launch, safety, frontier weight, and ETA tolerance —
   aggressive expansion early, racing finish late.

Everything else (lead-aim physics, sun-dodge, rotation sign inference, defensive
threat accounting via fleet-arrival timeline) is reused from v2.
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from ..registry import register

# ---------- Physics constants (shared with v1/v2) ----------
SUN_CX = 50.0
SUN_CY = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0
MAX_SPEED = 6.0
SPEED_LOG_DENOM = math.log(1000.0)
ROTATION_RADIUS_LIMIT = 50.0
LEAD_AIM_ITERS = 6

# ---------- Phase tuning ----------
EPISODE_STEPS = 500
EARLY_TURN_LIMIT = 40
OPENING_TURN_LIMIT = 80
LATE_REMAINING = 60
VERY_LATE_REMAINING = 25

# Each phase tuple: (neutral_bonus, defense_buffer, min_launch,
#                    safety_buffer, frontier_weight, swarm_eta_tolerance)
# - neutral_bonus < 1.0 → prefer neutral targets (cheaper)
# - defense_buffer    → ships held back as garrison floor
# - min_launch        → smallest fleet to ever send
# - safety_buffer     → extra ships beyond defended count
# - frontier_weight   → exponent on (1 + frontier_dist/50). >1 = stronger preference
#                       for nearby targets; <1 = OK to reach further
# - swarm_eta_tolerance → max turns between fleet arrivals to count as a coalition
PHASE_TABLE = {
    "EARLY":     (0.70, 2, 4, 2, 1.0, 1),
    "OPENING":   (0.80, 3, 5, 3, 1.2, 1),
    "MID":       (0.90, 4, 5, 3, 1.0, 2),
    "LATE":      (0.85, 3, 5, 2, 0.9, 2),
    "VERY_LATE": (0.70, 1, 3, 1, 0.7, 3),
}

# ---------- Swarm constants ----------
SWARM_MAX_SOURCES = 4
SWARM_EXTRA_MARGIN = 4   # ships beyond defended_total to confirm a coalition wins


def phase_of(step: int) -> str:
    remaining = EPISODE_STEPS - step
    if remaining < VERY_LATE_REMAINING:
        return "VERY_LATE"
    if remaining < LATE_REMAINING:
        return "LATE"
    if step < EARLY_TURN_LIMIT:
        return "EARLY"
    if step < OPENING_TURN_LIMIT:
        return "OPENING"
    return "MID"


# ---------- Reused physics from v2 ----------
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


def compute_surplus(source: Planet, enemy_fleets: list[Fleet], defense_buffer: int) -> int:
    threats = []
    for f in enemy_fleets:
        eta = fleet_eta_to_planet(f, source)
        if eta is not None:
            threats.append((eta, f.ships))
    if not threats:
        return max(0, source.ships - defense_buffer)
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
    return int(max(0, math.floor(min_garrison) - defense_buffer))


# ---------- v4-specific scoring + coalition logic ----------
def frontier_distance(target: Planet, my_planets: list[Planet]) -> float:
    """Min distance from any of my planets to target. Lower = closer to my border."""
    if not my_planets:
        return float("inf")
    return min(math.hypot(p.x - target.x, p.y - target.y) for p in my_planets)


def score_target(
    turns: float,
    target: Planet,
    frontier_dist: float,
    neutral_bonus: float,
    frontier_w: float,
) -> float:
    base = turns / max(1, target.production)
    # Frontier penalty: distant targets are worse. (1 + d/50) so d=0 → 1, d=50 → 2.
    frontier_term = (1.0 + frontier_dist / 50.0) ** frontier_w
    score = base * frontier_term
    if target.owner == -1:
        score *= neutral_bonus
    return score


def candidates_for_source(
    source: Planet,
    targets: list[Planet],
    my_planets: list[Planet],
    surplus: int,
    av_signed: float,
    angular_velocity: float,
    neutral_bonus: float,
    frontier_w: float,
    safety: int,
) -> list[tuple]:
    """Return [(target, eta, ships_needed, angle, score), ...] sorted by score asc."""
    out = []
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
        ships_needed = ships_on_arrival + safety
        if ships_needed > surplus:
            continue
        fd = frontier_distance(target, my_planets)
        sc = score_target(turns, target, fd, neutral_bonus, frontier_w)
        angle = math.atan2(py - source.y, px - source.x)
        out.append((target, turns, ships_needed, angle, sc))
    out.sort(key=lambda x: x[4])
    return out


def find_coalition(
    target: Planet,
    sources_with_surplus: list[tuple[Planet, int]],
    av_signed: float,
    angular_velocity: float,
    eta_tolerance: int,
    safety: int,
    min_launch: int,
):
    """Try to assemble 2–4 of our planets to swarm `target` within eta_tolerance.

    Returns (target, [(source, ships, angle, eta), ...]) or None.
    """
    orbiting = is_orbiting(target, angular_velocity)
    candidates = []
    for source, surplus in sources_with_surplus:
        if surplus < min_launch:
            continue
        # Use a guess fleet size for the lead-aim computation; refined below.
        fleet_guess = min(surplus, max(20, target.ships + 10))
        px, py, eta = lead_aim(source, target, fleet_guess, av_signed, orbiting)
        if crosses_sun(source.x, source.y, px, py):
            continue
        angle = math.atan2(py - source.y, px - source.x)
        candidates.append((source, eta, angle, surplus))
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda x: x[1])  # by ETA

    # Try larger coalitions first.
    for size in range(min(SWARM_MAX_SOURCES, len(candidates)), 1, -1):
        for i in range(len(candidates) - size + 1):
            window = candidates[i : i + size]
            eta_range = window[-1][1] - window[0][1]
            if eta_range > eta_tolerance:
                continue
            # Use the latest ETA in the window for production-during-travel,
            # since defenders have until then to grow.
            latest_eta = window[-1][1]
            if target.owner == -1:
                defended = target.ships
            else:
                defended = target.ships + int(target.production * latest_eta)
            total_needed = defended + safety + SWARM_EXTRA_MARGIN
            total_surplus = sum(s for _, _, _, s in window)
            if total_surplus < total_needed:
                continue
            # Allocate proportional to surplus, capped at each source's surplus.
            per_source = []
            for src, eta, angle, surplus in window:
                fair = int(math.ceil(total_needed * surplus / total_surplus))
                ships = max(min_launch, min(surplus, fair))
                per_source.append((src, ships, angle, eta))
            # Sanity: combined fleet must still beat defenders even after caps.
            if sum(ships for _, ships, _, _ in per_source) < total_needed:
                continue
            return (target, per_source)
    return None


# ---------- Agent ----------
@register(
    "physical_v4",
    "physical_v2 + phase tuning + frontier-distance scoring + multi-source ETA-coordinated swarms.",
)
def physical_v4_agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    raw_planets = get("planets") or []
    raw_fleets = get("fleets") or []
    angular_velocity = abs(float(get("angular_velocity") or 0.0))
    initial_planets = get("initial_planets") or []
    step = int(get("step") or 0)

    phase = phase_of(step)
    neutral_bonus, defense_buffer, min_launch, safety, frontier_w, eta_tol = PHASE_TABLE[phase]

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    av_sign = infer_rotation_sign(planets, initial_planets)
    av_signed = angular_velocity * av_sign

    my_planets = [p for p in planets if p.owner == player and p.ships >= min_launch]
    targets = [p for p in planets if p.owner != player]
    enemy_fleets = [f for f in fleets if f.owner != player and f.owner >= 0]
    if not my_planets or not targets:
        return []

    sources_with_surplus = []
    for source in my_planets:
        surplus = compute_surplus(source, enemy_fleets, defense_buffer)
        if surplus >= min_launch:
            sources_with_surplus.append((source, surplus))
    if not sources_with_surplus:
        return []

    moves: list[list] = []
    used_sources: set[int] = set()

    # ----- STAGE 1: multi-source coalitions -----
    # Score targets by frontier distance + production to pick which we'd most
    # like to swarm; then for each, try to assemble a coalition.
    target_priority = sorted(
        targets,
        key=lambda t: (
            frontier_distance(t, my_planets) / max(1, t.production)
            * (neutral_bonus if t.owner == -1 else 1.0)
        ),
    )
    for target in target_priority:
        coalition = find_coalition(
            target,
            [(s, surplus) for (s, surplus) in sources_with_surplus if s.id not in used_sources],
            av_signed,
            angular_velocity,
            eta_tol,
            safety,
            min_launch,
        )
        if coalition is None:
            continue
        _, per_source = coalition
        for src, ships, angle, _eta in per_source:
            moves.append([src.id, angle, ships])
            used_sources.add(src.id)

    # ----- STAGE 2: independent per-source picks for unused sources -----
    for source, surplus in sources_with_surplus:
        if source.id in used_sources:
            continue
        cands = candidates_for_source(
            source,
            targets,
            my_planets,
            surplus,
            av_signed,
            angular_velocity,
            neutral_bonus,
            frontier_w,
            safety,
        )
        if not cands:
            continue
        _target, _eta, ships_needed, angle, _score = cands[0]
        moves.append([source.id, angle, ships_needed])

    return moves
