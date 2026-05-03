"""Shared trajectory + launch-planning utilities for physical agents.

Two layers of API:

1. Low-level helpers — `fleet_speed`, `crosses_sun`, `_seg_hits_circle`,
   `find_first_collision`, `lead_aim`, `predict_garrison_at_arrival`,
   `validate_launch` — operate on raw env tuples for cheapness and
   composability.

2. High-level `plan_launch(from_planet, to_planet, ...) -> Launch` — the
   recommended entry point. It dispatches into separate static / orbital
   / comet planners, sizes the fleet, predicts garrison-at-arrival from
   the in-flight fleet timeline, validates the trajectory, and returns a
   single ``Launch`` record with ``.ok``, ``.reason``, ``.angle``,
   ``.eta``, and ``.ships``.

   Trajectory determinism: static and orbiting planet positions are fully
   determined by ``(initial position, angular_velocity)`` at any future
   turn. Comets, by contrast, only become predictable once they SPAWN
   (their pre-computed path appears in ``obs["comets"]`` at the spawn
   step). When comet metadata is provided, this module now aims them
   against their explicit path instead of approximating them as orbiters.

Planets/fleets are passed as raw env tuples
``[id, owner, x, y, radius, ships, production]``
``[id, owner, x, y, angle, from_planet_id, ships]``
to keep imports light (no Planet/Fleet namedtuple construction needed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

SUN_CX = 50.0
SUN_CY = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0
BOARD_MIN = 0.0
BOARD_MAX = 100.0
MAX_SPEED = 6.0
SPEED_LOG_DENOM = math.log(1000.0)

# Planet tuple field indices
P_ID = 0
P_OWNER = 1
P_X = 2
P_Y = 3
P_RADIUS = 4
P_SHIPS = 5
P_PRODUCTION = 6

# Fleet tuple field indices
F_ID = 0
F_OWNER = 1
F_X = 2
F_Y = 3
F_ANGLE = 4
F_FROM = 5
F_SHIPS = 6

# Orbital geometry
ROTATION_RADIUS_LIMIT = 50.0
LEAD_AIM_ITERS = 6


@dataclass
class Launch:
    """Result of `plan_launch`.

    Always populated even on failure (so callers can inspect why we passed
    on a candidate). `.ok` is the gate; `.reason` is one of:

      "capture"               — will capture target (success)
      "reinforcement"         — first hit IS intended target, target is ours
      "intercept"             — first hit is a *different* planet but the
                                launch still lands cleanly (friendly or
                                weak garrison). ``ok=True`` because ships
                                aren't lost, but the planned target is NOT
                                reached. ``actual_hit_id`` carries the
                                planet that was actually hit. Sniper-style
                                callers should reject this.
      "no_surplus"            — sized fleet exceeds available surplus
      "sun"                   — trajectory crosses the sun (within
                                ``SUN_RADIUS + SUN_MARGIN`` of the center)
      "boundary"              — trajectory exits the 100×100 board
      "no_collision"          — trajectory hits nothing within max distance
      "wrong_planet_<id>_garrison_<g>"
                              — first hit is a different planet with too much garrison
      "insufficient_garrison_<g>_vs_ships_<n>"
                              — target garrison-at-arrival ≥ our fleet
      "comet_expired_at_arrival"
                              — intended target is a comet whose path
                                exhausts before our fleet arrives
      "target_unreachable"    — target path / motion could not be intercepted
    """

    from_id: int
    target_id: int
    angle: float
    eta: float
    ships: int
    ok: bool
    reason: str
    # Planet id the trajectory's collision sweep actually hits first.
    # ``None`` for non-planet outcomes (sun, boundary, no_collision).
    # Sniper-style callers compare this to ``target_id`` to reject
    # incidental wrong-planet intercepts (the trajectory stops at a
    # *different* planet before reaching the intended target).
    actual_hit_id: int | None = None


def fleet_speed(ships: int) -> float:
    """Speed (units/turn) for a fleet of `ships` — same formula as v1/v2/v3/v4."""
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / SPEED_LOG_DENOM) ** 1.5


def crosses_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    """True if segment (x1,y1)→(x2,y2) passes within SUN_RADIUS+SUN_MARGIN of the sun."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - SUN_CX, y1 - SUN_CY) <= SUN_RADIUS + SUN_MARGIN
    t = max(0.0, min(1.0, ((SUN_CX - x1) * dx + (SUN_CY - y1) * dy) / len_sq))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(cx - SUN_CX, cy - SUN_CY) <= SUN_RADIUS + SUN_MARGIN


def _seg_hits_circle(
    x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, r: float
) -> bool:
    """True if segment (x1,y1)→(x2,y2) intersects disc centered at (cx,cy) radius r."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - cx, y1 - cy) <= r
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / len_sq))
    px, py = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy) <= r


def _point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Minimum distance from point ``(px, py)`` to segment ``(x1, y1)``→``(x2, y2)``."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0.0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _segment_projection_fraction(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Projection fraction of point onto a segment, clamped to ``[0, 1]``."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0.0:
        return 0.0
    return max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))


def _swept_disc_intersection(
    fx0: float, fy0: float, fx1: float, fy1: float,
    tx0: float, ty0: float, tx1: float, ty1: float,
    radius: float,
) -> float | None:
    """Earliest ``t ∈ [0, 1]`` such that ``|F(t) - T(t)| ≤ radius``, else None.

    Both fleet and target move LINEARLY within the step:
      ``F(t) = (fx0, fy0) + t·((fx1, fy1) - (fx0, fy0))``
      ``T(t) = (tx0, ty0) + t·((tx1, ty1) - (tx0, ty0))``

    Solving ``|F(t) - T(t)|² = radius²`` is a quadratic in ``t`` —
    closed-form, no per-substep approximation. Used by both the static
    sweep (target stationary) and the dynamic sweep (orbital / comet).
    """
    dx0 = fx0 - tx0
    dy0 = fy0 - ty0
    ddx = (fx1 - fx0) - (tx1 - tx0)
    ddy = (fy1 - fy0) - (ty1 - ty0)
    a = ddx * ddx + ddy * ddy
    b = 2.0 * (dx0 * ddx + dy0 * ddy)
    c = dx0 * dx0 + dy0 * dy0 - radius * radius
    if a < 1e-12:
        # No relative motion — either always inside the disc or always out.
        return 0.0 if c <= 0.0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    t_enter = (-b - sqrt_disc) / (2.0 * a)
    t_exit = (-b + sqrt_disc) / (2.0 * a)
    # Already inside at t=0?
    if t_enter <= 0.0 <= t_exit:
        return 0.0
    if 0.0 <= t_enter <= 1.0:
        return t_enter
    return None


def find_first_collision(
    src_x: float,
    src_y: float,
    src_radius: float,
    src_id: int,
    angle: float,
    fleet_ships: int,
    planets: list,
    max_distance: float = 200.0,
) -> dict | None:
    """Walk the fleet's straight-line trajectory against STATIC planets.

    Mirrors the env's turn order for static snapshots:
      1. move the fleet by one turn of speed
      2. check boundary (env-exact: |coord| > board)
      3. check sun crossing (with SUN_MARGIN as a safety buffer — the
         env's hard cut is ``SUN_RADIUS`` but we drop launches that
         would pass within ``SUN_RADIUS + SUN_MARGIN`` of the sun
         center, mirroring :func:`crosses_sun`)
      4. check planet collisions, picking the EARLIEST ``t`` (continuous
         time within the step) — important when the fleet's segment
         clips multiple discs in one turn.
    """
    speed = fleet_speed(fleet_ships)
    if speed <= 0:
        return None
    ch = math.cos(angle)
    sh = math.sin(angle)
    launch_offset = src_radius + 0.1
    lx = src_x + launch_offset * ch
    ly = src_y + launch_offset * sh
    sun_radius_eff = SUN_RADIUS + SUN_MARGIN

    max_steps = int(max_distance / speed) + 1
    fleet_x, fleet_y = lx, ly
    for step in range(1, max_steps + 1):
        old_x = fleet_x
        old_y = fleet_y
        fleet_x += speed * ch
        fleet_y += speed * sh

        if (
            fleet_x < BOARD_MIN
            or fleet_x > BOARD_MAX
            or fleet_y < BOARD_MIN
            or fleet_y > BOARD_MAX
        ):
            return {"kind": "boundary", "step": step, "x": fleet_x, "y": fleet_y}

        # Sun: stationary disc, swept-disc intersection at sun-margin
        # radius. Use ``<=`` to match the original ``crosses_sun`` semantics.
        t_sun = _swept_disc_intersection(
            old_x, old_y, fleet_x, fleet_y,
            SUN_CX, SUN_CY, SUN_CX, SUN_CY,
            sun_radius_eff,
        )
        if t_sun is not None:
            return {
                "kind": "sun",
                "step": step,
                "x": old_x + t_sun * (fleet_x - old_x),
                "y": old_y + t_sun * (fleet_y - old_y),
                "eta": (step - 1) + t_sun,
            }

        # Planets: find earliest entry across all candidates within the
        # step. Iteration-order picks miss inter-planet ordering when
        # the fleet's segment grazes multiple discs.
        best_t = None
        best_planet = None
        for p in planets:
            pid = p[P_ID]
            px, py, pr = p[P_X], p[P_Y], p[P_RADIUS]
            if pid == src_id and math.hypot(old_x - src_x, old_y - src_y) < src_radius + 1.0:
                continue
            t = _swept_disc_intersection(
                old_x, old_y, fleet_x, fleet_y,
                px, py, px, py,
                pr,
            )
            if t is None:
                continue
            if best_t is None or t < best_t:
                best_t = t
                best_planet = p
        if best_planet is not None:
            return {
                "kind": "planet",
                "planet": best_planet,
                "step": step,
                "eta": (step - 1) + best_t,
            }

    return None


def _fleet_eta_to_planet(fleet, planet) -> float | None:
    """Closest-approach time for a fleet's straight trajectory to a static
    planet disc. None if it misses or is moving away."""
    speed = fleet_speed(fleet[F_SHIPS])
    if speed <= 0:
        return None
    ch = math.cos(fleet[F_ANGLE])
    sh = math.sin(fleet[F_ANGLE])
    dx = planet[P_X] - fleet[F_X]
    dy = planet[P_Y] - fleet[F_Y]
    t = (dx * ch + dy * sh) / speed
    if t < 0:
        return None
    cx = fleet[F_X] + speed * t * ch
    cy = fleet[F_Y] + speed * t * sh
    if math.hypot(cx - planet[P_X], cy - planet[P_Y]) > planet[P_RADIUS]:
        return None
    return t


def predict_garrison_at_arrival(
    target,
    eta: float,
    player: int,
    fleets: list | None = None,
) -> int:
    """Return the expected defender count when *our* fleet reaches `target`.

    Walks the timeline of all in-flight fleets aimed at `target`, applying
    production growth between events and resolving each arrival's combat
    against the running garrison/owner. Returns the garrison the launching
    player will face.

    Combat resolution per arrival:
        - reinforcement (same owner): `garrison += ships`
        - capture attempt: if `attacker_ships > garrison`, ownership flips
          and `garrison = attacker_ships - garrison`; else `garrison -= attacker_ships`
        - simultaneous arrivals from same owner are summed; opposing
          simultaneous arrivals partially cancel.
    """
    powner = target[P_OWNER]
    pprod = target[P_PRODUCTION]
    garrison = float(target[P_SHIPS])
    cur_owner = powner

    arrivals: list[tuple[float, int, int]] = []
    if fleets:
        for f in fleets:
            t = _fleet_eta_to_planet(f, target)
            if t is None or t >= eta:
                continue
            arrivals.append((t, f[F_OWNER], f[F_SHIPS]))
    arrivals.sort(key=lambda x: x[0])

    last_t = 0.0
    i = 0
    while i < len(arrivals):
        t = arrivals[i][0]
        # Production accrues per turn for owned planets.
        if cur_owner >= 0:
            garrison += pprod * max(0.0, t - last_t)
        # Group all arrivals at this same t (within 0.5 turns).
        ships_per_owner: dict[int, int] = {}
        while i < len(arrivals) and arrivals[i][0] - t < 0.5:
            _, ow, sh = arrivals[i]
            ships_per_owner[ow] = ships_per_owner.get(ow, 0) + sh
            i += 1
        # Reinforcement from current owner first.
        if cur_owner in ships_per_owner:
            garrison += ships_per_owner.pop(cur_owner)
        # Resolve attackers among themselves: top vs second.
        if ships_per_owner:
            sorted_atk = sorted(ships_per_owner.items(), key=lambda kv: -kv[1])
            top_o, top_s = sorted_atk[0]
            second_s = sorted_atk[1][1] if len(sorted_atk) > 1 else 0
            if top_s == second_s:
                # Tied attackers annihilate each other; garrison untouched.
                pass
            else:
                survivor = top_s - second_s
                if survivor > garrison:
                    cur_owner = top_o
                    garrison = float(survivor - garrison)
                else:
                    garrison = float(garrison - survivor)
        last_t = t

    # Final stretch of production until our arrival.
    if cur_owner >= 0 and cur_owner != -1:
        garrison += pprod * max(0.0, eta - last_t)

    # If by the time we arrive the planet would already be ours (captured by an
    # ally, or was never hostile), report 0 — caller treats as no threat.
    if cur_owner == player:
        return 0
    return int(math.ceil(garrison))


def validate_launch(
    source: Any,
    angle: float,
    fleet_ships: int,
    intended_target_id: int,
    planets: list,
    player: int,
    safety_buffer: int = 3,
    eta_override: float | None = None,
    fleets: list | None = None,
    *,
    angular_velocity: float = 0.0,
    av_signed: float | None = None,
    comet_planet_ids: set[int] | None = None,
    comets: list | None = None,
) -> tuple[bool, str]:
    """Return (is_valid, reason). False ⇒ drop the launch.

    `source` and entries of `planets` are raw env tuples
    ``[id, owner, x, y, radius, ships, production]``.
    `fleets` (optional) — list of in-flight fleets in the env's tuple format
    ``[id, owner, x, y, angle, from_planet_id, ships]``. When provided, the
    validator predicts garrison-at-arrival by walking the per-arrival
    combat timeline (incoming enemy reinforcements + concurrent friendly
    fleets), which substantially reduces "annihilated" outcomes from
    targets that were softer-looking than the static obs would suggest.

    `eta_override` (optional) — turns until the fleet reaches the intended
    target as estimated by the caller (e.g. ``lead_aim`` for orbiting
    targets). Used in place of the validator's straight-line eta when
    computing production-during-travel.

    Motion-aware collision sweep: when ``angular_velocity > 0`` (orbital)
    or ``comet_planet_ids`` is non-empty, the validator dispatches to
    :func:`_find_first_collision_dynamic` so the trajectory is checked
    against moving planets / comets — same path :func:`plan_launch` uses.
    Static planets fall through to :func:`find_first_collision`. If none
    of the motion args are supplied, behaviour matches the legacy
    static-only validator.
    """
    src_x = source[P_X]
    src_y = source[P_Y]
    src_r = source[P_RADIUS]
    src_id = source[P_ID]

    comet_ids = set(comet_planet_ids or [])
    comet_lookup = _build_comet_lookup(comets) if comets else {}
    abs_av = abs(float(angular_velocity))
    if av_signed is None:
        av_signed = abs_av  # caller didn't supply sign — assume +1.

    # Decide whether ANY moving entity could appear on the trajectory.
    # If yes, use the dynamic sweep; if no, the static path is faster.
    has_motion = abs_av > 0.0 or bool(comet_ids)
    if has_motion:
        hit = _find_first_collision_dynamic(
            src_x,
            src_y,
            src_r,
            src_id,
            angle,
            fleet_ships,
            planets,
            angular_velocity=abs_av,
            av_signed=float(av_signed),
            comet_lookup=comet_lookup,
        )
    else:
        hit = find_first_collision(
            src_x, src_y, src_r, src_id, angle, fleet_ships, planets
        )
    if hit is None:
        return (False, "no_collision")
    if hit["kind"] == "sun":
        return (False, "sun")
    if hit["kind"] == "boundary":
        return (False, "boundary")
    # Planet hit.
    p = hit["planet"]
    pid = p[P_ID]
    powner = p[P_OWNER]
    pships = p[P_SHIPS]
    pprod = p[P_PRODUCTION]
    eta = hit["eta"]
    # Use the longer of validator-eta and caller-eta when computing how much
    # production the defender will have stacked up by arrival. Conservative
    # (we'd rather drop a marginal launch than approve one that gets
    # annihilated mid-flight by an unforeseen orbital crawl).
    eta_for_growth = max(eta, eta_override) if eta_override is not None else eta
    if pid != intended_target_id:
        # Hit a planet we didn't aim at. If it's mine, that's fine (reinforcement-ish).
        # If it's enemy/neutral with garrison >= fleet, we'd waste ships.
        if powner != player:
            if fleets is not None:
                garrison = predict_garrison_at_arrival(p, eta_for_growth, player, fleets)
            else:
                garrison = pships if powner == -1 else pships + int(pprod * eta_for_growth)
            if garrison >= fleet_ships:
                return (
                    False,
                    f"hits_wrong_planet_{pid}_garrison_{garrison}",
                )
        return (True, "ok")
    # Hit intended target.
    if powner == player:
        return (True, "ok")
    if fleets is not None:
        # Fleet-aware: predict per-arrival combat timeline. For OUR target, we
        # only want to *soften* the prediction (use timeline garrison), not
        # short-circuit. If an in-flight friendly fleet would capture it
        # cleanly, let the launch proceed as reinforcement — the marginal
        # cost is small compared to dropping a swarm and falling behind.
        garrison = predict_garrison_at_arrival(p, eta_for_growth, player, fleets)
    else:
        garrison = pships if powner == -1 else pships + int(pprod * eta_for_growth)
    if fleet_ships <= garrison:
        return (
            False,
            f"insufficient_garrison_{garrison}_vs_ships_{fleet_ships}",
        )
    return (True, "ok")


# ---------- High-level launch planner ----------
def _is_orbiting_xy(x: float, y: float, radius: float, angular_velocity: float) -> bool:
    if angular_velocity == 0:
        return False
    return math.hypot(x - SUN_CX, y - SUN_CY) + radius < ROTATION_RADIUS_LIMIT


def _infer_rotation_sign_raw(planets: list, initial_planets: list) -> int:
    """Infer the sign of ``angular_velocity`` from raw env tuples."""
    init = {int(row[P_ID]): row for row in initial_planets}
    for p in planets:
        pid = int(p[P_ID])
        if pid not in init:
            continue
        ip = init[pid]
        ix = float(ip[P_X])
        iy = float(ip[P_Y])
        ir = math.hypot(ix - SUN_CX, iy - SUN_CY)
        cx = float(p[P_X])
        cy = float(p[P_Y])
        cr = math.hypot(cx - SUN_CX, cy - SUN_CY)
        if abs(ir - cr) > 0.5:
            continue
        ia = math.atan2(iy - SUN_CY, ix - SUN_CX)
        ca = math.atan2(cy - SUN_CY, cx - SUN_CX)
        delta = (ca - ia + math.pi) % (2 * math.pi) - math.pi
        if abs(delta) > 1e-3:
            return 1 if delta > 0 else -1
    return 1


def _build_comet_lookup(comets: list | None) -> dict[int, tuple[list, int]]:
    """Map ``planet_id`` → ``(path, path_index)`` for active comet planets."""
    lookup: dict[int, tuple[list, int]] = {}
    for comet_meta in comets or []:
        pids = list(comet_meta.get("planet_ids") or [])
        paths = list(comet_meta.get("paths") or [])
        path_index = int(comet_meta.get("path_index", 0) or 0)
        for seat, pid in enumerate(pids):
            if seat >= len(paths):
                continue
            lookup[int(pid)] = (paths[seat], path_index)
    return lookup


def _target_motion_kind(
    planet: tuple,
    angular_velocity: float,
    comet_planet_ids: set[int],
) -> str:
    """Return one of ``static``, ``orbital``, ``comet`` for ``planet``."""
    pid = int(planet[P_ID])
    if pid in comet_planet_ids:
        return "comet"
    if _is_orbiting_xy(
        float(planet[P_X]),
        float(planet[P_Y]),
        float(planet[P_RADIUS]),
        angular_velocity,
    ):
        return "orbital"
    return "static"


def _predict_orbital_xy(
    x: float,
    y: float,
    turns: float,
    av_signed: float,
) -> tuple[float, float]:
    dx = x - SUN_CX
    dy = y - SUN_CY
    r = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) + av_signed * turns
    return SUN_CX + r * math.cos(angle), SUN_CY + r * math.sin(angle)


def _predict_comet_xy(
    planet: tuple,
    turns: float,
    comet_lookup: dict[int, tuple[list, int]],
) -> tuple[float, float] | None:
    """Interpolate a comet's path at ``turns`` future turns."""
    entry = comet_lookup.get(int(planet[P_ID]))
    if entry is None:
        return None
    path, path_index = entry
    x0 = float(planet[P_X])
    y0 = float(planet[P_Y])
    if turns <= 0:
        return x0, y0

    whole = int(math.floor(turns))
    frac = turns - whole

    def _sample(step: int) -> tuple[float, float] | None:
        if step <= 0:
            return x0, y0
        idx = path_index + step
        if 0 <= idx < len(path):
            px, py = path[idx]
            return float(px), float(py)
        return None

    start = _sample(whole)
    if start is None:
        return None
    if frac == 0.0:
        return start
    end = _sample(whole + 1)
    if end is None:
        return start
    return (
        start[0] + (end[0] - start[0]) * frac,
        start[1] + (end[1] - start[1]) * frac,
    )


def _predict_planet_xy(
    planet: tuple,
    turns: float,
    *,
    angular_velocity: float,
    av_signed: float,
    comet_lookup: dict[int, tuple[list, int]],
) -> tuple[float, float] | None:
    kind = _target_motion_kind(planet, angular_velocity, set(comet_lookup))
    if kind == "comet":
        return _predict_comet_xy(planet, turns, comet_lookup)
    if kind == "orbital":
        return _predict_orbital_xy(
            float(planet[P_X]),
            float(planet[P_Y]),
            turns,
            av_signed,
        )
    return float(planet[P_X]), float(planet[P_Y])


def _find_first_collision_dynamic(
    src_x: float,
    src_y: float,
    src_radius: float,
    src_id: int,
    angle: float,
    fleet_ships: int,
    planets: list,
    *,
    angular_velocity: float,
    av_signed: float,
    comet_lookup: dict[int, tuple[list, int]],
    max_distance: float = 200.0,
) -> dict | None:
    """Per-step swept-disc intersection against MOVING planets / comets.

    The previous implementation split the per-step check into two
    approximations (fleet segment vs planet's start position, then
    planet's sweep vs fleet endpoint). That double-pass produced both
    false positives (calling HIT when relative motion never actually
    closed the gap) and false negatives (missing a hit that occurred
    mid-step when the planet drifted into the fleet's path). Either
    error breaks downstream callers — a false positive in particular
    makes ``validate_launch`` approve a trajectory that the real env
    keeps flying straight, often into the sun or board edge.

    The closed-form fix: for every planet (and the sun), compute
    ``F(t) - T(t)`` as a linear function of ``t ∈ [0, 1]`` within the
    step and solve the quadratic for the earliest hit. Within each
    step we still pick the EARLIEST t across all moving entities, then
    convert to an absolute eta.

    Step ordering still matches the env: boundary → sun → planets.
    Planet positions are advanced once per step so the next step uses
    the new positions as its ``old`` (start-of-step) values.
    """
    speed = fleet_speed(fleet_ships)
    if speed <= 0:
        return None
    ch = math.cos(angle)
    sh = math.sin(angle)
    launch_offset = src_radius + 0.1
    lx = src_x + launch_offset * ch
    ly = src_y + launch_offset * sh
    sun_radius_eff = SUN_RADIUS + SUN_MARGIN

    max_steps = int(max_distance / speed) + 1
    fleet_x, fleet_y = lx, ly
    planet_pos: dict[int, tuple[float, float]] = {
        int(p[P_ID]): (float(p[P_X]), float(p[P_Y]))
        for p in planets
    }
    comet_indices: dict[int, int] = {
        pid: int(path_index)
        for pid, (_path, path_index) in comet_lookup.items()
    }
    comet_ids = set(comet_lookup)
    # Track comets that have run out of path. Once dead they vanish from
    # the collision sweep entirely — matches :func:`extrapolate_trajectory`'s
    # semantics (returns None past path end). Previously dead comets
    # were left as stationary obstacles at their last known spot, which
    # caused both false-positive intercepts and stuck phantom blockers
    # for any subsequent fleet passing through that location.
    dead_comets: set[int] = set()

    for step in range(1, max_steps + 1):
        old_fx = fleet_x
        old_fy = fleet_y
        fleet_x += speed * ch
        fleet_y += speed * sh

        # Boundary first (env-exact: tested against the fleet's endpoint).
        if (
            fleet_x < BOARD_MIN
            or fleet_x > BOARD_MAX
            or fleet_y < BOARD_MIN
            or fleet_y > BOARD_MAX
        ):
            return {"kind": "boundary", "step": step, "x": fleet_x, "y": fleet_y}

        # Compute every entity's NEW position for this step. Static
        # planets stay put; orbital rotate by 1 turn; comets advance
        # one path index, and are marked DEAD when the path runs out.
        new_planet_pos: dict[int, tuple[float, float]] = {}
        for p in planets:
            pid = int(p[P_ID])
            if pid in dead_comets:
                continue  # already removed from sweep
            old_pos = planet_pos[pid]
            if pid in comet_ids:
                path, _idx0 = comet_lookup[pid]
                next_idx = comet_indices[pid] + 1
                if next_idx >= len(path):
                    # Path exhausted — comet has died this turn. Don't
                    # add to new_planet_pos so it's skipped going forward.
                    dead_comets.add(pid)
                    planet_pos.pop(pid, None)
                    comet_indices[pid] = next_idx
                    continue
                nx, ny = path[next_idx]
                new_pos = (float(nx), float(ny))
                comet_indices[pid] = next_idx
            elif _is_orbiting_xy(
                float(p[P_X]),
                float(p[P_Y]),
                float(p[P_RADIUS]),
                angular_velocity,
            ):
                new_pos = _predict_orbital_xy(old_pos[0], old_pos[1], 1.0, av_signed)
            else:
                new_pos = old_pos
            new_planet_pos[pid] = new_pos

        # Sun: stationary, with the safety margin.
        t_sun = _swept_disc_intersection(
            old_fx, old_fy, fleet_x, fleet_y,
            SUN_CX, SUN_CY, SUN_CX, SUN_CY,
            sun_radius_eff,
        )

        # Planets: closed-form swept-disc test against (old_pos → new_pos).
        # Dead comets are excluded — they no longer exist in the env.
        best_t = None
        best_planet = None
        for p in planets:
            pid = int(p[P_ID])
            if pid in dead_comets or pid not in new_planet_pos:
                continue
            old_pos = planet_pos[pid]
            new_pos = new_planet_pos[pid]
            if pid == src_id and math.hypot(old_fx - src_x, old_fy - src_y) < src_radius + 1.0:
                continue
            t = _swept_disc_intersection(
                old_fx, old_fy, fleet_x, fleet_y,
                old_pos[0], old_pos[1], new_pos[0], new_pos[1],
                float(p[P_RADIUS]),
            )
            if t is None:
                continue
            if best_t is None or t < best_t:
                best_t = t
                best_planet = p

        # Sun beats planet only if it lands EARLIER in the step. The
        # env's order says sun is checked before planet within a step,
        # so on a tie we still report sun.
        if t_sun is not None and (best_t is None or t_sun <= best_t):
            return {
                "kind": "sun",
                "step": step,
                "x": old_fx + t_sun * (fleet_x - old_fx),
                "y": old_fy + t_sun * (fleet_y - old_fy),
                "eta": (step - 1) + t_sun,
            }
        if best_planet is not None:
            return {
                "kind": "planet",
                "planet": best_planet,
                "step": step,
                "eta": (step - 1) + best_t,
            }

        # Advance planet positions for the next step.
        planet_pos = new_planet_pos

    return None


def _lead_aim_static(
    sx: float,
    sy: float,
    tx0: float,
    ty0: float,
    fleet_ships: int,
) -> tuple[float, float, float]:
    speed = fleet_speed(fleet_ships)
    dist = math.hypot(tx0 - sx, ty0 - sy)
    return tx0, ty0, dist / max(speed, 1e-6)


def _lead_aim_orbital(
    sx: float,
    sy: float,
    tx0: float,
    ty0: float,
    fleet_ships: int,
    av_signed: float,
    iters: int = LEAD_AIM_ITERS,
) -> tuple[float, float, float]:
    speed = fleet_speed(fleet_ships)
    if av_signed == 0 or speed <= 0:
        return _lead_aim_static(sx, sy, tx0, ty0, fleet_ships)
    px, py = tx0, ty0
    turns = math.hypot(px - sx, py - sy) / speed
    for _ in range(iters):
        px, py = _predict_orbital_xy(tx0, ty0, turns, av_signed)
        turns = math.hypot(px - sx, py - sy) / speed
    return px, py, turns


def _lead_aim_comet(
    sx: float,
    sy: float,
    target: tuple,
    fleet_ships: int,
    comet_lookup: dict[int, tuple[list, int]],
    iters: int = LEAD_AIM_ITERS,
) -> tuple[float, float, float] | None:
    speed = fleet_speed(fleet_ships)
    if speed <= 0:
        return None
    px = float(target[P_X])
    py = float(target[P_Y])
    turns = math.hypot(px - sx, py - sy) / speed
    for _ in range(iters):
        predicted = _predict_comet_xy(target, turns, comet_lookup)
        if predicted is None:
            return None
        px, py = predicted
        turns = math.hypot(px - sx, py - sy) / speed
    return px, py, turns


def lead_aim(
    sx: float, sy: float,
    tx0: float, ty0: float,
    target_radius: float,
    fleet_ships: int,
    av_signed: float,
    target_orbiting: bool,
    iters: int = LEAD_AIM_ITERS,
) -> tuple[float, float, float]:
    """Compute (predicted_target_x, predicted_target_y, eta_turns).

    For static or zero-angular-velocity targets, this is a straight line.
    For orbiting targets, iterates: estimate eta from current target
    position → predict where target will be after eta → recompute eta to
    that predicted position → repeat. Converges in <6 iterations because
    the displacement is small per step.
    """
    speed = fleet_speed(fleet_ships)
    if not target_orbiting or av_signed == 0 or speed <= 0:
        return _lead_aim_static(sx, sy, tx0, ty0, fleet_ships)
    return _lead_aim_orbital(sx, sy, tx0, ty0, fleet_ships, av_signed, iters=iters)


def _finalize_launch(
    *,
    from_planet: tuple,
    to_planet: tuple,
    planets: list,
    fleets: list,
    player: int,
    ships: int,
    angle: float,
    eta: float,
    surplus: int | None,
    motion_kind: str,
    angular_velocity: float,
    av_signed: float,
    comet_lookup: dict[int, tuple[list, int]],
) -> Launch:
    sx = float(from_planet[P_X])
    sy = float(from_planet[P_Y])
    sr = float(from_planet[P_RADIUS])
    src_id = int(from_planet[P_ID])
    tid = int(to_planet[P_ID])
    towner = int(to_planet[P_OWNER])

    if surplus is not None and ships > surplus:
        return Launch(src_id, tid, angle, eta, ships, ok=False, reason="no_surplus")

    has_dynamic_obstacles = bool(comet_lookup) or angular_velocity != 0.0
    if motion_kind == "static" and not has_dynamic_obstacles:
        hit = find_first_collision(sx, sy, sr, src_id, angle, ships, planets)
    else:
        hit = _find_first_collision_dynamic(
            sx,
            sy,
            sr,
            src_id,
            angle,
            ships,
            planets,
            angular_velocity=angular_velocity,
            av_signed=av_signed,
            comet_lookup=comet_lookup,
        )
    if hit is None:
        return Launch(src_id, tid, angle, eta, ships, ok=False, reason="no_collision")
    if hit["kind"] == "sun":
        return Launch(src_id, tid, angle, eta, ships, ok=False, reason="sun")
    if hit["kind"] == "boundary":
        return Launch(src_id, tid, angle, eta, ships, ok=False, reason="boundary")

    hit_planet = hit["planet"]
    hit_pid = int(hit_planet[P_ID])
    # Prefer the swept-disc-derived eta when available (continuous time
    # from collision, more accurate than the lead-aim turns estimate).
    hit_eta = float(hit.get("eta", eta))

    # Comet-alive-at-arrival check. If the intended target is a comet
    # and its path runs out before our fleet would arrive, the env will
    # see no comet there — even if the swept-disc sweep "hit" the
    # comet's last-known position. Reject so callers (including
    # plan_launch users that aren't sniper) don't queue a launch
    # against a target that won't exist on touchdown. Matches
    # ``extrapolate_trajectory``'s "None past path end" semantics.
    if tid in comet_lookup:
        path, path_index = comet_lookup[tid]
        eta_steps = int(math.ceil(hit_eta))
        if path_index + eta_steps >= len(path):
            return Launch(
                src_id, tid, angle, eta, ships,
                ok=False, reason="comet_expired_at_arrival",
                actual_hit_id=hit_pid,
            )
    if hit_pid != tid:
        h_owner = int(hit_planet[P_OWNER])
        if h_owner != player:
            h_garrison = predict_garrison_at_arrival(hit_planet, hit_eta, player, fleets)
            if h_garrison >= ships:
                return Launch(
                    src_id,
                    tid,
                    angle,
                    eta,
                    ships,
                    ok=False,
                    reason=f"wrong_planet_{hit_pid}_garrison_{h_garrison}",
                    actual_hit_id=hit_pid,
                )
        # The trajectory's first hit isn't the intended target — distinguish
        # this from a real reinforcement so callers (e.g. sniper) can
        # reject incidental wrong-planet intercepts. ``ok`` stays True
        # because the launch still does *something* useful (lands on a
        # friendly / weak planet); it's just not the planned outcome.
        return Launch(
            src_id, tid, angle, eta, ships,
            ok=True, reason="intercept", actual_hit_id=hit_pid,
        )

    if towner == player:
        return Launch(
            src_id, tid, angle, eta, ships,
            ok=True, reason="reinforcement", actual_hit_id=hit_pid,
        )

    garrison = predict_garrison_at_arrival(to_planet, hit_eta, player, fleets)
    if ships <= garrison:
        return Launch(
            src_id,
            tid,
            angle,
            eta,
            ships,
            ok=False,
            reason=f"insufficient_garrison_{garrison}_vs_ships_{ships}",
            actual_hit_id=hit_pid,
        )
    return Launch(
        src_id, tid, angle, eta, ships,
        ok=True, reason="capture", actual_hit_id=hit_pid,
    )


def _plan_launch_for_motion(
    *,
    from_planet: tuple,
    to_planet: tuple,
    planets: list,
    fleets: list,
    player: int,
    fleet_ships: int | None,
    surplus: int | None,
    safety_buffer: int,
    convergence_iters: int,
    motion_kind: str,
    intercept_fn,
    angular_velocity: float,
    av_signed: float,
    comet_lookup: dict[int, tuple[list, int]],
) -> Launch:
    sx = float(from_planet[P_X])
    sy = float(from_planet[P_Y])
    src_id = int(from_planet[P_ID])
    tid = int(to_planet[P_ID])
    tships = int(to_planet[P_SHIPS])

    if fleet_ships is not None:
        ships = int(fleet_ships)
    else:
        ships = max(20, tships + 10)
        if surplus is not None:
            ships = min(ships, surplus)

    eta = 0.0
    angle = 0.0
    for _ in range(convergence_iters):
        intercept = intercept_fn(ships)
        if intercept is None:
            # For comet motion, ``intercept_fn`` returns None when the
            # comet's path is too short to predict a future position
            # at the iterative ETA — i.e., the comet would die before
            # the fleet arrives. Surface that as the explicit reason
            # so plan_launch users (not just sniper) can distinguish
            # timing failures from straight unreachables.
            reason = (
                "comet_expired_at_arrival"
                if motion_kind == "comet"
                else "target_unreachable"
            )
            return Launch(src_id, tid, angle, eta, ships, ok=False, reason=reason)
        px, py, eta = intercept
        angle = math.atan2(py - sy, px - sx)
        if fleet_ships is not None:
            break
        garrison = predict_garrison_at_arrival(to_planet, eta, player, fleets)
        new_ships = garrison + safety_buffer
        if surplus is not None:
            new_ships = min(new_ships, surplus)
        if new_ships <= ships:
            ships = new_ships
            break
        ships = new_ships

    return _finalize_launch(
        from_planet=from_planet,
        to_planet=to_planet,
        planets=planets,
        fleets=fleets,
        player=player,
        ships=ships,
        angle=angle,
        eta=eta,
        surplus=surplus,
        motion_kind=motion_kind,
        angular_velocity=angular_velocity,
        av_signed=av_signed,
        comet_lookup=comet_lookup,
    )


def _plan_launch_static(
    *,
    from_planet: tuple,
    to_planet: tuple,
    planets: list,
    fleets: list,
    player: int,
    angular_velocity: float,
    av_signed: float,
    comet_lookup: dict[int, tuple[list, int]],
    fleet_ships: int | None,
    surplus: int | None,
    safety_buffer: int,
    convergence_iters: int,
) -> Launch:
    sx = float(from_planet[P_X])
    sy = float(from_planet[P_Y])
    tx0 = float(to_planet[P_X])
    ty0 = float(to_planet[P_Y])
    return _plan_launch_for_motion(
        from_planet=from_planet,
        to_planet=to_planet,
        planets=planets,
        fleets=fleets,
        player=player,
        fleet_ships=fleet_ships,
        surplus=surplus,
        safety_buffer=safety_buffer,
        convergence_iters=convergence_iters,
        motion_kind="static",
        intercept_fn=lambda ships: _lead_aim_static(sx, sy, tx0, ty0, ships),
        angular_velocity=angular_velocity,
        av_signed=av_signed,
        comet_lookup=comet_lookup,
    )


def _plan_launch_orbital(
    *,
    from_planet: tuple,
    to_planet: tuple,
    planets: list,
    fleets: list,
    player: int,
    angular_velocity: float,
    av_signed: float,
    comet_lookup: dict[int, tuple[list, int]],
    fleet_ships: int | None,
    surplus: int | None,
    safety_buffer: int,
    convergence_iters: int,
) -> Launch:
    sx = float(from_planet[P_X])
    sy = float(from_planet[P_Y])
    tx0 = float(to_planet[P_X])
    ty0 = float(to_planet[P_Y])
    return _plan_launch_for_motion(
        from_planet=from_planet,
        to_planet=to_planet,
        planets=planets,
        fleets=fleets,
        player=player,
        fleet_ships=fleet_ships,
        surplus=surplus,
        safety_buffer=safety_buffer,
        convergence_iters=convergence_iters,
        motion_kind="orbital",
        intercept_fn=lambda ships: _lead_aim_orbital(
            sx,
            sy,
            tx0,
            ty0,
            ships,
            av_signed,
        ),
        angular_velocity=angular_velocity,
        av_signed=av_signed,
        comet_lookup=comet_lookup,
    )


def _plan_launch_comet(
    *,
    from_planet: tuple,
    to_planet: tuple,
    planets: list,
    fleets: list,
    player: int,
    angular_velocity: float,
    av_signed: float,
    comet_lookup: dict[int, tuple[list, int]],
    fleet_ships: int | None,
    surplus: int | None,
    safety_buffer: int,
    convergence_iters: int,
) -> Launch:
    sx = float(from_planet[P_X])
    sy = float(from_planet[P_Y])
    return _plan_launch_for_motion(
        from_planet=from_planet,
        to_planet=to_planet,
        planets=planets,
        fleets=fleets,
        player=player,
        fleet_ships=fleet_ships,
        surplus=surplus,
        safety_buffer=safety_buffer,
        convergence_iters=convergence_iters,
        motion_kind="comet",
        intercept_fn=lambda ships: _lead_aim_comet(
            sx,
            sy,
            to_planet,
            ships,
            comet_lookup,
        ),
        angular_velocity=angular_velocity,
        av_signed=av_signed,
        comet_lookup=comet_lookup,
    )


def plan_launch(
    from_planet: tuple,
    to_planet: tuple,
    *,
    planets: list,
    fleets: list,
    player: int,
    angular_velocity: float,
    av_sign: int = 1,
    comet_planet_ids: set[int] | None = None,
    comets: list | None = None,
    fleet_ships: int | None = None,
    surplus: int | None = None,
    safety_buffer: int = 3,
    convergence_iters: int = 3,
) -> Launch:
    """Compute the best from→to launch and validate it end-to-end.

    Steps performed:
      1. Detect target's motion class (static / orbiting / comet).
      2. Iteratively converge ``(eta, ships_needed)`` against
         ``predict_garrison_at_arrival`` so production growth, in-flight
         enemy reinforcements and friendly attacks are all factored in
         when sizing the fleet.
      3. Validate trajectory against the same motion class: static planets
         use the old static collision checker; orbital planets and comets
         use a moving-target collision sweep.
      4. Return a Launch record with ``.ok`` True iff the launch is sound.

    `fleet_ships`: pin the fleet size. If None, sizes to the smallest count
    that captures the predicted defenders (clamped by `surplus`).

    `surplus`: maximum ships available from the source. If the planner's
    sized fleet exceeds this, returns Launch(ok=False, reason="no_surplus").
    """
    av_signed = abs(float(angular_velocity)) * (1 if av_sign >= 0 else -1)
    comet_ids = set(comet_planet_ids or [])
    comet_lookup = _build_comet_lookup(comets)
    motion_kind = _target_motion_kind(to_planet, abs(float(angular_velocity)), comet_ids)

    if motion_kind == "comet":
        return _plan_launch_comet(
            from_planet=from_planet,
            to_planet=to_planet,
            planets=planets,
            fleets=fleets,
            player=player,
            angular_velocity=abs(float(angular_velocity)),
            av_signed=av_signed,
            comet_lookup=comet_lookup,
            fleet_ships=fleet_ships,
            surplus=surplus,
            safety_buffer=safety_buffer,
            convergence_iters=convergence_iters,
        )
    if motion_kind == "orbital":
        return _plan_launch_orbital(
            from_planet=from_planet,
            to_planet=to_planet,
            planets=planets,
            fleets=fleets,
            player=player,
            angular_velocity=abs(float(angular_velocity)),
            av_signed=av_signed,
            comet_lookup=comet_lookup,
            fleet_ships=fleet_ships,
            surplus=surplus,
            safety_buffer=safety_buffer,
            convergence_iters=convergence_iters,
        )
    return _plan_launch_static(
        from_planet=from_planet,
        to_planet=to_planet,
        planets=planets,
        fleets=fleets,
        player=player,
        angular_velocity=abs(float(angular_velocity)),
        av_signed=av_signed,
        comet_lookup=comet_lookup,
        fleet_ships=fleet_ships,
        surplus=surplus,
        safety_buffer=safety_buffer,
        convergence_iters=convergence_iters,
    )


# ---------- Trajectory extrapolation ----------
def extrapolate_trajectory(
    planet_id: int,
    n_steps: int,
    obs: Any,
) -> list[tuple[float, float] | None]:
    """Predicted ``(x, y)`` for ``planet_id`` at turns t+1, t+2, ..., t+n_steps.

    Resolution by entity type, in priority order:

      1. ``planet_id in obs['comet_planet_ids']`` — read positions from
         the matching ``obs['comets'][k].paths[seat]`` array, indexed
         starting at the comet's current ``path_index + 1``. Returns
         ``None`` for any horizon past the path's end (the comet will be
         gone by then).
      2. Orbiting planet (body fits inside ``ROTATION_RADIUS_LIMIT`` and
         ``angular_velocity != 0``) — rotate the radial vector by
         ``angular_velocity * h`` per step. Always returns a valid
         position; planets persist for the whole episode.
      3. Static planet (outer ring or zero ω) — repeats current ``(x, y)``
         for every horizon.

    If the planet id isn't present in ``obs['planets']`` at all, returns
    ``[None] * n_steps``.
    """
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    planets = get("planets") or []
    angular_velocity = float(get("angular_velocity") or 0.0)
    obs_step = int(get("step", 0) or 0)
    comet_planet_ids = set(get("comet_planet_ids") or [])

    planet = next((p for p in planets if int(p[P_ID]) == planet_id), None)
    if planet is None:
        return [None] * n_steps

    x = float(planet[P_X])
    y = float(planet[P_Y])
    radius = float(planet[P_RADIUS])

    # Comet path lookup.
    if planet_id in comet_planet_ids:
        for comet_meta in (get("comets") or []):
            pids = list(comet_meta.get("planet_ids") or [])
            if planet_id not in pids:
                continue
            seat = pids.index(planet_id)
            paths = comet_meta.get("paths") or []
            if seat >= len(paths):
                break
            path = paths[seat]
            path_index = int(comet_meta.get("path_index", 0) or 0)
            results: list[tuple[float, float] | None] = []
            for h in range(1, n_steps + 1):
                idx = path_index + h
                if 0 <= idx < len(path):
                    px, py = path[idx]
                    results.append((float(px), float(py)))
                else:
                    results.append(None)
            return results
        # Marked as comet but no matching entry in obs['comets'] — degenerate.
        return [None] * n_steps

    # Orbiting planet: deterministic rotation.
    if _is_orbiting_xy(x, y, radius, angular_velocity):
        dx = x - SUN_CX
        dy = y - SUN_CY
        out: list[tuple[float, float] | None] = []
        for h in range(1, n_steps + 1):
            # In the env, the first post-reset observation (step 0) and the
            # next observation (step 1) share the same orbital positions; the
            # first actual orbital advance appears at step 2. After that,
            # motion advances one angular-velocity tick per turn.
            turn_delta = h if obs_step >= 1 else max(0, h - 1)
            theta = angular_velocity * turn_delta
            c, s = math.cos(theta), math.sin(theta)
            out.append((SUN_CX + dx * c - dy * s, SUN_CY + dx * s + dy * c))
        return out

    # Static planet — never moves.
    return [(x, y)] * n_steps


# ---------- Sniper launcher ----------
def sniper(
    from_planet: tuple,
    to_planet: tuple,
    num_ships: int,
    obs: Any,
    *,
    player: int,
    safety_buffer: int = 3,
    max_game_turns: int = 500,
    av_sign: int | None = None,
) -> tuple[int, float, float]:
    """Validate a launch of exactly ``num_ships`` from ``from_planet`` to
    ``to_planet`` against the current observation. Returns
    ``(ships, angle, eta)``: ``ships == 0`` means the launch is invalid;
    a positive ``ships`` value (always equal to ``num_ships``) means
    every gate passed and the caller can issue the move with the
    returned ``angle``.

    A launch is *invalid* (returns ``(0, 0.0, 0.0)``) if any of:

      * ``num_ships <= 0``.
      * ``plan_launch`` reports failure — sun crossing, board boundary
        exit, wrong-planet collision, or insufficient ships vs the
        predicted garrison-at-arrival.
      * Target is a comet whose path runs out before the fleet arrives
        (comet has died / left, planet position is no longer the comet's
        true future location).
      * The episode ends (``current_step + ceil(eta) >= max_game_turns``)
        before the fleet arrives.

    The angle/eta are returned alongside the ships count so callers that
    pass the gate can build the move directly without a second
    ``plan_launch`` call.
    """
    if num_ships <= 0:
        return 0, 0.0, 0.0

    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    planets = get("planets") or []
    fleets = get("fleets") or []
    initial_planets = get("initial_planets") or []
    angular_velocity = abs(float(get("angular_velocity") or 0.0))
    current_step = int(get("step", 0) or 0)
    comet_planet_ids = set(get("comet_planet_ids") or [])
    if av_sign is None:
        av_sign = _infer_rotation_sign_raw(planets, initial_planets)

    launch = plan_launch(
        from_planet, to_planet,
        planets=planets, fleets=fleets,
        player=player, angular_velocity=angular_velocity,
        av_sign=av_sign,
        comet_planet_ids=comet_planet_ids,
        comets=get("comets") or [],
        fleet_ships=int(num_ships),
        safety_buffer=safety_buffer,
    )
    if not launch.ok:
        return 0, 0.0, 0.0

    # Sniper precision: the trajectory's actual first-hit planet MUST
    # equal the intended target. ``plan_launch`` exposes the swept-disc
    # collision result via ``actual_hit_id``, so we don't have to parse
    # reason strings or special-case the friendly-target branch (the
    # previous version only rejected wrong-planet intercepts when the
    # intended target was an enemy, missing the case where the intended
    # target itself was friendly and the trajectory landed on a
    # *different* friendly planet first).
    intended_pid = int(to_planet[P_ID])
    if launch.actual_hit_id is None or launch.actual_hit_id != intended_pid:
        return 0, 0.0, 0.0

    # Game ends before arrival?
    eta_steps = int(math.ceil(launch.eta))
    if current_step + eta_steps >= max_game_turns:
        return 0, 0.0, 0.0

    # Comet target still alive at arrival?
    target_pid = int(to_planet[P_ID])
    if target_pid in comet_planet_ids:
        for comet_meta in (get("comets") or []):
            pids = list(comet_meta.get("planet_ids") or [])
            if target_pid not in pids:
                continue
            seat = pids.index(target_pid)
            paths = comet_meta.get("paths") or []
            if seat >= len(paths):
                return 0, 0.0, 0.0
            path = paths[seat]
            path_index = int(comet_meta.get("path_index", 0) or 0)
            if path_index + eta_steps >= len(path):
                return 0, 0.0, 0.0
            break
        else:
            # Marked as comet but no matching entry — treat as invalid.
            return 0, 0.0, 0.0

    return int(num_ships), float(launch.angle), float(launch.eta)
