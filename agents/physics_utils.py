"""Shared trajectory + launch-planning utilities for physical agents.

Two layers of API:

1. Low-level helpers — `fleet_speed`, `crosses_sun`, `_seg_hits_circle`,
   `find_first_collision`, `lead_aim`, `predict_garrison_at_arrival`,
   `validate_launch` — operate on raw env tuples for cheapness and
   composability.

2. High-level `plan_launch(from_planet, to_planet, ...) -> Launch` — the
   recommended entry point. Sizes the fleet, runs deterministic lead-aim,
   predicts garrison-at-arrival from the in-flight fleet timeline,
   validates the trajectory, and returns a single ``Launch`` record with
   ``.ok``, ``.reason``, ``.angle``, ``.eta``, and ``.ships``.

   Trajectory determinism: static and orbiting planet positions are fully
   determined by ``(initial position, angular_velocity)`` at any future
   turn. Comets, by contrast, only become predictable once they SPAWN
   (their pre-computed path appears in ``obs["comets"]`` at the spawn
   step). Before spawn, no agent — heuristic or learned — can target a
   comet correctly, so we don't try; comets in flight are treated as
   orbiting planets here, which is approximate (their orbits are
   elliptical, not circular) but adequate for the agent's purposes.

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
      "reinforcement"         — target is already ours (success)
      "no_surplus"            — sized fleet exceeds available surplus
      "sun"                   — trajectory crosses the sun
      "boundary"              — trajectory exits the 100×100 board
      "no_collision"          — trajectory hits nothing within max distance
      "wrong_planet_<id>_garrison_<g>"
                              — first hit is a different planet with too much garrison
      "insufficient_garrison_<g>_vs_ships_<n>"
                              — target garrison-at-arrival ≥ our fleet
    """

    from_id: int
    target_id: int
    angle: float
    eta: float
    ships: int
    ok: bool
    reason: str


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
    """Walk the fleet's straight-line trajectory and return the first event.

    Steps in increments of `speed = fleet_speed(fleet_ships)` (= one turn of motion)
    starting from just outside the source planet's rim. Returns one of:
        {"kind": "sun", "step": N, "x": float, "y": float}
        {"kind": "boundary", "step": N, "x": float, "y": float}
        {"kind": "planet", "planet": tuple, "step": N, "eta": float}
        None  (no hit within max_distance turns)
    """
    speed = fleet_speed(fleet_ships)
    if speed <= 0:
        return None
    ch = math.cos(angle)
    sh = math.sin(angle)
    # Launch point: just outside the source rim.
    launch_offset = src_radius + 0.1
    lx = src_x + launch_offset * ch
    ly = src_y + launch_offset * sh

    max_steps = int(max_distance / speed) + 1
    prev_x, prev_y = lx, ly
    for step in range(1, max_steps + 1):
        cur_x = lx + step * speed * ch
        cur_y = ly + step * speed * sh

        # Sun check on per-step segment.
        if crosses_sun(prev_x, prev_y, cur_x, cur_y):
            return {"kind": "sun", "step": step, "x": cur_x, "y": cur_y}

        # Planet check — first planet whose disc the segment hits.
        # Skip the source planet IF launch point is still inside src+1.0 buffer.
        best_planet = None
        best_t = float("inf")
        for p in planets:
            pid = p[P_ID]
            px, py, pr = p[P_X], p[P_Y], p[P_RADIUS]
            if pid == src_id:
                # Skip source if we're still very close to it.
                if math.hypot(prev_x - src_x, prev_y - src_y) < src_radius + 1.0:
                    continue
            if not _seg_hits_circle(prev_x, prev_y, cur_x, cur_y, px, py, pr):
                continue
            # Estimate hit-distance from launch point along the ray.
            t_along = (px - lx) * ch + (py - ly) * sh
            if t_along < best_t:
                best_t = t_along
                best_planet = p
        if best_planet is not None:
            eta = max(0.0, best_t / speed)
            return {"kind": "planet", "planet": best_planet, "step": step, "eta": eta}

        # Boundary check on the new endpoint.
        if (
            cur_x < BOARD_MIN
            or cur_x > BOARD_MAX
            or cur_y < BOARD_MIN
            or cur_y > BOARD_MAX
        ):
            return {"kind": "boundary", "step": step, "x": cur_x, "y": cur_y}

        prev_x, prev_y = cur_x, cur_y

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
    """
    src_x = source[P_X]
    src_y = source[P_Y]
    src_r = source[P_RADIUS]
    src_id = source[P_ID]
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
        dist = math.hypot(tx0 - sx, ty0 - sy)
        return tx0, ty0, dist / max(speed, 1e-6)
    dx = tx0 - SUN_CX
    dy = ty0 - SUN_CY
    r = math.hypot(dx, dy)
    initial_angle = math.atan2(dy, dx)
    px, py = tx0, ty0
    turns = math.hypot(px - sx, py - sy) / speed
    for _ in range(iters):
        a = initial_angle + av_signed * turns
        px = SUN_CX + r * math.cos(a)
        py = SUN_CY + r * math.sin(a)
        turns = math.hypot(px - sx, py - sy) / speed
    return px, py, turns


def plan_launch(
    from_planet: tuple,
    to_planet: tuple,
    *,
    planets: list,
    fleets: list,
    player: int,
    angular_velocity: float,
    av_sign: int = 1,
    fleet_ships: int | None = None,
    surplus: int | None = None,
    safety_buffer: int = 3,
    convergence_iters: int = 3,
) -> Launch:
    """Compute the best from→to launch and validate it end-to-end.

    Steps performed:
      1. Detect target's motion class (static / orbiting).
      2. Iteratively converge ``(eta, ships_needed)`` against
         ``predict_garrison_at_arrival`` so production growth, in-flight
         enemy reinforcements and friendly attacks are all factored in
         when sizing the fleet.
      3. Validate trajectory: sun-cross, board-exit, first-hit-must-be-target.
      4. Return a Launch record with ``.ok`` True iff the launch is sound.

    `fleet_ships`: pin the fleet size. If None, sizes to the smallest count
    that captures the predicted defenders (clamped by `surplus`).

    `surplus`: maximum ships available from the source. If the planner's
    sized fleet exceeds this, returns Launch(ok=False, reason="no_surplus").
    """
    sx = from_planet[P_X]
    sy = from_planet[P_Y]
    sr = from_planet[P_RADIUS]
    src_id = from_planet[P_ID]

    tid = to_planet[P_ID]
    tx0 = to_planet[P_X]
    ty0 = to_planet[P_Y]
    tr = to_planet[P_RADIUS]
    tships = to_planet[P_SHIPS]
    towner = to_planet[P_OWNER]

    av_signed = float(angular_velocity) * (1 if av_sign >= 0 else -1)
    target_orbiting = _is_orbiting_xy(tx0, ty0, tr, angular_velocity)

    # ---- Iterative size-and-aim convergence ----
    # Initial fleet guess: enough for the static defender + a margin. This
    # primes lead_aim with a reasonable speed so the eta estimate is sane.
    if fleet_ships is not None:
        ships = int(fleet_ships)
    else:
        ships = max(20, tships + 10)
        if surplus is not None:
            ships = min(ships, surplus)

    eta = 0.0
    angle = 0.0
    for _ in range(convergence_iters):
        px, py, eta = lead_aim(sx, sy, tx0, ty0, tr, ships, av_signed, target_orbiting)
        angle = math.atan2(py - sy, px - sx)
        if fleet_ships is not None:
            break  # caller fixed the size; just compute angle/eta with it
        # Re-size against predicted garrison so we don't under-fund the launch
        # when production-during-travel + enemy reinforcement is high.
        garrison = predict_garrison_at_arrival(to_planet, eta, player, fleets)
        new_ships = garrison + safety_buffer
        if surplus is not None:
            new_ships = min(new_ships, surplus)
        if new_ships <= ships:
            ships = new_ships
            break
        ships = new_ships

    # ---- Surplus check ----
    if surplus is not None and ships > surplus:
        return Launch(src_id, tid, angle, eta, ships, ok=False, reason="no_surplus")

    # ---- Trajectory validation ----
    hit = find_first_collision(sx, sy, sr, src_id, angle, ships, planets)
    if hit is None:
        return Launch(src_id, tid, angle, eta, ships, ok=False, reason="no_collision")
    if hit["kind"] == "sun":
        return Launch(src_id, tid, angle, eta, ships, ok=False, reason="sun")
    if hit["kind"] == "boundary":
        return Launch(src_id, tid, angle, eta, ships, ok=False, reason="boundary")

    hit_planet = hit["planet"]
    hit_pid = hit_planet[P_ID]
    if hit_pid != tid:
        # First-hit isn't the intended target. If it's enemy/neutral with
        # enough garrison to absorb our fleet, fail. If it's our own planet,
        # the intervening collision is a free reinforcement — that's fine.
        h_owner = hit_planet[P_OWNER]
        if h_owner != player:
            h_garrison = predict_garrison_at_arrival(hit_planet, eta, player, fleets)
            if h_garrison >= ships:
                return Launch(
                    src_id, tid, angle, eta, ships, ok=False,
                    reason=f"wrong_planet_{hit_pid}_garrison_{h_garrison}",
                )
        # Non-target collision but it's ours / undefended; allow.
        return Launch(src_id, tid, angle, eta, ships, ok=True, reason="reinforcement")

    # Hit intended target.
    if towner == player:
        return Launch(src_id, tid, angle, eta, ships, ok=True, reason="reinforcement")

    garrison = predict_garrison_at_arrival(to_planet, eta, player, fleets)
    if ships <= garrison:
        return Launch(
            src_id, tid, angle, eta, ships, ok=False,
            reason=f"insufficient_garrison_{garrison}_vs_ships_{ships}",
        )
    return Launch(src_id, tid, angle, eta, ships, ok=True, reason="capture")
