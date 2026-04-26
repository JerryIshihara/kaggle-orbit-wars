"""Shared trajectory-validation utilities for physical agents.

Centralizes the segment-vs-disc collision math, fleet-speed formula,
and a `validate_launch` filter that drops obviously wasteful launches:

    - trajectory crosses the sun
    - fleet exits the board without hitting anything
    - first hit is a non-target enemy/neutral with garrison >= our fleet
    - first hit is the intended target but garrison-on-arrival >= our fleet

Planets/fleets are passed as raw env tuples
``[id, owner, x, y, radius, ships, production]``
``[id, owner, x, y, angle, from_planet_id, ships]``
to keep the validator import-light (no Planet/Fleet construction needed).
"""

from __future__ import annotations

import math
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


def validate_launch(
    source: Any,
    angle: float,
    fleet_ships: int,
    intended_target_id: int,
    planets: list,
    player: int,
    safety_buffer: int = 3,
    eta_override: float | None = None,
) -> tuple[bool, str]:
    """Return (is_valid, reason). False ⇒ drop the launch.

    `source` and entries of `planets` are raw env tuples
    ``[id, owner, x, y, radius, ships, production]``.

    `eta_override` (optional) — turns until the fleet reaches the intended
    target as estimated by the caller (e.g. ``lead_aim`` for orbiting
    targets). If provided, used in place of the validator's straight-line
    ETA when computing production-during-travel for garrison growth — this
    closes the orbital-target gap where the validator's eta is shorter
    than the actual lead-aim eta and so under-estimates the defender.
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
            garrison = pships
            if powner != -1:
                garrison = pships + int(pprod * eta_for_growth)
            if garrison >= fleet_ships:
                return (
                    False,
                    f"hits_wrong_planet_{pid}_garrison_{garrison}",
                )
        return (True, "ok")
    # Hit intended target.
    if powner == player:
        return (True, "ok")
    if powner == -1:
        garrison = pships
    else:
        garrison = pships + int(pprod * eta_for_growth)
    if fleet_ships <= garrison:
        return (
            False,
            f"insufficient_garrison_{garrison}_vs_ships_{fleet_ships}",
        )
    return (True, "ok")
