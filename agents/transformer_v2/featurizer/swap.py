"""Observation seat-swap layer for transformer_v2 featurization.

This module is intentionally dormant: nothing in the live agent calls it by
default. It provides a small stateful layer that can be placed before the
current observation featurizer to relabel player seats and spatial positions
consistently for one episode.

The clockwise mapping is old seat -> new seat:

    0 -> 1, 1 -> 2, ..., N-1 -> 0

Neutral ownership (-1), planet ids, fleet ids, source planet ids, and target
planet ids are preserved. Player/owner seat ids are relabelled, and all spatial
coordinates are rotated around the sun/board center so the old seat's starting
position moves to the next clockwise seat. Fleet headings and comet paths are
rotated by the same angle.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable

MAX_OWNER_SLOTS = 4
P_OWNER = 1
P_X = 2
P_Y = 3
F_OWNER = 1
F_X = 2
F_Y = 3
F_ANGLE = 4
SUN_CX = 50.0
SUN_CY = 50.0
TAU = 2.0 * math.pi


def clockwise_owner_map(num_players: int) -> tuple[int, ...]:
    """Return old-seat -> new-seat mapping for one clockwise seat rotation."""
    n = int(num_players)
    if n < 1 or n > MAX_OWNER_SLOTS:
        raise ValueError(f"num_players must be in [1, {MAX_OWNER_SLOTS}], got {num_players}")
    return tuple((i + 1) % n for i in range(n))


def clockwise_owner_map_from_initial_obs(
    obs: Any,
    *,
    num_players: int | None = None,
) -> tuple[int, ...]:
    """Return old-seat -> new-seat mapping from initial home positions.

    Owner ids are ordered by their initial planet angle around the sun in screen
    coordinates, where increasing ``atan2(y - cy, x - cx)`` is clockwise. The
    owner at each position maps to the owner at the next clockwise position.
    Falls back to numeric seat order when a complete one-planet-per-player
    initial layout is not available.
    """
    n = (
        int(num_players)
        if num_players is not None
        else infer_num_players_from_obs(obs, default=MAX_OWNER_SLOTS)
    )
    numeric = clockwise_owner_map(n)
    get = _getter(obs)
    rows = list(get("initial_planets") or get("planets") or [])
    seats: dict[int, tuple[float, float]] = {}
    for row in rows:
        owner = _row_owner(row, P_OWNER)
        if owner is None or not (0 <= owner < n) or owner in seats:
            continue
        try:
            seats[owner] = (float(row[P_X]), float(row[P_Y]))
        except (TypeError, ValueError, IndexError):
            continue
    if len(seats) != n:
        return numeric
    clockwise = sorted(
        seats,
        key=lambda owner: _screen_clockwise_angle(seats[owner][0], seats[owner][1]),
    )
    mapping = list(range(n))
    for i, owner in enumerate(clockwise):
        mapping[owner] = clockwise[(i + 1) % n]
    return tuple(mapping)


def spatial_rotation_from_initial_obs(
    obs: Any,
    owner_map: Iterable[int],
    *,
    num_players: int | None = None,
) -> float:
    """Return one global rotation angle that moves each old home to its new home.

    The angle is derived from the same initial-seat geometry as
    :func:`clockwise_owner_map_from_initial_obs`. If a complete initial layout is
    not available, fall back to the regular clockwise step ``2π / num_players``.
    """
    mapping = tuple(int(x) for x in owner_map)
    n = int(num_players) if num_players is not None else len(mapping)
    if n <= 1:
        return 0.0
    seats = _initial_owner_positions(obs, n)
    if len(seats) != n:
        return TAU / n
    deltas: list[float] = []
    for owner, target_owner in enumerate(mapping[:n]):
        if owner not in seats or target_owner not in seats:
            continue
        ax = _screen_clockwise_angle(*seats[owner])
        bx = _screen_clockwise_angle(*seats[target_owner])
        deltas.append(_wrap_pi(bx - ax))
    if not deltas:
        return TAU / n
    return _circular_mean(deltas)


def infer_num_players_from_obs(obs: Any, *, default: int = MAX_OWNER_SLOTS) -> int:
    """Infer active player count from ``obs`` owner/player ids.

    The caller should pass the real game ``num_players`` when it is known. This
    fallback is for scripts/smoke checks and keeps 2-player games from being
    remapped as 4-player games when the observation only contains owners 0 and 1.
    """
    get = _getter(obs)
    seats: set[int] = set()
    player = get("player")
    if isinstance(player, int) and player >= 0:
        seats.add(player)
    for key in ("initial_planets", "planets"):
        for row in get(key, []) or []:
            owner = _row_owner(row, P_OWNER)
            if owner is not None and owner >= 0:
                seats.add(owner)
    for row in get("fleets", []) or []:
        owner = _row_owner(row, F_OWNER)
        if owner is not None and owner >= 0:
            seats.add(owner)
    if seats:
        return max(1, min(MAX_OWNER_SLOTS, max(seats) + 1))
    return max(1, min(MAX_OWNER_SLOTS, int(default)))


def swap_observation_seats(
    obs: Any,
    owner_map: Iterable[int],
    *,
    rotation_radians: float | None = None,
    rotate_spatial: bool = True,
) -> Any:
    """Return a copied observation with player/owner seats and space remapped.

    ``owner_map[old] = new``. Owners outside the map, including neutral ``-1``,
    are left unchanged. When ``rotate_spatial`` is true, coordinates are rotated
    around ``(50, 50)`` by ``rotation_radians``; if no angle is supplied it is
    inferred from the observation's initial seat positions.
    """
    mapping = tuple(int(x) for x in owner_map)
    rot = (
        spatial_rotation_from_initial_obs(obs, mapping, num_players=len(mapping))
        if rotate_spatial and rotation_radians is None
        else float(rotation_radians or 0.0)
    )
    out = copy.deepcopy(obs)
    get = _getter(out)
    setv = _setter(out)

    player = get("player")
    if isinstance(player, int):
        setv("player", _map_owner(player, mapping))

    planets = get("planets")
    if planets is not None:
        setv(
            "planets",
            [_replace_planet_row(row, mapping, rot, rotate_spatial) for row in planets],
        )

    initial_planets = get("initial_planets")
    if initial_planets is not None:
        setv(
            "initial_planets",
            [
                _replace_planet_row(row, mapping, rot, rotate_spatial)
                for row in initial_planets
            ],
        )

    fleets = get("fleets")
    if fleets is not None:
        setv(
            "fleets",
            [_replace_fleet_row(row, mapping, rot, rotate_spatial) for row in fleets],
        )

    comets = get("comets")
    if comets is not None and rotate_spatial:
        setv("comets", [_replace_comet_meta(row, rot) for row in comets])

    return out


@dataclass
class ClockwiseSeatSwap:
    """Persistent clockwise seat-swap layer for one game stream.

    Call :meth:`apply` once per observation before featurization. The mapping is
    initialized from the first observation with owner evidence, then reused
    until :meth:`reset` is called or the observed step counter moves backward.
    """

    num_players: int | None = None
    default_num_players: int = MAX_OWNER_SLOTS
    use_initial_positions: bool = True
    rotate_spatial: bool = True

    owner_map: tuple[int, ...] | None = None
    rotation_radians: float | None = None
    last_step: int | None = None

    def reset(self) -> None:
        self.owner_map = None
        self.rotation_radians = None
        self.last_step = None

    def apply(self, obs: Any) -> Any:
        get = _getter(obs)
        step = _safe_int(get("step"), default=0)
        if self.last_step is not None and step < self.last_step:
            self.reset()
        if self.owner_map is None:
            if self.num_players is None and len(_world_owner_slots(obs)) < 2:
                self.last_step = step
                return copy.deepcopy(obs)
            n = (
                int(self.num_players)
                if self.num_players is not None
                else infer_num_players_from_obs(obs, default=self.default_num_players)
            )
            self.owner_map = (
                clockwise_owner_map_from_initial_obs(obs, num_players=n)
                if self.use_initial_positions
                else clockwise_owner_map(n)
            )
            self.rotation_radians = (
                spatial_rotation_from_initial_obs(obs, self.owner_map, num_players=n)
                if self.rotate_spatial
                else 0.0
            )
        self.last_step = step
        return swap_observation_seats(
            obs,
            self.owner_map,
            rotation_radians=self.rotation_radians,
            rotate_spatial=self.rotate_spatial,
        )


def _getter(obj: Any):
    if isinstance(obj, dict):
        return obj.get
    return lambda key, default=None: getattr(obj, key, default)


def _setter(obj: Any):
    if isinstance(obj, dict):
        return lambda key, value: obj.__setitem__(key, value)
    return lambda key, value: setattr(obj, key, value)


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _map_owner(owner: int, owner_map: tuple[int, ...]) -> int:
    if 0 <= owner < len(owner_map):
        return owner_map[owner]
    return owner


def _screen_clockwise_angle(x: float, y: float) -> float:
    return math.atan2(y - SUN_CY, x - SUN_CX)


def _initial_owner_positions(obs: Any, num_players: int) -> dict[int, tuple[float, float]]:
    get = _getter(obs)
    rows = list(get("initial_planets") or get("planets") or [])
    seats: dict[int, tuple[float, float]] = {}
    for row in rows:
        owner = _row_owner(row, P_OWNER)
        if owner is None or not (0 <= owner < num_players) or owner in seats:
            continue
        try:
            seats[owner] = (float(row[P_X]), float(row[P_Y]))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return seats


def _world_owner_slots(obs: Any) -> set[int]:
    get = _getter(obs)
    seats: set[int] = set()
    for key, owner_index in (
        ("initial_planets", P_OWNER),
        ("planets", P_OWNER),
        ("fleets", F_OWNER),
    ):
        for row in get(key, []) or []:
            owner = _row_owner(row, owner_index)
            if owner is not None and owner >= 0:
                seats.add(owner)
    return seats


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % TAU - math.pi


def _circular_mean(angles: list[float]) -> float:
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def _rotate_xy(x: float, y: float, theta: float) -> tuple[float, float]:
    dx = x - SUN_CX
    dy = y - SUN_CY
    ct = math.cos(theta)
    st = math.sin(theta)
    return SUN_CX + dx * ct - dy * st, SUN_CY + dx * st + dy * ct


def _rotate_angle(angle: float, theta: float) -> float:
    return (angle + theta) % TAU


def _row_owner(row: Any, index: int) -> int | None:
    if isinstance(row, dict):
        owner = row.get("owner")
        return int(owner) if owner is not None else None
    try:
        return int(row[index])
    except (TypeError, ValueError, IndexError, KeyError):
        owner = getattr(row, "owner", None)
        return int(owner) if owner is not None else None


def _replace_planet_row(
    row: Any,
    owner_map: tuple[int, ...],
    rotation_radians: float,
    rotate_spatial: bool,
) -> Any:
    out = _replace_owner(row, P_OWNER, owner_map)
    if not rotate_spatial:
        return out
    return _replace_xy(out, P_X, P_Y, rotation_radians)


def _replace_fleet_row(
    row: Any,
    owner_map: tuple[int, ...],
    rotation_radians: float,
    rotate_spatial: bool,
) -> Any:
    out = _replace_owner(row, F_OWNER, owner_map)
    if not rotate_spatial:
        return out
    out = _replace_xy(out, F_X, F_Y, rotation_radians)
    return _replace_angle(out, F_ANGLE, rotation_radians)


def _replace_owner(row: Any, index: int, owner_map: tuple[int, ...]) -> Any:
    if isinstance(row, dict):
        out = copy.deepcopy(row)
        if "owner" in out:
            out["owner"] = _map_owner(int(out["owner"]), owner_map)
        return out

    if hasattr(row, "owner") and not _looks_sequence_like(row):
        out = copy.deepcopy(row)
        out.owner = _map_owner(int(out.owner), owner_map)
        return out

    vals = list(row)
    if len(vals) <= index:
        return row
    vals[index] = _map_owner(int(vals[index]), owner_map)
    return tuple(vals) if isinstance(row, tuple) else vals


def _replace_xy(row: Any, x_index: int, y_index: int, rotation_radians: float) -> Any:
    if isinstance(row, dict):
        out = copy.deepcopy(row)
        if "x" in out and "y" in out:
            out["x"], out["y"] = _rotate_xy(float(out["x"]), float(out["y"]), rotation_radians)
        return out

    if hasattr(row, "x") and hasattr(row, "y") and not _looks_sequence_like(row):
        out = copy.deepcopy(row)
        out.x, out.y = _rotate_xy(float(out.x), float(out.y), rotation_radians)
        return out

    vals = list(row)
    if len(vals) <= max(x_index, y_index):
        return row
    vals[x_index], vals[y_index] = _rotate_xy(
        float(vals[x_index]), float(vals[y_index]), rotation_radians,
    )
    return tuple(vals) if isinstance(row, tuple) else vals


def _replace_angle(row: Any, angle_index: int, rotation_radians: float) -> Any:
    if isinstance(row, dict):
        out = copy.deepcopy(row)
        if "angle" in out:
            out["angle"] = _rotate_angle(float(out["angle"]), rotation_radians)
        return out

    if hasattr(row, "angle") and not _looks_sequence_like(row):
        out = copy.deepcopy(row)
        out.angle = _rotate_angle(float(out.angle), rotation_radians)
        return out

    vals = list(row)
    if len(vals) <= angle_index:
        return row
    vals[angle_index] = _rotate_angle(float(vals[angle_index]), rotation_radians)
    return tuple(vals) if isinstance(row, tuple) else vals


def _replace_comet_meta(row: Any, rotation_radians: float) -> Any:
    out = copy.deepcopy(row)
    if not isinstance(out, dict):
        return out
    paths = out.get("paths")
    if not paths:
        return out
    out["paths"] = [
        [_replace_path_point(pt, rotation_radians) for pt in path]
        for path in paths
    ]
    return out


def _replace_path_point(point: Any, rotation_radians: float) -> Any:
    if isinstance(point, dict):
        out = copy.deepcopy(point)
        if "x" in out and "y" in out:
            out["x"], out["y"] = _rotate_xy(float(out["x"]), float(out["y"]), rotation_radians)
        return out
    vals = list(point)
    if len(vals) < 2:
        return point
    vals[0], vals[1] = _rotate_xy(float(vals[0]), float(vals[1]), rotation_radians)
    return tuple(vals) if isinstance(point, tuple) else vals


def _looks_sequence_like(row: Any) -> bool:
    try:
        len(row)
        row[0]
        return True
    except (TypeError, KeyError, IndexError):
        return False
