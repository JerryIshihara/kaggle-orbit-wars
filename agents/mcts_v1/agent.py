"""mcts_v1 — flat-rollout MCTS over candidate launches.

This is "MCTS" in the loose sense: it's a tree of depth 1 (root + per-source
candidates) where each candidate is evaluated by a single forward rollout. No
UCB selection, no expansion, no backprop — just rollout-based action scoring.
This is the pattern used by frizzerdk/orbit-wars-agent and the most pragmatic
form of search for orbit_wars given the 1-second per-turn actTimeout.

Per turn:
  1. For each owned planet, generate top-K candidate launches using
     physical_v4-style frontier-weighted scoring.
  2. For each candidate, deepcopy the world state and forward-simulate
     `HORIZON` turns where:
        - this candidate fires THIS turn from THIS source,
        - every other planet (mine and enemy) plays a fast heuristic
          (a stripped-down physical_v2 that runs purely on the sim state).
  3. Score the rollout = my_total_ships − sum(enemy_total_ships)
     evaluated at the horizon.
  4. Pick, per source, the candidate with the highest score (or skip if
     even the best is worse than holding).

The simulator handles 2- and 4-player games. It models orbiting planets,
fleet movement with the orbit_wars speed formula, sun-collision destruction,
fleet–planet collisions, multi-fleet combat resolution per the env's rules,
and per-turn production. It deliberately does NOT model comet spawns
during rollout — short horizons make this effectively zero-impact.
"""

from __future__ import annotations

import copy
import math
import time

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from ..registry import register

# ---------- Physics constants ----------
SUN_CX = 50.0
SUN_CY = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0
MAX_SPEED = 6.0
SPEED_LOG_DENOM = math.log(1000.0)
ROTATION_RADIUS_LIMIT = 50.0
LEAD_AIM_ITERS = 6

# ---------- MCTS knobs ----------
HORIZON = 5                     # turns rolled out per candidate (short to limit sim drift)
K_CANDIDATES_PER_SOURCE = 2     # top-K candidates evaluated per source
MAX_SOURCES_TO_EVALUATE = 4     # evaluate at most top-N sources by surplus
TIME_BUDGET_S = 0.85            # leave 0.15s margin under 1s actTimeout
SAFETY_BUFFER = 3
DEFENSE_BUFFER = 3
MIN_LAUNCH_SHIPS = 5
NEUTRAL_BONUS = 0.85


# ============================================================
# Physics helpers (shared with physical_v*)
# ============================================================
def fleet_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / SPEED_LOG_DENOM) ** 1.5


def _dist_from_sun(x: float, y: float) -> float:
    return math.hypot(x - SUN_CX, y - SUN_CY)


def crosses_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return _dist_from_sun(x1, y1) <= SUN_RADIUS + SUN_MARGIN
    t = max(0.0, min(1.0, ((SUN_CX - x1) * dx + (SUN_CY - y1) * dy) / len_sq))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(cx - SUN_CX, cy - SUN_CY) <= SUN_RADIUS + SUN_MARGIN


def _seg_hits_circle(x1, y1, x2, y2, cx, cy, r):
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - cx, y1 - cy) <= r
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / len_sq))
    px, py = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy) <= r


def is_orbiting_at(x: float, y: float, radius: float) -> bool:
    return (_dist_from_sun(x, y) + radius) < ROTATION_RADIUS_LIMIT


def predict_position(x: float, y: float, av_signed: float, turns: float):
    dx, dy = x - SUN_CX, y - SUN_CY
    r = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) + av_signed * turns
    return SUN_CX + r * math.cos(angle), SUN_CY + r * math.sin(angle)


def lead_aim(sx, sy, tx, ty, target_radius, fleet_ships, av_signed, target_orbiting):
    speed = fleet_speed(fleet_ships)
    if not target_orbiting or av_signed == 0:
        return tx, ty, math.hypot(tx - sx, ty - sy) / speed
    px, py = tx, ty
    turns = math.hypot(px - sx, py - sy) / speed
    for _ in range(LEAD_AIM_ITERS):
        px, py = predict_position(tx, ty, av_signed, turns)
        turns = math.hypot(px - sx, py - sy) / speed
    return px, py, turns


def fleet_eta_to_planet(fx, fy, fangle, fships, px, py, prad):
    speed = fleet_speed(fships)
    if speed <= 0:
        return None
    ch = math.cos(fangle)
    sh = math.sin(fangle)
    dx = px - fx
    dy = py - fy
    t_closest = (dx * ch + dy * sh) / speed
    if t_closest < 0:
        return None
    cx = fx + speed * t_closest * ch
    cy = fy + speed * t_closest * sh
    if math.hypot(cx - px, cy - py) > prad:
        return None
    return t_closest


def infer_rotation_sign(planets, initial_planets):
    init = {row[0]: row for row in initial_planets}
    for p in planets:
        pid = p[0]
        if pid not in init:
            continue
        ip = init[pid]
        ix, iy = ip[2], ip[3]
        ir = math.hypot(ix - SUN_CX, iy - SUN_CY)
        cr = _dist_from_sun(p[2], p[3])
        if abs(ir - cr) > 0.5:
            continue
        ia = math.atan2(iy - SUN_CY, ix - SUN_CX)
        ca = math.atan2(p[3] - SUN_CY, p[2] - SUN_CX)
        delta = (ca - ia + math.pi) % (2 * math.pi) - math.pi
        if abs(delta) > 1e-3:
            return 1 if delta > 0 else -1
    return 1


# ============================================================
# Lightweight forward simulator
# ============================================================
def _compute_surplus(source, enemy_fleets):
    """v2-style threat-timeline surplus."""
    threats = []
    for f in enemy_fleets:
        eta = fleet_eta_to_planet(f[2], f[3], f[4], f[6], source[2], source[3], source[4])
        if eta is not None:
            threats.append((eta, f[6]))
    if not threats:
        return max(0, source[5] - DEFENSE_BUFFER)
    threats.sort(key=lambda x: x[0])
    garrison = float(source[5])
    min_garrison = garrison
    last_t = 0.0
    for t, ships in threats:
        garrison += source[6] * (t - last_t)
        garrison -= ships
        if garrison < min_garrison:
            min_garrison = garrison
        last_t = t
    return int(max(0, math.floor(min_garrison) - DEFENSE_BUFFER))


def _heuristic_actions(planets, fleets, av, av_signed, player):
    """Stripped-down physical_v2 that operates on raw lists. Used as the
    rollout policy for non-candidate sources and the opponent."""
    my_planets = [p for p in planets if p[1] == player and p[5] >= MIN_LAUNCH_SHIPS]
    targets = [p for p in planets if p[1] != player]
    enemy_fleets = [f for f in fleets if f[1] != player and f[1] >= 0]
    if not my_planets or not targets:
        return []

    moves = []
    for source in my_planets:
        surplus = _compute_surplus(source, enemy_fleets)
        if surplus < MIN_LAUNCH_SHIPS:
            continue

        best_score = float("inf")
        best_angle = 0.0
        best_ships = 0

        for target in targets:
            t_orbiting = av != 0 and is_orbiting_at(target[2], target[3], target[4])
            fleet_guess = min(max(target[5] + 10, 20), surplus)
            if fleet_guess < 1:
                continue
            px, py, turns = lead_aim(
                source[2], source[3], target[2], target[3], target[4],
                fleet_guess, av_signed, t_orbiting,
            )
            if crosses_sun(source[2], source[3], px, py):
                continue
            if target[1] == -1:
                ships_on_arrival = target[5]
            else:
                ships_on_arrival = target[5] + int(target[6] * turns)
            ships_needed = ships_on_arrival + SAFETY_BUFFER
            if ships_needed > surplus:
                continue
            base = turns / max(1, target[6])
            sc = base * (NEUTRAL_BONUS if target[1] == -1 else 1.0)
            if sc < best_score:
                best_score = sc
                best_angle = math.atan2(py - source[3], px - source[2])
                best_ships = ships_needed

        if best_ships > 0:
            moves.append([source[0], best_angle, best_ships])
    return moves


def _resolve_combat(planet, attackers):
    """Apply orbit_wars combat rules at one planet.

    attackers: list of (owner, ships) — incoming fleets only, NOT defender.
    Mutates planet [id, owner, x, y, radius, ships, prod] in place.
    """
    if not attackers:
        return
    incoming = {}
    for owner, ships in attackers:
        incoming[owner] = incoming.get(owner, 0) + ships
    sorted_in = sorted(incoming.items(), key=lambda x: -x[1])
    if len(sorted_in) >= 2 and sorted_in[0][1] == sorted_in[1][1]:
        # Top tie → all incoming annihilate, garrison untouched.
        return
    if len(sorted_in) == 1:
        survivor_owner, survivor_ships = sorted_in[0]
    else:
        survivor_owner = sorted_in[0][0]
        survivor_ships = sorted_in[0][1] - sorted_in[1][1]
    if survivor_owner == planet[1]:
        planet[5] += survivor_ships
    else:
        if survivor_ships > planet[5]:
            planet[1] = survivor_owner
            planet[5] = survivor_ships - planet[5]
        else:
            planet[5] -= survivor_ships


def _step(planets, fleets, av, av_signed, actions_per_player, next_fid):
    # 1. Apply launches
    for player, action_list in actions_per_player.items():
        for move in action_list:
            if not move or len(move) < 3:
                continue
            from_id, angle, ships = move[0], float(move[1]), int(move[2])
            src = next((p for p in planets if p[0] == from_id), None)
            if src is None or src[1] != player or ships < 1 or src[5] < ships:
                continue
            src[5] -= ships
            fx = src[2] + (src[4] + 0.1) * math.cos(angle)
            fy = src[3] + (src[4] + 0.1) * math.sin(angle)
            fleets.append([next_fid, player, fx, fy, angle, from_id, ships])
            next_fid += 1

    # 2. Production (after launches; matches dylan's "production fires same turn as launch")
    for p in planets:
        if p[1] >= 0:
            p[5] += p[6]

    # 3. Move orbiting planets
    if av != 0:
        for p in planets:
            r = math.hypot(p[2] - SUN_CX, p[3] - SUN_CY)
            if r + p[4] < ROTATION_RADIUS_LIMIT:
                ang = math.atan2(p[3] - SUN_CY, p[2] - SUN_CX) + av_signed
                p[2] = SUN_CX + r * math.cos(ang)
                p[3] = SUN_CY + r * math.sin(ang)

    # 4. Move fleets, detect collisions
    survivors = []
    arrivals: dict[int, list[tuple[int, int]]] = {}
    for f in fleets:
        owner = f[1]
        fx, fy, angle, fships = f[2], f[3], f[4], f[6]
        speed = fleet_speed(fships)
        nx = fx + speed * math.cos(angle)
        ny = fy + speed * math.sin(angle)
        if crosses_sun(fx, fy, nx, ny):
            continue  # destroyed
        collided = None
        for p in planets:
            if p[0] == f[5] and math.hypot(fx - p[2], fy - p[3]) < p[4] + 0.5:
                continue  # don't collide with the source we just left
            if _seg_hits_circle(fx, fy, nx, ny, p[2], p[3], p[4]):
                collided = p[0]
                break
        if collided is not None:
            arrivals.setdefault(collided, []).append((owner, fships))
        else:
            f[2] = nx
            f[3] = ny
            survivors.append(f)
    fleets[:] = survivors

    # 5. Combat
    for pid, atks in arrivals.items():
        planet = next(p for p in planets if p[0] == pid)
        _resolve_combat(planet, atks)

    return next_fid


def _ship_score(planets, fleets, my_player, num_players):
    """Score = my_total - max(enemy_totals). Higher = better for me."""
    totals = [0] * num_players
    for p in planets:
        if 0 <= p[1] < num_players:
            totals[p[1]] += p[5]
    for f in fleets:
        if 0 <= f[1] < num_players:
            totals[f[1]] += f[6]
    my = totals[my_player]
    enemies = [totals[i] for i in range(num_players) if i != my_player]
    return my - max(enemies) if enemies else my


def _rollout(planets, fleets, av, av_signed, my_player, num_players,
             my_action, horizon, next_fid):
    """Forward-simulate `horizon` turns. Returns ship-delta score at horizon."""
    p = [list(x) for x in planets]
    f = [list(x) for x in fleets]
    fid = next_fid

    for turn in range(horizon):
        actions = {p_idx: _heuristic_actions(p, f, av, av_signed, p_idx)
                   for p_idx in range(num_players)}
        if turn == 0 and my_action:
            # On turn 0, override my candidate's source with the candidate move;
            # other my-planets keep their heuristic moves so the rollout isn't
            # handicapped by one turn of inactivity.
            candidate_sources = {m[0] for m in my_action}
            other = [m for m in actions[my_player] if m[0] not in candidate_sources]
            actions[my_player] = my_action + other
        fid = _step(p, f, av, av_signed, actions, fid)

        # Early termination: only one player owns any planets
        owners = {pp[1] for pp in p if pp[1] >= 0}
        if len(owners) <= 1:
            break

    return _ship_score(p, f, my_player, num_players)


# ============================================================
# Candidate generation (lifted from physical_v4 scoring)
# ============================================================
def _frontier_distance(target, my_planets):
    if not my_planets:
        return float("inf")
    return min(math.hypot(p[2] - target[2], p[3] - target[3]) for p in my_planets)


def _candidates_for_source(source, targets, my_planets, surplus, av, av_signed):
    out = []
    for target in targets:
        t_orbiting = av != 0 and is_orbiting_at(target[2], target[3], target[4])
        fleet_guess = min(max(target[5] + 10, 20), surplus)
        if fleet_guess < 1:
            continue
        px, py, turns = lead_aim(
            source[2], source[3], target[2], target[3], target[4],
            fleet_guess, av_signed, t_orbiting,
        )
        if crosses_sun(source[2], source[3], px, py):
            continue
        if target[1] == -1:
            ships_on_arrival = target[5]
        else:
            ships_on_arrival = target[5] + int(target[6] * turns)
        ships_needed = ships_on_arrival + SAFETY_BUFFER
        if ships_needed > surplus:
            continue
        fd = _frontier_distance(target, my_planets)
        base = turns / max(1, target[6])
        score = base * (1.0 + fd / 50.0)
        if target[1] == -1:
            score *= NEUTRAL_BONUS
        angle = math.atan2(py - source[3], px - source[2])
        out.append((target[0], angle, ships_needed, score))
    out.sort(key=lambda x: x[3])
    return out


# ============================================================
# Agent
# ============================================================
@register(
    "mcts_v1",
    "Flat-rollout MCTS: enumerate top-K candidate launches per source, "
    "forward-simulate H turns with physical_v2 as the rollout policy, "
    "pick the candidate with the best ship-delta at horizon.",
)
def mcts_v1_agent(obs):
    t0 = time.time()
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    raw_planets = list(get("planets") or [])
    raw_fleets = list(get("fleets") or [])
    angular_velocity = abs(float(get("angular_velocity") or 0.0))
    initial_planets = list(get("initial_planets") or [])

    # Convert tuples → mutable lists for the simulator.
    planets = [list(p) for p in raw_planets]
    fleets = [list(f) for f in raw_fleets]
    av_sign = infer_rotation_sign(planets, initial_planets)
    av_signed = angular_velocity * av_sign

    num_players = max((p[1] for p in planets if p[1] >= 0), default=1) + 1
    num_players = max(num_players, 2)

    my_planets = [p for p in planets if p[1] == player and p[5] >= MIN_LAUNCH_SHIPS]
    targets = [p for p in planets if p[1] != player]
    enemy_fleets = [f for f in fleets if f[1] != player and f[1] >= 0]
    if not my_planets or not targets:
        return []

    # Surplus per source, then pick top sources by surplus to keep within budget.
    sources_with_surplus = []
    for source in my_planets:
        surplus = _compute_surplus(source, enemy_fleets)
        if surplus >= MIN_LAUNCH_SHIPS:
            sources_with_surplus.append((source, surplus))
    sources_with_surplus.sort(key=lambda x: -x[1])
    sources_with_surplus = sources_with_surplus[:MAX_SOURCES_TO_EVALUATE]
    if not sources_with_surplus:
        return []

    next_fid = (max((f[0] for f in fleets), default=-1) + 1)

    # Heuristic baseline action for each of my sources (what physical_v2 would emit).
    heuristic_moves = _heuristic_actions(planets, fleets, angular_velocity, av_signed, player)
    heur_by_source = {m[0]: [m[1], m[2]] for m in heuristic_moves}

    moves = []
    # Per-source: rollout-evaluate the heuristic pick + frontier candidates;
    # always emit the winner so the agent never under-acts.
    for source, surplus in sources_with_surplus:
        if time.time() - t0 > TIME_BUDGET_S:
            # No more time — fall through to heuristic for remaining sources.
            if source[0] in heur_by_source:
                ang, sh = heur_by_source[source[0]]
                moves.append([source[0], ang, sh])
            continue

        cands = _candidates_for_source(
            source, targets, my_planets, surplus, angular_velocity, av_signed
        )[:K_CANDIDATES_PER_SOURCE]

        # Always include the heuristic action for this source as a candidate.
        if source[0] in heur_by_source:
            h_ang, h_sh = heur_by_source[source[0]]
            if not any(abs(c[1] - h_ang) < 1e-3 and c[2] == h_sh for c in cands):
                cands.append((source[0], h_ang, h_sh, -1.0))  # heur_score N/A

        if not cands:
            continue

        best_score = float("-inf")
        best_move = None
        for from_id, angle, ships, _ in cands:
            if time.time() - t0 > TIME_BUDGET_S:
                break
            score = _rollout(
                planets, fleets, angular_velocity, av_signed,
                player, num_players,
                my_action=[[from_id, angle, ships]],
                horizon=HORIZON, next_fid=next_fid,
            )
            if score > best_score:
                best_score = score
                best_move = [from_id, angle, ships]

        # Fallback to heuristic if rollouts were all skipped (time budget).
        if best_move is None and source[0] in heur_by_source:
            h_ang, h_sh = heur_by_source[source[0]]
            best_move = [source[0], h_ang, h_sh]

        if best_move is not None:
            moves.append(best_move)

    return moves
