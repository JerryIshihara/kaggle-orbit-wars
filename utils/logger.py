"""Post-match logging utilities.

Two features:

1. **Fleet waste ratio** — fraction of launched fleets that did not contribute
   to capturing or reinforcing a planet (annihilated, destroyed by sun, exited
   map, or unresolved). See ``waste_ratio`` and ``format_waste_summary``.

2. **Time-to-target** — for each fleet, the number of turns between launch and
   resolution (whether by collision, sun, or end-of-game). Aggregates per
   player and per outcome. See ``time_to_target`` and ``format_tto_summary``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agents.physics_utils import (
    P_ID,
    P_OWNER,
    P_RADIUS,
    P_SHIPS,
    P_X,
    P_Y,
    _build_comet_lookup,
    _find_first_collision_dynamic,
    _infer_rotation_sign_raw,
    _is_orbiting_xy,
    _lead_aim_comet,
    _lead_aim_orbital,
    _lead_aim_static,
    find_first_collision,
)

# Sun + map constants (mirror the env's geometry).
SUN_CX = 50.0
SUN_CY = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0
BOARD_MIN = 0.0
BOARD_MAX = 100.0


@dataclass
class FleetRecord:
    fleet_id: int
    owner: int
    launch_step: int       # first env.step the fleet appeared in
    end_step: int          # first env.step where it was GONE
    initial_ships: int
    from_id: int           # source planet ID
    target_planet_id: int | None
    outcome: str           # "captured" | "reinforced" | "annihilated"
                           # | "destroyed_sun" | "out_of_map"
                           # | "still_in_flight" | "unknown"

    @property
    def travel_time(self) -> int:
        """Turns between launch and resolution (i.e., end_step - launch_step)."""
        return self.end_step - self.launch_step


@dataclass
class LaunchMotionRecord:
    """One accepted replay launch classified by source/target motion kind.

    Outcomes are sourced from :func:`trace_fleets` (which walks the env's
    fleet lifecycle for each fleet that actually appeared), so the miss
    classification matches the env's ground truth — not a re-simulation.
    Each launch motion record corresponds to exactly one accepted fleet:
    moves the env rejected (e.g., insufficient running ship surplus
    after earlier multi-target launches from the same source) produce no
    record.

    For hits, ``target_planet_id`` and ``actual_hit_id`` are the planet
    the fleet actually collided with per the env trace. For trajectory
    misses (``reason ∈ {"sun", "boundary", "unknown"}``), ``target_planet_id``
    is the planet inferred from the launch angle via
    :func:`_infer_intended_target` because the raw replay only carries
    ``[source, angle, ships]``.

    ``miss`` is trajectory-level, not combat-level: it is True when the
    env says the fleet was removed without landing on the intended planet
    (``destroyed_sun`` → ``"sun"``, ``out_of_map`` → ``"boundary"``,
    ``unknown`` → ``"unknown"``). A fleet that landed but was annihilated
    in combat is NOT a trajectory miss. ``wrong_planet`` flags the case
    where the fleet hit a planet but the lead-aim-inferred intended
    target was a different planet (e.g., the agent aimed past a friendly
    blocker that absorbed the fleet); inference is reliable when the
    agent uses one of the repo's ``shoot_*`` helpers.
    """

    step: int
    owner: int
    from_id: int
    target_planet_id: int | None
    actual_hit_id: int | None
    source_kind: str
    target_kind: str
    category: str
    miss: bool
    reason: str
    ships: int
    angle: float


def _seg_hits_circle(x1, y1, x2, y2, cx, cy, r):
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - cx, y1 - cy) <= r
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / len_sq))
    px, py = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy) <= r


def _crosses_sun(x1, y1, x2, y2):
    return _seg_hits_circle(x1, y1, x2, y2, SUN_CX, SUN_CY, SUN_RADIUS + SUN_MARGIN)


def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    speed = 1.0 + 5.0 * (math.log(max(2, ships)) / math.log(1000.0)) ** 1.5
    return min(speed, 6.0)


_MOTION_KINDS = ("static", "orbit", "comet")


def _wrap_abs_angle(theta: float) -> float:
    return abs((theta + math.pi) % (2.0 * math.pi) - math.pi)


def _get_obs_field(obs: dict, key: str, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def _planet_motion_kind(planet: list | tuple, obs: dict) -> str:
    comet_ids = set(_get_obs_field(obs, "comet_planet_ids", []) or [])
    pid = int(planet[P_ID])
    if pid in comet_ids:
        return "comet"
    angular_velocity = abs(float(_get_obs_field(obs, "angular_velocity", 0.0) or 0.0))
    if _is_orbiting_xy(
        float(planet[P_X]),
        float(planet[P_Y]),
        float(planet[P_RADIUS]),
        angular_velocity,
    ):
        return "orbit"
    return "static"


def _state_action(state) -> list:
    if isinstance(state, dict):
        return state.get("action") or []
    return getattr(state, "action", None) or []


def _infer_intended_target(
    source: list | tuple,
    angle: float,
    ships: int,
    obs: dict,
    owner: int,
) -> tuple[list | tuple | None, str]:
    """Infer intended target from a replay action's angle.

    Orbit Wars actions do not contain target IDs. For diagnostics we choose
    the planet whose motion-aware lead-aim angle best matches the emitted
    angle. This is robust for the agents in this repo because their public
    launch helpers all emit exactly those lead-aim angles.
    """
    planets = list(_get_obs_field(obs, "planets", []) or [])
    if not planets:
        return None, "unknown"

    sx = float(source[P_X])
    sy = float(source[P_Y])
    sid = int(source[P_ID])
    angular_velocity = abs(float(_get_obs_field(obs, "angular_velocity", 0.0) or 0.0))
    initial_planets = list(_get_obs_field(obs, "initial_planets", []) or [])
    av_sign = _infer_rotation_sign_raw(planets, initial_planets)
    av_signed = angular_velocity * av_sign
    comet_lookup = _build_comet_lookup(list(_get_obs_field(obs, "comets", []) or []))
    current_step = int(_get_obs_field(obs, "step", 0) or 0)

    best: tuple[float, float, list | tuple, str] | None = None
    for planet in planets:
        pid = int(planet[P_ID])
        if pid == sid:
            continue
        kind = _planet_motion_kind(planet, obs)
        if kind == "comet":
            lead = _lead_aim_comet(sx, sy, planet, int(ships), comet_lookup)
            if lead is None:
                lead = _lead_aim_static(
                    sx, sy, float(planet[P_X]), float(planet[P_Y]), int(ships),
                )
        elif kind == "orbit":
            lead = _lead_aim_orbital(
                sx,
                sy,
                float(planet[P_X]),
                float(planet[P_Y]),
                int(ships),
                av_signed,
                current_step=current_step,
            )
        else:
            lead = _lead_aim_static(
                sx, sy, float(planet[P_X]), float(planet[P_Y]), int(ships),
            )
        px, py, _eta = lead
        dx = px - sx
        dy = py - sy
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            continue
        aim = math.atan2(dy, dx)
        err = _wrap_abs_angle(aim - angle)
        # Tiny penalty for friendly planets: if a friendly blocker and an
        # enemy target sit on nearly the same ray, the enemy/non-owned target
        # is a better proxy for "intended target". The penalty is small enough
        # that clear reinforcements are still classified correctly.
        friendly_penalty = 0.002 if int(planet[P_OWNER]) == int(owner) else 0.0
        score = err + friendly_penalty
        key = (score, dist)
        if best is None or key < (best[0], best[1]):
            best = (score, dist, planet, kind)

    if best is None:
        return None, "unknown"
    return best[2], best[3]


def _first_collision_for_launch(
    source: list | tuple,
    angle: float,
    ships: int,
    obs: dict,
) -> dict | None:
    planets = list(_get_obs_field(obs, "planets", []) or [])
    angular_velocity = abs(float(_get_obs_field(obs, "angular_velocity", 0.0) or 0.0))
    comet_lookup = _build_comet_lookup(list(_get_obs_field(obs, "comets", []) or []))
    if angular_velocity > 0.0 or comet_lookup:
        av_sign = _infer_rotation_sign_raw(
            planets,
            list(_get_obs_field(obs, "initial_planets", []) or []),
        )
        return _find_first_collision_dynamic(
            float(source[P_X]),
            float(source[P_Y]),
            float(source[P_RADIUS]),
            int(source[P_ID]),
            float(angle),
            int(ships),
            planets,
            angular_velocity=angular_velocity,
            av_signed=angular_velocity * av_sign,
            comet_lookup=comet_lookup,
            current_step=int(_get_obs_field(obs, "step", 0) or 0),
        )
    return find_first_collision(
        float(source[P_X]),
        float(source[P_Y]),
        float(source[P_RADIUS]),
        int(source[P_ID]),
        float(angle),
        int(ships),
        planets,
    )


# Mapping from ``trace_fleets`` outcomes to ``(miss, reason)`` for the
# motion miss-rate aggregator. Combat-loss (``annihilated``) is NOT a
# trajectory miss — the fleet reached the target planet, it just lost
# the fight. ``still_in_flight`` is treated as "ok" (didn't have time
# to resolve before episode end); revisit if we ever want to surface
# stranded fleets as a separate diagnostic.
_FLEET_OUTCOME_TO_MISS: dict[str, tuple[bool, str]] = {
    "captured":         (False, "ok"),
    "reinforced":       (False, "ok"),
    "annihilated":      (False, "ok"),
    "still_in_flight":  (False, "still_in_flight"),
    "destroyed_sun":    (True,  "sun"),
    "out_of_map":       (True,  "boundary"),
    "unknown":          (True,  "unknown"),
}


def trace_launch_motion(env) -> list[LaunchMotionRecord]:
    """Classify accepted replay launches into source→target motion buckets.

    Buckets are the 3×3 matrix:
      static/orbit/comet source → static/orbit/comet inferred target.

    Hit/miss labels come from :func:`trace_fleets`, which walks
    ``env.steps`` to track each fleet's true outcome — matching the env's
    ground truth instead of re-simulating physics. Moves the env rejected
    (no matching fleet ever appeared) are silently skipped, which fixes
    the multi-target-per-source surplus-drain over-count: a coalition
    launch from one source whose later moves the env dropped because the
    source's running ship pool was exhausted won't produce phantom
    records.

    Uses only the replay/env object, not agent-specific logs, so it can
    analyze every player slot in the same replay.
    """
    records: list[LaunchMotionRecord] = []

    # Pre-build ground-truth fleet outcomes. Group by (owner, launch_step,
    # from_id) and pop FIFO per group as we walk moves — coalition
    # launches from the same source on the same turn become a queue.
    # ``trace_fleets.launch_step`` is the first env step a fleet was
    # observed (the step the action was applied INTO), so it matches
    # ``env.steps[step_idx][...].action`` for action at step_idx.
    fleet_records = trace_fleets(env)
    fleet_queue: dict[tuple[int, int, int], list[FleetRecord]] = {}
    for f in fleet_records:
        key = (int(f.owner), int(f.launch_step), int(f.from_id))
        fleet_queue.setdefault(key, []).append(f)
    for q in fleet_queue.values():
        q.sort(key=lambda fr: fr.fleet_id)

    # Kaggle stores each action on the *resulting* step. A move listed at
    # env.steps[t][player].action was chosen from env.steps[t-1]'s observation
    # and then applied during the transition into t. Analyze it against
    # that launch-time observation, not the post-move observation.
    for step_idx, step in enumerate(env.steps):
        if step_idx <= 0 or not step:
            continue
        prev_step = env.steps[step_idx - 1]
        if not prev_step:
            continue
        obs = prev_step[0].observation
        planets = list(_get_obs_field(obs, "planets", []) or [])
        if not planets:
            continue
        raw_by_id = {int(p[P_ID]): p for p in planets}
        for owner, state in enumerate(step):
            for move in _state_action(state):
                if not isinstance(move, (list, tuple)) or len(move) < 3:
                    continue
                try:
                    from_id = int(move[0])
                    angle = float(move[1])
                    ships = int(move[2])
                except (TypeError, ValueError):
                    continue
                if ships <= 0 or not math.isfinite(angle):
                    continue
                source = raw_by_id.get(from_id)
                if source is None:
                    continue
                if int(source[P_OWNER]) != int(owner):
                    continue

                # Look up the corresponding fleet. Multi-target rows from
                # the same source produce a queue keyed by (owner, step,
                # from_id) — pop FIFO so the k-th move maps to the k-th
                # accepted fleet. Moves the env rejected (running ship
                # pool exhausted by earlier moves) have no matching fleet
                # and are skipped: they never actually launched.
                key = (int(owner), int(step_idx), int(from_id))
                queue = fleet_queue.get(key)
                if not queue:
                    continue
                fleet = queue.pop(0)

                source_kind = _planet_motion_kind(source, obs)
                outcome = fleet.outcome
                actual_hit_id: int | None = (
                    int(fleet.target_planet_id)
                    if fleet.target_planet_id is not None else None
                )
                miss, reason = _FLEET_OUTCOME_TO_MISS.get(
                    outcome, (True, "unknown"),
                )

                if not miss and actual_hit_id is not None:
                    # Fleet landed on a planet. Compare against the
                    # lead-aim-inferred intended target; if they differ,
                    # demote to ``wrong_planet`` (the agent aimed past a
                    # blocker that absorbed the fleet).
                    intended, intended_kind = _infer_intended_target(
                        source, angle, ships, obs, owner,
                    )
                    if (
                        intended is not None
                        and int(intended[P_ID]) != actual_hit_id
                    ):
                        target_id = int(intended[P_ID])
                        target_kind = intended_kind
                        miss = True
                        reason = "wrong_planet"
                    else:
                        target = raw_by_id.get(actual_hit_id)
                        target_id = actual_hit_id
                        target_kind = (
                            _planet_motion_kind(target, obs)
                            if target is not None else "static"
                        )
                elif not miss:
                    # ``still_in_flight`` — fleet didn't resolve by episode
                    # end. Bucket against the lead-aim-inferred intended
                    # so we can still attribute it to a motion class.
                    intended, intended_kind = _infer_intended_target(
                        source, angle, ships, obs, owner,
                    )
                    target_id = (
                        int(intended[P_ID]) if intended is not None else None
                    )
                    target_kind = intended_kind
                else:
                    # Trajectory miss: sun / boundary / unknown. Infer the
                    # intended target so the heatmap can still bucket it.
                    intended, intended_kind = _infer_intended_target(
                        source, angle, ships, obs, owner,
                    )
                    target_id = (
                        int(intended[P_ID]) if intended is not None else None
                    )
                    target_kind = intended_kind

                category = f"{source_kind}->{target_kind}"
                records.append(LaunchMotionRecord(
                    step=step_idx - 1,
                    owner=int(owner),
                    from_id=from_id,
                    target_planet_id=target_id,
                    actual_hit_id=actual_hit_id,
                    source_kind=source_kind,
                    target_kind=target_kind,
                    category=category,
                    miss=miss,
                    reason=reason,
                    ships=ships,
                    angle=angle,
                ))
    return records


#: Canonical miss-reason buckets surfaced in the dashboard's reason heatmap.
#: ``trace_launch_motion`` emits exactly these strings (or ``"ok"`` on hits).
LAUNCH_MISS_REASONS: tuple[str, ...] = (
    "wrong_planet", "sun", "boundary", "no_collision",
)


def launch_motion_miss_stats(env) -> dict[int, dict]:
    """Per-player launch miss counts by source→target motion category.

    Returned shape (per player owner key):

      .. code-block:: python

         {
           "total": int, "hit": int, "miss": int, "miss_rate": float,
           # top-level by-reason aggregation across all motion categories.
           "by_reason": {reason: int, ...},
           # per (src→tgt) motion cell aggregations.
           "by_category": {
             "static->orbit": {
               "total": int, "hit": int, "miss": int, "miss_rate": float,
               "by_reason": {reason: int, ...},
             },
             ...
           },
           # cross-tab: reason → (src→tgt) cell counts. The dashboard's
           # per-reason heatmap reads directly from here.
           "by_reason_category": {
             "wrong_planet": {"static->static": int, "static->orbit": int, ...},
             "sun": {...}, "boundary": {...}, "no_collision": {...},
           },
         }
    """
    records = trace_launch_motion(env)
    n_players = len(env.steps[0]) if getattr(env, "steps", None) and env.steps else 0
    categories = [f"{s}->{t}" for s in _MOTION_KINDS for t in _MOTION_KINDS]

    def _empty_player() -> dict:
        return {
            "total": 0,
            "hit": 0,
            "miss": 0,
            "by_reason": {},
            "by_category": {
                cat: {"total": 0, "hit": 0, "miss": 0, "by_reason": {}}
                for cat in categories
            },
            "by_reason_category": {
                reason: {cat: 0 for cat in categories}
                for reason in LAUNCH_MISS_REASONS
            },
        }

    out: dict[int, dict] = {owner: _empty_player() for owner in range(n_players)}

    for r in records:
        d = out.setdefault(r.owner, _empty_player())
        d["total"] += 1
        d["miss" if r.miss else "hit"] += 1
        if r.category not in d["by_category"]:
            d["by_category"][r.category] = {
                "total": 0, "hit": 0, "miss": 0, "by_reason": {},
            }
        c = d["by_category"][r.category]
        c["total"] += 1
        c["miss" if r.miss else "hit"] += 1
        if r.miss:
            c["by_reason"][r.reason] = c["by_reason"].get(r.reason, 0) + 1
            d["by_reason"][r.reason] = d["by_reason"].get(r.reason, 0) + 1
            rc = d["by_reason_category"].setdefault(
                r.reason, {cat: 0 for cat in categories},
            )
            rc[r.category] = rc.get(r.category, 0) + 1

    for d in out.values():
        d["miss_rate"] = d["miss"] / d["total"] if d["total"] else 0.0
        for c in d["by_category"].values():
            c["miss_rate"] = c["miss"] / c["total"] if c["total"] else 0.0
    return out


def trace_fleets(env) -> list[FleetRecord]:
    """Walk env.steps and produce one FleetRecord per fleet that ever appeared."""
    # Map fleet_id → list of (step_idx, fleet_tuple). Fleet tuple format:
    #   (id, owner, x, y, angle, from_id, ships)
    fleets_seen: dict[int, list[tuple[int, tuple]]] = {}
    for step_idx, step in enumerate(env.steps):
        obs = step[0].observation
        for f in obs.get("fleets") or []:
            fleets_seen.setdefault(f[0], []).append((step_idx, tuple(f)))

    records: list[FleetRecord] = []
    n_steps = len(env.steps)

    for fid, history in fleets_seen.items():
        first_step, first_f = history[0]
        last_step, last_f = history[-1]
        owner = first_f[1]
        from_id = first_f[5]
        initial_ships = first_f[6]
        last_x, last_y, last_angle, last_ships = last_f[2], last_f[3], last_f[4], last_f[6]
        end_step = last_step + 1

        if end_step >= n_steps:
            records.append(FleetRecord(
                fid, owner, first_step, end_step, initial_ships, from_id,
                target_planet_id=None, outcome="still_in_flight",
            ))
            continue

        # The segment between last_step's position and where the fleet would have
        # been on end_step reveals what it collided with.
        speed = _fleet_speed(last_ships)
        next_x = last_x + speed * math.cos(last_angle)
        next_y = last_y + speed * math.sin(last_angle)

        # Sun?
        if _crosses_sun(last_x, last_y, next_x, next_y):
            records.append(FleetRecord(
                fid, owner, first_step, end_step, initial_ships, from_id,
                target_planet_id=None, outcome="destroyed_sun",
            ))
            continue

        # Find first planet whose disc the segment crossed. Orbiting planets
        # move 1–3 units per step, so we must check their positions at BOTH
        # the start and end of the transition: a fleet that legitimately
        # collided with an orbiting planet will only intersect one of those.
        # Also extend the planet's effective radius slightly to absorb the
        # planet's own swept-arc length during the step.
        obs_end = env.steps[end_step][0].observation
        obs_before = env.steps[last_step][0].observation
        planets_end = obs_end.get("planets") or []
        planets_before = obs_before.get("planets") or []
        before_by_id = {p[0]: p for p in planets_before}
        end_by_id = {p[0]: p for p in planets_end}

        target_id = None
        target_position_planet = None  # the (x,y,r) we used for the hit test
        for pid in set(before_by_id) | set(end_by_id):
            pb = before_by_id.get(pid)
            pe = end_by_id.get(pid)
            # Estimate orbital sweep: distance between start and end positions.
            sweep = 0.0
            if pb and pe:
                sweep = math.hypot(pe[2] - pb[2], pe[3] - pb[3])
            for p in (pb, pe):
                if p is None:
                    continue
                # Inflate the disc by the orbital sweep so we catch fleets
                # whose collision point falls between the start- and
                # end-of-step planet positions.
                if _seg_hits_circle(
                    last_x, last_y, next_x, next_y,
                    p[2], p[3], p[4] + sweep,
                ):
                    target_id = pid
                    target_position_planet = p
                    break
            if target_id is not None:
                break

        if target_id is None:
            outside = not (
                BOARD_MIN <= next_x <= BOARD_MAX and BOARD_MIN <= next_y <= BOARD_MAX
            )
            outcome = "out_of_map" if outside else "unknown"
            records.append(FleetRecord(
                fid, owner, first_step, end_step, initial_ships, from_id,
                target_planet_id=None, outcome=outcome,
            ))
            continue

        # Compare planet state before/after the resolution to classify outcome.
        before = before_by_id.get(target_id)
        after = end_by_id.get(target_id)

        outcome = "unknown"
        if before is not None and after is not None:
            before_owner, before_ships, before_prod = before[1], before[5], before[6]
            after_owner, after_ships = after[1], after[5]
            if after_owner == owner and before_owner != owner:
                outcome = "captured"
            elif after_owner == owner and before_owner == owner:
                # Reinforcement: ships should exceed what production alone would yield.
                expected_no_fleet = before_ships + before_prod  # one turn of production
                if after_ships > expected_no_fleet:
                    outcome = "reinforced"
                else:
                    # Owner kept the planet but our fleet was absorbed by an
                    # opposing inbound (rare); still useful from a defense pov,
                    # but ships were ultimately consumed without growing the
                    # garrison. Conservative: count as annihilated.
                    outcome = "annihilated"
            else:
                outcome = "annihilated"

        records.append(FleetRecord(
            fid, owner, first_step, end_step, initial_ships, from_id,
            target_planet_id=target_id, outcome=outcome,
        ))

    return records


_USEFUL_OUTCOMES = {"captured", "reinforced"}


def waste_ratio(env) -> dict[int, dict]:
    """Per-owner fleet-waste summary.

    Returns a dict keyed by player slot, each value is::

        {
            "total":            <fleets launched>,
            "useful":           <captures + reinforcements>,
            "useless":          <annihilated + sun + out_of_map + unknown>,
            "in_flight":        <fleets still moving when env ended>,
            "ships_total":      <sum of initial_ships over all launches>,
            "ships_wasted":     <ships in useless fleets>,
            "by_outcome":       {outcome: count, ...},
            "waste_ratio":      useless / (total - in_flight),
            "ship_waste_ratio": ships_wasted / ships_total,
        }
    """
    records = trace_fleets(env)
    out: dict[int, dict] = {}

    for r in records:
        d = out.setdefault(r.owner, {
            "total": 0,
            "useful": 0,
            "useless": 0,
            "in_flight": 0,
            "ships_total": 0,
            "ships_wasted": 0,
            "by_outcome": {},
        })
        d["total"] += 1
        d["ships_total"] += r.initial_ships
        d["by_outcome"][r.outcome] = d["by_outcome"].get(r.outcome, 0) + 1

        if r.outcome in _USEFUL_OUTCOMES:
            d["useful"] += 1
        elif r.outcome == "still_in_flight":
            d["in_flight"] += 1
        else:
            d["useless"] += 1
            d["ships_wasted"] += r.initial_ships

    for d in out.values():
        resolved = d["total"] - d["in_flight"]
        d["waste_ratio"] = d["useless"] / resolved if resolved > 0 else 0.0
        d["ship_waste_ratio"] = (
            d["ships_wasted"] / d["ships_total"] if d["ships_total"] > 0 else 0.0
        )

    return out


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def time_to_target(env, include_in_flight: bool = False) -> dict[int, dict]:
    """Per-owner time-to-target statistics.

    Returns a dict keyed by player slot. Each value::

        {
            "n_fleets":         <fleets considered>,
            "n_ships":          <total ships in those fleets>,
            "tt_mean":          <mean travel time, turns>,
            "tt_median":        <median travel time>,
            "tt_p25":           <25th percentile>,
            "tt_p75":           <75th percentile>,
            "tt_min":           <min>,
            "tt_max":           <max>,
            "tt_ship_weighted": <Σ(ships * tt) / Σ(ships) — average turn a ship spends in transit>,
            "by_outcome":       {outcome: {"n_fleets": int, "tt_mean": float, "tt_median": float}, ...},
        }

    `include_in_flight=False` (default) excludes fleets still moving when the
    game ended (they don't have a real "time to target"). Set True to include
    them with their elapsed-so-far value.
    """
    records = trace_fleets(env)
    by_owner: dict[int, list] = {}
    for r in records:
        if not include_in_flight and r.outcome == "still_in_flight":
            continue
        by_owner.setdefault(r.owner, []).append(r)

    out: dict[int, dict] = {}
    for owner, recs in by_owner.items():
        tts = sorted(r.travel_time for r in recs)
        ships_total = sum(r.initial_ships for r in recs)
        ship_weighted = (
            sum(r.initial_ships * r.travel_time for r in recs) / ships_total
            if ships_total > 0 else 0.0
        )

        # Per-outcome breakdown
        per_outcome: dict[str, list[int]] = {}
        for r in recs:
            per_outcome.setdefault(r.outcome, []).append(r.travel_time)
        outcome_stats = {}
        for outcome, vals in per_outcome.items():
            vs = sorted(vals)
            outcome_stats[outcome] = {
                "n_fleets": len(vs),
                "tt_mean": sum(vs) / len(vs),
                "tt_median": _percentile(vs, 0.5),
            }

        out[owner] = {
            "n_fleets": len(recs),
            "n_ships": ships_total,
            "tt_mean": sum(tts) / len(tts) if tts else 0.0,
            "tt_median": _percentile(tts, 0.5),
            "tt_p25": _percentile(tts, 0.25),
            "tt_p75": _percentile(tts, 0.75),
            "tt_min": tts[0] if tts else 0,
            "tt_max": tts[-1] if tts else 0,
            "tt_ship_weighted": ship_weighted,
            "by_outcome": outcome_stats,
        }
    return out


def format_tto_summary(env, agent_ids: list[str] | None = None,
                       include_in_flight: bool = False) -> str:
    """Pretty-printable per-player time-to-target summary."""
    stats = time_to_target(env, include_in_flight=include_in_flight)
    lines = []
    for owner in sorted(stats):
        label = agent_ids[owner] if agent_ids and owner < len(agent_ids) else f"p{owner}"
        d = stats[owner]
        outcome_bits = []
        # Show top 3 outcomes by count.
        for outcome, od in sorted(d["by_outcome"].items(),
                                  key=lambda kv: -kv[1]["n_fleets"])[:3]:
            outcome_bits.append(f"{outcome}: n={od['n_fleets']} med={od['tt_median']:.1f}")
        outcome_str = " | ".join(outcome_bits)
        lines.append(
            f"  {label:<14} "
            f"n={d['n_fleets']:>3}  ships={d['n_ships']:>5}  "
            f"tt_mean={d['tt_mean']:5.1f}  med={d['tt_median']:5.1f}  "
            f"p25={d['tt_p25']:4.1f}  p75={d['tt_p75']:5.1f}  "
            f"min={d['tt_min']:>2}  max={d['tt_max']:>3}  "
            f"ship_w={d['tt_ship_weighted']:5.1f}  "
            f"[{outcome_str}]"
        )
    return "\n".join(lines)


def format_waste_summary(env, agent_ids: list[str] | None = None) -> str:
    """Pretty-printable per-player waste summary."""
    stats = waste_ratio(env)
    lines = []
    for owner in sorted(stats):
        label = agent_ids[owner] if agent_ids and owner < len(agent_ids) else f"p{owner}"
        d = stats[owner]
        outcomes = ", ".join(f"{k}={v}" for k, v in sorted(d["by_outcome"].items()))
        lines.append(
            f"  {label:<14} "
            f"fleets={d['total']:>3}  "
            f"useful={d['useful']:>3}  useless={d['useless']:>3}  "
            f"flight={d['in_flight']:>2}  "
            f"waste={d['waste_ratio']:.1%} "
            f"ship_waste={d['ship_waste_ratio']:.1%}  "
            f"[{outcomes}]"
        )
    return "\n".join(lines)
