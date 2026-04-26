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
    return 1.0 + 5.0 * (math.log(max(2, ships)) / math.log(1000.0)) ** 1.5


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
