"""Post-match logging utilities.

Currently exposes a single feature: **fleet waste ratio** — the fraction of
fleets a player launched that did not contribute to capturing or reinforcing
a planet (i.e., were annihilated in combat, destroyed by the sun, or sailed
off the map).

Usage:

    from utils import run_match
    from utils.logger import waste_ratio

    r = run_match(["physical_v4", "physical_v2"], seed=0)
    stats = waste_ratio(r.env)
    for owner, d in stats.items():
        print(f"player {owner}: {d['useless']}/{d['total']} wasted "
              f"= {d['waste_ratio']:.1%}  (ships: {d['ship_waste_ratio']:.1%})")
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

        # Find first planet whose disc the segment crossed (use planet positions
        # from the END-step obs, since orbiting planets move per step).
        obs_end = env.steps[end_step][0].observation
        planets_end = obs_end.get("planets") or []
        target = None
        for p in planets_end:
            pid, _, px, py, prad, _, _ = p
            if _seg_hits_circle(last_x, last_y, next_x, next_y, px, py, prad):
                target = p
                break

        if target is None:
            # No planet hit. Out of map?
            outside = not (
                BOARD_MIN <= next_x <= BOARD_MAX and BOARD_MIN <= next_y <= BOARD_MAX
            )
            outcome = "out_of_map" if outside else "unknown"
            records.append(FleetRecord(
                fid, owner, first_step, end_step, initial_ships, from_id,
                target_planet_id=None, outcome=outcome,
            ))
            continue

        target_id = target[0]
        # Compare planet state before/after the resolution to classify outcome.
        obs_before = env.steps[last_step][0].observation
        before = next(
            (pp for pp in obs_before.get("planets") or [] if pp[0] == target_id),
            None,
        )
        after = next((pp for pp in planets_end if pp[0] == target_id), None)

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
