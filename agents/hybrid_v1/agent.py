"""Hybrid dispatcher agent v1.

Inspects the current observation each turn and delegates the action to ONE
of the proven physical agents (`physical_v1`, `physical_v2`, `physical_v4`).
No ensembling — exactly one sub-agent runs per turn. Dispatch is O(planets +
fleets) per turn; the actual heavy work happens inside the chosen sub-agent.

Game-phase rules (see ``STRATEGY.md`` for full rationale):

  step <  20    → v1 fast expansion, BUT v2 if any enemy fleets already inbound.
  20-79         → v4 (phase tuning + frontier scoring + swarm coordination).
  80-349        → v2 mid-game defensive consolidation;
                  v4 if behind in planet count.
  350-474       → v4 LATE phase racing finish.
  >= 475        → v4 VERY_LATE phase (defensive holds released).

Two global overrides (checked first):
  my_total_ships > 2.5 * enemy_total_ships  → v2 (consolidate the lead).
  my_total_ships < 0.5 * enemy_total_ships  → v4 (desperate, swarms).

A class-level counter records how often each sub-agent fires across an
entire run. Tests print this summary at the end.
"""

from __future__ import annotations

from collections import Counter

from ..registry import register

# ---------- Dispatch tuning constants ----------
EARLY_STEP = 20
EARLY_MID_STEP = 80
MID_STEP = 350
LATE_STEP = 475
SHIP_DOMINANCE_RATIO = 2.5
SHIP_DESPERATE_RATIO = 0.5

# ---------- Telemetry ----------
DISPATCH_COUNTS: Counter = Counter()


def _get(obs, key, default=None):
    """Read a field from a dict-like or attribute-style obs."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _select_subagent(obs) -> str:
    """Decide which physical_v* sub-agent should handle this turn."""
    player = _get(obs, "player", 0)
    step = int(_get(obs, "step") or 0)
    planets = _get(obs, "planets") or []
    fleets = _get(obs, "fleets") or []

    # Aggregate ship and planet counts in a single pass each.
    # planet schema: [id, owner, x, y, radius, ships, production]
    my_ships = enemy_ships = 0
    my_planets = enemy_planets = 0
    for p in planets:
        owner = p[1]
        ships = p[5]
        if owner == player:
            my_ships += ships
            my_planets += 1
        elif owner >= 0:
            enemy_ships += ships
            enemy_planets += 1

    # fleet schema: [id, owner, x, y, angle, ttl?, ships, ...]; ships is index 6.
    enemy_inbound = 0
    for f in fleets:
        owner = f[1]
        if owner == player:
            my_ships += f[6]
        elif owner >= 0:
            enemy_ships += f[6]
            enemy_inbound += 1

    # ----- Global overrides on ship dominance -----
    if enemy_ships > 0:
        ratio = my_ships / enemy_ships
        if ratio > SHIP_DOMINANCE_RATIO:
            return "physical_v2"  # consolidate the lead
        if ratio < SHIP_DESPERATE_RATIO:
            return "physical_v4"  # desperate, need swarms

    # ----- Phase-based selection -----
    if step < EARLY_STEP:
        # Very early: raw expansion. Switch to v2 if anyone is already shooting.
        return "physical_v2" if enemy_inbound > 0 else "physical_v1"
    if step < EARLY_MID_STEP:
        return "physical_v4"  # contested expansion; phase tuning + frontier scoring
    if step < MID_STEP:
        # Mid-game defensive consolidation; v4 if behind on planet count.
        if my_planets < enemy_planets:
            return "physical_v4"
        return "physical_v2"
    if step < LATE_STEP:
        return "physical_v4"  # LATE phase tuning kicks in here
    return "physical_v4"  # VERY_LATE phase tuning kicks in here


@register(
    "hybrid_v1",
    "Dispatcher: routes each turn to physical_v1 / physical_v2 / physical_v4 "
    "based on game phase and ship/planet ratios. No ensembling.",
)
def hybrid_v1_agent(obs):
    # Lazy import to avoid circular dependency: agents/__init__.py imports
    # this module and would not yet have ``Agent`` defined when this file
    # was first imported.
    from agents import Agent

    sub_id = _select_subagent(obs)
    DISPATCH_COUNTS[sub_id] += 1
    return Agent(id=sub_id).fn(obs)
