# hybrid_v1 — Dispatcher Strategy

Routes each turn to ONE of `physical_v1`, `physical_v2`, or `physical_v4`.
No voting, no ensembling — `_select_subagent(obs)` picks a single ID and
that sub-agent's registered function generates the move list.

The dispatcher's only job is fast obs inspection (one pass over planets,
one over fleets — O(N) per turn). All heavy logic lives in the delegate.

`physical_v3` is excluded because it is known to lose to `physical_v2` in
head-to-head. `random_v1` and `sniper_v1` are excluded as baselines.

## Dispatch table

Phase boundaries are step indices (0-indexed). Episode is 500 steps.

| Step       | Default       | Override                                          |
| ---------- | ------------- | ------------------------------------------------- |
| 0–19       | `physical_v1` | `physical_v2` if any enemy fleet is inbound       |
| 20–79      | `physical_v4` | —                                                 |
| 80–349     | `physical_v2` | `physical_v4` if `my_planet_count < enemy_count`  |
| 350–474    | `physical_v4` | —                                                 |
| 475+       | `physical_v4` | —                                                 |

## Global overrides (checked BEFORE phase rules)

Computed from total ships across planets + in-flight fleets:

- `my_total_ships > 2.5 × enemy_total_ships` → `physical_v2`
  (decisively ahead — defensive consolidation wins).
- `my_total_ships < 0.5 × enemy_total_ships` → `physical_v4`
  (desperate — coalitions/swarms are the only way back).

Overrides skip when `enemy_ships == 0` (avoid divide-by-zero; phase rule
fires).

## Tuning constants

```python
EARLY_STEP = 20
EARLY_MID_STEP = 80
MID_STEP = 350
LATE_STEP = 475
SHIP_DOMINANCE_RATIO = 2.5
SHIP_DESPERATE_RATIO = 0.5
```

## Why these choices

- **`physical_v1` early**: simpler scoring → marginally faster expansion
  when the map is uncontested. Switches to v2 the moment threats appear.
- **`physical_v4` early-mid**: its `OPENING` phase is tuned for
  contested expansion (`neutral_bonus=0.80`, `frontier_w=1.2`, `eta_tol=1`).
- **`physical_v2` mid**: head-to-head tournament data shows v2 beats
  v3 and is competitive with v4 once positions are set. The threat-aware
  surplus calculation keeps fragile mid-game garrisons alive.
- **`physical_v4` late + very late**: its `LATE` and `VERY_LATE` phase
  tables release defensive holds (`defense_buffer=1` at very-late) for a
  proper racing finish.

## Telemetry

`agent.DISPATCH_COUNTS` is a module-level `collections.Counter` that
records every dispatch. Reset with `DISPATCH_COUNTS.clear()` between
runs if you need a clean slate. Tests print it as a summary.
