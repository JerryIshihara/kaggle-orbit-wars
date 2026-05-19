# physical_v4 — Strategy Summary

A heuristic agent stacking three patterns observed in the top public Kaggle kernels on top of `physical_v2`'s threat-aware foundation. Designed to capture the strategies that currently dominate the public leaderboard without any learned components.

## Lineage

```
sniper          ← greedy nearest, fixed 20 ships
physical_v1     ← physics: lead-aim, sun-dodge, fleet-speed formula
physical_v2     ← + defensive threat accounting (incoming-fleet timeline)
physical_v3     ← v2 + per-source multi-target greedy   ← REGRESSION (loses to v2)
physical_v4     ← v2 + multi-source swarm + frontier scoring + phase tuning
```

`v4` is a *parallel* successor to `v2` (not a successor to `v3`); the per-source multi-target idea from `v3` is intentionally not retained because it lost games against `v2` in earlier benchmarking.

## What it does, in three layers

### Layer 1 — Multi-source coordinated swarm (highest leverage)

**The idea (Pascal's `orbitwork-v14`):** instead of each owned planet picking its own target independently, identify enemy planets where 2–4 of our planets can launch simultaneously *this turn* and have their fleets land within `±ETA_TOLERANCE` turns of each other. The combined arrival overwhelms a defender that could have repelled any single fleet alone.

**Implementation** (`find_coalition`):
1. For each target, compute each potential source's ETA and required angle (with full lead-aim and sun-dodge from v2).
2. Sort sources by ETA. Slide a window of size 2–4 over them; the window is a valid coalition iff `eta_range <= eta_tolerance`.
3. Compute `total_needed` = `target.ships + production·max_eta_in_window + safety + SWARM_EXTRA_MARGIN`. (Use the *latest* ETA in the window so defenders have time to grow.)
4. If `Σ surpluses ≥ total_needed`, allocate proportional ship contributions per source, capped at each source's surplus, with a min of `min_launch` per launching planet.
5. Bigger coalitions are tried first (4 → 3 → 2). The first feasible one wins for that target.

A source is **consumed by at most one coalition** per turn. Sources that don't fit into any coalition fall through to Layer 2.

### Layer 2 — Frontier-distance target scoring

**The idea (ykhnkf's `distance-prioritized-agent`):** rank candidate targets not by their distance from the launching source, but by their distance from *anywhere on our border* — i.e., from the closest of all our owned planets. This biases us toward consolidating outward from our frontier and away from sending fleets across the map to capture stranded distant planets.

**Score formula:**

```
score(target, source) =
    (turns_to_arrive / production)             # base "value per second of effort"
  · (1 + frontier_dist / 50) ** frontier_w      # penalty for being far from any of my planets
  · neutral_bonus  if target.owner == -1        # cheaper neutrals ⇒ lower score
```

Lower score = better target. `frontier_dist` is `min(dist(p, target) for p in my_planets)`. The exponent `frontier_w` is phase-tuned: stronger preference for nearby targets in early/opening phases, more willingness to reach for distant kills in late game.

For coalition target prioritization, we use a simplified version: `frontier_dist / production · neutral_bonus` (without the per-source travel time, since coalitions are *about* the target).

### Layer 3 — Phase-aware constants

**The idea (debugendless's `sun-dodging-baseline`):** the right doctrine changes throughout the game. A single set of constants doesn't fit early-expansion *and* late-game racing simultaneously.

**Phases (function of `step` and `remaining`):**

| Phase | Range | Doctrine |
|---|---|---|
| `EARLY` | step < 40 | Aggressive expansion onto neutrals; thin defenses. Smaller min_launch (4) so we don't sit on ships. |
| `OPENING` | 40 ≤ step < 80 | Contest the frontier. Strong frontier penalty (`frontier_w=1.2`). Defensive buffer rises to 3. |
| `MID` | 80 ≤ step, remaining ≥ 60 | Balanced consolidation. Standard buffer (4). Coalition tolerance loosens to 2 turns. |
| `LATE` | remaining < 60 | Push for ship-count lead. Lower neutral bonus (still attractive but less so), looser coalition tolerance. |
| `VERY_LATE` | remaining < 25 | All-in. Drop defense buffer to 1, ship-fraction to 3. Coalition tolerance up to 3 turns — every fleet possible. |

Every phase has its own `(neutral_bonus, defense_buffer, min_launch, safety_buffer, frontier_w, swarm_eta_tolerance)` tuple in `PHASE_TABLE`, set in one block at the top of the file for easy retuning.

## Reused machinery from `physical_v2`

- `fleet_speed(ships)` — `1 + (max−1)·(log(ships)/log(1000))^1.5`
- `is_orbiting(p)` — `dist_from_sun + radius < ROTATION_RADIUS_LIMIT`
- `crosses_sun(s, t)` — point-to-segment distance ≤ `SUN_RADIUS + SUN_MARGIN`
- `infer_rotation_sign(planets, initial_planets)` — diff initial vs current planet angle
- `lead_aim(s, t, ships, av, orbiting)` — 6-iteration fixed point on the intercept
- `fleet_eta_to_planet(fleet, planet)` — closest-approach time check
- `compute_surplus(source, enemy_fleets, buffer)` — min-garrison timeline simulation; returns ships available for offense without compromising defense

The defensive threat model from `v2` is unchanged — we still walk the timeline of incoming enemy fleets per source and only spend the surplus above min-garrison.

## Per-turn execution order

```
1. Derive phase → fetch phase constants.
2. Compute per-source surplus (= ships safely spendable on offense this turn).
3. Build target priority list (frontier_dist / production / neutral_bonus).
4. STAGE 1 — coalitions: walk targets in priority order; for each, try to assemble
   a 4 → 3 → 2 source coalition. Mark consumed sources.
5. STAGE 2 — independents: each unused source picks its single best target by the
   frontier-weighted score.
6. Emit all launches (coalition + independent) as one move list.
```

## What's intentionally NOT in v4 (and why)

- **Per-source multi-target greedy** (the v3 idea). v3 spreads a source's surplus across multiple targets in one turn; this loses to v2 because the spread fleets often each fall short of a target's defense by themselves. v4 spreads across *sources to one target* instead, which is the correct direction.
- **Bayesian opponent fingerprinting / SPNE** (emanuellcs). Niche, complex, and only useful when facing varied opponents — overkill for our current evaluation against `physical_v2` in self-play.
- **Lanchester ROI calculations** (emanuellcs). The simpler `ships_needed = defended + safety` already does the work.
- **MCTS rollout search** (frizzerdk). Reserved for a future `mcts_v1` agent — pure search adds value but isn't a heuristic refinement.

## Tuning knobs

All in `PHASE_TABLE` and the constants block. No runtime config is exposed. To tune:

1. Pick a phase that's underperforming (look at win-rates split by step bucket).
2. Adjust the relevant tuple entry — usually `frontier_w` or `defense_buffer`.
3. Re-run the eval harness vs `physical_v2`.

The two most impactful knobs are `swarm_eta_tolerance` (looser → more coalitions, but more "missed timing" failures) and `neutral_bonus` (lower → more aggressive neutral grabs, but neglects enemy frontier).

## Measured baseline matchup (initial tournament, 10 games each)

| Opponent | win rate | record |
|---|---|---|
| `random_v1` | 100% | 10-0-0 |
| `sniper_v1` | 100% | 10-0-0 |
| `physical_v1` | 60% | 6-4-0 |
| `physical_v2` | 40% | 4-6-0 |

v4 dominates the simple baselines and edges v1. **v4 loses 4-6 to v2 in this sample** — within the binomial CI of "tied" (95% CI ≈ 14%-73%) but not the clean win we wanted. The first round of tuning should focus on:

1. `SWARM_EXTRA_MARGIN=4` may be too generous → fewer coalitions trigger because they fail the "Σ surpluses ≥ total_needed" check, *or* triggered ones overspend.
2. `frontier_w` of 1.2 in OPENING may bias too hard toward our own border — v2's pure travel-time/production score sometimes correctly picks a soft distant neutral that v4 skips.
3. Phase boundaries (40 / 80 / 60-rem / 25-rem) are guesses; the right values likely differ.

Suggested next steps before declaring v4 the new best heuristic:

- Sweep `SWARM_EXTRA_MARGIN ∈ {0, 2, 4, 6}` at 30-game evals vs v2.
- Try `frontier_w` set to 0.8 in `OPENING` and 0.6 elsewhere (weaker bias).
- Add an ablation flag (`enable_swarm: bool`, `enable_frontier: bool`) to isolate which of the three layers is helping vs hurting.
