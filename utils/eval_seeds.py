"""Stratified 128-seed eval panel for Orbit Wars A/B testing.

128 seeds across 32 game-shape archetypes (4 production bins × 4
rotating-share bins × 2 size-split bins = 32 cells, 4 seeds per cell).
Source: community share by chrisleitescha on the Kaggle Orbit Wars
forum — see ``docs/EVAL_SEEDS.md`` for the methodology, sampling
rationale, and preview-image link.

Two recommended habits when running the panel:

  1. Play **both seats** for every seed. Otherwise a change that
     helps when you're seat 0 (home-planet bottom-left) but hurts
     when you're seat 1 can look like a net win.
  2. Aggregate **per archetype**, not just overall. A net-winrate
     improvement can hide a regression on a specific board class
     (e.g., your change helps on high-production games but loses
     consistently on low-production rotating boards).

``SEEDS`` is interleaved by archetype: the first 32 entries cover
all 32 cells once, the first 64 entries cover each cell twice, and so
on. So for a quick balanced A/B take ``SEEDS[:32]`` and still get
balanced geometry coverage.

Notes from the original share:

  * Built against the simulator in ``data/main.py``. Seeds in
    ``range(0, 10_000)`` are reproducible there.
  * 2-player only. 4-player games sample the same initial-planet
    RNG, so geometry archetypes still apply, but the 4P-specific
    dynamics (3 opponents, scoring) aren't covered.
  * Archetype labels are human-readable bin names — nothing
    magical; re-bin however you like for your own analysis.
"""

from __future__ import annotations

#: Interleaved 128-seed list. ``SEEDS[:32]`` already covers all 32
#: archetypes once; each subsequent block of 32 is a fresh sweep with
#: a different seed per cell. Order is preserved exactly as published.
SEEDS: tuple[int, ...] = (
    # Block 1 (32 seeds, one per archetype):
    5199, 2083, 3493, 1649, 3233,  405, 3335, 1030, 1467,   78,   32, 1900,  647,  417,    1, 2560,
     272,  585, 1265,  741,  489, 2537,  422,  787,  455,  324,  119,  828, 1049,  906, 1117, 1990,
    # Block 2:
    5274, 2661, 3774, 2794, 3578, 7045, 4333, 1153, 2412, 1750, 2078, 2957, 1843,  451, 1725, 4676,
     662, 1217, 4461, 4785, 5675, 3403, 4814, 1336, 2996, 2509, 3959, 2867, 2572, 2476, 1282, 2393,
    # Block 3:
    7542, 6328, 5923, 4252, 4027, 9408, 4693, 2726, 3154, 7166, 6858, 4393, 2177, 2663, 1948, 6475,
    4353, 6462, 5981, 8516, 7770, 6593, 4963, 3473, 3520, 7337, 8763, 9017, 6202, 5047, 1571, 6294,
    # Block 4:
    8909, 9327, 8514, 9240, 6548, 9658, 9812, 8686, 8717, 8866, 7079, 9306, 7314, 8782, 2419, 8526,
    9118, 9984, 8855, 9227, 8624, 8968, 7186, 9013, 4633, 8603, 9102, 9547, 9453, 7947, 3462, 9427,
)
assert len(SEEDS) == 128, f"expected 128 seeds, got {len(SEEDS)}"


#: Per-archetype 4-seed bucket. Archetype key encodes the three axes:
#: ``<prod_bin>__<rotating_share>__<size_split>`` where
#:
#:   prod_bin       ∈ {low_prod, med_low_prod, med_high_prod, high_prod}
#:   rotating_share ∈ {mostly_static, mixed_static, mixed_rotating, mostly_rotating}
#:   size_split     ∈ {big_static, big_rotating}
#:
#: 4 × 4 × 2 = 32 cells, 4 seeds per cell. Every seed appears exactly
#: once across the dict; their union equals ``set(SEEDS)``.
BY_ARCHETYPE: dict[str, list[int]] = {
    "low_prod__mostly_static__big_static":          [2560, 4676, 6475, 8526],
    "low_prod__mostly_static__big_rotating":        [   1, 1725, 1948, 2419],
    "low_prod__mixed_static__big_static":           [1900, 2957, 4393, 9306],
    "low_prod__mixed_static__big_rotating":         [  32, 2078, 6858, 7079],
    "low_prod__mixed_rotating__big_static":         [  78, 1750, 7166, 8866],
    "low_prod__mixed_rotating__big_rotating":       [1467, 2412, 3154, 8717],
    "low_prod__mostly_rotating__big_static":        [ 417,  451, 2663, 8782],
    "low_prod__mostly_rotating__big_rotating":      [ 647, 1843, 2177, 7314],
    "med_low_prod__mostly_static__big_static":      [1990, 2393, 6294, 9427],
    "med_low_prod__mostly_static__big_rotating":    [1117, 1282, 1571, 3462],
    "med_low_prod__mixed_static__big_static":       [ 828, 2867, 9017, 9547],
    "med_low_prod__mixed_static__big_rotating":     [ 119, 3959, 8763, 9102],
    "med_low_prod__mixed_rotating__big_static":     [ 324, 2509, 7337, 8603],
    "med_low_prod__mixed_rotating__big_rotating":   [ 455, 2996, 3520, 4633],
    "med_low_prod__mostly_rotating__big_static":    [ 906, 2476, 5047, 7947],
    "med_low_prod__mostly_rotating__big_rotating":  [1049, 2572, 6202, 9453],
    "med_high_prod__mostly_static__big_static":     [ 787, 1336, 3473, 9013],
    "med_high_prod__mostly_static__big_rotating":   [ 422, 4814, 4963, 7186],
    "med_high_prod__mixed_static__big_static":      [ 741, 4785, 8516, 9227],
    "med_high_prod__mixed_static__big_rotating":    [1265, 4461, 5981, 8855],
    "med_high_prod__mixed_rotating__big_static":    [ 585, 1217, 6462, 9984],
    "med_high_prod__mixed_rotating__big_rotating":  [ 272,  662, 4353, 9118],
    "med_high_prod__mostly_rotating__big_static":   [2537, 3403, 6593, 8968],
    "med_high_prod__mostly_rotating__big_rotating": [ 489, 5675, 7770, 8624],
    "high_prod__mostly_static__big_static":         [1030, 1153, 2726, 8686],
    "high_prod__mostly_static__big_rotating":       [3335, 4333, 4693, 9812],
    "high_prod__mixed_static__big_static":          [1649, 2794, 4252, 9240],
    "high_prod__mixed_static__big_rotating":        [3493, 3774, 5923, 8514],
    "high_prod__mixed_rotating__big_static":        [2083, 2661, 6328, 9327],
    "high_prod__mixed_rotating__big_rotating":      [5199, 5274, 7542, 8909],
    "high_prod__mostly_rotating__big_static":       [ 405, 7045, 9408, 9658],
    "high_prod__mostly_rotating__big_rotating":     [3233, 3578, 4027, 6548],
}
assert len(BY_ARCHETYPE) == 32
assert sum(len(v) for v in BY_ARCHETYPE.values()) == 128
# Every BY_ARCHETYPE entry must also appear in SEEDS exactly once.
_flat = [s for seeds in BY_ARCHETYPE.values() for s in seeds]
assert sorted(_flat) == sorted(SEEDS), "BY_ARCHETYPE seeds don't match SEEDS"


#: 32-entry preview slice — first sweep through all archetypes.
SEEDS_QUICK: tuple[int, ...] = SEEDS[:32]


def archetype_for_seed(seed: int) -> str | None:
    """Return the archetype label for ``seed``, or ``None`` if not in
    the panel. O(32) — fine for the eval-panel volume."""
    for name, seeds in BY_ARCHETYPE.items():
        if seed in seeds:
            return name
    return None
