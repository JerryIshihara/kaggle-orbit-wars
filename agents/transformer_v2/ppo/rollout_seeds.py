"""Curated archetype-stratified game seeds for PPO rollout.

128 orbit_wars map seeds covering **32 archetypes × 4 seeds each**, spanning:
  - production level: low / med_low / med_high / high
  - planet motion:    mostly_static / mixed_static / mixed_rotating / mostly_rotating
  - big-planet:       big_static / big_rotating

Using this stratified set instead of random seeds gives the rollout BALANCED
coverage of the game space (no archetype over/under-sampled), so the PPO update
sees a representative distribution. The seed picks the MAP; the player count
(2P/4P) is chosen separately by the rollout. ``SEEDS`` is interleaved across
archetypes, so any contiguous slice already spans many archetypes.

Use from the rollout via ``rollout_worker --seed-set archetype`` (cycles SEEDS by
the global game index); see ``_episode_specs``.
"""
from __future__ import annotations

SEEDS = [
    5199, 2083, 3493, 1649, 3233,  405, 3335, 1030, 1467,   78,   32, 1900,  647,  417,    1, 2560,
     272,  585, 1265,  741,  489, 2537,  422,  787,  455,  324,  119,  828, 1049,  906, 1117, 1990,
    5274, 2661, 3774, 2794, 3578, 7045, 4333, 1153, 2412, 1750, 2078, 2957, 1843,  451, 1725, 4676,
     662, 1217, 4461, 4785, 5675, 3403, 4814, 1336, 2996, 2509, 3959, 2867, 2572, 2476, 1282, 2393,
    7542, 6328, 5923, 4252, 4027, 9408, 4693, 2726, 3154, 7166, 6858, 4393, 2177, 2663, 1948, 6475,
    4353, 6462, 5981, 8516, 7770, 6593, 4963, 3473, 3520, 7337, 8763, 9017, 6202, 5047, 1571, 6294,
    8909, 9327, 8514, 9240, 6548, 9658, 9812, 8686, 8717, 8866, 7079, 9306, 7314, 8782, 2419, 8526,
    9118, 9984, 8855, 9227, 8624, 8968, 7186, 9013, 4633, 8603, 9102, 9547, 9453, 7947, 3462, 9427,
]

BY_ARCHETYPE = {
    "low_prod__mostly_static__big_static":         [2560, 4676, 6475, 8526],
    "low_prod__mostly_static__big_rotating":       [   1, 1725, 1948, 2419],
    "low_prod__mixed_static__big_static":          [1900, 2957, 4393, 9306],
    "low_prod__mixed_static__big_rotating":        [  32, 2078, 6858, 7079],
    "low_prod__mixed_rotating__big_static":        [  78, 1750, 7166, 8866],
    "low_prod__mixed_rotating__big_rotating":      [1467, 2412, 3154, 8717],
    "low_prod__mostly_rotating__big_static":       [ 417,  451, 2663, 8782],
    "low_prod__mostly_rotating__big_rotating":     [ 647, 1843, 2177, 7314],
    "med_low_prod__mostly_static__big_static":     [1990, 2393, 6294, 9427],
    "med_low_prod__mostly_static__big_rotating":   [1117, 1282, 1571, 3462],
    "med_low_prod__mixed_static__big_static":      [ 828, 2867, 9017, 9547],
    "med_low_prod__mixed_static__big_rotating":    [ 119, 3959, 8763, 9102],
    "med_low_prod__mixed_rotating__big_static":    [ 324, 2509, 7337, 8603],
    "med_low_prod__mixed_rotating__big_rotating":  [ 455, 2996, 3520, 4633],
    "med_low_prod__mostly_rotating__big_static":   [ 906, 2476, 5047, 7947],
    "med_low_prod__mostly_rotating__big_rotating": [1049, 2572, 6202, 9453],
    "med_high_prod__mostly_static__big_static":    [ 787, 1336, 3473, 9013],
    "med_high_prod__mostly_static__big_rotating":  [ 422, 4814, 4963, 7186],
    "med_high_prod__mixed_static__big_static":     [ 741, 4785, 8516, 9227],
    "med_high_prod__mixed_static__big_rotating":   [1265, 4461, 5981, 8855],
    "med_high_prod__mixed_rotating__big_static":   [ 585, 1217, 6462, 9984],
    "med_high_prod__mixed_rotating__big_rotating": [ 272,  662, 4353, 9118],
    "med_high_prod__mostly_rotating__big_static":  [2537, 3403, 6593, 8968],
    "med_high_prod__mostly_rotating__big_rotating":[ 489, 5675, 7770, 8624],
    "high_prod__mostly_static__big_static":        [1030, 1153, 2726, 8686],
    "high_prod__mostly_static__big_rotating":      [3335, 4333, 4693, 9812],
    "high_prod__mixed_static__big_static":         [1649, 2794, 4252, 9240],
    "high_prod__mixed_static__big_rotating":       [3493, 3774, 5923, 8514],
    "high_prod__mixed_rotating__big_static":       [2083, 2661, 6328, 9327],
    "high_prod__mixed_rotating__big_rotating":     [5199, 5274, 7542, 8909],
    "high_prod__mostly_rotating__big_static":      [ 405, 7045, 9408, 9658],
    "high_prod__mostly_rotating__big_rotating":    [3233, 3578, 4027, 6548],
}

# reverse lookup: seed -> archetype label
ARCHETYPE_OF = {s: a for a, seeds in BY_ARCHETYPE.items() for s in seeds}


def draw_seed(i: int) -> int:
    """The i-th rollout game's seed, cycling the curated set so coverage stays
    balanced across iterations/actors. ``i`` is the GLOBAL game index
    (``seed_base + episode``), so different iters/actors take different slices."""
    return SEEDS[i % len(SEEDS)]


def archetype_of(seed: int) -> str | None:
    return ARCHETYPE_OF.get(seed)
