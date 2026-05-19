# Stratified 128-seed eval panel

A community-shared seed list for local A/B testing, stratified across 32 game-shape "archetypes" so a winrate comparison can show **which kinds of boards** a change helps or hurts — not just an overall number.

Source: shared by chrisleitescha on the Kaggle Orbit Wars forum.
Preview images: <https://www.kaggle.com/datasets/chrisleitescha/orbit-wars-seed-panel-preview>

## Construction

128 seeds total = 32 archetypes × 4 seeds per cell. Archetypes come from three axes:

| Axis | Bins |
|---|---|
| **Production level** (game pace proxy) | `low_prod` / `med_low_prod` / `med_high_prod` / `high_prod` |
| **Rotating share** (orbital-path coverage) | `mostly_static` / `mixed_static` / `mixed_rotating` / `mostly_rotating` |
| **Size split** (where the big planets live) | `big_static` / `big_rotating` |

`4 × 4 × 2 = 32` cells, each with 4 seeds. All seeds drawn from `range(0, 10_000)`.

`SEEDS` is **interleaved by archetype**: the first 32 entries cover all 32 cells once, the first 64 cover each cell twice, etc. So:

- Quick balanced A/B → `SEEDS[:32]`
- Standard sweep → `SEEDS[:64]`
- Full panel → all 128

## Usage

Import the constants directly:

```python
from utils.eval_seeds import SEEDS, BY_ARCHETYPE, SEEDS_QUICK, archetype_for_seed
```

Run a stratified A/B with both-seat coverage:

```python
from kaggle_environments import make
from utils.eval_seeds import BY_ARCHETYPE

wins = {arch: [0, 0] for arch in BY_ARCHETYPE}     # [my_wins, total]

for archetype, seeds in BY_ARCHETYPE.items():
    for seed in seeds:
        for my_seat in (0, 1):                      # play both sides
            env = make("orbit_wars", configuration={"seed": seed})
            agents = ["my_agent.py", "opponent.py"]
            if my_seat == 1:
                agents = agents[::-1]
            env.run(agents)
            # ... score game, update wins[archetype][0] += win, wins[archetype][1] += 1
```

## Two habits worth keeping

1. **Play both seats for every seed.** Otherwise a change that helps when you're seat 0 (home-planet bottom-left) but hurts as seat 1 can look like a net win.
2. **Aggregate per archetype, not just overall.** A new version that wins in aggregate can still lose every game on a specific archetype slice — that's exactly the kind of regression a stratified panel catches.

## Caveats

- Built against the simulator in `data/main.py`. Seeds in `range(0, 10_000)` are reproducible there.
- **2-player only.** 4-player games sample the same initial-planet RNG so geometry archetypes still apply, but 4P-specific dynamics (3 opponents, coalition scoring) aren't covered.
- Archetype labels are human-readable bin names — nothing magical; re-bin however you like for your own analysis.
- The original share contained a single typo (`047` → `5047`) which is fixed in `utils/eval_seeds.py`. All 128 seeds are unique and the 32 buckets cover them bijectively.

## Files

| Path | Purpose |
|---|---|
| `utils/eval_seeds.py` | Importable `SEEDS`, `BY_ARCHETYPE`, `SEEDS_QUICK`, `archetype_for_seed()` |
| `docs/EVAL_SEEDS.md` | This doc |
