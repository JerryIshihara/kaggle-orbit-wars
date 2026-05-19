# `agents/` — agent registry and per-agent architectures

Every agent lives in its own subpackage and registers itself with `agents.registry.register(id, description)` at import time. The Kaggle submission entrypoint and the dashboard look agents up by their registered `id`.

Top-level layout:

```
agents/
├── registry.py          # Agent / AgentSpec dataclasses + register()
├── physics_utils.py     # shared constants (board, sun, max_speed, …) used by featurizers
├── heuristic/           # rule-based / search agents (each subpackage registers itself)
│   ├── random_v1/                       # baselines
│   ├── physical_v{1..4}/                # heuristic ladder; v4 is the strongest hand-coded
│   ├── physical_{static,orbit,comet}_v1/# single-entity-class ablations
│   ├── sniper_v{1,2}/                   # opportunistic snipe + defend
│   ├── mcts_v1/                         # search baseline (uses physical_v4 rollouts)
│   └── hybrid_v1/                       # rule prior + small learned re-weighter
├── archive/             # legacy learned agents (still loadable for inference)
│   └── transformer_v1/  # first transformer line (action head + pair head + target ranker)
└── transformer_v2/      # current transformer line (multi-stage specialists, 4-layer stack)
```

The `mlp_v1` and `cnn_v1` packages were retired alongside `run.py --mode train`; the current learned line trains via standalone CLIs under `agents/transformer_v2/pretrain/` and matching Colab notebooks.

This README focuses on **how the models are structured** (encoders, heads, training stages). For game/feature/action details see the per-agent `DESIGN.md` files.

---

## Rule-based and hybrid (`agents/heuristic/`)

| Agent | What it does | Where the brain lives |
|---|---|---|
| `random_v1` | Uniform random launches. Sanity baseline. | `heuristic/random_v1/__init__.py` |
| `physical_v1..v4` | Greedy expand: pick the highest expected-value target reachable this turn, size fleets by surplus heuristics. v4 adds inbound-threat awareness and surplus-aware sizing. | `heuristic/physical_v4/__init__.py` |
| `physical_{static,orbit,comet}_v1` | Restricted variants that only act on one planet class. Used to debug class-specific heuristics. | `heuristic/physical_*_v1/__init__.py` |
| `sniper_v1, sniper_v2` | Watches enemy fleet vectors; launches the minimum-ships counter that lands before the threat arrives. v2 adds motion-aware launch validation against orbital and comet targets. | `heuristic/sniper_v2/__init__.py` |
| `mcts_v1` | Monte Carlo tree search with rollout policy = `physical_v4`. Single-thread, time-budgeted at decision time. | `heuristic/mcts_v1/__init__.py` |
| `hybrid_v1` | Take `physical_v4` proposal, apply a small learned re-weighter over its candidates. | `heuristic/hybrid_v1/__init__.py` |

These all live in a single `__init__.py` per package and use `physics_utils.py` for board constants. No frozen weights or pretrain stages.

---

## Archived: `transformer_v1` — first transformer (`agents/archive/transformer_v1/`)

Three modules: `FleetEncoder`, `PlanetEncoder`, plus a cross-attention block that fuses them. Outputs a `PairScoreHead` (per (source, target) planet pair logits) and an optional `FracHead` (Gaussian over ships-fraction). Trained in stages:

1. **Encoder pretraining** — `FleetEncoder` / `PlanetEncoder` against per-entity supervision (own state, future state).
2. **Pair-score training** — frozen encoders + new pair head, supervised on expert (source, target) choices.
3. **Frac head** — joint pair + frac training; pair head can be frozen or co-trained.
4. **(optional) PPO** — `agents/transformer_v1/ppo.py`.

The pretrain CLIs are in `agents/transformer_v1/pretrain/`; checkpoints land in `data/runs/{fleet,planet,pair_score}/<run_dir>/`. Inference glues them in `agents/transformer_v1/runner.py`.

`transformer_v1` is still used by the dashboard's default agent (latest `pair_score_best.pt` under `data/runs/pair_score/`).

---

## `transformer_v2` — current line

The active learned line is documented in detail in
`agents/transformer_v2/README.md`. The short version:

```text
L0 frozen specialists
  PlanetEncoder      138 raw planet dims; scalar-only ckpt consumes first 18
  CometPastModel     123 dims = 18 scalars + 35×(dx,dy,valid)
  FleetEncoder       24 raw fleet dims
        │
        ├─ planet/comet slots hard-routed with torch.where(is_comet, comet, planet)
        ▼
L1 PlanetEntityEncoder
  planet←fleet cross-attention over relation-aware fleets:
  [fleet_tok || source_entity_tok || target_entity_tok]
        ▼
L2 CrossEntityAttention
  2-layer Pre-LN Transformer over [CLS, T×P entity tokens], current T=6
        ▼
L3 DualRoleAttention
  parallel source→target and target→source role branches
        ▼
L4 JointRoleAttention
  concat source/target role streams, self-attend over 2P, split back
        ▼
PairHead
  emits (B, P, P) source→target logits
```

Current pretrain target: expert **pair-set** behavior cloning from raw
replay launches. The loss is masked BCE-with-logits over the pair grid,
not the older flattened softmax and not the older 9 per-planet entity
heads. Multi-source coalition turns are preserved because several
`pair_labels[source, target]` cells can be true in the same snapshot.

Frozen during this stage:

- `PlanetEncoder`
- `CometPastModel.encoder + norm`
- `FleetEncoder`

Trainable during this stage:

- `PlanetEntityEncoder`
- `CrossEntityAttention`
- `DualRoleAttention`
- `JointRoleAttention`
- `PairHead`

The current cache path is:

```text
data/datasets/_pair_cache/Orbital_Occle_T6/OrbitalOccle_T6_p64_f1024_acted.pt
```

It carries single-frame snapshots; `CachedPairDataset` builds the T=6
history window `(t-5, t-4, t-3, t-2, t-1, t)` lazily at training time.
Older files such as `pretrain/cross_entity.py`, `pretrain/pair_score.py`,
and `pretrain/target_rank.py` are retained for ablations / previous
experiments, but `pretrain/entity_encoder.py` is the current active
training entry point.

---

## Adding a new agent

1. Create `agents/<your_id>/__init__.py` and call `register("<your_id>", "<short description>")` at import time.
2. Implement `agent(obs)` returning `[[from_planet_id, angle, num_ships], ...]`.
3. Add the package to `agents/__init__.py`'s import list (with a `try/except ImportError` guard if it has heavy deps like torch).
4. The dashboard (`python -m app.server`), local play (`python run.py --mode play --agents <your_id> ...`), and submission packing (`python run.py --mode submit --agents <your_id>`) will all pick it up automatically.
