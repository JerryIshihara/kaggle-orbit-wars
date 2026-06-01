# Cross-entity attention

Sits one level above the per-planet entity encoder. The entity encoder
gives each planet a token that has folded in *its own* inbound-fleet
picture; this layer lets every token see every *other* token, so each
planet ends up contextualized by the global state (frontiers,
neighbors, sector balance, what other planets are about to do).

```
fleet feats ─► FleetEncoder ──► fleet tokens   ──┐
                                                  ├─► PlanetEntityEncoder
planet feats ► PlanetEncoder ► planet tokens  ──┘     │
                                                       ▼
                                     entity tokens (per-planet, with
                                                     inbound fleets)
                                                       │
                                                       ▼
                                       CrossEntityAttention   ◄── this package
                                                       │
                                                       ▼
                                       contextual_tokens (B, P, d)
                                       global_token      (B, d)
```

## Boundary: Perception Ends Here

`CrossEntityAttention` is the last world-perception layer. Its job is to
produce a coherent, learner-relative world state:

- `contextual_tokens (B, P, d)` answer "what is each planet/comet in the
  context of the whole board?"
- `global_token (B, d)` answers "what does this snapshot look like overall?"

Do not push action-role, player-intent, or doctrine learning back into this
layer unless diagnostics show perception itself is wrong. Above this boundary,
the architecture should split into decision learners:

```text
CrossEntityAttention
  -> ctx_now, glob
        |
        ├─ PlayerContextLearner
        |    player_ctx (B, S, d), learner_ctx (B, d)
        |
        ├─ StrategyLearner
        |    strategy_ctx (B, d) or K strategy tokens
        |
        └─ ActionLearner
             source/target role tokens + PairHead/PPO heads
```

The current code uses `DualRoleAttention`, `JointRoleAttention`, and
`PairHead` as a compact `ActionLearner`. The planned refactor should insert
explicit `PlayerContextLearner` and `StrategyLearner` modules between L2 and
the action learner rather than adding more generic self-attention to L2.

### PlayerContextLearner

Purpose: summarize the learner and opponents after perception is complete.

Inputs:
- `ctx_now`, `glob`, `planet_mask`
- owner-slot masks derived from learner-relative ownership features
- cheap scalar totals: ships, production, planet count, frontier pressure

Outputs:
- `player_ctx (B, S, d)` for learner + enemy slots
- `learner_ctx (B, d)` as the policy's point-of-view token

Good auxiliary labels:
- current and future ship/production/planet totals by player slot
- learner advantage margin at horizons 10/25/50
- per-player aggression/launch-rate summaries from recent turns when history
  is available

### StrategyLearner

Purpose: choose the strategic mode before selecting concrete launches.

Inputs:
- `glob`
- `learner_ctx`
- `player_ctx`
- optional pooled frontier/opportunity tokens from `ctx_now`

Outputs:
- `strategy_ctx (B, d)` or a small set of strategy tokens
- optional interpretable strategy logits: expand, defend, attack, reinforce,
  evacuate, race/endgame

Good auxiliary labels:
- phase bucket from turn/remaining turns
- expert action archetype from replay launches
- future delta labels: gained planet, lost planet, defended threatened planet,
  net ship/production swing

### ActionLearner

Purpose: translate world + player + strategy context into executable semantic
actions.

The current action learner is:

```text
ctx_now -> DualRoleAttention -> JointRoleAttention -> PairHead
```

The refactored action learner should condition that path on
`learner_ctx + strategy_ctx`, preferably by FiLM or cross-attention, instead of
asking PairHead to infer strategic mode only from pair tokens.

## Architecture

Vanilla `nn.TransformerEncoder` over `(B, P, d_model)` entity tokens,
with one tweak: a **learned `[CLS]`** prepended to the sequence so we
get a single global summary token for snapshot-level heads (value,
expert-acted, win-prob, …).

```python
class CrossEntityAttention(nn.Module):
    """Multi-layer self-attention over entity tokens.

    Inputs:
        entity_tokens  (B, P, d)       from PlanetEntityEncoder
        entity_mask    (B, P) bool     True = real entity

    Output:
        contextual_tokens  (B, P, d)
        global_token       (B, d)
    """
    def __init__(self, d_model=64, n_heads=4, n_layers=2,
                 ff_mult=2, dropout=0.0):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            batch_first=True, activation='gelu', dropout=dropout,
            norm_first=True,    # Pre-LN — more stable for shallow stacks
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
```

### Why these choices

- **`TransformerEncoder`** — dynamic-P just works because PyTorch
  handles `src_key_padding_mask` natively. The
  `EntitySnapshotDataset` already pads to `max_planets`; the mask
  flows through directly (`src_key_padding_mask = ~entity_mask`).
- **`[CLS]` token** — a single learned vector prepended at sequence
  position 0, never masked, that the self-attention layers gradually
  fill with a global summary of the snapshot. Acts as the snapshot's
  "header" — every other token attends to it, it attends to every
  real entity, and the final state at index 0 is read out as the
  global feature for snapshot-level heads (winner, expert-acted,
  win-prob…). Cheaper and cleaner than a separate masked mean-pool.
- **Pre-LN (`norm_first=True`)** — more stable than post-LN for
  shallow stacks. 2 layers without a warmup schedule (v2 dropped
  from the original 3 layers to reduce destructive context mixing
  per MODEL_TRAINING_ROADMAP §Phase-2).
- **2 layers, 8 heads, `ff_mult=2`** — matches entity-token
  complexity. Per-layer QKV is `64 × 64` ≈ 12k params; total under
  70k for the L2 stack. Tiny by transformer standards but enough
  capacity for `P ≤ 64` self-attention.
- **No geometric bias in the MVP** — planet position is already in
  the planet token, so the model can rediscover proximity from data.
  Keep an `attention_bias` shim around for a v2 if attention patterns
  look diffuse during evaluation.

### Pipeline integration

The whole stack is element-wise / mask-aware end-to-end, so dynamic
`P` (planets coming and going as comets spawn/die) needs no special
handling beyond the existing `entity_mask`.

### Multi-step (temporal) extension

The MVP processes one snapshot at a time. To give the layer recent
history (e.g., last 3 turns), the cleanest extension is to **stack
along the sequence dim**:

```
sequence = [CLS, e_{t-2}^0..e_{t-2}^{P-1}, e_{t-1}^0..e_{t-1}^{P-1}, e_t^0..e_t^{P-1}]
```

with two added inputs:

- **Step embedding** — a learned `d_model` vector per relative time
  offset (`t-2 → emb_0`, `t-1 → emb_1`, `t → emb_2`). Added to each
  token before the encoder so the attention can tell turns apart.
  Same role as positional embeddings in a vanilla transformer, but
  over time rather than position-in-sequence.
- **Per-step entity_mask** — the same `(P,) bool` mask, replicated
  per step and concatenated, so padded entities at any timestep are
  ignored. The CLS slot stays unmasked across all steps.

Total sequence length: `1 + 3 × P` (so 193 with `P=64`); attention
is still O(seq²) but stays tractable for `T ≤ 4`.

#### Cold-start (first two turns)

Episodes start at `t=0`, so for `t < 2` we don't have a real
history. The robust handling is to **pad with empty frames + mask
them out**, not branch on the step count:

- Always allocate slots for `T` history steps.
- For unavailable steps, fill with zero tokens and set their
  per-step entity_mask to all-False.
- The encoder ignores the masked tokens — the attention output for
  the present-turn tokens just doesn't get any history contribution.

Why padding-with-mask beats a separate branch:

1. **Single model, single forward path.** No conditional branching
   at training time, no special-case batches at inference.
2. **No distribution shift between cold-start and steady-state.**
   With branching, the first few turns of every episode would use a
   different sub-model, and the supervision signal for those turns
   would be sparser.
3. **The mask is the inductive bias.** The encoder learns "no info
   from those slots = treat them as void," which is exactly what we
   want; it doesn't have to also learn "different model architecture
   for different turn counts."

## Labels that *require* cross-entity reasoning

The per-token labels we already write (`owner_t+K`, `is_target_this_turn`,
…) get richer signal once cross-attention is added — predicting "is
this planet under threat" is much easier when the model can see whether
nearby friendlies could reinforce. Keep all of them; **add** the
following so cross-attention gets supervision that *fundamentally*
needs neighbor info.

### Tier 1 — Spatial / structural

Cheap to compute (pure obs aggregation), force the model to look
beyond a single token.

| Label | Type | Forces cross-entity attention because… |
|---|---|---|
| `frontier_class` | categorical (4): interior / friendly_border / contested / isolated | Definition is *about neighbors* — count K nearest planets, check owner mix |
| `n_friendly_within_R_norm` | regression | Counts of *other* planets within radius R |
| `n_enemy_within_R_norm` | regression | Same, enemy variant |
| `nearest_friendly_dist_norm` | regression | Distance to nearest friendly *other* planet |
| `nearest_enemy_dist_norm` | regression | Distance to nearest enemy *other* planet |
| `sector_advantage_log` | regression (signed-log) | `signed_log1p(Σ_friendly_ships_within_R − Σ_enemy_ships_within_R)` — pure aggregate over multiple entities |

`R` defaults to ~30 board units (≈ 30% of the board), tunable.

### Tier 2 — Imitation / decision

Attached to the **CLS** token, not per-planet.

| Label | Type | Notes |
|---|---|---|
| `expert_source_logit` | per-planet binary (BCE) over masked tokens | "Did the expert launch FROM this planet this turn?" — each token gets a binary head; comparing *across tokens* is what cross-attention buys |
| `expert_target_logit` | per-planet binary | Same for first-hit targets |
| `expert_acted_this_turn` | binary on CLS | Did the expert launch *anything* this turn? Cheap, dense signal |

The per-planet binary form (sigmoid + BCE) handles "expert launched
from 2 different planets" cleanly — no P-way softmax constraint.

### Tier 3 — Game-level value (CLS, sparse but high-value)

| Label | Type | How to compute |
|---|---|---|
| `winner_seat` | categorical (`NUM_OWNER_SLOTS`+1, learner-relative) | Final winner of the episode, broadcast to every snapshot |
| `score_advantage_at_end_log` | regression (signed-log) | `signed_log1p(my_total_ships_at_t=END − max_other_player_ships_at_t=END)` |
| `turns_until_episode_end` | regression | `min(1, (T − t) / EPISODE_STEPS)` |

These give cross-attention a strong global shape signal: "what does
this game look like at the global level?" `winner_seat` is essentially
a coarse value function and helps the layer learn long-horizon
reasoning even though the per-step labels are dense.

### Tier 4 — Pairwise (skip in this layer)

`(src, tgt)` capture probability is the natural cross-entity label,
but P² pairs are sparse and big. Handle it in the post-L2 ActionLearner
(`DualRoleAttention` / `JointRoleAttention` / `PairHead` today), which
queries the cross-attention output per pair as needed.

## Implementation order

1. **`CrossEntityAttention`** module here in `aggregator/` —
   straightforward MVP with pre-LN encoder + CLS.
2. **Tier-1 spatial labels** in `featurizer/entity_featurizer.py` —
   6 new columns, derived from `obs.planets` graph ops at write time.
3. **Tier-2 imitation heads** — reuse existing
   `is_source_this_turn` / `is_target_this_turn` labels (already
   written by entity featurizer); add a single CLS `expert_acted`
   flag derived from `next_fleet_id` deltas.
4. **Tier-3 game-level labels** — one-pass replay post-process to
   broadcast `winner_seat` and `score_advantage_at_end_log` to every
   snapshot; `turns_until_episode_end` is a trivial scalar.
5. **`pretrain/cross_entity.py`** — load the entity-encoder
   checkpoint frozen, train `CrossEntityAttention` + heads on the
   combined per-planet + CLS labels.

## Future work / open questions

- **Geometric attention bias.** If attention is too diffuse, add an
  additive `1/distance` and `1/eta` bias to the attention logits
  (separate path, applied post-attention). Default off in MVP.
- **3-step temporal stack** (see "Multi-step extension" above) is
  planned as the first follow-up after the single-snapshot MVP
  trains cleanly. Step embedding + cold-start padding+mask, no
  architectural branching.
- **PlayerContextLearner.** A single CLS summarizes the snapshot but mixes
  perspectives. The post-L2 `PlayerContextLearner` should replace the old
  "per-player CLS" idea with explicit per-player summary tokens built from
  owner-slot masks and `ctx_now`.
- **Sparsity.** Most snapshots have ~30 real planets but we pad to
  64. With `P ≤ 64` the O(P²) attention is fine; if `P` grows we'd
  switch to flash-attention or local-windowed attention.
- **Stacking with action decoder.** Cross-attention output → action
  decoder in agent inference. The training-time data flow is one-way
  (no recurrent dependency), so the layer can be trained
  independently of the decoder.
