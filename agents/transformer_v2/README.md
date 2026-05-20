# `transformer_v2/` — current learned line

Pair-score policy for Orbit Wars. Three frozen L0 specialist encoders feed a four-layer trainable stack (L1 → L2 → L3 → L4), capped by a `PairHead` that emits two `(B, P, P)` outputs through an L1-conditioned FiLM block: `pair_logits` (source→target compatibility) and `pair_frac` (fraction of source's ships to send). Behavior-cloned on top-leaderboard replays with a T=6 history window.

## TL;DR

```
L0 (frozen, per-entity MLP encoders)
   ├─ PlanetEncoder    (18 → 256)
   ├─ CometEncoder     (123 → 256)         ─► where(is_comet, ...) ─► entity_self (B, T, P, 256)
   └─ FleetEncoder     (24 → 256)                                                    │
                                                                                     ▼
L1 PlanetEntityEncoder  (cross-attn: planets ←→ relation-aware fleet tokens)
                                                                                     ▼ entity_tokens (B, T, P, 256)
L2 CrossEntityAttention (planet ↔ planet, multi-step Pre-LN encoder + CLS)
                                                                                     ▼ ctx_now (B, P, 256)
L3 DualRoleAttention    (parallel source-to-target / target-to-source branches)
                                                                                     ▼ source_aware, target_aware
L4 JointRoleAttention   (concat both, self-attn on 2P sequence, split back)
                                                                                     ▼ source_joint, target_joint
PairHead                ([src_r, ctx_r, tgt_r, ctx_r, src⊙tgt, ctx⊙ctx] → trunk h[s,t])
                              ↑ FiLM cond: [L1_src ‖ L1_tgt ‖ pair_type_emb]
                                 h_film = h + α · (γ · h + β)        (γ, β init=0; α=1 → identity)
                                                                                     ▼
                                                ┌───────────────┐
                                                ▼               ▼
                                          pair_logits        pair_frac
                                          (B, P, P)          (B, P, P, raw)
                                          BCE pw=600         MSE on sigmoid
                                          masked by          masked to positive
                                          pair_valid         cells only
```

**Trainable:** ~3.70M params (L1: 658k, L2: 1.05M, L3: 528k, L4: 528k, PairHead: 929k incl. FiLM). **Frozen L0:** 374k params.

## Architecture details

### L0 — three frozen specialist encoders

Each L0 specialist is a 3-Linear MLP + LayerNorm, pretrained independently against per-entity multi-task supervision (own state, future trajectory). Their checkpoints live under `data/runs/{planet,comet,fleet}/` and are loaded with `requires_grad_(False)`.

| Encoder | Input | Output | Best ckpt dir |
|---|---|---|---|
| `PlanetEncoder` | 18 scalars (sun-relative geom + owner one-hot + garrison log) | (B, T, P, 256) | `specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep` |
| `CometEncoder` | 18 scalars + 35 path slots × (dx, dy, valid) = 123 dims | (B, T, P, 256) | `fullpath_scalar_multitask_d256_40k_lr1e4_120ep` |
| `FleetEncoder` | 24 dims (mission, target rel, ships, ETA, heading, ...) | (B, T, F, 256) | `specialist_fleet_d256_40k_lr1e4_120ep` |

The unified per-entity `entity_self (B, T, P, 256)` stream is built by `torch.where(is_comet[..., None], comet_tok, planet_tok)` — same `d_model` everywhere so no projection bridge is needed.

### L1 — `PlanetEntityEncoder` (`encoder/entity_encoder.py`)

Cross-attention from each planet over a **relation-aware fleet representation**:

```python
src_tok = gather(entity_self, fleet_source_idx)     # zero where idx=-1
tgt_tok = gather(entity_self, fleet_target_idx)     # zero where idx=-1
fleet_repr = Linear(3·d → d)(concat[fleet_tok, src_tok, tgt_tok])
ctx, _    = MultiheadAttention(query=entity_self, key=value=fleet_repr,
                               key_padding_mask=~fleet_mask, n_heads=4)
entity_tok = Fuse(concat[entity_self, ctx])         # concat-fuse, not residual
```

Runs per-timestep over the T axis (flatten T into batch, reshape back). 658k params.

### L2 — `CrossEntityAttention` (`aggregator/cross_entity.py`)

Standard 2-layer Pre-LN `TransformerEncoder` over `[CLS, entity_tokens × T history frames]` (`T·P + 1` sequence length). 8 heads, ff=512, GELU. Internal residuals are the standard transformer skip + LayerNorm pattern. 1.05M params.

Outputs:
- `ctx_now (B, P, 256)` — per-planet contextual embedding at the current step (slice `[:, -1]` from `(B, T, P, d)`)
- `glob (B, 256)` — snapshot CLS readout (reserved for future snapshot-level heads)

### L3 — `DualRoleAttention` (`aggregator/dual_role_attention.py`)

Two parallel cross-attention branches with **additive role embeddings**:

```python
source_tok = ctx_now + source_role        # additive embedding
target_tok = ctx_now + target_role
source_aware = LayerNorm(source_tok + MHA(Q=source_tok, K=V=target_tok))
target_aware = LayerNorm(target_tok + MHA(Q=target_tok, K=V=source_tok))
```

Each branch is a single `MultiheadAttention` block with explicit residual + LayerNorm. 528k params total. The role embeddings disambiguate the same per-planet token in source vs target use.

### L4 — `JointRoleAttention` (`aggregator/joint_role_attention.py`)

Concatenates the two role-aware streams into a `(B, 2P, d)` sequence and runs a 1-layer Pre-LN `TransformerEncoder` so source-mode and target-mode slots attend to each other globally:

```python
src_tagged = source_aware + source_role_l4   # fresh role embeddings, independent of L3's
tgt_tagged = target_aware + target_role_l4
seq        = concat[src_tagged, tgt_tagged]         # (B, 2P, d)
out        = TransformerEncoder(seq, key_padding_mask=~tile(planet_mask))
source_joint = out[:, :P]    target_joint = out[:, P:]
```

528k params.

### PairHead (`aggregator/pair_head.py`)

Per-`(source, target)` compatibility scorer + ship-size regressor. Projects role and context tokens to `d_pair=d_model=256` (no down-projection by default), broadcasts to `(B, P, P, 6·d_pair=1536)`, runs a 2-Linear MLP trunk → `h`, then **L1-conditioned FiLM** → 2 heads:

```python
src_r = src_proj(source_joint)   # (B, P, 256)
tgt_r = tgt_proj(target_joint)
ctx_r = ctx_proj(ctx_now)
pair_feat[s, t] = concat[src_r[s], ctx_r[s], tgt_r[t], ctx_r[t],
                          src_r[s] ⊙ tgt_r[t], ctx_r[s] ⊙ ctx_r[t]]
h = trunk(pair_feat)                                   # (B, P, P, 256)

# FiLM conditioner — uses the L1 tokens (skip from L1, bypassing L2/L3/L4)
# + a 27-way source/target type embedding. Identity-init: γ=β=0 and α=1,
# so h_film = h on the first forward while the conditioner receives gradients.
pair_type[s, t] = source_type[s] * 9 + target_relation[t] * 3 + target_type[t]
cond[s, t] = concat[L1_src[s], L1_tgt[t], Embed27(pair_type[s, t])]
γ, β = chunk(film_proj(cond), 2, dim=-1)               # (B, P, P, 256) each
h_film = h + α * (γ * h + β)                           # α: learnable scalar, init=1

pair_logits = pair_head(h_film).squeeze(-1)            # (B, P, P)
pair_frac   = pair_frac_head(h_film).squeeze(-1)       # (B, P, P) raw logit
```

929k params total (trunk 460k + film_proj 271k + 3 input projections 197k + 2 heads ~0.5k + film_alpha 1).

**Why FiLM from L1?** The trunk's role-aware tokens (from L4) and role-agnostic context (from L2) capture **strategic** state. L1 carries **tactical** state — "this source already has outgoing fleets", "this target has an enemy inbound", "this planet is being contested" — which gets compressed by the time it reaches L2. The FiLM block lets the head tilt scores per-(s, t) on local tactical state without disrupting the trunk's pre-trained pathway. The identity-init means a legacy pre-FiLM ckpt loaded into the new module behaves bit-identically to its trained behavior; because `α` starts non-zero, the new conditioner is trainable from the first backward pass.

### Loss & metrics

```python
# pair_logits — per-cell BCE
loss_pair = BCE-with-logits(pair_logits, pair_labels, pos_weight=600, reduction='none')
loss_pair = (loss_pair * pair_valid).sum() / pair_valid.sum()

# pair_frac — MSE on sigmoid vs fraction of source ships (positive cells only)
source_ships = expm1(planet_features[:, -1, :, f006] * SHIPS_LOG_MAX)  # f006 = ships log
target_frac = pair_ships / source_ships.clamp(min=1).unsqueeze(-1)
loss_frac = ((sigmoid(pair_frac) - target_frac) ** 2 * pos_mask).sum() / pos_mask.sum()

total = loss_pair + loss_frac
```

`pos_weight=600` counters the ~0.16% positive cell rate (observed `n_neg / n_pos ≈ 600` on the bow+Ebi cache). **Many-to-one supported natively**: BCE-per-cell handles coalition launches (one target, multiple sources) without information loss.

Per-epoch eval prints:
- `pair_logits.loss`, `recall_true` (per-cell), `recall_false` (per-cell)
- `recall_at_{1,5,10}` (per snapshot — does any true pair fall in the top-k of the flat P² grid?)
- `row_recall_at_{1,5,10}` (per row — does any true target for this source fall in its top-k?)
- `pair_frac.loss` — MSE on positive cells
- `n_pos`, `n_neg`, `pos_frac`

## Code layout

```
agents/transformer_v2/
├── README.md              # this file
├── DESIGN.md              # initial design doc (pre-PairHead era)
├── HANDOFF.md
├── MODEL_TRAINING_ROADMAP.md
├── history.py             # HISTORY_OFFSETS (legacy 9-step) and N_HISTORY constants
├── paths.py               # data/run dir layout
├── runner.py              # inference / agent registration entrypoint
├── ppo.py                 # PPO bring-up (not used in this pretrain)
│
├── featurizer/            # raw obs → tensor features (one-shot, used at dataset-build time)
│   ├── planet_featurizer.py    # 18 scalars + 30 future-traj anchors per planet (138 dims)
│   ├── fleet_featurizer.py     # 24-dim per-fleet vector + helpers
│   ├── entity_featurizer.py    # per-(planet, owner_slot) inbound stats labels
│   └── inference.py            # runtime featurization for the agent runner
│
├── encoder/               # L0 + L1 modules
│   ├── planet_encoder.py       # 3-Linear MLP, single-frame
│   ├── fleet_encoder.py        # 3-Linear MLP
│   └── entity_encoder.py       # PlanetEntityEncoder (L1) + QueryConditionedPool (legacy)
│
├── aggregator/            # L2 / L3 / L4 / head modules
│   ├── cross_entity.py         # CrossEntityAttention (L2)
│   ├── dual_role_attention.py  # DualRoleAttention (L3)
│   ├── joint_role_attention.py # JointRoleAttention (L4)
│   └── pair_head.py            # PairHead (output)
│
└── pretrain/              # training scripts (each is a self-contained CLI)
    ├── planet_encoder.py       # L0 planet specialist pretrain
    ├── fleet_encoder.py        # L0 fleet specialist pretrain
    ├── comet_past_encoder.py   # L0 comet specialist pretrain
    ├── entity_encoder.py       # ★ CURRENT: L1+L2+L3+L4 + PairHead pretrain
    ├── cross_entity.py         # legacy: L1+L2 standalone pretrain
    ├── pair_score.py           # legacy: v1-style pair-score head (action dataset)
    ├── target_rank.py          # legacy: target-rank stage A/B
    ├── expert_action.py        # action featurizer + dataset class
    └── feature_probe_train.py  # diagnostic head training
```

The current active pretrain path is `pretrain/entity_encoder.py`. The others either pretrain L0 specialists (still useful for the freeze chain) or belong to a previous design generation (kept for ablation comparison).

## Dataset generation

### Stage 1: raw replays → per-stem feature CSVs

```
data/replays/<player>/<stem>.json.gz
              │
              ▼
scripts/build_encoder_dataset.py   (uses transformer_v1 featurizers — pre-dating v2's split)
              │
              ├─▶ data/datasets/planet/planet_<stem>.csv         (138 feat cols + meta)
              ├─▶ data/datasets/fleet/fleet_<stem>.csv           (24 feat cols + meta)
              ├─▶ data/datasets/entity/entity_<stem>.csv         (per-planet labels)
              ├─▶ data/datasets/cross_entity/cross_entity_<stem>.csv
              └─▶ data/datasets/action/action_<stem>.csv         (first-launch-per-turn pairs)
```

### Stage 2: comet feature CSVs

`action_<stem>.csv` only stores the **first** launch per turn — useful for L1's binary classification supervision but loses coalition launches. The comet specialist also wants per-comet 35-path-slot features that the v1 featurizer didn't emit.

```
scripts/build_entity_comet_features.py
   walks each stem's replay, emits 18-scalar + 35×(dx,dy,valid) per is_comet=1 row
              │
              ▼
data/datasets/entity_comet/comet_<stem>.csv    (123 input dims + meta)
```

### Stage 3: pair-set cache for the current PairHead pretrain

The big lift, done by a one-shot script:

```
scripts/build_pair_dataset_orbital_occle.py
              │
              │  1. enumerate viable stems under data/replays/Orbital Occle/
              │  2. for each stem, run the existing EntitySnapshotDataset to build
              │     the per-turn input tensors (planet_features (P, 138),
              │     comet_features (P, 123), fleet_features (F, 24), routing, masks)
              │  3. *** walk the raw replay JSON *** to recover the FULL expert pair set
              │     per turn — including coalition launches the action CSV drops.
              │     For each acted turn, build pair_labels (P, P) bool where
              │     [s, t] = True iff the expert launched ≥1 fleet from slot s to slot t.
              │  4. mask invalid pairs: pair_valid = planet_mask[s] & planet_mask[t] & (s != t)
              │  5. drop non-acted snapshots (they have empty pair sets)
              │
              ▼
data/datasets/_pair_cache/Orbital_Occle_T6/OrbitalOccle_T6_p64_f1024_acted.pt
   ~3.8 GB, 18,582 acted snapshots from 238 stems
   keys = list of (episode_uuid, turn)
   config = {player, history_offsets, max_planets, max_fleets, keep_non_acted}
   snapshots = list of dicts with all input tensors (single-frame in the cache) +
               pair_labels (P, P) bool + pair_valid (P, P) bool
```

**Why walk the replay JSON instead of trusting action CSVs?** The v1 `action_*.csv` only stores one (source, target) per turn. 7,468 of 18,582 acted snapshots (~40%) are coalition turns where the expert launched from multiple sources at once. Using the JSON walk surfaces them as **multi-positive** pair labels — without this, 40% of the supervision signal would be missing or wrong.

### Stage 4: training-time consumption

```
pretrain/entity_encoder.py
   loads CachedPairDataset(...)
              │
              │  CachedPairDataset.__getitem__(idx):
              │    return single-frame snap[idx]  if history_offsets is None
              │    else
              │      stack T frames at offsets (5, 4, 3, 2, 1, 0) of the same episode,
              │      zero-fill + all-False-mask any past frames before the episode start,
              │      keep pair_labels / pair_valid current-turn only (no T axis).
              │
              ▼
              batch with planet_features (B, T=6, P, 138), pair_labels (B, P, P), etc.
              │
              ▼ episode-level 80/10/10 split (train_eps disjoint from val/test eps)
              ▼
              train(): L0 frozen forward → entity_self → L1 → L2 → L3 → L4 → PairHead
                       → compute_pair_loss → backward → AdamW step
```

The cache stores **single-frame** snapshots; T=6 history is built on-the-fly in `__getitem__` via the same lookup pattern `EntitySnapshotDataset` uses (`_key_to_idx`). For acted-only training, the cache must still retain non-acted context frames and store the supervised acted rows separately as `acted_indices`; otherwise most `t-1..t-5` slots get zero-filled. This keeps the cache smaller than pre-stacking while preserving real temporal context.

## Reproducing the current run

Pretrain order, top to bottom:

```bash
# 1. L0 specialists (run once, then reuse the ckpts) — each takes ~30-60 min on Colab
python -m agents.transformer_v2.pretrain.planet_encoder       --d-model 256 --epochs 120
python -m agents.transformer_v2.pretrain.fleet_encoder        --d-model 256 --epochs 120
python -m agents.transformer_v2.pretrain.comet_past_encoder   --d-model 256 --epochs 120

# 2. Build the comet feature CSVs (one-shot, ~5 min on the full 1,100-stem set)
python scripts/build_entity_comet_features.py

# 3. Build the per-stem feature CSVs for the target player (if missing) — ~4 min
python scripts/build_encoder_dataset.py --replay-dir "data/replays/Orbital Occle" \
    --datasets planet,fleet,entity,cross_entity,action --num-episodes 250

# 4. Build the pair-set cache (one-shot, ~3 min, ~3.8 GB output)
python scripts/build_pair_dataset_orbital_occle.py

# 5. Run the PairHead pretrain
python -m agents.transformer_v2.pretrain.entity_encoder \
    --d-model 256 --batch-size 16 --epochs 30 --lr 1e-4 \
    --pair-pos-weight 600.0 --device mps
```

Default `--pair-cache-path` already points at the OO T=6 cache, so no flag override needed.

## Status (as of this README)

| Component | Status |
|---|---|
| L0 specialists (Planet / Fleet / Comet) | trained, ckpts under `data/runs/{planet,fleet,comet}/specialist_*_d256_*` |
| L1 PlanetEntityEncoder | cross-attn rewrite (replaced the dense owner-slot pool); smoke-tested |
| L2 CrossEntityAttention | 2-layer Pre-LN encoder; T=6 multi-step verified |
| L3 DualRoleAttention | 2-branch parallel cross-attn; gradient flow verified |
| L4 JointRoleAttention | 1-layer concat+self-attn+split; smoke-tested |
| PairHead | wired, smoke-tested; first full training run pending |
| Orbital Occle pair cache | built (18,582 acted snapshots, 100% comet coverage, 40% multi-positive) |
| Single-player generalization | OO done; the same pipeline can rebuild for any player in `data/replays/<name>/` by editing the cache build script's player constant |

## Evaluation

For A/B comparisons against baselines (sniper_v2, physical_v4, etc.), use the stratified 128-seed panel from `utils/eval_seeds.py`:

```python
from utils.eval_seeds import SEEDS, BY_ARCHETYPE, SEEDS_QUICK
```

`SEEDS_QUICK` (32 seeds) covers all 32 game-shape archetypes once; the full 128-seed `SEEDS` gives 4× per cell. Always play **both seats** per seed and aggregate **per archetype** — a net winrate gain can hide a regression on a specific board class. See `docs/EVAL_SEEDS.md` for the full methodology.

## PPO roadmap

`agents/transformer_v2/ppo.py` is currently a stub; PPO should be reintroduced
only after the supervised PairHead runtime policy is stable. The two-machine
CPU rollout/learner protocol is documented in:

```text
docs/PPO_TWO_CPU_PROTOCOL.md
```

Key constraint from that protocol: keep rollouts synchronous and strictly
on-policy at first, use `plan_launch` as the environment projection gate, and
track launch-miss matrices during eval so PPO does not learn unsafe semantic
actions that the launcher silently drops.

## Memory & runtime notes

- **Cache size:** 3.8 GB single-file `.pt`. Loads in ~3 min on first access (one-time cost; subsequent epochs reuse the in-memory snapshots).
- **Per-batch memory:** with `d_pair=256`, PairHead's `(B, P, P, 6·d_pair)` feature tensor is `B·64·64·1536·4` bytes: ≈805 MB at `B=32` fp32 before backward. FiLM adds pairwise conditioner/output activations (`2d+type_emb=544`, then `2·hidden=512`). Use `BATCH_SIZE=16` as the safe Colab default; increase only after checking peak memory.
- **Per-epoch time:** ~3-4 min/epoch on MPS (M2/M3, batch=32), ~30 s/epoch on a Colab T4.
- **T=6 cost:** ~1.5-2× the L1 cross-attn FLOPs vs single-frame, plus ~6× the L2 sequence length. Worth it for the temporal context on fleet movement.
