# Orbit Wars

Agent development for the Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) competition — a real-time strategy game where 2 or 4 AI agents compete to conquer planets orbiting a central sun.

- **Prize**: $50,000 USD
- **Deadline**: 2026-06-23
- **Submission**: a `main.py` with an `agent(obs)` function returning `[[from_planet_id, angle, num_ships], ...]`

Players start with one home planet and launch fleets to capture neutral and enemy planets. Planets produce 1–5 ships/turn (inner ones orbit the sun, outer ones are static). Fleet speed scales log-style with size; fleets crossing the sun are destroyed. Comets spawn at steps 50/150/250/350/450 as temporary extra planets. The game ends at step 500 or when only one player is left — most total ships (on planets + in flight) wins.

## Quickstart

```bash
pip install -r requirements.txt

python run.py --mode play --agents sniper random   # play a match
python -m app.server                                # interactive dashboard
python run.py --mode submit --agents sniper        # pack and submit to Kaggle
```

---

## Current model: pair-score policy with 5 jointly-trained heads

Active learned line lives under [`agents/transformer_v2/`](agents/transformer_v2/README.md). It's a 4-layer transformer stack on top of 3 frozen per-entity specialist encoders, capped by a shared 2-layer trunk that feeds 5 simultaneous output heads. Heuristic and legacy agents are kept in [`agents/heuristic/`](agents/heuristic/) and [`agents/archive/`](agents/archive/) respectively.

### Architecture

```
L0 (frozen, per-entity MLP specialists, ~374k params)
   ├─ PlanetEncoder       (18 → 256)
   ├─ CometPastModel      (123 → 256)   ─► where(is_comet, ...) ─► entity_self (B, T=6, P, 256)
   └─ FleetEncoder        (24 → 256)                                            │
                                                                                ▼
L1 PlanetEntityEncoder       cross-attn: planets ←→ relation-aware fleets       (658k)
                              ([fleet_tok ‖ source_planet_tok ‖ target_planet_tok])
                                                                                ▼
L2 CrossEntityAttention      planet ↔ planet self-attention, multi-step over T=6,
                              learned CLS, 2-layer Pre-LN encoder                (1.05M)
                                                                                ▼ ctx_now (B, P, 256)
L3 DualRoleAttention         parallel source→target / target→source branches    (528k)
                                                                                ▼ source_aware, target_aware
L4 JointRoleAttention        concat 2P, 1-layer self-attn, split back           (528k)
                                                                                ▼ source_joint, target_joint
PairHead                     2-layer shared trunk → 5 single-Linear heads       (362k)
                              Linear(768 → 256) → GELU → Linear(256 → 256) → GELU
                                                  │
        ┌───────────────────┬─────────┴─────────┬─────────────────────┬───────────────────┐
        ▼                   ▼                   ▼                     ▼                   ▼
  pair_logits          pair_frac            source_act          target_aim            glob_act
   (B, P, P)            (B, P, P)            (B, P)              (B, P)               (B,)
   BCE pw=600           MSE on sigmoid       BCE pw=100          BCE pw=100           BCE pw=1
```

**Total trainable params: ~3.13M.** Loss is the sum of all 5 head losses, each masked appropriately (`pair_valid` for the cell heads, `planet_mask` for the per-planet heads, no mask for the snapshot head).

### Detailed flow with explicit residuals

`⊕` = residual skip + add; `+` = additive embedding (role / step), not a residual.

```
DATASET — EntitySnapshotDataset  (lazy T=6 stack via HISTORY_OFFSETS_T6=(5,4,3,2,1,0))
  planet_features (B,T,P,138) [first 18 dims used]   comet_features (B,T,P,123)
  fleet_features  (B,T,F,24)                          is_comet        (B,T,P) bool
  fleet_{target,source}_idx, fleet_mask, planet_mask, fleet_owner/ships/eta routing
  pair_labels (B,P,P) bool   pair_valid (B,P,P) bool   pair_ships (B,P,P) int32

═══════════════════════════════════════════════════════════════════════════════
L0 — frozen MLP specialists (no_grad, no residuals)
═══════════════════════════════════════════════════════════════════════════════
  planet_features[..., :18] ─▶ PlanetEncoder    (18 → 256)  ─▶ planet_tok (B,T,P,256)
  comet_features            ─▶ CometPastModel   (123 → 256) ─▶ comet_tok  (B,T,P,256)
  fleet_features            ─▶ FleetEncoder     (24 → 256)  ─▶ fleet_tok  (B,T,F,256)

           ┌── where-scatter (no residual) ──┐
           │ entity_self = where(is_comet,   │
           │                     comet_tok,  │
           │                     planet_tok) │
           └────────────┬─────────────────────┘
                        ▼
                entity_self (B, T, P, 256)

═══════════════════════════════════════════════════════════════════════════════
L1 — PlanetEntityEncoder  (per-timestep; flatten T into batch)
═══════════════════════════════════════════════════════════════════════════════
  src_tok = gather(entity_self_t, fleet_source_idx_t)   (B,F,256)  zero where idx=-1
  tgt_tok = gather(entity_self_t, fleet_target_idx_t)   (B,F,256)  zero where idx=-1
  fleet_repr = Linear(3·256 → 256)(concat[fleet_tok, src_tok, tgt_tok])    (no residual)

  ctx = MultiheadAttention(                                                (no residual)
          query = entity_self_t,
          key=value = fleet_repr,
          key_padding_mask = ~fleet_mask_t,
          n_heads = 4)                                          (B, P, 256)
  ctx = nan_to_num(ctx)                                          [empty-fleet rows]

  entity_tokens = Sequential(                                              (concat-fuse:
                    Linear(2·256 → 256), GELU,                              entity_self enters
                    Linear(256 → 256), LayerNorm                            via concat, not ⊕)
                  )(concat[entity_self_t, ctx])                  (B, P, 256)

═══════════════════════════════════════════════════════════════════════════════
L2 — CrossEntityAttention  (2-layer Pre-LN TransformerEncoder, multi-step T=6 + CLS)
═══════════════════════════════════════════════════════════════════════════════
  seq_t   = entity_tokens_t + step_embed[t]              (additive step embed)
  seq     = concat[CLS, flatten over (t, p)]             (B, 1 + T·P, 256)

  Pre-LN TransformerEncoderLayer × 2:

      x_in ──┬───────────┐                              x ──┬───────────┐
             ▼           │                                  ▼           │
          LayerNorm      │                               LayerNorm      │
             ▼           │                                  ▼           │
       MultiheadAttn(self, 8h) ⊕ residual 1               FFN(d→ff·d→d) ⊕ residual 2
             ▼           │                                  ▼           │
             ⊕  ◀────────┘                                  ⊕  ◀────────┘
             ▼                                              ▼

  glob    = out[:, 0]                                    (B, 256)        [CLS readout]
  ctx_full= out[:, 1:].reshape(B, T, P, 256)
  ctx_now = ctx_full[:, -1]                              (B, P, 256)     [current step]

═══════════════════════════════════════════════════════════════════════════════
L3 — DualRoleAttention  (two parallel cross-attn branches)
═══════════════════════════════════════════════════════════════════════════════
  source_tok = ctx_now + source_role                    (additive role embed)
  target_tok = ctx_now + target_role                    (additive role embed)

  ┌── Branch A: source-to-target ──────┐    ┌── Branch B: target-to-source ──────┐
  │ source_tok ─┬───────────┐          │    │ target_tok ─┬───────────┐          │
  │             ▼           │          │    │             ▼           │          │
  │   MHA(Q=src, K=V=tgt)   │          │    │   MHA(Q=tgt, K=V=src)   │          │
  │             ▼           │          │    │             ▼           │          │
  │             ⊕  ◀────────┘ residual │    │             ⊕  ◀────────┘ residual │
  │             ▼                      │    │             ▼                      │
  │         LayerNorm                  │    │         LayerNorm                  │
  │             ▼                      │    │             ▼                      │
  │       source_aware (B,P,256)       │    │       target_aware (B,P,256)       │
  └────────────────┬───────────────────┘    └────────────────┬───────────────────┘
                   └─────────────────┬────────────────────────┘
                                     ▼

═══════════════════════════════════════════════════════════════════════════════
L4 — JointRoleAttention  (concat 2P → self-attn → split)
═══════════════════════════════════════════════════════════════════════════════
  src_tagged = source_aware + source_role_l4            (additive role embed; fresh)
  tgt_tagged = target_aware + target_role_l4
  seq        = concat[src_tagged, tgt_tagged]            (B, 2P, 256)

  Pre-LN TransformerEncoderLayer × 1 (same residual layout as L2):
      x ──┬── LN ── MHA ──⊕──┬── LN ── FFN ──⊕──▶ out
          └──────────────────┘  └──────────────────┘
                  residual 1            residual 2

  source_joint = out[:,  :P]                             (B, P, 256)
  target_joint = out[:, P: ]                             (B, P, 256)

═══════════════════════════════════════════════════════════════════════════════
PairHead — shared 2-layer trunk + 5 single-Linear heads
═══════════════════════════════════════════════════════════════════════════════
  Project to d_pair=128:
    src_r = Linear(256→128)(source_joint)                (B, P, 128)
    tgt_r = Linear(256→128)(target_joint)
    ctx_r = Linear(256→128)(ctx_now)

  Broadcast across (P, P):
    pair_feat = concat[src_r, ctx_r, tgt_r, ctx_r, src_r⊙tgt_r, ctx_r⊙ctx_r]  (B,P,P,768)

  Shared trunk (no residual — purely sequential):
    Linear(768 → 256) → GELU → Linear(256 → 256) → GELU      ─▶ trunk (B, P, P, 256)

  5 heads (each Linear(256 → 1) on the trunk, no residual):
    pair_head        : trunk[s,t]                    → pair_logits  (B, P, P)
    pair_frac_head   : trunk[s,t]                    → pair_frac    (B, P, P) sigmoid
    source_act_head  : masked_mean(trunk, dim=tgt)    → source_act   (B, P)
    target_aim_head  : masked_mean(trunk, dim=src)    → target_aim   (B, P)
    glob_act_head    : masked_mean(trunk, dim=(s,t))  → glob_act     (B,)
```

#### Residuals summary

| Layer | Has residual? | Notes |
|---|---|---|
| L0 specialists | No | 3-Linear MLP + final LN, no skip |
| L0 → entity_self (where-scatter) | No | Slot routing, not addition |
| L1 fleet_repr projection | No | Concat + Linear |
| L1 cross-attention → fuse | **No additive residual** | `entity_self` enters via `concat`, then 2-Linear fuse MLP projects 2d→d (concat-fuse) |
| L2 TransformerEncoderLayer × 2 | **Yes ⊕** | Standard Pre-LN: residual around MHA + residual around FFN |
| L3 DualRoleAttention (each branch) | **Yes ⊕** | `LayerNorm(role_tok + attn_out)` |
| L4 TransformerEncoderLayer × 1 | **Yes ⊕** | Same Pre-LN block as L2 |
| PairHead trunk + 5 heads | No | Purely sequential MLP + per-head Linear |

#### Additive embeddings (NOT residuals — additive type/position signals)

| Where | What | Why |
|---|---|---|
| L2 per timestep | `seq = entity_tokens + step_embed[t]` | Tells the encoder which step a token came from |
| L3 (both branches) | `ctx_now + source_role` / `+ target_role` | Disambiguates source vs target use of the same per-planet token |
| L4 (both halves) | `source_aware + source_role_l4` / `target_aware + target_role_l4` | Fresh role embedding so the joint encoder can tell the two halves apart in the (B, 2P, d) sequence |

### What each head predicts

| Head | Output | Supervision |
|---|---|---|
| `pair_logits`   | per (source, target) compatibility logit | expert pair-set: `[s, t] = True` if expert launched ≥1 fleet from `s` to `t` |
| `pair_frac`     | sigmoid → fraction of source's ships sent to target | `pair_ships[s, t] / row_sum`; masked to positive cells only |
| `source_act`    | per-planet binary "this planet launches a fleet" | `pair_labels.any(dim=-1)` |
| `target_aim`    | per-planet binary "this planet is targeted" | `pair_labels.any(dim=-2)` |
| `glob_act`      | per-snapshot binary "any action this turn" | `pair_labels.any(dim=(-1, -2))` |

### Supervision pipeline

Single-stage behavior cloning from raw replay JSONs with a T=6 history window on the inputs:

```
data/replays/<player>/*.json.gz
        │
        ▼
scripts/build_pair_dataset_orbital_occle.py
   (despite the filename, accepts comma-separated --player list)
        │
        │  • walks each replay, diffs obs[t+1].fleets vs obs[t].fleets
        │    to recover EVERY learner launch (not just the first per turn
        │    like the v1 action CSV stores), so coalition launches survive
        │  • emits one snapshot per (episode, turn) with
        │    pair_labels (P, P) bool + pair_valid (P, P) bool +
        │    pair_ships (P, P) int32
        │  • mixes multiple players with deterministic per-player RNG
        │  • keeps non-acted snapshots if --keep-non-acted (subsampled to
        │    hit --acted-min-ratio 0.30 floor) so the glob_act head gets
        │    real positive/negative examples
        │
        ▼
data/datasets/_pair_cache/<PlayerSlug>_T6/
   <PlayerSlug>_T6_p64_f1024_{acted,all}.pt
```

The cache stores **single-frame** snapshots; the T=6 window is rebuilt lazily at `__getitem__` time by indexing into prior turns of the same episode (`HISTORY_OFFSETS_T6 = (5, 4, 3, 2, 1, 0)`, oldest first). Missing past frames (start of episode or non-acted gaps) get zero-filled with all-False `planet_mask`, which L2's `key_padding_mask` cleanly ignores.

Current default cache: **bowwowforeach + Ebi** mixed (rank #1 and #3 on the leaderboard). 60,424 snapshots, 30.8% acted, 13.3 GB.

### Training entrypoint

```bash
python -m agents.transformer_v2.pretrain.entity_encoder \
    --pair-cache-path data/datasets/_pair_cache/bowwowforeach_Ebi_T6/bowwowforeach_Ebi_T6_p64_f1024_all.pt \
    --d-model 256 --batch-size 128 --epochs 30 --lr 5e-5 \
    --pair-pos-weight 600 \
    --source-act-pos-weight 100 \
    --target-aim-pos-weight 100 \
    --glob-act-pos-weight 1.0 \
    --device cuda
```

Per-epoch logging prints a 5-row table covering each head's train/val loss, `recall_true`, `recall_false`, `pos_frac`, and (for `pair_logits`) `recall_at_{1, 5, 10}`. See [`notebooks/train_entity_encoder_colab.ipynb`](notebooks/train_entity_encoder_colab.ipynb) for the Colab variant — the bundle lives at `gs://orbit-wars-shipping/entity/`.

### Frozen L0 ckpts

The three specialists are pre-trained once (multi-task per-entity supervision) and reused across all entity-pretrain runs:

| Encoder | Best ckpt dir under `data/runs/` |
|---|---|
| Planet | `planet/specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep` |
| Fleet  | `fleet/specialist_fleet_d256_40k_lr1e4_120ep` |
| Comet  | `comet/fullpath_scalar_multitask_d256_40k_lr1e4_120ep` |

Re-pretrain via the matching CLIs in `agents/transformer_v2/pretrain/{planet,fleet,comet_past}_encoder.py`.

### Where the rest of the code lives

- [`agents/transformer_v2/README.md`](agents/transformer_v2/README.md) — deep dive into each layer, code layout per module, training-time memory + runtime notes.
- [`agents/heuristic/`](agents/heuristic/) — rule-based and search agents (random, physical_v1..v4, sniper, mcts, hybrid). The strongest hand-coded baseline is `sniper_v2`; `physical_v4` is `mcts_v1`'s rollout policy.
- [`agents/archive/transformer_v1/`](agents/archive/transformer_v1/) — previous transformer line, kept loadable for the dashboard's `app/server.py` and ablation comparisons.
- [`docs/EVAL_SEEDS.md`](docs/EVAL_SEEDS.md) + [`utils/eval_seeds.py`](utils/eval_seeds.py) — stratified 128-seed panel across 32 game-shape archetypes for A/B testing.

The submission entrypoint (`run.py --mode submit`) packs whichever agent you select; pre-trained ckpts live under `data/runs/<stage>/`.
