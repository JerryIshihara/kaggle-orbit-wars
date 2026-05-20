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

## Known issues / problem list

Live punch list of observed failures and gaps. Add new items as they're found; strike (`~~done~~`) or remove once fixed. Cite a replay / seed when possible so we can re-check after a change.

1. **Fleets miss the target and waste resources** — happens most often on comet targets, but also static planets near the board edge. The lead-aim estimate puts the fleet on a path that misses the moving target by more than its radius, so the fleet flies past, exits the board, and gets removed. Each miss spends the source's surplus for no return. Likely contributors: (a) `_lead_aim_comet` falls back to the comet's current position when the iterative ETA outruns the path window; (b) for fast-moving sources/targets the integer step rounding compounds; (c) the multi-target inference rule (`pair_logits > 2.0`) can fire low-confidence cells whose trajectories were never validated by `plan_launch`. Need to add a per-launch validation gate in `_target_to_moves` that hard-rejects when `plan_launch.ok=False` instead of falling back to a learned-frac launch.
2. **Comet-borne garrisons carried out of the map** — when a comet's path runs off the board, ships sitting on that comet at expiry vanish with it. The agent doesn't pre-emptively evacuate / launch from a comet whose path is about to end, so any garrison the agent built up there (or any planet captured *via* a comet) is lost the moment the comet expires. Fix candidates: (a) feature the comet's remaining-path-length and tail-distance-to-edge in `comet_features`; (b) add an evacuation rule in the runner that forces a launch from any comet within K turns of expiry; (c) penalize stationing ships on short-lived comets at the policy / scoring level.
3. **Agent ignores simple targets** — undefended neutrals or weakly-held enemies adjacent to the agent's own planets sometimes get no launch even though `physical_v4` would grab them every time. Suspected causes: `pair_logits > 2.0` threshold is calibrated for the average snapshot density and may suppress confident-but-not-extreme cells in low-action turns; `source_act` head isn't gating inference (it's trained but unused at runtime); the model was trained on bow+Ebi who play a tighter style than physical_v4. Could fix by lowering the threshold late-game, or by adding a "must-fire" fallback when `glob_act` is high but the threshold-filtered cells are empty (use top-K instead).
4. **Model can't recognize blocking planets on the launch path** — the agent picks a (source, target) pair whose straight-line trajectory passes through an intervening planet (own, neutral, or enemy). The intercepting body absorbs the fleet, so the intended target is never reached and the source's ships are spent on the wrong planet. Today the runner relies on `plan_launch` *after* the model picks the pair, but the LEARNED scorer has no feature describing "is there a planet sitting on the ray from src to tgt?". The relation tokens at L1 are `[fleet_tok ‖ src_planet ‖ tgt_planet]` — they don't include third-party planets that geometrically sit between the pair. Fix candidates: (a) add a per-(s, t) geometric feature (count / nearest-distance / soonest-blocker-arrival) and feed it into PairHead via `pair_scalars`; (b) at training time, mask out or down-weight cells where `plan_launch.reason == "wrong_planet_*"` so the model learns to score them low rather than just being rejected at inference.
5. ~~**Miss-rate calculation diverges from env's actual outcomes**~~ — fixed in `utils/logger.py:trace_launch_motion` (2026-05-20). The aggregator now consumes `trace_fleets` outcomes directly (the env's ground-truth fleet lifecycle) instead of re-simulating physics. Mapping: `captured`/`reinforced`/`annihilated` → `ok`; `destroyed_sun` → `sun`; `out_of_map` → `boundary`; `unknown` → `unknown`. Multi-target moves the env rejected (running ship pool exhausted) produce no record now (FIFO match against fleets via `(owner, launch_step, from_id)`). Verified on seed=1729 transformer_v2 vs physical_v4: analyzer reports `boundary=44, sun=11` matching env's `out_of_map=44, destroyed_sun=11` exactly. `_first_collision_for_launch` is no longer called — kept as an optional diagnostic helper. (The original symptom was 0% reported miss rate vs ~15% real — caused by cumulative orbital-rotation drift between the local simulator and the env's absolute-angle planet positions.)
6. **Performance degrades on large maps** — agent plays competently on seeds with fewer planets (~16-24 real planets, common static-heavy generations) but stops scaling on maps with the maximum number of planet groups (~40 planets, multiple orbital rings + comet spawns). Hypotheses: (a) **training distribution skew** — bow+Ebi's 555 replays may under-represent maximal-planet seeds, so the model never learned how to spread surplus across many fronts; (b) **per-cell pair_logits calibration shifts with P** — with `pos_weight=600` BCE trained against an average of ~30 valid cells per source, the absolute logit magnitude depends on local pair density, so the `> 2.0` inference threshold may be miscalibrated when the planet count doubles (more competing targets dilute confidence); (c) **L2/L4 attention saturation** — the planet↔planet self-attention sees 40+ tokens (vs ~16-20 typical) and may not have learned to allocate attention budget across that many entities cleanly; the model was trained with `max_planets=64` padding but the *actual* P distribution in training data is skewed low; (d) **action-budget mismatch** — the multi-target threshold rule scales linearly with active source count, so a 40-planet map can issue 10+ launches/turn while bow+Ebi typically issued 2-3; the model never saw rollouts with that launch density. Investigation candidates: bucket the eval-seed panel by P and report win rate per bucket; histogram pair_logits by snapshot P at val time to confirm the calibration shift; check whether physical_v4 also degrades on large maps (if yes, it's an env-difficulty axis; if no, it's a learning gap).



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
PairHead                     2-layer shared trunk → 5 single-Linear heads       (658k)
                              Linear(1536 → 256) → GELU → Linear(256 → 256) → GELU
                              (d_pair = d_model = 256 by default — no down-projection;
                               pass --d-pair 128 to reproduce the legacy narrowed layout)
                                                  │
        ┌───────────────────┬─────────┴─────────┬─────────────────────┬───────────────────┐
        ▼                   ▼                   ▼                     ▼                   ▼
  pair_logits          pair_frac            source_act          target_aim            glob_act
   (B, P, P)            (B, P, P)            (B, P)              (B, P)               (B,)
   BCE pw=600           MSE on sigmoid       BCE pw=100          BCE pw=100           BCE pw=1
```

**Total trainable params: ~3.43M.** Loss is the sum of all 5 head losses, each masked appropriately (`pair_valid` for the cell heads, `planet_mask` for the per-planet heads, no mask for the snapshot head).

### Per-layer features and pretrain tasks

| Layer | Input features (per timestep) | Output | Pretrain task (standalone) | Role in joint stack |
|---|---|---|---|---|
| **L0 PlanetEncoder** | 18 sun-relative scalars per planet: `is_comet` flag, polar coords, radius, ships_log, owner one-hot (4 slots), production, angular_velocity | `planet_tok (P, 256)` | multi-horizon future-state regression: planet ships@t+K, owner@t+K, "is source"/"is target" labels at K∈{1, 3, 5, 10, 15, 20} | Frozen scalar encoder for **static + orbital** planets; comet slots are routed elsewhere |
| **L0 CometPastModel** | 123 dims per comet planet = 18 scalars + 35 path slots × (dx, dy, valid) | `comet_tok (P, 256)` | path-aware analog: future ships/owner + trajectory interpolation supervision over the 35-step path window | Frozen encoder for **comet** planets only; selected via `where(is_comet, comet_tok, planet_tok)` |
| **L0 FleetEncoder** | 24 dims per fleet: source/target planet routing one-hots, ships_log, eta_norm, owner one-hot, in-flight angle, board-edge clearances | `fleet_tok (F, 256)` | per-fleet supervision: owner-of-target-at-arrival, fleet-survives-the-trip, eta-bucket | Frozen per-fleet representation; consumed by L1 cross-attention |
| **L1 PlanetEntityEncoder** | `entity_self (P, 256)` from L0 where-scatter + `fleet_tok (F, 256)` + relation routing (`fleet_source_idx`, `fleet_target_idx`, `fleet_mask`) | `entity_tokens (P, 256)` | not pretrained — trained jointly | Aggregates "what fleets are coming at / leaving from each planet" via relation-aware cross-attn; relation tokens are `[fleet_tok ‖ src_planet_tok ‖ tgt_planet_tok]` |
| **L2 CrossEntityAttention** | `entity_tokens` over **T=6 timesteps** + learned CLS + additive `step_embed[t]` | `ctx_now (P, 256)` + `glob (256)` (CLS) | jointly trained | Planet↔planet self-attention with step-position awareness; CLS pools the whole snapshot |
| **L3 DualRoleAttention** | `ctx_now + source_role` (branch A) / `ctx_now + target_role` (branch B) | `source_aware (P, 256)`, `target_aware (P, 256)` | jointly trained | Parallel cross-attn: A asks "which target does each source pick?", B asks "which sources might target each target?" |
| **L4 JointRoleAttention** | `concat[source_aware + source_role_l4, target_aware + target_role_l4]` as a 2P-token sequence | `source_joint (P, 256)`, `target_joint (P, 256)` | jointly trained | One self-attention pass over the 2P sequence, then split halves back — lets source and target halves cross-condition on the same matchup |
| **PairHead** | `source_joint`, `target_joint`, **`ctx_now`** (skip from L2) | 5 heads (see "What each head predicts") | jointly trained, multi-task loss sum | Broadcasts trios to (P, P), runs the shared 2-layer trunk, then 5 single-Linear heads |

### Layer-to-layer signal flow

| Hop | Signal carried | Shape | Mechanism / what gets dropped |
|---|---|---|---|
| L0 → L1 | `planet_tok`, `comet_tok`, `fleet_tok` | (P, 256) × 2 + (F, 256) | where-scatter merges planet/comet into `entity_self`; fleet path stays separate |
| L1 → L2 | `entity_tokens` | (T, P, 256) | concat-fuse with `entity_self` enters via the concat side — soft residual through L1 |
| L2 → L3 | `ctx_now` (only **current** step) | (P, 256) | T=6 past steps are dropped here; L2's CLS exits the stack (not used downstream) |
| L3 → L4 | `source_aware`, `target_aware` | (P, 256) × 2 | Pre-concat into 2P-token sequence with fresh role embeddings |
| L4 → Head | `source_joint`, `target_joint` | (P, 256) × 2 | Split halves back from the 2P self-attn output |
| **L2 → Head (skip L3 + L4)** | **`ctx_now`** | **(P, 256)** | **Layer-skipping signal: direct `Linear(ctx_now)` into PairHead's 6-way concat — bypasses both L3 and L4** |

### Layer-skipping residuals (signals that bypass layers)

The standard ⊕-residuals listed in the per-layer table below stay within a single layer (around an MHA or FFN sub-block). Two **layer-skipping** signals are also load-bearing — they're concat-fed *into* downstream layers, but their effect is the same: gradient flow and feature preservation across the bypassed layers.

1. **`entity_self` → L1 fuse** *(skips L1's cross-attention)*. L0's per-entity token is `concat`-fused with L1's MHA output before the 2-Linear fuse MLP. Without this, L1's pure attention output would have to re-derive each planet's identity from the cross-attention residue alone. Acts like a soft residual through L1.

2. **`ctx_now` → PairHead** *(skips L3 and L4 entirely)*. L2's per-planet representation after planet↔planet self-attention is projected and concat'd into PairHead's 6-way feature stack alongside `source_joint`/`target_joint` from L4:

   ```
   L2 ─ ctx_now ──────────────────────────────────────────────►  PairHead
              │                                                     ▲ ctx_proj
              ▼                                                     │
            L3 DualRoleAttention                                    │
              │                                                     │
              ▼                                                     │
            L4 JointRoleAttention ──► source_joint / target_joint ──┘
   ```

   Rationale: L3 and L4 produce **role-specialized** (source-vs-target) tokens that drop some of the symmetric planet-context information L2 had. By preserving `ctx_now`, PairHead's trunk sees both views — role-aware (from L4) and role-agnostic (from L2) — when scoring pair compatibility. The 6-way concat is `[src_r, ctx_r, tgt_r, ctx_r, src_r⊙tgt_r, ctx_r⊙ctx_r]` (see `agents/transformer_v2/aggregator/pair_head.py:153–164`).

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
  Project to d_pair=d_model=256 (no down-projection by default; pass --d-pair 128
  to reproduce the legacy layout):
    src_r = Linear(256→256)(source_joint)                (B, P, 256)
    tgt_r = Linear(256→256)(target_joint)
    ctx_r = Linear(256→256)(ctx_now)

  Broadcast across (P, P):
    pair_feat = concat[src_r, ctx_r, tgt_r, ctx_r, src_r⊙tgt_r, ctx_r⊙ctx_r]  (B,P,P,1536)

  Shared trunk (no residual — purely sequential):
    Linear(1536 → 256) → GELU → Linear(256 → 256) → GELU     ─▶ trunk (B, P, P, 256)

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
