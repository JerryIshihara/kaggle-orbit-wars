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

## Milestones

Rough sequencing for the next round of model work, in order. Strike when complete; refine as scope changes.

| # | Name | Description | Status |
|---|------|-------------|--------|
| 1 | **L2 perceptor refinement** | Pretrain `CrossEntityAttention` on the cross_entity dataset with `T=10` history (`HISTORY_OFFSETS = (45, 40, …, 0)`, ~50-turn lookback) and the ~25-head Tier-1/2/3/4 supervision menu (frontier / spatial / leader_k / score_advantage_k / tactical-horizon). Wired through `notebooks/L2_crossheads_crossT10_d256_colab.ipynb`. | in progress |
| 2 | **L3 / L4 ablation** | Train the pair-action stack with `--skip-l34 --no-consolidator --freeze-perception` and compare against the May-21 baseline (full L1-L4 + shallow PairHead) on the same cache. Question: is the role-specialized stack load-bearing, or does `ctx_now → PairHead` suffice with deeper FiLM. Notebook: `Lall_noL34_PairHead_pairT10_d256_colab.ipynb`. Chunked pair-cache staging supported via `pair_cache_t10.manifest.json` + `.part_*`. | B-side at epoch 7 |
| 3 | **Finalize actor + critic block design** | Lock in: actor branch level (`ctx_now` post-L2 vs `source_joint/target_joint` post-L4), FiLM conditioner + head depth, source/target/frac contract; critic input (`glob` CLS, `player_state[:, 0]` from PlayerConsolidator, or a new dedicated head), value MLP shape; wire both into `PPOActorCritic.forward` / `freeze_for_phase`. | pending |
| 4 | **Actor + critic supervised pretrain** | With the finalized block design, pretrain heads against expert pair labels (actor: `pair_logits` BCE + `pair_frac` MSE) and per-player current/future/outcome targets (critic: Stage A `CurrentStateHead`, then v1 Future/Outcome/Matchup after cache rebuild). Goal: a warm-started `entity_encoder_best.pt` PPO loads with zero re-init. Notebooks: `Cons_StageA_currentstate_pairT6_d256_colab.ipynb` (critic) + `Lall_PairHead_pairT6_d256_colab.ipynb` (actor). | pending |
| 5 | **Action-effect / player-state transition pretrain** | Teach the model how actions interact with the environment and change strategic state before PPO asks it to improve actions. Add action-conditioned transition heads over `player_state` / `slot_state`: given current world state plus sampled/expert launches, predict short-horizon deltas in ships, planet ownership, production, survival/rank, and pairwise player advantage. Include negative/no-op and invalid-action contrast so the model distinguishes "action looked plausible" from "action actually improved the future state." Success: action-conditioned predictions beat no-action baselines across K={5,10,20,50}, and learned deltas correlate with realized player-state/value changes. | pending |
| 6 | **PPO rollout training** | Phase 0 self-play (frozen snapshot opponent per iter) + BC anchor on expert pair cache + soft cap on Bernoulli target count. Two-CPU protocol from `docs/PPO_TWO_CPU_PROTOCOL.md` with file-mediated gradient averaging A↔B. Success: winrate > 0.6 over 30 self-play iters, KL < 0.02, invalid-launch rate trending down. | pending |

**Milestone 1 L2 data staging:** `notebooks/L2_crossheads_crossT10_d256_colab.ipynb` defaults to sharded CSV staging for faster Colab prep (`DATASET_SHARDS = 8`, batch 256). Build/upload the shard set from an extracted local `data/datasets/{cross_entity,entity,fleet,planet}/` tree with:

```bash
SHARDS=8 UPLOAD=1 bash scripts/chunk_cross_entity_dataset.sh
```

This uploads `cross_entity_dataset_shard_00.tgz` through `cross_entity_dataset_shard_07.tgz` plus `cross_entity_dataset_shard.manifest.json` to `gs://orbit-wars-shipping/cross_entity/`. Set `DATASET_SHARDS = 0` in the notebook only when falling back to streaming the legacy single object `cross_entity_dataset.tgz`.

---

## Known issues / problem list

Live punch list of observed failures and gaps. Add new items as they're found; strike (`~~done~~`) or remove once fixed. Cite a replay / seed when possible so we can re-check after a change.

1. **Fleets miss the target and waste resources** — happens most often on comet targets, but also static planets near the board edge. The lead-aim estimate puts the fleet on a path that misses the moving target by more than its radius, so the fleet flies past, exits the board, and gets removed. Each miss spends the source's surplus for no return. Likely contributors: (a) `_lead_aim_comet` falls back to the comet's current position when the iterative ETA outruns the path window; (b) for fast-moving sources/targets the integer step rounding compounds; (c) the multi-target inference rule (`pair_logits > 2.0`) can fire low-confidence cells whose trajectories were never validated by `plan_launch`. Need to add a per-launch validation gate in `_target_to_moves` that hard-rejects when `plan_launch.ok=False` instead of falling back to a learned-frac launch.
2. **Comet-borne garrisons carried out of the map** — when a comet's path runs off the board, ships sitting on that comet at expiry vanish with it. The agent doesn't pre-emptively evacuate / launch from a comet whose path is about to end, so any garrison the agent built up there (or any planet captured *via* a comet) is lost the moment the comet expires. Fix candidates: (a) feature the comet's remaining-path-length and tail-distance-to-edge in `comet_features`; (b) add an evacuation rule in the runner that forces a launch from any comet within K turns of expiry; (c) penalize stationing ships on short-lived comets at the policy / scoring level.
3. **Agent ignores simple targets** — undefended neutrals or weakly-held enemies adjacent to the agent's own planets sometimes get no launch even though `physical_v4` would grab them every time. Suspected causes: `pair_logits > 2.0` threshold is calibrated for the average snapshot density and may suppress confident-but-not-extreme cells in low-action turns; the model was trained on bow+Ebi who play a tighter style than physical_v4. Could fix by lowering the threshold late-game, or by adding a "must-fire" fallback when no cells clear the threshold (e.g., fall back to top-1 per source, or to the flat-`argmax(P²)` cell if its logit exceeds a softer threshold).
4. **Model can't recognize blocking planets on the launch path** — the agent picks a (source, target) pair whose straight-line trajectory passes through an intervening planet (own, neutral, or enemy). The intercepting body absorbs the fleet, so the intended target is never reached and the source's ships are spent on the wrong planet. Today the runner relies on `plan_launch` *after* the model picks the pair, but the LEARNED scorer has no feature describing "is there a planet sitting on the ray from src to tgt?". The relation tokens at L1 are `[fleet_tok ‖ src_planet ‖ tgt_planet]` — they don't include third-party planets that geometrically sit between the pair. Fix candidates: (a) add a per-(s, t) geometric feature (count / nearest-distance / soonest-blocker-arrival) and feed it into PairHead via `pair_scalars`; (b) at training time, mask out or down-weight cells where `plan_launch.reason == "wrong_planet_*"` so the model learns to score them low rather than just being rejected at inference.
5. ~~**Miss-rate calculation diverges from env's actual outcomes**~~ — fixed in `utils/logger.py:trace_launch_motion` (2026-05-20). The aggregator now consumes `trace_fleets` outcomes directly (the env's ground-truth fleet lifecycle) instead of re-simulating physics. Mapping: `captured`/`reinforced`/`annihilated` → `ok`; `destroyed_sun` → `sun`; `out_of_map` → `boundary`; `unknown` → `unknown`. Multi-target moves the env rejected (running ship pool exhausted) produce no record now (FIFO match against fleets via `(owner, launch_step, from_id)`). Verified on seed=1729 transformer_v2 vs physical_v4: analyzer reports `boundary=44, sun=11` matching env's `out_of_map=44, destroyed_sun=11` exactly. `_first_collision_for_launch` is no longer called — kept as an optional diagnostic helper. (The original symptom was 0% reported miss rate vs ~15% real — caused by cumulative orbital-rotation drift between the local simulator and the env's absolute-angle planet positions.)
6. **Performance degrades on large maps** — agent plays competently on seeds with fewer planets (~16-24 real planets, common static-heavy generations) but stops scaling on maps with the maximum number of planet groups (~40 planets, multiple orbital rings + comet spawns). Hypotheses: (a) **training distribution skew** — bow+Ebi's 555 replays may under-represent maximal-planet seeds, so the model never learned how to spread surplus across many fronts; (b) **per-cell pair_logits calibration shifts with P** — with `pos_weight=600` BCE trained against an average of ~30 valid cells per source, the absolute logit magnitude depends on local pair density, so the `> 2.0` inference threshold may be miscalibrated when the planet count doubles (more competing targets dilute confidence); (c) **L2/L4 attention saturation** — the planet↔planet self-attention sees 40+ tokens (vs ~16-20 typical) and may not have learned to allocate attention budget across that many entities cleanly; the model was trained with `max_planets=64` padding but the *actual* P distribution in training data is skewed low; (d) **action-budget mismatch** — the multi-target threshold rule scales linearly with active source count, so a 40-planet map can issue 10+ launches/turn while bow+Ebi typically issued 2-3; the model never saw rollouts with that launch density. Investigation candidates: bucket the eval-seed panel by P and report win rate per bucket; histogram pair_logits by snapshot P at val time to confirm the calibration shift; check whether physical_v4 also degrades on large maps (if yes, it's an env-difficulty axis; if no, it's a learning gap).
7. ~~**Eval seed panel was not actually deterministic**~~ — fixed in `utils/runner.py:run_match` (2026-06-02). The Kaggle config seed was passed through, but `orbit_wars.py` also uses module-level Python `random`, so two machines could report the same seed list while generating different maps. `run_match()` now patches `kaggle_environments.envs.orbit_wars.orbit_wars.random` to a local `random.Random(seed)` for the episode, then restores the module binding; this keeps map/comet RNG deterministic without mutating process-global Python `random`. Machine B was also pinned back to `kaggle-environments==1.28.1` to match Machine A. Re-check: the 5-game latest L3/L4 PairHead eval vs `physical_v4` now matches exactly on A and B for seeds `[5199, 2083, 3493, 1649, 3233]`, including map hashes and outcomes: 3 wins / 0 draws / 2 losses.
8. **PairHead actor over-reinforces same-owner corridors and loses mid-game production control** — in the seed-fixed latest model eval (`Lall_L3L4_novalue_top4T10_dropbad`, replays under `data/runs/play_Lall_L3L4_vs_physical_v4_seedfixed_A_20260602/replays`), the agent's opening is not the main failure. It expands, then falls into high-volume self-shipping loops while `physical_v4` takes contested production. Loss seed 1649: at t75 the agent is still even/ahead (`15 planets / 45 prod` vs `15 / 35`), but by t150 it is behind (`10 / 28` vs `22 / 64`) and finishes at `2 / 6` vs `30 / 86`; of 157,899 launched ships, 149,070 went to same-owner targets and only 4,337 to enemy targets. Loss seed 3233 is similar: t150 `7 / 24` vs `17 / 64`, final wipe; of 143,259 launched ships, 139,678 went same-owner and only 1,722 to enemy targets. Top loop corridors are symmetric shuttles such as `6->22->6`, `22->2->22`, `21->9->21`, and `9->5->9`. Fix candidates: ownership-conditioned same-owner target gating unless a defense need is detected, per-corridor cooldown / top-k cap, an explicit action-type head (`expand` / `attack` / `reinforce` / `defend` / `no-op`), and a value or map-control signal so the policy learns when to switch from reinforcement to denying enemy production.
9. ~~**BC pretrain logits were calibrated for independent grid BCE, not PPO's action factorization**~~ — fixed for new training runs in `agents/transformer_v2/pretrain/entity_encoder.py` and `agents/transformer_v2/ppo/loss.py` (2026-06-02). The issue was real: supervised actor pretrain optimized every `pair_logits[s,t]` as an independent Bernoulli, while PPO samples a NOOP-or-source categorical from row-max logits and then Bernoulli target bits on the chosen source row. New pretrain keeps the original pair BCE for greedy threshold inference but adds PPO-alignment losses by default: source CE over `[NOOP, legal learner-owned sources]` and target-row BCE on the canonical expert source. The PPO BC anchor now also receives the same source legality mask, so it no longer trains non-learner-owned rows as possible source choices. Optional `--ppo-frac-logit-weight` adds a row-share logit target for PPO's normalized frac allocation; default stays `0.0` to preserve greedy inference's absolute-fraction target.



Active learned line lives under [`agents/transformer_v2/`](agents/transformer_v2/README.md). It's a 4-layer transformer stack on top of 3 frozen per-entity specialist encoders, capped by a shared 2-layer trunk that feeds **2 output heads** through an L1-conditioned FiLM modulation block. Heuristic and legacy agents are kept in [`agents/heuristic/`](agents/heuristic/) and [`agents/archive/`](agents/archive/) respectively.

### Architecture

```
L0 (frozen, per-entity MLP specialists, ~374k params)
   ├─ PlanetEncoder       (18 → 256)
   ├─ CometEncoder        (123 → 256)   ─► where(is_comet, ...) ─► entity_self (B, T=6, P, 256)
   └─ FleetEncoder        (24 → 256)                                            │
                                                                                ▼
L1 PlanetEntityEncoder       cross-attn: planets ←→ relation-aware fleets       (658k)
                              ([fleet_tok ‖ source_planet_tok ‖ target_planet_tok])
                                                                                ▼ entity_tokens (B, T, P, 256)
L2 CrossEntityAttention      planet ↔ planet self-attention, multi-step over T=6,
                              learned CLS, 2-layer Pre-LN encoder                (1.05M)
                                                                                ▼ ctx_now (B, P, 256)
L3 DualRoleAttention         parallel source→target / target→source branches    (528k)
                                                                                ▼ source_aware, target_aware
L4 JointRoleAttention        concat 2P, 1-layer self-attn, split back           (528k)
                                                                                ▼ source_joint, target_joint
PairHead                     2-layer shared trunk → FiLM conditioner → 2 heads  (929k)
                              Linear(1536 → 256) → GELU → Linear(256 → 256) → GELU     ← trunk
                                                  │
                                                  ▼  h[s, t] (B, P, P, 256)
                              FiLM:  γ, β = MLP([L1_src ‖ L1_tgt ‖ pair_type_emb])
                                     h_film = h + α · (γ · h + β)        ← γ, β init=0; α=1 (identity)
                                                  │
                                          ┌───────┴───────┐
                                          ▼               ▼
                                    pair_logits        pair_frac
                                    (B, P, P)          (B, P, P)
                                    BCE pw=600         MSE on sigmoid
                                    (loss masked      (loss masked to
                                     by pair_valid)    positive cells)
```

**Total trainable params: ~3.70M.** Loss is the sum of two terms: per-cell BCE on `pair_logits` (masked by `pair_valid`, `pos_weight=600`) + MSE-on-sigmoid for `pair_frac` (masked to positive cells where `pair_ships > 0`). The earlier auxiliary heads (`source_act` / `target_aim` / `glob_act`) were removed in 2026-05-20 — at inference the runner only consumes `pair_logits` + `pair_frac`, and the aux heads were fighting the joint optimizer for capacity without affecting actions.

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
| **PairHead** | `source_joint`, `target_joint`, **`ctx_now`** (skip from L2), **`l1_tokens`** (skip from L1, FiLM cond), `pair_type_ids` (27-way source/target type) | `pair_logits (B, P, P)`, `pair_frac (B, P, P)` | jointly trained, 2-term loss | Broadcasts trios to (P, P), runs the shared 2-layer trunk, applies L1-conditioned FiLM (`h_film = h + α·(γ·h + β)`, identity-init), then 2 single-Linear heads |

### Layer-to-layer signal flow

| Hop | Signal carried | Shape | Mechanism / what gets dropped |
|---|---|---|---|
| L0 → L1 | `planet_tok`, `comet_tok`, `fleet_tok` | (P, 256) × 2 + (F, 256) | where-scatter merges planet/comet into `entity_self`; fleet path stays separate |
| L1 → L2 | `entity_tokens` | (T, P, 256) | concat-fuse with `entity_self` enters via the concat side — soft residual through L1 |
| L2 → L3 | `ctx_now` (only **current** step) | (P, 256) | T=6 past steps are dropped here; L2's CLS exits the stack (not used downstream) |
| L3 → L4 | `source_aware`, `target_aware` | (P, 256) × 2 | Pre-concat into 2P-token sequence with fresh role embeddings |
| L4 → Head | `source_joint`, `target_joint` | (P, 256) × 2 | Split halves back from the 2P self-attn output |
| **L2 → Head (skip L3 + L4)** | **`ctx_now`** | **(P, 256)** | **Layer-skipping signal: direct `Linear(ctx_now)` into PairHead's 6-way concat — bypasses both L3 and L4** |
| **L1 → Head (skip L2 + L3 + L4)** | **`l1_now`** (entity_tokens at current step) + **27-way `pair_type_ids`** | **(P, 256) + (P, P)** | **FiLM conditioner: L1_src ‖ L1_tgt ‖ pair_type_emb → γ, β. Modulates the post-trunk `h[s, t]`. Bypasses L2/L3/L4 to carry L1's local-tactical-state directly to the score.** |

### Layer-skipping residuals (signals that bypass layers)

The standard ⊕-residuals listed in the per-layer table below stay within a single layer (around an MHA or FFN sub-block). Three **layer-skipping** signals are also load-bearing — they're concat-fed (or FiLM-fed) *into* downstream layers, but their effect is the same: gradient flow and feature preservation across the bypassed layers.

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

   Rationale: L3 and L4 produce **role-specialized** (source-vs-target) tokens that drop some of the symmetric planet-context information L2 had. By preserving `ctx_now`, PairHead's trunk sees both views — role-aware (from L4) and role-agnostic (from L2) — when scoring pair compatibility. The 6-way concat is `[src_r, ctx_r, tgt_r, ctx_r, src_r⊙tgt_r, ctx_r⊙ctx_r]` (see `agents/transformer_v2/aggregator/pair_head.py`).

3. **`l1_tokens` → PairHead FiLM** *(skips L2, L3, and L4 entirely)*. L1's per-planet token — which fuses L0 entity self-encoding with the **fleet relation context** (inbound/outbound fleets, ship counts, ETAs) — feeds the FiLM conditioner between trunk and the two heads:

   ```
   L1 ─ l1_tokens ───────────────────────────────────────────►  PairHead.film_proj
              │       (broadcast across (P, P), concat with 27-way pair type embedding)
              ▼                                                     ▲
            L2 ─ ctx_now ─► L3 ─► L4 ─► trunk ─► h[s, t] ──────────┘
                                                                    │
                                          γ, β = chunk(film_proj(cond))
                                          h_film = h + α · (γ · h + β)
                                          α: learnable scalar scale, init=1
   ```

   Rationale: the trunk's role-aware representation (from L4) and role-agnostic context (from L2) capture **strategic** state — "is this a good source–target geometry?", "are the rest of the planets aligned to support this move?". L1 carries **tactical** state that's already been compressed by the time it reaches L2 — "this source already has outgoing fleets", "this target has an enemy inbound", "this planet is being contested". FiLM lets the head tilt scores per-(s, t) based on that local state without disrupting the trunk's pre-trained pathway. The identity init (`γ = β = 0` at step 0, so `h_film = h` exactly) means FiLM behaves as a no-op initially; `α` starts at 1 so the conditioner still receives gradients from the first backward pass while legacy ckpt behavior remains bit-stable.

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
PairHead — shared 2-layer trunk + L1-conditioned FiLM + 2 single-Linear heads
═══════════════════════════════════════════════════════════════════════════════
  Project to d_pair=d_model=256 (no down-projection by default; pass --d-pair 128
  to reproduce the legacy layout):
    src_r = Linear(256→256)(source_joint)                (B, P, 256)
    tgt_r = Linear(256→256)(target_joint)
    ctx_r = Linear(256→256)(ctx_now)

  Broadcast across (P, P):
    pair_feat = concat[src_r, ctx_r, tgt_r, ctx_r, src_r⊙tgt_r, ctx_r⊙ctx_r]  (B,P,P,1536)

  Shared trunk (no residual — purely sequential):
    Linear(1536 → 256) → GELU → Linear(256 → 256) → GELU     ─▶ h (B, P, P, 256)

  FiLM conditioner (skip-from-L1 + 27-way source/target type embedding):
    pair_type[s,t] = source_type[s] * 9 + target_relation[t] * 3 + target_type[t]
      source_type, target_type ∈ {0=static, 1=orbital, 2=comet}
      target_relation ∈ {0=enemy, 1=neutral, 2=own}
    cond[s,t] = concat[L1_src, L1_tgt, Embed27(pair_type[s,t])]   (B,P,P, 2·256 + 32)
    γ, β      = chunk(Linear(544 → 256) → GELU → Linear(256 → 512)(cond), 2, dim=-1)
    h_film    = h + α · (γ · h + β)                                  α: scalar Param, init=1

    At init γ, β = 0 (film_proj output Linear is zero-init), so h_film = h
    exactly. Alpha is non-zero so gradients flow into the conditioner; this
    keeps the trunk's pre-trained pathway intact when legacy ckpts load while
    making FiLM trainable immediately.

  2 heads (each Linear(256 → 1) on h_film, no residual):
    pair_head       : h_film[s,t]  → pair_logits  (B, P, P)
    pair_frac_head  : h_film[s,t]  → pair_frac    (B, P, P) raw logit, loss sigmoids
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
| PairHead trunk | No | Purely sequential MLP |
| **PairHead FiLM gate** | **Yes ⊕** | `h_film = h + α · (γ · h + β)`; `γ, β` zero-init and `α=1` — identity at start, trainable immediately |
| PairHead pair_head / pair_frac_head | No | Per-head Linear on h_film |

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
| `pair_frac`     | raw logit; sigmoid → fraction of source's ships sent to target | `pair_ships[s, t] / source_ships_before_launch[s]`; masked to positive cells only |

The runner consumes both at inference: pair_logits chooses the (s, t) cells to fire on (currently `pair_logits > 2.0` threshold, multi-target per source), and `sigmoid(pair_frac)` sizes each launch as a fraction of the source's ships.

---

## Reinforcement-learning layer (PPO)

PPO is added on top of the supervised model as a wrapper — not a rewrite. The
supervised L0–PairHead stack stays as the world-perception + action backbone;
PPO contributes the policy-gradient training loop, three new heads, and a
discipline about which layers are allowed to move per phase. The full bring-up
protocol lives in [`docs/PPO_TWO_CPU_PROTOCOL.md`](docs/PPO_TWO_CPU_PROTOCOL.md);
the algorithm-level sketch lives in [`PPO_PSEUDOCODE.md`](PPO_PSEUDOCODE.md).
This section is the design overview.

> **Current PPO critic design:** the critic branches from **L1**, not actor L2.
> The old 3-Linear `value_head(glob)` path is debug-only. Production PPO uses
> `actor L1 tokens -> separate critic L2 -> PlayerConsolidator -> ValueDecoder`.

### What PPO adds

**Actor = the existing `pair_logits` and `pair_frac` outputs from `PairHead`.
No new actor heads.** Critic is a separate `CrossEntityCriticModel` attached
to actor L1. It runs critic L2 + `PlayerConsolidator` and trains the
`ValueDecoder`; actor L2/L3/L4 are not the critic attachment point.

**Preconditions on the supervised PairHead ckpt that PPO bootstraps from:**

| Module | Minimum depth | `pair_head.py` flag |
|---|---|---|
| FiLM-conditioned adaptor | ≥ 3 Linear stages | `conditioner_n_layers >= 2` |
| `pair_logits` head | ≥ 3 Linear stages | `head_n_layers >= 3` |
| `pair_frac` head   | ≥ 3 Linear stages | `head_n_layers >= 3` |

PPO trains the actor *through* these modules in Phase 1+, so the capacity
has to be in the ckpt — depths can't be added post hoc. Any candidate
supervised ckpt trained with shallower defaults must be retrained before
starting PPO. Details in "PairHead minimum depths" below.

```
                 INPUT  (per learner turn)
       planet_features  comet_features  fleet_features  masks + routing
                                  │
   ─────────────────────────────── │ ──────────────────────────────────────
   SUPERVISED STACK                ▼
                ┌─────────────────────────────────────────┐
   L0           │  PlanetEnc │ CometEnc │ FleetEnc        │ frozen always
                │  18→256    │ 123→256  │ 24→256          │
                │  └─── entity_self (B,T,P,256) ──────────┐
                │              │                          │
   L1           │  PlanetEntityEncoder  cross-attn        │ Phase 0/1/2: frozen
                │              ▼                          │
                │           entity_tokens (B,T,P,256)     │
                │              │                          │
   L2           │  CrossEntityAttention + CLS, T=6 steps  │ Phase 0/1/2: frozen
                │              │                          │ Phase 3 only with evidence
                │       ┌──────┴──────┐                   │
                │       │             │                   │
                │   glob (B,256)   ctx_now (B,P,256)      │
                └───┬───┴───────────────────┬─────────────┘
                    │                       │
                    │           ╔═══════════│════════════════════════════╗
                    │           ║ post-L2   ▼  (ActionLearner pathway)   ║
                    │           ║   PlayerContextLearner  (planned/stub) ║
                    │           ║   StrategyLearner       (planned/stub) ║
                    │           ║   L3 DualRoleAttention                 ║
                    │           ║   L4 JointRoleAttention                ║ Phase 0: trunk frozen
                    │           ║   PairHead  trunk + FiLM               ║ Phase 1+: trainable
                    │           ║          │                             ║
                    │           ║   ┌──────┴──────┐                      ║
                    │           ║   ▼             ▼                      ║
                    │           ║ pair_logits  pair_frac                 ║ Phase 0: output
                    │           ║  (B,P,P)     (B,P,P)                   ║ Linears trainable
                    │           ╚═════│═════════════│════════════════════╝
                    │                 │             │
   ═══════════════ │ ══════════════ │ ═══════════ │ ════════════════════
   PPO WRAPPER      │                 │             │
                    ▼                 ▼             ▼
            ┌─────────────────┐ ┌──────────────────────────────────────┐
            │   CRITIC        │ │   ACTOR (no new heads)               │
            │                 │ │                                      │
            │ value_head      │ │  Categorical(logits =                │
            │  Linear(256→256)│ │    where(legality_mask,              │
            │  GELU           │ │          pair_logits.flatten(),      │
            │  Linear(256→256)│ │          −inf))                      │
            │  GELU           │ │       ▼  sample (s, t)               │
            │  Linear(256→1)  │ │                                      │
            │   ↑ NEW (~132k) │ │  LogitNormal(loc = pair_frac[s,t],   │
            │                 │ │              σ = 0.35 fixed)         │
            │  V(s) scalar    │ │       ▼  sample frac via sigmoid(z)  │
            └────────┬────────┘ └───────┬──────────────────────────────┘
                     │                  │
                     │                  ▼
                     │     project_to_env: plan_launch(s, t,
                     │       ships = round(clamp(frac,.02,1)·src.ships))
                     │       ┌────────┴────────┐
                     │       │                 │
                     │   ok=True             ok=False
                     │       │                 │
                     │       ▼                 ▼
                     │   [s, angle, ships]   NOOP + invalid penalty
                     │                       (NEVER resample;
                     │                        stored logp stays)
                     │
   ═════════════════ │ ════════════════════════════════════════════════
   GRADIENT FLOW    (during ppo_update)
                     │
   value_loss   ───► critic ValueDecoder
                       ↑
                    critic L2 + PlayerConsolidator
                       ↑
                    actor L1 tokens

   policy_loss  ───► (NOT into critic ValueDecoder)
   entropy      ───► pair_logits, pair_frac (output Linears) ─► PairHead trunk
   bc_loss      ───►                                              (Phase 1+)
                                                                   │
                                                            ─► L3 / L4 / FiLM
                                                                   │  Phase 1+
                                                            ─► PlayerContext / Strategy
                                                                   │  Phase 1+
                                                            ─► L2  Phase 2 only

   Critic branches from actor L1 and owns critic L2 + PlayerConsolidator.
   Actor L2/L3/L4 are not the value scaffold.
   BC anchor (pair_cache → bce(pair_logits, expert)) lives ONLY on A's
   gradient in distributed Phase 1+; bc_coef doubles 0.05 → 0.10 to keep
   effective weight after averaging.
```

**Why this split:**
- **Decoupled gradients.** Value-loss enters the critic `ValueDecoder`, not
  L3/L4/PairHead. Actor-loss never enters the critic.
- **L1 is the critic branch point.** It preserves entity/fleet perception
  without forcing value learning through actor-side L2/L3/L4 source-target
  specialization.
- **Critic ValueDecoder.** The critic's own L2 + PlayerConsolidator build the
  player-level state; PPO fine-tunes only the ValueDecoder in Phase 0.
- **Cleaner Phase-2 transition.** When L2 eventually unfreezes, value-loss
  on L2 is a single well-conditioned scalar-regression signal — much
  friendlier than the multi-head action zoo PairHead would dump on L2.

### PairHead minimum depths (FiLM + action heads)

PPO trains the actor *through* PairHead's FiLM conditioner and the two
output heads in Phase 1+. Each of those modules needs enough capacity to
adapt strategy, not just apply small additive corrections. The supervised
ckpt must be trained with the depths below — depths can't be added later.

**FiLM conditioner — `conditioner_n_layers >= 2` (≥ 3 Linears):**

```text
conditioner_n_layers = 2  →  Linear(cond_in → H) → GELU →
                             Linear(H → H)       → GELU →
                             Linear(H → 2·trunk_h)        # γ + β, zero-init
                                                           = 3 Linears total
```

`conditioner_n_layers = 1` (2 Linears) is too thin once PPO starts adapting
strategy through the FiLM and is **not** supported as a bootstrap source.

The identity-init for the final Linear (`γ = β = 0` at step 0) is preserved
regardless of depth, so loading a deeper ckpt is bit-stable on the first
forward.

**`pair_logits` and `pair_frac` heads — `head_n_layers >= 3` (3 Linears each):**

```text
head_n_layers = 3  →  Linear(trunk_h → H) → GELU →
                      Linear(H → H)       → GELU →
                      Linear(H → 1)                       = 3 Linears total
```

`head_n_layers = 1` (single Linear) was the legacy supervised default — it
is **not** enough capacity for PPO to fine-tune the action distribution on
top of strategy/role-aware tokens. With `head_n_layers = 3`, the actor heads
can express richer per-`(s, t)` decisions when L3/L4/PairHead-trunk gradients
land in Phase 1+. The same `nn.init.normal_(std=0.02)` initialization on the
final Linear is preserved, so initial pair_logits remain near-chance and
training is stable.

**Bootstrap rule:** any supervised PairHead checkpoint PPO consumes must
have been trained with `--conditioner-n-layers 2` (or higher) **and**
`--head-n-layers 3` (or higher). Train (or retrain) the supervised ckpt
with both flags before starting PPO.

### Freeze schedule (three phases)

PPO is brought up conservatively — value + actor output Linears first, action
trunk next, perception last. Each phase only starts after the previous one's
eval is stable.

**PPO never trains anything earlier than L4.** L0–L3 (and the planned
PlayerContext / Strategy) are frozen for the entire PPO run; there is no
Phase 2 perception-unfreeze.

| Phase | Trainable | Frozen | New params | Goal |
|---|---|---|---:|---|
| **0 — Plumbing** | `value_head` (3-Linear MLP) + PairHead `pair_logits` head + PairHead `pair_frac` head | L0–L3, PlayerContext / Strategy, L4, PairHead trunk + FiLM | ~132 k | Prove PPO update + GAE + eval gate + distributed loop work without destroying the BC policy. |
| **1 — Action adaptation** | + L4 JointRoleAttention + PairHead trunk + FiLM | L0–L3, PlayerContext / Strategy | + ~1–2 M | Action adaptation strictly **after L3**. L3's role-aware tokens stay fixed; L4 re-mixes them under PPO. Caps PPO's ability to disturb the perception/role substrate. |
| **—** (no Phase 2) | (no further unfreeze) | L0–L3 always | — | If diagnostics show perception is stale, retrain the supervised PairHead and restart PPO from the new ckpt — do not unfreeze L0-L3 in flight. |

The actor stays compatible with the supervised PairHead: BC anchor loss
(`bc_coef · BCE(pair_logits, expert_labels)`) is mixed in throughout PPO to
prevent collapse, sampled from the existing `_pair_cache`.

### Sampling and env projection

Phase 0 samples exactly one (source, target) pair per learner turn from a
`Categorical` over the flattened `P × P` `pair_logits` with the legality mask
applied. **There is no explicit NOOP slot.** NOOP only emerges when
`plan_launch` rejects the sampled pair (out-of-surplus, blocking planet,
boundary, sun, …); the rollout then earns an invalid-launch penalty. This
penalty does double-duty: it teaches the policy to put low logit on
infeasible cells, which is the same thing as learning NOOP via avoiding
all-infeasible turns.

`frac` is sampled from `LogitNormal(loc=pair_frac[s,t], σ=0.35_fixed)` — σ
is a CLI-tunable hyperparameter, not a learned parameter. Stored
`frac_sample = clamp(sigmoid(z), 1e-4, 1−1e-4)` (numerical only); the
launch-side clamp `[0.02, 1.00]` is applied only at env projection. PPO
recomputes logprob from the stored value, so the launch clamp **must not**
be applied at sample time — otherwise the stored logp would not match the
recomputed logp at update time and the PPO ratio would silently desync.

Invalid actions are **never resampled** — the original pair stays in the
buffer with its original logprob.

**Multi-launch gap:** the supervised runner fires multiple sources per turn
via `pair_logits > 2.0` threshold; PPO v1 here fires one. Track
`emitted_launch_rate` + launch-miss matrix; upgrade to a per-source
Bernoulli or autoregressive top-K contract in Phase 1+ if eval shows it
matters.

### Self-play pool

Rollouts are **100% self-play**. Opponents come only from the frozen
**promoted** lineage plus `transformer_v2_baseline` (the supervised May-20
ckpt — the load-bearing floor). The current learner `policy_vK` never plays
itself; doing so would train against a moving target whose logprobs aren't
tracked.

```
per-episode opponent sampling:
  50%  latest promoted PPO ckpt   (baseline fallback until first promotion)
  30%  transformer_v2_baseline    (the floor — winrate must stay near 50%)
  20%  uniform older promoted     (prevents cyclic chasing; baseline fallback)
```

Pool cap `N_pool = 8`. Promoted only — failed iters are kept on disk for
debug but not added to the rollout pool.

### Training topology — two CPU machines

There is no GPU. Two CPU machines run in parallel:

```
Machine A  (the dev box, /Users/agent/dev/kaggle-orbit-wars)
  • owns run dir, advantages, optim state, promotion gate, archive
  • ~50% of rollout episodes
  • Phase 0: full training locally
  • Phase 1+: half of each paired minibatch + applies the optimizer step

Machine B  (Tailscale 100.109.180.124, same workdir path)
  • ~50% of rollout episodes
  • Phase 1+: half of each paired minibatch + writes grad to file (no optim step)
  • cold storage for archived rollouts, out-of-pool checkpoints, eval JSONs

Connectivity: A → B SSH works; B → A is not reachable.
All sync is rsync over SSH, initiated from A.
```

Phase 1+ training is data-parallel via **file-mediated gradient averaging**
(one-way SSH rules out NCCL all-reduce). Per minibatch:

```
A: forward+backward on mb_A         ┐  parallel
B: forward+backward on mb_B → file  ┘
A: rsync-pull grad_B; grad = (grad_A + grad_B) / 2
A: optimizer.step(); write new weights to file
B: rsync-pull weights; reload; proceed to mb_{M+1}
```

The BC anchor loss only lives on A's gradient (the 47 GB `_pair_cache` stays on
A); A doubles its `bc_coef` from `0.05 → 0.10` so the averaged gradient sees
the same effective `0.05` coefficient. CPU-perf optimizations (batched envs
per worker, `torch.compile(mode="reduce-overhead")`, shared opponent model
instances, eval split 16/16 between A/B) are detailed in
[`docs/PPO_TWO_CPU_PROTOCOL.md`](docs/PPO_TWO_CPU_PROTOCOL.md) under "CPU
performance".

### Promotion gate

After every PPO iteration, deterministic eval runs against three mandatory
opponents (`previous_promoted`, `transformer_v2_baseline`, `physical_v4`) on
SEEDS_QUICK = 32 seeds × both seats. A candidate `policy_v(K+1)` is promoted
only if all five conditions hold:

```
1. winrate vs previous_promoted ≥ 50% with non-negative paired seed-seat score
2. winrate vs transformer_v2_baseline ≥ previous_promoted's baseline winrate − 2pp
3. winrate vs physical_v4 drops by ≤ 5pp
4. invalid_launch_rate ≤ 1.1 × previous_promoted's rate
5. no new high-severity launch-miss-matrix regression
```

Non-promoted iters still archive the checkpoint for debug but do not enter the
self-play pool. A startup calibration eval (`baseline` self-eval) bootstraps
the relative thresholds for iter 1 and also catches eval-harness bugs
(baseline-vs-baseline winrate must be `0.50 ± 0.05`).

### Detail references

| Doc | What's in it |
|---|---|
| [`docs/PPO_TWO_CPU_PROTOCOL.md`](docs/PPO_TWO_CPU_PROTOCOL.md) | Full two-machine protocol: roles, connectivity, rsync recipes, CPU-perf budget, distributed grad-averaging, archive policy, policy/action contract, reward shaping, hyperparameters, freeze schedule details, BC anchor math, evaluation gate, failure-mode table, startup calibration, implementation checklist, practical first-run config |
| [`PPO_PSEUDOCODE.md`](PPO_PSEUDOCODE.md) | Algorithm sketch: coordinator + peer drivers, batched rollout worker, action sampling + env projection, GAE, `ppo_update_local` (Phase 0) and `ppo_update_distributed` (Phase 1+), self-play pool, promotion gate, archive step, train-log field reference |
| [`agents/transformer_v2/ppo.py`](agents/transformer_v2/ppo.py) | Currently a stub. Will be the public CLI wrapper once `ppo_learner.py`, `ppo_rollout_worker.py`, and `ppo_train_peer.py` exist |

---

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
    --d-model 256 --batch-size 16 --epochs 30 --lr 5e-5 \
    --pair-pos-weight 600 \
    --device cuda
```

Per-epoch logging prints the two active heads (`pair_logits`, `pair_frac`) with train/val loss, `recall_true`, `recall_false`, `pos_frac`, and pair-ranking metrics (`recall_at_{1, 5, 10}`, `pair_recall_at_{1, 5, 10}`, `row_recall_at_{1, 5, 10}`). See [`notebooks/train_entity_encoder_colab.ipynb`](notebooks/train_entity_encoder_colab.ipynb) for the Colab variant — the bundle lives at `gs://orbit-wars-shipping/entity/`.

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
