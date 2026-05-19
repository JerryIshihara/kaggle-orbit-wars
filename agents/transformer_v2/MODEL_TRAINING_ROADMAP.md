# Transformer V1 Model / Training Roadmap

Last updated: 2026-05-15 (afternoon; covers two-stage ranker + inference UI)

This note summarizes the current Transformer V1 training iteration, the
bugs fixed during target-ranker bring-up, the latest target-rank results,
and the recommended roadmap for the next model revisions.

## Current Architecture

The current action target model is a ranker over candidate target planets.
It reuses the existing encoder stack and replaces the old pair-summary path
with direct attention inside the target head.

```text
fleet_features, planet_features
        |
        v
FleetEncoder + PlanetEncoder
        |
        v
PlanetEntityEncoder (L1)
  -> entity_now (B, P, d)
        |
        v
CrossEntityAttention (L2)
  -> ctx_now (B, P, d)
  -> glob    (B, d)
        |
        v
TargetRanker
  target_base = Linear([ctx_now, entity_now, glob, target_scalars])

  Stage A: target-to-source attention
    Q = candidate target tokens
    K,V = owned-source planet tokens
    key mask = ~src_valid
    diagonal mask excludes source == target

  Stage B: target self-attention
    each source-aware target attends over all real target candidates

  MLP scorer
    score_feat = [rank_ctx, source_aware, target_base, target_scalars]
    -> target_logits (B, P)
```

Important current defaults:

- `d_model = 64`
- `n_history = 3`
- `max_fleets = 1024`
- target candidates are all real planets: own, neutral, and enemy
- source candidates during training are broad owned-source planets:
  `owned by learner and ships > 0`
- runtime may tighten source legality with a surplus/launchable rule
- training force-includes the expert source/target labels to preserve BC
  supervision

The ranker intentionally has no `PairScoreHead`. The old pair head emitted
`(B, P, P)` pair logits, but target training consumed only a 5-scalar
column reduction per target. That threw away most per-source detail. Stage A
target-to-source attention now provides the source-conditioned target
summary directly.

## Iteration Summary

### 1. Pair-score / dataset upload failures

The Colab pair-score notebook initially failed because placeholders such as
`{ENCODER_CKPT}` were passed literally into Python commands. Separately,
the Colab upload/pull flow did not always ship the required tarballs.

Current state:

- `pack_for_gpu.sh` builds `code.tgz`, `data.tgz`, and `weights.tgz`.
- The target-rank notebook now checks required GCS objects before pulling.
- The current GCS objects are:
  - `gs://orbit-wars-shipping/code.tgz`
  - `gs://orbit-wars-shipping/data.tgz`
  - `gs://orbit-wars-shipping/weights.tgz`

### 2. Action mask sidecar bug

The action dataset added `_masks/<stem>.npz` sidecars containing:

- `src_valid`: owned-source planets
- `tgt_valid`: real planets

However, the existing uploaded `data.tgz` did not contain `_masks/*.npz`.
The first fallback implementation initialized both masks all-False and then
force-included only the gold target. This collapsed target training to one
candidate per row:

```text
avg_cand = 1.0
baseline_ce = 0.0
loss = 0.0
top1 = 1.0
```

That was a data/mask bug, not model performance.

Current fix:

- Missing mask sidecars fall back to `tgt_valid = planet_mask`.
- Missing source masks fall back to `src_valid = owned & ships > 0`.
- The ranker also repairs legacy gold-only masks by expanding targets back
  to `planet_mask` when multiple real planets exist.

### 3. Resume-from-checkpoint

`train_target_rank(args, ...)` accepts an `init_from=<ckpt>` path. It
loads all four encoders + the `target_ranker` state before
`unfreeze_all()` runs and the optimizer is built. **Optimizer state is
not carried** — pass a lowered `lr` for the continued phase. The
notebook (`train_target_rank_colab.ipynb`) now defaults to:

```
INIT_FROM_GCS = gs://orbit-wars-shipping/runs/Ebi_20260514-144051/target_rank_best.pt
LR            = 2e-4         # was 5e-4 (fresh init)
WEIGHT_DECAY  = 1e-4         # combat the train/val gap that opened ~ep 12
DROPOUT       = 0.10         # both Stage A and Stage B MHA blocks
```

Epoch-1 of a continue run should land at `val_top1 ≈ 0.44` (where the
prior run left off); if it regresses, resume isn't loading correctly or
the LR is too high.

### 4. Data integrity audits

Three classes of latent corruption were found and fixed during the
Stage-2 / Stage-3 bring-up:

1. **Fleet CSVs truncated mid-write.** Smoking gun:
   `fleet_75610892_4_2.csv` stopped at turn 409 while
   the replay ran through turn 438 (with ~750 fleets in flight).
   `scripts/build_encoder_dataset.py --audit-only` now cross-checks
   each fleet CSV's last turn against its replay's last
   **fleet-active** turn (not `len(steps) - K`, which over-flagged 78
   of Ebi's 434 stems that were actually fine).
2. **Action-mask sidecars absent from prior `data.tgz`.** The
   `_masks/*.npz` files that populate `src_valid` / `tgt_valid` were
   not in the shipped tarball. The build script's `already_processed`
   check now treats a missing sidecar as "not done" for the action
   dataset (in both strict and default modes), so a default
   `--datasets action` rebuild backfills them. The ranker stack also
   has a runtime fallback derived from planet ownership — never falls
   back to `mask_now` for `src_valid` (would allow enemy/neutral
   sources).
3. **`max_fleets=256` cap truncated the long tail.** Real max
   observed was 813 fleets per snapshot; 3,183 snapshots exceeded the
   cap. Defaults bumped to `max_fleets=1024` in
   `ActionSnapshotDataset`, `EntitySnapshotDataset`,
   `CrossEntitySnapshotDataset`, `prepare_dataset`, and the CLI.
   Old caches keyed by `_p64_f256_h3.pt` are still valid; new builds
   write `_p64_f1024_h3.pt`.

### 5. Target-ranker training result

Latest run:

```text
gs://orbit-wars-shipping/runs/Ebi_20260514-144051/
```

The corrected run is no longer degenerate:

```text
avg_cand ~= 29.8
baseline_ce ~= 3.377
```

Best validation point:

```text
epoch 17
val_loss      = 2.0458
val_top1      = 0.4455
val_top3      = 0.6891
val_top5      = 0.7734
val_logit_std = 1.3404
```

Curve summary:

```text
epoch  1: val_loss 2.7462, top1 0.2277, top5 0.5761
epoch 10: val_loss 2.2094, top1 0.3881, top5 0.7353
epoch 17: val_loss 2.0458, top1 0.4455, top5 0.7734
epoch 20: val_loss 2.0592, top1 0.4445, top5 0.7703
```

Use `target_rank_best.pt`, not `target_rank_last.pt`. Validation loss starts
to flatten or slightly regress after epoch 17 while train loss continues to
fall.

## Known Bottleneck

Representation probes showed that the frozen/intermediate representation
does not reliably preserve core tactical facts:

- positional signal survives reasonably well
- garrison / local counts are weak or lost
- `inbound_total_h10` is partly encoded before L2, then degraded after
  `CrossEntityAttention`
- per-slot inbound threat is weak even before L2

Interpretation:

The L2 cross-entity block is useful for global context, but it should not be
the only path by which local tactical facts reach action heads. Treat L2 as
contextual/global reasoning, and route raw/local facts around it.

## Current Design Principles

Keep these unless intentionally running an ablation:

1. Target candidates are all real planets.
   Own planets are valid targets because reinforcement moves exist.

2. Training source mask is broad owned-source, not runtime surplus-based.
   Tightening training to runtime surplus can discard expert launches.

3. `entity_now` and raw tactical scalars should bypass L2 into action heads.
   L2 can destroy some tactical facts.

4. `avg_candidate_count` and `uniform_ce_baseline` are mandatory sanity
   metrics.
   If `avg_cand == 1.0`, the target task has collapsed.

5. Compare target loss to `uniform_ce_baseline`, not to padded `log(64)`.

## Roadmap

### Phase 1: Strengthen the TargetRanker Without Rebuilding Encoders

Highest leverage next changes:

1. Add source-side tactical scalars into Stage A.

   Current target ranker has `target_scalars`, but source values are only
   represented through learned source tokens. Add explicit source facts:

   - source ships / garrison
   - production
   - owned-source flag
   - outbound commitment
   - inbound own/enemy h10
   - local friendly/enemy counts
   - nearest enemy distance

2. Add pairwise source-target geometry/battle features.

   Target choice is inherently relational. Stage A should know direct
   `(source, target)` facts rather than infer them from independent tokens:

   - distance source -> target
   - estimated travel time
   - source ships vs target garrison
   - target owner type
   - inbound threat to target
   - friendly reinforcement pressure

   Implementation options:

   - attention logit bias: `attn_logits += PairBias(source, target)`
   - pair feature projection added to values
   - side MLP score mixed into final target scorer

3. Keep raw/stat skips in the final scorer.

   The scorer should continue to see:

   ```text
   [rank_ctx, source_aware, target_base, target_scalars]
   ```

   Do not regress to `ctx_now`-only scoring.

### Phase 2: Make L2 Less Destructive

1. Add auxiliary heads on `entity_now` and/or `ctx_now`.

   Train representation-level tactical retention directly:

   - owner class
   - garrison / ships
   - inbound_enemy_h10
   - inbound_own_h10
   - local friendly/enemy support counts
   - nearest enemy distance

   This is the most direct way to stop L2 from dropping facts required by
   downstream action heads.

2. Add gated skip/fusion around L2.

   Instead of consuming raw `ctx_now` alone:

   ```text
   ctx_fused = LayerNorm(ctx_now + gate * entity_now)
   ```

   or:

   ```text
   ctx_fused = MLP([ctx_now, entity_now])
   ```

   Then pass `ctx_fused` into rank heads.

3. Consider shallower L2.

   If L2 is destructive at `d_model=64`, 3 Transformer layers may be too
   much. Try 1-2 layers with a stronger residual/fusion path before adding
   depth.

### Phase 3: Capacity / Temporal Context Experiments

1. Increase history from `T=3` to `T=5`.

   Expected effect:

   - more recent motion/fleet context
   - better inbound and ownership-change reasoning
   - almost no parameter increase except step embeddings

   Main cost is attention compute:

   ```text
   (5P)^2 / (3P)^2 = 25/9 ~= 2.8x L2 attention compute
   ```

2. Increase `d_model` from 64 to 128.

   Expected effect:

   - more representation capacity
   - possible improvement if the model is width-limited

   Costs:

   - many linear/attention/MLP weights scale roughly with `d^2`
   - `64 -> 128` is about 4x parameters in those blocks
   - activations are about 2x wider
   - existing `d_model=64` checkpoints are not directly compatible

3. Recommended sequence:

   ```text
   A. T=5, d_model=64
   B. add source-target pair bias/features
   C. add L2 auxiliary tactical losses
   D. T=5, d_model=128 if still underfitting
   ```

   Avoid jumping straight to `T=5, d_model=128` unless GPU budget is not a
   concern. It can be much slower and may require batch size 16 or 32.

## Training / Validation Checklist

Before trusting a run:

```text
avg_candidate_count > 1
uniform_ce_baseline > 0
target_logit_std > 0 after epoch 1
val_loss < uniform_ce_baseline
top1 meaningfully above 1 / avg_candidate_count
```

Watch for:

- `avg_cand = 1.0`: target mask collapse
- `loss = 0.0, top1 = 1.0`: degenerate one-class target task
- train loss falling while val loss rises: use best checkpoint, not last
- `target_logit_std = 0`: logits are uniform or the scorer is not learning

## Colab / GCS Workflow

Local pack/upload:

```bash
BUCKET=gs://orbit-wars-shipping INCLUDE_PAIR_SCORE_ASSETS=0 UPLOAD=1 ./scripts/pack_for_gpu.sh
gsutil ls -lh gs://orbit-wars-shipping/code.tgz gs://orbit-wars-shipping/data.tgz gs://orbit-wars-shipping/weights.tgz
```

If only code changes, it is enough to rebuild/upload `code.tgz`; the current
`data.tgz` can train correctly because the loader handles missing mask
sidecars.

Colab output upload path:

```text
gs://orbit-wars-shipping/runs/<run_name>/
```

For the current target-rank run:

```text
gs://orbit-wars-shipping/runs/Ebi_20260514-144051/
```

## Inference + Visualization Tooling

A small inference pipeline now wraps the trained ranker for offline
replay analysis and the dashboard UI.

### `agents/transformer_v2/inference/target_ranker_scorer.py`

- `load_target_ranker_stack(ckpt)` reconstructs the full stack
  (4 encoders + cross + ranker) from a saved `target_rank_best.pt` and
  returns `(stack, config)`. Config has `d_model / d_rank / n_heads /
  max_planets / n_history / player` so downstream code doesn't have to
  guess shapes.
- `score_replay(steps, ckpt, slot)` walks a replay's `env.steps`
  rolling the `n_history` window, calls the stack per turn, and emits
  per-planet `{logit, prob, target_valid}`. Drives the dashboard view
  via the server endpoint below.
- Per-turn label tensors (`ships_arriving_within_10`,
  `n_*_within_R_norm`, `nearest_enemy_dist_norm`) are stubbed at
  zero at inference — `featurize_observation` doesn't compute the
  cross-entity labels live. The encoders still see the right input
  shape; only `target_scalars` loses two channels' real signal.
  Follow-up to plug those in via a runtime cross-entity feature
  extractor would tighten inference-time accuracy.

### `app/server.py` — streaming endpoint

`GET /api/target_scores/stream?run_id=<>&slot=<>&mode=play` emits
NDJSON:

```
{"type": "init",     "n_total": 213, "config": {…}, "num_players": 2}
{"type": "progress", "current": 1,   "total": 213}
… one progress event per turn …
{"type": "done",     "steps": [{turn, planets, edges}, …], "n_steps": 213}
```

The synchronous `/api/target_scores` is still available for callers
that don't need progress. The stack is lazy-loaded once and cached.

The build script (`scripts/build_encoder_dataset.py`) and the trainer
both treat `_masks/*.npz` as required for the action dataset; missing
sidecars trigger a rebuild on next default-mode invocation.

### `app/target_view.html` — side-by-side replay

Standalone canvas-based renderer. Drawn next to the kaggle
`env.render()` iframe in the dashboard's Play tab. Per turn:

- **Heatmap rings** on each real planet, color and size = the
  ranker's per-target prob (saturated at the turn's argmax).
- **Owner core** colored with the Wong palette
  (`#0072B2 / #E69F00 / #009E73 / #F0E442`) matching the kaggle
  renderer; learner halo (white ring) marks the seat being scored.
- **Source→target attention edges** (Stage A): for each of the top-5
  targets by prob, the top-3 attended sources are drawn as arrowed
  lines with width/alpha scaled by attention weight. Edges below 0.05
  weight are dropped to avoid clutter.
- Reads the streaming endpoint with `fetch + ReadableStream` and
  drives a determinate `<progress>` bar in the header.

### Stage A attention extraction

`TargetRanker.forward(..., return_attn=True)` returns
`(target_logits, t2s_attn)` with `t2s_attn` shape `(B, P_target,
P_source)` (head-averaged). `TargetRankerStack.forward(...,
return_attn=True)` returns `(target_logits, tgt_valid, src_valid,
t2s_attn)`. Used by the scorer to emit per-turn edges; can also be
used for debugging the model's source-attention behavior offline.

### Dashboard scrubbing

A single timeline slider in the dashboard drives **both** replays:

- `input` event → `postMessage({type: 'setStep'})` to the target-view
  iframe (instant).
- `change` event → RAF-chunked `ArrowRight` / `ArrowLeft` keydown
  bursts on the kaggle iframe (40 keys per RAF frame; ~500 frames
  takes 1–2 s).
- A 4 Hz poll reads `contentWindow.kaggle.step` to keep the slider
  in sync during normal playback; suppressed while the user holds
  the thumb.
- Speed buttons and a unified Play/Pause control mirror their state
  to both viewers.

Useful as a debugging aid: park the slider on any turn, see what the
ranker thinks and which sources it attended to.

## Near-Term Recommendation

The best next implementation target is:

```text
TargetRanker Stage A + pairwise source-target bias/features
```

This directly addresses the remaining action-ranking problem: target choice
depends on which owned planets can act on the candidate target, how far away
they are, and whether their ships can change the outcome. That information
should be explicit in Stage A instead of being inferred indirectly from
independent source and target embeddings.

In parallel, add L2 auxiliary tactical losses so `CrossEntityAttention`
learns to preserve garrison, inbound pressure, and local support rather than
compressing them away.

## Open Items / Not Yet Done

- **Runtime agent wiring.** The TargetRanker is trained and visualized
  but no `Agent` subclass actually uses it for live play yet — the
  existing `TransformerAgent` still loads the old `pair_score` head
  via mtime-newest pick (which now points at the target_only_v2 ckpt,
  not the new ranker). Need a `target_rank_agent` that loads
  `target_rank_best.pt`, picks the argmax `(source, target)` from
  Stage A + ranker output, and combines with `physical_v4`'s
  ship-sizing heuristic until a frac head is reintroduced.
- **Cross-entity runtime labels.** Inference scorer stubs
  `ships_arriving_within_10` etc. at zero. Plug a runtime cross-entity
  feature computer in so the ranker sees the same `target_scalars`
  channels it trained on.
- **Per-head attention visualization.** Currently the dashboard
  averages heads via `average_attn_weights=True`. Exposing per-head
  attention would let us inspect whether different heads learned
  different things (e.g. distance-based vs. ownership-based source
  selection).
- **Stage B (target self-attention) edges.** Same extraction pattern
  but for the "is this target better than those?" comparison. Smaller
  payload (target↔target square matrix) and complementary view.
- **`max_fleets` regenerate.** The existing `data.tgz` on GCS was
  built at `max_fleets=256`; the dataset loader still works (it pads
  inside the snapshot) but truncated rows lose accurate inbound
  labels. A full rebuild + re-pack would close the loop on the
  truncation bug.
