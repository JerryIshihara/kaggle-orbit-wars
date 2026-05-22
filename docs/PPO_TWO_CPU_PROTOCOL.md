# Transformer v2 PPO protocol — two CPU machines

This is the recommended bring-up protocol for PPO after the supervised
`transformer_v2` PairHead has a usable runtime policy. It assumes **two CPU
machines** can run in parallel and no dedicated GPU is available.

Current repo status: `agents/transformer_v2/ppo.py` is intentionally still a
stub. This document defines the protocol to implement next; it is not a claim
that the PPO CLI already exists.

## Goal

Use PPO to fine-tune the learned source→target policy against the actual game
objective while preserving the useful supervised perception stack:

```text
L0 specialists                    frozen
L1-L2 world perception            frozen at first, touched only with evidence
PlayerContext/Strategy learners   planned post-L2 trainable modules
ActionLearner (current L3/L4)      trainable after PPO plumbing
PairHead output layers            trainable
noop/value PPO heads              trainable
```

The first PPO version should be conservative: keep the environment-facing
launches safe through `physics_utils.plan_launch`, keep rollouts strictly
on-policy, and measure policy quality with deterministic A/B evaluation after
every PPO iteration.

## Design critique / corrections

The earlier version of this plan was directionally useful but too optimistic
in several places. These are load-bearing corrections for the implementation:

- **The active model is not yet an actor-critic.** `EntityPretrainModel`
  currently returns only `pair_logits` and raw `pair_frac`. PPO must wrap it in
  a `PPOActorCritic` module that adds player/strategy context when available,
  a scalar `noop_logit`, a value head, and a stochastic fraction distribution.
  Do not pretend the existing PairHead already exposes those tensors.
- **Phase 0 uses a single-launch semantic action only as plumbing.** The
  runtime `TransformerAgent` can emit multiple thresholded launches per turn
  with per-source budgeting. A single categorical launch is easier to make
  mathematically correct for PPO, but it is a behavior mismatch. Treat it as
  bring-up; before submission either prove the single-launch policy beats the
  runtime rule or implement an autoregressive/per-source multi-launch sampler.
- **The learner must never be its own opponent in rollout.** Opponents come
  only from frozen *promoted* checkpoints plus the supervised baseline. The
  current `policy_vK` being updated is on-policy for learner actions, but it is
  not a valid self-play pool member.
- **Promotion cannot be gated by self-play winrate alone.** Self-play is
  non-transitive and can hide regressions. Gate against the previous promoted
  checkpoint, the supervised baseline, and at least one external heuristic.
- **Start with debuggable shards.** Featurized shards are larger, but they make
  logprob/value/GAE bugs visible. Switch to raw-observation shards only after a
  parity test proves re-featurization exactly matches stored rollout features
  and action masks.

## Machine roles

Use a synchronous coordinator setup where **both machines do rollouts** and,
from Phase 1 onward, **both contribute to the PPO update**. A is the
coordinator (owns the run directory, makes the optimizer step) and B is a
peer (computes gradients on its half of the data and ships them to A). This is
data-parallel training with one-way file-mediated grad sync, not classic NCCL
DDP — the one-way SSH constraint (see "Connectivity" below) rules out
symmetric all-reduce.

| Machine | Rollout | Training | Eval | Coordination |
|---|---|---|---|---|
| Machine A | ~50% of episodes | Phase 0: all training. Phase 1+: half of each paired minibatch, applies optimizer | quick-eval half + full-eval | owns run dir, advantages, optim state, promotion gate, archive |
| Machine B | ~50% of episodes | Phase 0: none. Phase 1+: half of each paired minibatch, writes grad to file | quick-eval half | passive — runs the worker process A drives via rsync |

Why split this way:

- **Rollout is the dominant CPU cost** (500 steps × N episodes × forward
  pass on a 256-dim transformer). Splitting 50/50 doubles rollout
  throughput, which is the biggest win.
- **The PPO update is much cheaper** but worth distributing once the
  trunk is unfrozen (Phase 1+). Cost of distribution = grad/weight file
  rsync per minibatch (see "CPU performance" and "Distributed training"
  below).
- **Eval also splits.** Default quick-eval is 32 seeds × 2 seats × 3
  mandatory opponents ≈ 192 episodes; halving it between A and B saves
  real wall-clock.

### Concrete machines (this project)

| Role | Host | User | Workdir |
|---|---|---|---|
| Machine A (learner) | the dev box this repo lives on | `agent` | `/Users/agent/dev/kaggle-orbit-wars` |
| Machine B (rollout worker) | `100.109.180.124` (Tailscale) — MacBook Air, Apple Silicon, Darwin 24.1.0 | `hirotakaishihara` | `/Users/agent/dev/kaggle-orbit-wars` |

**Workdir invariant:** the repo lives at the **same absolute path** on both
machines (`/Users/agent/dev/kaggle-orbit-wars`). All rsync source and
destination paths are therefore identical strings — no path translation needed
in any sync command.

**Connectivity (one-way):**

- A → B: SSH works (`ssh hirotakaishihara@100.109.180.124`).
- B → A: **not reachable.** Machine B cannot initiate a connection back to A.
- Therefore **every** rsync invocation (in either direction) must be started
  *from Machine A* using `rsync ... B:` or `rsync ... B:... .`. Machine B
  is strictly a passive endpoint of any transfer.

This rules out a worker-push model where B uploads its own shards. The
learner on A must pull them.

Recommended run directory:

```text
data/runs/ppo/<run_id>/
  config.json
  checkpoints/
    policy_v0.pt
    policy_v1.pt
    ...
  rollouts/
    v0/
      machine_a/
        shard_000001.pt
      machine_b/
        shard_000001.pt
  eval/
    v0.json
  pool/
    active_checkpoints.txt   # checkpoint basenames, e.g. policy_v17.pt
  train_log.jsonl
```

The two machines synchronize this directory with `rsync` over SSH, always
initiated from Machine A (see "Connectivity" above). A shared filesystem or
GCS bucket would also work but is unnecessary given the same-path invariant.

The important rule is atomic rollout writes:

```text
write shard_000123.pt.tmp
fsync/close
rename shard_000123.pt.tmp -> shard_000123.pt
```

The learner must never read `*.tmp`, and rsync must never transfer them
either — see the `--exclude '*.tmp'` flag below.

### rsync commands

Both directions are run on Machine A. The paths on B match A exactly because
of the workdir invariant.

```bash
# A -> B : publish the new policy checkpoint after a PPO update
rsync -av --partial --inplace \
  data/runs/ppo/<run_id>/checkpoints/policy_v${K}.pt \
  hirotakaishihara@100.109.180.124:/Users/agent/dev/kaggle-orbit-wars/data/runs/ppo/<run_id>/checkpoints/

# B -> A : pull B's rollout shards for iteration K
rsync -av --partial --inplace --exclude '*.tmp' \
  hirotakaishihara@100.109.180.124:/Users/agent/dev/kaggle-orbit-wars/data/runs/ppo/<run_id>/rollouts/v${K}/machine_b/ \
  data/runs/ppo/<run_id>/rollouts/v${K}/machine_b/

# A -> B : refresh the promoted-opponent pool manifest
rsync -av --delete \
  data/runs/ppo/<run_id>/pool/ \
  hirotakaishihara@100.109.180.124:/Users/agent/dev/kaggle-orbit-wars/data/runs/ppo/<run_id>/pool/
```

Notes:

- `--partial --inplace` lets large shard transfers resume after a transient
  Tailscale relay hiccup without restarting from zero.
- `--exclude '*.tmp'` is **mandatory** for any rsync that reads from a
  rollout directory — otherwise the learner could load a half-written
  shard whose `.tmp → final` rename hadn't happened on B yet.
- `--delete` is only safe on the pool-manifest direction (A is the source of
  truth for which promoted checkpoints are active). Checkpoint files named in
  `pool/active_checkpoints.txt` must also exist under `checkpoints/` on B
  before rollout starts. **Never** use
  `--delete` when pulling rollouts from B; B may still be writing more
  shards for the same iteration.

## CPU performance — where the time actually goes

This is a CPU-only setup. Two CPUs, no GPU. The design has to assume
**model forward passes dominate wall-clock**, not data movement or env
step. The recommendations below are ordered by expected impact.

### Rollout (dominant cost)

A single rollout episode of 500 steps requires ~500 learner forwards +
~500 opponent forwards (alternating seats in 2P). With 128 episodes per
iteration, that's ~128k model forwards per iter. Optimizations:

1. **Batched envs per worker process.** The model `forward(...)` already
   takes a leading batch dim `B` (verified in
   `agents/transformer_v2/encoder/entity_encoder.py:209`). Run `N_env`
   envs per worker process and call the policy once with a `[N_env, P, d]`
   tensor instead of `N_env` separate `[1, P, d]` calls. This is the
   single biggest CPU win for rollouts. Start with `N_env = 4–8` per
   worker, tune by `episodes/sec`.

2. **Worker process count = physical cores − 1.** Apple Silicon B has
   4 performance + 4 efficiency cores. Run ~7 worker processes (one
   reserved for the rsync/coordinator daemon). Oversubscription past
   physical core count slows total throughput because the env step is
   pure Python and contends for the GIL.

3. **`torch.inference_mode()` + `eval()`** during rollout. Skips autograd
   graph allocation. Mandatory, not optional.

4. **`torch.compile(policy, mode="reduce-overhead")`** on the rollout
   forward. First call costs ~30–60s of compile; amortized over 128k
   forwards per iter the payoff is large. Verify the compiled graph
   handles the legality mask correctly (boolean indexing is the usual
   compile gotcha).

5. **Share opponent model instances across workers.** A pool checkpoint
   loaded once per worker process is fine; loading it per episode is
   wasted I/O. The opponent model is also a `torch.compile` candidate
   since it's frozen.

6. **Keep T=1.** Storing T=6 fleet history per step would 6× the
   featurization cost and the shard size. T=1 was already the v1
   choice; do not revisit until eval shows tactical regression.

7. **Featurization caching is tempting but risky.** The
   `agents/transformer_v2/featurizer/` modules are ~3000 lines of pure
   Python and run every step. Incremental featurization (cache prev
   step, patch the diffs) could halve rollout time, but the bugs hide
   for many iterations because PPO still trains, just on slightly wrong
   features. Defer to a Phase 1+ effort after the loop is stable.

### Training (secondary cost)

Per-iteration PPO update on heads-only is ~1–2M trainable parameters →
forward+backward at batch 2048 ≈ 0.5–1s on CPU. Phase 1 unfreezes
PairHead + L4 → ~5–10M params → ~2–4s per minibatch. Either way the
update is small relative to rollouts, but distributing it still helps
once the trunk is unfrozen.

1. **Heads-only update is cheap enough to keep on A only in Phase 0.**
   Distribution overhead (grad rsync, weight rsync, see
   "Distributed training") would exceed the compute saved. Move to
   distributed only when Phase 1 is active.

2. **`torch.compile(policy)`** for the train forward too. The forward
   compiled for rollouts may be the same graph; PyTorch will reuse.

3. **Minibatch size 1024–2048 on CPU, not 4096.** CPU L2/L3 cache misses
   dominate past 2048-step minibatches; smaller minibatches with more
   gradient steps are usually faster on CPU than fewer larger ones.

4. **BC anchor minibatch sampling stays on A.** `_pair_cache` is ~47GB
   and lives on A (see "Archive policy"). Sampling the BC minibatch
   once per PPO step and including only its loss on A's gradient is
   simpler than replicating the cache to B. Trade-off: BC anchor only
   appears in A's gradient, so its effective coefficient is halved
   relative to the policy/value loss in the averaged gradient — bump
   `bc_coef` by 2× to compensate.

   **Why exactly 2×, not a heuristic:**

   ```text
   grad_A     = ∇L_PPO_A + bc_coef_A · ∇L_BC
   grad_B     = ∇L_PPO_B                          (no BC term — no cache)
   grad_avg   = 0.5 · (grad_A + grad_B)
              = 0.5 · (∇L_PPO_A + ∇L_PPO_B)  +  0.5 · bc_coef_A · ∇L_BC
   ```

   The effective BC coefficient applied at the optimizer step is
   `0.5 · bc_coef_A`. To recover the non-distributed setting where
   `bc_coef = 0.05` is the coefficient seen by the optimizer, set
   `bc_coef_A = 0.10`. The PPO/value/entropy losses are unaffected by
   the averaging because they're evaluated on disjoint mb_A/mb_B halves
   — their average is the unbiased estimator over the global minibatch.

### Eval

Quick-eval default = 32 seeds × 2 seats × 3 mandatory opponents
(`previous_promoted`, `transformer_v2_baseline`, `physical_v4`) =
192 episodes per iteration. That's 1.5× a 128-episode rollout budget —
eval can easily dominate if run only on A. `sniper_v2` is still important,
but run it on the full-eval cadence or when the quick gate is marginal.

1. **Split quick-eval 50/50 between A and B.** A runs seeds 0..15, B
   runs seeds 16..31 (both seats, same opponent set). A pulls B's eval
   JSON via rsync, merges, applies promotion gate.

2. **Full-eval (128 seeds) every 5–10 iterations**, not every
   iteration. Quick-eval is sufficient for the per-iter promotion gate.

3. **Skip eval only when iteration KL was negligible and do not promote.**
   If approx_KL < 0.2 × target_KL, the policy barely changed; the eval result
   will be nearly identical to the prior promoted checkpoint. Save the CPU,
   log `promotion_skipped_low_kl`, and leave the pool unchanged.

### Shard format (storage / rsync bandwidth)

Two options. Use Option A for smoke/bring-up, then switch to Option B only
after replay-parity tests pass.

| Option | Shard contents | Pros | Cons |
|---|---|---|---|
| **A. Featurized** | `planet_features`, `comet_features`, `fleet_features`, routing tensors, action mask, action, logprob, value, reward, done | Training is fast — just forward through the stored tensors | Large (~MB per episode); slow rsync from B; 2–4× the disk |
| **B. Raw obs + policy outputs** | raw env obs dict, action mask, action_id, frac, logprob, value, reward, done | ~10× smaller shards, fast rsync | Training has to re-run featurization, costing ~3× CPU per epoch (×N PPO epochs) |

For CPU + Tailscale, **Option B is recommended after the loop is verified**.
Storage and rsync win exceeds the re-featurization cost when PPO epochs ≤ 3,
but it is the wrong first implementation because it hides featurizer/action-mask
drift.

## Distributed training — file-mediated gradient averaging

One-way SSH (`A → B`) rules out NCCL all-reduce. Both machines instead
sync via files: B writes grads, A pulls, averages, applies the step,
publishes new weights, B pulls. Per minibatch.

```text
sync/v<K>/
  epoch_<E>/
    mb_<M>_grad_B.pt.tmp  -> mb_<M>_grad_B.pt   # written by B, pulled by A
    mb_<M>_weights.pt.tmp -> mb_<M>_weights.pt  # written by A, pulled by B
```

One PPO minibatch step:

```text
1. A: forward + backward on mb_A     →  grad_A           (parallel with B)
1. B: forward + backward on mb_B     →  grad_B; write to sync/.../grad_B.pt.tmp
2. B: rename grad_B.pt.tmp -> grad_B.pt          (atomic)
3. A: rsync pull grad_B.pt
4. A: grad = (grad_A + grad_B) / 2
5. A: optimizer.step(); zero_grad()
6. A: write weights.pt.tmp -> weights.pt         (atomic, on A's local fs)
7. A: rsync push weights.pt to B
8. B: load new weights from disk
9. both: proceed to minibatch M+1
```

### Sync overhead estimate

| Phase | Trainable params | Grad/weight size | Sync per minibatch | × 192 mb/iter |
|---|---:|---:|---:|---:|
| 0 (heads only) | ~1–2 M | ~4–8 MB | ~60 ms | ~12 s/iter |
| 1 (+ PairHead, L4) | ~5–10 M | ~20–40 MB | ~150 ms | ~30 s/iter |
| 2 (+ L3) | ~15–25 M | ~60–100 MB | ~400 ms | ~80 s/iter |

For Phase 0, this is ~10% of total iter wall-clock — not worth
distributing. Keep A-only training in Phase 0. **Switch to
distributed training starting Phase 1.**

### Less-frequent sync (local SGD) for Phase 2

If Phase 2's 80 s/iter sync overhead is intolerable, use *local SGD*:
each machine takes K=4 optimizer steps locally, then averages weights
(not grads) every 4 steps. Cuts sync count by 4× at the cost of mild
divergence between A's and B's model copies. Apply only after Phase 1
shows the distributed update converges.

### Sync atomicity

All grad and weight files use the same `.tmp → rename` rule as rollout
shards. Reader (A pulling grad, B pulling weights) **must** exclude
`*.tmp` from rsync. Loading a half-written tensor file silently produces
garbage updates without crashing.

## Archive policy

Disk on A (the dev box) is the working set: source, current iter,
recent checkpoints. **Old artifacts move to B's
`/Users/agent/dev/kaggle-orbit-wars/archive/`** dir. B has spare disk
(rollout-only workload, no training datasets stored long-term) and the
same workdir path so rsync targets are trivial.

### What stays hot on A

| Path | Why |
|---|---|
| `agents/`, `app/`, `scripts/`, `tests/` etc. | working tree |
| `data/runs/ppo/<run_id>/pool/active_checkpoints.txt` | source of truth for active promoted opponent checkpoint basenames |
| `data/runs/ppo/<run_id>/checkpoints/<basenames listed in pool/active_checkpoints.txt>` | active self-play pool |
| `data/runs/ppo/<run_id>/checkpoints/policy_v[K-2..K+1].pt` | recent trained ckpts for debug, including just-trained ckpt |
| `data/runs/ppo/<run_id>/checkpoints/policy_v0.pt`, `transformer_v2_baseline.pt` | the floor and the init — referenced forever |
| `data/runs/ppo/<run_id>/rollouts/v[K-1..K]/` | current iter + the one needed for any K-1 debug |
| `data/runs/ppo/<run_id>/eval/v[K-10..K].json` | recent eval history for plots |
| `data/runs/ppo/<run_id>/train_log.jsonl` | tiny, append-only |
| `data/runs/ppo/<run_id>/sync/v<K>/` | current iter's grad/weight files (deleted at end of iter) |
| `data/datasets/_pair_cache/` (47 GB) | BC anchor source — read every PPO step on A |
| `data/datasets/_bc_cache/` (10 GB) | secondary BC cache — read at iter boundaries |

### What moves to B's `archive/`

| Path on A | Path on B | When |
|---|---|---|
| `data/runs/ppo/<run_id>/checkpoints/policy_v*.pt` not in pool manifest, not v0, not recent debug window | `archive/runs/ppo/<run_id>/checkpoints/` | after each iteration |
| `data/runs/ppo/<run_id>/rollouts/v[0..K-2]/` | `archive/runs/ppo/<run_id>/rollouts/` | after iter K's update completes |
| `data/runs/ppo/<run_id>/eval/v[0..K-11].json` | `archive/runs/ppo/<run_id>/eval/` | rolling |
| `data/runs/ppo/<run_id>/replays/` (promoted only) | `archive/runs/ppo/<run_id>/replays/` | after each promotion |
| `data/datasets/entity/`, `planet/`, `fleet/`, `cross_entity/`, `comet_only_*/` (~15 GB pretraining datasets) | `archive/datasets/` | one-time — these produced the supervised checkpoints, no longer read |

The `data/datasets/_pair_cache/` (47 GB) and `_bc_cache/` (10 GB) stay
on A because they are read every PPO step for the BC anchor.

### Archive script

Run from A after each iteration's promotion gate:

```bash
# scripts/ppo_archive.sh <run_id> <K>
# K = rollout policy version that just finished training/eval; policy_v(K+1)
# is the just-trained checkpoint.
run_id="$1"; K="$2"
next=$((K + 1))
recent_floor=$((K - 2))
roll_floor=$((K - 2))
eval_floor=$((K - 11))
pool_manifest="data/runs/ppo/$run_id/pool/active_checkpoints.txt"

# 1. ship cold checkpoints to B.
# Do not rely on version arithmetic for the pool: promotion is sparse, so an
# old policy_v17 may still be active when the current iter is v80.
for path in data/runs/ppo/"$run_id"/checkpoints/policy_v*.pt; do
  [ -f "$path" ] || continue
  base="$(basename "$path")"
  v="${base#policy_v}"; v="${v%.pt}"
  [ "$v" = "0" ] && continue          # keep v0 hot (the init)
  [ "$v" -ge "$recent_floor" ] && continue
  [ "$v" -le "$next" ] || continue
  if [ -f "$pool_manifest" ] && grep -qx "$base" "$pool_manifest"; then
    continue                          # active promoted opponent
  fi
  rsync -av --remove-source-files \
    "$path" \
    "hirotakaishihara@100.109.180.124:/Users/agent/dev/kaggle-orbit-wars/archive/runs/ppo/$run_id/checkpoints/"
done

# 2. ship cold rollouts
for v in $(seq 0 "$roll_floor"); do
  [ -d "data/runs/ppo/$run_id/rollouts/v$v" ] || continue
  rsync -av --remove-source-files \
    "data/runs/ppo/$run_id/rollouts/v$v/" \
    "hirotakaishihara@100.109.180.124:/Users/agent/dev/kaggle-orbit-wars/archive/runs/ppo/$run_id/rollouts/v$v/"
  rmdir "data/runs/ppo/$run_id/rollouts/v$v" 2>/dev/null
done

# 3. ship cold eval JSONs
find "data/runs/ppo/$run_id/eval/" -maxdepth 1 -name 'v*.json' | \
  awk -F'v|.json' -v floor="$eval_floor" '$2+0 < floor' | \
  xargs -I{} rsync -av --remove-source-files {} \
    "hirotakaishihara@100.109.180.124:/Users/agent/dev/kaggle-orbit-wars/archive/runs/ppo/$run_id/eval/"
```

`--remove-source-files` deletes the local file *only after* a successful
transfer to B. Combined with the atomic write rule, this is safe even
if rsync is interrupted mid-batch.

**Never** archive a checkpoint that is in the current self-play pool —
B's rollout workers load pool checkpoints from the *hot* path. The pool
manifest, not version arithmetic, enforces this. Promotion is sparse, so
"older than K-N" is not equivalent to "not in the active pool".

## Policy/action contract

### Actor-critic wrapper

The active `EntityPretrainModel` is still a supervised PairHead model. Its
forward returns:

```text
pair_logits  (B, P, P)
pair_frac    (B, P, P) raw logit; runtime uses sigmoid(pair_frac)
```

PPO should add a wrapper instead of rewriting world perception. The wrapper can
start thin and later grow explicit post-L2 context modules:

```text
PPOActorCritic(EntityPretrainModel):
  pair_logits  = base_out["pair_logits"]
  frac_loc     = base_out["pair_frac"]        # logit-normal mean
  learner_ctx  = PlayerContextLearner(ctx_now, glob)       # planned
  strategy_ctx = StrategyLearner(glob, learner_ctx, players) # planned
  noop_logit   = Linear([glob, learner_ctx, strategy_ctx] -> 1)
  value        = Linear([glob, learner_ctx, strategy_ctx] -> 1)
  frac_log_std = learned scalar, clamped to log([0.15, 0.80])
```

Use the L2 global token returned by `CrossEntityAttention` for `noop_logit` and
`value` if the implementation exposes it cleanly. If not, mean-pool the current
valid `ctx_now` tokens as a first pass and leave a TODO to switch to the real
global token. Keep this wrapper checkpoint-compatible with the supervised
PairHead ckpt: missing PPO-only keys initialize from scratch.

Architectural boundary for PPO:

```text
L0-L2 = world perception; freeze by default.
PlayerContextLearner + StrategyLearner + ActionLearner = policy adaptation.
```

This gives PPO room to learn "who am I, what strategy should I use, what move
implements it?" without corrupting the already-useful board perception.

**Naming note — `ActionLearner` is conceptual.** There is no `ActionLearner`
class today. The name refers to the existing post-L2 action-selection stack
that PPO updates:

```text
ActionLearner ≡ aggregator/dual_role_attention.py   (L3 — DualRoleAttention)
              + aggregator/joint_role_attention.py  (L4 — JointRoleAttention)
              + aggregator/pair_head.py trunk + FiLM (PairHead body, not output layers)
```

`PairHead`'s output layers (`pair_logits` head, `pair_frac` head) are listed
separately because Phase 0 trains those output layers but freezes the trunk.

### Phase 0 semantic action

Phase 0 samples exactly one semantic decision per learner turn:

```text
action 0                = NOOP
action 1 + source*P+tgt = launch(source, target)
```

This single-launch distribution is intentionally a bring-up simplification.
It does **not** reproduce the production runner's thresholded multi-launch
behavior. Track that gap explicitly with emitted-launch rate, per-source
multi-target opportunities, and eval versus the frozen `transformer_v2_baseline`.

Use a cheap legality mask before sampling:

- source exists
- source is owned by the learner
- source has surplus above `min_launch`
- target exists
- source != target

Do **not** mask by expensive full physics for every pair in v1. That makes CPU
rollout too slow and can also hide useful negative signal.

### Fraction distribution

Pin the fraction distribution now; do not leave it as an implementation choice.
Reuse the existing `pair_frac` raw logit as the mean of a logit-normal:

```text
z            ~ Normal(frac_loc[source, target], exp(frac_log_std))
frac_sample  = clamp(sigmoid(z), 1e-4, 1 - 1e-4)    # numerical clamp only
logp_frac    = Normal.log_prob(logit(frac_sample))
               - log(frac_sample) - log(1 - frac_sample)
```

**Store `frac_sample` (the `[1e-4, 1-1e-4]`-clamped value) in the shard.**
PPO recomputes `logp_frac` at update time from the stored `frac_sample`, so
the two must agree exactly — that means the value stored must be the same
value the log-prob was computed on.

The launch-side clamp is applied **later, only at env projection**, not at
sampling time:

```text
ship_fraction_for_launch = clamp(frac_sample, 0.02, 1.00)
ships                    = round(ship_fraction_for_launch * source.ships)
```

This separation is load-bearing: if the launch clamp were applied at
sampling time and the clamped value stored, every step where
`sigmoid(z) < 0.02` would silently desync stored logprob from the
sampled action and break the PPO ratio. Add a rollout-replay test that
recomputes `logp_frac` from `frac_sample` and asserts equality to
`logprob_frac` in the shard.

For `NOOP`, store `frac_sample = 0` and `logp_frac = 0`; the action logprob is
only the categorical NOOP logprob.

### Environment projection

After sampling a launch, convert it to an env action through
`physics_utils.plan_launch`.

If `plan_launch.ok == True`:

```text
emit [source_planet_id, launch.angle, ships]
```

If `plan_launch.ok == False`:

```text
emit NOOP
record invalid_launch = 1
record invalid_reason = launch.reason
apply a small invalid-action penalty
```

Do **not** resample until valid. Resampling changes the executed action without
matching the stored PPO log-probability and makes the update mathematically
wrong. The sampled invalid action should stay in the rollout with its original
logprob, receive a penalty, and teach the policy away from that region.

### Phase 1+ multi-launch upgrade

Before using PPO-trained checkpoints for submission, decide whether the
single-launch simplification is acceptable. If it leaves obvious value on the
table, upgrade the sampler to one of these contracts:

| Contract | Logprob shape | Notes |
|---|---|---|
| Per-source categorical | one NOOP/target decision per launchable source | closest to current row-wise budgeting; can emit several sources per turn |
| Autoregressive top-K | sequentially sample pair/STOP with source budgets updated after each pick | closest to threshold runner; most code |
| Bernoulli per legal pair | independent fire/no-fire per cell | simple but high-variance and needs strict cap/budget rules |

Keep the chosen semantic action in the rollout exactly as sampled. Projection
through `plan_launch` may drop invalid launches, but it must not alter the
stored semantic action used by PPO.

### Initial T setting

Although the supervised PairHead was trained with `T=6`, the runtime runner
already supports `T=1`. For CPU PPO, start with:

```text
T = 1
```

Reason: storing or recomputing full T=6 fleet tensors for every PPO step is
expensive on CPU, especially at `max_fleets=1024`. Once the PPO loop is stable,
reintroduce T=6 only if evaluation shows a clear tactical regression from
single-frame input.

## Rollout shard format

Each shard should contain complete episodes or episode fragments that never
cross episode boundaries. Complete episodes are simpler and preferred.

Minimum fields:

```python
{
    "policy_version": int,
    "machine_id": str,
    "created_at": str,
    "git_sha": str,
    "policy_action_contract": "single_launch_v1",
    "episodes": [
        {
            "seed": int,
            "seat": int,
            "agent_slots": list[str],
            "opponent_ckpt": str,
            "winner": int,
            "reward_final": float,
            "steps": {
                # model inputs at action time
                "planet_features": Tensor[n, P, planet_dim],
                "comet_features":  Tensor[n, P, comet_dim],
                "fleet_features":  Tensor[n, F, fleet_dim],
                "planet_mask":     Tensor[n, P],
                "fleet_mask":      Tensor[n, F],
                "routing":         dict[str, Tensor],

                # PPO data
                "action_mask":      Tensor[n, 1 + P*P], # exact mask used for sampling
                "action_id":       Tensor[n],      # 0 = noop, >0 = pair
                "frac_sample":     Tensor[n],
                "logprob":         Tensor[n],      # logp_action + logp_frac
                "logprob_action":  Tensor[n],
                "logprob_frac":    Tensor[n],
                "value":           Tensor[n],
                "reward":          Tensor[n],
                "done":            Tensor[n],

                # diagnostics
                "invalid_launch":  Tensor[n],
                "invalid_reason":   list[str],
                "emitted_launch":   Tensor[n],      # bool
                "source_idx":       Tensor[n],
                "target_idx":       Tensor[n],
                "source_pid":       Tensor[n],
                "target_pid":       Tensor[n],
            },
        },
    ],
}
```

Use CPU tensors and save with `torch.save`. Prefer `float16` or compressed
`npz` only after the loop is correct; debuggability matters more at first.
`action_mask` is mandatory for featurized shards because the cheap legality
mask depends on source ownership, surplus, padding, and diagonal removal; do
not try to reconstruct it from normalized model features during the PPO update.

If rollout files become too large, the first optimization is to store raw obs
plus policy outputs and re-featurize on the learner. That path must have a
test that featurizes a sampled raw shard and matches the stored Phase-0
features/action masks before it replaces featurized shards.

## Startup calibration (once per run, before iter 0)

The promotion gate is written in terms of `previous_promoted`'s measured
metrics. On iter 0 there is no `previous_promoted`; the protocol uses
`transformer_v2_baseline` as the bootstrap stand-in (the pool starts as
`[baseline]`, all opponent slots fall back to baseline). For the gate's
relative thresholds ("≥ prev's − 2pp", "drops by ≤ 5pp") to be defined
on iter 1, we need actual measured metrics for the baseline first.

Run a one-shot calibration eval before the first PPO iteration:

```text
baseline_calibration = quick_eval(
    policy = transformer_v2_baseline,
    seeds  = SEEDS_QUICK,                              # split A/B as usual
    opponents = [transformer_v2_baseline,              # self-play floor
                 physical_v4]                          # external floor
)
```

Persist to `data/runs/ppo/<run_id>/eval/v0_baseline.json`. This produces:

```text
metrics_of(transformer_v2_baseline) = {
    winrate.previous_promoted = ~0.50,                 # baseline vs baseline
    winrate.baseline          = ~0.50,                 # same opponent
    winrate.physical_v4       = measured (~0.75 per prior panel),
    invalid_launch_rate       = measured,
    paired_score              = ~0,
    ...
}
```

These values fill `metrics_of(promoted)` on iter 1 so the promotion gate
has well-defined thresholds. After the first PPO promotion,
`metrics_of(promoted)` is replaced by the promoted ckpt's measured eval
and calibration is no longer consulted.

The calibration eval is also a sanity check: if baseline-vs-baseline
winrate is not ≈0.50 ± 0.05, the eval harness is broken (RNG seeding,
seat-symmetric scoring, etc.) — fix before iter 0.

## Iteration lifecycle

One synchronous PPO iteration. The "training" step has two variants:
**Phase 0 = A-only** (heads update too cheap to distribute);
**Phase 1+ = distributed** via file-mediated grad averaging.

```text
1.  A: publish checkpoints/policy_vK.pt and current promoted-opponent pool;
    rsync push both to B.
2.  Both: load exactly policy_vK.pt for learner actions and load only
    promoted frozen checkpoints for opponents.
3.  Both: collect rollout shards in parallel using frozen learner policy_vK.
4.  Both: write shards into rollouts/vK/<machine_id>/ atomically.
5.  A: background-rsync-pull B's shards as they appear.
6.  A: stop when rollout budget hit; ignore late shards.
7.  A: compute GAE on the merged rollout set (advantages, returns).
8a. Phase 0: A performs the PPO update locally, writes policy_vK+1.pt.
8b. Phase 1+: distributed update —
      i.   A builds paired minibatches: global minibatch = mb_A ∪ mb_B,
           local minibatch size = 1024-2048 per machine; rsync push mb_B
           sequence to B.
      ii.  for each (epoch E, minibatch M):
             A computes grad_A on mb_A[E][M]                (parallel)
             B computes grad_B on mb_B[E][M], writes file   (parallel)
             A rsync-pulls grad_B
             A averages grad = (grad_A + grad_B) / 2
             A optim step; writes new weights to file
             A rsync-pushes weights to B
             B loads new weights
      iii. final weights = policy_vK+1.pt on A; same weights now on B.
9.  Both: run quick-eval (32 seeds split 16/16 between A and B).
10. A: pull B's eval JSON, merge, apply promotion gate, append to train_log.jsonl.
11. A: run archive script (cold checkpoints, cold rollouts, old eval → B/archive/).
12. Repeat.
```

Strict on-policy rule for v1:

```text
train vK update only on rollouts generated by policy_vK
```

Do not train on stale shards from `vK-1`. With only two machines,
synchronous collection is simpler and reliable enough. The grad/weight
sync files in step 8b live in `sync/v<K>/` and are deleted at the end
of the iteration (they are not archived).

## Rollout budget

Start small enough to iterate quickly, then scale.

| Phase | Total rollout per iteration | Purpose |
|---|---:|---|
| Smoke | 8-16 episodes | verify tensors/logprobs/advantages/update |
| Bring-up | 64-128 episodes | tune reward/action penalties |
| Stable PPO | 256-512 episodes | real learning signal |

Split episodes roughly 50/50 between A and B. Both machines now
contribute equally to training (after Phase 0) so neither side has a
systematic obligation that warrants giving the other more rollout work:

```text
Machine A: 45-50% of rollout budget
Machine B: 50-55% of rollout budget
```

B gets a slight tilt because A also runs the GAE merge + optimizer
step + archive — those are CPU-light but wall-clock-serial.

Run one env worker process per physical core minus one (Apple Silicon
B: 4P + 4E → ~7 workers; A: depends). Inside each worker, batch
`N_env = 4–8` envs per forward pass — see "CPU performance →
Rollout" above. Avoid total worker count past physical cores; the env
step is pure Python and contends for the GIL.

## Opponent schedule — self-play only

Start with 2-player games. Add 4-player only after 2P improves.

The supervised baseline (`transformer_v2_baseline`, the frozen May-20
8-head ckpt) already beats `physical_v4` ~75% on the stratified 16-seed
panel and posts row R@1 = 0.42 / row R@5 = 0.74 on the held-out split.
Training against `physical_v4` / `sniper_v2` from here mostly burns
rollout budget on opponents that don't provide a useful gradient.

**Rollouts are 100% self-play-style training.** The learner sees frozen
snapshots of its own promoted lineage plus the supervised baseline. This keeps
the rollout signal near the current policy's level and avoids the classic PPO
trap of collapsing to a stable point against weak external opponents.

### Self-play pool

Maintain a pool of frozen **promoted** policy checkpoints. Each rollout
episode samples one opponent from the pool:

| Probability | Opponent |
|---:|---|
| 50% | **latest promoted PPO checkpoint** (`policy_v<promoted>.pt`; fallback to baseline until the first promotion) |
| 30% | **`transformer_v2_baseline`** (frozen May-20 supervised ckpt — the lower bound; the policy must keep beating it) |
| 20% | **uniformly sampled older promoted PPO checkpoint** (fallback to baseline when empty) |

Pool size `N_pool = 8` is a reasonable starting point. The older-uniform
slot exists to prevent cyclic policy chasing (NFSP / Fictitious Self-Play
intuition): if the learner forgets how to beat a strategy from 5 iters
ago, the pool resurfaces it.

The current policy `vK` never plays itself directly — always against
*promoted frozen* opponents. Do not add failed or merely trained checkpoints to
the rollout pool; that trains against regressions and makes the next gradient
less meaningful. Opponent moves are sampled but their logprobs aren't tracked.

### Seat assignment

Play both seats. For every sampled seed, assign learner to seat 0 or
seat 1 with equal probability. Evaluation must always run both seats.

### Eval vs rollout — keep external opponents for the gate

`physical_v4` and `sniper_v2` disappear from rollouts but remain in the
deterministic eval gate (see "Evaluation gate" below). They're cheap
external yardsticks: if self-play improves but `physical_v4` winrate
drops, the policy is overfitting to the pool — block promotion and
investigate.

## Reward design

Use terminal reward as the anchor:

```text
terminal win  = +1.0
terminal loss = -1.0
draw/tie      =  0.0
```

Add small dense shaping to reduce variance. Pure ship-count margin is
gameable because a policy can hoard instead of capturing, so include a small
territory term:

```text
potential_t =
    (my_total_ships + 5.0 * my_owned_planets)
    - max_enemy(my_total_ships + 5.0 * owned_planets)

dense_t = clip((potential_t - potential_t-1) / 200.0, -0.02, 0.02)
```

Recommended first reward:

```text
reward_t = dense_t
if done:
    reward_t += env_terminal_reward
reward_t -= 0.01 * invalid_launch
```

Keep shaping small. If dense reward dominates terminal win/loss, PPO will
learn to optimize the potential instead of winning. Log
`abs(sum_dense) / max(1e-6, abs(terminal_reward))` per episode; if its median
exceeds `0.25`, reduce the dense scale or disable shaping for the next run.

## PPO hyperparameters

Initial CPU-friendly defaults:

| Parameter | Value |
|---|---:|
| `gamma` | `0.995` |
| `gae_lambda` | `0.95` |
| PPO clip | `0.10` |
| target KL | `0.01` |
| update epochs | `2-3` |
| minibatch size | `1024-2048` learner steps (per machine in Phase 1+) |
| policy/head LR | `1e-4` |
| unfrozen trunk LR | `1e-5` |
| value coef | `0.5` |
| entropy coef | `0.01` initially, decay to `0.002` |
| max grad norm | `0.5` |
| invalid launch penalty | `0.01` |

If KL early-stops every iteration, lower LR or clip to `0.05`. If entropy
collapses before winrate improves, increase entropy coef or add a supervised
BC anchor.

## Freeze schedule

Do not unfreeze everything at once.

### Phase 0 — PPO plumbing

Train only:

```text
noop head
PairHead pair_logits output layer
PairHead pair_frac output layer
frac_log_std
value head
```

Freeze L0-L2 world perception and freeze the current compact ActionLearner
trunk (L3/L4 + PairHead trunk/FiLM). Goal: prove PPO logprobs, value learning,
GAE, and eval plumbing work without destroying the BC policy.

### Phase 1 — post-perception adaptation

Unfreeze:

```text
PlayerContextLearner (if implemented)
StrategyLearner (if implemented)
ActionLearner L3/L4
PairHead
value head
ship-fraction head
noop head
```

Keep L0-L2 frozen. Use low LR for the current action trunk (L3/L4 and PairHead
trunk/FiLM); keep the PPO-only heads at the higher head LR.

**Fallback when post-L2 stubs aren't implemented yet:** if
`PlayerContextLearner` and `StrategyLearner` are still identity / zero modules
(see "Actor-critic wrapper"), Phase 1 has no new parameters at those layers.
It then reduces to: unfreeze the existing `aggregator/dual_role_attention.py`
(L3) + `aggregator/joint_role_attention.py` (L4) + full `PairHead` (trunk +
FiLM + output layers). The conceptual L0-L2 / post-L2 boundary is preserved
for Phase 2 planning, but the actual trainable parameter set is the same as
the prior design's Phase 1.

### Phase 2 — perception adaptation, only if needed

If Phase 1 plateaus and diagnostics show world perception is stale or missing
facts, unfreeze:

```text
CrossEntityAttention (L2)
PlanetEntityEncoder (L1), only after L2 evidence
post-L2 learner stack
```

Keep L0 specialists frozen unless there is strong evidence that perception
itself is wrong. L0 was pretrained for physical/entity perception and should
not be rewritten by sparse PPO rewards early.

## BC anchor

Use a small supervised anchor during PPO to prevent policy collapse:

```text
loss = ppo_loss
     + value_coef * value_loss
     - entropy_coef * entropy
     + bc_coef * BCE(pair_logits, expert_pair_labels)
```

Start:

```text
bc_coef = 0.05
```

Decay toward `0.0` only after deterministic eval beats the BC-only policy.
Sample BC minibatches from the existing pair cache, not from rollout shards.

## Evaluation gate

After every PPO iteration, run deterministic eval:

```text
SEEDS_QUICK = 32 seeds
both seats
mandatory opponents = previous_promoted, transformer_v2_baseline, physical_v4
periodic/full opponents = sniper_v2, latest older pool members
```

Report:

- winrate by opponent
- winrate by seat
- winrate by archetype (`docs/EVAL_SEEDS.md`)
- average reward
- invalid launch rate
- emitted launch rate
- launch miss matrix from the dashboard analysis
- entropy
- approx KL
- value explained variance
- mean/median episode length

Promotion rule:

```text
Promote candidate `policy_v(K+1)` only if all are true on SEEDS_QUICK:
  1. winrate vs previous_promoted >= 50% and paired seed-seat score is non-negative
  2. winrate vs transformer_v2_baseline >= previous_promoted's baseline winrate - 2pp
  3. winrate vs physical_v4 does not drop by more than 5pp
  4. invalid launch rate <= 1.1x previous_promoted's rate
  5. launch miss matrix has no new high-severity category regression
```

Run the full 128-seed panel before considering a submission.

## Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| KL early-stops every iter | LR too high or rollout too small | lower LR, increase rollout episodes |
| entropy collapses | PPO overfits sparse reward | raise entropy coef, add/raise BC anchor |
| invalid launch rate rises | policy exploiting unsafe semantic actions | keep invalid penalty, add cheap physics features, do not resample |
| winrate up but miss matrix worse | unsafe projection hiding policy errors | block promotion; inspect category matrix |
| value loss explodes | rewards too sparse/noisy | normalize advantages, reduce dense scale, lower value LR |
| self-play winrate up but heuristic eval drops | policy overfit to current pool; cyclic chase | raise the older-uniform pool slot from 20% → 40%; lower the latest-promoted slot accordingly; check `transformer_v2_baseline` winrate is still near the floor |
| pool keeps cycling (winrate vs each pool member oscillates) | pool too small / latest-only sampling | grow `N_pool` from 8 → 16; lower the latest-promoted slot to ≤ 40% |
| rollout opponent is `policy_vK` | pool builder used the learner checkpoint instead of promoted ckpts | hard-fail rollout startup; `policy_vK` may be learner only, never opponent |
| PPO ratio NaNs or changes under replay | missing/stale `action_mask` or unstable fraction clamp | store exact action mask and clamp fraction before logit/logprob; add rollout replay test |
| CPU rollout too slow | one env per worker, no `torch.compile` | enable batched envs (`N_env=4-8`), `torch.compile(mode="reduce-overhead")`, share opponent model across workers (see "CPU performance → Rollout") |
| grad sync dominates iter wall-clock in Phase 1+ | per-minibatch full-grad rsync | switch to local SGD (avg every K=4 steps), or shrink minibatch count |
| disk on A fills up | rollouts/checkpoints accumulating | run `scripts/ppo_archive.sh` after each iteration's promotion gate; the script moves cold data to B's `archive/` |
| B's pool checkpoint sampled but file missing on B | new pool member not yet rsynced | A must push the new ckpt to B *before* publishing v(K+1), not after |
| `policy_vK+1` differs between A and B after distributed update | non-atomic weight write or B loaded stale weights | verify `.tmp → rename` for weight files; verify B's load step fences against partial reads |

## Minimal implementation checklist

1. Add `PPOActorCritic` around current `EntityPretrainModel`. Make sure
   the forward accepts batch dim > 1 and returns `pair_logits`, `frac_loc`,
   `noop_logit`, `value`, and `frac_log_std`.
2. Implement the single-launch categorical action contract plus logit-normal
   fraction sampling/logprob. Store the exact `action_mask`, semantic
   `action_id`, `frac_sample`, and logprob components in every shard.
3. Implement `ppo_rollout_worker.py`:
   - load checkpoint version + promoted opponent pool (loaded once per worker process)
   - hard-fail if the current learner checkpoint appears in the opponent pool
   - run assigned episodes with `torch.inference_mode()` + `torch.compile`
   - `N_env` envs per worker, batched forward
   - write rollout shards atomically (`.tmp → rename`)
4. Implement `ppo_learner.py`:
   - wait for rollout budget (with wall-clock timeout)
   - rsync-pull B's shards
   - **hard-fail on mismatch** of shard `policy_version` (must equal current
     `vK`), `policy_action_contract` (must equal the contract the learner is
     configured for), and presence/shape of `action_mask`. Never silently
     train on cross-version or cross-contract shards — log the offending
     shard path and abort the iteration.
   - compute GAE
   - **Phase 0**: local PPO update only
   - **Phase 1+**: distributed PPO update via grad/weight file sync
     (see "Distributed training")
   - publish next checkpoint; rsync push to B
5. Add `ppo_train_peer.py` for Machine B (Phase 1+):
   - load minibatch sequence A pushed
   - per minibatch: forward+backward, write `grad_B.pt` atomically, poll for new weights
6. Add deterministic eval script using `utils.eval_seeds`, split-aware
   (so A runs seeds 0..15 and B runs 16..31, both seats) and mandatory
   opponents `previous_promoted`, `transformer_v2_baseline`, `physical_v4`.
7. Add `scripts/ppo_archive.sh` — runs after each iteration's promotion gate, see "Archive policy".
8. Add dashboard replay save for promoted checkpoints (hot on A, archive on B).
9. Keep `ppo.py` as the public CLI wrapper once the above exists.

## Practical first run

Suggested first real run:

```text
run_id: ppo_v2_cpu_<timestamp>
policy init: best supervised entity_encoder PairHead checkpoint (latest top4 ckpt)
T: 1
mode: 2P only
action contract: single_launch_v1
freeze: Phase 0 (PairHead output layers + PPO-only heads); no distributed training yet

rollouts: 100% self-play, parallel on A and B
  pool init: [transformer_v2_baseline]  # all opponent slots fall back to baseline
  pool grows only by adding promoted PPO checkpoints, cap N_pool = 8
  per-episode opponent sampling:
    50% latest promoted PPO checkpoint (baseline fallback until first promotion)
    30% transformer_v2_baseline (floor)
    20% uniform from older promoted pool members (baseline fallback when empty)

rollout per iter: 128 episodes total
  Machine A: 60 episodes (+ GAE merge + optim + eval half)
  Machine B: 68 episodes (+ eval half)
  N_env per worker: 4
  worker processes: physical_cores - 1 per machine

training (Phase 0 — A only):
  update epochs: 3
  minibatch: 1024  (CPU-friendly; smaller than the supervised default)
  lr heads: 1e-4
  lr trunk: frozen
  fraction distribution: logit-normal, shared learned log_std
  bc_coef: 0.05  (anchor to supervised pair cache on A; doubled to 0.10
                  once Phase 1 distributed training starts, since BC loss
                  lives only on A's gradient)

eval (deterministic, every iteration, split A/B):
  - SEEDS_QUICK vs previous_promoted                (direct promotion opponent)
  - SEEDS_QUICK vs transformer_v2_baseline           (must stay near or above floor)
  - SEEDS_QUICK vs physical_v4                       (external sanity yardstick)

full eval (every 10 iterations):
  - 128 seeds vs previous_promoted + transformer_v2_baseline + physical_v4 + sniper_v2

archive (after every iteration's promotion gate):
  - scripts/ppo_archive.sh moves cold checkpoints + rollouts + eval JSON
    to B's archive/ dir; deletes local after rsync success
```

Promotion rule (see "Evaluation gate"):

```text
Promote candidate `policy_v(K+1)` only if all are true:
  1. winrate vs previous_promoted ≥ 50% with non-negative paired seed-seat score
  2. winrate vs transformer_v2_baseline ≥ previous_promoted baseline winrate - 2pp
  3. winrate vs physical_v4 drops by ≤ 5pp
  4. invalid launch rate ≤ 1.1 × previous_promoted's rate

The direct previous-promoted match is the load-bearing signal. The baseline
and physical_v4 matches are regression guards.
```

Do not scale to larger asynchronous rollouts until this synchronous two-machine
loop shows stable KL, stable entropy, and non-degrading miss matrices.
