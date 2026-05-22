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
L0 specialists       frozen initially
L1-L4 perception     frozen at first, optionally unfrozen late
Pair policy head     trainable
ship-fraction head   trainable
value head           trainable
```

The first PPO version should be conservative: keep the environment-facing
launches safe through `physics_utils.plan_launch`, keep rollouts strictly
on-policy, and measure policy quality with deterministic A/B evaluation after
every PPO iteration.

## Machine roles

Use a synchronous coordinator setup where **both machines do rollouts AND
both contribute to the PPO update**. A is the coordinator (owns the run
directory, makes the optimizer step) and B is a peer (computes gradients on
its half of the data and ships them to A). This is data-parallel training
with one-way file-mediated grad sync, not classic NCCL DDP — the one-way
SSH constraint (see "Connectivity" below) rules out symmetric all-reduce.

| Machine | Rollout | Training | Eval | Coordination |
|---|---|---|---|---|
| Machine A | ~50% of episodes | half of every minibatch (forward+backward, applies optimizer) | quick-eval half + full-eval | owns run dir, advantages, optim state, promotion gate, archive |
| Machine B | ~50% of episodes | half of every minibatch (forward+backward, writes grad to file) | quick-eval half | passive — runs the worker process A drives via rsync |

Why split this way:

- **Rollout is the dominant CPU cost** (500 steps × N episodes × forward
  pass on a 256-dim transformer). Splitting 50/50 doubles rollout
  throughput, which is the biggest win.
- **The PPO update is much cheaper** but worth distributing once the
  trunk is unfrozen (Phase 1+). Cost of distribution = grad/weight file
  rsync per minibatch (see "CPU performance" and "Distributed training"
  below).
- **Eval also splits.** Quick-eval is 32 seeds × 2 seats × 3 opponents
  ≈ 192 episodes; halving it between A and B saves real wall-clock.

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
    policy_v000.pt
    policy_v001.pt
    ...
  rollouts/
    v000/
      machine_a/
        shard_000001.pt
      machine_b/
        shard_000001.pt
  eval/
    v000.json
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

# A -> B : refresh the self-play opponent pool (frozen older PPO ckpts + baseline)
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
- `--delete` is only safe on the pool-publish direction (A is the source of
  truth for which frozen checkpoints are in the pool). **Never** use
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

### Eval

Quick-eval = 32 seeds × 2 seats × 3 opponents = 192 episodes per
iteration. That's 1.5× the rollout budget — eval can easily dominate
if run only on A.

1. **Split quick-eval 50/50 between A and B.** A runs seeds 0..15, B
   runs seeds 16..31 (both seats). A pulls B's eval JSON via rsync,
   merges, applies promotion gate.

2. **Full-eval (128 seeds) every 5–10 iterations**, not every
   iteration. Quick-eval is sufficient for the per-iter promotion gate.

3. **Skip eval when iteration KL was negligible.** If approx_KL < 0.2 ×
   target_KL, the policy barely changed; the eval result will be
   nearly identical to the prior promoted checkpoint. Save the CPU.

### Shard format (storage / rsync bandwidth)

Two options. Pick before implementation.

| Option | Shard contents | Pros | Cons |
|---|---|---|---|
| **A. Featurized** | `planet_features`, `comet_features`, `fleet_features`, masks, action, logprob, value, reward, done | Training is fast — just forward through the stored tensors | Large (~MB per episode); slow rsync from B; 2–4× the disk |
| **B. Raw obs + policy outputs** | raw env obs dict, action_id, frac, logprob, value, reward, done | ~10× smaller shards, fast rsync | Training has to re-run featurization, costing ~3× CPU per epoch (×N PPO epochs) |

For CPU + Tailscale, **Option B is recommended** once the loop is
verified. Storage and rsync win exceeds the re-featurization cost when
PPO epochs ≤ 3.

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
| `data/runs/ppo/<run_id>/checkpoints/policy_v[K-N_pool+1..K+1].pt` | active self-play pool + just-trained ckpt |
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
| `data/runs/ppo/<run_id>/checkpoints/policy_v[0..K-N_pool].pt` | `archive/runs/ppo/<run_id>/checkpoints/` | after each promotion |
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
run_id="$1"; K="$2"
pool_floor=$((K - 8))      # N_pool = 8
roll_floor=$((K - 2))
eval_floor=$((K - 11))

# 1. ship cold checkpoints to B
for v in $(seq 0 "$pool_floor"); do
  [ -f "data/runs/ppo/$run_id/checkpoints/policy_v$v.pt" ] || continue
  [ "$v" = "0" ] && continue  # keep v0 hot (the init)
  rsync -av --remove-source-files \
    "data/runs/ppo/$run_id/checkpoints/policy_v$v.pt" \
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
B's rollout workers load pool checkpoints from the *hot* path. The
script's `pool_floor` arithmetic enforces this.

## Policy/action contract

### Semantic action

The PPO policy samples a semantic action:

```text
noop_or_launch
source_idx
target_idx
ship_fraction
```

For the current PairHead design, the simplest distribution is:

```text
pair_logits:       (P, P) source→target logits
noop_logit:        scalar
frac_distribution: conditional on sampled pair
value:             scalar V(s)
```

Flatten action candidates as:

```text
action 0                = NOOP
action 1 + source*P+tgt = launch(source, target)
```

Use a cheap legality mask before sampling:

- source exists
- source is owned by the learner
- source has surplus above `min_launch`
- target exists
- source != target

Do **not** mask by expensive full physics for every pair in v1. That makes CPU
rollout too slow and can also hide useful negative signal.

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
    "episodes": [
        {
            "seed": int,
            "seat": int,
            "agent_slots": list[str],
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
                "action_id":       Tensor[n],      # 0 = noop, >0 = pair
                "frac_sample":     Tensor[n],
                "logprob":         Tensor[n],
                "value":           Tensor[n],
                "reward":          Tensor[n],
                "done":            Tensor[n],

                # diagnostics
                "invalid_launch":  Tensor[n],
                "emitted_launch":   Tensor[n],      # bool
                "source_pid":       Tensor[n],
                "target_pid":       Tensor[n],
            },
        },
    ],
}
```

Use CPU tensors and save with `torch.save`. Prefer `float16` or compressed
`npz` only after the loop is correct; debuggability matters more at first.

If rollout files become too large, the first optimization is to store raw obs
plus policy outputs and re-featurize on the learner. Do not optimize storage
before the PPO math is verified.

## Iteration lifecycle

One synchronous PPO iteration. The "training" step has two variants:
**Phase 0 = A-only** (heads update too cheap to distribute);
**Phase 1+ = distributed** via file-mediated grad averaging.

```text
1.  A: publish checkpoints/policy_vK.pt; rsync push to B.
2.  Both: load exactly policy_vK.pt (eval mode, inference_mode for rollout).
3.  Both: collect rollout shards in parallel using frozen policy_vK.
4.  Both: write shards into rollouts/vK/<machine_id>/ atomically.
5.  A: background-rsync-pull B's shards as they appear.
6.  A: stop when rollout budget hit; ignore late shards.
7.  A: compute GAE on the merged rollout set (advantages, returns).
8a. Phase 0: A performs the PPO update locally, writes policy_vK+1.pt.
8b. Phase 1+: distributed update —
      i.   A splits minibatches; rsync push mb_B sequence to B.
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

**Rollouts are 100% self-play.** The learner sees a snapshot of itself
from a pool of frozen prior versions. This keeps the rollout signal at
or above the current policy's level and avoids the classic PPO trap of
collapsing to a stable point against weak external opponents.

### Self-play pool

Maintain a pool of frozen policy checkpoints. Each rollout episode
samples one opponent from the pool:

| Probability | Opponent |
|---:|---|
| 50% | **latest frozen PPO checkpoint** (`policy_v(K-1).pt`) |
| 30% | **`transformer_v2_baseline`** (frozen May-20 supervised ckpt — the lower bound; the policy must keep beating it) |
| 20% | **uniformly sampled older PPO checkpoint** (`policy_v_i` for `i ∈ [max(0, K-N_pool), K-2]`) |

Pool size `N_pool = 8` is a reasonable starting point. The older-uniform
slot exists to prevent cyclic policy chasing (NFSP / Fictitious Self-Play
intuition): if the learner forgets how to beat a strategy from 5 iters
ago, the pool resurfaces it.

The current policy `vK` never plays itself directly — always against
*frozen* opponents. This keeps the policy gradient on the learner side
unbiased; opponent moves are sampled but their logprobs aren't tracked.

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

Add small dense shaping to reduce variance:

```text
score_margin_t = my_total_ships - max_enemy_total_ships
dense_t = clip((score_margin_t - score_margin_t-1) / 100.0, -0.05, 0.05)
```

Recommended first reward:

```text
reward_t = dense_t
reward_terminal += env_terminal_reward
reward_t -= 0.01 * invalid_launch
reward_t -= 0.002 * emitted_launch_that_hits_boundary_or_sun_if_detected
```

Keep shaping small. If dense reward dominates terminal win/loss, PPO will
learn to inflate short-term ship count instead of winning.

## PPO hyperparameters

Initial CPU-friendly defaults:

| Parameter | Value |
|---|---:|
| `gamma` | `0.995` |
| `gae_lambda` | `0.95` |
| PPO clip | `0.10` |
| target KL | `0.01` |
| update epochs | `2-4` |
| minibatch size | `1024-4096` steps |
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
pair policy head
ship-fraction head
value head
```

Everything else frozen. Goal: prove PPO update works and does not destroy the
BC policy.

### Phase 1 — action adaptation

Unfreeze:

```text
PairHead
JointRoleAttention (L4)
value head
ship-fraction head
```

Keep L0-L3 frozen. Use low LR for L4.

### Phase 2 — strategic adaptation

If Phase 1 plateaus and eval is stable, unfreeze:

```text
DualRoleAttention (L3)
JointRoleAttention (L4)
PairHead
value/fraction/noop heads
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
opponents = physical_v4, sniper_v2
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
Only mark policy_vK as promoted if it beats the previous promoted checkpoint
on SEEDS_QUICK without increasing invalid/miss rate materially.
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
| self-play winrate up but heuristic eval drops | policy overfit to current pool; cyclic chase | raise the older-uniform pool slot from 20% → 40%; lower the latest-frozen slot accordingly; check `transformer_v2_baseline` winrate is still ≥ 50% (it's the floor) |
| pool keeps cycling (winrate vs each pool member oscillates) | pool too small / latest-only sampling | grow `N_pool` from 8 → 16; lower the latest-frozen slot to ≤ 40% |
| CPU rollout too slow | one env per worker, no `torch.compile` | enable batched envs (`N_env=4-8`), `torch.compile(mode="reduce-overhead")`, share opponent model across workers (see "CPU performance → Rollout") |
| grad sync dominates iter wall-clock in Phase 1+ | per-minibatch full-grad rsync | switch to local SGD (avg every K=4 steps), or shrink minibatch count |
| disk on A fills up | rollouts/checkpoints accumulating | run `scripts/ppo_archive.sh` after each promotion; the script moves cold data to B's `archive/` |
| B's pool checkpoint sampled but file missing on B | new pool member not yet rsynced | A must push the new ckpt to B *before* publishing v(K+1), not after |
| `policy_vK+1` differs between A and B after distributed update | non-atomic weight write or B loaded stale weights | verify `.tmp → rename` for weight files; verify B's load step fences against partial reads |

## Minimal implementation checklist

1. Add actor/value wrapper around current `EntityPretrainModel`. Make sure
   the forward accepts batch dim > 1 (it already does — verified in
   `agents/transformer_v2/encoder/entity_encoder.py:209`) and add a
   batched rollout path that runs `N_env` envs through one forward.
2. Add `noop_logit` and `pair_frac` distribution head if not already exposed
   in the runtime policy. Pin the `frac` distribution choice (Beta vs
   clamped-Gaussian vs K-way Categorical) before writing logprob code.
3. Implement `ppo_rollout_worker.py`:
   - load checkpoint version + opponent pool (loaded once per worker process)
   - run assigned episodes with `torch.inference_mode()` + `torch.compile`
   - `N_env` envs per worker, batched forward
   - write rollout shards atomically (`.tmp → rename`)
4. Implement `ppo_learner.py`:
   - wait for rollout budget (with wall-clock timeout)
   - rsync-pull B's shards
   - compute GAE
   - **Phase 0**: local PPO update only
   - **Phase 1+**: distributed PPO update via grad/weight file sync
     (see "Distributed training")
   - publish next checkpoint; rsync push to B
5. Add `ppo_train_peer.py` for Machine B (Phase 1+):
   - load minibatch sequence A pushed
   - per minibatch: forward+backward, write `grad_B.pt` atomically, poll for new weights
6. Add deterministic eval script using `utils.eval_seeds`, split-aware
   (so A runs seeds 0..15 and B runs 16..31, both seats).
7. Add `scripts/ppo_archive.sh` — runs after each promotion, see "Archive policy".
8. Add dashboard replay save for promoted checkpoints (hot on A, archive on B).
9. Keep `ppo.py` as the public CLI wrapper once the above exists.

## Practical first run

Suggested first real run:

```text
run_id: ppo_v2_cpu_<timestamp>
policy init: best supervised entity_encoder PairHead checkpoint (latest top4 ckpt)
T: 1
mode: 2P only
freeze: Phase 0 (heads only); no distributed training yet

rollouts: 100% self-play, parallel on A and B
  pool init: [transformer_v2_baseline]
  pool grows by adding each promoted PPO checkpoint, cap N_pool = 8
  per-episode opponent sampling:
    50% latest frozen PPO checkpoint
    30% transformer_v2_baseline (floor)
    20% uniform from older frozen pool members

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
  bc_coef: 0.05  (anchor to supervised pair cache on A; doubled to 0.10
                  once Phase 1 distributed training starts, since BC loss
                  lives only on A's gradient)

eval (deterministic, every iteration, split A/B):
  - SEEDS_QUICK vs transformer_v2_baseline          (must stay > 50%)
  - SEEDS_QUICK vs latest 3 pool members             (track pool dominance)
  - SEEDS_QUICK vs physical_v4                       (external sanity yardstick)

full eval (every 10 iterations):
  - 128 seeds vs transformer_v2_baseline + physical_v4 + sniper_v2

archive (every promotion):
  - scripts/ppo_archive.sh moves cold checkpoints + rollouts + eval JSON
    to B's archive/ dir; deletes local after rsync success
```

Promotion rule (see "Evaluation gate"):

```text
Promote policy_vK only if BOTH:
  1. winrate vs transformer_v2_baseline ≥ winrate(prev_promoted vs baseline)
  2. invalid launch rate ≤ 1.1 × prev_promoted's rate

The baseline winrate is the load-bearing signal — if PPO can't beat the
frozen supervised checkpoint we started from, it's regressing.
```

Do not scale to larger asynchronous rollouts until this synchronous two-machine
loop shows stable KL, stable entropy, and non-degrading miss matrices.
