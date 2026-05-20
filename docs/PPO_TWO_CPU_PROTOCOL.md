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

Use a synchronous coordinator/worker setup.

| Machine | Role | Work |
|---|---|---|
| Machine A | learner + local rollout worker | owns run directory, publishes policy checkpoints, waits for rollout shards, performs PPO update, runs eval |
| Machine B | rollout worker only | polls for latest policy checkpoint, collects rollout shards, uploads/writes them atomically |

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

The two machines can synchronize this directory through a shared filesystem,
`rsync`, or the existing GCS bucket workflow. The important rule is atomic
rollout writes:

```text
write shard_000123.pt.tmp
fsync/close
rename shard_000123.pt.tmp -> shard_000123.pt
```

The learner must never read `*.tmp`.

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

One synchronous PPO iteration:

```text
1. Learner publishes checkpoints/policy_vK.pt.
2. Both machines load exactly policy_vK.pt.
3. Workers collect rollout shards using frozen policy_vK.
4. Workers write shards into rollouts/vK/<machine_id>/.
5. Learner waits until target rollout budget is reached.
6. Learner ignores any late shards after the cutoff.
7. Learner computes advantages from policy_vK values.
8. Learner performs PPO update and writes policy_vK+1.pt.
9. Learner runs deterministic evaluation for policy_vK+1.
10. Repeat.
```

Strict on-policy rule for v1:

```text
train vK update only on rollouts generated by policy_vK
```

Do not train on stale shards from `vK-1`. With only two machines, synchronous
collection is simpler and reliable enough.

## Rollout budget

Start small enough to iterate quickly, then scale.

| Phase | Total rollout per iteration | Purpose |
|---|---:|---|
| Smoke | 8-16 episodes | verify tensors/logprobs/advantages/update |
| Bring-up | 64-128 episodes | tune reward/action penalties |
| Stable PPO | 256-512 episodes | real learning signal |

On two CPU machines, split the target approximately by available cores:

```text
Machine A: 35-45% of rollout budget, because it also trains/evals
Machine B: 55-65% of rollout budget
```

If each machine has many cores, run one env process per physical core minus
one. Avoid oversubscription; Orbit Wars is Python-heavy and too many workers
can slow total throughput.

## Opponent schedule

Start with 2-player games. Add 4-player only after 2P improves.

Recommended opponent mixture per rollout episode:

| Probability | Opponent |
|---:|---|
| 40% | `physical_v4` |
| 20% | `sniper_v2` |
| 20% | latest frozen PPO checkpoint from the last promoted version |
| 10% | current self-play mirror |
| 10% | `random_v1` or weaker heuristic for exploration sanity |

Play both seats. For every sampled seed, assign learner to seat 0 or seat 1
with equal probability. Evaluation must always run both seats.

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
| self-play improves but heuristic eval drops | overfitting to current self | increase `physical_v4`/`sniper_v2` mixture |
| CPU rollout too slow | too much tensor storage or T=6 | start T=1, fewer evals during smoke, profile featurization |

## Minimal implementation checklist

1. Add actor/value wrapper around current `EntityPretrainModel`.
2. Add `noop_logit` and `pair_frac` distribution head if not already exposed
   in the runtime policy.
3. Implement `ppo_rollout_worker.py`:
   - load checkpoint version
   - run assigned episodes
   - write rollout shards atomically
4. Implement `ppo_learner.py`:
   - wait for rollout budget
   - compute GAE
   - PPO update
   - publish next checkpoint
5. Add deterministic eval script using `utils.eval_seeds`.
6. Add dashboard replay save for promoted checkpoints.
7. Keep `ppo.py` as the public CLI wrapper once the above exists.

## Practical first run

Suggested first real run:

```text
run_id: ppo_v2_cpu_<timestamp>
policy init: best supervised entity_encoder PairHead checkpoint
T: 1
mode: 2P only
rollout per iter: 128 episodes total
Machine A: 48 episodes + learner/eval
Machine B: 80 episodes
update epochs: 3
minibatch: 2048
lr heads: 1e-4
lr trunk: frozen
bc_coef: 0.05
eval: every iteration on SEEDS_QUICK vs physical_v4
full eval: every 10 iterations on 128 seeds vs physical_v4 + sniper_v2
```

Do not scale to larger asynchronous rollouts until this synchronous two-machine
loop shows stable KL, stable entropy, and non-degrading miss matrices.
