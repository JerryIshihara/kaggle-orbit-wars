# PPO pseudo code — transformer_v2

Algorithm-level sketch of the PPO loop described in
`docs/PPO_TWO_CPU_PROTOCOL.md`. The real implementation does not exist yet
(`agents/transformer_v2/ppo.py` is a stub); this file is the reference the
implementation should match.

Notation:

```
P              = number of planet slots (planet + comet)
T              = 1 frame per step in v1 (fleet history disabled for CPU rollout)
policy_vK      = frozen PPO checkpoint version K
baseline       = transformer_v2_baseline (May-20 supervised ckpt, the floor)
pool           = ring buffer of frozen PPO ckpts, cap N_pool = 8
```

---

## Top-level driver

```python
def train_ppo(run_dir, init_ckpt, max_iters):
    policy = load(init_ckpt)                      # supervised PairHead
    save_checkpoint(run_dir, policy, version=0)
    pool = [baseline]                             # self-play opponent pool
    promoted = 0

    for K in range(max_iters):
        publish_checkpoint(run_dir, K)            # workers see policy_vK

        # ---- 1. synchronous rollout (both machines) ---------------------
        shards = wait_for_rollout_budget(run_dir, version=K,
                                         target_episodes=128)
        rollouts = load_shards(shards)            # strictly on-policy: vK only

        # ---- 2. PPO update ----------------------------------------------
        policy_next = ppo_update(policy, rollouts, bc_minibatch_source=pair_cache)

        # ---- 3. deterministic eval gate ---------------------------------
        metrics = eval_quick(policy_next,
                             opponents=[baseline] + pool[-3:] + [physical_v4])

        # ---- 4. promotion + pool maintenance ----------------------------
        if promote_ok(metrics, prev=metrics_of(promoted)):
            promoted = K + 1
            pool.append(freeze(policy_next))
            if len(pool) > N_pool: pool.pop(1)    # keep baseline at index 0

        policy = policy_next
        save_checkpoint(run_dir, policy, version=K + 1)
        log_jsonl(run_dir, K, metrics)
```

---

## Rollout worker (Machine A and Machine B)

```python
def rollout_worker(run_dir, machine_id):
    while True:
        K, policy_vK = wait_for_new_checkpoint(run_dir)
        opp_sampler = SelfPlayPool(latest=policy_vK,
                                    baseline=baseline,
                                    older=pool[:-1],
                                    weights=(0.50, 0.30, 0.20))
        assigned = episodes_for_machine(machine_id, K)
        for seed in assigned:
            opp = opp_sampler.sample()
            ep  = run_episode(policy_vK, opp, seed,
                              learner_seat=random.choice([0, 1]))
            atomic_write_shard(run_dir, K, machine_id, ep)
```

```python
def run_episode(policy, opponent, seed, learner_seat):
    obs    = env.reset(seed=seed)
    buffer = EpisodeBuffer()
    done   = False

    while not done:
        if obs.current_seat == learner_seat:
            feats             = featurize(obs, T=1)
            pair_logits, noop_logit, frac_dist, value = policy(feats)
            mask              = legality_mask(obs, learner_seat)      # cheap mask
            action_id, logp_a = sample_action(pair_logits, noop_logit, mask)
            frac, logp_f      = sample_frac(frac_dist, action_id)
            env_action, ok    = project_to_env(obs, action_id, frac)  # plan_launch
            buffer.add(feats, action_id, frac, logp_a + logp_f,
                       value, invalid=not ok)
        else:
            env_action = opponent.act(obs)

        obs, reward, done, info = env.step(env_action)
        if buffer.last_was_learner:
            shaped = reward_shaping(info, invalid=buffer.last_invalid)
            buffer.set_reward(shaped)

    buffer.set_terminal(env.winner == learner_seat)
    return buffer.finalize()
```

---

## Action sampling and env projection

```python
def legality_mask(obs, seat):
    # Source must be own & have surplus; target must exist & != source.
    own       = (obs.planets.owner == seat)
    has_ships = (obs.planets.ships > obs.planets.min_launch)
    exists    = obs.planets.exists
    src_ok    = own & has_ships & exists                       # (P,)
    tgt_ok    = exists                                         # (P,)
    pair_ok   = src_ok[:, None] & tgt_ok[None, :]              # (P, P)
    pair_ok  &= ~eye(P, dtype=bool)                            # src != tgt
    return concat([true_scalar(), pair_ok.flatten()])          # +1 NOOP slot

def sample_action(pair_logits, noop_logit, mask):
    flat = concat([noop_logit[None], pair_logits.flatten()])
    flat = where(mask, flat, -inf)
    dist = Categorical(logits=flat)
    a    = dist.sample()
    return a, dist.log_prob(a)

def project_to_env(obs, action_id, frac):
    if action_id == 0:                                  # NOOP
        return [], ok=True
    src, tgt = unflatten(action_id - 1, P, P)
    ships    = max(int(frac * obs.planets[src].ships), 1)
    launch   = physics_utils.plan_launch(obs, src, tgt, ships)
    if not launch.ok:
        return [], ok=False                              # invalid -> penalty
    return [[src, launch.angle, ships]], ok=True
```

Never resample on invalid: the sampled action stays in the buffer with its
original logprob and earns an `invalid_launch_penalty`. Resampling would
desync stored logprob from executed action and break the PPO ratio.

---

## Reward shaping

```python
def reward_shaping(info, invalid):
    score_margin = info.my_ships - max(info.enemy_ships)
    dense        = clip((score_margin - info.prev_margin) / 100.0, -0.05, 0.05)
    r  = dense
    r -= 0.010 * invalid
    r -= 0.002 * info.boundary_or_sun_hit
    if info.done:
        r += +1.0 if info.winner == info.learner_seat else (
              0.0 if info.draw else -1.0)
    return r
```

---

## Advantages (GAE)

```python
def compute_advantages(rollouts, gamma=0.995, lam=0.95):
    for ep in rollouts:
        v = ep.values; r = ep.rewards; d = ep.dones
        gae = 0; adv = zeros_like(r)
        for t in reversed(range(len(r))):
            v_next      = 0 if d[t] else v[t + 1] if t + 1 < len(r) else 0
            delta       = r[t] + gamma * v_next * (1 - d[t]) - v[t]
            gae         = delta + gamma * lam * (1 - d[t]) * gae
            adv[t]      = gae
        ep.advantages = adv
        ep.returns    = adv + v
    flat_adv = concat([ep.advantages for ep in rollouts])
    return normalize(flat_adv)                                 # mean 0, std 1
```

---

## PPO update

```python
def ppo_update(policy, rollouts,
               clip=0.10, target_kl=0.01, epochs=3, minibatch=2048,
               lr_heads=1e-4, lr_trunk=None,             # trunk frozen in Phase 0
               value_coef=0.5, ent_coef=0.01,
               bc_coef=0.05, max_grad_norm=0.5):

    compute_advantages(rollouts)
    batch = flatten(rollouts)                            # one big tensor dict
    opt   = build_optimizer(policy, lr_heads, lr_trunk)

    for epoch in range(epochs):
        approx_kl_running = 0.0
        for mb in iter_minibatches(batch, minibatch, shuffle=True):
            pair_logits, noop_logit, frac_dist, value = policy(mb.feats)
            new_logp  = action_logprob(pair_logits, noop_logit, frac_dist,
                                       mb.action_id, mb.frac)
            ratio     = exp(new_logp - mb.old_logp)
            unclipped = ratio * mb.adv
            clipped   = clip(ratio, 1 - clip, 1 + clip) * mb.adv
            policy_loss = -mean(min(unclipped, clipped))

            value_loss  = mean((value - mb.returns) ** 2)
            entropy     = mean(action_entropy(pair_logits, noop_logit, frac_dist))

            # BC anchor — pulled from supervised pair cache, NOT rollouts.
            bc_batch    = pair_cache.sample(mb.size)
            bc_logits, *_ = policy(bc_batch.feats)
            bc_loss     = bce(bc_logits, bc_batch.expert_pair_labels)

            loss = (policy_loss
                    + value_coef * value_loss
                    - ent_coef   * entropy
                    + bc_coef    * bc_loss)

            opt.zero_grad()
            loss.backward()
            clip_grad_norm(policy.parameters(), max_grad_norm)
            opt.step()

            approx_kl_running += mean(mb.old_logp - new_logp).item()

        if approx_kl_running / num_minibatches > 1.5 * target_kl:
            break                                              # early stop
    return policy
```

---

## Self-play pool sampling

```python
class SelfPlayPool:
    # weights = (latest, baseline, older_uniform) = (0.50, 0.30, 0.20)
    def sample(self):
        u = random()
        if u < 0.50: return self.latest               # policy_v(K-1)
        if u < 0.80: return self.baseline             # transformer_v2_baseline
        return random.choice(self.older) if self.older else self.baseline
```

Pool invariants:
- `latest` is always a *frozen* checkpoint; the current policy never plays itself.
- `baseline` (`transformer_v2_baseline`) stays in the pool forever — it is the
  load-bearing floor. If baseline winrate dips below 50%, block promotion.
- `older` capped at `N_pool - 1 = 7` to prevent cyclic chasing.

---

## Promotion gate

```python
def promote_ok(metrics, prev):
    wr_vs_baseline = metrics.winrate["baseline"]
    inv_rate       = metrics.invalid_launch_rate
    return (wr_vs_baseline >= prev.winrate["baseline"]
            and inv_rate     <= 1.1 * prev.invalid_launch_rate)
```

If `promote_ok` returns false the checkpoint still gets archived but is **not**
added to the self-play pool and is **not** considered for submission.

---

## Freeze schedule (parameter groups)

| Phase | Trainable | Frozen | Notes |
|---|---|---|---|
| 0 | noop / pair / frac / value heads | L0-L4 perception, PairHead trunk | "PPO plumbing" — prove the update doesn't destroy the BC policy |
| 1 | + PairHead, + JointRoleAttention (L4) | L0-L3 | LR for L4 = `1e-5`; heads stay at `1e-4` |
| 2 | + DualRoleAttention (L3) | L0 specialists | Only if Phase 1 plateaus *and* eval stays stable |

---

## End-of-iteration log (one JSONL row per iter K)

```
iter, policy_version, n_rollout_eps,
winrate_baseline, winrate_pool_latest, winrate_pool_older, winrate_physical_v4,
winrate_by_seat[0|1], reward_mean,
invalid_launch_rate, emitted_launch_rate, miss_matrix_summary,
entropy, approx_kl, value_explained_var, ep_len_mean, ep_len_median,
promoted (bool)
```

The dashboard reads this file directly; no additional aggregation step needed.
