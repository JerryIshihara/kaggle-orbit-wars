# PPO pseudo code — transformer_v2

Algorithm-level sketch of the PPO loop described in
`docs/PPO_TWO_CPU_PROTOCOL.md`. The real implementation does not exist yet
(`agents/transformer_v2/ppo.py` is a stub); this file is the reference the
implementation should match.

Notation:

```
P              = number of planet slots (planet + comet)
T              = 1 frame per step in v1 (fleet history disabled for CPU rollout)
N_env          = envs batched per worker forward pass (start at 4)
policy_vK      = frozen PPO checkpoint version K
baseline       = transformer_v2_baseline (May-20 supervised ckpt, the floor)
pool           = ring buffer of promoted frozen PPO ckpts, cap N_pool = 8
A, B           = Machine A (coordinator) and Machine B (peer)
```

Both machines do rollouts in parallel. Both contribute to the PPO update
once Phase 1 unfreezes the trunk; Phase 0 (heads only) is A-only because
the grad sync overhead exceeds the compute saved.

---

## Top-level driver (Machine A — coordinator)

```python
def train_ppo(run_dir, init_ckpt, max_iters, phase):
    policy = PPOActorCritic(load(init_ckpt))      # supervised PairHead + PPO heads
    save_checkpoint(run_dir, policy, version=0)
    rsync_push_to_B(run_dir, "checkpoints/policy_v0.pt")
    pool = PromotedPool(baseline=baseline, max_size=N_pool)
    promoted = baseline                           # latest promoted opponent

    # ---- 0. startup calibration (once per run) --------------------------
    # Bootstrap metrics_of(promoted) for the iter-1 relative thresholds.
    # See docs/PPO_TWO_CPU_PROTOCOL.md → "Startup calibration".
    metrics_history = {baseline: eval_quick(baseline, seeds=range(0, 32),
                                             opponents=[baseline, physical_v4])}
    persist_eval(run_dir, version="v0_baseline", metrics=metrics_history[baseline])
    assert 0.45 <= metrics_history[baseline].winrate[baseline] <= 0.55, (
        "Baseline self-eval is not ~0.50 — eval harness is broken; fix before iter 0.")

    for K in range(max_iters):
        publish_checkpoint(run_dir, K)            # workers see policy_vK
        rsync_push_to_B(run_dir, f"checkpoints/policy_v{K}.pt")
        publish_pool(run_dir, pool)               # promoted opponents only
        rsync_push_to_B(run_dir, "pool/")          # manifest
        rsync_push_active_pool_ckpts_to_B(run_dir, pool)

        # ---- 1. parallel rollout (both machines, ~50/50 split) ----------
        spawn_local_workers(K, pool, share=0.50)   # A's rollout workers
        # B's rollout workers are spawned by a peer daemon — see Worker B.
        shards = wait_for_rollout_budget_with_pull(
                     run_dir, version=K, target_episodes=128,
                     pull_from_B=True)              # background rsync pull
        rollouts = load_shards(shards)              # strictly on-policy: vK only

        # ---- 2. PPO update ----------------------------------------------
        if phase == 0:
            policy_next = ppo_update_local(policy, rollouts, pair_cache)
        else:                                     # phase >= 1
            policy_next = ppo_update_distributed(policy, rollouts,
                                                  pair_cache, K)

        save_checkpoint(run_dir, policy_next, version=K + 1)
        rsync_push_to_B(run_dir, f"checkpoints/policy_v{K+1}.pt")

        # ---- 3. eval (split between A and B) ----------------------------
        metrics_A = eval_quick(policy_next, seeds=range(0, 16),
                               opponents=[promoted, baseline, physical_v4])
        request_peer_eval(version=K + 1,
                          opponents=[promoted, baseline, physical_v4])
        metrics_B = rsync_pull_eval_from_B(run_dir, version=K + 1)
        metrics   = merge_eval(metrics_A, metrics_B)

        # ---- 4. promotion + pool maintenance ----------------------------
        if promote_ok(metrics, prev=metrics_history[promoted]):
            promoted = freeze(policy_next, version=K + 1)
            pool.add(promoted)
            metrics_history[promoted] = metrics       # gate uses these next iter

        policy = policy_next
        log_jsonl(run_dir, K, metrics)

        # ---- 5. archive cold artifacts to B -----------------------------
        run("scripts/ppo_archive.sh", run_id, K)  # cold ckpts/rollouts/eval -> B
```

## Peer driver (Machine B — passive)

```python
def run_peer(run_dir):
    # B has no top-level loop of its own; it polls the run dir for
    # signals A drops via rsync push.
    while True:
        K = wait_for_new_checkpoint(run_dir)      # policy_v(K).pt appeared
        policy_vK = load(checkpoint(K))
        pool = current_promoted_pool(run_dir)

        # 1. spawn B's rollout workers (independent of A's)
        spawn_local_workers(K, pool=pool, share=0.50)
        wait_for_rollout_complete(K)

        # 2. if Phase 1+, wait for mb_B sequence A pushes, then peer-train.
        if exists(run_dir, f"sync/v{K}/mb_B_ready"):
            ppo_peer_train_loop(run_dir, policy_vK, K)

        # 3. quick-eval is requested after A publishes policy_v(K+1).
        eval_req = wait_for_eval_request()
        eval_policy = load(checkpoint(eval_req.version))
        eval_quick_peer(eval_policy, seeds=range(16, 32),
                        opponents=eval_req.opponents,
                        write_to=f"{run_dir}/eval/v{eval_req.version}_B.json")
```

---

## Actor-critic wrapper

```python
class PPOActorCritic(nn.Module):
    def __init__(self, entity_model):
        self.entity_model = entity_model
        self.player_context = PlayerContextLearner(entity_model.d_model)  # planned
        self.strategy = StrategyLearner(entity_model.d_model)             # planned
        self.noop_head = nn.Linear(entity_model.d_model, 1)
        self.value_head = nn.Linear(entity_model.d_model, 1)
        self.frac_log_std = nn.Parameter(tensor(log(0.35)))

    def forward(self, feats):
        base = self.entity_model.forward_with_context(feats)
        # base exposes pair_logits, pair_frac, and either global_context or ctx_now.
        glob = base.global_context
        if glob is None:
            glob = masked_mean(base.ctx_now, feats.planet_mask, dim=1)
        learner_ctx, player_ctx = self.player_context(base.ctx_now, glob, feats)
        strategy_ctx = self.strategy(glob, learner_ctx, player_ctx)
        policy_ctx = glob + learner_ctx + strategy_ctx
        return {
            "pair_logits": base.pair_logits,
            "frac_loc": base.pair_frac,             # logit-normal mean
            "noop_logit": self.noop_head(policy_ctx).squeeze(-1),
            "value": self.value_head(policy_ctx).squeeze(-1),
            "frac_log_std": clamp(self.frac_log_std, log(0.15), log(0.80)),
        }
```

The real implementation may initially replace `PlayerContextLearner` and
`StrategyLearner` with identity/zero modules. The architectural boundary still
matters: L0-L2 are world perception; player context, strategy, and action
heads are policy adaptation.

---

## Rollout worker (one process per physical core minus one)

The worker batches `N_env` envs per forward pass — the single biggest CPU
win over the naive one-env-per-step design.

```python
def rollout_worker(run_dir, machine_id, worker_id, N_env=4):
    K, policy_vK = wait_for_new_checkpoint(run_dir)
    policy_vK    = torch.compile(policy_vK.eval(), mode="reduce-overhead")
    opp_pool     = load_promoted_pool_once(run_dir)   # never includes policy_vK
    assert checkpoint_id(policy_vK) not in opp_pool.ids
    for opp_id, opp in opp_pool.models.items():
        opp_pool.models[opp_id] = torch.compile(opp.eval(), mode="reduce-overhead")
    sampler      = SelfPlayPool(latest=opp_pool.latest_or_baseline(),
                                baseline=opp_pool.baseline,
                                older=opp_pool.older_promoted(),
                                weights=(0.50, 0.30, 0.20))

    assigned = seeds_for_worker(machine_id, worker_id, K)   # disjoint subset
    # Group seeds into N_env-sized batches, then run all batches.
    for seed_batch in chunked(assigned, N_env):
        run_batched_episodes(policy_vK, sampler, seed_batch, run_dir,
                              machine_id, worker_id, K)
```

```python
def run_batched_episodes(policy, sampler, seeds, run_dir, mid, wid, K):
    # N envs run in lockstep. Each env has its own opponent and seat.
    envs    = [env.reset(seed=s) for s in seeds]
    opps    = [sampler.sample() for _ in seeds]
    seats   = [random.choice([0, 1]) for _ in seeds]
    buffers = [EpisodeBuffer() for _ in seeds]
    done    = [False] * len(seeds)

    with torch.inference_mode():
        while not all(done):
            # Group env indices by whose turn it is.
            learner_idxs = [i for i, e in enumerate(envs)
                            if not done[i] and e.current_seat == seats[i]]
            opp_idxs     = [i for i, e in enumerate(envs)
                            if not done[i] and e.current_seat != seats[i]]

            # --- one batched forward for ALL learner envs this tick ---
            if learner_idxs:
                feats_batch = stack([featurize(envs[i], T=1) for i in learner_idxs])
                out = policy(feats_batch)
                for j, i in enumerate(learner_idxs):
                    mask              = legality_mask(envs[i], seats[i])
                    action_id, logp_a = sample_action(
                        out["pair_logits"][j], out["noop_logit"][j], mask)
                    frac, logp_f      = sample_frac(
                        out["frac_loc"][j], out["frac_log_std"], action_id)
                    env_act, ok       = project_to_env(envs[i], action_id, frac)
                    buffers[i].add(feats_batch[j], mask, action_id, frac,
                                    logp_a + logp_f,
                                    logp_action=logp_a, logp_frac=logp_f,
                                    value=out["value"][j], invalid=not ok)
                    step_env(i, env_act)

            # --- opponent forwards grouped by ckpt for batching reuse ---
            for opp_ckpt, group in groupby(opp_idxs, key=lambda i: id(opps[i])):
                group = list(group)
                feats_batch = stack([featurize(envs[i], T=1) for i in group])
                actions = opps[group[0]].act_batch(feats_batch)
                for j, i in enumerate(group):
                    step_env(i, actions[j])

            for i in range(len(seeds)):
                if done[i]: continue
                obs, r, d, info = envs[i].last_step()
                if buffers[i].last_was_learner:
                    buffers[i].set_reward(reward_shaping(info, invalid=buffers[i].last_invalid))
                if d:
                    buffers[i].set_terminal(envs[i].winner == seats[i])
                    atomic_write_shard(run_dir, K, mid, wid, buffers[i].finalize())
                    done[i] = True
```

Notes:

- `torch.inference_mode()` is mandatory — autograd allocation would
  otherwise dominate rollout cost.
- `policy = torch.compile(...)` compiles once; the per-worker fixed cost
  (~30–60 s) amortizes over ~18k forwards per worker per iter.
- The "group by opponent ckpt" trick lets opponent inference batch too:
  ~50% of opponent steps in a typical iter use the same pool member.

---

## Action sampling and env projection

```python
def legality_mask(obs, seat):
    # Source must be own & have surplus; target must exist & != source.
    own       = (obs.planets.owner == seat)
    surplus   = compute_surplus_vector(obs, seat)
    has_ships = (surplus >= obs.min_launch)
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

def sample_frac(frac_loc, frac_log_std, action_id):
    """Returns (frac_sample, logp_frac).

    `frac_sample` is the value stored in the shard. PPO recomputes
    logp_frac from `frac_sample` at update time, so the two must agree
    exactly — that means we must NOT apply the launch clamp here.
    The launch clamp lives in `project_to_env`.
    """
    if action_id == 0:
        return 0.0, 0.0                          # NOOP: no frac decision
    src, tgt = unflatten(action_id - 1, P, P)
    normal = Normal(frac_loc[src, tgt], exp(frac_log_std))
    z = normal.sample()
    # The [1e-4, 1-1e-4] clamp is numerical only (avoid logit(0/1) = ±inf).
    # It IS applied to the stored value, and logp uses the same value.
    frac_sample = clamp(sigmoid(z), 1e-4, 1 - 1e-4)
    # log p(sigmoid(z)) = log p(z) - log |d sigmoid(z) / dz|
    logp = normal.log_prob(logit(frac_sample)) - log(frac_sample) - log(1 - frac_sample)
    return frac_sample, logp

def project_to_env(obs, action_id, frac_sample):
    if action_id == 0:                                  # NOOP
        return [], ok=True
    src, tgt = unflatten(action_id - 1, P, P)
    source   = obs.planets[src]
    surplus  = compute_surplus(source, obs.enemy_fleets)
    # Launch clamp goes HERE, not in sample_frac. Keeps the stored
    # frac_sample matching what PPO will recompute logp on.
    frac_launch = clamp(frac_sample, 0.02, 1.00)
    ships    = clamp(round(frac_launch * source.ships), obs.min_launch, surplus)
    if ships < obs.min_launch:
        return [], ok=False
    launch   = physics_utils.plan_launch(obs, src, tgt, ships)
    if not launch.ok:
        return [], ok=False                              # invalid -> penalty
    return [[src, launch.angle, ships]], ok=True
```

Never resample on invalid: the sampled action stays in the buffer with its
original logprob and earns an `invalid_launch_penalty`. Resampling would
desync stored logprob from executed action and break the PPO ratio.

**Rollout-replay test (mandatory before Phase 0 promotion):** for a random
sample of shards, recompute `logp_frac` from the stored `frac_sample` using
the model state that produced it, and assert byte-equality with the stored
`logprob_frac`. Catches any silent drift between sampling and storage.

---

## Reward shaping

```python
def reward_shaping(info, invalid):
    potential = (
        info.my_ships + 5.0 * info.my_owned_planets
        - max(e.ships + 5.0 * e.owned_planets for e in info.enemies)
    )
    dense = clip((potential - info.prev_potential) / 200.0, -0.02, 0.02)
    r  = dense
    r -= 0.010 * invalid
    if info.done:
        r += +1.0 if info.winner == info.learner_seat else (
              0.0 if info.draw else -1.0)
    return r
```

---

## Advantages (GAE)

```python
def compute_advantages(rollouts, gamma=0.995, lam=0.95):
    all_adv = []
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
        all_adv.append(adv)
    mean, std = moments(concat(all_adv))
    for ep in rollouts:
        ep.advantages = (ep.advantages - mean) / (std + 1e-8)
    return rollouts
```

---

## PPO update — Phase 0 (A-only, local)

Heads-only update is too cheap to distribute. Sync overhead exceeds the
compute saved, so A runs the whole update locally.

```python
def ppo_update_local(policy, rollouts, pair_cache,
                     clip=0.10, target_kl=0.01, epochs=3, minibatch=1024,
                     lr_heads=1e-4, lr_trunk=None,             # trunk frozen
                     value_coef=0.5, ent_coef=0.01,
                     bc_coef=0.05, max_grad_norm=0.5):

    compute_advantages(rollouts)
    batch = flatten(rollouts)
    opt   = build_optimizer(policy, lr_heads, lr_trunk)

    for epoch in range(epochs):
        approx_kl_running = 0.0
        for mb in iter_minibatches(batch, minibatch, shuffle=True):
            loss, kl = ppo_minibatch_loss(policy, mb, pair_cache, bc_coef,
                                           value_coef, ent_coef, clip)
            opt.zero_grad()
            loss.backward()
            clip_grad_norm(policy.parameters(), max_grad_norm)
            opt.step()
            approx_kl_running += kl

        if approx_kl_running / num_minibatches > 1.5 * target_kl:
            break                                              # early stop
    return policy
```

## PPO update — Phase 1+ (distributed, file-mediated grad average)

Both machines compute gradients on their half of every minibatch. B
writes its grad to a file; A pulls, averages, applies the optimizer
step, publishes new weights; B pulls and reloads. Per minibatch.

```python
def ppo_update_distributed(policy, rollouts, pair_cache, K,
                            clip=0.10, target_kl=0.01,
                            epochs=3, local_minibatch=1024,
                            lr_heads=1e-4, lr_trunk=1e-5,
                            value_coef=0.5, ent_coef=0.01,
                            bc_coef=0.10,            # doubled — BC only on A
                            max_grad_norm=0.5):

    compute_advantages(rollouts)
    batch     = flatten(rollouts)
    paired_mbs = list(iter_paired_minibatches(
        batch, local_size=local_minibatch, shuffle=True))
    mb_A = [a for a, _b in paired_mbs]
    mb_B = [b for _a, b in paired_mbs]

    # Push B's paired minibatch sequence to B's local disk; tell B to start.
    write_mbs(f"sync/v{K}/mb_B.pt", mb_B)
    rsync_push_to_B(f"sync/v{K}/")
    touch_signal_on_B(f"sync/v{K}/mb_B_ready")       # peer driver polls this

    opt = build_optimizer(policy, lr_heads, lr_trunk)

    for E in range(epochs):
        kl_running = 0.0
        for M in range(len(mb_A)):
            # 1. A: local grad on its half (with BC anchor)
            local_loss, kl = ppo_minibatch_loss(policy, mb_A[M], pair_cache,
                                                 bc_coef, value_coef, ent_coef, clip)
            opt.zero_grad()
            local_loss.backward()
            grad_A = [p.grad.detach().clone() for p in policy.parameters()]

            # 2. wait for B's grad file (B is computing in parallel)
            grad_B_path = rsync_pull_from_B(f"sync/v{K}/E{E}/mb{M}_grad_B.pt",
                                             timeout_s=30.0,
                                             exclude="*.tmp")
            grad_B = torch.load(grad_B_path)

            # 3. average grads (no BC on B — see bc_coef doubling note)
            for p, gA, gB in zip(policy.parameters(), grad_A, grad_B):
                p.grad = 0.5 * (gA + gB)

            # 4. optim step on the averaged grad
            clip_grad_norm(policy.parameters(), max_grad_norm)
            opt.step()

            # 5. publish new weights to B
            wpath = f"sync/v{K}/E{E}/mb{M}_weights.pt"
            atomic_save(wpath, policy.state_dict())
            rsync_push_to_B(wpath)

            kl_running += kl

        if kl_running / len(mb_A) > 1.5 * target_kl:
            signal_B_early_stop(K, E)
            break

    cleanup_sync_dir(K)                              # not archived; deleted
    return policy
```

```python
def ppo_peer_train_loop(run_dir, policy, K):
    """Runs on Machine B. Mirrors ppo_update_distributed but writes grads
    and pulls weights instead of applying the optim step."""
    mb_B = read_mbs(f"sync/v{K}/mb_B.pt")
    epochs = 3
    for E in range(epochs):
        for M in range(len(mb_B)):
            # B has no pair cache → no BC term; pass bc_coef=0
            local_loss, _ = ppo_minibatch_loss(policy, mb_B[M], pair_cache=None,
                                                bc_coef=0.0,
                                                value_coef=0.5, ent_coef=0.01,
                                                clip=0.10)
            policy.zero_grad()
            local_loss.backward()
            atomic_save(f"sync/v{K}/E{E}/mb{M}_grad_B.pt",
                         [p.grad.detach() for p in policy.parameters()])

            # wait for A's new weights, load, continue
            wpath = wait_for_file(f"sync/v{K}/E{E}/mb{M}_weights.pt",
                                   exclude="*.tmp", timeout_s=60.0)
            policy.load_state_dict(torch.load(wpath))

            if early_stop_signal(K, E): return
```

```python
def ppo_minibatch_loss(policy, mb, pair_cache, bc_coef,
                        value_coef, ent_coef, clip):
    out = policy(mb.feats)
    new_logp  = action_logprob(out["pair_logits"], out["noop_logit"],
                                out["frac_loc"], out["frac_log_std"],
                                mb.action_mask, mb.action_id, mb.frac)
    ratio     = exp(new_logp - mb.old_logp)
    policy_loss = -mean(min(ratio * mb.adv,
                             clip(ratio, 1 - clip, 1 + clip) * mb.adv))
    value_loss  = mean((out["value"] - mb.returns) ** 2)
    entropy     = mean(action_entropy(out["pair_logits"], out["noop_logit"],
                                      out["frac_log_std"], mb.action_mask))

    bc_loss = 0.0
    if bc_coef > 0 and pair_cache is not None:
        bc_batch = pair_cache.sample(mb.size)
        bc_out = policy(bc_batch.feats)
        bc_loss = masked_bce(bc_out["pair_logits"],
                             bc_batch.expert_pair_labels,
                             bc_batch.pair_valid)

    loss = (policy_loss + value_coef * value_loss
            - ent_coef * entropy + bc_coef * bc_loss)
    approx_kl = mean(mb.old_logp - new_logp).item()
    return loss, approx_kl
```

Notes:

- BC anchor only lives on A's gradient (B has no pair cache locally).
  To keep its effective coefficient unchanged after averaging,
  `bc_coef` doubles from `0.05 → 0.10` in Phase 1+.
- Grad sync per minibatch costs ~60–400 ms depending on phase (see
  `docs/PPO_TWO_CPU_PROTOCOL.md` → "Sync overhead estimate"). Not free,
  but tolerable given the 2× compute gain.
- All grad/weight files use `.tmp → rename`. Rsync pulls must
  `--exclude '*.tmp'` to avoid loading half-written tensors.

---

## Self-play pool sampling

```python
class SelfPlayPool:
    # latest/older contain promoted frozen PPO checkpoints only.
    # They never contain the current learner policy_vK.
    # weights = (latest_promoted, baseline, older_uniform) = (0.50, 0.30, 0.20)
    def sample(self):
        u = random()
        if u < 0.50: return self.latest               # fallback baseline if none
        if u < 0.80: return self.baseline             # transformer_v2_baseline
        return random.choice(self.older) if self.older else self.baseline
```

Pool invariants:
- `latest` is always a *promoted frozen* checkpoint; the current policy never
  plays itself.
- `baseline` (`transformer_v2_baseline`) stays in the pool forever — it is the
  load-bearing floor/regression guard.
- `older` capped at `N_pool - 1 = 7` to prevent cyclic chasing.

---

## Promotion gate

```python
def promote_ok(metrics, prev):
    return (
        metrics.winrate["previous_promoted"] >= 0.50
        and metrics.paired_score["previous_promoted"] >= 0
        and metrics.winrate["baseline"] >= prev.winrate["baseline"] - 0.02
        and metrics.winrate["physical_v4"] >= prev.winrate["physical_v4"] - 0.05
        and metrics.invalid_launch_rate <= 1.1 * prev.invalid_launch_rate
        and not metrics.miss_matrix_has_new_severe_regression
    )
```

If `promote_ok` returns false the checkpoint still gets archived but is **not**
added to the self-play pool and is **not** considered for submission.

---

## Freeze schedule (parameter groups)

| Phase | Trainable | Frozen | Notes |
|---|---|---|---|
| 0 | noop / value / frac_log_std / PairHead output layers | L0-L2 world perception, current L3/L4 ActionLearner trunk, PairHead trunk+FiLM | "PPO plumbing" — prove the update doesn't destroy the BC policy |
| 1 | PlayerContextLearner / StrategyLearner / current L3-L4 ActionLearner / full PairHead | L0-L2 | Post-perception adaptation; PPO heads stay at `1e-4`, post-L2 trunk at `1e-5` |
| 2 | + CrossEntityAttention (L2), then L1 only with evidence | L0 specialists | Only if diagnostics show world perception is stale or missing facts |

---

## Archive step (after each iteration, A only)

Run after the promotion gate and log write. Moves cold artifacts to B's
`archive/` dir, freeing A's disk for the next iteration.

```python
def archive_after_iter(run_dir, run_id, K, N_pool=8):
    next_version = K + 1                          # just-trained ckpt
    recent = set(range(max(0, K - 2), next_version + 1))
    active_pool = read_pool_manifest(run_dir)     # parse promoted ckpt versions
    roll_floor = K - 2                            # rollouts older than K-1
    eval_floor = K - 11                           # eval JSONs older than 10

    # 1. cold checkpoints (skip v0, recent debug window, active pool)
    for v in checkpoint_versions(f"{run_dir}/checkpoints"):
        if v == 0 or v in recent or v in active_pool:
            continue
        rsync_move_to_B(f"{run_dir}/checkpoints/policy_v{v}.pt",
                         f"archive/runs/ppo/{run_id}/checkpoints/")

    # 2. cold rollouts (whole directories)
    for v in range(0, roll_floor + 1):
        rsync_move_to_B(f"{run_dir}/rollouts/v{v}/",
                         f"archive/runs/ppo/{run_id}/rollouts/v{v}/")

    # 3. cold eval JSONs
    for f in glob(f"{run_dir}/eval/v*.json"):
        v = parse_version(f)
        if v < eval_floor:
            rsync_move_to_B(f, f"archive/runs/ppo/{run_id}/eval/")

    # 4. sync/v{K}/ — delete locally (and on B); never archived
    rmtree(f"{run_dir}/sync/v{K}")
    ssh_B(f"rm -rf /Users/agent/dev/kaggle-orbit-wars/{run_dir}/sync/v{K}")
```

`rsync_move_to_B` = `rsync -av --remove-source-files SRC B:DEST` —
local file is deleted only after a successful transfer. See
`docs/PPO_TWO_CPU_PROTOCOL.md` → "Archive policy" for the bash
implementation.

**Invariant:** never archive a checkpoint that's still in the pool. Use the
pool manifest, not `K - N_pool` arithmetic; promotion is sparse, so active
promoted checkpoints are not guaranteed to be near the current version number.

---

## End-of-iteration log (one JSONL row per iter K)

```
iter, policy_version, n_rollout_eps,
winrate_previous_promoted, paired_score_previous_promoted,
winrate_baseline, winrate_pool_latest, winrate_pool_older, winrate_physical_v4,
winrate_by_seat[0|1], reward_mean,
invalid_launch_rate, emitted_launch_rate, miss_matrix_summary,
entropy, approx_kl, value_explained_var, ep_len_mean, ep_len_median,
# performance / distribution metrics
rollout_sec_A, rollout_sec_B, train_sec, eval_sec, archive_sec,
grad_sync_sec_total, grad_sync_avg_ms_per_mb,    # phase 1+ only
promoted (bool)
```

### Field definitions

- `winrate_X`: fraction of evaluated games where the learner won (draws count
  as 0). Computed over both seats; reported per-opponent.
- `paired_score_previous_promoted`: anti-variance score over the
  SEEDS_QUICK × {seat 0, seat 1} grid against `previous_promoted`.
  For each `(seed, both seats played)` pair, classify as `+1` if learner wins
  both seats, `-1` if loses both, `0` if split. Sum across pairs. Positive ⇒
  the learner is strictly stronger on this seed panel (not just lucky on one
  seat). Used in promotion gate condition 1.
- `winrate_pool_latest`: vs the most-recent promoted ckpt other than the one
  under test. Equals `winrate_previous_promoted` for promoted iters.
- `winrate_pool_older`: mean winrate vs a uniform sample from older pool
  members (excluding `previous_promoted`). Empty/baseline-fallback during
  pool ramp-up.
- `miss_matrix_summary`: dict keyed by miss category (`wrong_planet_neutral`,
  `wrong_planet_own`, `boundary`, `sun`, etc.) → rate. A "new severe
  regression" (promotion gate condition 5) is any category whose rate
  exceeds `1.5 × prev` AND `prev + 0.02`.
- `grad_sync_sec_total` / `grad_sync_avg_ms_per_mb`: only populated in
  Phase 1+; tracks the file-mediated sync overhead per minibatch.

The dashboard reads this file directly; no additional aggregation step needed.
