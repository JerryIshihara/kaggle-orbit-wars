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
pool           = ring buffer of frozen PPO ckpts, cap N_pool = 8
A, B           = Machine A (coordinator) and Machine B (peer)
```

Both machines do rollouts in parallel. Both contribute to the PPO update
once Phase 1 unfreezes the trunk; Phase 0 (heads only) is A-only because
the grad sync overhead exceeds the compute saved.

---

## Top-level driver (Machine A — coordinator)

```python
def train_ppo(run_dir, init_ckpt, max_iters, phase):
    policy = load(init_ckpt)                      # supervised PairHead
    save_checkpoint(run_dir, policy, version=0)
    rsync_push_to_B(run_dir, "checkpoints/policy_v0.pt")
    pool = [baseline]                             # self-play opponent pool
    promoted = 0

    for K in range(max_iters):
        publish_checkpoint(run_dir, K)            # workers see policy_vK
        rsync_push_to_B(run_dir, f"checkpoints/policy_v{K}.pt")

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
                               opponents=[baseline] + pool[-3:] + [physical_v4])
        metrics_B = rsync_pull_eval_from_B(run_dir, version=K + 1)
        metrics   = merge_eval(metrics_A, metrics_B)

        # ---- 4. promotion + pool maintenance ----------------------------
        if promote_ok(metrics, prev=metrics_of(promoted)):
            promoted = K + 1
            pool.append(freeze(policy_next))
            if len(pool) > N_pool: pool.pop(1)    # keep baseline at index 0

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

        # 1. spawn B's rollout workers (independent of A's)
        spawn_local_workers(K, pool=current_pool(run_dir), share=0.50)
        wait_for_rollout_complete(K)

        # 2. if Phase 1+, wait for mb_B sequence A pushes, then peer-train.
        if exists(run_dir, f"sync/v{K}/mb_B_ready"):
            ppo_peer_train_loop(run_dir, policy_vK, K)

        # 3. quick-eval the OTHER half of the seeds
        eval_quick_peer(policy_vK, seeds=range(16, 32),
                        write_to=f"{run_dir}/eval/v{K}_B.json")
```

---

---

## Rollout worker (one process per physical core minus one)

The worker batches `N_env` envs per forward pass — the single biggest CPU
win over the naive one-env-per-step design.

```python
def rollout_worker(run_dir, machine_id, worker_id, N_env=4):
    K, policy_vK = wait_for_new_checkpoint(run_dir)
    policy_vK    = torch.compile(policy_vK.eval(), mode="reduce-overhead")
    opp_pool     = load_opponent_pool_once(run_dir)   # loaded once, not per-ep
    for opp_ckpt in opp_pool.values():
        opp_pool[opp_ckpt] = torch.compile(opp_pool[opp_ckpt].eval(),
                                            mode="reduce-overhead")
    sampler      = SelfPlayPool(latest=policy_vK, baseline=baseline,
                                older=opp_pool.older, weights=(0.50,0.30,0.20))

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
                pair_logits, noop_logit, frac_dist, value = policy(feats_batch)
                for j, i in enumerate(learner_idxs):
                    mask              = legality_mask(envs[i], seats[i])
                    action_id, logp_a = sample_action(pair_logits[j], noop_logit[j], mask)
                    frac, logp_f      = sample_frac(frac_dist[j], action_id)
                    env_act, ok       = project_to_env(envs[i], action_id, frac)
                    buffers[i].add(feats_batch[j], action_id, frac,
                                    logp_a + logp_f, value[j], invalid=not ok)
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
                            epochs=3, minibatch=1024,
                            lr_heads=1e-4, lr_trunk=1e-5,
                            value_coef=0.5, ent_coef=0.01,
                            bc_coef=0.10,            # doubled — BC only on A
                            max_grad_norm=0.5):

    compute_advantages(rollouts)
    batch     = flatten(rollouts)
    mb_seq    = list(iter_minibatches(batch, minibatch, shuffle=True))
    mb_A, mb_B = split_half(mb_seq)                  # disjoint halves

    # Push B's minibatch sequence to B's local disk; tell B to start.
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
    pair_logits, noop_logit, frac_dist, value = policy(mb.feats)
    new_logp  = action_logprob(pair_logits, noop_logit, frac_dist,
                                mb.action_id, mb.frac)
    ratio     = exp(new_logp - mb.old_logp)
    policy_loss = -mean(min(ratio * mb.adv,
                             clip(ratio, 1 - clip, 1 + clip) * mb.adv))
    value_loss  = mean((value - mb.returns) ** 2)
    entropy     = mean(action_entropy(pair_logits, noop_logit, frac_dist))

    bc_loss = 0.0
    if bc_coef > 0 and pair_cache is not None:
        bc_batch = pair_cache.sample(mb.size)
        bc_logits, *_ = policy(bc_batch.feats)
        bc_loss = bce(bc_logits, bc_batch.expert_pair_labels)

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

## Archive step (after each iteration, A only)

Run after the promotion gate and log write. Moves cold artifacts to B's
`archive/` dir, freeing A's disk for the next iteration.

```python
def archive_after_iter(run_dir, run_id, K, N_pool=8):
    pool_floor = K - N_pool                       # ckpts older than pool
    roll_floor = K - 2                            # rollouts older than K-1
    eval_floor = K - 11                           # eval JSONs older than 10

    # 1. cold checkpoints (skip v0 — keep init hot forever)
    for v in range(1, pool_floor + 1):
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

**Invariant:** never archive a checkpoint that's still in the pool. The
`pool_floor = K - N_pool` arithmetic enforces this; if you change
`N_pool` mid-run, update both places or B's rollout workers will fail
to load the opponent ckpt they sampled.

---

## End-of-iteration log (one JSONL row per iter K)

```
iter, policy_version, n_rollout_eps,
winrate_baseline, winrate_pool_latest, winrate_pool_older, winrate_physical_v4,
winrate_by_seat[0|1], reward_mean,
invalid_launch_rate, emitted_launch_rate, miss_matrix_summary,
entropy, approx_kl, value_explained_var, ep_len_mean, ep_len_median,
# performance / distribution metrics
rollout_sec_A, rollout_sec_B, train_sec, eval_sec, archive_sec,
grad_sync_sec_total, grad_sync_avg_ms_per_mb,    # phase 1+ only
promoted (bool)
```

The dashboard reads this file directly; no additional aggregation step needed.
