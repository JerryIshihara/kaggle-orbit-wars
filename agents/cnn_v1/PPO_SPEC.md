# PPO Spec Reference for cnn_v1

A walkthrough of every signal in the iter summary line, how it's computed, and the pseudo-code for the update. All code references point to `agents/cnn_v1/ppo.py` and `agents/cnn_v1/agent.py`.

---

## The action distribution

The CNN emits three heads per planet (up to 16 planets, padded). The policy is a product of three independent distributions:

```
launch_logit  ∈ ℝ           → Bernoulli(logits=launch_logit)         # launch yes/no
target_logits ∈ ℝ^2500      → Categorical(logits=target_logits)      # which 50×50 cell to aim at
ship_logit    ∈ ℝ           → Normal(sigmoid(ship_logit), std)       # what fraction of garrison to send
```

`std` is `exp(model.frac_log_std)` — a single learnable scalar. Initialized at `log(0.2)`, clamped after each Adam step to `[log(0.05), log(1.0)]`.

**Joint log-prob per step** (sum over planets, masked):

```
log_prob_step = Σ_planets [
    Bernoulli.log_prob(launch_action)              # always
  + launch_action · Categorical.log_prob(target)   # only if planet launched
  + launch_action · Normal.log_prob(frac)          # only if planet launched
]
```

Stored at rollout time as `old_log_prob`.

---

## The shaped reward

Per-step reward (potential-based shaping, Ng et al. 1999):

```
Φ(s)  = (my_planets − enemy_planets) / 16
      + 0.5 · (my_prod − enemy_prod) / 50

r_t   = γ · Φ(s_{t+1}) − Φ(s_t)            for t < T-1
r_T-1 = final_reward − Φ(s_{T-1})           # final = ±1 from env
```

Telescoping sum across the episode = `final_reward + γ^T·Φ(s_T) − Φ(s_0)` ≈ `±1`, so shaping cannot dominate the win signal.

---

## GAE (advantages and returns)

```
δ_t   = r_t + γ · V(s_{t+1}) − V(s_t)
A_t   = δ_t + (γ·λ) · δ_{t+1} + (γ·λ)^2 · δ_{t+2} + ...
R_t   = A_t + V(s_t)                        # return = critic target
```

Constants: `γ=0.99`, `λ=0.95`. Computed in reverse with the recurrence:

```
adv[T-1] = δ[T-1]
adv[t]   = δ[t] + γ·λ·adv[t+1]
```

Inside `_ppo_update`, **advantages are normalized** per batch:

```
adv = (adv − mean(adv)) / (std(adv) + 1e-8)
```

Standard variance-reduction trick; doesn't change the policy gradient sign but stabilizes scale across iters.

---

## The PPO clipped surrogate

For each minibatch:

```
new_log_prob = Σ over planets of (Bernoulli + Categorical·la + Normal·la) · mask
log_ratio    = clamp(new_log_prob − old_log_prob, −log(ratio_max), log(ratio_max))
ratio        = exp(log_ratio)                          # ≈ 1 right after rollout, drifts during epochs
```

`ratio_max = 3.0` — a hard guardrail to stop the per-step product of 16 planets' log-probs from blowing up exp() into 1e10 territory on any single weird sample. The PPO `clip` (below) does the principled bounding; this just prevents numerical explosion before the clip kicks in.

```
unclipped = ratio · adv
clipped   = clamp(ratio, 1−clip, 1+clip) · adv         # clip = 0.2
policy_loss = − mean( min(unclipped, clipped) )
```

The `min` is the conservative side: take the tighter (smaller) of the two surrogate objectives. This caps how much the policy can shift per update step.

`clip_frac` in the printout = fraction of samples where `|ratio − 1| > clip` (i.e., the clip actually bit). Typical healthy range: 10–30%. Very high (>80%) means the policy is moving too fast per epoch.

---

## Value loss + entropy bonus

```
value_loss = MSE(value_pred, ret)                       # ret = adv + V_old, GAE return target
entropy    = mean over batch of (Bernoulli.entropy() + Categorical.entropy()) · mask

loss = policy_loss + 0.5·value_loss − 0.01·entropy
```

`value_coef = 0.5` (critic learns half as fast as actor), `entropy_coef = 0.01` (mild push for exploration; counters policy collapse).

---

## KL early stopping

Schulman's approximate KL between old and new policy:

```
approx_kl = mean( (ratio − 1) − log_ratio )            # ≥ 0, second-order accurate
```

Compared against `kl_stop_now`, which **decays linearly** from `kl_stop_start=0.4` to `kl_stop_end=0.05` over the full `iterations`:

```
kl_stop_now = 0.4 + (0.05 − 0.4) · (it / (iterations − 1))
```

Loose early (let the policy move while it's still random), tight late (precision when converging). If `approx_kl > kl_stop_now`, break out of the epoch loop — `early_stopped=True`, marked with `*` in the print.

---

## Gradient + parameter safeguards

```
loss.backward()
if any non-finite grad:        skip this minibatch  (counted as skipped_mb)
if loss not finite:            skip this minibatch
clip_grad_norm_(params, max_norm=0.1)                 # tight cap, was 0.5
opt.step()
clamp_(model.frac_log_std, log(0.05), log(1.0))       # std stays in [0.05, 1.0]
```

After the full update:

```
if any param non-finite anywhere:                     # NaN-recovery
    model.load_state_dict(best_state)                 # revert to last best-by-eval
```

---

## Pseudo-code: one full iteration

```python
for it in range(iterations):
    rewards_hist, ep_steps, buffers = [], [], []

    # ---- ROLLOUT ----
    for ep in range(episodes_per_iter):                # 32
        opp = pick_opponent(self_play_ratio, snapshots, ext_opponents)
        slot = ep % 2
        env = make("orbit_wars", seed=...)
        for step in range(500):
            obs = env.observe(slot)
            ch, sc, my_planets = featurize(obs)
            launch, target, ship = forward(ch, sc, coords)
            launch_a = Bernoulli(logits=launch).sample()
            target_a = Categorical(logits=target).sample()
            frac_a   = Normal(sigmoid(ship), exp(frac_log_std)).sample().clamp(0.01, 1)
            log_prob = sum_planet_log_probs(...)
            value    = model.value(...)
            phi      = Φ(obs, slot)
            store(traj, ch, sc, coords, launch_a, target_a, frac_a, log_prob, value, phi)
            env.step(...)
        final_reward = env.reward(slot)
        traj_packed = pack(traj, final_reward, γ=0.99, λ=0.95, use_shaping=True)
        # → fills .adv, .ret via GAE on shaped rewards
        buffers.append(traj_packed)

    if not buffers:                                    # all rollouts empty
        revert_to_best_state();  continue

    batch = concat(buffers)                            # ~8–12k step samples

    # ---- UPDATE ----
    kl_stop_now = linear_schedule(it, iterations, 0.4, 0.05)
    for epoch in range(2):
        for minibatch in shuffle(batch, size=32):
            # recompute policy on current weights
            new_log_prob = recompute_joint_log_prob(...)
            log_ratio    = clamp(new_log_prob − old_log_prob, −log(3), log(3))
            ratio        = exp(log_ratio)
            adv          = normalize(adv)
            policy_loss  = − mean(min(ratio·adv, clip(ratio, 0.8, 1.2)·adv))
            value_loss   = MSE(value, ret)
            entropy      = mean(Bernoulli.entropy + Categorical.entropy)
            loss = policy_loss + 0.5·value_loss − 0.01·entropy

            opt.zero_grad();  loss.backward()
            if not all_finite(grads):  skip
            clip_grad_norm_(params, 0.1)
            opt.step()
            clamp_(frac_log_std, log(0.05), log(1.0))

            approx_kl = mean((ratio − 1) − log_ratio)
            if approx_kl > kl_stop_now:  early_stop = True; break

    if any_param_non_finite(model):
        load_state_dict(best_state)                    # NaN-recovery

    if (it + 1) % snapshot_every == 0:
        snapshots.append(deepcopy(model).eval())

    if (it + 1) % eval_every == 0:
        eval_wr = evaluate vs physical_v2 × 20 games
        if eval_wr > best_winrate:
            best_winrate, best_state = eval_wr, deepcopy(state_dict)

    save(latest.pt)                                    # every iter, resume-friendly
    log_jsonl(...)
```

---

## Reading the iter summary

```
iter   1/200  W/L/D=15/12/5  r=+0.094  steps=437
              pi=+0.81  v=0.09  ent=65  kl=0.57* clip=0.18  std=0.20
              opp=22s/10p/0x  snap=2  n=11200  ep_t=8.4/7.9s
              t=2.4m  elapsed=2.4m  ETA=8.0h  mem=0.3G  eval=0.05(best=0.05)
```

| Field | Meaning |
|---|---|
| `W/L/D` | Wins/losses/draws over the iter's 32 episodes (raw terminal r) |
| `r` | Mean of those terminal rewards |
| `steps` | Mean episode length |
| `pi` | Last minibatch's policy_loss (clipped surrogate, neg = good gradient) |
| `v` | Last minibatch's value MSE |
| `ent` | Last minibatch's joint Bernoulli+Categorical entropy (mean over batch) |
| `kl` | Mean approx_kl over all minibatches; `*` = early-stop triggered |
| `clip` | Fraction of samples with `|ratio−1| > 0.2` |
| `std` | Current `exp(frac_log_std)` for the ship-fraction Gaussian |
| `opp` | Opponent breakdown: `Ns` = self, `Np` = snapshot, `Nx` = external |
| `snap` | Snapshot ring buffer size (max 8) |
| `n` | Total step samples in the batch |
| `ep_t` | Per-epoch wall-times for the PPO update (slash-separated, may show only 1 if early-stopped) |
| `t` / `elapsed` / `ETA` | Iter time / cumulative training time / linear-extrapolation ETA |
| `mem` | Peak GPU memory ever allocated (`max_memory_allocated`) |
| `eval=X(best=Y)` | Win rate vs `physical_v2` × 20 games (only on eval iters); `best` tracks best-so-far for the checkpoint gate |
