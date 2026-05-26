# PPO Phase 0 training runs — results log

Append-only record of every completed PPO training run on the
`transformer_v2` stack. Lives in repo so we can diff configs and
trajectories across runs. The actual checkpoints and per-iter train logs
live under `data/runs/ppo/<run_id>/` and are git-ignored.

For the design these runs implement see `docs/PPO_TWO_CPU_PROTOCOL.md`
and `PPO_PSEUDOCODE.md`.

---

## Run 1 — vs `random_v1` (sanity)

```
run_id:       ppo_phase0_shallow_20260525-180554
ckpt:         top4_pair2head_film_d256_h8_lr5e-05_b256_30ep_20260521-003000 (shallow heads)
opponent:     random_v1
iters:        10
episodes:     10/iter
device:       cpu
config:       minibatch=256, epochs=3, lr_heads=1e-4, clip=0.10,
              target_kl=0.01, sigma=0.35, bc_coef=0 (no BC anchor)
soft_cap:     OFF (pre-fix run)
total wall:   101 min
```

Mean winrate: **1.00** (10/10 every iter — sanity check; the shallow
PairHead bootstrap is already competent vs uniform random launches).

Validates: rollout loop, GAE, PPO update, value head learning. Says
nothing about whether PPO can actually improve policy.

---

## Run 2 — self-play, 20 iters

```
run_id:       ppo_selfplay_shallow_20260525-220612 (iters 0-9)
              ppo_selfplay_shallow_cont_20260525-232033 (iters 10-19, resumed)
ckpt:         same shallow bootstrap
opponent:     frozen snapshot of learner at start of each iter
iters:        10 + 10 (resume via --resume-ppo-ckpt policy_v10.pt)
episodes:     10/iter
device:       cpu
config:       minibatch=256, epochs=3, bc_coef=0 (no BC anchor)
soft_cap:     OFF (pre-fix run)
total wall:   157 min (74 + 83)
```

| k  | win    | kl     | v_loss | entropy | clip_f | note |
|---:|------:|-------:|-------:|--------:|-------:|------|
|  0 | 0.60 | 0.0041 |  2.187 |   10.44 |  0.216 |      |
|  1 | 0.40 | 0.0046 |  1.509 |   11.43 |  0.167 |      |
|  2 | 0.60 | 0.0068 |  0.991 |   11.38 |  0.283 |      |
|  3 | 0.20 | 0.0205 |  2.027 |   11.47 |  0.412 | EARLY |
|  4 | 0.50 | 0.0036 |  0.912 |   11.17 |  0.248 |      |
|  5 | 0.50 | 0.0118 |  1.500 |   11.37 |  0.291 |      |
|  6 | 0.80 | 0.0159 |  1.119 |   11.26 |  0.337 | EARLY |
|  7 | 0.60 | 0.0061 |  1.762 |   11.49 |  0.216 |      |
|  8 | 0.50 | 0.0126 |  2.496 |   11.78 |  0.323 |      |
|  9 | 0.30 | 0.0046 |  3.956 |    9.31 |  0.222 |      |
| 10 | 0.30 | 0.0020 |  2.039 |   10.79 |  0.198 | resume |
| 11 | 0.60 | 0.0050 |  1.810 |   11.75 |  0.232 |      |
| 12 | 0.70 | 0.0065 |  1.723 |   10.17 |  0.238 |      |
| 13 | 0.30 | 0.0074 |  1.835 |   11.67 |  0.309 |      |
| 14 | 0.60 | 0.0068 |  1.767 |   11.33 |  0.228 |      |
| 15 | 0.40 | 0.0049 |  1.555 |   11.11 |  0.191 |      |
| 16 | 0.30 | 0.0061 |  2.114 |   11.26 |  0.214 |      |
| 17 | 0.60 | 0.0124 |  1.740 |   11.92 |  0.299 |      |
| 18 | 0.60 | 0.0063 |  1.380 |   11.29 |  0.256 |      |
| 19 | 0.50 | 0.0083 |  1.860 |    9.87 |  0.305 |      |

**Mean winrate 0.495** — statistically indistinguishable from 50/50
(10-ep std error ~15 pp). Mean KL 0.0078, mean value_loss 1.81, 2 early
stops (out of 20). Resume across runs landed clean (`missing=0
unexpected=0`).

**Diagnosis:** with only 257 trainable params per actor head and no BC
anchor, the policy cannot move meaningfully against its own snapshot.
The value head learned (132 k params) but the actor surface is too thin
to escape its bootstrap.

Aggregate invalid-launch rate: ~8.4 per learner step — the Bernoulli
selects more targets than survive the `min_launch` allocation gate.

---

## Run 3 — fixes v3 (soft cap, BC anchor, 32 ep/iter)

```
run_id:       ppo_fixes_v3_20260526-054701
ckpt:         same shallow bootstrap
opponent:     frozen snapshot of learner per iter (self-play)
iters:        10
episodes:     32/iter  (was 10 — 3.2x more rollout data per iter)
device:       cpu
config:       minibatch=256, epochs=3
soft_cap:     target_cap_lambda=0.001, target_cap_k_max=4
              (penalty per step = 0.001 * max(0, k-4)^2; at k=10 → 0.036)
bc_anchor:    ON, bc_coef=0.05, pair cache = bowwowforeach_T6 (1.4 GB,
              6,681 acted snapshots)
total wall:   10.7 h
```

Earlier v1 / v2 attempts crashed:
- v1: aborted by user choice to kill the supervised retrain that was
  competing for unified memory (system thrashed; PPO iter 0 took 46 min).
- v2: launched with `target_cap_lambda=0.005`; value_loss blew up to 51
  on iter 0 because the soft cap penalty at typical k≈10 dwarfed the
  ±1 terminal reward. Killed at iter 1 and relaunched at λ=0.001.
- A separate fix (PR #14, `_PPOWithL0` comet_features shape) was needed
  before the BC anchor could load T=6 history-stacked minibatches.

| k  | win  |    kl  | v_loss | entropy | clip  |   emit |    inv | i/s | k_p95 | >cap  |
|---:|-----:|-------:|-------:|--------:|------:|-------:|-------:|----:|------:|------:|
|  0 | 0.44 | 0.0049 |  4.364 |   11.15 | 0.226 | 32,243 | 91,528 | 7.7 |    22 | 9,333 |
|  1 | 0.28 | 0.0050 |  6.592 |   11.57 | 0.248 | 25,258 |123,071 | 9.5 |    24 |10,433 |
|  2 | 0.41 | 0.0090 |  3.859 |   10.55 | 0.286 | 30,478 | 89,260 | 7.3 |    20 | 9,541 |
|  3 | 0.41 | 0.0073 |  3.586 |   10.51 | 0.294 | 23,885 | 84,515 | 7.8 |    20 | 8,464 |
|  4 | 0.50 | 0.0064 |  5.797 |   11.32 | 0.274 | 27,495 | 97,906 | 8.3 |    22 | 9,566 |
|  5 | 0.59 | 0.0030 |  4.120 |   11.01 | 0.261 | 27,334 | 82,040 | 7.5 |    20 | 8,636 |
|  6 | 0.53 | 0.0040 |  4.338 |   10.13 | 0.238 | 24,756 | 73,079 | 7.1 |    20 | 7,823 |
|  7 | 0.81 | 0.0078 |  6.061 |   11.53 | 0.317 | 37,337 | 85,984 | 7.3 |    22 | 9,183 |
|  8 | 0.50 | 0.0064 |  5.050 |   11.10 | 0.278 | 24,856 | 95,923 | 8.2 |    21 | 9,345 |
|  9 | 0.72 | 0.0036 |  6.004 |   11.74 | 0.332 | 34,744 | 87,391 | 7.4 |    22 | 9,310 |

**Aggregates:**
- Mean winrate 0.519 (still ~50/50 in aggregate, but with **two real
  spikes**: iter 7 = 0.81, iter 9 = 0.72)
- Mean KL 0.0057
- **Zero early stops** (vs 2 in the no-fix 20-iter run)
- Mean invalid/step 7.83 (vs 8.4 in no-fix — modest reduction)
- Mean target_count_p95 21.3 (Bernoulli still selects ~20 of every legal
  target row)

**Invalid-reason histogram (run total): `min_launch: 910,697` (100 %)**
— every single invalid is "allocated ships < per-phase min_launch
floor". The Bernoulli's high selection rate slices the source budget
into too-thin pieces; the soft cap at λ=0.001 was too gentle to reshape
this.

**Verdict:**
- BC anchor + bigger episode count produced real spike iters that the
  no-fix run never hit. The policy CAN move; it just bounces back.
- Soft cap λ=0.001 is too gentle. λ=0.005 was too aggressive
  (value_loss blew up). The sweet spot is somewhere ~0.002, or move
  from quadratic to linear, or push the cap into the action contract
  (per-source top-k truncation in Phase 1+).

---

## Open issues, prioritized

1. **Actor heads are too shallow.** 257 params per head is the binding
   constraint on whether PPO can move the policy. The deep-heads
   supervised retrain (`head_n_layers=3`, `conditioner_n_layers=2`) is
   the unlock — when that ckpt lands, every result here should be
   redone on it.

2. **Soft cap λ tuning.** Bisect between 0.001 (too gentle) and 0.005
   (too aggressive). Or switch to linear `λ · max(0, k - k_max)`.

3. **Per-source top-k truncation at the action contract** ("Phase 1+
   multi-source upgrade" in the protocol) is the cleaner long-term fix
   for high invalid rate — caps Bernoulli output rather than penalizing
   it after.

4. **BC cache size.** Current run used the 1.4 GB bowwowforeach cache
   (single player). The bigger 33 GB cache has the top-4 players but
   needs concurrent-memory headroom we don't have when retrain also
   runs on MPS.

5. **Distributed Phase 1+ training** still not implemented; only
   single-machine Phase 0 has been exercised. The protocol's
   file-mediated gradient averaging across A and B is unblocked by all
   of the above.

---

## Run-config dimension cheatsheet

The flags below map to the design parameters captured in
`docs/PPO_TWO_CPU_PROTOCOL.md` and `PPO_PSEUDOCODE.md`. Anything not
listed is intentionally hard-coded at the design's default.

```
--episodes-per-iter      rollout budget per PPO iteration (default 32)
--minibatch-size         PPO minibatch size (default 256)
--epochs                 inner PPO epochs per iter (default 3)
--clip                   PPO clip ratio (default 0.10)
--target-kl              KL early-stop threshold (default 0.01)
--ent-coef               entropy bonus (default 0.01)
--value-coef             value loss weight (default 0.5)
--sigma                  fixed logit-normal stddev for frac (default 0.35)
--noop-logit-bias        fixed bias on NOOP slot (default 0.0)
--target-cap-k-max       soft cap threshold (default 4)
--target-cap-lambda      soft cap quadratic coef (default 0.005)
--bc-coef                BC anchor coefficient (default 0.05; off without --pair-cache-path)
--bc-target-weight       relative weight of bc_target vs bc_source (default 1.0)
--pair-cache-path        PATH to _pair_cache .pt (enables BC anchor)
--opponent               registered agent id; UNSET = self-play snapshot per iter
--resume-ppo-ckpt        PATH to a prior policy_v*.pt to continue training
```
