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

---

# 2026-06 update — multi-target contract, deep heads, agent-B-vs-A

Everything above (Runs 1–3) used the **shallow** PairHead (257 params/head) and
the single-shot action rule. The stack has since moved to **deep heads
(`head_num_layers=3`) + the multi-target action contract**, and the active goal
changed from "PPO can move the policy at all" to **"train an agent B that beats a
fixed agent A"** (A = a strong multi-target checkpoint that beats `physical_v4`
~93.8% in deploy). This section captures the current design, config, and the
agent-B trials run on the RunPod 3090 (`infserver` rollout).

## Current model design (`transformer_v2`)

```
d_model = 256,  T = 10 history (HISTORY_OFFSETS strided: t, t-5, … t-45)
L0  frozen specialists:  planet_enc (18 scalars→256), fleet_enc, comet_enc (18+105→256)
L1  per-entity tokens  →  L2 CrossEntityAttention (temporal, runs over the T=10 stack)
L3  dual_role  →  L4 joint_role  →  PlayerConsolidator   (all load-bearing; see ablation)
PairHead:  trunk + FiLM (conditioned on player/context) over source⊗target⊗context
           ├─ pair_head        → pair_logits[s,t]   (SELECT logits)
           └─ pair_frac_head   → frac_loc[s,t]      (ALLOCATION logits)
Critic:    reward-decomp (pretrained win head + trainable residual value_decoder),
           fed the full T=10 L1 stack (in-distribution with its pretraining)
max_planets = 64,  max_fleets = 512  (fleet p99≈624; <512 truncates ~14% of steps)
```

**Action contract — `bernoulli_select_multinomial_alloc_v1` (multi-target):**
per OWNED source row `s` (legal = owned + surplus ≥ `min_launch`):
1. **Select (Bernoulli):** for each *legal* target column, fire ~ `Bernoulli(σ(pair_logits[s,t]/τ − select_logit_bias))`.
2. **Allocate (Multinomial):** distribute the source's *full* ship count `N` over `[fired targets … , self]` with
   `logits = [frac_loc[s, fired] , pair_logits[s,s]]/τ`. Fired-target counts = launch sizes; the self/HOLD count stays home.
   - NB: the **HOLD logit reuses the select head's diagonal** `pair_logits[s,s]`, which was **never supervised in pretraining** → "launch vs hold" is shaped entirely by PPO from a near-random init (a likely early-instability contributor).

**Deploy (shipped) rule — `inference_mode="threshold"`, `logit_threshold=2.0`:**
every legal cell with `pair_logits>2.0` fires, sized `sigmoid(frac)·ships`, each launch
**pre-validated through `plan_launch`** (drops wrong-planet/sun/boundary) + surplus-budgeted.
This is the eval that matters; PPO trains the SAMPLED distribution above.

## Current PPO config (agent-B vs frozen A)

```
rollout:    --rollout infserver  (two-policy: learner-seat→B, other seats→frozen A
            via per-request is_learner flag; ~8x over batched). procs=32, T=10,
            games=64 (mix 2P/4P), spool off /dev/shm → /root/ow_spool (OOM guard),
            per-iter janitor prunes old rollouts/ (MooseFS quota guard).
critic:     --reward-decomp (PBRS Φ-shaping), value-gamma 0.997
reward:     win-weight 0.5  +  signal-weights [0.1, 0.2, 0.1, 0, 0.35]
            (ship_adv, prod_adv, planet_adv, safety, fleet_spd from compute_p1_signals)
contract:   --opponent-ckpt <A>  (frozen agent A);  sigma 0.35;  select-logit-bias 1.0
update:     unfreeze L2, epochs 3, minibatch 64, value-coef 0.5
logging:    per-iter PBRS signals bucketed in 25-step windows (apply_phi_shaping log_fn)
infra:      RunPod 3090, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404,
            125 GB cgroup; T=10 update peaks ~107 GB (clear dead-run rollout cache first).
```

`--rollout shm` (shared-memory request plane, infserver_shm_rollout.py) is wired
+ `--shm-forwarders`, but **stalls through `train_local_trial` on the 3090**
(forwarder↔worker handshake → games=0/64; standalone test passes on L40). **Not
production-ready — keep `infserver`.** The 1.6× shm win is marginal vs the env.step floor.

## Trials — agent B vs frozen A (2026-06-06, RunPod 3090)

All warm-started from a multi-target B checkpoint, opponent = frozen A (`policy_v5`).
Sampled winrate ≈ 30–40 % throughout (= A's own training range; the deploy number is
what matters). The lever under test was **invalid-launch reduction** (later shown moot).

| trial | clip | lr-heads | target-kl | ent-coef | inval-coef | KL/update | inval_frac | outcome |
|---|---|---|---|---|---|---|---|---|
| **sig** (baseline) | 0.20 | 5e-5 | 0.05 | −0.05 | 0 | ~0.002 | 0.89 | **frozen** (KL 25× under target) |
| **p1inval** | 0.20 | 1e-4 | 0.10 | −0.025 | 0.005 | 0.006→0.003 | flat ~0.88 | moves a little then re-freezes; invalid flat |
| **accel** | 0.30 | **2e-4** | 0.10 | −0.025 | 0.01 | **0.084** | 0.895→**0.909** | **DETONATED** — invalid ↑, emit collapse, win 30→23% |
| **controlled** | 0.30 | 1.25e-4 | 0.03 | −0.025 | 0.005 | ~0.005 | flat ~0.88 | back to **stuck** regime (no detonation, no progress) |

**There is a cliff with no working middle:** lr-heads ≤1.25e-4 → KL ~0.005, invalid
flat (policy doesn't move); lr-heads 2e-4 → KL 0.084, policy runs away into a *worse*
region (detonation). 1.6× lr → 16× KL.

### Key findings (these reframe the whole effort)

1. **The ~88 % invalid-launch rate is BENIGN — it is *not* illegal moves.** The
   sampler only ever fires *legal* pairs; the "invalid" count is essentially all
   `min_launch` (the Multinomial spreads `N` across many fired targets + HOLD, so most
   fired cells fall below the per-phase `min_launch` floor and are dropped). **Agent A —
   the 93.8 % champion — trained at `inval_frac ≈ 0.857` with NO penalty.** And the
   **deploy path emits env-valid launches** (it budgets surplus into a few targets +
   `plan_launch`-validates), so sampled invalid_frac has **zero** effect on the shipped
   policy. ⇒ Phase-1 "decrease invalid first" was a **red herring**; abandoned.

2. **Deploy eval (the number that matters):** B (`policy_v4`, threshold) vs A
   (`policy_v5`, threshold) = **29/64 = 45.3 %** (seat0 40.6 %, seat1 50.0 %). B ≈ A —
   **genuine symmetric parity.** Separately, B beats `physical_v4` **4/4** in deploy, so
   B is a strong agent; it is specifically *A* it cannot pass.

3. **20/20 may be structurally unattainable.** Deploy is deterministic given a seed +
   there's strong seat asymmetry (2p seat-0 ≈ 54 %). Even *champion A* only beats the
   far-weaker `physical_v4` 93.8 % (not 20/20) — i.e. seat/seed variance caps winrate
   below 100 %. Beating the *stronger* A on 20/20 seat-balanced seeds is almost certainly
   above that ceiling. **Target definition needs pinning before more training** (literal
   20/20 vs B-favorable-seat vs "high-winrate dominance").

### Recommended next direction (not yet run — awaiting decision)

Warm B **from A itself** (`policy_v5`) and exploit-train vs frozen A:
win-weight 1.0 (optimize winning, not signals), **ent-coef +0.01** (real exploration —
phase 2), no invalid penalty, controlled lr 1.25e-4 / target-kl 0.03 / clip 0.30. Since A
is frozen + deterministic, this is an *exploitation* problem; the under-supervised HOLD
diagonal (above) is the least-constrained spot where B can diverge from A.

## 2026-06-11 — contract v2: HOLD logit moved to the frac head's diagonal

`bernoulli_select_multinomial_alloc_v1` → `_v2`: the allocation softmax's
self/HOLD category now reads `frac_loc[s, s]` instead of `pair_logits[s, s]`.
One head owns the whole alloc softmax (one output scale), and select
(pair_head) / alloc (pair_frac_head) gradients decouple at the head level.
v1's slot only existed to inherit the single-target NOOP diagonal; current
warm-starts are multi-target pretrains with unsupervised diagonals, so nothing
is lost. No new params; prior ckpts load 0/0. `shards.POLICY_ACTION_CONTRACT`
bumped — v1 rollout shards are rejected by the version gate (intended).
Changed: sampler.py, loss.py (logprob recompute + entropy), alloc_labels.py
(pretrain CE), notebook. Parity suite re-passed (ratio=1, float64 exact).
`pair_logits`' diagonal is now fully dead. NOTE: A's PPO ckpts carry their
learned hold behavior in the OLD slot — warm-chaining B from A under v2 loses
that calibration (acceptable: this track re-pretrains from the merged base).

## 2026-06-11 — beat-the-baseline run launched (v2 contract end-to-end)

```
run_id:     beatbase_v2alloc_p28_f512_lr3e5_20260611-035912
pod:        RunPod COMMUNITY 4090 (r79mlvidz9n9aa, 256 vCPU host, >=100GB cgroup, $0.34/hr)
base/init:  joint_mt_alloc_d256_T10_head3_20260610-164152/joint_best.pt
            (v2 alloc-CE pretrain: 75.0% vs physical_v4, 20.3% vs A, hold_mae 0.029)
opponent:   SAME ckpt, FROZEN (--opponent-ckpt; loader fixed to overlay supervised-format
            'model' ckpts under the entity_model. prefix — was a silent no-op)
goal:       beat the frozen baseline 20/20 (deploy gate: scripts/gate_beat_baseline.sh,
            10 stratified seeds x both seats, alloc_softmax decode both sides)
rollout:    infserver, PROCS=48, games=64/iter (2P/4P mix), T=10, f512, spool /dev/shm
reward:     --reward-decomp, win 0.5 + signals [0.1,0.2,0.1,0,0.35], gamma 0.997, inval_coef 0
update:     lr 3e-5/4e-6/4e-7 (heads/trunk/perception), clip 0.20, target-kl 0.03,
            ent +0.01 (BONUS — phase-2 explore), epochs 3, mb 64, unfreeze L2
new guards: in-epoch KL brake (running-avg >= 1.5x target after 4 mb, or single-mb > 4x
            -> stop update); per-iter 25-step-window tables now include the ROLLOUT table
            (inval/emit/inval%/fired/fleet sizes) alongside the 5 PBRS signals.
pre-launch: zero-lr parity EXACT through the real trial path (kl=0.0000 ratio=1.0000);
            lr calibrated from KL ~ lr^2 (1.25e-4 gave 0.50/epoch -> 3e-5 ~ on-target).
```

## 2026-06-11 — v3 lineage: first gate + the deploy-decode lesson

Run lineage beatbase_v3topk (joint_best + reinit frac head, frozen v2 baseline
opponent, equal signal weights [.15,.15,.15,0,.15], ent 0, lr 3e-5): sampled
winrate climbed to ~58-61% by 5-7 updates, inval=0 throughout, analyzer shows a
persistent bimodal loss mode (passive: ~1.5 emit/s, ~50% noop steps vs ~10/s,
~11% in wins; seat-1-heavy in 2P).

GATE at 5 updates (10 panel seeds x both seats vs frozen baseline):
  topk_self decode (fire iff logit > pair_logits[s,s], k-cap):   3/20 = 15%
  same + surplus lifted:                                          3/20 = 15%
  EXPECTATION-MATCHED decode (fire iff 1-(1-p)^k >= 0.5 from the
  training softmax [legal, self]; k-cap; capped surplus):       11/20 = 55%

Lesson: pair_logits[s,s] was DEAD in the v2 pretrain (v2 HOLD = frac diag), so
thresholding deploy on it collapses to noise-gating; matching the SAMPLER'S
MARGINALS deploys the trained distribution. OW_V3_DECODE=expmatch is now the
v3 default (selfthresh kept as A/B arm); OW_V3_TRUST_GARRISON toggles the
surplus lift (A/B pending under the good decode). Infra en route: /workspace
MooseFS silently dropped data writes (rc=0, 0 bytes) -> run relocated to
container disk + tmux; cross-iter OOM fixed by pre-update free (episodes/ok/
live_post_packed); in-epoch KL brake first fired at update 7 (kept 0.055).
Surplus A/B under expmatch decode (5-update ckpt, same 20 games): capped
11/20 = 55% vs lifted 8/20 = 40% -> guard stays the deploy default
(OW_V3_TRUST_GARRISON=0); re-test the lift at later gates.

## 2026-06-12 — graft + env 1.30.1 break the deploy plateau (62.5%)

Deploy policy pivoted to SAMPLED decode + surplus lifted (OW_V3_DECODE=sample,
OW_V3_TRUST_GARRISON=1) as the production default — "the bar shouldn't be
picked manually": deploy draws from the trained contract exactly as training
does, no threshold, no guard. Lineage gates under that arm (16 seeds x both
seats vs frozen deployed baseline): 25.0% (5up) -> 31.2 (12up) -> 34.4 (19up)
-> 25.0 (24up) -> 25.0 (31up) — a hard plateau, seat-0 collapsing to 1-3/16
while SAMPLED-form winrate vs the same ckpt sat at 60-93% (opponent-form
mismatch, the diagnosed binding constraint).

Two interventions, then the break:
1. ALLOC GRAFT (user-approved): copied pair_frac_head tensors from
   joint_mt_alloc joint_best into the PPO ckpt. hold 0.46-0.48 -> 0.02-0.06,
   alloc top1 0.86-0.95, emits -25% (fewer, bigger fleets).
2. kaggle_environments 1.28.1 -> 1.30.1 (swept_pair_hit continuous collisions,
   new map gen, native seeding). Run restarted from the grafted ckpt as
   ppo_beatbase_v3topk_p48_f512_lr75env130_20260611-181540. First update ate
   one detonating minibatch (kl 0.264, in-epoch brake at mb=1); KL settled to
   0.024-0.035 by iters 1-4. ALLOC hold oscillates 0.02-0.14 (no monotone
   re-softening of the graft). inval=0 throughout.

GATE at v0005 (5 updates post-graft under 1.30.1, sampled+lifted, 16 seeds x
both seats): seat0 8/16 = 50.0%, seat1 12/16 = 75.0%, OVERALL 20/32 = 62.5%
— first above-baseline deploy gate of the lineage, and the seat-0 collapse is
gone. Attribution is graft-dominant (decisive allocation survives the deploy
form); env shift affects both seats symmetrically. Next reads: gate again at
~v0010-v0012 to separate PPO progress from graft step-change; 20/20 still the
target.
GATE at v0010 (same 16-seed panel): seat0 8/16 = 50.0%, seat1 14/16 = 87.5%,
OVERALL 22/32 = 68.8% (+6.3pp over v0005) — post-graft PPO adds deploy
progress on top of the graft step-change; trajectory toward 20/20 positive.
Update-loop note: epoch-1 hot-brakes in 3 of 4 updates (kl 0.066-0.090, epoch 0
always in-target) — brake bounding damage; --epochs 1 is the standing lever if
gates stall. Next gate ~v0015.
GATE at v0015 (same panel): seat0 6/16 = 37.5%, seat1 12/16 = 75.0%,
OVERALL 18/32 = 56.2% — first down-tick (-12.6pp vs v0010; ~1.4 SE on a
32-game panel, so within noise, but direction-consistent with the
epoch-1 hot-brake pattern: 4 of updates 7-14 braked hot in epoch 1
(kl 0.064-0.090, one at ratio 0.69 / policy_loss 0.60), and iters 13-14
were the weakest sampled rollouts of the run. Lineage best stays v0010
(68.8%). Decision pending (user): --epochs 1 (epoch 1 is mostly braked
anyway; saves ~80-130s/iter) vs ride to a v0020 gate for a noise read.
GATE at v0020 (same panel): seat0 7/16, seat1 11/16, OVERALL 18/32 = 56.2%
— identical to v0015 despite updates 16-19 all landing clean (kl
0.017-0.031, no brakes). Hot-epoch-damage hypothesis weakened. Pooled
v0005/10/15/20: 78/128 = 60.9% +/- 4.3 (all four gates within ~1.5 SE of
flat 61%; seat0 pooled 29/64 = 45%, seat1 49/64 = 77%). Read: the graft
moved deploy from 25% -> ~61% and PPO is HOLDING it there, not climbing.
20/20 needs a different lever (topmeta re-anchor pretrain / opponent-form
fix / ent phase), not more iters of this config.

## 2026-06-12 — sigadv relaunch: reward what discriminates wins

Run ppo_beatbase_v3topk_p48_f512_sigadv_20260612-033427, resumed from the
env130 lineage's v0020 (md5-verified copy /root/ow/resume_sigadv.pt).
ONE change (user-approved): --signal-weights 0.08,0.08,0.08,0,0.26 ->
0.14,0.14,0.14,0,0.08. Rationale: across all 20 analyzed iters the W-L
mid-game gaps were ship/prod/planet_adv +0.35-0.47 every iter vs
fleet_spd +0.01-0.12 (and early-game identical W/L — consequence, not
cause); the graft already owns fleet sizing. Safety stays 0, win 0.5,
ent 0.0, epochs 2, lr unchanged. Watch: value_loss bump while the critic
residual re-fits the new Phi; gates on the same 16-seed panel at v0005+.
SIGADV GATE at v0005 (16-seed panel): seat0 7/16 = 43.8%, seat1 14/16 =
87.5%, OVERALL 21/32 = 65.6% — +9.4pp over the v0020 resume point after 5
updates on the advantage-weighted mix; best seat-1 panel of the lineage.
Behavior shift under the new mix: SEL p_self collapsed 0.24 -> 0.06 (fires
nearly every step), SEL ppl ~4.5 (broader targeting), ALLOC decisive
(top1 0.85-0.91, hold 0.05-0.07), KL clean 0.013-0.016 throughout.
Next sigadv gate at v0010.
