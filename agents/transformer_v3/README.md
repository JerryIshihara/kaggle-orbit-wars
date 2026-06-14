# transformer_v3 — current agent design (2026-06-14)

The active model/agent line for Orbit Wars. Everything below is implemented
and committed on `ppo/multi-target-contract`; per-component status is at the
bottom.

> **TL;DR of the RL campaign (read `## Training campaign` before launching runs).**
> A pretrained v4 actor sits at **~65.6% vs `physical_v4`**. Across **7 PPO
> runs / ~100 iters / two objectives (self-mirror + opponent league) / every
> entropy regime**, *nothing has beaten that pretrain by more than ~1 SE.*
> The wall is **not** the training signal — it is a **structural force leak**:
> ~30% of intended launches never fire (≈13% physics-doomed at the sun/
> boundary, ≈15% under-sized), which is exactly `physical_v4`'s edge (it
> never wastes force). The current bet (**v5 Q-head**) attacks that leak
> directly. Entropy/KL/cap/penalty tuning only changes *how* a run fails.

## Architecture

```
INPUT: ONE history stack at the 18-frame UNION of two windows
       LONG  T=10 @ stride 5  (offsets 45..0)  — economy/momentum
       SHORT T=10 @ stride 2  (offsets 18..0)  — fine recency (1.30 swept physics)
  │
L0 frozen specialists (planet / fleet / comet, d256)      [per frame]
L1 PlanetEntityEncoder (frozen)                            [per frame]
  │            + zero-init OWNER one-hot projection (v3.1)
  ▼
L2 = DualRateCrossEntity
  ├─ LONG branch  CrossEntityAttention(T=10) ─┐  each with 4 PLAYER CLS
  ├─ SHORT branch CrossEntityAttention(T=10) ─┤  tokens behind an asymmetric
  │                                           │  mask (planets/global blind
  │   pre-fusion taps: short→ShortHorizonHeads│  to them → init-equivalence)
  ▼                                           ▼
  zero-init [I|0] fusions: tokens, global CLS, player tokens (512→256)
  │
  ├─ ctx_now (B,P,256)  → L3 DualRole → L4 JointRole → PairHead (+FiLM)
  │                          → pair_logits (select) + pair_frac (alloc mean)
  │                          → q_value (per-pair launch Q, v5; opt-in)
  │                          → α0 concentration head (off L4 source tokens)
  ├─ player_state (B,4,256) ┐
  └─ glob (B,256)           ┴→ value trunk → win + fwd heads (+ inbound aux)
```

Heads on the shared PairHead `h_film` (per-pair B,P,P features): `pair_logits`
(select), `pair_frac` (alloc mean), and — when `with_q_head=True` (v5) — a
third `q_value` head. All three read the *same* features; `with_q_head=False`
leaves a model byte-identical to before (existing ckpts load 0-missing/0-extra).

Key invariant: every zero-init gate (fusions, owner proj) makes a
warm-started model reproduce its parent bit-for-bit at init; new capacity
fades in through training. Smoke-proven in `smoke_dual_rate.py`.
The PlayerConsolidator is REMOVED (v3.1): player state = in-L2 player
tokens (−1.3M params, one less attention pass).

## Action contract v4 — `bounded_k_select_dirichlet_alloc_v4`

- SELECT: bounded-k multinomial over [legal targets + self], k = min(3,
  ships//min_launch). Confidence surfaced at decode as the fire marginal
  `1−(1−p)^k` (no training change).
- ALLOC: Dirichlet over [fired…, HOLD]: mean = the existing frac softmax
  (warm-starts exactly from v2/v3 heads), α0 = ALPHA_MAX·sigmoid(conc head)
  = the actor's LEARNED confidence. α0 replaces the hardcoded `--sigma`
  exploration knob: per-source, state-dependent spread.
- Pretrain loss: Dirichlet NLL on ε-smoothed expert shares (`dirichlet_alloc.py`;
  near-one-hot experts REQUIRE the smoothing + cap). Known instability: α0
  can saturate then detonate (seen stage-B ep17-19) — retune ALPHA_MAX↓ /
  conc-lr↓; watch `dir/alpha0_satfrac`.
- PPO (BUILT, `ppo/sampler.py::sample_dirichlet_k` + `ppo/loss.py`): logprob =
  select term (v3 machinery) + closed-form Dirichlet log-density of the DRAWN
  shares (ε-clamped identically at draw and re-eval → ratio≡1 unchanged-policy,
  parity `test_dirichlet_parity` < 1e-3). Analytic Dirichlet entropy makes
  ent-coef price confidence directly; conc head in its own slow LR group
  (`--lr-conc 5e-6`).
- **α0 cap is checkpoint CALIBRATION, not a free knob** (`ALPHA_MAX=200`,
  `OW_V4_ALPHA0_CEIL` deploy/runtime override). A conc head trained at cap C
  and run under cap C′≠C rescales every α0 → train/deploy mismatch (cost ~25pp
  in the A100 regression). Applied IDENTICALLY at sample (sampler), update
  (loss recompute + entropy), and deploy (runner + krank) or the ratio/
  distribution desync. Capping α0 floors the Dirichlet entropy — the lever
  that makes the determinism crater unreachable (see campaign).
- **Per-component PPO clipping** (v4 only): the Dirichlet log-DENSITY swings
  much harder per weight-delta than a discrete contract, so a whole-action
  ratio clips 30–60% of samples (zeroing the step's gradient). The v4 surrogate
  stores per-row logps (`select_row_logp`/`alloc_row_logp`) and clips EACH
  component's ratio independently. `target_kl` is per-component; the v2-era
  0.01 chronically early-stops v4 at 1/3 epochs — use ~0.02–0.03.

## Staged pretrain (top-meta 300 replays: JW 194 / HM 68 / TonyK 25 / M&J&M 13)

| stage | trains | tasks | artifact |
|---|---|---|---|
| A `l2_pretrain` | dual L2 only (ALL 72k snapshots) | planet/player/global forecasts × short(t+5)/long(t+10) on the FUSED outputs | `l2_best` → merged into `joint_warm_merged` |
| B `actor_pretrain` | + L3/L4 + PairHead + α0 (no value) | select BCE + Dirichlet NLL + sh5 aux; per-part LRs | `actor_best` (65.6% vs physical_v4, sampled) |
| D `joint_v4_pretrain` | actor (slow) + value heads (fast) | PRUNED value set: win + fwd kept; back/rank/survives REMOVED; ADDED temporal contrast + player inbound aux | `jointv4_best` (holdout win_acc 0.85, contrast 0.68) |

LR tiers (stage D): value 1e-4 / action heads 2e-5 / L3+L4 1e-5 / L2 5e-6.
Value labels come from the cross cache — REQUIRES the v2 cross-entity
featurizer (the v1 writer lacks final_rank/player_valid; bug fixed
2026-06-12, preflights now assert `player_valid` is live).

## Deploy mechanisms (`runner.py` + `krank.py`)

- Runner loads v3/v4 ckpts (arch dispatch from `config.arch`); inference is
  SINGLE-FRAME (no deque); owner one-hot threaded at forward.
- Decodes: `OW_V3_DECODE=sample` (production: sampled select+alloc — measured
  65.6% vs physical_v4 vs 31.2% for deterministic expmatch at k=3),
  `expmatch/expcount/selfthresh` deterministic arms,
  `OW_V4_ALLOC=dirichlet` (alloc shares drawn from the learned Dirichlet).
- K-RANK (`KRankAgent`, simulate-then-score): per step draw K full action
  sets (cand 0 = deterministic), apply each to the obs PRE-TICK (garrison
  debit + rim spawn only — opponent cancels across candidates), score the
  post-action state with `1.0·σ(win) + Σ_{h∈5,10,20} {.5,.3,.2}·(sig-w·fwd)`,
  emit the argmax set. 280ms/step @ K=4 CPU. DEPLOY/GATES ONLY — greedier
  than the recorded policy; never for PPO rollouts. Planned: N×M hierarchy
  (N select-samples × M alloc-samples), gate A/B K∈{1,4,8}.
- **Q(s,a) head (v5, BUILT — `agents/transformer_v3/q_head.py`)**: per-pair
  `Q[s,t]` scoring each candidate launch, on the SAME `h_film` features as
  select. Earlier rejected on EXPERT data (experts never make doomed launches
  → OOD); the fix is to train on **PPO-ROLLOUT** data where exploration
  produces them. Target = DENSE plan_launch doomed label (every legal pair →
  `Q_DOOMED=−1.0`, the free anti-doomed signal) + SPARSE TD return (fired pairs
  → GAE return). Deploy gate `q_gate_select_logits` (`OW_V5_QGATE` soft bias /
  `OW_V5_QFLOOR` hard learned-reachability mask) biases SELECT toward high-Q
  (reachable) targets — UPSTREAM of the Dirichlet alloc, never replacing it.
  Learnability proven (synthetic sep +0.05→−1.4; real-rollout sep −0.27 by
  iter 2). This is the current bet against the doomed-launch bottleneck.

## RL / PPO design (as built)

- **Objective** (reward-decomp critic, design A): terminal `±1` (win) +
  potential-based shaping `r_t = γΦ(s_{t+1}) − Φ(s_t)`, γ=0.997,
  `Φ(s)=Σ wᵢ·sᵢ` over the 5 P1 signals (ship/prod/planet adv, safety, fleet
  speed), all wᵢ=0.1 so Φ∈[0,0.5]; mean |r_shape|≈0.005/step (terminal still
  dominates by ~100×). Critic = pretrained win head − Φ + zero-init residual;
  stored value = V−Φ so GAE stays aligned. Optional direct invalid-launch cost
  `−inval_coef·doomed_rate` (see findings — it does not bite).
- **Contract**: `--action-contract v4` (Dirichlet). `--opponent-contract`
  selects opponent-seat decode (v4 for a frozen-v4 mirror).
- **Opponent / league** (`--opponent-pool`, infserver only): per-game opponent
  cycled over `{self, ckpt, <heuristic id>}`. `self`→learner policy (forwarder),
  `ckpt`→`--opponent-ckpt` (two-policy forwarder), any registry id (e.g.
  `physical_v4`)→worker-LOCAL heuristic (no GPU). The forwarder's 5th request
  field became `use_learner_policy` (learner seat ∪ self-game opp seats → B;
  ckpt-game opp seats → A; heuristic seats skip the forwarder). Lets a single
  run mix self-play + a frozen mirror + a directional heuristic.
- **Entropy control**: hard floor via the α0 cap (above) + a soft adaptive
  controller `--ent-target` (multiplicative SAC-style: ent_coef ×=
  exp(rate·tanh((target−H)/2)), clamped [1e-4,0.05]). The cap is prevention,
  the controller is a reactive backstop (it LAGS a fast dive — see findings).
- **Rollout/infra**: infserver (CPU env workers + one GPU forwarder, single-
  forwarder-serial — throughput ceiling, not GPU/worker-bound); shm multi-
  forwarder path is v4-capable too. T18 union history threaded as
  `history_offsets`. StepRecord index fields stored int16 (4× pack shrink).
  See `docs/PPO_RUNPOD_INFSERVER_PROTOCOL.md` and the OOM note below.

## Training campaign — problems, findings, bottleneck (2026-06-12→14)

**Result ladder vs `physical_v4` (sampled decode, env 1.30; 32-game SE≈8pp,
64-game SE≈6pp):**

| model | vs physical_v4 | note |
|---|---|---|
| v4 pretrain (`jointv4_best`) | **65.6%** | the number to beat |
| run3-iter10 (per-component clip) | **68.8%** (32g) | best PPO artifact; within 1 SE of pretrain |
| run3-iter20 / run4-iter10 | 40.6% / 43.8% | post-peak collapse |
| run5 (unthrottle + α0 cap 40) iter3→10 | 62.5%→45.3% | over-diffused; peak moved earlier |
| run6 league iter10 / iter25 | 65.6% / 60.9% (64g) | league did NOT break the wall |

**Finding 1 — the objective wall.** Every config peaks ≈63–69% then degrades;
peak ≈ pretrain within 1 SE. A frozen-self **mirror** only teaches beating a
copy of yourself (transfers to ~parity). The **opponent league** (50%
`physical_v4`) was built to add a directional gradient — it held entropy
stable but still capped at 65.6%. ⇒ the bottleneck is not the training signal.

**Finding 2 — the entropy-band law.** v4 deploys well only in a narrow band
(H≈−2 to −3). Below ~−7 the Dirichlet goes near-deterministic → the policy
re-fires its top picks every step (a cadence pathology) → gates crater to
~40%. Above (too diffuse, H≳−1) it under-commits force → also weak. Failure
MODE depends on the knobs (concentration crater vs diffusion drift); the
~65% CEILING does not move.

**Finding 3 — the real bottleneck is a force leak, not entropy.** Replay
analysis (`scripts/analyze_target_kinds.py`) is identical across mirror and
league runs: static 71–74% / orbital 25–28% / comet 1% of picks, and **~30% of
intended launches never fire** — ~13% physics-DOOMED (orbital lead-aim crosses
the sun/boundary; comet 94% dead) + ~15% under-sized (`insufficient_garrison`).
`physical_v4` validates every launch and never wastes force — that gap is its
edge. No entropy/KL setting closes a 30% force leak.

**Finding 4 — penalties and the critic cannot teach reachability.** A direct
`--inval-coef` cost held the doomed rate flat at 0.02 (run4) AND 0.1 (run7).
The value head can't price it either: V(s) scores STATES not launches, and the
doomed consequence reaches it only as DILUTED (≈40 ships of hundreds), DELAYED
(fleet flies ~10–15 steps to the sun), CONFOUNDED ship loss — uncreditable in
~100 iters. ⇒ the lever must make doomed targets **observable/unselectable**,
not merely penalized. Two paths: a hard reachability mask in `legality_masks`,
or the **v5 Q-head** (chosen — learnable, soft, expresses "risky" vs "dead";
the dense plan_launch label gives the signal the reward cannot). Current run:
`ppo_v5_qrun` (warm jointv4 + `--with-q-head --q-coef 0.5`, league).

**Operational hazards (each cost a run this campaign):**
- **Orphaned `multiprocessing.spawn` workers** stack across launch/kill cycles
  (98 ≈ 80GB on a 109GB-cgroup pod) → the next run's pack SIGKILLs after the
  `Φ-shaping:` line, no traceback. `pkill -f train_local_trial` does NOT reap
  them — ALWAYS `pkill -9 -f multiprocessing.spawn` before relaunch, in a
  SEPARATE ssh command from `cat>code.tgz` (the broad -9 self-kills the ssh
  session and truncates the upload). See `project_ppo_infserver_orphan_oom`.
- **cgroup OOM at pack**: 64 games × T18 OOMs a 109GB pod; 32 games fits.
- **Pod `/workspace` MooseFS silently drops appends** (logs vanish) — keep
  OUT_DIR on local disk; rsync ckpts to the controller EARLY (a pod exit/
  terminate loses everything local).
- PPO ckpts load in the runner directly (policy→model key strip) — required,
  else gate evals are 0/32 crash-walkovers.

## Physics / eval substrate

`physics_utils.plan_launch` is env-1.30 parity-fixed (contact-shrink eta,
comet horizon clamp + end-of-path margin, planets-first swept tick order):
100% hit-or-honest-refuse on the empirical launch matrix
(`scripts/sanity_launch_matrix.py`). The deploy runner validates every
emitted pair through it. Re-run the matrix after any env upgrade.

## Status / TODO

- DONE: arch (A/B/D pretrain stages); runner v4 load + decodes; K-rank +
  N×M sampler + `--krank` gate flag; **v4 PPO fully built** (Dirichlet sampler/
  logprob/entropy + per-component clip + α0 cap + adaptive entropy + parity);
  opponent league; **v5 Q-head built + validated** (module, full PPO path,
  deploy gate); 300-replay caches on GCS.
- BEST ARTIFACT: `run3-iter10` (68.8% vs physical_v4, on GCS
  `runs/ppo_v4_run3/`) — but within 1 SE of the `jointv4_best` pretrain (65.6%).
- ACTIVE: `ppo_v5_qrun` (Q-head) — gate plan: 64g vs physical_v4 ×
  {plain sample, `OW_V5_QGATE=1.0`, `+OW_V5_QFLOOR`} + replay doomed-rate.
- OPEN LEVERS (if the Q-head plateaus): hard reachability mask in
  `legality_masks`; reachability INPUT features (so the actor sees what's
  reachable, not just gets gated); per-launch Q target via simulate-then-score
  distillation (vs the coarse step-level GAE return); multi-neural league
  (past PPO snapshots in the forwarder); bigger gate panels (≥64g) on finalists.
- Baselines to beat: `physical_v4` (the scored opponent; GOAL 10:10), frozen
  `jointv4`, old v2-PPO bests (`data/runs/ppo/compete/`).
