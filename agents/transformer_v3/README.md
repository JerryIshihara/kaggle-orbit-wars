# transformer_v3 — current agent design (2026-06-12)

The active model/agent line for Orbit Wars. Everything below is implemented
and committed on `ppo/multi-target-contract`; per-component status is at the
bottom.

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
  │                          → α0 concentration head (off L4 source tokens)
  ├─ player_state (B,4,256) ┐
  └─ glob (B,256)           ┴→ value trunk → win + fwd heads (+ inbound aux)
```

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
- PPO plan (task #17): logprob = select term (v3 machinery) + closed-form
  Dirichlet log-density of the DRAWN shares (ε-clamped identically at draw
  and re-eval); analytic Dirichlet entropy makes ent-coef price confidence
  directly (the α0-collapse guard); conc head in its own slow LR group.

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
- Q(s,a) head: rejected for now (expert-only data can't supervise off-policy
  actions; argmax selects Q's extrapolation errors). Revisit as distillation
  from the simulator, or properly trainable from PPO rollout data.

## Physics / eval substrate

`physics_utils.plan_launch` is env-1.30 parity-fixed (contact-shrink eta,
comet horizon clamp + end-of-path margin, planets-first swept tick order):
100% hit-or-honest-refuse on the empirical launch matrix
(`scripts/sanity_launch_matrix.py`). The deploy runner validates every
emitted pair through it. Re-run the matrix after any env upgrade.

## Status / TODO

- DONE: arch (A/B/D stages trained at least once), runner v4 load + decodes,
  K-rank mechanism, 300-replay pair+cross caches on GCS, notebooks for all
  stages, compete opponents prepared (old-PPO bests).
- TODO (task #17): v4 PPO sampler/logprob/entropy + shards version + parity;
  α0 stability retune; K-rank N×M sampler + gate A/B; eval-harness `--krank`;
  value-stage probe (expert-vs-random scoring calibration).
- Baselines to beat: frozen `joint_best` (v2 pretrain), `physical_v4`,
  old-PPO bests (v2 lineage closed at ~60% deployed vs its baseline; best
  ckpts in `data/runs/ppo/compete/`).
