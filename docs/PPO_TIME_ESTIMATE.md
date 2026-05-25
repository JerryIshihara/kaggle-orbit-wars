# PPO Phase 0 time estimate (without running)

A-priori wall-clock estimate for one **PPO iteration** of the design
captured in `docs/PPO_TWO_CPU_PROTOCOL.md`. No measurements taken here —
all numbers are derived from (a) the supervised training timing
documented in `agents/transformer_v2/README.md`, (b) the model parameter
counts, and (c) Apple-Silicon-CPU peak/sustained throughput.

The "epoch" in the user-facing ask maps to **one PPO iteration** = one
sync of `rollout → GAE → train → eval → archive`. The inner PPO `epochs`
hyperparameter (default 3) is the number of times the train phase sweeps
the rollout minibatches.

---

## Inputs

### Model (transformer_v2, `EntityPretrainModel`)

| Block | Params | Notes |
|---|---:|---|
| L0 specialists (frozen always) | 374 k | Planet / Comet / Fleet encoders |
| L1 PlanetEntityEncoder | 658 k | cross-attn(planets ↔ fleets) |
| L2 CrossEntityAttention | 1.05 M | 2-layer Pre-LN + CLS |
| L3 DualRoleAttention | 528 k | 2 cross-attn branches |
| L4 JointRoleAttention | 528 k | 1-layer encoder on 2P |
| PairHead trunk + FiLM (`conditioner_n_layers = 2`) | ~1.1 M | new spec depth |
| PairHead `pair_logits` head (`head_n_layers = 3`) | ~132 k | 3-Linear MLP, new spec depth |
| PairHead `pair_frac` head (`head_n_layers = 3`) | ~132 k | 3-Linear MLP, new spec depth |
| **value_head** (3-Linear MLP) | **~132 k** | the only NEW PPO module |
| **Total trainable in Phase 0** | **~396 k** | value_head + 2 action heads |
| **Total trainable in Phase 1** | **~2.4 M** | + trunk + FiLM + L4 |

### Hardware (per machine — Apple M1 MacBook Air, the spec'd "Machine B")

- 4 performance cores @ ~3.2 GHz, NEON FMA 4-wide → **~25 GFLOPS** fp32
  peak per core, **~50 GFLOPS** sustained across the 4 P-cores after
  thread-overhead. The 4 E-cores add ~10 GFLOPS on top under PyTorch /
  Accelerate.
- Total realistic sustained throughput for PyTorch matmul on CPU: **50–80
  GFLOPS fp32**.
- Machine A (the dev box) is likely an x86 or another Apple silicon —
  assume similar order of magnitude; the iteration is gated by the
  slower side anyway because rollouts split 50/50.

### Anchor: supervised training timing on MPS

From `agents/transformer_v2/README.md`:

> Per-epoch time: ~3-4 min/epoch on MPS (M2/M3, batch=32), ~30 s/epoch on
> a Colab T4. T=6 cost: ~1.5-2× the L1 cross-attn FLOPs vs single-frame,
> plus ~6× the L2 sequence length.

→ 240 s / (60 424 snapshots / 32) ≈ **127 ms per minibatch at B = 32 on
MPS at T = 6** (forward + backward, all of L1–L4 + PairHead trainable).

Backward typically dominates this (≈ 60 %) → forward alone ≈ **50 ms /
minibatch on MPS at B = 32, T = 6**.

### CPU-vs-MPS scaling factor

Apple's Accelerate-backed CPU matmul on M-series is roughly **3× slower**
than MPS for these transformer shapes. PPO uses **T = 1** (CPU
optimization), which cuts L2 cost ~36× (sequence length 65 vs 385) but
that's a small part of the total because PairHead dominates. Net: PPO
forward at T = 1 on CPU is about the same ms-cost as supervised forward
at T = 6 on MPS, give or take a factor of 2.

**Working assumption: per-snapshot forward on CPU at T = 1**
- B = 1 (rollout):   **~15 ms / snapshot** (poor BLAS efficiency)
- B = 32:             **~5 ms / snapshot**
- B = 256:            **~3 ms / snapshot** (BLAS sweet spot for M1)
- B = 1024:           **~3 ms / snapshot** (memory bandwidth bound past here)

---

## Per-stage estimates (one PPO iteration)

### 1. Rollout — 128 episodes total, split 50/50 across A and B

```
per-episode forwards:
  500 env steps × 2 forwards/step (learner + opponent) = 1000 forwards
per-episode time at 15 ms/forward (B = 1):
  1000 × 15 ms ≈ 15 s

per machine (64 episodes):
  64 × 15 s = 960 s ≈ 16 min

both machines in parallel: rollout phase ≈ 16 min wall-clock
```

**Caveat:** this is the **un-optimized** path. The protocol's CPU-perf
section (batched envs per worker with N_env = 4, `torch.compile`, shared
opponent model) shrinks this 3–5×. With those optimizations applied,
rollout drops to **~4–6 min wall-clock**. The MVP implementation in this
PR does not include batched envs / `torch.compile` — those are the next
optimization PR, gated on measured throughput.

### 2. GAE

Pure tensor ops on ~32 000 steps (128 episodes × ~250 learner steps).

```
GAE ≈ 30 s (one Python loop per episode; vectorizable to ~3 s if it shows up).
```

### 3. Train — Phase 0 (A only)

```
samples to train on:  128 episodes × ~250 learner steps ≈ 32 000 steps
minibatch size:       1024
minibatches/epoch:    32
PPO epochs:           3
total minibatch steps: 96

per minibatch step at B = 1024:
  forward (full model):       1024 × 3 ms ≈ 3 s
  head-only backward:         ~0.3 s   (only ~400 k trainable params downstream of frozen trunk)
  BC anchor forward:          ~3 s
  BC head backward:           ~0.3 s
  optimizer step:             ~10 ms
                              -------
  per minibatch step:         ~7 s

train phase wall-clock:        96 × 7 s = 672 s ≈ 11 min
```

**Phase 0 does NOT distribute** (per the design — grad sync overhead
exceeds compute saved when only ~400 k params are trainable). All 11 min
fall on Machine A.

### 4. Eval — quick gate, 192 episodes, split 50/50

```
SEEDS_QUICK = 32 seeds × 2 seats × 3 mandatory opponents = 192 episodes

per-episode inference time (no backward): ~15 s
per machine (96 episodes):                96 × 15 s = 1440 s ≈ 24 min

both machines in parallel: eval phase ≈ 24 min wall-clock
```

Same un-optimized caveat — with batched envs eval drops to ~6–8 min.

### 5. Archive

```
~30 s of file moves (rsync --remove-source-files of cold checkpoints,
                     rollouts, eval JSONs).
```

---

## Total per PPO iteration (Phase 0, un-optimized MVP)

The phases are sequential because each one's output gates the next
(rollout → GAE → train → eval → archive).

| Phase | Wall-clock | Notes |
|---|---:|---|
| Rollout | ~16 min | both machines in parallel |
| GAE | <1 min | A only, tensor ops |
| Train (Phase 0) | ~11 min | A only; no distribution in Phase 0 |
| Eval | ~24 min | both machines in parallel |
| Archive | <1 min | A only, file moves |
| **Total per iteration** | **~52 min** | end-to-end wall-clock |

### With the CPU optimizations spec'd in the protocol

Batched envs (N_env = 4) + `torch.compile(mode="reduce-overhead")` +
shared opponent model across workers per the protocol's "CPU performance
→ Rollout" section bring rollout and eval down ~3×:

| Phase | Un-optimized | Optimized |
|---|---:|---:|
| Rollout | ~16 min | ~4–6 min |
| GAE | <1 min | <1 min |
| Train | ~11 min | ~11 min (already big-batch) |
| Eval | ~24 min | ~6–8 min |
| Archive | <1 min | <1 min |
| **Total** | **~52 min** | **~22–26 min** |

### Phase 1 projection (after Phase 0 stabilizes)

When PPO moves to Phase 1, the trainable set grows from ~400 k to
~2.4 M, and training distributes via file-mediated grad averaging. From
the protocol's "Sync overhead estimate":

```
Phase 1 sync: ~150 ms per minibatch × 96 mb/iter ≈ 30 s per iter
```

Distributed train wall-clock ≈ `~11 min / 2 + 30 s sync ≈ 6 min`. Net
iteration drops from ~22 min (Phase 0 optimized) to ~17 min (Phase 1
distributed + optimized). The break-even where distributing pays off is
about Phase 1's 2.4 M trainable params.

---

## What the "one epoch" of `--epochs 3` costs (the inner PPO epoch)

If "one epoch" means the inner PPO epoch (a single sweep of the 32
minibatches in the rollout buffer), that's:

```
32 minibatches × ~7 s/minibatch ≈ 3 min 45 s per inner epoch (Phase 0)
```

With 3 default inner epochs, that's the ~11 min train phase above.

---

## Error bars

The above carries ~2× uncertainty per stage; the most likely sources:

1. **CPU sustained GFLOPS** can swing 2× depending on thermal state and
   whether PyTorch picks the Accelerate kernels (M-series benefits a lot
   from BLAS fastpath).
2. **B = 1 forward latency** is dominated by Python overhead and tensor
   allocation, not matmul. Real measurement can be anywhere from 5 ms
   (best-case `torch.compile`) to 30 ms (cold).
3. **Episode length** is ~500 steps for typical 2P games; some games end
   early on a runaway victory which lowers the average.
4. **BC anchor minibatch size** is assumed equal to PPO minibatch size
   (1024). If the cache loader is slow it can be the bottleneck.

Recommendation: budget ~1 hour per Phase 0 iteration for the un-optimized
MVP. Confirm against the first 2–3 measured iterations, then enable the
spec'd CPU optimizations once the loop is verified correct.

---

## How to validate this estimate cheaply

Three measurements pin down most of the uncertainty:

1. **Per-snapshot forward at B = 1 on CPU** — `python -c "..."`-style
   timing of `PPOActorCritic.forward` on a synthetic minibatch.
2. **Per-minibatch forward + backward at B = 1024** with Phase 0 freeze
   applied — same kind of microbenchmark on a batch from the existing
   pair cache.
3. **One short rollout episode** — wire the env adapter, time `run_episode`
   end-to-end with a random opponent.

(1) and (2) pin down the per-step compute; (3) catches env-step overhead
that's harder to estimate a-priori. Together they let you re-derive the
table above and either confirm or correct the budget before launching a
full PPO run.
