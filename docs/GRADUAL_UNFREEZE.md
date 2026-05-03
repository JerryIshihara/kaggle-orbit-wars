# Gradual unfreeze plan — cross-entity & full-stack fine-tune

The pretraining pipeline trained encoders in isolation:
``FleetEncoder`` → ``PlanetEncoder`` → ``PlanetEntityEncoder``. The
cross-entity layer (``CrossEntityAttention``) sits on top of all
three. By default, the cross-entity pretrain freezes everything
upstream so the fresh attention layer + heads see a stable input
distribution while they find a sane local minimum. Once they have,
unfreezing the upstream layers gives the whole stack one more
opportunity to coordinate end-to-end — but doing it naively (all at
once, single LR) breaks the pretraining signal.

This doc lays out the staged unfreezing schedule and the
discriminative learning-rate table that goes with it.

## Implemented entrypoint

The staged resume path now lives in the existing cross-entity trainer:

```bash
python -m agents.transformer_v1.pretrain.cross_entity \
  --train-mode gradual-unfreeze \
  --resume-cross-pt data/runs/cross_entity/<run>/cross_entity_best.pt \
  --stage-epochs 5 \
  --batch-size 64 --num-workers 2 --eval-every 1 --device cuda
```

Notes:

* `--train-mode frozen` remains the original behavior: only
  `CrossEntityAttention` + heads train.
* `--train-mode gradual-unfreeze` assumes that frozen run already
  happened and starts at **Stage 1** below.
* `--stage-epochs` is a comma-separated list of durations for
  Stage 1 / 2 / 3 / 4. Examples:
  * `5`       → Stage 1 only
  * `5,5`     → Stage 1 then Stage 2
  * `5,5,5`   → Stage 1 → 2 → 3
  * `5,5,5,5` → Stage 1 → 2 → 3 → 4
* If `--resume-cross-pt` is omitted, the trainer auto-discovers the
  latest `cross_entity_best.pt` under `data/runs/cross_entity/`.

## Colab handoff

When repacking for a later Colab fine-tune, include the latest
cross-entity checkpoint in `weights.tgz`:

```bash
INCLUDE_CROSS_ENTITY=1 bash scripts/pack_for_gpu.sh
```

Or point at a specific checkpoint:

```bash
INCLUDE_CROSS_ENTITY=1 \
CROSS_ENTITY_RESUME=data/runs/cross_entity/<run>/cross_entity_best.pt \
bash scripts/pack_for_gpu.sh
```

## Layer hierarchy (top → bottom)

```
Heads (per-label classifiers/regressors)            ← top, freshest
CrossEntityAttention   (3 transformer layers + CLS)
PlanetEntityEncoder    (attention pool + fuse MLP)
PlanetEncoder          (scalar MLP + 1D-conv traj + gated fusion)
FleetEncoder           (Linear → GELU → Linear → LN)
                                                    ← bottom, oldest
```

"Top" layers are closest to the task — they should adapt first and
fastest. "Bottom" layers carry generic per-element semantics and are
the most expensive to lose, so they unfreeze last and only with very
small learning rates.

## Why gradual unfreezing

1. **Catastrophic forgetting.** Encoders were trained on multi-task
   labels with much larger per-batch signal than the cross-entity
   labels alone. If they unfreeze with the same LR as the new heads,
   noisy gradients early in training move them off-distribution and
   the multi-task pretraining is wasted.
2. **Gradient warmup, naturally.** A freshly-unfrozen layer has zero
   optimizer momentum and stale `requires_grad` history. Letting one
   layer thaw at a time means the adjacent (already-warm) layers
   provide stable gradients to it.
3. **Effective LR control.** Discriminative LRs (different LR per
   layer) are the standard tool, but they only really help if the
   layers also see different gradient magnitudes — which is
   exactly what staged unfreezing produces.

## Schedule

5 stages, ~5 epochs each (tunable per dataset size). Total ~25
epochs. Validation runs after every epoch; the *best by val mean
loss* across all stages is the keeper.

### Stage 0 — heads only (5 epochs)

Train the cross-entity transformer + per-label heads with everything
below frozen. Lets the new modules see a stable input distribution
and find their initial optimum.

```
FleetEncoder            requires_grad=False
PlanetEncoder           requires_grad=False
PlanetEntityEncoder     requires_grad=False
CrossEntityAttention    requires_grad=True   lr=1e-3
Heads                   requires_grad=True   lr=1e-3
```

Convergence signal to graduate: train loss + val loss both
plateauing for 2 epochs.

### Stage 1 — unfreeze the entity encoder (5 epochs)

The entity encoder is the closest upstream layer. Unfreezing it
first lets the model fix any "the new attention wants slightly
different inbound-fleet summary stats" mismatch.

```
FleetEncoder            requires_grad=False
PlanetEncoder           requires_grad=False
PlanetEntityEncoder     requires_grad=True   lr=1e-4    (10× lower)
CrossEntityAttention    requires_grad=True   lr=1e-3
Heads                   requires_grad=True   lr=1e-3
```

Why 10× smaller LR for the entity encoder: it's been trained for 30
epochs on its own pretrain objective; we want gentle adjustment, not
overhaul.

### Stage 2 — unfreeze planet & fleet encoder *top* layers (5 epochs)

Only the *outermost* projection of each per-element encoder. For
``PlanetEncoder`` that's the LayerNorm + the second `Linear` of each
branch + the gated-fusion gate. For ``FleetEncoder`` that's `fc2` +
LayerNorm. Bottom feature-extraction stays frozen.

```
FleetEncoder.fc1         requires_grad=False
FleetEncoder.fc2, norm   requires_grad=True   lr=1e-5
PlanetEncoder.scalar.fc1, traj.conv*
                         requires_grad=False
PlanetEncoder.scalar.fc2, traj.proj, gate, norm
                         requires_grad=True   lr=1e-5
PlanetEntityEncoder      requires_grad=True   lr=1e-4
CrossEntityAttention     requires_grad=True   lr=1e-3
Heads                    requires_grad=True   lr=1e-3
```

100× smaller for newly-thawed layers. The bottom-half features keep
their multi-task pretrain interpretation; only the projection that
hands tokens to the next layer gets to specialize.

### Stage 3 — full unfreeze with discriminative LRs (5 epochs)

Everything trainable, but each layer at its own LR:

| Layer | LR | Note |
|---|---|---|
| Heads | 1e-3 | task-specific, fastest |
| CrossEntityAttention | 1e-3 | freshest |
| PlanetEntityEncoder | 1e-4 | adapt-the-summary |
| PlanetEncoder top half | 1e-5 | gentle final tune |
| PlanetEncoder bottom half | 1e-6 | "don't break it" |
| FleetEncoder top | 1e-5 | gentle |
| FleetEncoder bottom | 1e-6 | "don't break it" |

A single roughly-geometric `1e-3 / 1e-4 / 1e-5 / 1e-6` ladder.

### Stage 4 — fine-tune at uniform low LR (5 epochs, optional)

If train + val are still falling at end of Stage 3, do one more
sweep at a uniformly low LR (`1e-5` everywhere) to settle the
balance. Skip if val plateaued.

## Implementation hooks

Two helpers, dropped into `pretrain/cross_entity.py`:

### `set_trainable(model, spec)`

```python
def set_trainable(model, *, freeze: list[str], unfreeze: list[str]) -> None:
    """Toggle requires_grad on submodule paths.

    ``freeze``/``unfreeze`` are dotted attr paths into ``model`` —
    e.g., ``"fleet_encoder"``, ``"planet_encoder.scalar.fc2"``.
    Resolves each path, sets every leaf parameter under it.
    """
    for path in freeze:
        for p in _resolve(model, path).parameters():
            p.requires_grad_(False)
    for path in unfreeze:
        for p in _resolve(model, path).parameters():
            p.requires_grad_(True)
```

### `build_param_groups(model, lr_table)`

```python
def build_param_groups(model, lr_table: dict[str, float]) -> list[dict]:
    """Return ``torch.optim`` param groups: one per (path, lr) pair
    in ``lr_table``. Skips params with requires_grad=False.

    Order in lr_table matters — paths are matched longest-first so a
    ``planet_encoder.traj.proj`` entry overrides a generic
    ``planet_encoder`` entry.
    """
```

Usage in the train loop:

```python
SCHEDULE = [
    # (start_epoch, freeze_paths, unfreeze_paths, lr_table)
    (0, ['fleet_encoder', 'planet_encoder', 'entity_encoder'], [],
        {'cross': 1e-3, 'heads': 1e-3}),
    (5, [], ['entity_encoder'],
        {'entity_encoder': 1e-4, 'cross': 1e-3, 'heads': 1e-3}),
    (10, [], ['fleet_encoder.fc2', 'fleet_encoder.norm',
              'planet_encoder.scalar.fc2', 'planet_encoder.traj.proj',
              'planet_encoder.gate',     'planet_encoder.norm'],
        {'fleet_encoder.fc2': 1e-5, 'fleet_encoder.norm': 1e-5,
         'planet_encoder.scalar.fc2': 1e-5, 'planet_encoder.traj.proj': 1e-5,
         'planet_encoder.gate': 1e-5,        'planet_encoder.norm': 1e-5,
         'entity_encoder': 1e-4, 'cross': 1e-3, 'heads': 1e-3}),
    (15, [], ['fleet_encoder', 'planet_encoder'],   # full unfreeze
        {'fleet_encoder.fc1': 1e-6, 'fleet_encoder.fc2': 1e-5,
         'fleet_encoder.norm': 1e-5,
         'planet_encoder.scalar.fc1': 1e-6, 'planet_encoder.traj.conv1': 1e-6,
         'planet_encoder.traj.conv2': 1e-6,
         'planet_encoder.scalar.fc2': 1e-5, 'planet_encoder.traj.proj': 1e-5,
         'planet_encoder.gate': 1e-5,        'planet_encoder.norm': 1e-5,
         'entity_encoder': 1e-4, 'cross': 1e-3, 'heads': 1e-3}),
]

for epoch in range(args.epochs):
    new_stage = [s for s in SCHEDULE if s[0] == epoch]
    if new_stage:
        _, freeze, unfreeze, lr_table = new_stage[0]
        set_trainable(model, freeze=freeze, unfreeze=unfreeze)
        opt = torch.optim.AdamW(
            build_param_groups(model, lr_table),
            weight_decay=args.weight_decay,
        )
        print(f"[stage] epoch={epoch}  param-groups={len(opt.param_groups)}")
    train_one_epoch(...)
    val = evaluate(...)
```

Replacing the optimizer at each stage transition is intentional —
fresh optimizer state for newly-thawed layers prevents the AdamW
running-mean-of-squared-grads from being stale and causing huge
effective LRs on first step. (Alternatively: keep the same optimizer
and `add_param_group` for new layers; equivalent for AdamW since old
params keep their state.)

## Pitfalls

* **Don't reset the LR schedule** when transitioning. If using
  cosine/warmup, advance the global step counter through stage
  transitions; resetting causes a learning-rate spike that wipes
  the prior stage's progress.
* **Watch val carefully across transitions.** A sudden val spike at
  Stage 1 onset is normal (1–2 epochs) — the entity encoder briefly
  fights the cross-attention. If val keeps climbing, the LR for the
  newly-thawed layer is too high; halve it and resume.
* **Save checkpoints after each stage**, not just at end. The
  best-by-val checkpoint is often at Stage 1 or Stage 2 end, not
  Stage 4 — full fine-tuning sometimes overshoots.
* **Track per-stage val curves separately** in `log.json`. The
  schedule transition epochs are landmarks: a label `"stage": 0/1/2/3`
  on each log entry makes diagnostics much easier.
* **LayerNorm and CLS are not in any encoder.** They live in
  `CrossEntityAttention` and always update from Stage 0. No special
  handling needed, but worth remembering when spec'ing freeze paths.

## Validation strategy

Three things to check at every stage transition:

1. **Held-out val mean loss** — should fall or stay flat across the
   transition; if it spikes ≥10% and doesn't recover by epoch 2,
   roll back the LR.
2. **Per-encoder probing accuracy** (optional). On the val split,
   re-evaluate the original encoder pretraining heads against the
   *currently-unfrozen* encoder. Catches catastrophic forgetting:
   if FleetEncoder's pretraining `mission_type_coarse` accuracy
   drops from 99% to 80% during Stage 3, the multi-task signal is
   being eaten. Add a small "pretraining preservation" auxiliary
   loss as a safety net.
3. **Gradient norms per layer.** Print average grad norm per param
   group at the end of each epoch. Healthy gradients are roughly
   inverse to LR (`grad_norm × lr ≈ const`). A deeply-unfrozen layer
   with `grad_norm > 100×` lower than a top layer means the LR is
   wasted there.

## Concrete first-iteration recipe

Start with Stages 0 and 1 only (10 epochs total, 5 each). If the
cross-entity heads converge cleanly and Stage 1 doesn't degrade
encoder pretraining accuracy, escalate to Stage 2/3 in a
follow-up run. **Don't run all 4 stages on the first try** — too
many failure modes mask each other; debug one transition at a
time.

With the implemented trainer, that current recommendation translates to:

1. Run the existing frozen Colab job as-is.
2. Pull `cross_entity_best.pt` back from GCS into `data/runs/cross_entity/<run>/`.
3. Re-pack with `INCLUDE_CROSS_ENTITY=1`.
4. Launch the gradual resume with `--train-mode gradual-unfreeze --stage-epochs 5`.
