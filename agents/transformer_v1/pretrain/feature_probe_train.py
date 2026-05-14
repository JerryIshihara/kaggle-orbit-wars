"""End-to-end feature-probe ablation.

**Question this experiment answers:** the linear probe in
``scripts/probe_encoder_representation.py`` showed the FROZEN
:class:`PlanetEntityEncoder` already loses most per-planet info (e.g.
``inbound_total_h10`` r² ≈ 0.33, ``inbound_slot{0..3}_h10`` and
``n_*_R`` all r² < 0). That probe doesn't distinguish two hypotheses:

  A. The entity-encoder *architecture* genuinely can't fit a per-
     planet representation of these basics → we need a wider / deeper
     encoder, or a different aggregation.
  B. The architecture is fine, but no training signal ever pushed the
     encoder to encode garrison / per-slot inbound — the action-stage
     BC loss only cares about the chosen ``(source, target)``, so the
     encoder optimized for something else.

This module tests (B): wire a per-planet regression head **directly on
top of the entity encoder output** (NO cross-attention), unfreeze
every encoder below it, and train the whole stack end-to-end to
reconstruct:

    [garrison, inbound_slot0, inbound_slot1, inbound_slot2, inbound_slot3]
    per planet, per snapshot                                    (B, P, 5)

If trained ``val_mae`` falls clearly below the frozen-probe ``mae``
(stored in ``data/runs/probe_encoder_Ebi_v2_5k.json``) → hypothesis B
holds, the architecture has the capacity, the BC objective just never
asked for it. If it doesn't → hypothesis A, the encoder is the wall.

Architecture (standalone, does not modify ``pair_score.py``):

    FleetEncoder ──┐
    PlanetEncoder ─┴─→ PlanetEntityEncoder ──→ entity_now (B, P, d)
                                                  │
                                            FeatureProbeHead
                                            (MLP: d → hidden → 5)
                                                  │
                                            (B, P, 5)  [garrison, inb0..3]

No cross-attention runs at all. The "global" context the v2
post-cross TargetHead relied on is intentionally absent — these labels
are local per-planet quantities and the test is whether the entity
encoder can carry them in its per-token state.

Labels — already present in every snapshot via
:class:`CrossEntitySnapshotDataset`:

  * ``garrison``     — ``planet_features[..., 6]``  (log1p(ships) / SHIPS_LOG_MAX)
                       Lives in the raw input; predicting it from the
                       encoder output tests whether the encoder's
                       nonlinearity distorts what's directly handed in.
  * ``inbound_slot{0..3}_h10`` — ``ships_arriving_within_10[..., k]``
                       NOT in the planet inputs; the encoder has to
                       *synthesize* them by pooling per-(planet, owner)
                       fleet tokens. This is the real test.

Loss: masked MSE over real planet slots only, summed across the 5
labels (equal weight). Per-label val_mae + r² + baseline_mae reported
each epoch so each label can be compared against the frozen-probe
numbers in ``data/runs/probe_encoder_Ebi_v2_5k.json`` cell by cell.

Run (CLI, local CPU):

    python -m agents.transformer_v1.pretrain.feature_probe_train \\
        --encoder-ckpt data/runs/action/20260505-143435/action_best.pt \\
        --player Ebi --filter all \\
        --epochs 8 --batch-size 64 --lr 1e-3 --device cpu \\
        --out-dir data/runs/feature_probe/Ebi_<TS>

Notebook (reuse an existing in-memory dataset across iterations):

    from agents.transformer_v1.pretrain.feature_probe_train import \\
        train_feature_probe_kwargs
    train_feature_probe_kwargs(
        encoder_ckpt='data/runs/action/20260505-143435/action_best.pt',
        out_dir='data/runs/feature_probe/Ebi_xyz',
        player='Ebi', epochs=8, lr=1e-3,
        dataset=existing_dataset,
    )
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from .cross_entity import _entity_tokens_per_step
from .expert_action import ActionSnapshotDataset
from .pair_score import (
    ACTION_DATASET_DIR,
    CROSS_ENTITY_DATASET_DIR,
    ENTITY_DATASET_DIR,
    FLEET_DATASET_DIR,
    PLANET_DATASET_DIR,
    prepare_dataset,
)
from ..encoder import FleetEncoder, PlanetEncoder, PlanetEntityEncoder


# ---------- Fleet-CSV integrity audit ----------
def _csv_max_turn(path: Path) -> int | None:
    """Return the largest ``turn`` value in a CSV with a ``turn`` column,
    or ``None`` if unreadable / missing the column. Mirrors the helper in
    ``scripts/build_encoder_dataset.py`` — kept local to avoid an awkward
    cross-package import from the script directory.
    """
    try:
        with path.open() as fh:
            reader = csv.DictReader(fh)
            if "turn" not in (reader.fieldnames or []):
                return None
            last: int | None = None
            for row in reader:
                t = row.get("turn")
                if t is None or t == "":
                    continue
                try:
                    last = int(t)
                except ValueError:
                    continue
            return last
    except OSError:
        return None


def _replay_last_fleet_turn(replay_path: Path) -> int | None:
    """Largest ``t`` for which any seat's observation reports a non-empty
    ``fleets`` list, or ``-1`` if the replay never had a fleet, or
    ``None`` on unreadable.

    Fleets in the Kaggle Orbit Wars replay are global (every seat sees
    the same set), so we sample seat 0. This is the right reference for
    "where should the fleet CSV reach": a replay that finishes its last
    150 turns with no fleets in flight (game won early, no more
    launches) genuinely has no fleet rows to write — flagging that as
    "incomplete" is a false positive (and the earlier ``len(steps) -
    5`` check did exactly that, over-excluding 78 of Ebi's 434 stems
    that were in fact fine).
    """
    try:
        with gzip.open(replay_path, "rt") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    steps = payload.get("steps") or []
    last_with_fleets = -1
    for t, step in enumerate(steps):
        if not step or not isinstance(step[0], dict):
            continue
        obs = step[0].get("observation") or {}
        if obs.get("fleets"):
            last_with_fleets = t
    return last_with_fleets


def _incomplete_fleet_stems(
    player: str | None,
    fleet_dir: Path,
    replay_dir: Path,
) -> set[str]:
    """Stems whose ``fleet_<stem>.csv`` stopped short of the last turn
    in the replay that **actually had fleets in flight**.

    Two failure modes this distinguishes from a benign empty-end-game:

      * Interrupted write — CSV stops well before the replay's last
        non-empty-fleet turn. Smoking gun: ``75610892_4_2``'s fleet CSV
        halted at turn 409 while fleets were still in flight through
        turn 438. Real corruption; the inbound label diverges from the
        fleet input.
      * No-late-fleets game — the player resigned or simply launched
        nothing in the last 100+ turns. CSV's last turn matches the
        last turn the replay had any fleet, which can be far earlier
        than ``len(steps)``. Benign; the CSV is complete relative to
        the actual fleet activity.

    Comparing ``csv_max`` against ``len(steps) - K`` (the original
    check) couldn't tell these apart; comparing against the replay's
    last-fleet turn does.

    Only ``fleet_*.csv`` is audited because the cross_entity-stage
    audit confirmed every other dataset (planet / entity /
    cross_entity / action) reached the expected last turn for every
    stem.
    """
    if player is not None:
        replay_files = sorted((replay_dir / player).glob("*.json.gz"))
    else:
        replay_files = sorted(replay_dir.rglob("*.json.gz"))
    bad: set[str] = set()
    t0 = time.time()
    n_checked = 0
    n_benign = 0
    for rp in replay_files:
        stem = rp.name[: -len(".json.gz")] if rp.name.endswith(".json.gz") else rp.stem
        fleet_csv = fleet_dir / f"fleet_{stem}.csv"
        if not fleet_csv.exists():
            continue
        last_fleet_t = _replay_last_fleet_turn(rp)
        if last_fleet_t is None:
            continue
        if last_fleet_t < 0:
            # Replay never had a fleet — empty CSV is correct.
            n_benign += 1
            n_checked += 1
            continue
        csv_max = _csv_max_turn(fleet_csv)
        if csv_max is None:
            bad.add(stem)
            n_checked += 1
            continue
        # 2-turn slack absorbs any off-by-one between the featurizer's
        # iteration end and the last fleet turn in the replay.
        if csv_max < last_fleet_t - 2:
            bad.add(stem)
        n_checked += 1
    print(
        f"[feature_probe] fleet integrity: checked {n_checked} stems for "
        f"player={player or 'any'}; {len(bad)} corrupted, "
        f"{n_benign} benign-no-late-fleets "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    return bad

# Channel offset of the ships-log-norm value within ``planet_features``:
# layout is [is_comet, 5×owner_one_hot, ships_log, production, ...].
# Defined in ``featurizer/planet_featurizer.py`` :: SCALAR_DIM block.
PLANET_FEATURE_SHIPS_LOG_IDX = 6

# Output ordering — also used as the per-epoch metric key suffix.
# Slot 0 is **learner-relative**: it is the LEARNER'S OWN fleets, not an
# enemy. The "enemy inbound threat" signal we actually care about is the
# sum of slots 1/2/3. We expose both so we can sanity-check that the
# encoder distinguishes own-fleet inbound from enemy-fleet inbound.
LABEL_KEYS: tuple[str, ...] = (
    "garrison",
    "inbound_own",            # learner-relative slot 0
    "inbound_enemy1",         # learner-relative slot 1
    "inbound_enemy2",         # learner-relative slot 2
    "inbound_enemy3",         # learner-relative slot 3
    "inbound_enemy_total",    # sum of slots 1+2+3
)
N_LABELS = len(LABEL_KEYS)


# ---------- Head ----------
class FeatureProbeHead(nn.Module):
    """Per-planet regression head emitting ``N_LABELS`` scalars.

    Input: one ``entity_now`` token per planet (d-dim).
    Output: ``(B, P, N_LABELS)``.

    The MLP is deliberately small — what we're testing is the
    representation quality of ``entity_now``, not the head's capacity.
    A wider head would only help if the information is *there* and
    needs a bigger reader.
    """

    def __init__(
        self,
        d_model: int = 64,
        hidden: int = 64,
        *,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2 (got {num_layers})")
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)

        layers: list[nn.Module] = []
        cur = d_model
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(cur, hidden))
            layers.append(nn.GELU())
            cur = hidden
        layers.append(nn.Linear(cur, N_LABELS))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)

    def forward(self, entity_now: torch.Tensor) -> torch.Tensor:
        # (B, P, d) → (B, P, N_LABELS)
        if entity_now.shape[-1] != self.d_model:
            raise ValueError(
                f"entity_now d={entity_now.shape[-1]} but head built for "
                f"d_model={self.d_model}"
            )
        return self.mlp(entity_now)


# ---------- Stack ----------
class FeatureProbeStack(nn.Module):
    """Fleet + planet + entity encoders + a per-planet regression head.

    No cross-attention; the head reads ``entity_now`` directly.
    """

    def __init__(
        self,
        *,
        fleet_encoder: FleetEncoder,
        planet_encoder: PlanetEncoder,
        entity_encoder: PlanetEntityEncoder,
        head: FeatureProbeHead,
    ):
        super().__init__()
        self.fleet_encoder = fleet_encoder
        self.planet_encoder = planet_encoder
        self.entity_encoder = entity_encoder
        self.head = head

    def unfreeze_all(self) -> None:
        for m in (self.fleet_encoder, self.planet_encoder,
                  self.entity_encoder, self.head):
            m.train()
            for p in m.parameters():
                p.requires_grad_(True)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        entity_tokens, _entity_mask = _entity_tokens_per_step(
            batch, self.fleet_encoder, self.planet_encoder, self.entity_encoder,
        )
        entity_now = (
            entity_tokens[:, -1] if entity_tokens.dim() == 4 else entity_tokens
        )
        return self.head(entity_now)


# ---------- Labels ----------
def _now(t: torch.Tensor) -> torch.Tensor:
    """Take the last time-step slice from an (B, T, …) tensor; pass
    (B, …) tensors through unchanged. Matches the helper used by the
    frozen probe so we get the same label slicing."""
    if t.dim() >= 3 and t.shape[1] == 3:
        return t[:, -1]
    return t


def _extract_labels(batch: dict[str, torch.Tensor]) -> tuple[
    torch.Tensor, torch.Tensor,
]:
    """Return ``(targets, mask)`` aligned with ``entity_now`` (B, P, …).

    targets: (B, P, N_LABELS) float
    mask:    (B, P) bool — real planet slots only
    """
    planet_features = _now(batch["planet_features"])           # (B, P, D_p)
    arr = _now(batch["ships_arriving_within_10"])              # (B, P, 4)
    planet_mask = _now(batch["planet_mask"]).bool()            # (B, P)

    garrison = planet_features[..., PLANET_FEATURE_SHIPS_LOG_IDX]  # (B, P)
    # arr[..., 0] is OWN inbound (learner-relative slot 0). Enemy
    # inbound is slots 1/2/3. Both the per-slot and the aggregate
    # ``enemy_total`` are exposed so post-training analysis can tell
    # apart "encoder confused own vs enemy" from "encoder lost the
    # signal entirely".
    enemy_total = arr[..., 1] + arr[..., 2] + arr[..., 3]
    targets = torch.stack(
        [
            garrison,
            arr[..., 0],   # inbound_own
            arr[..., 1],   # inbound_enemy1
            arr[..., 2],   # inbound_enemy2
            arr[..., 3],   # inbound_enemy3
            enemy_total,   # inbound_enemy_total
        ],
        dim=-1,
    )                                                           # (B, P, 6)
    return targets, planet_mask


def _masked_mse_per_label(
    pred: torch.Tensor,     # (B, P, N_LABELS)
    target: torch.Tensor,   # (B, P, N_LABELS)
    mask: torch.Tensor,     # (B, P) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(per_label_mse, total_loss)`` masked to real planets.

    ``per_label_mse`` is a (N_LABELS,) detached float tensor for the
    log; ``total_loss`` is the scalar to backprop, summed across
    labels and averaged over real-planet rows.
    """
    m = mask.unsqueeze(-1).float()                              # (B, P, 1)
    sq = (pred - target).pow(2) * m
    n_real = m.sum().clamp(min=1.0) * pred.shape[-1]            # real entries × labels
    total_loss = sq.sum() / n_real
    with torch.no_grad():
        denom_per_label = m.sum().clamp(min=1.0)
        per_label_mse = sq.sum(dim=(0, 1)) / denom_per_label    # (N_LABELS,)
    return per_label_mse, total_loss


def _masked_mae_per_label(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Same shape as ``_masked_mse_per_label`` but absolute error."""
    m = mask.unsqueeze(-1).float()
    abs_err = (pred - target).abs() * m
    denom = m.sum().clamp(min=1.0)
    return abs_err.sum(dim=(0, 1)) / denom


# ---------- Encoder loader ----------
def _load_encoders(
    encoder_ckpt: Path,
    *,
    d_model: int,
    device: torch.device,
) -> tuple[FleetEncoder, PlanetEncoder, PlanetEntityEncoder]:
    """Pull fleet/planet/entity weights from an action-stage ckpt.

    Cross-attention is intentionally NOT loaded — this experiment runs
    only the path up to and including the entity encoder.
    """
    ckpt = torch.load(encoder_ckpt, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"unexpected ckpt at {encoder_ckpt}: {type(ckpt)}")
    for k in ("fleet_encoder", "planet_encoder", "entity_encoder"):
        if k not in ckpt:
            raise KeyError(
                f"{encoder_ckpt} missing '{k}' — expected an action-stage ckpt."
            )
    fenc = FleetEncoder(d_model=d_model)
    fenc.load_state_dict(ckpt["fleet_encoder"])
    penc = PlanetEncoder(d_model=d_model)
    penc.load_state_dict(ckpt["planet_encoder"])
    eenc = PlanetEntityEncoder(d_model=d_model)
    eenc.load_state_dict(ckpt["entity_encoder"])
    for m in (fenc, penc, eenc):
        m.to(device)
    return fenc, penc, eenc


# ---------- Baselines ----------
def _compute_baseline_means(
    dataset: ActionSnapshotDataset,
    indices: list[int],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Per-label mean over real planet slots in the train split.

    Used to compute the constant-predictor baseline MAE so each label's
    learning curve has a "trivial floor" line for context.
    """
    # MPS doesn't support float64; use float32 accumulators (the values
    # are small log-normed scalars so float32 precision is fine here).
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)
    label_sum = torch.zeros(N_LABELS, dtype=torch.float32, device=device)
    count = torch.zeros((), dtype=torch.float32, device=device)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets, mask = _extract_labels(batch)
        m = mask.unsqueeze(-1).float()
        label_sum += (targets * m).sum(dim=(0, 1))
        count += m.sum()
    if count.item() == 0:
        return torch.zeros(N_LABELS)
    return (label_sum / count).cpu()


@torch.no_grad()
def _compute_baseline_mae(
    dataset: ActionSnapshotDataset,
    indices: list[int],
    *,
    means: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Per-label MAE of the constant-mean predictor on ``indices``."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)
    abs_sum = torch.zeros(N_LABELS, dtype=torch.float32, device=device)
    count = torch.zeros((), dtype=torch.float32, device=device)
    means = means.to(device)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets, mask = _extract_labels(batch)
        m = mask.unsqueeze(-1).float()
        diff = (targets - means.view(1, 1, -1)).abs() * m
        abs_sum += diff.sum(dim=(0, 1))
        count += m.sum()
    if count.item() == 0:
        return torch.zeros(N_LABELS)
    return (abs_sum / count).cpu()


# ---------- Train loop ----------
def _train_one_epoch(
    stack: FeatureProbeStack,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    stack.train()
    mse_sums = torch.zeros(N_LABELS, dtype=torch.float64)
    mae_sums = torch.zeros(N_LABELS, dtype=torch.float64)
    loss_sum = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets, mask = _extract_labels(batch)
        pred = stack(batch)                                     # (B, P, N_LABELS)
        per_label_mse, total_loss = _masked_mse_per_label(pred, targets, mask)
        per_label_mae = _masked_mae_per_label(pred, targets, mask)
        optim.zero_grad(set_to_none=True)
        total_loss.backward()
        optim.step()
        mse_sums += per_label_mse.detach().cpu().double()
        mae_sums += per_label_mae.detach().cpu().double()
        loss_sum += float(total_loss.item())
        n_batches += 1
    if n_batches == 0:
        return {}
    out = {
        "loss": loss_sum / n_batches,
    }
    for i, key in enumerate(LABEL_KEYS):
        out[f"mse_{key}"] = float(mse_sums[i].item() / n_batches)
        out[f"mae_{key}"] = float(mae_sums[i].item() / n_batches)
    return out


@torch.no_grad()
def _evaluate(
    stack: FeatureProbeStack,
    loader: DataLoader,
    device: torch.device,
    *,
    baseline_mae: torch.Tensor,
) -> dict[str, float]:
    stack.eval()
    mse_sums = torch.zeros(N_LABELS, dtype=torch.float64)
    mae_sums = torch.zeros(N_LABELS, dtype=torch.float64)
    var_sums = torch.zeros(N_LABELS, dtype=torch.float64)        # for r²
    val_label_sum = torch.zeros(N_LABELS, dtype=torch.float64)
    count = torch.zeros((), dtype=torch.float64)
    loss_sum = 0.0
    n_batches = 0
    # Two-pass to get a clean r² (mean over val): first pass for sums,
    # second pass for variance. But for a per-epoch metric this would
    # double eval time. Instead we use an online-stable approximation:
    # accumulate squared error sum + label-mean from the same pass and
    # compute r² = 1 - mse / var_against_val_mean.
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets, mask = _extract_labels(batch)
        pred = stack(batch)
        per_label_mse, total_loss = _masked_mse_per_label(pred, targets, mask)
        per_label_mae = _masked_mae_per_label(pred, targets, mask)
        m = mask.unsqueeze(-1).float()
        mse_sums += per_label_mse.detach().cpu().double()
        mae_sums += per_label_mae.detach().cpu().double()
        val_label_sum += (targets * m).sum(dim=(0, 1)).cpu().double()
        var_sums += ((targets ** 2) * m).sum(dim=(0, 1)).cpu().double()
        count += m.sum().cpu().double()
        loss_sum += float(total_loss.item())
        n_batches += 1
    if n_batches == 0:
        return {}
    # E[X²] - E[X]² over all real-planet val entries.
    mean_y = val_label_sum / count.clamp(min=1.0)
    var_y = (var_sums / count.clamp(min=1.0)) - mean_y * mean_y
    mse_per_label = mse_sums / n_batches                        # MSE averaged per batch
    mae_per_label = mae_sums / n_batches
    out: dict[str, float] = {"loss": loss_sum / n_batches}
    for i, key in enumerate(LABEL_KEYS):
        out[f"mse_{key}"] = float(mse_per_label[i].item())
        out[f"mae_{key}"] = float(mae_per_label[i].item())
        out[f"baseline_mae_{key}"] = float(baseline_mae[i].item())
        var_i = float(var_y[i].item())
        # r²: average per-snapshot MSE compared to overall val variance.
        out[f"r2_{key}"] = 1.0 - (float(mse_per_label[i].item()) / max(var_i, 1e-9))
        out[f"rel_mae_{key}"] = (
            float(mae_per_label[i].item()) / max(float(baseline_mae[i].item()), 1e-9)
        )
    return out


# ---------- Entry points ----------
def train_feature_probe(
    args: argparse.Namespace,
    *,
    dataset: ActionSnapshotDataset | None = None,
) -> Path:
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fenc, penc, eenc = _load_encoders(
        Path(args.encoder_ckpt), d_model=args.d_model, device=device,
    )

    if dataset is None:
        # Stems whose fleet CSV stopped short of the replay — building
        # over them would feed empty fleet inputs against the still-
        # nonzero entity-side inbound labels. The exclude set is empty
        # by default; opt in with --require-complete-fleet.
        exclude: set[str] = set()
        if getattr(args, "require_complete_fleet", False):
            exclude = _incomplete_fleet_stems(
                args.player, Path(args.fleet_dir), Path(args.replay_dir),
            )
            if exclude:
                preview = ", ".join(sorted(exclude)[:3])
                more = "" if len(exclude) <= 3 else f" (+{len(exclude) - 3} more)"
                print(
                    f"[feature_probe] excluding {len(exclude)} incomplete-"
                    f"fleet stems: {preview}{more}",
                    flush=True,
                )
        cache_dir = getattr(args, "cache_dir", None)
        rebuild_cache = bool(getattr(args, "rebuild_cache", False))
        dataset = prepare_dataset(
            player=args.player,
            filter_mode=args.filter,
            action_dir=args.action_dir,
            planet_dir=args.planet_dir,
            fleet_dir=args.fleet_dir,
            entity_dir=args.entity_dir,
            cross_entity_dir=args.cross_entity_dir,
            replay_dir=args.replay_dir,
            max_planets=args.max_planets,
            max_fleets=args.max_fleets,
            n_history=args.n_history,
            cache_dir=cache_dir,
            rebuild_cache=rebuild_cache,
            exclude_stems=exclude or None,
        )
    else:
        print(
            f"[feature_probe] using caller-provided dataset "
            f"({len(dataset)} snapshots)",
            flush=True,
        )

    n = len(dataset)
    # Every snapshot carries valid per-planet labels (garrison + inbound
    # are computed every turn) — we do NOT filter to acted rows.
    all_idx = list(range(n))
    if args.max_rows is not None:
        all_idx = all_idx[: args.max_rows]
        print(f"[feature_probe] capped to {len(all_idx)} snapshots", flush=True)

    if args.overfit:
        train_idx = all_idx
        val_idx = all_idx
    else:
        n_val = max(1, int(round(len(all_idx) * args.val_frac)))
        train_idx = all_idx[:-n_val]
        val_idx = all_idx[-n_val:]
    print(
        f"[feature_probe] train={len(train_idx)} val={len(val_idx)}",
        flush=True,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
    )

    # Train-split label means → val-split baseline MAE.
    t_b = time.time()
    print("[feature_probe] computing constant-predictor baseline ...", flush=True)
    means = _compute_baseline_means(
        dataset, train_idx, device=device, batch_size=args.batch_size,
    )
    baseline_mae = _compute_baseline_mae(
        dataset, val_idx, means=means,
        device=device, batch_size=args.batch_size,
    )
    print(
        "[feature_probe] baseline MAE per label: "
        + ", ".join(f"{k}={baseline_mae[i]:.4f}"
                    for i, k in enumerate(LABEL_KEYS))
        + f"  ({time.time() - t_b:.1f}s)",
        flush=True,
    )

    head = FeatureProbeHead(
        d_model=args.d_model, hidden=args.head_hidden,
        num_layers=args.head_num_layers,
    ).to(device)
    stack = FeatureProbeStack(
        fleet_encoder=fenc, planet_encoder=penc,
        entity_encoder=eenc, head=head,
    ).to(device)
    stack.unfreeze_all()
    trainable = [p for p in stack.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[feature_probe] trainable params: {n_params:,}", flush=True)
    optim = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_path = out_dir / "feature_probe_best.pt"
    last_path = out_dir / "feature_probe_last.pt"
    log_path = out_dir / "log.json"
    log_entries: list[dict[str, Any]] = []
    t0 = time.time()

    config = {
        "d_model": args.d_model,
        "head_hidden": args.head_hidden,
        "head_num_layers": args.head_num_layers,
        "max_planets": args.max_planets,
        "max_fleets": args.max_fleets,
        "n_history": args.n_history,
        "player": args.player,
        "filter": args.filter,
        "label_keys": list(LABEL_KEYS),
        "baseline_mae": {k: float(baseline_mae[i].item())
                          for i, k in enumerate(LABEL_KEYS)},
        "train_means": {k: float(means[i].item())
                         for i, k in enumerate(LABEL_KEYS)},
    }

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        tr = _train_one_epoch(stack, train_loader, optim, device)
        va = _evaluate(stack, val_loader, device, baseline_mae=baseline_mae)
        elapsed = time.time() - t0
        log = {
            "epoch": epoch,
            "elapsed_s": round(elapsed, 1),
            "epoch_s": round(time.time() - t_epoch, 1),
            "train": tr,
            "val": va,
        }
        log_entries.append(log)
        log_path.write_text(json.dumps(log_entries, indent=2))

        # Compact one-line summary: train MAE per label + val MAE per
        # label + r² per label. ``rel`` = val_mae / baseline_mae.
        def _fmt(prefix: str, metrics: dict[str, float]) -> str:
            return " ".join(
                f"{k}={metrics.get(f'{prefix}{k}', 0):.3f}"
                for k in LABEL_KEYS
            )
        print(
            f"[feature_probe] ep={epoch:3d} "
            f"tr_loss={tr.get('loss', 0):.5f} "
            f"val_loss={va.get('loss', 0):.5f}  |  "
            f"val MAE: {_fmt('mae_', va)}  |  "
            f"val rel: {_fmt('rel_mae_', va)}  |  "
            f"r²: {_fmt('r2_', va)}  "
            f"dt={time.time() - t_epoch:.1f}s",
            flush=True,
        )

        payload = {
            "epoch": epoch,
            "encoder_ckpt": str(args.encoder_ckpt),
            "config": config,
            "metrics": {"train": tr, "val": va},
            "fleet_encoder": stack.fleet_encoder.state_dict(),
            "planet_encoder": stack.planet_encoder.state_dict(),
            "entity_encoder": stack.entity_encoder.state_dict(),
            "feature_probe_head": stack.head.state_dict(),
        }
        torch.save(payload, last_path)
        val_loss = va.get("loss", float("inf"))
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(torch.load(last_path, weights_only=False), best_path)

    print(
        f"[feature_probe] done. best_val_loss={best_val_loss:.5f} "
        f"ckpts: {best_path.name}, {last_path.name}",
        flush=True,
    )
    return best_path


def train_feature_probe_kwargs(
    *,
    encoder_ckpt,
    out_dir,
    action_dir=ACTION_DATASET_DIR,
    planet_dir=PLANET_DATASET_DIR,
    fleet_dir=FLEET_DATASET_DIR,
    entity_dir=ENTITY_DATASET_DIR,
    cross_entity_dir=CROSS_ENTITY_DATASET_DIR,
    filter="all",
    player=None,
    replay_dir="data/replays",
    max_rows=None,
    overfit=False,
    val_frac=0.2,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.0,
    epochs=8,
    d_model=64,
    head_hidden=64,
    head_num_layers=2,
    max_planets=64,
    max_fleets=1024,
    n_history=3,
    device=None,
    cache_dir=None,
    rebuild_cache=False,
    dataset=None,
    require_complete_fleet=False,
) -> Path:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args = argparse.Namespace(
        encoder_ckpt=Path(encoder_ckpt),
        action_dir=Path(action_dir),
        planet_dir=Path(planet_dir),
        fleet_dir=Path(fleet_dir),
        entity_dir=Path(entity_dir),
        cross_entity_dir=Path(cross_entity_dir),
        filter=filter,
        player=player,
        replay_dir=Path(replay_dir),
        max_rows=max_rows,
        overfit=overfit,
        val_frac=val_frac,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        d_model=d_model,
        head_hidden=head_hidden,
        head_num_layers=head_num_layers,
        max_planets=max_planets,
        max_fleets=max_fleets,
        n_history=n_history,
        device=device,
        require_complete_fleet=bool(require_complete_fleet),
        out_dir=Path(out_dir),
        cache_dir=(Path(cache_dir) if cache_dir is not None else None),
        rebuild_cache=bool(rebuild_cache),
    )
    return train_feature_probe(args, dataset=dataset)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder-ckpt", type=Path, required=True,
                   help="Path to an action-stage ckpt with fleet/planet/entity "
                        "state dicts (e.g. action_best.pt).")
    p.add_argument("--action-dir", type=Path, default=ACTION_DATASET_DIR)
    p.add_argument("--planet-dir", type=Path, default=PLANET_DATASET_DIR)
    p.add_argument("--fleet-dir", type=Path, default=FLEET_DATASET_DIR)
    p.add_argument("--entity-dir", type=Path, default=ENTITY_DATASET_DIR)
    p.add_argument("--cross-entity-dir", type=Path, default=CROSS_ENTITY_DATASET_DIR)
    p.add_argument("--filter", choices=["winner", "all"], default="all")
    p.add_argument("--player", default=None)
    p.add_argument("--replay-dir", type=Path, default=Path("data/replays"))
    p.add_argument("--require-complete-fleet", action="store_true",
                   help="Skip stems whose fleet_<stem>.csv stopped short of "
                        "the replay's last few turns (interrupted writes). "
                        "Prevents training inbound labels against empty "
                        "fleet inputs. See the build script's --audit-only "
                        "for the precise integrity criterion.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap on snapshots (after dataset materialization). "
                        "Every snapshot carries valid per-planet labels, so "
                        "this isn't restricted to acted rows.")
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--head-num-layers", type=int, default=2)
    p.add_argument("--max-planets", type=int, default=64)
    p.add_argument("--max-fleets", type=int, default=1024)
    p.add_argument("--n-history", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()
    train_feature_probe(args)


if __name__ == "__main__":
    main()
