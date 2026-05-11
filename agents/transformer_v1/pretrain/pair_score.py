"""Encoder-freeze + single PairScoreHead smoke test.

Goal of this stage: answer one question.

    Can the current frozen encoder representation support direct expert
    ``(source, target)`` pair prediction from replay?

If the answer is "yes" (val pair top-1 clearly beats random-valid), we
layer NOOP / frac / value / PPO back on top in subsequent stages. If
"no" (the head can't even overfit a tiny subset), we know to fix the
encoder or the labels/masking before re-trying any policy work.

This file is intentionally minimal:

  * one trainable module — :class:`PairScoreHead` (a 2-layer MLP).
  * encoders frozen via ``requires_grad_(False)`` and ``.eval()``.
  * one loss — joint cross-entropy on flattened ``(P×P)`` pair logits.
  * one expert — ``--filter winner`` keeps only rows where the CSV's
    learner-slot matches the replay's ``winner_seat`` (proxy for "one
    strong expert" in the absence of per-replay agent metadata).
  * one dataset class reused — :class:`ActionSnapshotDataset` already
    emits ``source_planet_idx`` / ``target_planet_idx`` /
    ``src_valid`` / ``tgt_valid`` per snapshot.

Run from the repo root:

    python -m agents.transformer_v1.pretrain.pair_score \\
        --encoder-ckpt data/runs/action/<run>/action_best.pt \\
        --filter winner --max-rows 50 --overfit \\
        --epochs 200 --device cpu \\
        --out-dir data/runs/pair_score/$(date +%Y%m%d-%H%M%S)

For the small-real split (Experiment 2):

    python -m agents.transformer_v1.pretrain.pair_score \\
        --encoder-ckpt data/runs/action/<run>/action_best.pt \\
        --filter winner --max-rows 5000 \\
        --batch-size 64 --epochs 10 --lr 1e-3 --device cuda \\
        --out-dir data/runs/pair_score/$(date +%Y%m%d-%H%M%S)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..aggregator import CrossEntityAttention
from ..encoder.entity_encoder import PlanetEntityEncoder
from ..encoder.fleet_encoder import FleetEncoder
from ..encoder.planet_encoder import PlanetEncoder
from ..paths import (
    ACTION_DATASET_DIR,
    CROSS_ENTITY_DATASET_DIR,
    ENTITY_DATASET_DIR,
    FLEET_DATASET_DIR,
    PLANET_DATASET_DIR,
)
from .cross_entity import _entity_tokens_per_step
from .expert_action import ActionSnapshotDataset


# ---------- Model ----------
class PairScoreHead(nn.Module):
    """One MLP scoring every ``(source_i, target_j)`` pair.

    Pair feature per (i, j):
        h_ij = [ glob ‖ ctx_i ‖ ctx_j ‖ ctx_i ⊙ ctx_j ]   (4·d)

    Score:
        s_ij = MLP(h_ij)                                   (1)

    Output ``pair_logits`` has shape ``(B, P, P)``; the ``[i, j]`` cell
    is the pre-softmax score for "expert launches from i, aiming at j".
    Invalid pairs (per ``src_valid × tgt_valid``) are masked to ``-inf``.
    """

    def __init__(self, d_model: int = 64, hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)

    def forward(
        self,
        glob: torch.Tensor,                          # (B, d)
        ctx: torch.Tensor,                           # (B, P, d)
        src_valid: torch.Tensor | None = None,        # (B, P) bool
        tgt_valid: torch.Tensor | None = None,        # (B, P) bool
    ) -> torch.Tensor:                                # (B, P, P)
        B, P, d = ctx.shape
        if d != self.d_model:
            raise ValueError(
                f"ctx d={d} but head built for d_model={self.d_model}"
            )
        glob_b = glob.view(B, 1, 1, d).expand(B, P, P, d)
        src = ctx.unsqueeze(2).expand(B, P, P, d)
        tgt = ctx.unsqueeze(1).expand(B, P, P, d)
        had = src * tgt
        feat = torch.cat([glob_b, src, tgt, had], dim=-1)        # (B,P,P,4d)
        scores = self.mlp(feat).squeeze(-1)                       # (B,P,P)
        if src_valid is not None and tgt_valid is not None:
            pair_valid = src_valid.unsqueeze(2) & tgt_valid.unsqueeze(1)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~pair_valid, neg_inf)
        return scores


# Inference-side clamp on ``frac_log_std`` (matches the deleted PPO
# decoder's contract — see runner.py dead-code path). Training-side
# clamps are applied via ``torch.clamp`` in the loss / metric paths so
# the parameter is free to drift slightly outside before the next
# update; ``FRAC_LOG_STD_MIN``/``MAX`` correspond to σ ∈ [0.30, 1.0].
FRAC_LOG_STD_MIN: float = math.log(0.30)
FRAC_LOG_STD_MAX: float = math.log(1.0)
FRAC_LOG_STD_INIT: float = math.log(0.5)
FRAC_LABEL_EPS: float = 1e-3


class FracHead(nn.Module):
    """Per-``(source, target)`` launch-fraction predictor.

    Shares the pair-feature shape with :class:`PairScoreHead` but is its
    own module so existing ``pair_score_best.pt`` files (which only
    carry ``pair_score_head`` state) load cleanly via ``--init-from``
    — a fresh ``FracHead`` initialises from scratch on the first frac
    run and gets saved alongside ``pair_score_head`` on subsequent ones.

    Pair feature per ``(i, j)``:
        h_ij = [ glob ‖ ctx_i ‖ ctx_j ‖ ctx_i ⊙ ctx_j ]   (4·d)

    Output dict:
        frac_loc      (B, P, P)   — predicted mean of ``z = logit(frac)``
        frac_log_std  scalar      — Normal log-std shared across pairs

    Loss-side: sparse — supervise only at the ground-truth pair index.
    Inference-side: deterministic = ``sigmoid(clamp(loc, min=FRAC_Z_MIN))``,
    stochastic = truncated-Normal inverse-CDF at ``FRAC_Z_MIN`` (see
    ``agents/transformer_v1/runner.py``'s dead-code path for the
    reference implementation).
    """

    def __init__(self, d_model: int = 64, hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        # Global log-std for the per-pair Normal. Init log(0.5) so σ in
        # z space is ~0.5, which maps to a ~12% spread in frac space.
        self.frac_log_std = nn.Parameter(torch.tensor(FRAC_LOG_STD_INIT))

    def forward(
        self,
        glob: torch.Tensor,                  # (B, d)
        ctx: torch.Tensor,                   # (B, P, d)
    ) -> dict[str, torch.Tensor]:
        B, P, d = ctx.shape
        if d != self.d_model:
            raise ValueError(
                f"ctx d={d} but FracHead built for d_model={self.d_model}"
            )
        glob_b = glob.view(B, 1, 1, d).expand(B, P, P, d)
        src = ctx.unsqueeze(2).expand(B, P, P, d)
        tgt = ctx.unsqueeze(1).expand(B, P, P, d)
        had = src * tgt
        feat = torch.cat([glob_b, src, tgt, had], dim=-1)         # (B,P,P,4d)
        loc = self.mlp(feat).squeeze(-1)                           # (B,P,P)
        return {"frac_loc": loc, "frac_log_std": self.frac_log_std}


class PairScoreStack(nn.Module):
    """Frozen encoders + trainable :class:`PairScoreHead`.

    Forward expects the same batch dict as :class:`ActionSnapshotDataset`
    emits — `_entity_tokens_per_step` handles the (B,T,P,...) history
    layout transparently.
    """

    def __init__(
        self,
        *,
        fleet_encoder: FleetEncoder,
        planet_encoder: PlanetEncoder,
        entity_encoder: PlanetEntityEncoder,
        cross: CrossEntityAttention,
        pair_score_head: PairScoreHead,
        frac_head: FracHead | None = None,
    ):
        super().__init__()
        self.fleet_encoder = fleet_encoder
        self.planet_encoder = planet_encoder
        self.entity_encoder = entity_encoder
        self.cross = cross
        self.pair_score_head = pair_score_head
        # FracHead is optional: when ``--frac-weight 0`` (default) the
        # head is never built and the stack remains byte-identical to
        # the pre-frac contract. When non-None, it adds ~33k params and
        # contributes ``frac_loc``/``frac_log_std`` to ``forward()``.
        self.frac_head = frac_head

    # Encoder modules accessible by short name. Keep this as the
    # authoritative ordering — CLI arg parsing + checkpoint save/load
    # walk the same names.
    ENCODER_MODULES: tuple[str, ...] = (
        "fleet_encoder",
        "planet_encoder",
        "entity_encoder",
        "cross",
    )

    def freeze_encoders(self) -> None:
        """Backwards-compat alias: freeze every encoder."""
        self.set_freeze_state(unfrozen=())

    def set_freeze_state(
        self,
        unfrozen: tuple[str, ...] | list[str] | set[str],
        *,
        freeze_pair_head: bool = False,
    ) -> None:
        """Set each encoder's train/grad mode based on whether its name
        appears in ``unfrozen``. The frac head (when present) is always
        trainable; the pair head is trainable by default but can be
        frozen via ``freeze_pair_head=True``.

        ``unfrozen`` may contain ``"cross"``, ``"entity"`` (= entity_encoder),
        ``"planet"`` (= planet_encoder), or ``"fleet"`` (= fleet_encoder).
        Anything else raises.

        Stage-2 frac-only training pattern: pass ``freeze_pair_head=True``
        + ``unfrozen=()`` so only the FracHead trains. The pair head's
        already-tuned calibration is preserved bit-exact, and the
        encoders' representation stays fixed too.
        """
        canon = self._canonicalize(unfrozen)
        for name in self.ENCODER_MODULES:
            module = getattr(self, name)
            if name in canon:
                module.train()
                for p in module.parameters():
                    p.requires_grad_(True)
            else:
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)
        # Pair head: trainable by default, frozen on opt-in.
        if freeze_pair_head:
            self.pair_score_head.eval()
            for p in self.pair_score_head.parameters():
                p.requires_grad_(False)
        else:
            self.pair_score_head.train()
            for p in self.pair_score_head.parameters():
                p.requires_grad_(True)
        # Frac head (when present) is always trainable — that's the
        # whole point of building it.
        if self.frac_head is not None:
            self.frac_head.train()
            for p in self.frac_head.parameters():
                p.requires_grad_(True)

    @classmethod
    def _canonicalize(cls, names) -> set[str]:
        """Map user-facing aliases (e.g. ``entity`` → ``entity_encoder``)
        and validate against ``ENCODER_MODULES``.
        """
        aliases = {
            "fleet": "fleet_encoder",
            "planet": "planet_encoder",
            "entity": "entity_encoder",
            "cross": "cross",
        }
        out: set[str] = set()
        for n in names or ():
            n = n.strip()
            if not n:
                continue
            if n in cls.ENCODER_MODULES:
                out.add(n)
            elif n in aliases:
                out.add(aliases[n])
            elif n in ("head", "pair_score_head"):
                # Always trainable; silently accepted.
                continue
            else:
                raise ValueError(
                    f"unknown unfreeze target {n!r}. "
                    f"valid: {sorted(set(aliases) | set(cls.ENCODER_MODULES))}"
                )
        return out

    def trainable_module_state(self, unfrozen: set[str]) -> dict[str, dict]:
        """Collect state-dicts for the pair head + (when present) the
        frac head + **every** encoder (frozen or not).

        Self-contained checkpoint: downstream consumers (PPO, inference,
        a next stage of pair_score) should only need this one file to
        reconstruct the stack — no second hop through the original
        ``--encoder-ckpt`` action_best.pt. Encoder states are small
        (~700 KB total) so always saving them keeps the ckpt under
        ~2 MB even with the frac head.

        ``unfrozen`` is accepted for backwards compat but no longer
        gates saving — the parameter is kept so the train loop's call
        site doesn't need to change.
        """
        out: dict[str, dict] = {"pair_score_head": self.pair_score_head.state_dict()}
        if self.frac_head is not None:
            out["frac_head"] = self.frac_head.state_dict()
        for name in self.ENCODER_MODULES:
            out[name] = getattr(self, name).state_dict()
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch,
            self.fleet_encoder,
            self.planet_encoder,
            self.entity_encoder,
        )
        ctx, glob = self.cross(entity_tokens, entity_mask)
        # Single-step input is shape (B, P, d); multi-step is (B, T, P, d).
        ctx_now = ctx[:, -1] if ctx.dim() == 4 else ctx
        if entity_mask.dim() == 3:
            mask_now = entity_mask[:, -1]
        else:
            mask_now = entity_mask

        P = ctx_now.shape[1]
        src_valid = batch.get("src_valid")
        tgt_valid = batch.get("tgt_valid")
        if src_valid is None:
            src_valid = mask_now
        if tgt_valid is None:
            tgt_valid = mask_now
        # Dataset masks may have been allocated wider than the encoder's
        # planet axis; clip so shapes match.
        src_valid = src_valid[..., :P].bool().clone()
        tgt_valid = tgt_valid[..., :P].bool().clone()
        mask_now = mask_now[..., :P].bool()

        # Older action CSV packs do not have the optional `_masks/*.npz`
        # side cache, which leaves src/tgt masks all-False. Fall back to
        # the current real-planet mask for those rows so pair CE remains
        # trainable, then force-include the supervised pair labels.
        if mask_now.shape == src_valid.shape:
            src_empty = ~src_valid.any(dim=-1)
            tgt_empty = ~tgt_valid.any(dim=-1)
            fallback = src_empty | tgt_empty
            src_valid[fallback] = mask_now[fallback]
            tgt_valid[fallback] = mask_now[fallback]

        src_idx = batch.get("source_planet_idx")
        tgt_idx = batch.get("target_planet_idx")
        if src_idx is not None:
            src_idx = src_idx.to(src_valid.device).long()
            rows = torch.nonzero((src_idx >= 0) & (src_idx < P), as_tuple=True)[0]
            if rows.numel() > 0:
                src_valid[rows, src_idx[rows]] = True
        if tgt_idx is not None:
            tgt_idx = tgt_idx.to(tgt_valid.device).long()
            rows = torch.nonzero((tgt_idx >= 0) & (tgt_idx < P), as_tuple=True)[0]
            if rows.numel() > 0:
                tgt_valid[rows, tgt_idx[rows]] = True

        pair_logits = self.pair_score_head(glob, ctx_now, src_valid, tgt_valid)
        out: dict[str, torch.Tensor] = {
            "pair_logits": pair_logits,
            "_ctx_now": ctx_now,
            "_glob": glob,
        }
        if self.frac_head is not None:
            frac_out = self.frac_head(glob, ctx_now)
            out["frac_loc"] = frac_out["frac_loc"]
            out["frac_log_std"] = frac_out["frac_log_std"]
        return out


# ---------- Loss + metrics ----------
def compute_pair_score_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint pair CE on rows where the expert acted.

    Returns (loss, metrics) where metrics contains top-1/3/5 accuracies,
    induced source/target accuracy, and the count of valid rows.
    """
    pair_logits = preds["pair_logits"]                               # (B, P, P)
    src_idx = batch["source_planet_idx"].to(pair_logits.device)       # (B,) Long
    tgt_idx = batch["target_planet_idx"].to(pair_logits.device)       # (B,)
    B, P, _ = pair_logits.shape

    valid = (src_idx >= 0) & (tgt_idx >= 0) & (src_idx < P) & (tgt_idx < P)
    n_valid = int(valid.sum().item())
    metrics: dict[str, float] = {
        "top1": 0.0, "top3": 0.0, "top5": 0.0,
        "src_top1": 0.0, "tgt_top1": 0.0,
        "n_valid": float(n_valid),
    }
    if n_valid == 0:
        # Zero-grad sentinel — keeps the optimizer step a no-op without
        # branching the train loop.
        return pair_logits.sum() * 0.0, metrics

    pl = pair_logits[valid]                                          # (Nv, P, P)
    si = src_idx[valid]
    ti = tgt_idx[valid]
    flat = pl.reshape(pl.shape[0], P * P)                            # (Nv, P*P)
    y = si * P + ti                                                  # (Nv,)
    loss = F.cross_entropy(flat, y)

    with torch.no_grad():
        k = min(5, P * P)
        top_ids = flat.topk(k, dim=-1).indices                       # (Nv, k)
        match = top_ids == y.unsqueeze(-1)
        metrics["top1"] = float(match[:, 0].float().mean().item())
        metrics["top3"] = float(match[:, : min(3, k)].any(-1).float().mean().item())
        metrics["top5"] = float(match.any(-1).float().mean().item())
        pred = flat.argmax(-1)
        metrics["src_top1"] = float((pred // P == si).float().mean().item())
        metrics["tgt_top1"] = float((pred % P == ti).float().mean().item())
    return loss, metrics


def compute_frac_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    frac_baseline_mae: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normal NLL on ``z = logit(frac_label)``, sparse over acted rows.

    Supervision is dense in ``frac_loc`` only at inference (the agent
    samples the cell of its chosen ``(src, tgt)``). At loss time, only
    the expert's actual ``(src_idx, tgt_idx)`` cell contributes gradient
    — gathering one prediction per acted row keeps the optimizer from
    confusing the head about pairs the expert didn't take.

    Returns ``(nll, metrics)``. Metrics keys (always present so the
    per-epoch summary doesn't branch):

      * ``frac_loss``         — the NLL (== ``nll.item()``)
      * ``frac_mae``          — mean ``|sigmoid(loc) - frac_label|`` in frac space
      * ``frac_baseline_mae`` — passes the train-mean baseline through (or NaN)
      * ``frac_sigma``        — ``exp(frac_log_std)`` after the inference clamp
      * ``n_frac_valid``      — number of supervised rows this batch
    """
    frac_loc = preds.get("frac_loc")
    frac_log_std = preds.get("frac_log_std")
    if frac_loc is None or frac_log_std is None:
        # Frac head not installed — nothing to do. Return a zero-grad
        # sentinel so the caller can still sum it into the total loss.
        zero = next(iter(preds.values())).sum() * 0.0
        return zero, {
            "frac_loss": 0.0,
            "frac_mae": 0.0,
            "frac_baseline_mae": (
                float(frac_baseline_mae) if frac_baseline_mae is not None else float("nan")
            ),
            "frac_sigma": 0.0,
            "n_frac_valid": 0.0,
        }

    device = frac_loc.device
    src_idx = batch["source_planet_idx"].to(device).long()
    tgt_idx = batch["target_planet_idx"].to(device).long()
    frac_label = batch["frac_label"].to(device).float()
    B, P, _ = frac_loc.shape

    valid = (
        (src_idx >= 0) & (tgt_idx >= 0)
        & (src_idx < P) & (tgt_idx < P)
        & torch.isfinite(frac_label)
        & (frac_label > 0.0)
    )
    n_valid = int(valid.sum().item())

    # Inference-side σ clamp (mirrors the deleted PPO decoder's contract).
    sigma_disp = float(
        frac_log_std.detach()
        .clamp(min=FRAC_LOG_STD_MIN, max=FRAC_LOG_STD_MAX)
        .exp()
        .item()
    )
    metrics: dict[str, float] = {
        "frac_loss": 0.0,
        "frac_mae": 0.0,
        "frac_baseline_mae": (
            float(frac_baseline_mae) if frac_baseline_mae is not None else float("nan")
        ),
        "frac_sigma": sigma_disp,
        "n_frac_valid": float(n_valid),
    }
    if n_valid == 0:
        zero = frac_loc.sum() * 0.0 + frac_log_std * 0.0
        return zero, metrics

    rows = torch.nonzero(valid, as_tuple=True)[0]
    loc_at = frac_loc[rows, src_idx[rows], tgt_idx[rows]]              # (n_valid,)

    f_clamped = frac_label[rows].clamp(min=FRAC_LABEL_EPS, max=1.0 - FRAC_LABEL_EPS)
    z_target = torch.log(f_clamped / (1.0 - f_clamped))

    # Differentiable σ — clamp via ``torch.clamp`` (gradient passes through
    # the active range). At training the optimizer can still drift the
    # underlying log-std slightly outside; the inference-side display uses
    # ``.detach()`` to read a static σ.
    sigma = torch.clamp(frac_log_std, min=FRAC_LOG_STD_MIN, max=FRAC_LOG_STD_MAX).exp()
    dist = torch.distributions.Normal(loc=loc_at, scale=sigma)
    nll = -dist.log_prob(z_target).mean()

    with torch.no_grad():
        frac_pred = torch.sigmoid(loc_at)
        metrics["frac_loss"] = float(nll.item())
        metrics["frac_mae"] = float((frac_pred - frac_label[rows]).abs().mean().item())
    return nll, metrics


def compute_frac_baseline_mae(
    dataset: ActionSnapshotDataset,
    indices,
) -> float:
    """Constant-predictor MAE: ``mean(|frac_global_mean - frac_label|)`` over
    every acted snapshot in ``indices``. Computed once (cheap, ≪1 s) so the
    per-epoch summary can display a fixed "trivial predictor" floor.
    """
    fracs: list[float] = []
    for i in indices:
        snap = dataset.snapshots[i]
        acted = float(snap["expert_acted"].item())
        if acted <= 0.5:
            continue
        f = float(snap["frac_label"].item())
        if not math.isfinite(f) or f <= 0.0:
            continue
        fracs.append(f)
    if not fracs:
        return float("nan")
    arr = torch.tensor(fracs)
    mean = arr.mean()
    return float((arr - mean).abs().mean().item())


@torch.no_grad()
def random_valid_baseline(
    batch: dict[str, torch.Tensor], device: torch.device,
) -> float:
    """Top-1 accuracy of picking a uniformly-random valid pair.

    Computed once per batch as ``1 / n_valid_pairs`` averaged over rows.
    """
    src_idx = batch["source_planet_idx"].to(device)
    tgt_idx = batch["target_planet_idx"].to(device)
    src_valid = batch["src_valid"].to(device).bool()
    tgt_valid = batch["tgt_valid"].to(device).bool()
    valid = (src_idx >= 0) & (tgt_idx >= 0)
    if not valid.any():
        return 0.0
    pair_valid = src_valid.unsqueeze(2) & tgt_valid.unsqueeze(1)     # (B,P,P)
    n_pairs = pair_valid[valid].reshape(int(valid.sum()), -1).sum(-1).clamp(min=1)
    return float((1.0 / n_pairs.float()).mean().item())


# ---------- Encoder-only ckpt loader ----------
def load_frozen_encoder_stack(
    ckpt_path: str | Path,
    *,
    d_model: int = 64,
    device: str = "cpu",
) -> tuple[FleetEncoder, PlanetEncoder, PlanetEntityEncoder, CrossEntityAttention]:
    """Build empty encoders and load only the encoder + cross weights
    from a previously-saved action checkpoint.

    The pre-deletion ``_save_checkpoint`` wrote keys
    ``cross``, ``fleet_encoder``, ``planet_encoder``, ``entity_encoder``,
    ``action_decoder``, ``global_decoder``. This loader takes the first
    four and ignores the rest.
    """
    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"unexpected ckpt format at {ckpt_path}: {type(ckpt)}")
    for k in ("fleet_encoder", "planet_encoder", "entity_encoder", "cross"):
        if k not in ckpt:
            raise KeyError(
                f"{ckpt_path} is missing '{k}' state-dict key — "
                "this loader expects an action-stage checkpoint."
            )

    fenc = FleetEncoder(d_model=d_model)
    fenc.load_state_dict(ckpt["fleet_encoder"])
    penc = PlanetEncoder(d_model=d_model)
    penc.load_state_dict(ckpt["planet_encoder"])
    eenc = PlanetEntityEncoder(d_model=d_model)
    eenc.load_state_dict(ckpt["entity_encoder"])
    cross = CrossEntityAttention(d_model=d_model)
    cross.load_state_dict(ckpt["cross"])

    for m in (fenc, penc, eenc, cross):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return fenc, penc, eenc, cross


# ---------- Dataset helpers ----------
def _csv_winner_slot_match(action_csv_path: Path) -> bool:
    """``True`` iff the CSV's learner-slot equals the replay's
    ``winner_seat`` for the first row.

    Filename convention: ``action_<replay>_<num_players>_<learner_slot>.csv``.
    Each CSV's rows share the same ``learner_slot``, so reading any row
    gives the answer.
    """
    try:
        learner_slot = int(action_csv_path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return False
    try:
        with action_csv_path.open() as fh:
            row = next(csv.DictReader(fh), None)
    except OSError:
        return False
    if row is None:
        return False
    try:
        winner_seat = int(row["winner_seat"])
    except (KeyError, ValueError):
        return False
    return winner_seat == learner_slot


def player_replay_stems(
    replay_dir: Path,
    player: str,
) -> set[str]:
    """Return the set of replay-stem strings (e.g. ``75365996_2_0``) under
    ``<replay_dir>/<player>/``.

    Layout is ``data/replays/<player>/<replay_id>_<num_players>_<seat>.json.gz``.
    The directory name is the player whose perspective the replay is from;
    the replay JSON's ``info.TeamNames[seat]`` matches that player.

    Action CSV filenames share the same stem with an ``action_`` prefix,
    so the returned set can be used to filter
    :func:`discover_action_csvs` outputs to one player.
    """
    pdir = Path(replay_dir) / player
    if not pdir.is_dir():
        raise FileNotFoundError(
            f"no replay directory for player={player!r} at {pdir}"
        )
    stems: set[str] = set()
    for p in pdir.iterdir():
        if not p.is_file():
            continue
        # ``75365996_2_0.json.gz`` → ``75365996_2_0``
        name = p.name
        if name.endswith(".json.gz"):
            stems.add(name[: -len(".json.gz")])
        elif name.endswith(".json"):
            stems.add(name[: -len(".json")])
    return stems


def discover_action_csvs(
    action_dir: Path,
    *,
    filter_mode: str,                  # "winner" or "all"
    player: str | None = None,
    replay_dir: Path | None = None,
) -> list[Path]:
    """Find action CSVs, optionally restricted to one player and/or
    winner-only perspectives.

    ``filter_mode``:
      * ``all``    — every CSV under ``action_dir``.
      * ``winner`` — keep only CSVs whose learner-slot equals
                     ``winner_seat`` (proxy for "perspective player won").

    ``player`` (optional): restrict to CSVs whose stem matches a replay
    under ``<replay_dir>/<player>/``. Composes with ``filter_mode`` —
    ``player='kovi', filter_mode='winner'`` keeps only kovi's actions in
    replays kovi won.
    """
    csvs = sorted(action_dir.glob("action_*.csv"))

    if player is not None:
        if replay_dir is None:
            raise ValueError("player= requires replay_dir=")
        keep_stems = player_replay_stems(replay_dir, player)
        csvs = [p for p in csvs if p.stem.removeprefix("action_") in keep_stems]

    if filter_mode == "all":
        return csvs
    if filter_mode == "winner":
        return [p for p in csvs if _csv_winner_slot_match(p)]
    raise ValueError(f"unknown filter_mode={filter_mode!r}")


def acted_only_indices(dataset: ActionSnapshotDataset) -> list[int]:
    """Return dataset indices where ``expert_acted > 0.5``."""
    out: list[int] = []
    for i in range(len(dataset)):
        snap = dataset.snapshots[i]
        if float(snap["expert_acted"].item()) > 0.5:
            out.append(i)
    return out


# ---------- Dataset materialization (with on-disk cache) ----------
DEFAULT_DATASET_CACHE_DIR: Path = (
    Path(__file__).resolve().parents[3] / "data" / "datasets" / "_cache"
)


def _dataset_cache_path(
    cache_dir: Path,
    *,
    player: str | None,
    filter_mode: str,
    max_planets: int,
    max_fleets: int,
    n_history: int,
) -> Path:
    """Stable cache filename keyed on (player, filter, dataset shape).

    Different ``--max-planets`` / ``--max-fleets`` / ``--n-history``
    materialize different tensor shapes; cache must not collide.
    """
    tag = (
        f"{player or 'any'}_{filter_mode}_"
        f"p{max_planets}_f{max_fleets}_h{n_history}.pt"
    )
    return Path(cache_dir) / tag


def prepare_dataset(
    *,
    player: str | None = None,
    filter_mode: str = "all",
    action_dir: Path | str = ACTION_DATASET_DIR,
    planet_dir: Path | str = PLANET_DATASET_DIR,
    fleet_dir: Path | str = FLEET_DATASET_DIR,
    entity_dir: Path | str = ENTITY_DATASET_DIR,
    cross_entity_dir: Path | str = CROSS_ENTITY_DATASET_DIR,
    replay_dir: Path | str = "data/replays",
    max_planets: int = 64,
    max_fleets: int = 256,
    n_history: int = 3,
    cache_dir: Path | str | None = None,
    rebuild_cache: bool = False,
) -> ActionSnapshotDataset:
    """Build the action-snapshot dataset for one expert (in-memory).

    Returns an :class:`ActionSnapshotDataset` ready to drop into the
    train loop. CSV → tensor materialization takes ~3 min for an
    Ebi-sized corpus (434 replays, ~120k snapshots), so the intended
    pattern is: **call once per Colab session, hold the returned dataset
    in a kernel variable, and pass it to** :func:`train_pair_score_kwargs`
    **via the** ``dataset=`` **kwarg** for every subsequent training
    re-run. That keeps training-cell re-runs near-instant.

    ``cache_dir`` enables an on-disk cache for survival across kernel
    restarts. **Off by default** — the cache file is ~9 GB for an
    Ebi-sized corpus and the save itself takes longer than rebuilding
    from CSVs, so it's only worth enabling for non-Colab CLI workflows
    where the same Python process won't stay alive long enough to hold
    the in-memory dataset across runs. Pass ``cache_dir='data/datasets/_cache'``
    (or any path) to opt in. ``rebuild_cache=True`` forces a fresh
    build even when a cache file exists — set after regenerating CSVs.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _dataset_cache_path(
            cache_dir,
            player=player, filter_mode=filter_mode,
            max_planets=max_planets, max_fleets=max_fleets, n_history=n_history,
        )

    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        print(f"[prepare] cache hit: {cache_path}", flush=True)
        t0 = time.time()
        dataset = ActionSnapshotDataset.from_cache(cache_path)
        print(
            f"[prepare] loaded {len(dataset)} snapshots from cache "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )
        return dataset

    if cache_path is not None:
        print(
            f"[prepare] cache miss at {cache_path}; building from CSVs ...",
            flush=True,
        )
    t0 = time.time()
    action_csvs = discover_action_csvs(
        Path(action_dir),
        filter_mode=filter_mode,
        player=player,
        replay_dir=Path(replay_dir) if player else None,
    )
    if not action_csvs:
        raise SystemExit(
            f"no action CSVs found under {action_dir} "
            f"with player={player!r} filter={filter_mode!r}"
        )
    print(
        f"[prepare] {len(action_csvs)} action CSVs "
        f"(player={player or 'any'}, filter={filter_mode!r})",
        flush=True,
    )

    def _other(stems: list[str], dir_path: Path, prefix: str) -> list[Path]:
        return [dir_path / f"{prefix}{s}.csv" for s in stems
                if (dir_path / f"{prefix}{s}.csv").exists()]
    stems = [p.stem.removeprefix("action_") for p in action_csvs]
    planet_csvs = _other(stems, Path(planet_dir), "planet_")
    fleet_csvs = _other(stems, Path(fleet_dir), "fleet_")
    entity_csvs = _other(stems, Path(entity_dir), "entity_")
    cross_csvs = _other(stems, Path(cross_entity_dir), "cross_entity_")

    dataset = ActionSnapshotDataset(
        planet_csv_paths=planet_csvs,
        fleet_csv_paths=fleet_csvs,
        entity_csv_paths=entity_csvs,
        cross_entity_csv_paths=cross_csvs,
        action_csv_paths=action_csvs,
        max_planets=max_planets,
        max_fleets=max_fleets,
        n_history=n_history,
    )
    build_s = time.time() - t0
    print(
        f"[prepare] built {len(dataset)} snapshots in {build_s:.1f}s",
        flush=True,
    )
    if cache_path is not None:
        print(f"[prepare] saving cache to {cache_path} ...", flush=True)
        t0 = time.time()
        dataset.save_cache(cache_path)
        print(f"[prepare] cache saved in {time.time() - t0:.1f}s", flush=True)
    return dataset


# ---------- Training loop ----------
def _train_one_epoch(
    stack: PairScoreStack,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    unfrozen: set[str],
    *,
    frac_weight: float = 0.0,
    frac_baseline_mae: float | None = None,
    freeze_pair_head: bool = False,
) -> dict[str, float]:
    stack.train()
    stack.set_freeze_state(unfrozen, freeze_pair_head=freeze_pair_head)
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = stack(batch)
        pair_loss, pair_metrics = compute_pair_score_loss(preds, batch)
        loss = pair_loss
        for k, v in pair_metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        if frac_weight > 0.0:
            frac_loss, frac_metrics = compute_frac_loss(
                preds, batch, frac_baseline_mae=frac_baseline_mae,
            )
            loss = loss + frac_weight * frac_loss
            for k, v in frac_metrics.items():
                sums[k] = sums.get(k, 0.0) + v
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        sums["loss"] = sums.get("loss", 0.0) + float(loss.item())
        sums["pair_loss"] = sums.get("pair_loss", 0.0) + float(pair_loss.item())
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in sums.items()}


@torch.no_grad()
def _evaluate(
    stack: PairScoreStack,
    loader: DataLoader,
    device: torch.device,
    *,
    frac_weight: float = 0.0,
    frac_baseline_mae: float | None = None,
) -> dict[str, float]:
    stack.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = stack(batch)
        pair_loss, pair_metrics = compute_pair_score_loss(preds, batch)
        total = pair_loss
        for k, v in pair_metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        if frac_weight > 0.0:
            frac_loss, frac_metrics = compute_frac_loss(
                preds, batch, frac_baseline_mae=frac_baseline_mae,
            )
            total = total + frac_weight * frac_loss
            for k, v in frac_metrics.items():
                sums[k] = sums.get(k, 0.0) + v
        sums["loss"] = sums.get("loss", 0.0) + float(total.item())
        sums["pair_loss"] = sums.get("pair_loss", 0.0) + float(pair_loss.item())
        sums["random_valid_top1"] = (
            sums.get("random_valid_top1", 0.0) + random_valid_baseline(batch, device)
        )
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in sums.items()}


def train_pair_score(
    args: argparse.Namespace,
    *,
    dataset: ActionSnapshotDataset | None = None,
) -> Path:
    """Run the pair-score (+ optional frac) training loop.

    ``dataset`` (kwarg-only): pass a pre-built :class:`ActionSnapshotDataset`
    to skip CSV parsing entirely. This is the fast notebook path —
    materialize once via :func:`prepare_dataset` into a kernel variable,
    then call this for each hyperparameter iteration. When ``None`` the
    function falls back to ``prepare_dataset(...)`` with the args'
    configuration.
    """
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. Resolve unfrozen modules ----
    unfreeze_list: list[str] = []
    if getattr(args, "unfreeze", None):
        unfreeze_list = [s.strip() for s in args.unfreeze.split(",") if s.strip()]
    unfrozen = PairScoreStack._canonicalize(unfreeze_list)
    if unfrozen:
        print(f"[pair_score] unfreezing {sorted(unfrozen)}", flush=True)

    # ---- 1. Load + freeze encoders ----
    fenc, penc, eenc, cross = load_frozen_encoder_stack(
        args.encoder_ckpt, d_model=args.d_model, device=str(device),
    )

    # ---- 2. Get the dataset ----
    # If the caller passed one in via the ``dataset`` kwarg (notebook
    # pattern: one ``prepare_dataset`` call earlier in the session, held
    # in a kernel variable), reuse it as-is. Otherwise build now,
    # respecting ``--cache-dir`` / ``--rebuild-cache``.
    if dataset is None:
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
        )
    else:
        print(
            f"[pair_score] using caller-provided dataset "
            f"({len(dataset)} snapshots) — skipping CSV parse",
            flush=True,
        )

    # ---- 3. Filter to acted rows; cap at --max-rows ----
    acted_idx = acted_only_indices(dataset)
    print(f"[pair_score] acted rows: {len(acted_idx)}", flush=True)
    if args.max_rows is not None:
        acted_idx = acted_idx[: args.max_rows]
        print(f"[pair_score] capped to {len(acted_idx)} rows", flush=True)

    if args.overfit:
        train_idx = acted_idx
        val_idx = acted_idx        # train==val on purpose for tiny-overfit
    else:
        n_val = max(1, int(round(len(acted_idx) * args.val_frac)))
        train_idx = acted_idx[:-n_val]
        val_idx = acted_idx[-n_val:]
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    print(f"[pair_score] train={len(train_set)} val={len(val_set)}", flush=True)

    # ---- 4. Build pair head + (optional) frac head + stack ----
    head = PairScoreHead(d_model=args.d_model, hidden=args.hidden).to(device)
    frac_weight = float(getattr(args, "frac_weight", 0.0) or 0.0)
    frac_head: FracHead | None = None
    if frac_weight > 0.0:
        frac_hidden = int(getattr(args, "frac_hidden", args.hidden) or args.hidden)
        frac_head = FracHead(d_model=args.d_model, hidden=frac_hidden).to(device)
        print(f"[pair_score] frac head enabled (weight={frac_weight}, "
              f"hidden={frac_hidden}, init log_std={FRAC_LOG_STD_INIT:.3f})", flush=True)
    stack = PairScoreStack(
        fleet_encoder=fenc,
        planet_encoder=penc,
        entity_encoder=eenc,
        cross=cross,
        pair_score_head=head,
        frac_head=frac_head,
    ).to(device)

    # ---- 4b. Optional resume from a prior pair_score_best.pt ----
    init_from = getattr(args, "init_from", None)
    if init_from:
        prior = torch.load(Path(init_from), map_location=device, weights_only=False)
        if "pair_score_head" not in prior:
            raise SystemExit(
                f"--init-from {init_from} has no 'pair_score_head' key — "
                "expected output of a prior pair_score run."
            )
        head.load_state_dict(prior["pair_score_head"])
        # If the prior run unfroze any encoders, prefer those weights —
        # otherwise the encoder state from --encoder-ckpt is kept.
        for name in PairScoreStack.ENCODER_MODULES:
            if name in prior:
                getattr(stack, name).load_state_dict(prior[name])
                print(f"[pair_score] init-from: loaded {name} state", flush=True)
        # Frac head reuse: only meaningful when we built one for this run.
        if frac_head is not None:
            if "frac_head" in prior:
                frac_head.load_state_dict(prior["frac_head"])
                print("[pair_score] init-from: loaded frac_head state", flush=True)
            else:
                print("[pair_score] init-from: frac_head missing in ckpt; "
                      "initialized from scratch", flush=True)
        print(f"[pair_score] init-from: loaded pair_score_head from "
              f"{init_from} (epoch={prior.get('epoch')})", flush=True)

    freeze_pair_head = bool(getattr(args, "freeze_pair_head", False))
    if freeze_pair_head:
        print("[pair_score] pair head FROZEN (stage-2 mode: only frac head + "
              "any --unfreeze encoders are trainable)", flush=True)
    stack.set_freeze_state(unfrozen, freeze_pair_head=freeze_pair_head)

    # ---- 4c. Train-split baseline for frac MAE (constant predictor) ----
    frac_baseline_mae: float | None = None
    if frac_weight > 0.0:
        frac_baseline_mae = compute_frac_baseline_mae(dataset, train_idx)
        if math.isfinite(frac_baseline_mae):
            print(f"[pair_score] frac baseline MAE (constant predictor on train): "
                  f"{frac_baseline_mae:.4f}", flush=True)
        else:
            print("[pair_score] frac baseline MAE: NaN (no acted train rows had "
                  "finite frac_label)", flush=True)

    # AdamW over every parameter that's actually trainable now (head +
    # any unfrozen encoders). filter is needed because frozen params
    # have requires_grad=False.
    trainable_params = [p for p in stack.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    n_train_params = sum(p.numel() for p in trainable_params)
    print(f"[pair_score] trainable params: {n_train_params:,}", flush=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
    )

    # ---- 5. Train + log ----
    best_val_loss = float("inf")
    best_path = out_dir / "pair_score_best.pt"
    last_path = out_dir / "pair_score_last.pt"
    log_path = out_dir / "log.json"
    log_entries: list[dict[str, Any]] = []

    sigma_pinned_epochs = 0  # count epochs where log_std hit either clamp

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_metrics = _train_one_epoch(
            stack, train_loader, optim, device, unfrozen,
            frac_weight=frac_weight, frac_baseline_mae=frac_baseline_mae,
            freeze_pair_head=freeze_pair_head,
        )
        val_metrics = _evaluate(
            stack, val_loader, device,
            frac_weight=frac_weight, frac_baseline_mae=frac_baseline_mae,
        )
        elapsed = time.time() - t0
        log = {
            "epoch": epoch,
            "elapsed_s": round(elapsed, 1),
            "epoch_s": round(time.time() - t_epoch, 1),
            "train": train_metrics,
            "val": val_metrics,
        }
        log_entries.append(log)
        log_path.write_text(json.dumps(log_entries, indent=2))

        # Per-epoch summary: always shows pair metrics; frac block only
        # appears when ``--frac-weight > 0`` so the no-frac path keeps
        # its existing log format (smoke regression #2).
        frac_block = ""
        if frac_weight > 0.0:
            frac_block = (
                f"  ||  tr_frac_mae={train_metrics.get('frac_mae', 0):.3f} "
                f"val_frac_mae={val_metrics.get('frac_mae', 0):.3f} "
                f"vs_baseline={val_metrics.get('frac_baseline_mae', float('nan')):.3f} "
                f"frac_sigma={val_metrics.get('frac_sigma', 0):.3f}"
            )
            sigma_disp = val_metrics.get("frac_sigma", 0.0)
            if sigma_disp <= math.exp(FRAC_LOG_STD_MIN) + 1e-6 \
                    or sigma_disp >= math.exp(FRAC_LOG_STD_MAX) - 1e-6:
                sigma_pinned_epochs += 1
        print(
            f"[pair_score] ep={epoch:3d} "
            f"tr_loss={train_metrics.get('loss', 0):.4f} "
            f"tr_top1={train_metrics.get('top1', 0):.3f}  |  "
            f"val_loss={val_metrics.get('loss', 0):.4f} "
            f"val_top1={val_metrics.get('top1', 0):.3f} "
            f"val_top3={val_metrics.get('top3', 0):.3f} "
            f"val_top5={val_metrics.get('top5', 0):.3f} "
            f"rand={val_metrics.get('random_valid_top1', 0):.3f}"
            f"{frac_block}  "
            f"dt={time.time() - t_epoch:.1f}s",
            flush=True,
        )

        # Save last + best. Include any unfrozen encoder state and the
        # frac head (when built) so a follow-up run can resume via
        # ``--init-from``.
        ckpt_payload: dict = {
            "epoch": epoch,
            "encoder_ckpt": str(args.encoder_ckpt),
            "init_from": str(init_from) if init_from else None,
            "unfrozen": sorted(unfrozen),
            "frac_weight": frac_weight,
            "config": {
                "d_model": args.d_model,
                "hidden": args.hidden,
                "max_planets": args.max_planets,
                "max_fleets": args.max_fleets,
                "n_history": args.n_history,
                "frac_hidden": int(getattr(args, "frac_hidden", args.hidden) or args.hidden),
            },
            "metrics": {"train": train_metrics, "val": val_metrics},
        }
        if frac_weight > 0.0:
            ckpt_payload["frac_baseline_mae"] = (
                float(frac_baseline_mae) if frac_baseline_mae is not None else float("nan")
            )
        ckpt_payload.update(stack.trainable_module_state(unfrozen))
        torch.save(ckpt_payload, last_path)
        if val_metrics.get("loss", float("inf")) < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(torch.load(last_path, weights_only=False), best_path)

    if frac_weight > 0.0 and args.epochs > 0:
        pin_ratio = sigma_pinned_epochs / args.epochs
        if pin_ratio >= 0.8:
            print(
                f"[pair_score] WARNING: frac_log_std pinned at a clamp "
                f"boundary for {sigma_pinned_epochs}/{args.epochs} epochs "
                f"({pin_ratio:.0%}). Pinned-low = overconfident, "
                f"pinned-high = uncertainty-dominated. Revisit "
                f"FRAC_LOG_STD_INIT or --frac-weight.",
                flush=True,
            )

    print(f"[pair_score] done. best_val_loss={best_val_loss:.4f} "
          f"ckpts: {best_path.name}, {last_path.name}", flush=True)
    return best_path


# ---------- In-kernel entry point ----------
def train_pair_score_kwargs(
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
    epochs=10,
    d_model=64,
    hidden=128,
    max_planets=64,
    max_fleets=256,
    n_history=3,
    device=None,
    unfreeze=None,
    init_from=None,
    frac_weight=0.0,
    frac_hidden=128,
    freeze_pair_head=False,
    cache_dir=None,
    rebuild_cache=False,
    dataset=None,
) -> Path:
    """Run :func:`train_pair_score` in the calling Python process.

    Same surface as the CLI but no ``argparse`` and no ``subprocess`` —
    every ``print(...)`` inside the trainer streams directly into the
    caller's stdout (e.g. a Jupyter cell), so progress is visible epoch
    by epoch instead of waiting for the subprocess to flush a buffer.

    ``device`` defaults to ``cuda`` if available, else ``cpu``.
    """
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
        hidden=hidden,
        max_planets=max_planets,
        max_fleets=max_fleets,
        n_history=n_history,
        device=device,
        out_dir=Path(out_dir),
        unfreeze=unfreeze,
        init_from=Path(init_from) if init_from is not None else None,
        frac_weight=float(frac_weight),
        frac_hidden=int(frac_hidden),
        freeze_pair_head=bool(freeze_pair_head),
        cache_dir=(Path(cache_dir) if cache_dir is not None else None),
        rebuild_cache=bool(rebuild_cache),
    )
    return train_pair_score(args, dataset=dataset)


# ---------- CLI ----------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder-ckpt", type=Path, required=True,
                   help="Path to a stack ckpt with fleet/planet/entity/cross state dicts.")
    p.add_argument("--action-dir", type=Path, default=ACTION_DATASET_DIR)
    p.add_argument("--planet-dir", type=Path, default=PLANET_DATASET_DIR)
    p.add_argument("--fleet-dir", type=Path, default=FLEET_DATASET_DIR)
    p.add_argument("--entity-dir", type=Path, default=ENTITY_DATASET_DIR)
    p.add_argument("--cross-entity-dir", type=Path, default=CROSS_ENTITY_DATASET_DIR)
    p.add_argument("--filter", choices=["winner", "all"], default="all",
                   help="winner = keep only CSVs whose learner_slot == winner_seat. "
                        "Composes with --player.")
    p.add_argument("--player", default=None,
                   help="Restrict to one player's replays (e.g. 'kovi', "
                        "'Shun_PI', 'Orbital Occle'). Replay tree is "
                        "<replay-dir>/<player>/<stem>.json.gz.")
    p.add_argument("--replay-dir", type=Path,
                   default=Path("data/replays"),
                   help="Root containing per-player replay subdirs.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap on acted rows (after filter). None = no cap.")
    p.add_argument("--overfit", action="store_true",
                   help="Use train==val (tiny-overfit Experiment 1).")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--max-planets", type=int, default=64)
    p.add_argument("--max-fleets", type=int, default=256)
    p.add_argument("--n-history", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--unfreeze", default=None,
                   help="Comma-separated encoder modules to thaw alongside "
                        "the head. Names: cross, entity (=entity_encoder), "
                        "planet (=planet_encoder), fleet (=fleet_encoder). "
                        "Example: --unfreeze cross,entity")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Resume from a prior pair_score_best.pt. Loads the "
                        "head, frac head (if present), and any previously-"
                        "thawed encoder state. Modules not in the prior "
                        "file fall back to fresh init / --encoder-ckpt.")
    p.add_argument("--frac-weight", type=float, default=0.0,
                   help="Weight for the launch-fraction Normal-NLL loss "
                        "(0.0 = disabled, no FracHead built, log format "
                        "matches the pre-frac contract). 0.5 is a sane "
                        "joint-training start.")
    p.add_argument("--frac-hidden", type=int, default=128,
                   help="Hidden width of the FracHead MLP. "
                        "Ignored when --frac-weight 0.")
    p.add_argument("--freeze-pair-head", action="store_true",
                   help="Freeze the pair-score head (no gradient flows). "
                        "Stage-2 frac-only pattern: pair head + all "
                        "encoders frozen, only FracHead trains. Combine "
                        "with --init-from <pair-trained best.pt>.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Optional on-disk cache for the materialized "
                        "ActionSnapshotDataset. Off by default — the cache "
                        "file can be multi-GB and saves take as long as a "
                        "rebuild, so it's only worth enabling for non-"
                        "Colab CLI workflows. Pass e.g. "
                        f"--cache-dir {DEFAULT_DATASET_CACHE_DIR} to opt in.")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force a fresh CSV parse even when a cache file exists "
                        "(set this after regenerating action CSVs).")
    args = p.parse_args()
    train_pair_score(args)


if __name__ == "__main__":
    main()
