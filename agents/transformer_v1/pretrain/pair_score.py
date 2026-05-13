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


class TargetHead(nn.Module):
    """Per-planet target-attractiveness score.

    Marginal-over-source companion to :class:`PairScoreHead`: instead
    of scoring every ``(source, target)`` pair, this head scores every
    planet on its own — own / neutral / enemy uniform, no ownership
    mask. Downstream code combines the per-planet logit with the pair
    head's conditional however it likes (e.g. ``pair_logits[src, :]
    + α · target_logits`` at inference, or treats it as an auxiliary
    BC loss in training).

    Per-planet feature (depending on ``use_entity``):

      * ``use_entity=False`` (v1 default for older ckpts):
          feat_i = [ glob ‖ ctx_i ]                       (2·d)
      * ``use_entity=True``  (v2 default):
          feat_i = [ glob ‖ ctx_i ‖ entity_i ]            (3·d)
        where ``entity_i`` is the post-:class:`PlanetEntityEncoder`
        token (after the per-(planet, owner) fleet pooling) and
        before cross-attention — i.e., the entity-local view of
        planet ``i`` that the cross attention then mixes globally.

    ``num_layers`` chooses 2- or 3-layer MLP depth. v1 used 2, v2
    bumps to 3 for more capacity since the input dim grew 2d → 3d.

    Parameter count (d=64, hidden=128):
      * v1 (2d, 2 layers): 2·64·128 + 128 + 128·1 + 1  ≈ **16 641**
      * v2 (3d, 3 layers): 3·64·128 + 128 + 128·128 + 128 + 128·1 + 1
                                                       ≈ **41 345**

    Loaders should pass ``use_entity`` and ``num_layers`` from the
    saved ckpt's config block so older 2d/2-layer ckpts still load
    state_dicts cleanly.
    """

    def __init__(
        self,
        d_model: int = 64,
        hidden: int = 128,
        *,
        use_entity: bool = True,
        num_layers: int = 3,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2 (got {num_layers})")
        self.d_model = d_model
        self.hidden = hidden
        self.use_entity = bool(use_entity)
        self.num_layers = int(num_layers)

        in_dim = (3 if self.use_entity else 2) * d_model
        layers: list[nn.Module] = []
        cur = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(cur, hidden))
            layers.append(nn.GELU())
            cur = hidden
        layers.append(nn.Linear(cur, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)

    def forward(
        self,
        glob: torch.Tensor,                          # (B, d)
        ctx: torch.Tensor,                           # (B, P, d)
        entity_now: torch.Tensor | None = None,      # (B, P, d) — required when use_entity=True
    ) -> torch.Tensor:
        B, P, d = ctx.shape
        if d != self.d_model:
            raise ValueError(
                f"ctx d={d} but TargetHead built for d_model={self.d_model}"
            )
        glob_b = glob.unsqueeze(1).expand(B, P, d)
        if self.use_entity:
            if entity_now is None:
                raise ValueError(
                    "TargetHead was built with use_entity=True; pass "
                    "entity_now (the post-PlanetEntityEncoder token, last "
                    "history step) when calling forward."
                )
            feat = torch.cat([glob_b, ctx, entity_now], dim=-1)   # (B, P, 3d)
        else:
            feat = torch.cat([glob_b, ctx], dim=-1)               # (B, P, 2d)
        return self.mlp(feat).squeeze(-1)                          # (B, P)


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
        target_head: TargetHead | None = None,
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
        # TargetHead is optional in the same way (controlled by
        # ``--target-weight``). When non-None, it adds ~17k params and
        # contributes ``target_logits`` to ``forward()``.
        self.target_head = target_head

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
        freeze_frac_head: bool = False,
        freeze_target_head: bool = False,
    ) -> None:
        """Set each encoder's train/grad mode based on whether its name
        appears in ``unfrozen``. The pair / frac / target heads default
        to trainable; each can be individually frozen with the matching
        keyword flag so stage-N runs can pin earlier-trained heads.

        ``unfrozen`` may contain ``"cross"``, ``"entity"`` (= entity_encoder),
        ``"planet"`` (= planet_encoder), or ``"fleet"`` (= fleet_encoder).
        Anything else raises.

        Stage-2 frac-only training pattern: pass
        ``freeze_pair_head=True`` + ``unfrozen=()`` so only the FracHead
        trains. Stage-3 target-only on top of a stage-2 ckpt: also pass
        ``freeze_frac_head=True`` so the frac calibration is preserved.
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
        # Frac head (when present) trainable by default, frozen on opt-in.
        if self.frac_head is not None:
            if freeze_frac_head:
                self.frac_head.eval()
                for p in self.frac_head.parameters():
                    p.requires_grad_(False)
            else:
                self.frac_head.train()
                for p in self.frac_head.parameters():
                    p.requires_grad_(True)
        # Target head (when present) trainable by default, frozen on opt-in.
        if self.target_head is not None:
            if freeze_target_head:
                self.target_head.eval()
                for p in self.target_head.parameters():
                    p.requires_grad_(False)
            else:
                self.target_head.train()
                for p in self.target_head.parameters():
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
        if self.target_head is not None:
            out["target_head"] = self.target_head.state_dict()
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

        # entity_now: the post-PlanetEntityEncoder token for the
        # current turn — required by the v2 TargetHead.
        entity_now = (
            entity_tokens[:, -1] if entity_tokens.dim() == 4 else entity_tokens
        )
        pair_logits = self.pair_score_head(glob, ctx_now, src_valid, tgt_valid)
        out: dict[str, torch.Tensor] = {
            "pair_logits": pair_logits,
            "_ctx_now": ctx_now,
            "_glob": glob,
            "_entity_now": entity_now,
        }
        if self.frac_head is not None:
            frac_out = self.frac_head(glob, ctx_now)
            out["frac_loc"] = frac_out["frac_loc"]
            out["frac_log_std"] = frac_out["frac_log_std"]
        if self.target_head is not None:
            if getattr(self.target_head, "use_entity", False):
                out["target_logits"] = self.target_head(glob, ctx_now, entity_now)
            else:
                out["target_logits"] = self.target_head(glob, ctx_now)
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


def compute_target_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy over the per-planet target head, sparse on acted rows.

    Same label as the pair head's target dimension —
    ``batch["target_planet_idx"]`` — but supervised against the
    standalone ``target_logits`` (no source conditioning). The head
    learns a marginal-over-source target preference. No
    ``tgt_valid`` masking is applied at loss time so the head sees
    own / neutral / enemy uniformly; the caller decides how to use
    it at inference.

    Returns ``(loss, metrics)`` with keys ``target_loss``,
    ``target_top1``, ``target_top3``, ``target_top5``, ``n_target_valid``.
    Always populated so the per-epoch summary doesn't branch.
    """
    target_logits = preds.get("target_logits")
    if target_logits is None:
        zero = next(iter(preds.values())).sum() * 0.0
        return zero, {
            "target_loss": 0.0,
            "target_top1": 0.0,
            "target_top3": 0.0,
            "target_top5": 0.0,
            "n_target_valid": 0.0,
        }

    device = target_logits.device
    tgt_idx = batch["target_planet_idx"].to(device).long()
    B, P = target_logits.shape
    valid = (tgt_idx >= 0) & (tgt_idx < P)
    n_valid = int(valid.sum().item())
    metrics: dict[str, float] = {
        "target_loss": 0.0,
        "target_top1": 0.0,
        "target_top3": 0.0,
        "target_top5": 0.0,
        "n_target_valid": float(n_valid),
    }
    if n_valid == 0:
        return target_logits.sum() * 0.0, metrics

    logits = target_logits[valid]                                  # (Nv, P)
    y = tgt_idx[valid]                                             # (Nv,)
    loss = F.cross_entropy(logits, y)

    with torch.no_grad():
        k = min(5, P)
        top_ids = logits.topk(k, dim=-1).indices                   # (Nv, k)
        match = top_ids == y.unsqueeze(-1)
        metrics["target_loss"] = float(loss.item())
        metrics["target_top1"] = float(match[:, 0].float().mean().item())
        metrics["target_top3"] = float(match[:, : min(3, k)].any(-1).float().mean().item())
        metrics["target_top5"] = float(match.any(-1).float().mean().item())
    return loss, metrics


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
    pair_weight: float = 1.0,
    frac_weight: float = 0.0,
    frac_baseline_mae: float | None = None,
    target_weight: float = 0.0,
    freeze_pair_head: bool = False,
    freeze_frac_head: bool = False,
    freeze_target_head: bool = False,
) -> dict[str, float]:
    stack.train()
    stack.set_freeze_state(
        unfrozen,
        freeze_pair_head=freeze_pair_head,
        freeze_frac_head=freeze_frac_head,
        freeze_target_head=freeze_target_head,
    )
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = stack(batch)
        pair_loss, pair_metrics = compute_pair_score_loss(preds, batch)
        # Always compute pair_metrics so logs stay informative even when
        # pair_weight=0 (caller may freeze the pair head and still want
        # to read off its frozen accuracy as a sanity check).
        for k, v in pair_metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        loss = pair_weight * pair_loss if pair_weight > 0.0 else pair_loss * 0.0
        if frac_weight > 0.0:
            frac_loss, frac_metrics = compute_frac_loss(
                preds, batch, frac_baseline_mae=frac_baseline_mae,
            )
            loss = loss + frac_weight * frac_loss
            for k, v in frac_metrics.items():
                sums[k] = sums.get(k, 0.0) + v
        if target_weight > 0.0:
            target_loss, target_metrics = compute_target_loss(preds, batch)
            loss = loss + target_weight * target_loss
            for k, v in target_metrics.items():
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
    pair_weight: float = 1.0,
    frac_weight: float = 0.0,
    frac_baseline_mae: float | None = None,
    target_weight: float = 0.0,
) -> dict[str, float]:
    stack.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = stack(batch)
        pair_loss, pair_metrics = compute_pair_score_loss(preds, batch)
        for k, v in pair_metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        total = pair_weight * pair_loss if pair_weight > 0.0 else pair_loss * 0.0
        if frac_weight > 0.0:
            frac_loss, frac_metrics = compute_frac_loss(
                preds, batch, frac_baseline_mae=frac_baseline_mae,
            )
            total = total + frac_weight * frac_loss
            for k, v in frac_metrics.items():
                sums[k] = sums.get(k, 0.0) + v
        if target_weight > 0.0:
            target_loss, target_metrics = compute_target_loss(preds, batch)
            total = total + target_weight * target_loss
            for k, v in target_metrics.items():
                sums[k] = sums.get(k, 0.0) + v
        sums["loss"] = sums.get("loss", 0.0) + float(total.item())
        sums["pair_loss"] = sums.get("pair_loss", 0.0) + float(pair_loss.item())
        sums["random_valid_top1"] = (
            sums.get("random_valid_top1", 0.0) + random_valid_baseline(batch, device)
        )
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in sums.items()}


# ---------- Target-only fast path: precompute frozen features ----------
@torch.no_grad()
def _precompute_frozen_features(
    stack: PairScoreStack,
    dataset: ActionSnapshotDataset,
    indices: list[int],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> list[dict[str, torch.Tensor]]:
    """Walk the indexed snapshots once through the frozen encoder + cross
    stack and cache the resulting ``glob`` + ``ctx_now`` per snapshot,
    plus the labels the TargetHead loss / metrics need. Returns a list
    of CPU-resident dicts, one per snapshot.

    The cache scales as ``len(indices) * (d + P * d) * 4 bytes`` —
    ~16 KB per snapshot at d=64 / P=64. 26 k Ebi acted rows ≈ 425 MB.
    Doesn't carry pair-specific tensors like ``src_valid`` because the
    pair head is assumed frozen here.
    """
    stack.eval()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)
    cached: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch,
            stack.fleet_encoder,
            stack.planet_encoder,
            stack.entity_encoder,
        )
        ctx, glob = stack.cross(entity_tokens, entity_mask)
        ctx_now = ctx[:, -1] if ctx.dim() == 4 else ctx
        entity_now = (
            entity_tokens[:, -1] if entity_tokens.dim() == 4 else entity_tokens
        )
        B = glob.shape[0]
        for i in range(B):
            cached.append({
                "glob": glob[i].detach().cpu(),
                "ctx": ctx_now[i].detach().cpu(),
                # Cache the pre-cross-attention entity token so a v2
                # TargetHead (use_entity=True) can train without re-
                # running the encoder. ~16 KB per snapshot at d=64/P=64
                # → adds ~450 MB for a full-Ebi acted-row cache.
                "entity": entity_now[i].detach().cpu(),
                "target_planet_idx": batch["target_planet_idx"][i].detach().cpu(),
                "source_planet_idx": batch["source_planet_idx"][i].detach().cpu(),
            })
    return cached


class _CachedFeatureDataset(torch.utils.data.Dataset):
    """Lightweight wrapper around a list of precomputed feature dicts."""

    def __init__(self, items: list[dict[str, torch.Tensor]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.items[i]


def _train_target_only_fast(
    target_head: TargetHead,
    pair_score_head: PairScoreHead,
    train_cached: list[dict[str, torch.Tensor]],
    val_cached: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    out_dir: Path,
    init_from_path: str | None,
    config: dict,
    save_extras: dict[str, dict] | None = None,
) -> Path:
    """Train ONLY the TargetHead on cached frozen-encoder features.

    Skips every encoder + cross-attention forward each batch — the
    cached ``(glob, ctx)`` per snapshot is the input to TargetHead +
    PairScoreHead (the latter only used to log frozen pair_top1 as a
    sanity check). Designed for the stage-N pattern where the pair
    head, frac head, and all encoders are frozen.

    Args:
      target_head: the only module that updates.
      pair_score_head: used in eval-only mode to recompute pair_top1
        per epoch from cached ctx. Stays on the CPU; small.
      train_cached / val_cached: outputs of :func:`_precompute_frozen_features`.
      save_extras: extra state-dicts to embed in the saved ckpt
        (e.g. frac_head, every encoder) so the output stays self-
        contained — same contract as the full path's ``trainable_module_state``.

    Returns the path to the best ckpt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target_head = target_head.to(device).train()
    pair_score_head = pair_score_head.to(device).eval()
    optim = torch.optim.AdamW(
        target_head.parameters(), lr=lr, weight_decay=weight_decay,
    )
    train_loader = DataLoader(
        _CachedFeatureDataset(train_cached),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        _CachedFeatureDataset(val_cached),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )

    best_val_loss = float("inf")
    best_path = out_dir / "pair_score_best.pt"
    last_path = out_dir / "pair_score_last.pt"
    log_path = out_dir / "log.json"
    log_entries: list[dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        target_head.train()
        # ---- Train ----
        n_train_batches = 0
        train_sums: dict[str, float] = {}
        for batch in train_loader:
            glob = batch["glob"].to(device)
            ctx = batch["ctx"].to(device)
            entity = batch["entity"].to(device) if "entity" in batch else None
            tgt_idx = batch["target_planet_idx"].to(device).long()
            if getattr(target_head, "use_entity", False):
                logits = target_head(glob, ctx, entity)
            else:
                logits = target_head(glob, ctx)
            B, P = logits.shape
            valid = (tgt_idx >= 0) & (tgt_idx < P)
            if not valid.any():
                continue
            loss = F.cross_entropy(logits[valid], tgt_idx[valid])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_sums["target_loss"] = train_sums.get("target_loss", 0.0) + float(loss.item())
            with torch.no_grad():
                k = min(5, P)
                top_ids = logits[valid].topk(k, dim=-1).indices
                yv = tgt_idx[valid]
                match = top_ids == yv.unsqueeze(-1)
                train_sums["target_top1"] = train_sums.get("target_top1", 0.0) + \
                    float(match[:, 0].float().mean().item())
                train_sums["target_top3"] = train_sums.get("target_top3", 0.0) + \
                    float(match[:, : min(3, k)].any(-1).float().mean().item())
            n_train_batches += 1
        train_metrics = {k: v / max(1, n_train_batches) for k, v in train_sums.items()}

        # ---- Eval (target loss + frozen pair sanity) ----
        target_head.eval()
        n_val_batches = 0
        val_sums: dict[str, float] = {}
        with torch.no_grad():
            for batch in val_loader:
                glob = batch["glob"].to(device)
                ctx = batch["ctx"].to(device)
                entity = batch["entity"].to(device) if "entity" in batch else None
                tgt_idx = batch["target_planet_idx"].to(device).long()
                src_idx = batch["source_planet_idx"].to(device).long()
                if getattr(target_head, "use_entity", False):
                    logits = target_head(glob, ctx, entity)
                else:
                    logits = target_head(glob, ctx)
                B, P = logits.shape
                valid_t = (tgt_idx >= 0) & (tgt_idx < P)
                if valid_t.any():
                    loss = F.cross_entropy(logits[valid_t], tgt_idx[valid_t])
                    val_sums["target_loss"] = val_sums.get("target_loss", 0.0) + float(loss.item())
                    k = min(5, P)
                    top_ids = logits[valid_t].topk(k, dim=-1).indices
                    yv = tgt_idx[valid_t]
                    match = top_ids == yv.unsqueeze(-1)
                    val_sums["target_top1"] = val_sums.get("target_top1", 0.0) + \
                        float(match[:, 0].float().mean().item())
                    val_sums["target_top3"] = val_sums.get("target_top3", 0.0) + \
                        float(match[:, : min(3, k)].any(-1).float().mean().item())
                    val_sums["target_top5"] = val_sums.get("target_top5", 0.0) + \
                        float(match.any(-1).float().mean().item())
                # Pair sanity (frozen): full pair_logits over (P, P) for the batch.
                pair_logits = pair_score_head(glob, ctx)
                valid_p = (src_idx >= 0) & (tgt_idx >= 0) & (src_idx < P) & (tgt_idx < P)
                if valid_p.any():
                    pl = pair_logits[valid_p]
                    si = src_idx[valid_p]
                    ti = tgt_idx[valid_p]
                    flat = pl.reshape(pl.shape[0], P * P)
                    y = si * P + ti
                    pred = flat.argmax(-1)
                    val_sums["pair_top1"] = val_sums.get("pair_top1", 0.0) + \
                        float((pred == y).float().mean().item())
                n_val_batches += 1
        val_metrics = {k: v / max(1, n_val_batches) for k, v in val_sums.items()}

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
        print(
            f"[pair_score-fast] ep={epoch:3d} "
            f"tr_loss={train_metrics.get('target_loss', 0):.4f} "
            f"tr_tgt_top1={train_metrics.get('target_top1', 0):.3f}  |  "
            f"val_loss={val_metrics.get('target_loss', 0):.4f} "
            f"val_tgt_top1={val_metrics.get('target_top1', 0):.3f} "
            f"val_tgt_top3={val_metrics.get('target_top3', 0):.3f} "
            f"val_tgt_top5={val_metrics.get('target_top5', 0):.3f}  ||  "
            f"pair_top1={val_metrics.get('pair_top1', 0):.3f} "
            f"(frozen sanity)  "
            f"dt={time.time() - t_epoch:.1f}s",
            flush=True,
        )

        ckpt_payload: dict = {
            "epoch": epoch,
            "init_from": init_from_path,
            "config": config,
            "metrics": {"train": train_metrics, "val": val_metrics},
            "target_weight": 1.0,
            "pair_weight": 0.0,
            "frac_weight": 0.0,
            "target_head": target_head.state_dict(),
            "pair_score_head": pair_score_head.state_dict(),
        }
        if save_extras:
            ckpt_payload.update(save_extras)
        torch.save(ckpt_payload, last_path)
        val_loss = val_metrics.get("target_loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(torch.load(last_path, weights_only=False), best_path)

    print(f"[pair_score-fast] done. best target_loss={best_val_loss:.4f} "
          f"ckpts: {best_path.name}, {last_path.name}", flush=True)
    return best_path


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

    # ---- 4. Build heads + stack ----
    # Optional heads are built when EITHER their weight > 0 OR the
    # --init-from ckpt already carries their state. This means a
    # target-only stage that resumes from a pair+frac ckpt rebuilds and
    # freezes the frac head (preserving it on save) instead of silently
    # dropping it from the output.
    head = PairScoreHead(d_model=args.d_model, hidden=args.hidden).to(device)
    pair_weight = float(getattr(args, "pair_weight", 1.0) or 0.0)
    if pair_weight == 0.0:
        print("[pair_score] pair_weight=0 — pair_loss excluded from total "
              "(still computed for metric display).", flush=True)
    frac_weight = float(getattr(args, "frac_weight", 0.0) or 0.0)
    target_weight = float(getattr(args, "target_weight", 0.0) or 0.0)

    # Peek at the init-from ckpt before deciding which heads to build so
    # we can preserve any extra heads it carries.
    init_from = getattr(args, "init_from", None)
    init_payload: dict[str, Any] | None = None
    if init_from is not None:
        init_payload = torch.load(
            Path(init_from), map_location=device, weights_only=False,
        )

    have_prior_frac = bool(init_payload and "frac_head" in init_payload)
    have_prior_target = bool(init_payload and "target_head" in init_payload)

    frac_head: FracHead | None = None
    if frac_weight > 0.0 or have_prior_frac:
        frac_hidden = int(getattr(args, "frac_hidden", args.hidden) or args.hidden)
        frac_head = FracHead(d_model=args.d_model, hidden=frac_hidden).to(device)
        why = "weight>0" if frac_weight > 0.0 else "preserved from --init-from"
        print(
            f"[pair_score] frac head enabled (weight={frac_weight}, "
            f"hidden={frac_hidden}, init log_std={FRAC_LOG_STD_INIT:.3f}, "
            f"reason={why})",
            flush=True,
        )

    target_head: TargetHead | None = None
    if target_weight > 0.0 or have_prior_target:
        target_hidden = int(getattr(args, "target_hidden", args.hidden) or args.hidden)
        # Honor the prior ckpt's TargetHead shape when resuming so its
        # state_dict loads. New training runs default to the v2 head
        # (use_entity=True, num_layers=3).
        prior_cfg = init_payload.get("config", {}) if init_payload else {}
        if have_prior_target:
            t_use_entity = bool(prior_cfg.get("target_use_entity", False))
            t_num_layers = int(prior_cfg.get("target_num_layers", 2))
        else:
            t_use_entity = bool(getattr(args, "target_use_entity", True))
            t_num_layers = int(getattr(args, "target_num_layers", 3))
        target_head = TargetHead(
            d_model=args.d_model, hidden=target_hidden,
            use_entity=t_use_entity, num_layers=t_num_layers,
        ).to(device)
        why = "weight>0" if target_weight > 0.0 else "preserved from --init-from"
        print(
            f"[pair_score] target head enabled (weight={target_weight}, "
            f"hidden={target_hidden}, use_entity={t_use_entity}, "
            f"num_layers={t_num_layers}, reason={why})",
            flush=True,
        )

    stack = PairScoreStack(
        fleet_encoder=fenc,
        planet_encoder=penc,
        entity_encoder=eenc,
        cross=cross,
        pair_score_head=head,
        frac_head=frac_head,
        target_head=target_head,
    ).to(device)

    # ---- 4b. Apply init-from state to the freshly built modules ----
    if init_payload is not None:
        prior = init_payload
        if "pair_score_head" not in prior:
            raise SystemExit(
                f"--init-from {init_from} has no 'pair_score_head' key — "
                "expected output of a prior pair_score run."
            )
        head.load_state_dict(prior["pair_score_head"])
        for name in PairScoreStack.ENCODER_MODULES:
            if name in prior:
                getattr(stack, name).load_state_dict(prior[name])
                print(f"[pair_score] init-from: loaded {name} state", flush=True)
        if frac_head is not None:
            if "frac_head" in prior:
                frac_head.load_state_dict(prior["frac_head"])
                print("[pair_score] init-from: loaded frac_head state", flush=True)
            else:
                print("[pair_score] init-from: frac_head missing in ckpt; "
                      "initialized from scratch", flush=True)
        if target_head is not None:
            if "target_head" in prior:
                target_head.load_state_dict(prior["target_head"])
                print("[pair_score] init-from: loaded target_head state", flush=True)
            else:
                print("[pair_score] init-from: target_head missing in ckpt; "
                      "initialized from scratch", flush=True)
        print(f"[pair_score] init-from: loaded pair_score_head from "
              f"{init_from} (epoch={prior.get('epoch')})", flush=True)

    freeze_pair_head = bool(getattr(args, "freeze_pair_head", False))
    freeze_frac_head = bool(getattr(args, "freeze_frac_head", False))
    freeze_target_head = bool(getattr(args, "freeze_target_head", False))
    frozen_heads = []
    if freeze_pair_head:
        frozen_heads.append("pair")
    if freeze_frac_head and frac_head is not None:
        frozen_heads.append("frac")
    if freeze_target_head and target_head is not None:
        frozen_heads.append("target")
    if frozen_heads:
        print(f"[pair_score] heads FROZEN: {frozen_heads} — only the unfrozen "
              "heads + any --unfreeze encoders will train", flush=True)
    stack.set_freeze_state(
        unfrozen,
        freeze_pair_head=freeze_pair_head,
        freeze_frac_head=freeze_frac_head,
        freeze_target_head=freeze_target_head,
    )

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

    # Detect target-only mode: every encoder frozen, pair head frozen,
    # frac head missing-or-frozen, target head present and trainable.
    # In that mode the encoder + cross + pair + frac forward never
    # changes — we precompute (glob, ctx) per snapshot once and iterate
    # the TargetHead alone, which lets full-Ebi training fit a local
    # CPU budget (~minutes instead of ~hours).
    encoders_frozen = not unfrozen
    frac_path_inert = (frac_head is None) or freeze_frac_head
    target_only_mode = (
        encoders_frozen
        and freeze_pair_head
        and frac_path_inert
        and target_head is not None
        and not freeze_target_head
    )
    if target_only_mode:
        print(
            "[pair_score] target-only fast path engaged: precomputing "
            "(glob, ctx) per snapshot, then training only the TargetHead "
            f"on {len(train_idx)} train + {len(val_idx)} val cached features.",
            flush=True,
        )
        t_pc = time.time()
        train_cached = _precompute_frozen_features(
            stack, dataset, train_idx,
            device=device, batch_size=args.batch_size,
        )
        val_cached = _precompute_frozen_features(
            stack, dataset, val_idx,
            device=device, batch_size=args.batch_size,
        )
        print(
            f"[pair_score] precomputed {len(train_cached) + len(val_cached)} "
            f"snapshot features in {time.time() - t_pc:.1f}s",
            flush=True,
        )
        # Self-contained ckpt extras: encoder + pair + frac states stay
        # bit-exact (none of them updates this run).
        save_extras: dict[str, dict] = {}
        for name in PairScoreStack.ENCODER_MODULES:
            save_extras[name] = getattr(stack, name).state_dict()
        if frac_head is not None:
            save_extras["frac_head"] = frac_head.state_dict()
        save_extras["encoder_ckpt"] = str(args.encoder_ckpt)
        config = {
            "d_model": args.d_model,
            "hidden": args.hidden,
            "max_planets": args.max_planets,
            "max_fleets": args.max_fleets,
            "n_history": args.n_history,
            "target_hidden": int(getattr(args, "target_hidden", args.hidden) or args.hidden),
            "target_use_entity": bool(getattr(target_head, "use_entity", False)),
            "target_num_layers": int(getattr(target_head, "num_layers", 2)),
        }
        return _train_target_only_fast(
            target_head=target_head,
            pair_score_head=head,
            train_cached=train_cached,
            val_cached=val_cached,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            out_dir=out_dir,
            init_from_path=str(init_from) if init_from else None,
            config=config,
            save_extras=save_extras,
        )

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
            pair_weight=pair_weight,
            frac_weight=frac_weight, frac_baseline_mae=frac_baseline_mae,
            target_weight=target_weight,
            freeze_pair_head=freeze_pair_head,
            freeze_frac_head=freeze_frac_head,
            freeze_target_head=freeze_target_head,
        )
        val_metrics = _evaluate(
            stack, val_loader, device,
            pair_weight=pair_weight,
            frac_weight=frac_weight, frac_baseline_mae=frac_baseline_mae,
            target_weight=target_weight,
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

        # Per-epoch summary: always shows pair metrics; per-head blocks
        # only appear when the corresponding weight > 0 so the no-extra
        # paths keep their existing log format (smoke regression test).
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
        target_block = ""
        if target_weight > 0.0:
            target_block = (
                f"  ||  tr_tgt_top1={train_metrics.get('target_top1', 0):.3f} "
                f"val_tgt_top1={val_metrics.get('target_top1', 0):.3f} "
                f"val_tgt_top3={val_metrics.get('target_top3', 0):.3f}"
            )
        print(
            f"[pair_score] ep={epoch:3d} "
            f"tr_loss={train_metrics.get('loss', 0):.4f} "
            f"tr_top1={train_metrics.get('top1', 0):.3f}  |  "
            f"val_loss={val_metrics.get('loss', 0):.4f} "
            f"val_top1={val_metrics.get('top1', 0):.3f} "
            f"val_top3={val_metrics.get('top3', 0):.3f} "
            f"val_top5={val_metrics.get('top5', 0):.3f} "
            f"rand={val_metrics.get('random_valid_top1', 0):.3f}"
            f"{frac_block}{target_block}  "
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
            "target_weight": target_weight,
            "config": {
                "d_model": args.d_model,
                "hidden": args.hidden,
                "max_planets": args.max_planets,
                "max_fleets": args.max_fleets,
                "n_history": args.n_history,
                "frac_hidden": int(getattr(args, "frac_hidden", args.hidden) or args.hidden),
                "target_hidden": int(getattr(args, "target_hidden", args.hidden) or args.hidden),
                "target_use_entity": bool(getattr(target_head, "use_entity", False)) if target_head is not None else False,
                "target_num_layers": int(getattr(target_head, "num_layers", 2)) if target_head is not None else 2,
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
    pair_weight=1.0,
    frac_weight=0.0,
    frac_hidden=128,
    target_weight=0.0,
    target_hidden=128,
    target_use_entity=True,
    target_num_layers=3,
    freeze_pair_head=False,
    freeze_frac_head=False,
    freeze_target_head=False,
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
        pair_weight=float(pair_weight),
        frac_weight=float(frac_weight),
        frac_hidden=int(frac_hidden),
        target_weight=float(target_weight),
        target_hidden=int(target_hidden),
        target_use_entity=bool(target_use_entity),
        target_num_layers=int(target_num_layers),
        freeze_pair_head=bool(freeze_pair_head),
        freeze_frac_head=bool(freeze_frac_head),
        freeze_target_head=bool(freeze_target_head),
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
    p.add_argument("--pair-weight", type=float, default=1.0,
                   help="Weight on pair_loss in the total. 1.0 (default) "
                        "= include normally. Set to 0.0 for target-only "
                        "or frac-only stages: pair_metrics are still "
                        "computed for the log, but pair_loss does not "
                        "contribute gradients (encoders won't shift "
                        "toward the pair objective). Combine with the "
                        "matching --freeze-*-head flags to fully isolate "
                        "one head.")
    p.add_argument("--freeze-pair-head", action="store_true",
                   help="Freeze the pair-score head (no gradient flows). "
                        "Stage-2 frac-only pattern: pair head + all "
                        "encoders frozen, only FracHead trains. Combine "
                        "with --init-from <pair-trained best.pt>.")
    p.add_argument("--freeze-frac-head", action="store_true",
                   help="Freeze the FracHead. Pairs with --init-from to "
                        "preserve a previously-trained frac while training "
                        "other heads.")
    p.add_argument("--freeze-target-head", action="store_true",
                   help="Freeze the TargetHead. Pairs with --init-from to "
                        "preserve a previously-trained target head while "
                        "training other heads.")
    p.add_argument("--target-weight", type=float, default=0.0,
                   help="Weight for the per-planet target-attractiveness CE "
                        "loss (0.0 = disabled, no TargetHead built). 0.5 is "
                        "a sane joint-training default.")
    p.add_argument("--target-hidden", type=int, default=128,
                   help="Hidden width of the TargetHead MLP. "
                        "Ignored when --target-weight 0.")
    p.add_argument("--target-use-entity", dest="target_use_entity",
                   action="store_true", default=True,
                   help="v2 TargetHead: concat the pre-cross-attention "
                        "entity token alongside glob/ctx (3·d input). "
                        "Default. Use --target-no-entity for the v1 2d head.")
    p.add_argument("--target-no-entity", dest="target_use_entity",
                   action="store_false",
                   help="Force the v1 TargetHead (no entity input).")
    p.add_argument("--target-num-layers", type=int, default=3,
                   help="MLP depth for the TargetHead. v1 used 2; v2 "
                        "default is 3. Resume runs override this from "
                        "the prior ckpt's config to keep state_dict shapes "
                        "consistent.")
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
