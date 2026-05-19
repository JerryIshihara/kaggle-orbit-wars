"""Target ranker — two-stage attention over candidate target planets.

**Goal:** predict ``target_planet_idx`` for an expert action. A target may
be any real planet (own / neutral / enemy); a source must be an
**owned-source** planet (learner-owned with at least one ship). The
ranker scores every real planet candidate with a single cross-entropy
loss.

**A note on the source mask.** ``src_valid`` here is the "owned-source"
mask, NOT the runtime surplus-based "launchable" mask. The action
featurizer computes ``own ∧ ships > 0``; the dataset force-includes the
expert's actual source on acted rows so BC supervision is never lost
to a stricter heuristic. Runtime call sites tighten the mask with the
surplus check at inference time — that train/inference mismatch is
intentional and load-bearing; do not "fix" it by tightening training to
match the runtime, you'll discard expert launches.

**Architecture (two attention stages on top of the existing encoder stack):**

  1. Upstream encoders (unchanged): FleetEncoder + PlanetEncoder +
     PlanetEntityEncoder + CrossEntityAttention produce ``entity_now``,
     ``ctx_now``, ``glob`` over real planets.
  2. Per-planet token assembly: project
     ``[ctx_now ‖ entity_now ‖ glob ‖ target_scalars]`` → ``target_base``.
     The same token pool is shared between "source role" and "target role";
     learned role embeddings disambiguate the two uses.
  3. **Stage A — target→source cross-attention.** Each candidate target
     queries the set of owned-source planets. ``key_padding_mask`` =
     ``~src_valid``; an additional **diagonal attn_mask** excludes
     ``source == target`` (a planet cannot launch to itself, even when it
     is a valid target for reinforcement).
  4. **Stage B — target self-attention.** Source-aware target tokens
     attend to every other candidate target. ``key_padding_mask`` =
     ``~target_valid``. This is where the model learns "is this target
     the best of N options".
  5. Final per-target MLP scorer over
     ``[rank_ctx ‖ source_aware ‖ target_base ‖ target_scalars]``.

**Information-flow guardrails:**

  * Cross-attention is a *learned source-conditioned summary* — not
    strictly lossless, but far richer than the old PairScoreHead
    column-reduce (max/top2/lse/count/mean → 5 scalars).
  * Source-self exclusion is *pairwise* (only source==target masked, not
    "source masked for all targets"), so attention with a single valid
    source (which also happens to be the current target) doesn't shadow
    the rest of the candidates. ``nan_to_num`` guards the edge case
    where every source ends up masked for some row.
  * Loss-time mask: ``target_valid = planet_mask`` with the gold target
    force-included. Without the force-include the loss collapses to
    ``log(P)`` when the side-cache mask is missing (the bug fixed in
    the previous iteration).

**No PairScoreHead in this design.** The pair head's information was
already lossy (column-reduce to 5 scalars per target) and its
``torch.finfo.min`` masking caused a numerical collapse downstream. The
direct target→source attention replaces it cleanly.

Run (CLI):

    python -m agents.transformer_v2.pretrain.target_rank \\
        --encoder-ckpt data/runs/action/<TS>/action_best.pt \\
        --player Ebi --filter all \\
        --epochs 8 --batch-size 64 --lr 5e-4 --device cuda \\
        --out-dir data/runs/target_rank/Ebi_<TS>
"""

from __future__ import annotations

import argparse
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
    acted_only_indices,
    prepare_dataset,
)
from ..aggregator import CrossEntityAttention
from ..encoder import FleetEncoder, PlanetEncoder, PlanetEntityEncoder
from ..history import HISTORY_OFFSETS, N_HISTORY

# Channel indices into ``planet_features`` (mirrors
# featurizer/planet_featurizer.py::SCALAR_DIM layout):
#   0    = is_comet
#   1..5 = owner one-hot (learner-relative; 1=learner, 2..4 enemies, 5 neutral)
#   6    = ships log-norm (garrison)
PFI_OWNER_SELF = 1
PFI_OWNER_ENEMY_START = 2
PFI_OWNER_ENEMY_END = 5      # exclusive — slots 2/3/4 are enemies
PFI_OWNER_NEUTRAL = 5
PFI_SHIPS_LOG = 6

C_AGG = 9      # 3 owner + garrison + n_fr + n_en + near_en + inb_own + inb_enemy


# ---------- Helpers ----------
def _current_planet_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Current-turn slice of ``planet_features``.

    ``planet_features`` is one of the history-stacked keys in
    :class:`CrossEntitySnapshotDataset._STACK_KEYS`: when ``n_history >= 1``
    it always has rank 4 (``(B, T, P, D)``), regardless of the specific
    ``n_history`` value, so we slice by rank — not by ``shape[1] == 3``.
    For an unstacked input (rank 3) we pass through.
    """
    pf = batch["planet_features"]
    if pf.dim() == 4:
        return pf[:, -1]
    return pf


def _target_agg_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-planet hand-built scalars. ``(B, P, C_AGG=9)``.

    Owner one-hot (3) + garrison_log (1) + n_friendly_R (1) + n_enemy_R (1)
    + nearest_enemy_dist (1) + inbound_own_h10 (1) + inbound_enemy_h10 (1).

    All values already normalized by the upstream featurizer. Only
    ``planet_features`` is history-stacked (per
    :attr:`CrossEntitySnapshotDataset._STACK_KEYS`); the ``ships_arriving``
    /``n_*_within_R_norm`` /``nearest_enemy_dist_norm`` labels are
    current-step-only, so we read them straight from the batch.
    """
    pf = _current_planet_features(batch)                             # (B, P, D_p)
    inbound = batch["ships_arriving_within_10"]                      # (B, P, 4) — not stacked
    n_friend = batch["n_friendly_within_R_norm"]                     # (B, P) — not stacked
    n_enemy = batch["n_enemy_within_R_norm"]                         # (B, P) — not stacked
    near_enemy = batch["nearest_enemy_dist_norm"]                    # (B, P) — not stacked

    owner_friendly = pf[..., PFI_OWNER_SELF]                                       # (B, P)
    owner_enemy = pf[..., PFI_OWNER_ENEMY_START:PFI_OWNER_ENEMY_END].sum(-1)       # (B, P)
    owner_neutral = pf[..., PFI_OWNER_NEUTRAL]                                     # (B, P)

    garrison = pf[..., PFI_SHIPS_LOG]                                # (B, P)
    inb_own = inbound[..., 0]                                        # (B, P) slot 0
    inb_enemy = inbound[..., 1:4].sum(-1)                            # (B, P) slots 1+2+3

    return torch.stack(
        [
            owner_friendly, owner_enemy, owner_neutral,
            garrison,
            n_friend, n_enemy, near_enemy,
            inb_own, inb_enemy,
        ],
        dim=-1,
    )                                                                # (B, P, 9)


# ---------- Core ranker module ----------
class TargetRanker(nn.Module):
    """Two-stage attention + scorer.

    Stage A: target→source cross-attention with **source-self exclusion**
             (a planet cannot launch to itself, even when it is a valid
             reinforcement target).
    Stage B: target self-attention over real-planet candidates.
    Scorer:  per-target MLP on a 4-way concat
             ``[rank_ctx ‖ source_aware ‖ target_base ‖ target_scalars]``.

    Parameter count @ d_model=128, d_rank=128, n_heads=4, mlp_hidden=128,
    mlp_layers=3:
      * token_proj            (3·64 + 9) → 128         ≈   25 856
      * source/target roles   2 × 128                   ≈      256
      * t2s_attn (MultiheadAttention, in_proj + out)    ≈   66 048
      * t2s_norm                                        ≈      256
      * t2t_attn                                        ≈   66 048
      * t2t_norm                                        ≈      256
      * scorer MLP (3·128 + 9 → 128 → 128 → 1)          ≈   66 049
      * Total                                            ≈ **224 769**
    """

    def __init__(
        self,
        *,
        d_model: int = 128,
        c_agg: int = C_AGG,
        d_rank: int = 128,
        n_heads: int = 4,
        mlp_hidden: int = 128,
        mlp_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mlp_layers < 2:
            raise ValueError(f"mlp_layers must be >= 2 (got {mlp_layers})")
        if d_rank % n_heads != 0:
            raise ValueError(
                f"d_rank={d_rank} must be divisible by n_heads={n_heads}"
            )
        self.d_model = int(d_model)
        self.c_agg = int(c_agg)
        self.d_rank = int(d_rank)
        self.n_heads = int(n_heads)

        # Per-planet token: [ctx ‖ entity ‖ glob ‖ scalars] = (3·d_model + c_agg)
        in_dim = 3 * d_model + c_agg
        self.token_proj = nn.Linear(in_dim, d_rank)

        # Role embeddings disambiguate the same token in source vs target
        # role. Small init keeps the symmetric breaking gentle at start.
        self.source_role = nn.Parameter(torch.zeros(d_rank))
        self.target_role = nn.Parameter(torch.zeros(d_rank))
        nn.init.trunc_normal_(self.source_role, std=0.02)
        nn.init.trunc_normal_(self.target_role, std=0.02)

        # Stage A — target→source cross-attention.
        self.t2s_attn = nn.MultiheadAttention(
            d_rank, n_heads, batch_first=True, dropout=dropout,
        )
        self.t2s_norm = nn.LayerNorm(d_rank)

        # Stage B — target self-attention.
        self.t2t_attn = nn.MultiheadAttention(
            d_rank, n_heads, batch_first=True, dropout=dropout,
        )
        self.t2t_norm = nn.LayerNorm(d_rank)

        # Final scorer: 4-way concat → MLP → logit.
        # score_feat = [rank_ctx ‖ source_aware ‖ target_base ‖ target_scalars]
        # dim        = 3·d_rank + c_agg
        score_dim = 3 * d_rank + c_agg
        layers: list[nn.Module] = []
        cur = score_dim
        for _ in range(mlp_layers - 1):
            layers.append(nn.Linear(cur, mlp_hidden))
            layers.append(nn.GELU())
            cur = mlp_hidden
        layers.append(nn.Linear(cur, 1))
        self.scorer = nn.Sequential(*layers)
        nn.init.zeros_(self.scorer[-1].bias)
        # Bigger final-layer init than the earlier head: end-to-end joint
        # training needs gradients large enough to escape the uniform-
        # softmax basin in epoch 1.
        nn.init.normal_(self.scorer[-1].weight, std=0.05)

    def forward(
        self,
        glob: torch.Tensor,            # (B, d_model)
        ctx_now: torch.Tensor,         # (B, P, d_model)
        entity_now: torch.Tensor,      # (B, P, d_model)
        target_scalars: torch.Tensor,  # (B, P, c_agg)
        *,
        src_valid: torch.Tensor,       # (B, P) bool — owned-source planets
        tgt_valid: torch.Tensor,       # (B, P) bool — valid target candidates
        return_attn: bool = False,
    ):
        """Returns ``target_logits`` of shape ``(B, P)`` when
        ``return_attn=False`` (default — matches the training contract).
        When ``return_attn=True`` returns ``(target_logits, t2s_attn)``
        where ``t2s_attn`` is ``(B, P_target, P_source)`` averaged across
        attention heads. Used by the inference visualizer to draw
        source→target edges on the replay canvas.
        """
        B, P, d = ctx_now.shape
        if d != self.d_model:
            raise ValueError(
                f"ctx_now d={d} but ranker built for d_model={self.d_model}"
            )

        # --- 1. Per-planet base token ----------------------------------
        glob_b = glob.unsqueeze(1).expand(B, P, d)
        target_raw = torch.cat(
            [ctx_now, entity_now, glob_b, target_scalars], dim=-1,
        )                                                                # (B, P, 3·d + c_agg)
        target_base = self.token_proj(target_raw)                        # (B, P, d_rank)

        # Role-aware variants. Adding a learned bias vector flips the
        # token's interpretation between source and target use even
        # though the underlying token is identical. Cheap (2·d_rank
        # params) and lets attention learn role-specific projections.
        target_query = target_base + self.target_role.view(1, 1, -1)
        source_keys = target_base + self.source_role.view(1, 1, -1)

        # --- 2. Stage A: target→source cross-attention -----------------
        # key_padding_mask: True = ignore (so ~src_valid masks padding +
        # non-owned planets).
        # attn_mask: True = forbidden. Diagonal eye(P) excludes source==
        # target on a per-(query, key) basis. A planet can't reinforce
        # itself even when it's a valid reinforcement target.
        diag_mask = torch.eye(P, dtype=torch.bool, device=target_base.device)
        source_ctx, t2s_attn = self.t2s_attn(
            query=target_query,
            key=source_keys,
            value=source_keys,
            key_padding_mask=~src_valid,
            attn_mask=diag_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        # NaN guard: if every key was masked for some row (rare — e.g.
        # only one owned source and it equals the target index),
        # MultiheadAttention emits NaN. Zero those out so they don't
        # corrupt downstream layers.
        source_ctx = torch.nan_to_num(source_ctx, nan=0.0, posinf=0.0, neginf=0.0)
        source_aware = self.t2s_norm(target_base + source_ctx)           # (B, P, d_rank)

        # --- 3. Stage B: target self-attention -------------------------
        # All real planets attend to all real planets (and to themselves —
        # self-attn doesn't need the diagonal exclusion of stage A).
        rank_ctx_raw, _ = self.t2t_attn(
            query=source_aware,
            key=source_aware,
            value=source_aware,
            key_padding_mask=~tgt_valid,
            need_weights=False,
        )
        rank_ctx_raw = torch.nan_to_num(
            rank_ctx_raw, nan=0.0, posinf=0.0, neginf=0.0,
        )
        rank_ctx = self.t2t_norm(source_aware + rank_ctx_raw)            # (B, P, d_rank)

        # --- 4. Score --------------------------------------------------
        score_feat = torch.cat(
            [rank_ctx, source_aware, target_base, target_scalars], dim=-1,
        )                                                                # (B, P, 3·d_rank + c_agg)
        target_logits = self.scorer(score_feat).squeeze(-1)              # (B, P)

        if return_attn:
            # t2s_attn: (B, P_target, P_source) when batch_first=True
            # and average_attn_weights=True. Each row sums to ~1 over
            # valid non-self sources (masked positions get 0).
            return target_logits, t2s_attn
        return target_logits


# ---------- Stack ----------
class TargetRankerStack(nn.Module):
    """Upstream encoders + cross-attn + TargetRanker.

    No PairScoreHead. Source-target interaction is handled inside the
    ranker's Stage A cross-attention.
    """

    ENCODER_MODULES: tuple[str, ...] = (
        "fleet_encoder", "planet_encoder", "entity_encoder", "cross",
    )

    def __init__(
        self,
        *,
        fleet_encoder: FleetEncoder,
        planet_encoder: PlanetEncoder,
        entity_encoder: PlanetEntityEncoder,
        cross: CrossEntityAttention,
        target_ranker: TargetRanker,
    ):
        super().__init__()
        self.fleet_encoder = fleet_encoder
        self.planet_encoder = planet_encoder
        self.entity_encoder = entity_encoder
        self.cross = cross
        self.target_ranker = target_ranker

    def unfreeze_all(self) -> None:
        for m in (
            self.fleet_encoder, self.planet_encoder, self.entity_encoder,
            self.cross, self.target_ranker,
        ):
            m.train()
            for p in m.parameters():
                p.requires_grad_(True)

    def _build_masks(
        self,
        batch: dict[str, torch.Tensor],
        mask_now: torch.Tensor,                                   # (B, P) bool
        P: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve ``src_valid`` and ``tgt_valid`` for the ranker.

        **Source-mask semantics.** ``src_valid`` here means "this planet
        could plausibly be a launching source" — concretely *owned by the
        learner and has at least one ship*. It is **not** the runtime
        surplus-based "launchable" rule. The mismatch is intentional:
        the runtime surplus heuristic disagrees with real expert launches
        on a meaningful fraction of rows, so a strict surplus mask would
        push the expert's actual source out of the valid set. The gold
        source is force-included below for any acted row, which keeps
        the BC supervision intact. Runtime call sites tighten the mask
        with the surplus check at inference time.

        Three layers of defense (in priority order):

          1. **Read** the side-cache mask from the batch (``src_valid`` is
             populated by :class:`ActionSnapshotDataset` from
             ``_masks/*.npz``).
          2. **Fallback for missing side-cache** — older / unrebuilt
             ``data.tgz`` packs ship without the mask side cache, so the
             batch's ``src_valid`` is then all-False per row. We fall
             back to a mask derived from **planet ownership** (the
             learner-relative owner one-hot at channel
             ``PFI_OWNER_SELF`` of ``planet_features``), AND-ed with
             ``mask_now`` so padding slots stay False. **Critically we
             do NOT fall back to ``mask_now`` directly** — that would
             let neutral / enemy planets serve as sources and the Stage
             A cross-attention would learn nonsense ("attack from
             enemy"). The target mask, by contrast, *can* be any real
             planet (own / neutral / enemy are all valid target types),
             so its fallback uses ``mask_now``.
          3. **Force-include the expert's chosen source / target** so
             Stage A always has at least one valid key for the gold
             target's row and the loss always has a valid class to
             maximize.
        """
        src_valid = batch.get("src_valid")
        tgt_valid = batch.get("tgt_valid")
        # When the key is missing entirely (e.g. a manual/synthetic
        # batch built outside ActionSnapshotDataset), seed src with
        # all-False so the owned-only fallback below populates it.
        # Crucially we do NOT seed src with mask_now here — that would
        # short-circuit the empty-row branch and let enemy / neutral
        # planets act as sources.
        if src_valid is None:
            src_valid = torch.zeros_like(mask_now)
        if tgt_valid is None:
            # Targets *can* be any real planet (own/neutral/enemy), so
            # mask_now is the correct fallback here.
            tgt_valid = mask_now
        src_valid = src_valid[..., :P].bool().clone()
        tgt_valid = tgt_valid[..., :P].bool().clone()

        # Owned-only fallback for src_valid. Derived from
        # planet_features owner one-hot (channel PFI_OWNER_SELF =
        # learner's slot in the learner-relative encoding). AND with
        # mask_now to clip padding slots.
        pf_now = _current_planet_features(batch)                     # (B, P, D_p)
        owned_now = (pf_now[..., PFI_OWNER_SELF] > 0.5) & mask_now    # (B, P) bool

        if mask_now.shape == src_valid.shape:
            src_count = src_valid.sum(dim=-1)
            tgt_count = tgt_valid.sum(dim=-1)
            owned_count = owned_now.sum(dim=-1)

            src_empty = src_count == 0
            tgt_empty = tgt_count == 0
            src_valid[src_empty] = owned_now[src_empty]
            tgt_valid[tgt_empty] = mask_now[tgt_empty]

            # Legacy/cache guard: older ActionSnapshotDataset builds
            # initialized masks all-False and then force-included only
            # the gold label. That produces avg_cand=1, CE=0, top1=1.
            # If a row has a single target candidate while multiple
            # real planets exist, rebuild the target mask from
            # planet_mask. Do the analogous source repair when there
            # are multiple owned planets but only one source candidate.
            tgt_gold_only = (tgt_count <= 1) & (mask_now.sum(dim=-1) > 1)
            src_gold_only = (src_count <= 1) & (owned_count > 1)
            tgt_valid[tgt_gold_only] = mask_now[tgt_gold_only]
            src_valid[src_gold_only] = owned_now[src_gold_only]

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
        return src_valid, tgt_valid

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_attn: bool = False,
    ):
        """Returns ``(target_logits, tgt_valid)`` by default.

        When ``return_attn=True``, returns
        ``(target_logits, tgt_valid, src_valid, t2s_attn)`` where
        ``t2s_attn`` is the Stage A target→source attention matrix
        ``(B, P_target, P_source)`` averaged across heads. Used by the
        inference visualizer to draw source→target edges on the replay
        canvas.

        Returning ``tgt_valid`` ensures the loss applies the same mask
        used inside Stage B — avoids the all-False-collapse bug that
        bit the prior iteration when the side-cache was missing.
        ``src_valid`` is also returned in the attn-mode tuple so the
        scorer knows which source positions are real (the attn matrix
        has zeros for padded sources, but the scorer needs the explicit
        mask to label edges by source planet id).
        """
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch, self.fleet_encoder, self.planet_encoder, self.entity_encoder,
        )
        ctx, glob = self.cross(entity_tokens, entity_mask)
        ctx_now = ctx[:, -1] if ctx.dim() == 4 else ctx
        entity_now = (
            entity_tokens[:, -1] if entity_tokens.dim() == 4 else entity_tokens
        )
        mask_now = entity_mask[:, -1] if entity_mask.dim() == 3 else entity_mask

        P = ctx_now.shape[1]
        mask_now = mask_now[..., :P].bool()
        src_valid, tgt_valid = self._build_masks(batch, mask_now, P)

        target_scalars = _target_agg_features(batch)[..., :P, :]
        if return_attn:
            target_logits, t2s_attn = self.target_ranker(
                glob, ctx_now, entity_now, target_scalars,
                src_valid=src_valid, tgt_valid=tgt_valid,
                return_attn=True,
            )
            return target_logits, tgt_valid, src_valid, t2s_attn
        target_logits = self.target_ranker(
            glob, ctx_now, entity_now, target_scalars,
            src_valid=src_valid, tgt_valid=tgt_valid,
        )
        return target_logits, tgt_valid


# ---------- Loss + metrics ----------
def _target_rank_loss(
    target_logits: torch.Tensor,    # (B, P)
    tgt_idx: torch.Tensor,          # (B,)
    target_valid: torch.Tensor,     # (B, P) bool
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked cross-entropy + ranking diagnostics.

    ``target_valid`` is the FINAL mask (planet_mask ∪ gold) returned by
    ``TargetRankerStack.forward`` — callers must not pass the raw batch
    mask, which can be all-False when the side-cache is missing.
    """
    B, P = target_logits.shape
    valid = (tgt_idx >= 0) & (tgt_idx < P)
    n_valid = int(valid.sum().item())
    metrics = {
        "target_loss": 0.0,
        "target_top1": 0.0, "target_top3": 0.0, "target_top5": 0.0,
        "uniform_ce_baseline": math.log(float(P)),
        "avg_candidate_count": float(P),
        "target_logit_std": 0.0,
        "n_target_valid": float(n_valid),
    }
    if n_valid == 0:
        return target_logits.sum() * 0.0, metrics

    logits = target_logits[valid]                                    # (Nv, P)
    y = tgt_idx[valid]                                                # (Nv,)
    tv = target_valid[valid].bool()                                   # (Nv, P)
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(tv, logits, torch.full_like(logits, neg_inf))
    loss = F.cross_entropy(masked, y)

    with torch.no_grad():
        cand = tv.float().sum(dim=-1).clamp(min=1.0)
        metrics["target_loss"] = float(loss.item())
        metrics["uniform_ce_baseline"] = float(cand.log().mean().item())
        metrics["avg_candidate_count"] = float(cand.mean().item())
        k = min(5, P)
        top_ids = masked.topk(k, dim=-1).indices
        match = top_ids == y.unsqueeze(-1)
        metrics["target_top1"] = float(match[:, 0].float().mean().item())
        metrics["target_top3"] = float(match[:, : min(3, k)].any(-1).float().mean().item())
        metrics["target_top5"] = float(match.any(-1).float().mean().item())
        # Std of logits over **valid** positions only. The earlier
        # std-over-all-P version hid the "uniform-over-valid-targets"
        # failure mode by averaging in the padded -inf positions.
        tvf = tv.float()
        n_per_row = tvf.sum(dim=-1).clamp(min=1.0)
        mean_per_row = (logits * tvf).sum(dim=-1) / n_per_row
        sq_dev = ((logits - mean_per_row.unsqueeze(-1)) ** 2) * tvf
        std_per_row = (sq_dev.sum(dim=-1) / n_per_row).clamp(min=0).sqrt()
        metrics["target_logit_std"] = float(std_per_row.mean().item())
    return loss, metrics


# ---------- Loader ----------
def _load_encoder_stack(
    encoder_ckpt: Path,
    *,
    d_model: int,
    device: torch.device,
) -> tuple[
    FleetEncoder, PlanetEncoder, PlanetEntityEncoder, CrossEntityAttention,
]:
    """Load encoders from an action-stage ckpt.

    Each module is warm-started only when the ckpt's state-dict shapes
    match the local module's shapes. Otherwise the module is fresh-
    initialized and the mismatch logged. Common reasons a v1 action
    ckpt fails the shape check:

      * ``d_model`` change (v2 = 128 vs v1 = 64) → every encoder shape
        differs; everything fresh-inits.
      * L2 layer count (v2 = 2 vs v1 = 3) → cross fresh-inits.

    This loader never raises on shape mismatch — that lets callers point
    at any old ckpt as a "best effort" seed. The training loop logs
    exactly which modules were warm-started so the call site can decide
    whether to abort.
    """
    ckpt = torch.load(encoder_ckpt, map_location=device, weights_only=False)

    def _load_or_fresh(
        module: nn.Module, key: str, label: str,
    ) -> None:
        state = ckpt.get(key)
        if state is None:
            print(
                f"[target_rank] {label}: ckpt has no '{key}' — fresh init."
            )
            return
        try:
            module.load_state_dict(state, strict=True)
            print(f"[target_rank] {label}: loaded from {encoder_ckpt}.")
        except (RuntimeError, KeyError) as e:
            # RuntimeError on shape mismatch, KeyError on missing keys
            # in a partial dict. Either way, fresh-init and surface
            # the reason so callers can see why warm-start was skipped.
            msg = str(e).splitlines()[0][:200]
            print(
                f"[target_rank] {label}: shape/key mismatch vs ckpt "
                f"({msg}) — fresh init."
            )

    fenc = FleetEncoder(d_model=d_model)
    _load_or_fresh(fenc, "fleet_encoder", "FleetEncoder (L0)")

    penc = PlanetEncoder(d_model=d_model)
    _load_or_fresh(penc, "planet_encoder", "PlanetEncoder (L0)")

    eenc = PlanetEntityEncoder(d_model=d_model)
    _load_or_fresh(eenc, "entity_encoder", "PlanetEntityEncoder (L1)")

    cross = CrossEntityAttention(d_model=d_model)
    _load_or_fresh(cross, "cross", "CrossEntityAttention (L2)")

    for m in (fenc, penc, eenc, cross):
        m.to(device)
    return fenc, penc, eenc, cross


# ---------- Train / eval ----------
def _train_one_epoch(
    stack: TargetRankerStack,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    stack.train()
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        tgt_idx = batch["target_planet_idx"].long()
        target_logits, tgt_valid = stack(batch)
        loss, metrics = _target_rank_loss(target_logits, tgt_idx, tgt_valid)
        if not metrics["n_target_valid"]:
            continue
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n_batches += 1
    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def _evaluate(
    stack: TargetRankerStack,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    stack.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        tgt_idx = batch["target_planet_idx"].long()
        target_logits, tgt_valid = stack(batch)
        _, metrics = _target_rank_loss(target_logits, tgt_idx, tgt_valid)
        if not metrics["n_target_valid"]:
            continue
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n_batches += 1
    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in sums.items()}


# ---------- Top-level entry points ----------
def train_target_rank(
    args: argparse.Namespace,
    *,
    dataset: ActionSnapshotDataset | None = None,
) -> Path:
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fenc, penc, eenc, cross = _load_encoder_stack(
        Path(args.encoder_ckpt), d_model=args.d_model, device=device,
    )

    if dataset is None:
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
            cache_dir=getattr(args, "cache_dir", None),
            rebuild_cache=bool(getattr(args, "rebuild_cache", False)),
        )
    else:
        print(f"[target_rank] using caller-provided dataset "
              f"({len(dataset)} snapshots)", flush=True)

    acted_idx = acted_only_indices(dataset)
    print(f"[target_rank] acted rows: {len(acted_idx)}", flush=True)
    if args.max_rows is not None:
        acted_idx = acted_idx[: args.max_rows]
        print(f"[target_rank] capped to {len(acted_idx)} rows", flush=True)

    if args.overfit:
        train_idx = acted_idx
        val_idx = acted_idx
    else:
        n_val = max(1, int(round(len(acted_idx) * args.val_frac)))
        train_idx = acted_idx[:-n_val]
        val_idx = acted_idx[-n_val:]
    print(f"[target_rank] train={len(train_idx)} val={len(val_idx)}", flush=True)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
    )

    ranker = TargetRanker(
        d_model=args.d_model,
        c_agg=C_AGG,
        d_rank=args.d_rank,
        n_heads=args.n_heads,
        mlp_hidden=args.head_hidden,
        mlp_layers=args.head_num_layers,
        dropout=args.dropout,
    ).to(device)
    stack = TargetRankerStack(
        fleet_encoder=fenc, planet_encoder=penc, entity_encoder=eenc,
        cross=cross, target_ranker=ranker,
    ).to(device)

    # Optional: resume from a prior target_rank ckpt (continue training).
    # Loads all 4 encoder modules + the target_ranker state. The fresh
    # AdamW state is intentional — we don't ship the optimizer state in
    # ckpts, so resume always restarts the moments. With a properly-
    # lowered LR (passed via args.lr) this is a clean "fine-tune from
    # where we left off" pattern, not a full hot-resume.
    init_from = getattr(args, "init_from", None)
    if init_from is not None:
        prior = torch.load(
            Path(init_from), map_location=device, weights_only=False,
        )
        loaded: list[str] = []
        skipped: list[tuple[str, str]] = []
        for name in TargetRankerStack.ENCODER_MODULES:
            if name not in prior:
                continue
            try:
                getattr(stack, name).load_state_dict(prior[name], strict=True)
                loaded.append(name)
            except (RuntimeError, KeyError) as e:
                skipped.append((name, str(e).splitlines()[0][:160]))
        if "target_ranker" in prior:
            try:
                stack.target_ranker.load_state_dict(
                    prior["target_ranker"], strict=True,
                )
                loaded.append("target_ranker")
            except (RuntimeError, KeyError) as e:
                skipped.append(("target_ranker", str(e).splitlines()[0][:160]))
        print(
            f"[target_rank] init_from: loaded {loaded} from {init_from} "
            f"(epoch={prior.get('epoch')})",
            flush=True,
        )
        for name, msg in skipped:
            print(f"[target_rank] init_from: skipped {name} ({msg})", flush=True)

    stack.unfreeze_all()
    trainable = [p for p in stack.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[target_rank] trainable params: {n_params:,}", flush=True)
    optim = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_path = out_dir / "target_rank_best.pt"
    last_path = out_dir / "target_rank_last.pt"
    log_path = out_dir / "log.json"
    log_entries: list[dict[str, Any]] = []
    t0 = time.time()

    config = {
        "d_model": args.d_model,
        "d_rank": args.d_rank,
        "n_heads": args.n_heads,
        "head_hidden": args.head_hidden,
        "head_num_layers": args.head_num_layers,
        "dropout": args.dropout,
        "max_planets": args.max_planets,
        "max_fleets": args.max_fleets,
        "n_history": args.n_history,
        "history_offsets": list(HISTORY_OFFSETS),
        "player": args.player,
        "filter": args.filter,
        "c_agg": C_AGG,
    }

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        tr = _train_one_epoch(stack, train_loader, optim, device)
        va = _evaluate(stack, val_loader, device)
        elapsed = time.time() - t0
        log = {
            "epoch": epoch,
            "elapsed_s": round(elapsed, 1),
            "epoch_s": round(time.time() - t_epoch, 1),
            "train": tr, "val": va,
        }
        log_entries.append(log)
        log_path.write_text(json.dumps(log_entries, indent=2))
        print(
            f"[target_rank] ep={epoch:3d} "
            f"tr_loss={tr.get('target_loss', 0):.4f} "
            f"tr_top1={tr.get('target_top1', 0):.3f} "
            f"tr_std={tr.get('target_logit_std', 0):.3f}  |  "
            f"val_loss={va.get('target_loss', 0):.4f} "
            f"val_top1={va.get('target_top1', 0):.3f} "
            f"val_top3={va.get('target_top3', 0):.3f} "
            f"val_top5={va.get('target_top5', 0):.3f}  "
            f"baseline_ce={va.get('uniform_ce_baseline', 0):.3f} "
            f"avg_cand={va.get('avg_candidate_count', 0):.1f}  "
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
            "cross": stack.cross.state_dict(),
            "target_ranker": stack.target_ranker.state_dict(),
        }
        torch.save(payload, last_path)
        val_loss = va.get("target_loss", float("inf"))
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(torch.load(last_path, weights_only=False), best_path)

    print(f"[target_rank] done. best_val_loss={best_val_loss:.4f}  "
          f"ckpts: {best_path.name}, {last_path.name}", flush=True)
    return best_path


def train_target_rank_kwargs(
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
    lr=5e-4,
    weight_decay=0.0,
    epochs=8,
    d_model=128,
    d_rank=128,
    n_heads=4,
    head_hidden=128,
    head_num_layers=3,
    dropout=0.0,
    max_planets=64,
    max_fleets=1024,
    n_history=N_HISTORY,
    device=None,
    cache_dir=None,
    rebuild_cache=False,
    dataset=None,
    init_from=None,
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
        d_rank=d_rank,
        n_heads=n_heads,
        head_hidden=head_hidden,
        head_num_layers=head_num_layers,
        dropout=dropout,
        max_planets=max_planets,
        max_fleets=max_fleets,
        n_history=n_history,
        device=device,
        out_dir=Path(out_dir),
        cache_dir=(Path(cache_dir) if cache_dir is not None else None),
        rebuild_cache=bool(rebuild_cache),
        init_from=(Path(init_from) if init_from is not None else None),
    )
    return train_target_rank(args, dataset=dataset)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder-ckpt", type=Path, required=True)
    p.add_argument("--action-dir", type=Path, default=ACTION_DATASET_DIR)
    p.add_argument("--planet-dir", type=Path, default=PLANET_DATASET_DIR)
    p.add_argument("--fleet-dir", type=Path, default=FLEET_DATASET_DIR)
    p.add_argument("--entity-dir", type=Path, default=ENTITY_DATASET_DIR)
    p.add_argument("--cross-entity-dir", type=Path, default=CROSS_ENTITY_DATASET_DIR)
    p.add_argument("--filter", choices=["winner", "all"], default="all")
    p.add_argument("--player", default=None)
    p.add_argument("--replay-dir", type=Path, default=Path("data/replays"))
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-rank", type=int, default=128,
                   help="Internal width of the target ranker (post token-proj). "
                        "Must be divisible by --n-heads.")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--head-hidden", type=int, default=128)
    p.add_argument("--head-num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--max-planets", type=int, default=64)
    p.add_argument("--max-fleets", type=int, default=1024)
    p.add_argument("--n-history", type=int, default=N_HISTORY)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Path to a prior target_rank_best.pt to resume from. "
                        "Loads all 4 encoders + the target_ranker state "
                        "before optimizer setup. Optimizer state is NOT "
                        "carried — pass a lowered --lr for the continued "
                        "phase.")
    args = p.parse_args()
    train_target_rank(args)


if __name__ == "__main__":
    main()
