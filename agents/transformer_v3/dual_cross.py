"""Dual-rate L2: two parallel CrossEntityAttention branches + zero-init fusion.

Drop-in replacement for the single ``CrossEntityAttention`` in
``EntityPretrainModel``: same call signature, same return arity, so the
v2 forward path needs no edits. The temporal input is the 18-frame
UNION stack (see ``transformer_v3.history``); each branch
index-selects its own 10 frames.

Return-shape contract (deliberate):

  * rank-4 input ``(B, N_UNION, P, d)`` -> returns ``(B, 1, P, d)``
    containing ONLY the fused current step. The v2 callers do
    ``ctx_full[:, -1]`` right after, which picks exactly that frame —
    per-history-step contextual tokens were never consumed downstream
    in v2, so we don't pay to fuse them.
  * rank-3 input ``(B, P, d)`` (cold-start / probe paths) -> both
    branches see the single frame as T=1 and the fused ``(B, P, d)``
    is returned, mirroring v2 single-frame semantics.

Fusion is ``Linear(2d -> d)`` with weight initialized ``[I | 0]`` and
zero bias: at init the module computes exactly the LONG branch, so a
v2 warm start reproduces v2 outputs bit-for-bit (see
``smoke_dual_rate.py`` for the assertion) and the SHORT branch fades
in through training. Tokens and the global CLS readout get separate
fusion layers (planet-space and snapshot-space shouldn't share one
adaptation surface).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..transformer_v2.aggregator.cross_entity import CrossEntityAttention
from ..transformer_v2.pretrain.entity_encoder import ENTITY_N_OWNER_CLASSES
from .history import (
    LONG_SLOT_IDX,
    SHORT_SLOT_IDX,
    N_BRANCH,
    N_UNION,
)

N_PLAYER_TOKENS = 4


def _zero_init_fusion(d_model: int) -> nn.Linear:
    """Linear(2d -> d) computing ``x_long`` exactly at init."""
    fuse = nn.Linear(2 * d_model, d_model)
    with torch.no_grad():
        fuse.weight.zero_()
        fuse.weight[:, :d_model].copy_(torch.eye(d_model))
        fuse.bias.zero_()
    return fuse


class DualRateCrossEntity(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        # Same constructor knobs as the v2 L2 for both branches; each
        # branch owns its 10-row step_embed table.
        self.long = CrossEntityAttention(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            ff_mult=ff_mult, dropout=dropout, n_steps=N_BRANCH,
            n_player_tokens=N_PLAYER_TOKENS,
        )
        self.short = CrossEntityAttention(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            ff_mult=ff_mult, dropout=dropout, n_steps=N_BRANCH,
            n_player_tokens=N_PLAYER_TOKENS,
        )
        self.fuse_tokens = _zero_init_fusion(d_model)
        self.fuse_glob = _zero_init_fusion(d_model)
        # v3.1 l2_tokens player state (replaces the PlayerConsolidator):
        # learner-relative owner one-hots are zero-init projected onto the
        # entity tokens BEFORE the branches (sharpens owner routing for
        # the player readers without disturbing the warm-started planet
        # path at init), and each branch's 4 player reader tokens are
        # fused like the global CLS.
        self.owner_proj = nn.Linear(ENTITY_N_OWNER_CLASSES, d_model, bias=False)
        with torch.no_grad():
            self.owner_proj.weight.zero_()
        self.fuse_player = _zero_init_fusion(d_model)
        self.last_player_state: torch.Tensor | None = None
        # Buffer-free index tuples (python ints survive state_dict round
        # trips trivially and index_select accepts lists).
        self._long_idx = list(LONG_SLOT_IDX)
        self._short_idx = list(SHORT_SLOT_IDX)
        # PRE-FUSION current-step branch tokens, stashed each forward
        # (graph-attached). The short-horizon aux heads read these to
        # supervise the short branch past the zero-init fusion gate.
        # Consumers must read them between THEIR forward and the next
        # one (each forward overwrites).
        self.last_ctx_long_now: torch.Tensor | None = None
        self.last_ctx_short_now: torch.Tensor | None = None

    def forward(
        self,
        entity_tokens: torch.Tensor,   # (B, N_UNION, P, d) or (B, P, d)
        entity_mask: torch.Tensor,     # (B, N_UNION, P)    or (B, P)
        owner_oh: torch.Tensor | None = None,  # (B, N_UNION, P, 5)/(B, P, 5)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if owner_oh is not None:
            # Zero-init projection: identical tokens at init, owner
            # routing fades in. None (legacy callers) == zeros.
            entity_tokens = entity_tokens + self.owner_proj(
                owner_oh.to(entity_tokens.dtype))
        if entity_tokens.dim() == 3:
            # Single frame: both branches see it as their current step
            # (step_embed[-1:]), fusion picks long-at-init.
            ctx_l, glob_l = self.long(entity_tokens, entity_mask)
            pl_l = self.long.last_player_tokens
            ctx_s, glob_s = self.short(entity_tokens, entity_mask)
            pl_s = self.short.last_player_tokens
            self.last_ctx_long_now, self.last_ctx_short_now = ctx_l, ctx_s
            ctx = self.fuse_tokens(torch.cat([ctx_l, ctx_s], dim=-1))
            glob = self.fuse_glob(torch.cat([glob_l, glob_s], dim=-1))
            self.last_player_state = self.fuse_player(
                torch.cat([pl_l, pl_s], dim=-1))
            return ctx, glob

        B, T, P, d = entity_tokens.shape
        if T != N_UNION:
            raise ValueError(
                f"DualRateCrossEntity expects the {N_UNION}-frame UNION "
                f"stack (transformer_v3.history.UNION_HISTORY_OFFSETS), "
                f"got T={T}. Restack the dataset/deque with the union "
                f"offsets — branch windows are index-selected internally."
            )
        ctx_l_full, glob_l = self.long(
            entity_tokens[:, self._long_idx], entity_mask[:, self._long_idx],
        )
        pl_l = self.long.last_player_tokens
        ctx_s_full, glob_s = self.short(
            entity_tokens[:, self._short_idx], entity_mask[:, self._short_idx],
        )
        pl_s = self.short.last_player_tokens
        # Current step of each branch (both branch windows end at offset 0).
        self.last_ctx_long_now = ctx_l_full[:, -1]
        self.last_ctx_short_now = ctx_s_full[:, -1]
        ctx = self.fuse_tokens(
            torch.cat([ctx_l_full[:, -1], ctx_s_full[:, -1]], dim=-1)
        )
        glob = self.fuse_glob(torch.cat([glob_l, glob_s], dim=-1))
        self.last_player_state = self.fuse_player(
            torch.cat([pl_l, pl_s], dim=-1))
        # Rank-4 out so the caller's ``ctx_full[:, -1]`` lands on the
        # fused frame.
        return ctx.unsqueeze(1), glob
