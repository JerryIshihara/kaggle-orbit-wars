"""Player-level consolidation block.

Branches off ``CrossEntityAttention`` (L2). L2 ends world perception with
per-planet contextual tokens ``ctx_now (B, P, d_model)`` and a single CLS
``glob (B, d_model)``. This module compresses the per-planet view into
**one strategic embedding per player**: ``player_state (B, 4, d_model)``.

Player slots are **learner-relative** to match the rest of the stack
(``planet_featurizer.py`` writes owner one-hots as
``[self, opp1, opp2, opp3, neutral]``). Slot 0 is always the learner. The
critic reads ``player_state[:, 0]``.

Architecture:

  1. **Owner additive encoding** — a 5-dim learner-relative owner one-hot
     (the relevant slice of the planet feature vector) is projected to
     ``d_model`` via ``Linear(5, d_model, bias=False)`` and **added** to
     ``ctx_now``. ``bias=False`` keeps planets at their L2 representation
     when no owner one-hot is supplied; otherwise the projection learns a
     per-owner offset (including a learned neutral offset, because the
     5-slot encoding has an explicit neutral channel — there is no magic
     "neutral = zero" assumption).

  2. **Player CLS cross-attention** — 4 learned query tokens (one per
     learner-relative player slot) cross-attend over the owner-encoded
     entity tokens. ``key_padding_mask = ~planet_mask`` excludes padding
     and dead planets. Neutrals stay readable (they're context for every
     player) — they are not masked here.

  3. **FFN + LayerNorm** — a standard pre-LN transformer block's second
     half on each player token.

  4. **Player self-attention** (optional, on by default) — a single
     pre-LN transformer encoder layer over the 4 player tokens so
     each player's state sees the others.

Returns ``player_state (B, 4, d_model)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Learner-relative owner one-hot is 5-dim: 4 player slots + neutral.
# Slot 0 is always the learner; slots 1-3 are opponents in fixed order;
# slot 4 is neutral. The four player tokens emitted by the consolidator
# correspond to slots 0..3 — neutral has no CLS token (it is context for
# every player, not a player itself).
_N_OWNER_CHANNELS: int = 5
_N_PLAYER_SLOTS: int = 4


class OwnerOneHotEncoding(nn.Module):
    """Project a 5-dim learner-relative owner one-hot to ``d_model``.

    ``bias=False`` so that a row of zeros (no owner info supplied) maps to
    zero — callers can pass an all-zero one-hot to disable this signal
    without disturbing the L2 representation. With a real one-hot the
    module learns a per-owner additive offset.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(_N_OWNER_CHANNELS, d_model, bias=False)
        # Small init so the additive signal starts gentle and L2's
        # representation dominates at step 0.
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, owner_oh: torch.Tensor) -> torch.Tensor:
        """``owner_oh (B, P, 5)`` → ``(B, P, d_model)``."""
        return self.proj(owner_oh)


class PlayerConsolidator(nn.Module):
    """Per-player strategic state from L2 entity tokens.

    Forward inputs:
      ctx_now       (B, P, d_model)    L2 per-planet contextual tokens.
      planet_mask   (B, P) bool        True = real planet (incl. neutrals).
      owner_oh      (B, P, 5)          learner-relative owner one-hot
                                       [self, opp1, opp2, opp3, neutral].

    Outputs:
      player_state  (B, 4, d_model)    per-player strategic embedding,
                                       learner-relative slot ordering.
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        n_heads: int = 8,
        ff_mult: int = 2,
        dropout: float = 0.0,
        n_player_self_attn_layers: int = 1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.d_model = d_model
        self.n_heads = n_heads

        # 1) Owner additive encoding (5-dim one-hot → d_model, no bias).
        self.owner_enc = OwnerOneHotEncoding(d_model)
        # Multiplicative scale on the owner additive signal. Init small
        # (0.1) so L2's ``ctx_now`` representation dominates at step 0;
        # the model is free to up-weight the owner tag if the task
        # demands it. Mirrors the FiLM ``alpha`` pattern in PairHead.
        self.owner_scale = nn.Parameter(torch.tensor(0.1))

        # 2) Four learned query tokens (one per learner-relative slot).
        #    Same init pattern as ``CrossEntityAttention.cls`` and the role
        #    bias vectors in ``DualRoleAttention``.
        self.player_cls = nn.Parameter(torch.zeros(_N_PLAYER_SLOTS, d_model))
        nn.init.trunc_normal_(self.player_cls, std=0.02)
        # Multiplicative scale on the player CLS queries. Init 1.0 — the
        # CLS tokens are already small via trunc_normal_(std=0.02); the
        # scale is a degree-of-freedom for the model to amplify or
        # suppress the slot identity signal independently of token norm.
        self.player_scale = nn.Parameter(torch.tensor(1.0))

        # 3) Player-CLS cross-attention (pre-LN).
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )

        # 4) FFN on each player token (pre-LN).
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

        # 5) Optional player self-attention over the 4 player tokens.
        self.n_player_self_attn_layers = int(n_player_self_attn_layers)
        if self.n_player_self_attn_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * ff_mult,
                batch_first=True,
                activation="gelu",
                dropout=dropout,
                norm_first=True,
            )
            try:
                self.player_self = nn.TransformerEncoder(
                    layer,
                    num_layers=self.n_player_self_attn_layers,
                    enable_nested_tensor=False,
                )
            except TypeError:
                self.player_self = nn.TransformerEncoder(
                    layer, num_layers=self.n_player_self_attn_layers,
                )
        else:
            self.player_self = None

    def forward(
        self,
        ctx_now: torch.Tensor,        # (B, P, d_model)
        planet_mask: torch.Tensor,    # (B, P) bool
        owner_oh: torch.Tensor,       # (B, P, 5)
    ) -> torch.Tensor:
        B, P, d = ctx_now.shape
        if d != self.d_model:
            raise ValueError(
                f"ctx_now d={d} but module built for d_model={self.d_model}"
            )
        if owner_oh.shape[:2] != (B, P) or owner_oh.shape[-1] != _N_OWNER_CHANNELS:
            raise ValueError(
                f"owner_oh shape {tuple(owner_oh.shape)} expected (B={B}, "
                f"P={P}, 5)"
            )

        # 1) Owner additive encoding on the KV stream, scaled.
        kv_tokens = ctx_now + self.owner_scale * self.owner_enc(owner_oh)

        # 2) Player queries broadcast to batch, scaled.
        q_tokens = (
            self.player_scale * self.player_cls
        ).unsqueeze(0).expand(B, -1, -1)                            # (B, 4, d)

        # 3) Pre-LN cross-attention. ``key_padding_mask`` uses True = ignore.
        q_n = self.cross_norm_q(q_tokens)
        kv_n = self.cross_norm_kv(kv_tokens)
        attn_out, _ = self.cross_attn(
            q_n, kv_n, kv_n,
            key_padding_mask=~planet_mask,
            need_weights=False,
        )
        # Snapshots with zero real planets would produce NaNs from
        # masked-softmax; guard them.
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        players = q_tokens + attn_out                                # (B, 4, d)

        # 4) Pre-LN FFN + residual.
        players = players + self.ffn(self.ffn_norm(players))         # (B, 4, d)

        # 5) Optional self-attention over the 4 player tokens.
        if self.player_self is not None:
            # All 4 slots are always valid (no padding among players).
            players = self.player_self(players)                      # (B, 4, d)

        return players
