"""Parallel source/target role-conditioned cross-attention.

Sits one level above ``CrossEntityAttention``. The cross-entity layer
gives every planet a token contextualized over the global game state
(``ctx_now``). This layer adds two **parallel** role-aware views on
top:

  * **source-to-target** — each planet in its *source* role attends to
    every other planet in its *target* role, learning "if I launch from
    here, what targets matter?"
  * **target-to-source** — each planet in its *target* role attends to
    every other planet in its *source* role, learning "if I'm targeted,
    which sources matter?"

Both directions share the same per-planet token pool; learned
``source_role`` / ``target_role`` bias vectors disambiguate the two
uses cheaply (2·d_model parameters). Each direction is a single
:class:`torch.nn.MultiheadAttention` block with a residual + LayerNorm
identical to a transformer block's first half.

No source-self exclusion is applied here — that's a task-specific
constraint that belongs in downstream pair/target rankers, not in the
representation layer. Empty rows (all keys masked) are guarded with
``torch.nan_to_num`` so a snapshot with zero real planets stays
numerically safe.

Returns a tuple ``(source_aware, target_aware)`` both of shape
``(B, P, d_model)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DualRoleAttention(nn.Module):
    """Two parallel role-conditioned attention branches.

    Forward inputs:
      ctx_now      (B, P, d_model)  per-planet contextual tokens from
                                    :class:`CrossEntityAttention`.
      planet_mask  (B, P) bool      True = real planet.

    Outputs:
      source_aware (B, P, d_model)  source-role tokens enriched with
                                    target-role context.
      target_aware (B, P, d_model)  target-role tokens enriched with
                                    source-role context.
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        # Learned role bias vectors. Small init keeps the symmetry
        # breaking gentle so the two branches don't diverge wildly at
        # epoch 0.
        self.source_role = nn.Parameter(torch.zeros(d_model))
        self.target_role = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.source_role, std=0.02)
        nn.init.trunc_normal_(self.target_role, std=0.02)

        self.s2t_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.t2s_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.s2t_norm = nn.LayerNorm(d_model)
        self.t2s_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        ctx_now: torch.Tensor,         # (B, P, d_model)
        planet_mask: torch.Tensor,     # (B, P) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, P, d = ctx_now.shape
        if d != self.d_model:
            raise ValueError(
                f"ctx_now d={d} but module built for d_model={self.d_model}"
            )

        # Role-conditioned variants of the same per-planet token pool.
        source_tok = ctx_now + self.source_role.view(1, 1, -1)
        target_tok = ctx_now + self.target_role.view(1, 1, -1)

        # ``key_padding_mask`` uses True = ignore (inverse of planet_mask).
        kpm = ~planet_mask

        # Source-to-target: source queries see target keys/values.
        s2t_raw, _ = self.s2t_attn(
            query=source_tok, key=target_tok, value=target_tok,
            key_padding_mask=kpm, need_weights=False,
        )
        s2t_raw = torch.nan_to_num(s2t_raw, nan=0.0, posinf=0.0, neginf=0.0)
        source_aware = self.s2t_norm(source_tok + s2t_raw)

        # Target-to-source: target queries see source keys/values.
        t2s_raw, _ = self.t2s_attn(
            query=target_tok, key=source_tok, value=source_tok,
            key_padding_mask=kpm, need_weights=False,
        )
        t2s_raw = torch.nan_to_num(t2s_raw, nan=0.0, posinf=0.0, neginf=0.0)
        target_aware = self.t2s_norm(target_tok + t2s_raw)

        return source_aware, target_aware
