"""Pair-score scorer with a shared 2-layer trunk and 5 output heads.

After the L1-L4 entity stack, this head produces five complementary
predictions per snapshot from the same `(B, P, P, 6·d_pair)` broadcast
tensor. The first two layers of the original PairHead scorer become a
**shared trunk** that all heads consume; each head is a single Linear
on top of the 256-wide trunk output.

Heads (per snapshot, output shapes):

  * ``pair_logits   (B, P, P)``  per-cell source→target compatibility
  * ``glob_act      (B,)``       "any action this turn?"
  * ``source_act    (B, P)``     "does planet p launch a fleet?"
  * ``target_aim    (B, P)``     "is planet p a target?"
  * ``pair_frac     (B, P, P)``  fraction of source's ships sent to t,
                                  in [0, 1]; supervised on positive
                                  cells only

Trunk structure (matches the spec discussed in design review):

    pair_feat  (B, P, P, 6·d_pair=768)
       │ Linear(768 → 256), GELU
       ▼
    h1         (B, P, P, 256)
       │ Linear(256 → 256), GELU
       ▼
    trunk      (B, P, P, 256)              ←── all 5 heads consume this

Heads:

    pair_head        : Linear(256 → 1)  on trunk[b, s, t, :]
    pair_frac_head   : Linear(256 → 1)  on trunk[b, s, t, :], sigmoid
    source_act_head  : Linear(256 → 1)  on mean(trunk, dim=target)
    target_aim_head  : Linear(256 → 1)  on mean(trunk, dim=source)
    glob_act_head    : Linear(256 → 1)  on mean(trunk, dim=(source, target))

Pooling for the per-planet / snapshot heads uses **valid-mask-aware
mean**: pad cells contribute zero with denominator counted from
``pair_valid``. Caller passes ``pair_valid (B, P, P) bool`` so masking
stays correct regardless of how many planets are real.

The head returns a dict of raw logits / fractions; the loss layer is
responsible for masking, BCE/MSE selection, and the pos_weight knobs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairHead(nn.Module):
    """Trunk + 5 heads sharing the (B, P, P, 256) representation.

    Forward inputs:
      source_joint  (B, P, d_model)   from :class:`JointRoleAttention`.
      target_joint  (B, P, d_model)   from :class:`JointRoleAttention`.
      ctx_now       (B, P, d_model)   from :class:`CrossEntityAttention`.
      pair_valid    (B, P, P) bool    optional; used by the per-planet
                                       / snapshot pools.

    Returns ``dict[str, Tensor]``:
      ``pair_logits  (B, P, P)``,
      ``pair_frac    (B, P, P)`` raw logit (caller sigmoids in loss),
      ``source_act   (B, P)``,
      ``target_aim   (B, P)``,
      ``glob_act     (B,)``.
    """

    HEAD_NAMES: tuple[str, ...] = (
        "pair_logits", "pair_frac", "source_act", "target_aim", "glob_act",
    )

    def __init__(
        self,
        d_model: int = 256,
        *,
        d_pair: int = 128,
        trunk_hidden: int = 256,
        c_scalars: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_pair = d_pair
        self.trunk_hidden = trunk_hidden
        self.c_scalars = c_scalars

        # Three independent projections — source, target, and ctx
        # (shared between source and target context use in the concat).
        self.src_proj = nn.Linear(d_model, d_pair)
        self.tgt_proj = nn.Linear(d_model, d_pair)
        self.ctx_proj = nn.Linear(d_model, d_pair)

        feat_dim = 6 * d_pair + c_scalars
        # 2-layer shared trunk. All 5 output heads consume the post-trunk
        # ``(B, P, P, trunk_hidden)`` token.
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, trunk_hidden),
            nn.GELU(),
            *([nn.Dropout(dropout)] if dropout > 0 else []),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            *([nn.Dropout(dropout)] if dropout > 0 else []),
        )

        # 5 single-Linear heads. Init final Linears small so logits
        # start near zero (sigmoid ~0.5) — lets the pos_weight-amplified
        # BCE pull them apart without first overcoming a random bias.
        self.pair_head = nn.Linear(trunk_hidden, 1)
        self.pair_frac_head = nn.Linear(trunk_hidden, 1)
        self.source_act_head = nn.Linear(trunk_hidden, 1)
        self.target_aim_head = nn.Linear(trunk_hidden, 1)
        self.glob_act_head = nn.Linear(trunk_hidden, 1)
        for h in (
            self.pair_head, self.pair_frac_head,
            self.source_act_head, self.target_aim_head,
            self.glob_act_head,
        ):
            nn.init.zeros_(h.bias)
            nn.init.normal_(h.weight, std=0.02)

    @staticmethod
    def _masked_mean(
        x: torch.Tensor,        # (..., N, D)
        mask: torch.Tensor,     # (..., N) bool
        dim: int = -2,
    ) -> torch.Tensor:
        """Mean over ``dim`` with cells where ``mask`` is False excluded.

        Returns the same shape minus ``dim``. Rows with zero valid
        entries collapse to zero (safe for downstream Linears).
        """
        m = mask.unsqueeze(-1).to(x.dtype)
        weighted = x * m
        denom = m.sum(dim=dim).clamp(min=1.0)
        return weighted.sum(dim=dim) / denom

    def forward(
        self,
        source_joint: torch.Tensor,        # (B, P, d_model)
        target_joint: torch.Tensor,        # (B, P, d_model)
        ctx_now: torch.Tensor,             # (B, P, d_model)
        pair_valid: torch.Tensor | None = None,    # (B, P, P) bool
        pair_scalars: torch.Tensor | None = None,  # (B, P, P, c_scalars)
    ) -> dict[str, torch.Tensor]:
        B, P, d = source_joint.shape
        if d != self.d_model:
            raise ValueError(
                f"source_joint d={d} but PairHead built for d_model={self.d_model}"
            )

        src_r = self.src_proj(source_joint)        # (B, P, d_pair)
        tgt_r = self.tgt_proj(target_joint)
        ctx_r = self.ctx_proj(ctx_now)

        src_b  = src_r.unsqueeze(2).expand(B, P, P, self.d_pair)
        tgt_b  = tgt_r.unsqueeze(1).expand(B, P, P, self.d_pair)
        ctxs_b = ctx_r.unsqueeze(2).expand(B, P, P, self.d_pair)
        ctxt_b = ctx_r.unsqueeze(1).expand(B, P, P, self.d_pair)
        st     = src_b * tgt_b
        ctx_st = ctxs_b * ctxt_b

        feats: list[torch.Tensor] = [src_b, ctxs_b, tgt_b, ctxt_b, st, ctx_st]
        if pair_scalars is not None:
            if pair_scalars.shape[-1] != self.c_scalars:
                raise ValueError(
                    f"pair_scalars last-dim={pair_scalars.shape[-1]} "
                    f"but PairHead built for c_scalars={self.c_scalars}"
                )
            feats.append(pair_scalars)
        feat = torch.cat(feats, dim=-1)            # (B, P, P, 6·d_pair + c_sc)

        trunk = self.trunk(feat)                    # (B, P, P, trunk_hidden)

        # Pair-cell heads — direct Linear on (s, t) trunk cell.
        pair_logits = self.pair_head(trunk).squeeze(-1)       # (B, P, P)
        pair_frac   = self.pair_frac_head(trunk).squeeze(-1)  # (B, P, P) raw

        # Per-source / per-target / snapshot heads. Pool the trunk along
        # the orthogonal axis with valid-pair masking so padded cells
        # don't pollute the mean. If pair_valid is None, fall back to
        # plain mean (treats all P×P cells as valid).
        if pair_valid is None:
            source_pool = trunk.mean(dim=2)                    # (B, P, d)
            target_pool = trunk.mean(dim=1)                    # (B, P, d)
            glob_pool   = trunk.mean(dim=(1, 2))               # (B, d)
        else:
            # Source row pool: average across targets for each source.
            source_pool = self._masked_mean(trunk, pair_valid, dim=2)
            # Target col pool: average across sources for each target.
            # mask axis matches dim=1 → transpose mask first.
            target_pool = self._masked_mean(
                trunk.transpose(1, 2), pair_valid.transpose(1, 2), dim=2,
            )
            # Global pool: mean over (P, P) restricted to valid cells.
            mask_f = pair_valid.to(trunk.dtype).unsqueeze(-1)
            glob_pool = (
                (trunk * mask_f).sum(dim=(1, 2))
                / mask_f.sum(dim=(1, 2)).clamp(min=1.0)
            )

        source_act = self.source_act_head(source_pool).squeeze(-1)  # (B, P)
        target_aim = self.target_aim_head(target_pool).squeeze(-1)  # (B, P)
        glob_act   = self.glob_act_head(glob_pool).squeeze(-1)      # (B,)

        return {
            "pair_logits": pair_logits,
            "pair_frac":   pair_frac,
            "source_act":  source_act,
            "target_aim":  target_aim,
            "glob_act":    glob_act,
        }
