"""Joint source/target self-attention over the concatenated role tokens.

Sits one level above :class:`DualRoleAttention` in the current compact
post-perception ActionLearner. The dual-role layer gives each planet two
role-conditioned views — ``source_aware`` and ``target_aware`` — by attending
across roles (sources query targets and vice versa). This layer concatenates
both streams into one
``(B, 2P, d_model)`` sequence and runs a single Pre-LN
``TransformerEncoder`` over it, so every (slot, role) token attends to
every other (slot, role):

  * sources can see other sources directly (not only via the
    target-mediated path L3 provides),
  * targets can see other targets,
  * the cross-role paths L3 already established are preserved through
    residual + attention recomputation.

A fresh learned role embedding is added at entry — slots ``0..P-1``
get ``source_role``, slots ``P..2P-1`` get ``target_role`` — so the
encoder distinguishes the two halves unambiguously, independent of how
diluted the upstream role residual signal is.

Returns ``(source_joint, target_joint)`` both of shape
``(B, P, d_model)``, ready for the per-role 3-Linear decoders.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class JointRoleAttention(nn.Module):
    """Concat source+target tokens → self-attention → split.

    Forward inputs:
      source_aware  (B, P, d_model)   from :class:`DualRoleAttention`.
      target_aware  (B, P, d_model)   from :class:`DualRoleAttention`.
      planet_mask   (B, P) bool       True = real planet. Tiled across
                                      both role halves for the
                                      attention key-padding mask.

    Outputs:
      source_joint  (B, P, d_model)   source-mode tokens after global
                                      cross-role + same-role attention.
      target_joint  (B, P, d_model)   target-mode tokens, same.
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        n_heads: int = 8,
        n_layers: int = 1,
        ff_mult: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        # Fresh role bias vectors — independent of the ones in
        # DualRoleAttention so this layer can shape them for its own
        # objective without coupling. Small init for gentle symmetry
        # breaking at epoch 0.
        self.source_role = nn.Parameter(torch.zeros(d_model))
        self.target_role = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.source_role, std=0.02)
        nn.init.trunc_normal_(self.target_role, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
            norm_first=True,
        )
        # Pre-LN disables PyTorch's nested-tensor fast path; setting
        # this explicitly avoids a noisy warning on every Colab run.
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=n_layers, enable_nested_tensor=False,
            )
        except TypeError:
            # Older PyTorch versions do not expose ``enable_nested_tensor``.
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        source_aware: torch.Tensor,    # (B, P, d_model)
        target_aware: torch.Tensor,    # (B, P, d_model)
        planet_mask: torch.Tensor,     # (B, P) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, P, d = source_aware.shape
        if d != self.d_model:
            raise ValueError(
                f"source_aware d={d} but module built for d_model={self.d_model}"
            )

        # Add fresh role embeddings so the encoder can tell source-mode
        # slots from target-mode slots without depending on upstream
        # residual signal.
        src_tagged = source_aware + self.source_role.view(1, 1, -1)
        tgt_tagged = target_aware + self.target_role.view(1, 1, -1)

        # Concatenate along the sequence axis: slots 0..P-1 are source,
        # slots P..2P-1 are target. The encoder sees a 2P-long sequence
        # of mixed-role tokens.
        seq = torch.cat([src_tagged, tgt_tagged], dim=1)              # (B, 2P, d)

        # ``key_padding_mask`` uses True = ignore. Both halves use the
        # same planet validity mask (a planet is valid as both source
        # and target iff it's real).
        kpm = ~torch.cat([planet_mask, planet_mask], dim=1)           # (B, 2P)

        out = self.encoder(seq, src_key_padding_mask=kpm)             # (B, 2P, d)

        source_joint = out[:, :P]
        target_joint = out[:, P:]
        return source_joint, target_joint
