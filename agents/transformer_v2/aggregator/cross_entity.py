"""Cross-entity self-attention over per-planet entity tokens.

Sits one level above ``encoder.entity_encoder.PlanetEntityEncoder``.
Each entity token already carries its own inbound-fleet picture; this
layer lets every token see every *other* token, contextualizing each
planet by the global game state (frontiers, neighbors, sector
balance).

The implementation is deliberately a thin wrapper around
``nn.TransformerEncoder``:

  * Pre-LN encoder layers (more stable than post-LN for shallow
    stacks; we run 2 layers without warmup — v2 dropped from 3→2).
  * A learned ``[CLS]`` token prepended at sequence position 0,
    never masked, that the encoder gradually fills with a global
    snapshot summary. Snapshot-level heads (winner classification,
    expert-acted, …) read it out from index 0.
  * Dynamic ``P`` (planets coming and going as comets spawn/die)
    handled natively via ``src_key_padding_mask`` — the upstream
    ``EntitySnapshotDataset`` already pads to ``max_planets``, so
    ``mask = ~entity_mask`` flows straight through.

See ``README.md`` for the design rationale and the label menu the
pretrain pipeline will train this layer against.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossEntityAttention(nn.Module):
    """Multi-layer self-attention over ``(B, T, P, d_model)`` entity tokens.

    Accepts either ``(B, T, P, d)`` (multi-step) or ``(B, P, d)`` (single
    step) inputs — single-step is treated as ``T=1``. Multi-step
    flattens the time axis into the sequence dim so attention can
    relate any (step, planet) pair to any other.

    Inputs:
      entity_tokens  (B, T, P, d) or (B, P, d)  per-planet entity tokens
      entity_mask    (B, T, P)    or (B, P)     True = real entity

    Returns:
      contextual_tokens  (B, T, P, d) or (B, P, d) — same rank as input,
                                                    post-attention values.
      global_token       (B, d)                   — CLS read-out.

    Multi-step adds a learned per-step embedding (positional encoding
    along time) so attention can distinguish ``e_{t-1}^j`` from
    ``e_t^j``. Step ``i`` (with ``i = 0`` being the oldest history
    frame and ``i = T-1`` the current turn) gets ``step_embed[i]``
    added to every planet token at that step.

    Cold-start handling: when an episode lacks ``T-1`` past turns, the
    caller passes zero tokens with ``entity_mask`` all-False for the
    missing slots; the masked softmax in the encoder ignores them.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.0,
        n_steps: int = 9,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_steps = n_steps
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        # Per-relative-step positional encoding. Step ``i`` of ``n_steps``
        # gets ``step_embed[i]`` added to every planet token at that step.
        # Same role as positional embeddings, but along time rather than
        # position-in-sequence.
        self.step_embed = nn.Parameter(torch.zeros(n_steps, d_model))
        nn.init.trunc_normal_(self.step_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
            norm_first=True,
        )
        # Pre-LN disables PyTorch's nested-tensor fast path; setting this
        # explicitly avoids a noisy warning on every Colab run.
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=n_layers, enable_nested_tensor=False,
            )
        except TypeError:
            # Older PyTorch versions do not expose enable_nested_tensor.
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        entity_tokens: torch.Tensor,        # (B, T, P, d) or (B, P, d)
        entity_mask: torch.Tensor,          # (B, T, P) or (B, P) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize to (B, T, P, d) / (B, T, P).
        if entity_tokens.dim() == 3:
            entity_tokens = entity_tokens.unsqueeze(1)
            entity_mask = entity_mask.unsqueeze(1)
            squeeze_t = True
        else:
            squeeze_t = False
        B, T, P, d = entity_tokens.shape
        if d != self.d_model:
            raise ValueError(
                f"entity_tokens has d={d} but module was built for "
                f"d_model={self.d_model}"
            )
        if T > self.n_steps:
            raise ValueError(
                f"got T={T} steps but module was built for "
                f"n_steps={self.n_steps}"
            )

        # Step embeddings — pick the last T entries so that index 0 is
        # the OLDEST history frame and index T-1 is the CURRENT turn.
        # This way single-step input always gets the "current" embedding.
        step_emb = self.step_embed[-T:].view(1, T, 1, d)
        seq = entity_tokens + step_emb                                 # (B, T, P, d)

        # Flatten (T, P) into one sequence dim.
        seq = seq.reshape(B, T * P, d)
        mask_flat = entity_mask.reshape(B, T * P)                      # bool

        # Prepend CLS (never masked).
        cls_tok = self.cls.expand(B, 1, d)
        seq = torch.cat([cls_tok, seq], dim=1)                         # (B, 1 + T*P, d)
        cls_unmasked = torch.zeros(
            B, 1, dtype=torch.bool, device=entity_tokens.device,
        )
        key_padding = torch.cat([cls_unmasked, ~mask_flat], dim=1)     # True = MASKED

        out = self.encoder(seq, src_key_padding_mask=key_padding)      # (B, 1+T*P, d)

        global_token = out[:, 0]                                        # (B, d)
        contextual = out[:, 1:].reshape(B, T, P, d)                     # (B, T, P, d)
        if squeeze_t:
            contextual = contextual.squeeze(1)
        return contextual, global_token
