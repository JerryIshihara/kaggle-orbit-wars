"""Per-fleet token encoder for transformer_v2 (v2-path supplement).

The v1 agent folds fleet info into per-planet aggregates (see
``../DESIGN.md``). This file implements the v2-path "promote fleets back
to tokens" alternative described there: each in-flight fleet becomes its
own token after a single ``Linear`` projection to ``d_model``.

The featurization (raw-field extraction, target resolution, tracking) is
in ``../featurizer/fleet_featurizer.py``. This module owns only the
trainable projection so the nn.Module stays pure and device-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..featurizer import FLEET_RAW_DIM


class FleetEncoder(nn.Module):
    """Project per-fleet raw features to ``d_model`` tokens.

    Input:  ``features`` (B, F, FLEET_RAW_DIM), ``mask`` (B, F).
    Output: ``tokens``   (B, F, d_model). Padding rows are projected too
            (cheaper than a gather), but the caller should pass the mask
            into the transformer's ``src_key_padding_mask``.

    Architecture: ``num_layers`` Linears with GELU between, then
    LayerNorm. ``num_layers=2`` is the historical default
    (``Linear → GELU → Linear → LayerNorm``). ``num_layers=3+`` adds
    deeper internal nonlinearity so the encoder can represent more
    structure linearly in its output — useful when downstream
    pretraining uses linear-probe-style heads (1-layer decoders) to
    force the encoder to carry information directly.

    The middle hidden width is ``d_hidden`` (defaults to ``d_model``).
    The op stays element-wise per fleet, so the encoder is still a
    drop-in for a downstream deep-set / transformer aggregator.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_hidden: int | None = None,
        layer_norm: bool = True,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1 (got {num_layers})")
        d_hidden = d_hidden or d_model
        layers: list[nn.Module] = []
        in_dim = FLEET_RAW_DIM
        for k in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, d_hidden))
            layers.append(nn.GELU())
            in_dim = d_hidden
        layers.append(nn.Linear(in_dim, d_model))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.mlp(features)
        x = self.norm(x)
        if mask is not None:
            # Zero out padded rows so downstream ops that ignore the mask
            # (e.g., naive mean-pooling) don't see garbage.
            x = x * mask.unsqueeze(-1)
        return x
