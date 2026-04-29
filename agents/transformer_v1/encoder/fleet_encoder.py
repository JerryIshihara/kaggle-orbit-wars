"""Per-fleet token encoder for transformer_v1 (v2-path supplement).

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

    Architecture: ``Linear → GELU → Linear → LayerNorm``. The hidden
    nonlinearity is needed so the projection can learn nonlinear
    functions of position (e.g., distance-to-sun bucket = quantile of
    ``hypot(x-50, y-50)``); a pure ``Linear`` cannot represent
    ``hypot``. The op stays element-wise per fleet, so the encoder is
    still a drop-in for a downstream deep-set / transformer aggregator.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_hidden: int | None = None,
        layer_norm: bool = True,
    ):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.fc1 = nn.Linear(FLEET_RAW_DIM, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.fc1(features)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        if mask is not None:
            # Zero out padded rows so downstream ops that ignore the mask
            # (e.g., naive mean-pooling) don't see garbage.
            x = x * mask.unsqueeze(-1)
        return x
