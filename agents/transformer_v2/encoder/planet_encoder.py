"""Per-planet/comet token encoder for transformer_v2.

Two-branch MLP design (v8 — deeper):

    input(138)
       │
       ├─ scalar block (18)   ─► 3-Linear MLP(18→128→128→128) ─► z_s (128)
       │
       └─ traj flat (120)     ─► 3-Linear MLP(120→128→128→128) ─► z_t (128)
                       concat(z_s, z_t) → (256)
                                │
                                ▼
                       fusion trunk 3-Linear MLP(256→128→128→128)
                                │
                                ▼
                          LayerNorm(128)
                                │
                                ▼
                          token (128)

Each branch/trunk is a 3-Linear MLP (``Linear → GELU → Linear → GELU →
Linear``) — 2 hidden layers per block. Encoder path is 2 branch hidden
+ 2 fusion hidden = 4 hidden layers total. The branches process
independently before the trunk mixes them, allocating capacity to
scalar geometry and trajectory anchors separately before fusion.

The featurization (raw-field extraction, comet path lookup, orbit math)
lives in ``../featurizer/planet_featurizer.py``. This module owns only
the trainable projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..featurizer.planet_featurizer import (
    N_FUTURE_ANCHORS,
    PLANET_RAW_DIM,
    SCALAR_DIM,
    TRAJ_CHANNELS,
)


def _mlp(
    in_dim: int,
    out_dim: int,
    *,
    n_hidden: int = 1,
    hidden: int | None = None,
) -> nn.Sequential:
    """N-hidden-layer MLP: ``[Linear → GELU] × n_hidden → Linear``.

    ``n_hidden=1`` reproduces the old ``_two_layer_mlp`` (Linear → GELU →
    Linear); ``n_hidden=2`` adds one more Linear+GELU before the output.
    """
    h = hidden if hidden is not None else out_dim
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(prev, h))
        layers.append(nn.GELU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PlanetEncoder(nn.Module):
    """Two-branch MLP encoder for per-planet / per-comet tokens.

    Input:  ``features`` (B, E, PLANET_RAW_DIM=138), optional ``mask`` (B, E).
    Output: ``tokens``   (B, E, d_model=128).

    Slice layout (matches ``planet_featurizer.to_vector``):
      * dims ``[0, SCALAR_DIM=18)`` → scalar branch
      * dims ``[SCALAR_DIM, PLANET_RAW_DIM)`` (= 120 dims, flat 30×4) → trajectory branch

    Architecture (v8 — 2 hidden layers per block):
      * Scalar branch:     ``Linear(18→128) → GELU → Linear(128→128) → GELU → Linear(128→128)``
      * Trajectory branch: ``Linear(120→128) → GELU → Linear(128→128) → GELU → Linear(128→128)``
      * Fusion trunk:      ``Linear(256→128) → GELU → Linear(128→128) → GELU → Linear(128→128)``
      * Final ``LayerNorm(128)``.

    Padding rows are projected the same as real ones, then zeroed out via
    the mask (caller still passes the mask to the downstream transformer's
    ``src_key_padding_mask``).
    """

    def __init__(
        self,
        d_model: int = 128,
        branch_dim: int = 128,
        branch_hidden_layers: int = 2,
        trunk_hidden_layers: int = 2,
        scalar_dim: int = SCALAR_DIM,
        n_anchors: int = N_FUTURE_ANCHORS,
        traj_channels: int = TRAJ_CHANNELS,
        layer_norm: bool = True,
        use_traj_branch: bool = True,
        scalar_only_hidden_layers: int = 2,
    ):
        super().__init__()
        expected = scalar_dim + n_anchors * traj_channels
        if expected != PLANET_RAW_DIM:
            raise ValueError(
                f"scalar_dim({scalar_dim}) + n_anchors({n_anchors})*traj_channels"
                f"({traj_channels}) = {expected} != PLANET_RAW_DIM={PLANET_RAW_DIM}"
            )
        self.scalar_dim = scalar_dim
        self.traj_dim = n_anchors * traj_channels
        self.use_traj_branch = use_traj_branch
        if use_traj_branch:
            self.scalar = _mlp(self.scalar_dim, branch_dim, n_hidden=branch_hidden_layers)
            self.traj = _mlp(self.traj_dim, branch_dim, n_hidden=branch_hidden_layers)
            self.trunk = _mlp(
                2 * branch_dim, d_model,
                n_hidden=trunk_hidden_layers, hidden=d_model,
            )
        else:
            # Ablation: drop the trajectory anchors entirely; the encoder
            # is one MLP mapping scalar kinematics (x, y, vx, vy, ω,
            # sun-relative geometry) directly to the token. The model has
            # to learn the rotation/extrapolation math from these alone.
            self.scalar = _mlp(
                self.scalar_dim, d_model,
                n_hidden=scalar_only_hidden_layers, hidden=d_model,
            )
            self.traj = None
            self.trunk = None
        self.norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scalar = features[..., : self.scalar_dim]
        z_s = self.scalar(scalar)
        if self.use_traj_branch:
            traj = features[..., self.scalar_dim :]
            z_t = self.traj(traj)
            z = torch.cat([z_s, z_t], dim=-1)
            z = self.trunk(z)
        else:
            z = z_s
        z = self.norm(z)
        if mask is not None:
            z = z * mask.unsqueeze(-1)
        return z
