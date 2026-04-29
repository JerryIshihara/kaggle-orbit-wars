"""Per-planet/comet token encoder for transformer_v1.

Two-branch architecture: a scalar branch over the static/kinematic
features, and a trajectory branch over the future-anchor sequence.
Outputs are summed and ``LayerNorm``'d to produce the entity token.

Why two branches: the raw vector is heterogeneous. ~18 dims describe
"what is this entity" (type, owner, ships, current pos/vel, sun
geometry); the remaining ~50 dims are a long sequence of future
``(dx, dy)`` anchors with strong temporal structure. A single
``Linear`` over the concatenated input has no notion of time and
forces the encoder to spend capacity preserving anchor info that a
1D conv would extract for free. Splitting:

  * lets the scalar branch specialize on semantics
  * gives the trajectory branch translation-equivariance over time
  * gives a per-anchor validity bit (when added) a natural channel

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
)


# ---------- Scalar branch ----------
class PlanetScalarEncoder(nn.Module):
    """Two-layer MLP over the static-feature block.

    Input:  ``x`` (..., scalar_dim).
    Output: (..., d_model).

    The first ``Linear`` widens to ``d_model``, ``GELU``, then a second
    ``Linear`` keeps the dim. No final activation/normalization here —
    the parent encoder sums with the trajectory branch and then applies
    ``LayerNorm``.
    """

    def __init__(self, scalar_dim: int = SCALAR_DIM, d_model: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(scalar_dim, d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------- Trajectory branch ----------
class PlanetTrajectoryEncoder(nn.Module):
    """1D-conv encoder over the ``(n_anchors, 2)`` future-trajectory tensor.

    Treats the trajectory as a 2-channel ``(dx, dy)`` sequence and runs
    two ``Conv1d`` layers (kernel 3, padding 1) followed by adaptive
    average pooling over time and a final ``Linear`` projection to
    ``d_model``.

    Why ``Conv1d``:
      * adjacent anchors are highly correlated; a conv exploits that
        smoothness for free
      * translation-equivariant over time, so a comet that's "about to
        speed up" looks similar regardless of *when* in the horizon it
        starts speeding up
      * tiny — ~1.5k params at the defaults

    Input:  ``x`` (..., n_anchors, 2).
    Output: (..., d_model).
    """

    def __init__(
        self,
        n_anchors: int = N_FUTURE_ANCHORS,
        d_model: int = 64,
        d_hidden: int = 32,
    ):
        super().__init__()
        self.n_anchors = n_anchors
        # Conv1d expects (N, C, T) → ``in_channels=2`` for (dx, dy).
        self.conv1 = nn.Conv1d(2, d_hidden // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_hidden // 2, d_hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten leading dims so Conv1d sees (N, 2, n_anchors), then
        # restore the batch shape on output.
        leading = x.shape[:-2]
        # ``contiguous()`` because Conv1d needs the underlying memory
        # layout after the transpose.
        x_flat = (
            x.reshape(-1, self.n_anchors, 2).transpose(1, 2).contiguous()
        )
        z = self.act(self.conv1(x_flat))
        z = self.act(self.conv2(z))
        z = self.pool(z).squeeze(-1)         # (N, d_hidden)
        z = self.proj(z)                      # (N, d_model)
        return z.reshape(*leading, -1)


# ---------- Combined encoder ----------
class PlanetEncoder(nn.Module):
    """Project per-entity raw features to ``d_model`` tokens via two
    branches that handle scalar features and trajectory anchors
    separately.

    Input:  ``features`` (B, E, PLANET_RAW_DIM), ``mask`` (B, E).
    Output: ``tokens``   (B, E, d_model).

    Slice layout (matches ``planet_featurizer.to_vector``):
      * dims ``[0, SCALAR_DIM)`` → scalar branch
      * dims ``[SCALAR_DIM, PLANET_RAW_DIM)`` reshaped to
        ``(n_anchors, 2)`` → trajectory branch

    Branch outputs are summed and pushed through ``LayerNorm``. Padding
    rows are projected the same as real ones, then zeroed out via the
    mask (caller still passes the mask to the downstream transformer).
    """

    def __init__(
        self,
        d_model: int = 64,
        scalar_dim: int = SCALAR_DIM,
        n_anchors: int = N_FUTURE_ANCHORS,
        d_traj_hidden: int = 32,
        layer_norm: bool = True,
    ):
        super().__init__()
        if scalar_dim + 2 * n_anchors != PLANET_RAW_DIM:
            raise ValueError(
                f"scalar_dim={scalar_dim} + 2*n_anchors={n_anchors} "
                f"!= PLANET_RAW_DIM={PLANET_RAW_DIM}"
            )
        self.scalar_dim = scalar_dim
        self.n_anchors = n_anchors
        self.scalar = PlanetScalarEncoder(scalar_dim, d_model)
        self.traj = PlanetTrajectoryEncoder(n_anchors, d_model, d_traj_hidden)
        self.norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        leading = features.shape[:-1]
        scalar = features[..., : self.scalar_dim]
        traj_flat = features[..., self.scalar_dim :]
        traj = traj_flat.reshape(*leading, self.n_anchors, 2)

        z_s = self.scalar(scalar)
        z_t = self.traj(traj)
        z = self.norm(z_s + z_t)
        if mask is not None:
            z = z * mask.unsqueeze(-1)
        return z
