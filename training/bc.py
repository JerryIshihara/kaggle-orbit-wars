"""Behavior cloning trainer — warm-start the CNN on a teacher policy."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from agents.agent_cnn_v1 import CNNv1

from .common import fresh_model, load_model, save_model


def _planet_features(model: CNNv1, channels: torch.Tensor, scalars: torch.Tensor, coords: torch.Tensor):
    """Mirror of CNNv1.act but batched. coords is (B, M, 2)."""
    feat_map = model.conv(channels)
    scalar_feat = model.scalar_mlp(scalars)
    B, M, _ = coords.shape
    grid = coords.clone()
    grid[..., 0] = (grid[..., 0] / 100.0) * 2 - 1
    grid[..., 1] = (grid[..., 1] / 100.0) * 2 - 1
    grid = grid.view(B, M, 1, 2)
    sampled = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=False)  # (B, D, M, 1)
    per_planet = sampled.squeeze(-1).permute(0, 2, 1)  # (B, M, D)
    sfeat = scalar_feat.unsqueeze(1).expand(-1, M, -1)  # (B, M, 32)
    x = model.head_trunk(torch.cat([per_planet, sfeat], dim=-1))  # (B, M, 96)
    launch = model.launch_head(x).squeeze(-1)  # (B, M)
    target = model.target_head(x)  # (B, M, GRID*GRID)
    ship = model.ship_head(x).squeeze(-1)  # (B, M)
    return launch, target, ship


def train_bc(
    dataset: dict[str, torch.Tensor],
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    launch_weight: float = 1.0,
    target_weight: float = 1.0,
    frac_weight: float = 0.5,
    resume_from: Path | str | None = None,
    save_to: Path | str | None = None,
    verbose: bool = True,
) -> CNNv1:
    model = load_model(resume_from) if resume_from else fresh_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        dataset["channels"],
        dataset["scalars"],
        dataset["coords"],
        dataset["launched"],
        dataset["targets"],
        dataset["fracs"],
        dataset["mask"],
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        t0 = time.time()
        sums = {"launch": 0.0, "target": 0.0, "frac": 0.0, "total": 0.0, "n": 0}
        for batch in loader:
            channels, scalars, coords, launched, targets, fracs, mask = batch
            pred_launch, pred_target, pred_frac = _planet_features(model, channels, scalars, coords)

            # Mask to live planets
            m = mask.bool()
            if not m.any():
                continue

            launch_loss = F.binary_cross_entropy_with_logits(
                pred_launch[m], launched[m], reduction="mean"
            )

            lm = m & launched.bool()
            if lm.any():
                target_loss = F.cross_entropy(pred_target[lm], targets[lm], reduction="mean")
                frac_loss = F.mse_loss(torch.sigmoid(pred_frac[lm]), fracs[lm], reduction="mean")
            else:
                target_loss = torch.zeros((), device=pred_target.device)
                frac_loss = torch.zeros((), device=pred_frac.device)

            loss = (
                launch_weight * launch_loss
                + target_weight * target_loss
                + frac_weight * frac_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            sums["launch"] += launch_loss.item()
            sums["target"] += target_loss.item()
            sums["frac"] += frac_loss.item()
            sums["total"] += loss.item()
            sums["n"] += 1

        if verbose and sums["n"]:
            n = sums["n"]
            print(
                f"epoch {epoch + 1}/{epochs}  "
                f"launch={sums['launch']/n:.4f}  "
                f"target={sums['target']/n:.4f}  "
                f"frac={sums['frac']/n:.4f}  "
                f"total={sums['total']/n:.4f}  "
                f"({time.time() - t0:.1f}s)",
                file=sys.stderr,
            )

    model.eval()
    if save_to is not None or save_to is None:
        path = save_model(model, save_to)
        if verbose:
            print(f"saved weights: {path}", file=sys.stderr)
    return model
