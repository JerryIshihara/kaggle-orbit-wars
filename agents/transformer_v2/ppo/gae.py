"""Generalized Advantage Estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Episode:
    """One learner episode laid out as parallel 1-D tensors.

    All tensors are length ``T`` (number of learner steps in the episode).
    ``values`` and ``rewards`` are aligned step-for-step. ``done[t]`` is
    1.0 when step ``t`` is the terminal learner step of the episode.
    """

    values: torch.Tensor      # (T,) float
    rewards: torch.Tensor     # (T,) float
    dones: torch.Tensor       # (T,) float in {0.0, 1.0}
    # populated by compute_advantages:
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None


def compute_advantages(
    episodes: list[Episode],
    *,
    gamma: float = 0.995,
    lam: float = 0.95,
    normalize: bool = True,
) -> torch.Tensor:
    """Fill ``episode.advantages`` and ``episode.returns`` in place; return
    the concatenated normalized advantage tensor for convenience.

    GAE within each episode; bootstrap value at non-terminal truncation is
    zero (Phase 0 rollouts collect complete episodes — see protocol "Rollout
    shard format").
    """
    if not episodes:
        return torch.empty(0)
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma out of (0, 1]: {gamma}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam out of [0, 1]: {lam}")

    all_adv: list[torch.Tensor] = []
    for ep in episodes:
        T = ep.values.shape[0]
        if T == 0:
            ep.advantages = torch.empty(0)
            ep.returns = torch.empty(0)
            continue
        adv = torch.zeros(T, dtype=torch.float32, device=ep.values.device)
        gae = 0.0
        for t in reversed(range(T)):
            done_t = float(ep.dones[t].item())
            v_t = float(ep.values[t].item())
            r_t = float(ep.rewards[t].item())
            if t + 1 < T:
                v_next = float(ep.values[t + 1].item())
            else:
                v_next = 0.0  # bootstrap = 0 at episode end
            delta = r_t + gamma * v_next * (1.0 - done_t) - v_t
            gae = delta + gamma * lam * (1.0 - done_t) * gae
            adv[t] = gae
        ep.advantages = adv
        ep.returns = adv + ep.values
        all_adv.append(adv)

    flat = torch.cat(all_adv)
    if normalize and flat.numel() > 1:
        mean = flat.mean()
        std = flat.std(unbiased=False).clamp_min(1e-8)
        for ep in episodes:
            if ep.advantages is not None and ep.advantages.numel() > 0:
                ep.advantages = (ep.advantages - mean) / std
        flat = (flat - mean) / std
    return flat
