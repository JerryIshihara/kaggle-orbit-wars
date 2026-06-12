"""Short-horizon auxiliary pretrain tasks for the dual-rate L2's SHORT branch.

The zero-init fusion gate means the SHORT branch receives no gradient
from the main action/value losses until the fusion weights move off
zero — correct for warm-start safety, but slow to bootstrap. These
heads give the branch its own supervision from step 0, reading the
branch's PRE-FUSION current-step tokens directly (gate bypassed), on
labels the pair cache already stores per snapshot (the original
entity-pretrain label set, t+5 horizon family — matched to the short
window's 20-turn @ stride-2 view):

  * ``owner_t_plus_5``            5-class CE   (who owns this planet in 5 turns)
  * ``log_ships_t_plus_5``        Huber        (garrison level in 5 turns, log-norm)
  * ``ships_arriving_within_5``   Huber        (per-player inbound mass, log-norm)
  * ``earliest_arrival_owner_slot`` 5-class CE (who strikes first; 4 = nobody)

The first two are masked by ``valid_t_plus_5`` (episode must still be
running) AND the current planet mask; the last two only need the planet
mask. All heads share one light GELU trunk off the short branch tokens;
internal task weights are fixed equal — scale the whole block with
``--short-aux-weight``.

Forecasting t+5 ownership/garrisons under the 1.30.x swept-collision
physics requires integrating production + in-flight fleets + collisions
over the next few turns — exactly the dynamics the short branch exists
to perceive.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer_v2.pretrain.entity_encoder import ENTITY_N_OWNER_CLASSES

#: player slots in the per-player arrival table (learner-relative slots).
N_PLAYER_SLOTS = 4
#: owner classes (4 player slots + neutral); also the earliest-arrival
#: class count (slots 0-3 + 4 = "no inbound fleet").
N_OWNER = ENTITY_N_OWNER_CLASSES

SHORT_AUX_LABEL_KEYS: tuple[str, ...] = (
    "owner_t_plus_5",
    "log_ships_t_plus_5",
    "valid_t_plus_5",
    "ships_arriving_within_5",
    "earliest_arrival_owner_slot",
)


class ShortHorizonHeads(nn.Module):
    """Per-planet t+5 forecast heads over the SHORT branch's ctx tokens."""

    def __init__(self, d_model: int, *, dropout: float = 0.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.owner_t5 = nn.Linear(d_model, N_OWNER)
        self.ships_t5 = nn.Linear(d_model, 1)
        self.arrivals5 = nn.Linear(d_model, N_PLAYER_SLOTS)
        self.earliest = nn.Linear(d_model, N_OWNER)

    def forward(self, ctx_short_now: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(ctx_short_now)                      # (B, P, d)
        return {
            "owner_t5": self.owner_t5(h),                  # (B, P, N_OWNER)
            "ships_t5": self.ships_t5(h).squeeze(-1),      # (B, P)
            "arrivals5": self.arrivals5(h),                # (B, P, 4)
            "earliest": self.earliest(h),                  # (B, P, N_OWNER)
        }


def short_horizon_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    planet_mask_now: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked mean of the four task losses + accuracy metrics."""
    device = planet_mask_now.device
    fut_ok = batch["valid_t_plus_5"].to(device) > 0.5      # (B, P)
    m_fut = planet_mask_now & fut_ok                        # owner/ships mask
    m_now = planet_mask_now                                 # arrivals/earliest

    terms: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}

    if m_fut.any():
        owner_lbl = batch["owner_t_plus_5"].to(device)
        lo = preds["owner_t5"][m_fut]
        terms["owner_t5"] = F.cross_entropy(lo, owner_lbl[m_fut])
        metrics["owner_acc"] = (
            (lo.argmax(-1) == owner_lbl[m_fut]).float().mean().item()
        )
        terms["ships_t5"] = F.huber_loss(
            preds["ships_t5"][m_fut],
            batch["log_ships_t_plus_5"].to(device)[m_fut],
            delta=0.25,
        )
    if m_now.any():
        terms["arrivals5"] = F.huber_loss(
            preds["arrivals5"][m_now],
            batch["ships_arriving_within_5"].to(device)[m_now],
            delta=0.25,
        )
        ea_lbl = batch["earliest_arrival_owner_slot"].to(device)
        le = preds["earliest"][m_now]
        terms["earliest"] = F.cross_entropy(le, ea_lbl[m_now])
        metrics["earliest_acc"] = (
            (le.argmax(-1) == ea_lbl[m_now]).float().mean().item()
        )

    if not terms:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {"empty_batch": 1.0}
    total = torch.stack(list(terms.values())).mean()
    metrics.update({k: v.item() for k, v in terms.items()})
    return total, metrics
