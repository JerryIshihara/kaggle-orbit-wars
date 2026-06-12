"""v4 value tasks: pruned head set + the two ranker-specific additions.

KEPT (the PPO critic + simulate-then-score ranker consumers):
  * win  (tier 1.0, BCE, player_valid-masked) — the critic anchor
  * fwd  (tier 0.5, Huber) — the 5 signals at horizons (5,10,15,20,50);
    the ranker reads the t∈{5,10,20} slice

REMOVED (no loss term; heads stay in the module for ckpt-shape compat,
parameters frozen-dead):
  * back  — anti-shortcut aux, superseded by the temporal-contrast task
  * rank  — cross-player ListMLE; in 2P it IS the win label, in 4P the
    win+fwd combination covers placement
  * survives — alive-at-horizon; nothing downstream consumes it

ADDED:
  * temporal CONTRAST (tier 0.25): sibling snapshots (t, t+Δ) of the same
    episode; BCE on sigmoid(score(t+Δ) − score(t)) vs the realized slot-0
    composite direction. Trains exactly the comparator monotonicity the
    sample-K-argmax ranker exploits. ``score`` = win logit + w·fwd@h10.
  * player INBOUND aux (tier 0.25): per-slot in-flight mass arriving
    within 5/10 ticks, derived ON THE FLY from fleet columns
    (eta_norm / ships_log / owner_slot) — no cache labels needed. Forces
    ``player_state`` to encode fleet timing, the exact channel that
    distinguishes candidate launch sets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer_v2.pretrain.value_signals import (
    N_P1_SIGNALS,
    P1_FWD_HORIZONS,
)

#: slot-0 composite weights over the 5 signals (advantage trio dominant —
#: the W/L-gap evidence) used by the contrast score AND the deploy ranker.
RANKER_SIGNAL_W = (0.30, 0.30, 0.30, 0.05, 0.05)
H10_IDX = P1_FWD_HORIZONS.index(10)
#: eta_norm denominators: fleet_eta_norm is eta / ETA_NORM_MAX (board-scale
#: ticks). Thresholds in normalized units for "arrives within K".
ETA_NORM_MAX = 100.0


class PlayerInboundAux(nn.Module):
    """player_state (B, 4, d) -> per-slot (inbound5, inbound10)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2),
        )

    def forward(self, player_state: torch.Tensor) -> torch.Tensor:
        return self.net(player_state)                    # (B, 4, 2)


def derive_inbound_targets(batch: dict) -> torch.Tensor:
    """(B, 4, 2) label-space inbound mass per slot within 5 / 10 ticks.

    Sum of ``fleet_ships_log`` over in-flight fleets owned by each slot
    with ``eta <= K`` — computed from the CURRENT frame of the stacked
    fleet columns. Label-space aggregate (consistent, unit-free).
    """
    eta = batch["fleet_eta_norm"]
    ships = batch["fleet_ships_log"]
    owner = batch["fleet_owner_slot"]
    mask = batch["fleet_mask"]
    if eta.dim() == 3:                                   # (B, T, F) -> t0
        eta, ships, owner, mask = (x[:, -1] for x in (eta, ships, owner, mask))
    B, F_ = eta.shape
    out = eta.new_zeros(B, 4, 2)
    for ki, k in enumerate((5.0, 10.0)):
        within = mask & (eta * ETA_NORM_MAX <= k)
        for slot in range(4):
            sel = within & (owner == slot)
            out[:, slot, ki] = (ships * sel).sum(-1)
    return out


def composite_score(out: dict) -> torch.Tensor:
    """Slot-0 ranker score: win logit + w·fwd@h=10. (B,)"""
    w = out["fwd"].new_tensor(RANKER_SIGNAL_W)
    return out["win"][:, 0] + (out["fwd"][:, 0, :, H10_IDX] * w).sum(-1)


def compute_value_v4_loss(
    out: dict,
    batch: dict,
    *,
    inbound_head: PlayerInboundAux | None = None,
    out_b: dict | None = None,
    batch_b: dict | None = None,
    w_win: float = 1.0,
    w_fwd: float = 0.5,
    w_contrast: float = 0.25,
    w_inbound: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    dev = out["win"].device
    terms: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}

    pv = batch["player_valid"].to(dev)                              # (B, 4)
    if pv.sum() > 0:
        # win label: final_rank == 0 per slot
        z = (batch["final_rank"].to(dev) == 0).float()
        bce = F.binary_cross_entropy_with_logits(
            out["win"], z, reduction="none")
        terms["win"] = w_win * (bce * pv).sum() / pv.sum()
        acc = (((out["win"] > 0).float() == z) * pv).sum() / pv.sum()
        metrics["win_acc"] = float(acc)

    p1v = batch["p1_valid"].to(dev)                                 # (B, NF)?
    fut = batch["p1_future"].to(dev)                                # (B,4,S,NF)
    if fut.dim() == 4 and p1v.numel():
        # p1_valid: (B, NF) horizon validity broadcast over slots/signals
        v = p1v.view(p1v.shape[0], 1, 1, -1).to(dev)
        hub = F.huber_loss(out["fwd"], fut, delta=0.25, reduction="none")
        terms["fwd"] = w_fwd * (hub * v).sum() / v.expand_as(hub).sum().clamp(min=1)

    if inbound_head is not None:
        tgt = derive_inbound_targets(batch).to(dev)
        pred = inbound_head(out["player_state"])
        terms["inbound"] = w_inbound * F.huber_loss(pred, tgt, delta=0.25)

    if out_b is not None and batch_b is not None:
        s_a = composite_score(out)
        s_b = composite_score(out_b)
        w = fut.new_tensor(RANKER_SIGNAL_W)
        now_a = (batch["p1_now"].to(dev)[:, 0] * w).sum(-1)
        now_b = (batch_b["p1_now"].to(dev)[:, 0] * w).sum(-1)
        label = (now_b > now_a).float()
        terms["contrast"] = w_contrast * F.binary_cross_entropy_with_logits(
            s_b - s_a, label)
        metrics["contrast_acc"] = float(
            (((s_b - s_a) > 0).float() == label).float().mean())

    if not terms:
        zero = out["win"].sum() * 0.0
        return zero, {"empty": 1.0}
    total = sum(terms.values())
    metrics.update({k: float(v.detach()) for k, v in terms.items()})
    return total, metrics
