"""L2-only pretrain tasks for the dual-rate L2 (stage: perception only).

All heads sit ON TOP OF the fused 512→256 outputs — the exact tensors the
later parts will consume — at three granularities, each split into a
SHORT-term (t+5) and LONG-term (t+10) family:

  PLANET aux  (fused ``ctx_now`` (B,P,256)):
      owner_t+{5,10}            5-class CE     (valid_t+K ∧ planet mask)
      log_ships_t+{5,10}        Huber          (same mask)
      ships_arriving_within_{5,10} per player slot, Huber (planet mask)
      earliest_arrival_owner    5-class CE, 4 = none (planet mask)

  PLAYER aux  (fused ``player_state`` (B,4,256), learner-relative slots):
      inbound_{5,10}    Σ_planets arrivals[:, slot]   (label-space aggregate)
      owned_frac_t+{5,10}  mean_planets 1[owner_t+K == slot]
      ships_t+{5,10}     Σ_planets log_ships_t+K · 1[owner_t+K == slot]

  GLOBAL aux  (fused ``glob`` (B,256)):
      churn_{5,10}      mean_planets 1[owner_now != owner_t+K]  (volatility)
      board_ships_t+{5,10}  mean_planets log_ships_t+K

Player/global targets are DERIVED on the fly from the per-planet labels the
pair cache already stores (no dataset rebuild). Aggregates are computed in
label space (the stored log-norm values) — consistent, learnable targets
without unit assumptions. Every label here is observation-derived
perception (what WILL the board look like), no expert-action supervision —
the action/value superstructure trains in the later joint stage.

This complements ``short_horizon.py`` (pre-fusion SHORT-branch tap, used by
the joint stage to bypass the zero-init gate). Here the gate is irrelevant:
these are the ONLY losses, the fusion halves receive gradient immediately,
and the intended warm start (the stopped v3dual run) already has the gate
open.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer_v2.pretrain.entity_encoder import (
    ENTITY_N_OWNER_CLASSES,
    _PLANET_OWNER_START_IDX,
)
from .short_horizon import N_PLAYER_SLOTS

N_OWNER = ENTITY_N_OWNER_CLASSES

L2_AUX_LABEL_KEYS: tuple[str, ...] = (
    "owner_t_plus_5", "owner_t_plus_10",
    "log_ships_t_plus_5", "log_ships_t_plus_10",
    "valid_t_plus_5", "valid_t_plus_10",
    "ships_arriving_within_5", "ships_arriving_within_10",
    "earliest_arrival_owner_slot",
)


def _mlp(d_in: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_out))


class DualL2AuxHeads(nn.Module):
    """Planet / player / global forecast heads over the fused L2 outputs."""

    def __init__(self, d_model: int):
        super().__init__()
        # planet: shared trunk, per-horizon task linears
        self.planet_trunk = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.p_owner5 = nn.Linear(d_model, N_OWNER)
        self.p_owner10 = nn.Linear(d_model, N_OWNER)
        self.p_ships5 = nn.Linear(d_model, 1)
        self.p_ships10 = nn.Linear(d_model, 1)
        self.p_arr5 = nn.Linear(d_model, N_PLAYER_SLOTS)
        self.p_arr10 = nn.Linear(d_model, N_PLAYER_SLOTS)
        self.p_earliest = nn.Linear(d_model, N_OWNER)
        # player: 6 scalars per slot (inbound/owned-frac/ships x 2 horizons)
        self.player_head = _mlp(d_model, 6)
        # global: 4 scalars (churn5, churn10, board_ships5, board_ships10)
        self.glob_head = _mlp(d_model, 4)

    def forward(
        self,
        ctx_now: torch.Tensor,        # (B, P, d) fused
        player_state: torch.Tensor,   # (B, 4, d) fused
        glob: torch.Tensor,           # (B, d)    fused
    ) -> dict[str, torch.Tensor]:
        h = self.planet_trunk(ctx_now)
        return {
            "owner5": self.p_owner5(h), "owner10": self.p_owner10(h),
            "ships5": self.p_ships5(h).squeeze(-1),
            "ships10": self.p_ships10(h).squeeze(-1),
            "arr5": self.p_arr5(h), "arr10": self.p_arr10(h),
            "earliest": self.p_earliest(h),
            "player": self.player_head(player_state),   # (B, 4, 6)
            "glob": self.glob_head(glob),                # (B, 4)
        }


def _derive_targets(batch: dict, planet_mask_now: torch.Tensor) -> dict:
    """Player/global label-space aggregates from the per-planet labels."""
    dev = planet_mask_now.device
    m = planet_mask_now.float()                                   # (B, P)
    denom = m.sum(1).clamp(min=1.0)                               # (B,)
    pf = batch["planet_features"]
    pf_now = pf[:, -1] if pf.dim() == 4 else pf
    owner_now = pf_now[..., _PLANET_OWNER_START_IDX:
                       _PLANET_OWNER_START_IDX + N_OWNER].argmax(-1)  # (B, P)
    out = {}
    for hz in ("5", "10"):
        ok = (batch[f"valid_t_plus_{hz}"].to(dev) > 0.5).float() * m  # (B, P)
        ok_denom = ok.sum(1).clamp(min=1.0)
        owner_k = batch[f"owner_t_plus_{hz}"].to(dev)             # (B, P)
        ships_k = batch[f"log_ships_t_plus_{hz}"].to(dev)         # (B, P)
        arr_k = batch[f"ships_arriving_within_{hz}"].to(dev)      # (B, P, 4)
        slots = torch.arange(N_PLAYER_SLOTS, device=dev).view(1, 1, -1)
        owned = (owner_k.unsqueeze(-1) == slots).float() * ok.unsqueeze(-1)
        out[f"pl_inbound{hz}"] = (arr_k * m.unsqueeze(-1)).sum(1)         # (B,4)
        out[f"pl_owned{hz}"] = owned.sum(1) / ok_denom.unsqueeze(-1)      # (B,4)
        out[f"pl_ships{hz}"] = (ships_k.unsqueeze(-1) * owned).sum(1)     # (B,4)
        out[f"g_churn{hz}"] = ((owner_now != owner_k).float() * ok).sum(1) / ok_denom
        out[f"g_ships{hz}"] = (ships_k * ok).sum(1) / ok_denom
        out[f"mask{hz}"] = ok > 0.5
    return out


def dual_l2_aux_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    planet_mask_now: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    dev = planet_mask_now.device
    tg = _derive_targets(batch, planet_mask_now)
    terms: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}
    m_now = planet_mask_now

    for hz in ("5", "10"):
        mk = tg[f"mask{hz}"]                                       # (B, P)
        if mk.any():
            owner_lbl = batch[f"owner_t_plus_{hz}"].to(dev)
            lo = preds[f"owner{hz}"][mk]
            terms[f"p/owner{hz}"] = F.cross_entropy(lo, owner_lbl[mk])
            metrics[f"p/owner{hz}_acc"] = (
                (lo.argmax(-1) == owner_lbl[mk]).float().mean().item())
            terms[f"p/ships{hz}"] = F.huber_loss(
                preds[f"ships{hz}"][mk],
                batch[f"log_ships_t_plus_{hz}"].to(dev)[mk], delta=0.25)
        if m_now.any():
            terms[f"p/arr{hz}"] = F.huber_loss(
                preds[f"arr{hz}"][m_now],
                batch[f"ships_arriving_within_{hz}"].to(dev)[m_now], delta=0.25)
        i = 0 if hz == "5" else 1
        terms[f"pl/inbound{hz}"] = F.huber_loss(
            preds["player"][..., 0 + i], tg[f"pl_inbound{hz}"], delta=0.25)
        terms[f"pl/owned{hz}"] = F.huber_loss(
            preds["player"][..., 2 + i], tg[f"pl_owned{hz}"], delta=0.25)
        terms[f"pl/ships{hz}"] = F.huber_loss(
            preds["player"][..., 4 + i], tg[f"pl_ships{hz}"], delta=0.25)
        terms[f"g/churn{hz}"] = F.huber_loss(
            preds["glob"][..., 0 + i], tg[f"g_churn{hz}"], delta=0.25)
        terms[f"g/ships{hz}"] = F.huber_loss(
            preds["glob"][..., 2 + i], tg[f"g_ships{hz}"], delta=0.25)

    if m_now.any():
        ea_lbl = batch["earliest_arrival_owner_slot"].to(dev)
        le = preds["earliest"][m_now]
        terms["p/earliest"] = F.cross_entropy(le, ea_lbl[m_now])
        metrics["p/earliest_acc"] = (
            (le.argmax(-1) == ea_lbl[m_now]).float().mean().item())

    total = torch.stack(list(terms.values())).mean()
    metrics.update({k: v.item() for k, v in terms.items()})
    return total, metrics
