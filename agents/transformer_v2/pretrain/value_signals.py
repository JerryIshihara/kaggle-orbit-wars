"""P1 shaping signals — all RELATIVE to the strongest enemy, in [0,1].

Each signal is ``own / (own + reference_enemy)`` (0.5 = parity with the
strongest opponent, >0.5 ahead, <0.5 behind, 0.5 when both sides are 0). This
uniform "advantage share vs max-other" form is what the value heads predict the
future level of, and what the PBRS potential Φ is built from.

  s1 ship_adv         = own_ships  / (own_ships  + max_other_ships)
  s2 production_adv   = own_prod   / (own_prod   + max_other_prod)
  s3 planet_adv       = own_planets/ (own_planets+ max_other_planets)
  s4 safety           = own_ships  / (own_ships  + incoming_enemy_ships)   # threat-relative
  s5 fleet_speed_adv  = own_speed  / (own_speed  + max_other_speed)

Returns (B, O, 5) per owner/player slot. Cycle-safe: imports only
consolidator_heads (current-state quantities) + featurizer constants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .consolidator_heads import (
    _FLEET_OWNER_SLICE,
    _FLEET_SHIPS_LOG_IDX,
    _N_PLAYER_SLOTS,
    _PLANET_OWNER_SLICE,
    compute_current_state_labels,
)
from ..featurizer.fleet_featurizer import (
    ETA_NORM,
    MAX_SPEED,
    SHIPS_LOG_MAX,
    SPEED_LOG_DENOM,
)

P1_SIGNAL_NAMES: tuple[str, ...] = (
    "ship_adv", "production_adv", "planet_adv", "safety", "fleet_speed_adv",
)
N_P1_SIGNALS: int = len(P1_SIGNAL_NAMES)

# Forward horizons the value heads predict the signal LEVEL at (fits fleet
# flight dynamics: median 5, p90 18, p99 45 — covers in-flight → fully-resolved).
P1_FWD_HORIZONS: tuple[int, ...] = (5, 10, 15, 20, 50)
# Backward horizons for the anti-shortcut aux heads (within the T=10 history).
P1_BACK_HORIZONS: tuple[int, ...] = (5, 20, 45)


def _cur4(x: torch.Tensor) -> torch.Tensor:
    """Current frame of a FEATURE tensor: (B,T,P,D)->(B,P,D); (B,P,D) passthrough."""
    return x[:, -1] if x.dim() == 4 else x


def _cur3(x: torch.Tensor) -> torch.Tensor:
    """Current frame of a MASK/ROUTING tensor: (B,T,F)->(B,F); (B,F) passthrough."""
    return x[:, -1] if x.dim() == 3 else x


def _max_other(x: torch.Tensor) -> torch.Tensor:
    """(B,S) -> (B,S): max over the OTHER slots (exclude self), floored at 0."""
    B, S = x.shape
    eye = torch.eye(S, dtype=torch.bool, device=x.device).view(1, S, S)
    other = x.unsqueeze(1).expand(B, S, S).masked_fill(eye, float("-inf"))
    return other.max(dim=-1).values.clamp_min(0.0)


def _rel(own: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """own / (own + ref) ∈ [0,1]; 0.5 when both are 0 (parity / no info)."""
    denom = own + ref
    return torch.where(denom > 0, own / denom.clamp_min(1e-6), torch.full_like(own, 0.5))


def _fleet_speed_from_ships(ships: torch.Tensor) -> torch.Tensor:
    """Mirror fleet_featurizer fleet-speed: bigger fleets travel faster."""
    s = torch.log(ships.clamp(min=1.0)) / SPEED_LOG_DENOM
    sp = (1.0 + (MAX_SPEED - 1.0) * s.clamp(min=0.0) ** 1.5).clamp(max=MAX_SPEED)
    return torch.where(ships <= 1.0, torch.ones_like(sp), sp)


def compute_p1_signals(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-player P1 shaping signals from a (current-frame) batch -> (B, O, 5)."""
    pf = batch["planet_features"]
    pm = batch["planet_mask"]
    ff = batch["fleet_features"]
    fm = batch["fleet_mask"]

    lab = compute_current_state_labels(pf, pm, ff, fm)             # per-player (B,O)
    total_ships = torch.expm1(lab.total_ships_log * SHIPS_LOG_MAX).clamp_min(0.0)
    prod = lab.production_sum.clamp_min(0.0)
    planets = lab.planet_count.clamp_min(0.0)

    # --- per-player average fleet speed (own) ---
    ffn = _cur4(ff)
    fmn = _cur3(fm).to(ffn.dtype)
    fowner = ffn[..., _FLEET_OWNER_SLICE] * fmn.unsqueeze(-1)       # (B,F,4)
    fships = torch.expm1(ffn[..., _FLEET_SHIPS_LOG_IDX] * SHIPS_LOG_MAX).clamp_min(0.0)
    fspeed = _fleet_speed_from_ships(fships)                       # (B,F)
    spd_sum = torch.einsum("bf,bfp->bp", fspeed, fowner)           # (B,4)
    n_fleets = fowner.sum(dim=1)                                   # (B,4)
    avg_speed = torch.where(n_fleets > 0, spd_sum / n_fleets.clamp_min(1.0),
                            torch.zeros_like(spd_sum))

    # --- incoming threat per defender p (enemy ships en route to p's planets) ---
    powner = _cur4(pf)[..., _PLANET_OWNER_SLICE][..., :_N_PLAYER_SLOTS]   # (B,P,4)
    fo = _cur3(batch["fleet_owner_slot"]).long().clamp(0, _N_PLAYER_SLOTS - 1)
    ft = _cur3(batch["fleet_target_idx"]).long()
    fe = _cur3(batch["fleet_eta_norm"])
    fsl = _cur3(batch["fleet_ships_log"])
    fmsk = _cur3(fm).bool()
    tgt_owner = torch.gather(                                       # (B,F,4) target planet owner
        powner, 1, ft.clamp(min=0).unsqueeze(-1).expand(-1, -1, _N_PLAYER_SLOTS),
    )
    valid = ((ft >= 0) & fmsk).unsqueeze(-1).to(tgt_owner.dtype)
    ships2 = torch.expm1(fsl * SHIPS_LOG_MAX).clamp_min(0.0)
    decay = 1.0 / (1.0 + fe * ETA_NORM)                            # nearer threat weighs more
    fo_oh = F.one_hot(fo, _N_PLAYER_SLOTS).to(tgt_owner.dtype)      # (B,F,4) owner one-hot
    w = (ships2 * decay).unsqueeze(-1) * valid                     # (B,F,1)
    # defender p threatened by fleets targeting p AND not owned by p
    incoming = (tgt_owner * (1.0 - fo_oh) * w).sum(dim=1)          # (B,4)

    s1 = _rel(total_ships, _max_other(total_ships))
    s2 = _rel(prod, _max_other(prod))
    s3 = _rel(planets, _max_other(planets))
    s4 = _rel(total_ships, incoming)
    s5 = _rel(avg_speed, _max_other(avg_speed))
    return torch.stack([s1, s2, s3, s4, s5], dim=-1)               # (B, O, 5)
