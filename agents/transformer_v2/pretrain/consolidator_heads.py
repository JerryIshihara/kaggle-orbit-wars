"""Supervised heads attached to the PlayerConsolidator output.

Each head consumes ``player_state (B, 4, d_model)`` from
``EntityPretrainModel.forward_with_context()`` and produces per-player
predictions. Targets are derived from the same snapshot — no cache
rebuild is needed for the v0 (CurrentStateHead-only) bundle.

The slot convention everywhere is learner-relative:
``player_state[:, 0]`` is the learner, ``[:, 1:4]`` are opponents (in
the same relative ordering used by the planet/fleet featurizers).

This module exports v0 only; FutureStateHead, OutcomeHead, MatchupHead
and the optional AffordanceHead are deferred to v1+ (they need either a
cache rebuild for future-snapshot lookups or per-episode outcome
labels). See ``train_consolidator.py`` for the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..featurizer.fleet_featurizer import SHIPS_LOG_MAX


_N_PLAYER_SLOTS: int = 4
_N_RANKS: int = 4

# Continuous channel order — kept as a fixed tuple so the loss code can
# refer to slices by index and so logging stays stable.
_CONTINUOUS_CHANNELS: tuple[str, ...] = (
    "total_ships_log",
    "planet_count",
    "production_sum",
    "fleet_ships_in_air_log",
    "comet_count",
)
_N_CONTINUOUS: int = len(_CONTINUOUS_CHANNELS)


def _build_mlp(
    *,
    in_dim: int,
    hidden: int,
    out_dim: int,
    n_layers: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build an ``n_layers``-Linear MLP with GELU between hidden layers.

    For ``n_layers == 1`` returns a bare ``Linear(in_dim, out_dim)``.
    For ``n_layers >= 2`` returns
    ``Linear → GELU → (Dropout) → Linear ... → Linear(hidden, out_dim)``.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if n_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    for _ in range(n_layers - 2):
        layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class CurrentStateHead(nn.Module):
    """Per-player current-state regressor + classifiers.

    Forward input:
      player_state  (B, 4, d_model)   per-player strategic embedding.

    Forward output (all tensors keyed in the returned dict):
      continuous    (B, 4, 5)         regressed current-state stats.
                                      Channel order in _CONTINUOUS_CHANNELS.
      alive_logits  (B, 4)            BCE logit, P(player is alive).
      rank_logits   (B, 4, 4)         CE logits over ranks 0..3 (0 = top).
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        hidden: int | None = None,
        trunk_n_layers: int = 2,
        head_n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if trunk_n_layers < 1:
            raise ValueError(f"trunk_n_layers must be >= 1, got {trunk_n_layers}")
        if head_n_layers < 2:
            raise ValueError(
                f"head_n_layers must be >= 2 (per design floor); "
                f"got {head_n_layers}"
            )
        h = hidden if hidden is not None else d_model
        # Shared per-player trunk. Applied independently to each of the 4
        # player tokens (linear layers are batched over the slot axis).
        # Each layer: Linear → GELU → Dropout. ``trunk_n_layers`` is the
        # number of Linears.
        trunk: list[nn.Module] = []
        for i in range(trunk_n_layers):
            in_dim = d_model if i == 0 else h
            trunk.extend([
                nn.Linear(in_dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.trunk = nn.Sequential(*trunk)
        # Per-task heads: at least 2-Linear MLPs (Linear → GELU → ... →
        # Linear). ``head_n_layers`` counts the Linears; the user-facing
        # floor is 2 so each head can shape its own task-specific
        # nonlinearity over the shared trunk features.
        self.continuous_head = _build_mlp(
            in_dim=h, hidden=h, out_dim=_N_CONTINUOUS,
            n_layers=head_n_layers, dropout=dropout,
        )
        self.alive_head = _build_mlp(
            in_dim=h, hidden=h, out_dim=1,
            n_layers=head_n_layers, dropout=dropout,
        )
        # Rank: per-player logit over (rank0..rank3). The final rank
        # decoder *could* be cross-player (a single softmax over the 4
        # players assigning ranks), but per-player CE with independent
        # 4-way logits is simpler and lets the head learn rank from each
        # player's view of itself; the cross-player consistency lives in
        # the Consolidator's player_self_attn layer.
        self.rank_head = _build_mlp(
            in_dim=h, hidden=h, out_dim=_N_RANKS,
            n_layers=head_n_layers, dropout=dropout,
        )

    def forward(self, player_state: torch.Tensor) -> dict[str, torch.Tensor]:
        # player_state: (B, 4, d_model)
        if player_state.dim() != 3 or player_state.shape[1] != _N_PLAYER_SLOTS:
            raise ValueError(
                f"player_state expected (B, 4, d_model); got "
                f"{tuple(player_state.shape)}"
            )
        h = self.trunk(player_state)                            # (B, 4, h)
        return {
            "continuous": self.continuous_head(h),               # (B, 4, 5)
            "alive_logits": self.alive_head(h).squeeze(-1),      # (B, 4)
            "rank_logits": self.rank_head(h),                    # (B, 4, 4)
        }


# --------------------------------------------------------------------------- #
# Label computation                                                            #
# --------------------------------------------------------------------------- #

# Planet feature index constants (mirror entity_encoder._PLANET_* but
# kept local so this module imports cleanly without the heavy entity
# encoder file).
_PLANET_IS_COMET_IDX: int = 0
_PLANET_OWNER_SLICE = slice(1, 6)   # 5-dim learner-relative one-hot
_PLANET_SHIPS_LOG_IDX: int = 6
_PLANET_PRODUCTION_IDX: int = 7

# Fleet feature indices (see featurizer/fleet_featurizer.py FLEET_RAW_DIM).
# Owners live at indices 11..14 (4-slot one-hot, no neutral — fleets
# always have an owner).
_FLEET_SHIPS_LOG_IDX: int = 7
_FLEET_OWNER_SLICE = slice(11, 15)  # (B, F, 4)


@dataclass
class CurrentStateLabels:
    """Tensors fed to ``current_state_loss``. All shape (B, 4)."""
    total_ships_log: torch.Tensor       # learner-relative slot order
    planet_count: torch.Tensor
    production_sum: torch.Tensor
    fleet_ships_in_air_log: torch.Tensor
    comet_count: torch.Tensor
    alive: torch.Tensor                 # binary float
    rank: torch.Tensor                  # long, 0..3 (0 = top)


def _current_planet_features(planet_features: torch.Tensor) -> torch.Tensor:
    """If a temporal (B, T, P, d) is passed, take the last step.

    Mirrors the helper of the same name in ``entity_encoder.py`` so
    callers can pass either shape transparently.
    """
    if planet_features.dim() == 4:
        return planet_features[:, -1]
    return planet_features


def _current_fleet_features(fleet_features: torch.Tensor) -> torch.Tensor:
    if fleet_features.dim() == 4:
        return fleet_features[:, -1]
    return fleet_features


def compute_current_state_labels(
    planet_features: torch.Tensor,         # (B, P, PLANET_RAW_DIM) or (B, T, P, ..)
    planet_mask: torch.Tensor,             # (B, P) bool or (B, T, P)
    fleet_features: torch.Tensor,          # (B, F, FLEET_RAW_DIM) or (B, T, F, ..)
    fleet_mask: torch.Tensor,              # (B, F) bool or (B, T, F)
) -> CurrentStateLabels:
    """Derive per-player current-state targets from the current snapshot.

    All outputs are learner-relative — slot 0 = self, 1..3 = opponents in
    the same order the featurizer uses. The neutral planet slot (5th of
    the planet owner one-hot) is **not** a player and is excluded from
    every aggregate here. Padding / dead planets are gated by
    ``planet_mask``; dead fleets by ``fleet_mask``.
    """
    pf = _current_planet_features(planet_features)               # (B, P, d_planet)
    ff = _current_fleet_features(fleet_features)                 # (B, F, d_fleet)
    if planet_mask.dim() == 3:
        pm = planet_mask[:, -1]
    else:
        pm = planet_mask
    if fleet_mask.dim() == 3:
        fm = fleet_mask[:, -1]
    else:
        fm = fleet_mask

    pm = pm.to(pf.dtype)                                          # (B, P)
    fm = fm.to(ff.dtype)                                          # (B, F)

    # ---- per-planet quantities, masked ----
    planet_owner_oh = pf[..., _PLANET_OWNER_SLICE]                # (B, P, 5)
    planet_owner_oh = planet_owner_oh * pm.unsqueeze(-1)
    # Player slots only (drop the neutral channel).
    planet_player_oh = planet_owner_oh[..., :_N_PLAYER_SLOTS]     # (B, P, 4)

    # Ship features in the planet/fleet vectors are
    # ``log1p(ships) / SHIPS_LOG_MAX`` (see featurizer/{planet,fleet}_featurizer).
    # Summing those across rows yields ``sum_p log1p(ships_p)/SHIPS_LOG_MAX``
    # — NOT what we want. Reconstruct raw counts first, sum, then
    # re-normalize once so the label means ``log1p(total_ships_i)/SHIPS_LOG_MAX``.
    planet_ships_raw = torch.expm1(
        pf[..., _PLANET_SHIPS_LOG_IDX] * SHIPS_LOG_MAX,
    ) * pm                                                         # (B, P)
    production = pf[..., _PLANET_PRODUCTION_IDX] * pm
    is_comet = pf[..., _PLANET_IS_COMET_IDX] * pm

    # ---- per-player aggregates over planets ----
    total_ships_raw_planet = torch.einsum(
        "bp,bpk->bk", planet_ships_raw, planet_player_oh,
    )                                                              # (B, 4)
    production_sum = torch.einsum("bp,bpk->bk", production, planet_player_oh)
    comet_count = torch.einsum("bp,bpk->bk", is_comet, planet_player_oh)
    # Planet count = number of player-owned planets (incl. comets).
    planet_count = planet_player_oh.sum(dim=1)                    # (B, 4)
    # Subtract comets if you want "planets only"; for v0 keep both — the
    # comet_count channel separates them, and planet_count remains a
    # simple "how many bodies do I own".

    # ---- per-fleet aggregates ----
    fleet_owner_oh = ff[..., _FLEET_OWNER_SLICE]                  # (B, F, 4)
    fleet_owner_oh = fleet_owner_oh * fm.unsqueeze(-1)
    fleet_ships_raw = torch.expm1(
        ff[..., _FLEET_SHIPS_LOG_IDX] * SHIPS_LOG_MAX,
    ) * fm                                                         # (B, F)
    total_ships_raw_fleet = torch.einsum(
        "bf,bfk->bk", fleet_ships_raw, fleet_owner_oh,
    )                                                              # (B, 4)
    # Re-normalize ONCE per-player. ``total_ships_log`` is the planets+
    # fleets combined per-player log1p (matches "this player's strength"
    # better than splitting); ``fleet_ships_in_air_log`` is fleets only
    # so the head can still learn that channel independently.
    total_ships_log = torch.log1p(
        total_ships_raw_planet + total_ships_raw_fleet,
    ) / SHIPS_LOG_MAX
    fleet_ships_in_air_log = torch.log1p(
        total_ships_raw_fleet,
    ) / SHIPS_LOG_MAX

    # ---- alive ----
    # Alive = owns at least one body OR has at least one in-air fleet.
    has_body = planet_count > 0
    has_fleet = fleet_owner_oh.sum(dim=1) > 0
    alive = (has_body | has_fleet).to(pf.dtype)                   # (B, 4)

    # ---- rank ----
    # Rank by ``total_ships_log`` (which already combines planet +
    # in-air fleet ships) with a tiny tiebreaker on planet count so
    # ties between empty stacks resolve deterministically. Dead players
    # get a very negative score so they always sort last among non-tied
    # dead. Long for CE; 0 = best (rank 0), 3 = worst.
    strength = total_ships_log + 1e-3 * planet_count
    strength = torch.where(
        alive.bool(), strength, torch.full_like(strength, -1e9),
    )
    # argsort descending → ordering[i] = which slot ranks i-th.
    ordering = torch.argsort(strength, dim=-1, descending=True)   # (B, 4) long
    rank = torch.empty_like(ordering)
    rank.scatter_(1, ordering, torch.arange(_N_PLAYER_SLOTS, device=ordering.device).expand_as(ordering))

    return CurrentStateLabels(
        total_ships_log=total_ships_log,
        planet_count=planet_count,
        production_sum=production_sum,
        fleet_ships_in_air_log=fleet_ships_in_air_log,
        comet_count=comet_count,
        alive=alive,
        rank=rank,
    )


# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #

def stack_continuous_targets(labels: CurrentStateLabels) -> torch.Tensor:
    """Stack the 5 continuous targets into ``(B, 4, 5)`` in
    ``_CONTINUOUS_CHANNELS`` order."""
    return torch.stack(
        [
            labels.total_ships_log,
            labels.planet_count,
            labels.production_sum,
            labels.fleet_ships_in_air_log,
            labels.comet_count,
        ],
        dim=-1,
    )


def current_state_loss(
    preds: dict[str, torch.Tensor],
    labels: CurrentStateLabels,
    *,
    huber_delta: float = 1.0,
    continuous_weight: float = 1.0,
    alive_weight: float = 1.0,
    rank_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the v0 supervised loss.

    Returns a dict with the total ``loss`` and per-task scalar diagnostics
    (``cont_loss``, ``alive_loss``, ``rank_loss``). Dead players are
    excluded from the rank loss (their target rank is undefined); the
    alive loss is the signal that lets the model learn who's dead.
    """
    cont_pred = preds["continuous"]                              # (B, 4, 5)
    cont_target = stack_continuous_targets(labels)               # (B, 4, 5)
    cont_loss = F.huber_loss(
        cont_pred, cont_target, delta=huber_delta, reduction="mean",
    )

    alive_loss = F.binary_cross_entropy_with_logits(
        preds["alive_logits"], labels.alive, reduction="mean",
    )

    # Rank CE — mask out dead players (their rank is the trailing tie).
    rank_logits = preds["rank_logits"]                           # (B, 4, 4)
    rank_target = labels.rank                                     # (B, 4) long
    alive_mask = labels.alive.bool()                              # (B, 4)
    if alive_mask.any():
        ce_per = F.cross_entropy(
            rank_logits[alive_mask],
            rank_target[alive_mask],
            reduction="mean",
        )
        rank_loss = ce_per
    else:
        rank_loss = rank_logits.new_zeros(())

    total = (
        continuous_weight * cont_loss
        + alive_weight * alive_loss
        + rank_weight * rank_loss
    )
    return {
        "loss": total,
        "cont_loss": cont_loss.detach(),
        "alive_loss": alive_loss.detach(),
        "rank_loss": rank_loss.detach(),
    }


__all__ = [
    "CurrentStateHead",
    "CurrentStateLabels",
    "compute_current_state_labels",
    "current_state_loss",
    "stack_continuous_targets",
]
