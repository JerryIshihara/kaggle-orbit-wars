"""Shared explicit value-pretrain heads (the /tmp value_pretrain_design.md set).

This module factors the value/momentum heads out of
:class:`agents.transformer_v2.pretrain.cross_entity.CrossEntityCriticModel` so
the SAME head set can be attached to the action model
(:class:`agents.transformer_v2.pretrain.entity_encoder.EntityPretrainModel`),
letting the action head (PairHead, off L3/L4) and the value heads (off the
PlayerConsolidator) **branch from the one shared L2 backbone** and train
jointly.

Forward consumes ``glob (B, d)`` + ``player_state (B, O, d)`` (O = owner/player
slots, normally 4) — exactly what ``forward_with_context`` already returns — and
emits the same dict keys as ``CrossEntityCriticModel.forward``'s value subset,
so ``cross_entity.compute_loss_value_pretrain`` consumes the output unchanged.

Kept import-light on purpose: it imports only ``_build_mlp`` (from
``consolidator_heads``) and ``CROSS_ENTITY_VALUE_HORIZONS`` (from the
featurizer). It must NOT import ``cross_entity`` (that would create an
``entity_encoder -> value_heads -> cross_entity -> entity_encoder`` cycle).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .consolidator_heads import _build_mlp
from ..featurizer.entity_featurizer import CROSS_ENTITY_VALUE_HORIZONS

# Mirrors cross_entity.VALUE_CURRENT_STAT_CHANNELS (kept here as the canonical
# definition; cross_entity may import these from this module later). Order is
# the channel order of the current-state regression head.
VALUE_CURRENT_STAT_CHANNELS: tuple[str, ...] = (
    "ship_share",
    "production_share",
    "planet_share",
    "fleet_ship_share",
    "score_adv_norm",
)
VALUE_N_CURRENT_STATS: int = len(VALUE_CURRENT_STAT_CHANNELS)


def _zero_last_linear(module: nn.Module) -> None:
    """Zero-init the final Linear (weight + bias) so the head starts neutral."""
    last = module[-1] if isinstance(module, nn.Sequential) else None
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


class ValuePretrainHeads(nn.Module):
    """Explicit supervised value/advantage heads over ``[glob ‖ player_state]``.

    Heads on the trunked feature ``value_h = value_trunk(feat)`` (board + own
    detail, zero-init final layer so they start neutral):

      * ``win_logit``        (B, O)      final-win BCE logit
      * ``rank_score``       (B, O)      ListMLE ranking score
      * ``final_score_adv``  (B, O)      terminal score-advantage (Huber)
      * ``current_stats``    (B, O, C)   current resource shares (Huber)

    Heads on the raw ``feat`` (anti-shortcut temporal anchors + horizon value):

      * ``is_ahead_logits``  (B, O, H)   per-horizon "ahead" logit (H horizons)
      * ``dships_back``/``dplanets_back``/``trend_back`` (B,O[,3])  backward momentum
      * ``dships_fwd``/``future_score_adv``/``survives_fwd`` (B,O)  forward momentum
      * ``slope_back``/``slope_fwd``  (B,)   learner-slot OLS slopes

    Output dict keys are identical to ``CrossEntityCriticModel.forward``'s value
    subset (minus ``pair_logits``/``player_state``/``glob`` which the host model
    already owns), so ``compute_loss_value_pretrain`` consumes it unchanged.
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_horizons: int = len(CROSS_ENTITY_VALUE_HORIZONS),
        trunk_n_layers: int = 2,
        head_n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        feat_dim = 2 * d_model                           # [glob ‖ player_state]

        self.value_trunk = _build_mlp(
            in_dim=feat_dim, hidden=d_model, out_dim=d_model,
            n_layers=trunk_n_layers, dropout=dropout,
        )
        # Heads over the trunked feature (B, O, d).
        self.win_logit_head = _build_mlp(
            in_dim=d_model, hidden=d_model, out_dim=1,
            n_layers=head_n_layers, dropout=dropout,
        )
        self.rank_score_head = _build_mlp(
            in_dim=d_model, hidden=d_model, out_dim=1,
            n_layers=head_n_layers, dropout=dropout,
        )
        self.final_score_adv_head = _build_mlp(
            in_dim=d_model, hidden=d_model, out_dim=1,
            n_layers=head_n_layers, dropout=dropout,
        )
        self.current_stats_head = _build_mlp(
            in_dim=d_model, hidden=d_model, out_dim=VALUE_N_CURRENT_STATS,
            n_layers=head_n_layers, dropout=dropout,
        )
        for head in (
            self.win_logit_head, self.rank_score_head,
            self.final_score_adv_head, self.current_stats_head,
        ):
            _zero_last_linear(head)

        # Heads over the raw [glob ‖ player_state] feature (B, O, 2d).
        self.is_ahead_head = _build_mlp(
            in_dim=feat_dim, hidden=d_model, out_dim=n_horizons,
            n_layers=head_n_layers, dropout=dropout,
        )
        _zero_last_linear(self.is_ahead_head)

        def _mlp() -> nn.Module:
            return _build_mlp(
                in_dim=feat_dim, hidden=d_model, out_dim=1, n_layers=2,
                dropout=dropout,
            )

        self.head_dships_back = _mlp()
        self.head_dplanets_back = _mlp()
        self.head_dships_fwd = _mlp()
        self.head_future_score_adv = _mlp()
        self.head_survives_fwd = _mlp()
        self.head_trend_back = _build_mlp(
            in_dim=feat_dim, hidden=d_model, out_dim=3, n_layers=2, dropout=dropout,
        )
        self.head_slope_back = _mlp()                    # learner slot only
        self.head_slope_fwd = _mlp()                     # learner slot only

    def forward(
        self,
        glob: torch.Tensor,            # (B, d)
        player_state: torch.Tensor,    # (B, O, d)
    ) -> dict[str, torch.Tensor]:
        if glob.dim() != 2 or player_state.dim() != 3:
            raise ValueError(
                f"expected glob (B,d) + player_state (B,O,d); got "
                f"{tuple(glob.shape)} and {tuple(player_state.shape)}"
            )
        O = player_state.size(1)
        glob_b = glob.unsqueeze(1).expand(-1, O, -1)                 # (B, O, d)
        feat = torch.cat([glob_b, player_state], dim=-1)            # (B, O, 2d)
        value_h = self.value_trunk(feat)                            # (B, O, d)
        return {
            "win_logit": self.win_logit_head(value_h).squeeze(-1),       # (B, O)
            "rank_score": self.rank_score_head(value_h).squeeze(-1),     # (B, O)
            "final_score_adv": self.final_score_adv_head(value_h).squeeze(-1),
            "current_stats": self.current_stats_head(value_h),          # (B, O, C)
            "is_ahead_logits": self.is_ahead_head(feat),               # (B, O, H)
            "dships_back": self.head_dships_back(feat).squeeze(-1),     # (B, O)
            "dplanets_back": self.head_dplanets_back(feat).squeeze(-1), # (B, O)
            "trend_back": self.head_trend_back(feat),                  # (B, O, 3)
            "dships_fwd": self.head_dships_fwd(feat).squeeze(-1),       # (B, O)
            "future_score_adv": self.head_future_score_adv(feat).squeeze(-1),
            "survives_fwd": self.head_survives_fwd(feat).squeeze(-1),   # (B, O)
            "slope_back": self.head_slope_back(feat[:, 0]).squeeze(-1),   # (B,)
            "slope_fwd": self.head_slope_fwd(feat[:, 0]).squeeze(-1),     # (B,)
        }
