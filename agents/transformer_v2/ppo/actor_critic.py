"""PPOActorCritic wrapper.

Wraps :class:`agents.transformer_v2.pretrain.entity_encoder.EntityPretrainModel`
with the single new PPO module — a 3-Linear MLP ``value_head`` on L2's
``glob``. The actor is the existing PairHead outputs; no new actor heads.

Design reference: ``docs/PPO_TWO_CPU_PROTOCOL.md`` → "Actor-critic wrapper".
"""

from __future__ import annotations

import math

import torch
from torch import nn

from agents.transformer_v2.pretrain.entity_encoder import EntityPretrainModel


class PPOActorCritic(nn.Module):
    """PPO wrapper around the supervised entity model.

    Forward returns a dict with:

        value         (B,)              critic — V(s) from glob
        pair_logits   (B, P, P)         actor — existing PairHead output
        frac_loc      (B, P, P)         actor — existing pair_frac (logit-normal mean)
        glob          (B, d_model)      L2 CLS readout (returned for diagnostics)
        ctx_now       (B, P, d_model)   L2 per-planet context (returned for diagnostics)
        sigma         float             fixed frac stddev (CLI hyperparameter)
    """

    def __init__(
        self,
        entity_model: EntityPretrainModel,
        *,
        sigma: float = 0.35,
        value_hidden: int | None = None,
    ):
        super().__init__()
        self.entity_model = entity_model
        d = entity_model.d_model
        h = value_hidden if value_hidden is not None else d

        # 3-Linear MLP. Identity-init the final Linear so V(s) starts near 0 —
        # avoids a large initial value bootstrap shock when GAE first runs.
        self.value_head = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

        # sigma is a fixed hyperparameter (CLI flag at the training driver),
        # registered as a buffer so it serializes with the checkpoint without
        # appearing as a trainable parameter.
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    # --------------------------------------------------------------------- #
    # Freeze utility — Phase 0 / Phase 1 freeze sets per the design.        #
    # --------------------------------------------------------------------- #
    def freeze_for_phase(self, phase: int) -> dict[str, int]:
        """Apply the design's freeze rules and return the trainable-param
        breakdown for logging.

        Phase 0: trainable = value_head + PairHead pair_logits/pair_frac heads.
        Phase 1: + PairHead trunk + FiLM + L4 JointRoleAttention.
        L0-L3 + PlayerContext / Strategy stay frozen at every phase.
        """
        # Default: freeze everything in the supervised stack.
        for p in self.entity_model.parameters():
            p.requires_grad_(False)

        ph = self.entity_model.pair_head
        head_modules = [ph.pair_head, ph.pair_frac_head]

        # Phase 0: unfreeze the two action heads.
        for mod in head_modules:
            for p in mod.parameters():
                p.requires_grad_(True)

        if phase >= 1:
            # Phase 1: also unfreeze PairHead trunk + FiLM + L4. Keep L3 frozen.
            phase1_modules = [
                ph.trunk,
                ph.film_proj,
                ph.src_proj,
                ph.tgt_proj,
                ph.ctx_proj,
                ph.pair_type_embed,
                self.entity_model.joint_role,  # L4
            ]
            for mod in phase1_modules:
                for p in mod.parameters():
                    p.requires_grad_(True)
            # film_alpha is a bare nn.Parameter, not a submodule
            if hasattr(ph, "film_alpha"):
                ph.film_alpha.requires_grad_(True)

        # value_head is always trainable.
        for p in self.value_head.parameters():
            p.requires_grad_(True)

        # Return param counts by group for the train log.
        return {
            "value_head": sum(p.numel() for p in self.value_head.parameters()
                              if p.requires_grad),
            "pair_logits_head": sum(p.numel() for p in ph.pair_head.parameters()
                                     if p.requires_grad),
            "pair_frac_head": sum(p.numel() for p in ph.pair_frac_head.parameters()
                                   if p.requires_grad),
            "pair_trunk_film": sum(
                p.numel()
                for mod in (ph.trunk, ph.film_proj, ph.src_proj, ph.tgt_proj,
                            ph.ctx_proj, ph.pair_type_embed)
                for p in mod.parameters() if p.requires_grad
            ),
            "joint_role_l4": sum(p.numel() for p in
                                  self.entity_model.joint_role.parameters()
                                  if p.requires_grad),
        }

    # --------------------------------------------------------------------- #
    # Forward                                                                #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        planet_tokens: torch.Tensor,
        fleet_tokens: torch.Tensor,
        routing: dict[str, torch.Tensor],
        planet_mask: torch.Tensor,
        *,
        is_comet: torch.Tensor | None = None,
        pair_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Use the supervised model's forward_with_context, which returns
        # ``glob`` and ``ctx_now`` alongside the actor's pair outputs.
        # If a future refactor strips ``forward_with_context``, this is the
        # single call to update.
        out = self.entity_model.forward_with_context(
            planet_tokens, fleet_tokens, routing, planet_mask,
            is_comet=is_comet, pair_type_ids=pair_type_ids,
        )

        value = self.value_head(out["glob"]).squeeze(-1)

        return {
            "value": value,
            "pair_logits": out["pair_logits"],
            "frac_loc": out["pair_frac"],
            "glob": out["glob"],
            "ctx_now": out["ctx_now"],
            "sigma": self.sigma,
        }

    # --------------------------------------------------------------------- #
    # Sigma schedule helper (sigma stays a buffer; CLI may set per iter).   #
    # --------------------------------------------------------------------- #
    def set_sigma(self, value: float) -> None:
        if not (1e-3 < value < 10.0):
            raise ValueError(f"sigma out of safe range: {value}")
        self.sigma.fill_(float(value))


def default_value_hidden(entity_model: EntityPretrainModel) -> int:
    """Pick a default hidden width for the value MLP.

    Mirrors the protocol's "H = d_model" default. Exposed as a helper so the
    CLI can grep one place if it ever changes.
    """
    return int(entity_model.d_model)


def log_sigma_for(schedule: str, iter_K: int, total_iters: int) -> float:
    """Return sigma for iter K under one of the simple schedules.

    Used by the CLI when the user passes ``--sigma-schedule``. Defaults to
    constant 0.35 if no schedule is requested.
    """
    if schedule == "constant":
        return 0.35
    if schedule == "linear":
        # Linearly decay 0.35 -> 0.15 over total_iters.
        frac = min(1.0, max(0.0, iter_K / max(1, total_iters - 1)))
        return 0.35 + (0.15 - 0.35) * frac
    if schedule == "cosine":
        frac = min(1.0, max(0.0, iter_K / max(1, total_iters - 1)))
        return 0.15 + 0.5 * (0.35 - 0.15) * (1 + math.cos(math.pi * frac))
    raise ValueError(f"unknown sigma schedule: {schedule}")
