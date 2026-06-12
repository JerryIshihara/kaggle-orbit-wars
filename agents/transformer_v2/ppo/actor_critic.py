"""PPOActorCritic wrapper.

Wraps :class:`agents.transformer_v2.pretrain.entity_encoder.EntityPretrainModel`
for PPO. L2 is the completed perception output. The actor uses the existing
post-L2 PairHead path; the critic reads the post-L2 ``player_state`` from
``PlayerConsolidator`` plus the L2 ``glob`` summary.

There is no PPO branch from L1. ``l1_now`` may still be returned by
``forward_with_context`` for diagnostics / actor FiLM internals, but the value
path must not consume it.

Design reference: ``docs/PPO_TWO_CPU_PROTOCOL.md`` → "Actor-critic wrapper".
"""

from __future__ import annotations

import math

import torch
from torch import nn

from agents.transformer_v2.pretrain.entity_encoder import EntityPretrainModel
from agents.transformer_v2.pretrain.cross_entity import PairCompareHead


class PPOActorCritic(nn.Module):
    """PPO wrapper around the supervised entity model.

    Forward returns a dict with:

        value         (B,)              critic from post-L2 player_state
        pair_logits   (B, P, P)         actor — existing PairHead output
        frac_loc      (B, P, P)         actor — existing pair_frac (logit-normal mean)
        glob          (B, d_model)      actor L2 CLS readout — diagnostic
        ctx_now       (B, P, d_model)   L2 per-planet context (diagnostics)
        player_state  (B, 4, d_model)   per-player state from PlayerConsolidator
                                        (actor-side diagnostic)
        sigma         float             fixed frac stddev (CLI hyperparameter)
    """

    def __init__(
        self,
        entity_model: EntityPretrainModel,
        *,
        sigma: float = 0.35,
        value_hidden: int | None = None,
        allow_debug_glob_critic: bool = False,
        critic_model: nn.Module | None = None,
        reward_decomp: bool = False,
        win_weight: float = 1.0,
        signal_weights: list[float] | None = None,
        value_gamma: float = 0.997,
    ):
        super().__init__()
        self.entity_model = entity_model
        d = entity_model.d_model
        h = value_hidden if value_hidden is not None else d
        self.allow_debug_glob_critic = bool(allow_debug_glob_critic)

        # Production critic. PairCompareHead is pretrained by the
        # cross-entity critic objective and returns ordered logits
        # P(player_i outperforms player_j). PPO averages learner slot 0
        # against valid opponents and maps the probability to [-1, +1].
        self.pair_compare = PairCompareHead(
            d,
            hidden=h,
            n_layers=3,
        )
        if critic_model is not None:
            if not hasattr(critic_model, "pair_compare"):
                raise ValueError(
                    "critic_model must be a redesigned CrossEntityCriticModel "
                    "with pair_compare.* weights."
                )
            self.pair_compare.load_state_dict(critic_model.pair_compare.state_dict())
            if self.entity_model.consolidator is None:
                raise ValueError(
                    "PPO critic checkpoint requires an EntityPretrainModel built "
                    "with PlayerConsolidator."
                )
            if hasattr(critic_model, "consolidator"):
                self.entity_model.consolidator.load_state_dict(
                    critic_model.consolidator.state_dict(),
                )

        # Legacy debug fallback only for plumbing against a ckpt built with
        # ``--no-consolidator``. Production PPO should use pair_compare over
        # post-L2 player_state.
        self.value_head = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

        # ---- Design A reward-decomposition critic (optional) ----
        # The shaped reward uses a terminal win indicator z∈{0,1} plus PBRS
        # γΦ(s')−Φ(s). Match that scale directly:
        #
        #   V(s) = win_weight·P(win|s) − Φ(s) + residual(glob)
        #
        # using ONLY the pretrained win head (learner slot 0). The forward/back/
        # survives/rank heads stay loaded to warm the shared value trunk, but are
        # auxiliary here, not value terms. The signal weights and γ live on the
        # reward side where Φ=Σwᵢsᵢ is calculated from stored step features.
        self.reward_decomp = bool(reward_decomp)
        self.win_weight = float(win_weight)
        if self.reward_decomp and getattr(entity_model, "value_heads", None) is None:
            raise ValueError(
                "reward_decomp critic needs entity_model built with "
                "with_value_heads=True (and the win head loaded).")

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

        Phase 0: trainable = post-L2 critic PairCompareHead + PairHead
        pair_logits/pair_frac heads. Legacy debug fallback trains value_head
        only if the model has no PlayerConsolidator.
        Phase 1: + PairHead trunk + FiLM + L4 JointRoleAttention.
        Phase 2: + L3 dual_role (DualRoleAttention).
        L0-L2 are perception and stay frozen at first; PlayerConsolidator also
        stays frozen in Phase 0 so only the pairwise critic head receives value-loss
        gradients.
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
            # Phase 1: also unfreeze PairHead trunk + FiLM + L4. (L3 stays
            # frozen until Phase 2 below.)
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

        if phase >= 2:
            # Phase 2: also unfreeze L3 DualRoleAttention. The attribute mirrors
            # joint_role (L4) on the entity model; guard in case it differs.
            if hasattr(self.entity_model, "dual_role") and \
                    self.entity_model.dual_role is not None:
                for p in self.entity_model.dual_role.parameters():
                    p.requires_grad_(True)

        for p in self.pair_compare.parameters():
            p.requires_grad_(True)
        for p in self.value_head.parameters():
            p.requires_grad_(False)
        if self.entity_model.consolidator is None and self.allow_debug_glob_critic:
            # Plumbing-only fallback for actor-only ckpts. This path still
            # reads L2 glob, never L1.
            for p in self.pair_compare.parameters():
                p.requires_grad_(False)
            for p in self.value_head.parameters():
                p.requires_grad_(True)

        # Return param counts by group for the train log.
        return {
            "value_head": sum(p.numel() for p in self.value_head.parameters()
                              if p.requires_grad),
            "critic_pair_compare": sum(
                p.numel() for p in self.pair_compare.parameters()
                if p.requires_grad
            ),
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
            "l3_dual_role": sum(
                p.numel()
                for p in getattr(self.entity_model, "dual_role", None).parameters()
                if p.requires_grad
            ) if getattr(self.entity_model, "dual_role", None) is not None else 0,
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
        planet_owner_oh: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Use the supervised model's forward_with_context, which returns
        # ``glob``, ``ctx_now``, and ``player_state`` alongside the actor's
        # pair outputs. If a future refactor strips ``forward_with_context``,
        # this is the single call to update.
        out = self.entity_model.forward_with_context(
            planet_tokens, fleet_tokens, routing, planet_mask,
            is_comet=is_comet, pair_type_ids=pair_type_ids,
            planet_owner_oh=planet_owner_oh,
        )

        if self.reward_decomp and out["player_state"] is not None:
            # Design A: calibrated win probability for the terminal z∈{0,1}
            # reward, minus the current PBRS potential, plus a zero-init residual
            # fine-tuned on shaped returns. The forward/survives/rank/back heads
            # stay loaded to warm the shared value trunk but are auxiliary here.
            vh = self.entity_model.value_heads(out["glob"], out["player_state"])
            win_logit = vh["win"][:, 0]                              # (B,) learner slot 0
            v_win = self.win_weight * torch.sigmoid(win_logit)
            residual = self.value_head(out["glob"]).squeeze(-1)      # zero-init, fine-tuned
            # subtract the CALCULATED potential Φ(s) (PBRS return ≈ win − Φ); phi is
            # carried in from the rollout/minibatch. None at rollout time (Φ not yet
            # computed) → the stored value is corrected by −Φ post-rollout instead.
            phi_term = phi if phi is not None else 0.0
            value = v_win - phi_term + residual
            player_valid = None     # not needed by the reward-decomp critic
        elif out["player_state"] is not None:
            # Post-L2 critic: PlayerConsolidator consumes L2 ctx_now, so the
            # value path is downstream of completed perception and never reads
            # L1 tokens.
            player_state = out["player_state"]                      # (B, 4, d)
            pair_value_logits = self.pair_compare(player_state, out["glob"])  # (B,4,4)
            player_valid = self._infer_player_valid(
                player_state=player_state,
                planet_owner_oh=planet_owner_oh,
                routing=routing,
            ).to(pair_value_logits.dtype)
            opp_mask = player_valid.clone()
            opp_mask[:, 0] = 0.0
            p0 = torch.sigmoid(pair_value_logits[:, 0, :])
            value_prob = (p0 * opp_mask).sum(dim=-1) / opp_mask.sum(dim=-1).clamp(min=1.0)
            value = 2.0 * value_prob - 1.0
        elif self.allow_debug_glob_critic:
            # Legacy debug fallback: critic reads actor ``glob`` (L2 CLS), not
            # L1. This is only for ckpts intentionally built without the
            # PlayerConsolidator.
            value = self.value_head(out["glob"]).squeeze(-1)
        else:
            raise RuntimeError(
                "PPO critic requires EntityPretrainModel(with_consolidator=True) "
                "so post-L2 player_state is available."
            )

        return {
            "value": value,
            "pair_logits": out["pair_logits"],
            "frac_loc": out["pair_frac"],
            # v4 contract: α0 head output (None on models without the head).
            "alloc_conc": out.get("alloc_conc"),
            "glob": out["glob"],
            "ctx_now": out["ctx_now"],
            "player_state": out["player_state"],
            "player_valid": player_valid if out["player_state"] is not None else None,
            "sigma": self.sigma,
        }

    # --------------------------------------------------------------------- #
    # Sigma schedule helper (sigma stays a buffer; CLI may set per iter).   #
    # --------------------------------------------------------------------- #
    def set_sigma(self, value: float) -> None:
        if not (1e-3 < value < 10.0):
            raise ValueError(f"sigma out of safe range: {value}")
        self.sigma.fill_(float(value))

    def _infer_player_valid(
        self,
        *,
        player_state: torch.Tensor,
        planet_owner_oh: torch.Tensor | None,
        routing: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Best-effort real-player mask from current/window ownership tensors."""
        B, S, _ = player_state.shape
        device = player_state.device
        valid = torch.zeros(B, S, dtype=torch.bool, device=device)
        valid[:, 0] = True

        if planet_owner_oh is not None:
            owner = planet_owner_oh[..., :S].to(device=device)
            if owner.dim() == 4:
                seen = owner.sum(dim=(1, 2)) > 0
            elif owner.dim() == 3:
                seen = owner.sum(dim=1) > 0
            else:
                seen = None
            if seen is not None:
                valid |= seen

        fleet_owner = routing.get("fleet_owner_slot")
        fleet_mask = routing.get("fleet_mask")
        if isinstance(fleet_owner, torch.Tensor) and isinstance(fleet_mask, torch.Tensor):
            owner = fleet_owner.to(device=device).long()
            mask = fleet_mask.to(device=device).bool()
            if owner.dim() == 3:
                owner = owner.reshape(B, -1)
                mask = mask.reshape(B, -1)
            seen_fleet = torch.zeros(B, S, dtype=torch.bool, device=device)
            for slot in range(S):
                seen_fleet[:, slot] = ((owner == slot) & mask).any(dim=-1)
            valid |= seen_fleet

        # If the current/window tensors only show the learner, keep one
        # opponent slot so V(s) remains well-defined instead of averaging over
        # an empty set.
        no_opp = ~valid[:, 1:].any(dim=-1)
        if bool(no_opp.any()):
            valid[no_opp, 1] = True
        return valid


def default_value_hidden(entity_model: EntityPretrainModel) -> int:
    """Pick a default hidden width for the pairwise value head.

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
