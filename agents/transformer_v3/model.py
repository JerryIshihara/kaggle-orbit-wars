"""EntityPretrainModelV3 — v2 model with the dual-rate L2 swapped in.

Subclasses ``transformer_v2.pretrain.entity_encoder.EntityPretrainModel``
and replaces ``self.cross`` with :class:`DualRateCrossEntity`. Nothing
else changes:

  * ``forward`` / ``forward_with_context`` are inherited verbatim — the
    dual module returns the fused current step as a rank-4 ``(B,1,P,d)``
    so every existing ``[:, -1]`` read (ctx, mask, l1, is_comet,
    pair_type, owner_oh) stays correct against the union stack, whose
    last frame is offset 0.
  * ``freeze_perception`` / ``freeze_l1_only`` / ``freeze_below_l2``
    iterate ``self.cross.parameters()`` — that now covers both branches
    plus the two fusion layers, which is exactly the L2 group.

Input contract: temporal inputs must be stacked at
``history.UNION_HISTORY_OFFSETS`` (T=18). ``n_steps`` passed by callers
is ignored (the union length is structural, not tunable) — a mismatch
is loudly logged rather than silently accepted.
"""

from __future__ import annotations

import torch

from ..transformer_v2.pretrain.entity_encoder import (
    EntityPretrainModel,
    _adapt_cross_step_embed,
)
from .dual_cross import DualRateCrossEntity
from .history import (
    LONG_HISTORY_OFFSETS,
    SHORT_HISTORY_OFFSETS,
    UNION_HISTORY_OFFSETS,
    N_UNION,
)


class EntityPretrainModelV3(EntityPretrainModel):
    ARCH = "dual_rate_l2_v3"

    def __init__(
        self,
        d_model: int = 256,
        *,
        n_steps: int | None = None,
        with_short_aux: bool = True,
        with_alloc_conc: bool = False,
        **kw,
    ):
        if n_steps is not None and int(n_steps) != N_UNION:
            print(
                f"[v3] n_steps={n_steps} ignored — dual-rate L2 fixes the "
                f"input stack to the {N_UNION}-frame union "
                f"(long T=10@5 + short T=10@2)",
                flush=True,
            )
        super().__init__(d_model=d_model, n_steps=N_UNION, **kw)
        # Replace the single-rate L2 built by the base ctor. Reuse the
        # same structural knobs so branch layers mirror v2's L2 exactly.
        self.cross = DualRateCrossEntity(
            d_model,
            n_heads=self.cross_n_heads,
            n_layers=self.cross_n_layers,
            ff_mult=kw.get("cross_ff_mult", 2),
            dropout=kw.get("dropout", 0.0),
        )
        # v3.1: player_state comes from the per-branch player CLS tokens
        # inside L2 (asymmetric mask keeps planet/global outputs
        # untouched) — the PlayerConsolidator is REMOVED (~-1.5M params,
        # one fewer attention pass). The base ctor built it because the
        # value heads demand with_consolidator=True; drop it here.
        self.consolidator = None
        # Short-horizon aux heads: direct supervision for the SHORT
        # branch (bypasses the zero-init fusion gate, which blocks main-
        # loss gradient to the branch until fusion weights move). Train-
        # time only; deploy/PPO never call them (small dead params).
        self.with_short_aux = bool(with_short_aux)
        if self.with_short_aux:
            from .short_horizon import ShortHorizonHeads
            self.short_heads = ShortHorizonHeads(
                d_model, dropout=kw.get("dropout", 0.0),
            )
        else:
            self.short_heads = None
        # Contract v4: per-source Dirichlet concentration α0 off the L4
        # source tokens (the mean stays the existing frac softmax).
        self.with_alloc_conc = bool(with_alloc_conc)
        if self.with_alloc_conc:
            from .dirichlet_alloc import AllocConcentrationHead
            self.alloc_conc_head = AllocConcentrationHead(d_model)
        else:
            self.alloc_conc_head = None

    def forward_with_context(
        self,
        planet_tokens: torch.Tensor,
        fleet_tokens: torch.Tensor,
        routing: dict[str, torch.Tensor],
        planet_mask: torch.Tensor,
        is_comet: torch.Tensor | None = None,
        pair_type_ids: torch.Tensor | None = None,
        planet_owner_oh: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """v3.1 override of the base method: the FULL owner one-hot stack
        is zero-init projected onto the entity tokens before the dual L2
        (the base only sliced ``[:, -1]`` for the consolidator), and
        ``player_state`` is read from the in-L2 player tokens instead of
        the removed PlayerConsolidator. Everything else mirrors the base.
        """
        is_temporal = planet_tokens.dim() == 4
        if is_temporal:
            B, T, P, d = planet_tokens.shape
            F = fleet_tokens.shape[2]
            entity_tokens = self.entity(
                planet_tokens.reshape(B * T, P, d),
                fleet_tokens.reshape(B * T, F, d),
                routing["fleet_target_idx"].reshape(B * T, F),
                routing["fleet_source_idx"].reshape(B * T, F),
                routing["fleet_owner_slot"].reshape(B * T, F),
                routing["fleet_ships_log"].reshape(B * T, F),
                routing["fleet_eta_norm"].reshape(B * T, F),
                routing["fleet_mask"].reshape(B * T, F),
                planet_mask=planet_mask.reshape(B * T, P),
            ).reshape(B, T, P, d)
        else:
            entity_tokens = self.entity(
                planet_tokens, fleet_tokens,
                routing["fleet_target_idx"], routing["fleet_source_idx"],
                routing["fleet_owner_slot"], routing["fleet_ships_log"],
                routing["fleet_eta_norm"], routing["fleet_mask"],
                planet_mask=planet_mask,
            )

        ctx_full, glob = self.cross(
            entity_tokens, planet_mask, owner_oh=planet_owner_oh,
        )
        if is_temporal:
            ctx_now = ctx_full[:, -1]
            planet_mask_now = planet_mask[:, -1]
            l1_now = entity_tokens[:, -1]
        else:
            ctx_now = ctx_full
            planet_mask_now = planet_mask
            l1_now = entity_tokens

        if is_comet is None:
            is_comet_now = torch.zeros(
                planet_mask_now.shape, dtype=torch.bool,
                device=planet_mask_now.device,
            )
        elif is_comet.dim() == 3:
            is_comet_now = is_comet[:, -1].to(torch.bool)
        else:
            is_comet_now = is_comet.to(torch.bool)

        if pair_type_ids is not None and pair_type_ids.dim() == 4:
            pair_type_now = pair_type_ids[:, -1].to(torch.long)
        elif pair_type_ids is not None:
            pair_type_now = pair_type_ids.to(torch.long)
        else:
            pair_type_now = None

        player_state = self.cross.last_player_state

        if self.skip_l34:
            source_joint = ctx_now
            target_joint = ctx_now
        else:
            source_aware, target_aware = self.dual_role(ctx_now, planet_mask_now)
            source_joint, target_joint = self.joint_role(
                source_aware, target_aware, planet_mask_now,
            )

        B_now, P_now = planet_mask_now.shape
        pair_valid = (
            planet_mask_now.unsqueeze(2)
            & planet_mask_now.unsqueeze(1)
        )
        eye = torch.eye(P_now, dtype=torch.bool, device=pair_valid.device)
        pair_valid = pair_valid & ~eye.unsqueeze(0)

        heads = self.pair_head(
            source_joint, target_joint, ctx_now,
            l1_tokens=l1_now,
            is_comet=is_comet_now,
            pair_type_ids=pair_type_now,
            pair_valid=pair_valid,
        )
        out = {
            "pair_logits": heads["pair_logits"],
            "pair_frac": heads["pair_frac"],
            "glob": glob,
            "ctx_now": ctx_now,
            "player_state": player_state,
            "source_joint": source_joint,
            "target_joint": target_joint,
            "l1_now": l1_now,
        }
        if self.alloc_conc_head is not None:
            out["alloc_conc"] = self.alloc_conc_head(source_joint)
        # Same merge as the base: one forward yields action + value preds.
        if self.value_heads is not None and player_state is not None:
            out.update(self.value_heads(glob, player_state))
        return out

    def short_aux_loss(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Short-branch aux loss for the batch JUST forwarded.

        Must be called between this model's forward on ``batch`` and any
        other forward — it reads the pre-fusion branch stash, which every
        forward overwrites.
        """
        from .short_horizon import short_horizon_loss
        assert self.short_heads is not None, "built with with_short_aux=False"
        ctx_s = self.cross.last_ctx_short_now
        assert ctx_s is not None, "forward must run before short_aux_loss"
        pm = batch["planet_mask"]
        pm_now = pm[:, -1] if pm.dim() == 3 else pm
        assert ctx_s.shape[0] == pm_now.shape[0], "stash/batch size mismatch"
        return short_horizon_loss(self.short_heads(ctx_s), batch, pm_now)

    #: Stamped into the run config by the pretrain driver so checkpoints
    #: are self-describing for the (later) runner/PPO adaptation.
    @property
    def config_extra(self) -> dict:
        return {
            "arch": self.ARCH,
            "n_steps": N_UNION,
            "history_offsets": list(UNION_HISTORY_OFFSETS),
            "long_history_offsets": list(LONG_HISTORY_OFFSETS),
            "short_history_offsets": list(SHORT_HISTORY_OFFSETS),
            "with_short_aux": self.with_short_aux,
            "with_consolidator": False,
            "player_state_source": "l2_player_tokens",
            **(
                {
                    "action_contract":
                        "bounded_k_select_dirichlet_alloc_v4",
                    "select_k_max": 3,
                    "with_alloc_conc": True,
                }
                if self.with_alloc_conc else {}
            ),
        }


def adapt_v2_state_dict(
    sd: dict[str, torch.Tensor],
    model: EntityPretrainModelV3 | None = None,
) -> dict[str, torch.Tensor]:
    """Map a v2 checkpoint state dict onto the v3 module tree.

    * ``cross.<k>``  -> ``cross.long.<k>`` (identical window — exact copy)
                     -> ``cross.short.<k>`` (same structure; better-than-
                        random init for the fresh branch). The short
                        branch's ``step_embed`` rows are re-mapped by
                        nearest offset (45..0 -> 18..0) via the existing
                        v2 adapter so its few overlapping offsets (10, 0)
                        keep their trained rows.
    * everything else passes through unchanged.
    * fusion layers are intentionally ABSENT from the result — they keep
      their zero-init ``[I|0]``, which is what makes the warm-started v3
      reproduce v2 exactly at init.

    Use with ``model.load_state_dict(adapted, strict=False)`` and assert
    the only missing keys are ``cross.fuse_*``.
    """
    if any(k.startswith("cross.long.") for k in sd):
        # Already v3-shaped (e.g. warm-starting v3.1 from a stopped v3
        # dual-rate run): branch/fusion/aux keys map 1:1 — only the
        # removed consolidator is dropped; the player-token machinery
        # (owner_proj / player_tokens / fuse_player) stays fresh.
        return {k: v for k, v in sd.items()
                if not k.startswith("consolidator.")}

    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("consolidator."):
            # v3.1: PlayerConsolidator removed — player_state comes from
            # the in-L2 player tokens. Nothing to map these onto.
            continue
        if not k.startswith("cross."):
            out[k] = v
            continue
        rest = k[len("cross."):]
        out[f"cross.long.{rest}"] = v
        if rest == "step_embed":
            dst = v.new_zeros(len(SHORT_HISTORY_OFFSETS), v.shape[1])
            out["cross.short.step_embed"] = _adapt_cross_step_embed(
                src=v,
                dst=dst,
                src_offsets=LONG_HISTORY_OFFSETS,
                dst_offsets=SHORT_HISTORY_OFFSETS,
            )
        else:
            out[f"cross.short.{rest}"] = v
    return out
