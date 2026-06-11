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

    def __init__(self, d_model: int = 256, *, n_steps: int | None = None, **kw):
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
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
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
