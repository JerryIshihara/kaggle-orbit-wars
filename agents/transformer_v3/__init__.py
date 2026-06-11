"""transformer_v3 — dual-rate world perception (two-timescale L2).

v3 = the newest transformer_v2 design (d256 / T=10-strided history /
head3+cond3 / consolidator + value heads / multinomial-alloc contracts)
plus ONE architectural change at L2:

    L2 becomes TWO parallel ``CrossEntityAttention`` branches over the
    same L1 entity tokens, at different temporal rates:

      * LONG  — T=10 @ stride 5  → offsets (45, 40, ..., 5, 0), ~50
        turns of lookback. Identical to v2's window: economy trends,
        production curves, momentum.
      * SHORT — T=10 @ stride 2  → offsets (18, 16, ..., 2, 0), 20
        turns. Fine recency: launch cadence, skirmish outcomes,
        dodge/intercept dynamics (more valuable under the 1.30.x
        swept-collision physics).

    After L2 the two current-step readouts are CONCATENATED and fused
    back to d_model by a zero-initialized projection (weight = [I; 0]):
    at init the fused output is EXACTLY the long branch, so a model
    warm-started from a v2 checkpoint reproduces the v2 forward bit-for-
    bit and the short branch fades in through training. The same applies
    to the global CLS token.

Everything above L2 (consolidator, L3 dual-role, L4 joint-role, pair
head, value heads) is inherited from v2 unchanged — they consume the
fused ``ctx_now``/``glob`` exactly as before.

Input contract: ONE history stack at the UNION of both offset sets
(18 frames; see ``history.UNION_HISTORY_OFFSETS``). The branches
index-select their frames inside the module, so datasets/runners only
need to walk a different offset tuple — no second tensor set, and L0/L1
run once over 18 frames instead of 10 (~1.8x perception FLOPs, L2 2x).

The pair/value caches store every turn as single frames and stack
lazily at ``__getitem__``; switching to union offsets is a metadata
override, NOT a cache rebuild (same mechanism as the v2 T6->T10
restack, including the coverage probe).

Code-only for now (2026-06-12): pretrain driver wires through
``transformer_v2.pretrain.joint_pretrain.train_joint`` injection
points; no notebook yet, PPO/runner adaptation comes after a v3
pretrain exists.
"""

from .history import (
    LONG_HISTORY_OFFSETS,
    SHORT_HISTORY_OFFSETS,
    UNION_HISTORY_OFFSETS,
    LONG_SLOT_IDX,
    SHORT_SLOT_IDX,
    N_UNION,
    HISTORY_MAX_LOOKBACK,
)
from .dual_cross import DualRateCrossEntity
from .model import EntityPretrainModelV3, adapt_v2_state_dict

__all__ = [
    "LONG_HISTORY_OFFSETS",
    "SHORT_HISTORY_OFFSETS",
    "UNION_HISTORY_OFFSETS",
    "LONG_SLOT_IDX",
    "SHORT_SLOT_IDX",
    "N_UNION",
    "HISTORY_MAX_LOOKBACK",
    "DualRateCrossEntity",
    "EntityPretrainModelV3",
    "adapt_v2_state_dict",
]
