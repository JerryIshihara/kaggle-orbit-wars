"""Dual-rate history windows for transformer_v3.

v2 ships one uniform window (``transformer_v2.history.HISTORY_OFFSETS``,
T=10 @ stride 5). v3 keeps that as the LONG branch and adds a SHORT
branch at stride 2 covering the last 20 turns:

    LONG  slot:    0    1    2    3    4    5    6    7    8    9
          offset:  45   40   35   30   25   20   15   10   5    0

    SHORT slot:    0    1    2    3    4    5    6    7    8    9
          offset:  18   16   14   12   10   8    6    4    2    0

Both windows are oldest -> newest so each branch's per-step positional
embedding keeps the v2 convention (``step_embed[-T:]``; slot T-1 = the
current turn).

Datasets and the deploy deque walk ONE stack at the UNION of the two
offset sets (oldest -> newest, 18 frames — LONG and SHORT overlap only
at offsets {10, 0}); ``DualRateCrossEntity`` index-selects each
branch's frames out of the union via ``LONG_SLOT_IDX`` /
``SHORT_SLOT_IDX``. The union ends at offset 0, so every existing
``[:, -1]`` current-step read in the v2 forward path stays correct
unchanged.
"""

from __future__ import annotations

from ..transformer_v2.history import HISTORY_OFFSETS as LONG_HISTORY_OFFSETS

SHORT_HISTORY_OFFSETS: tuple[int, ...] = (18, 16, 14, 12, 10, 8, 6, 4, 2, 0)

#: One stack to feed them both: union of LONG ∪ SHORT, oldest first.
UNION_HISTORY_OFFSETS: tuple[int, ...] = tuple(
    sorted(set(LONG_HISTORY_OFFSETS) | set(SHORT_HISTORY_OFFSETS), reverse=True)
)

#: Positions of each branch's frames inside the union stack (branch
#: slot order preserved, oldest -> newest).
LONG_SLOT_IDX: tuple[int, ...] = tuple(
    UNION_HISTORY_OFFSETS.index(off) for off in LONG_HISTORY_OFFSETS
)
SHORT_SLOT_IDX: tuple[int, ...] = tuple(
    UNION_HISTORY_OFFSETS.index(off) for off in SHORT_HISTORY_OFFSETS
)

N_UNION: int = len(UNION_HISTORY_OFFSETS)
N_BRANCH: int = len(LONG_HISTORY_OFFSETS)

#: Deploy deque retention — unchanged from v2 (the long branch still
#: looks back farthest): max offset + 1 = 46 turns.
HISTORY_MAX_LOOKBACK: int = max(UNION_HISTORY_OFFSETS) + 1

assert UNION_HISTORY_OFFSETS[-1] == 0, "union must end at the current turn"
assert len(LONG_HISTORY_OFFSETS) == len(SHORT_HISTORY_OFFSETS) == 10
assert N_UNION == 18, f"expected 18 union frames, got {N_UNION}"
