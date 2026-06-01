"""History window offsets for transformer_v2.

v1 used a dense 3-step window (offsets [2, 1, 0]: t-2, t-1, t). v2
ships a **10-step uniform-spaced window** at 5-turn spacing:

    slot:        0    1    2    3    4    5    6    7    8    9
    offset:      45   40   35   30   25   20   15   10   5    0
    meaning:    t-45 t-40 t-35 t-30 t-25 t-20 t-15 t-10 t-5   t

A typical Orbit Wars episode runs ~180 turns; this window covers ~25%
of an episode and ~50 turns of lookback — wider than the earlier sparse
9-step window (max lookback t-26) and easier for the per-step positional
embedding to interpret linearly because the spacing is uniform.

Order is **oldest → newest** so the model's per-step positional
embedding indexing (``step_embed[-T:]`` in
``CrossEntityAttention.forward``) maps slot 0 to the oldest frame and
slot T-1 to the current turn — same convention as v1.

This constant is the single source of truth for v2. The dataset uses
it directly in ``__getitem__``; the model's ``n_steps`` is sized to
``len(HISTORY_OFFSETS)``; the inference scorer's deque is sized to
``max(HISTORY_OFFSETS) + 1`` and walks the same offsets.
"""

from __future__ import annotations


HISTORY_OFFSETS: tuple[int, ...] = (45, 40, 35, 30, 25, 20, 15, 10, 5, 0)

# Derived for convenience — keep ``n_history`` semantics (= window
# length) consistent with v1 call sites that take an integer count.
N_HISTORY: int = len(HISTORY_OFFSETS)

# Number of past turns the inference deque must retain (so off=11 still
# resolves to a real frame on a 12+ turn replay).
HISTORY_MAX_LOOKBACK: int = max(HISTORY_OFFSETS) + 1
