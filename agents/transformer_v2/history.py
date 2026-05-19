"""History window offsets for transformer_v2.

v1 used a dense 3-step window (offsets [2, 1, 0]: t-2, t-1, t). v2
widens this to a **9-step sparse window**:

    slot:        0    1    2    3    4    5    6    7    8
    offset:      26   21   16   11   8    5    2    1    0
    meaning:    t-26 t-21 t-16 t-11 t-8  t-5  t-2  t-1   t

The dense recent triplet (t, t-1, t-2) captures short-horizon dynamics
(acceleration, direction). The medium anchors at t-5, t-8, t-11 carry
fleet-launch and territory-flip context before arrival. The long
anchors at t-16, t-21, t-26 give the model a strategic-scale view —
roughly one full inbound-fleet cycle for the typical map — so the
ranker can condition on momentum across the whole engagement window.

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


HISTORY_OFFSETS: tuple[int, ...] = (26, 21, 16, 11, 8, 5, 2, 1, 0)

# Derived for convenience — keep ``n_history`` semantics (= window
# length) consistent with v1 call sites that take an integer count.
N_HISTORY: int = len(HISTORY_OFFSETS)

# Number of past turns the inference deque must retain (so off=11 still
# resolves to a real frame on a 12+ turn replay).
HISTORY_MAX_LOOKBACK: int = max(HISTORY_OFFSETS) + 1
