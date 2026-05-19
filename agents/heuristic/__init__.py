"""Heuristic (rule-based / search-based) agents.

Grouped here to keep ``agents/`` top-level focused on the active
learned-model lines. Each subpackage registers itself with
``agents.registry`` at import time, so the registry behavior is
unchanged — only the import path moved (e.g.,
``agents.physical_v4`` → ``agents.heuristic.physical_v4``).

Roster:
  * ``random_v1`` — uniform random launches (sanity baseline).
  * ``physical_v{1..4}`` — greedy-expand ladder; v4 is the strongest.
  * ``physical_{static, orbit, comet}_v1`` — single-class ablations.
  * ``sniper_v{1, 2}`` — opportunistic snipe + defend; v2 adds
    motion-aware launch validation.
  * ``mcts_v1`` — Monte Carlo tree search with ``physical_v4`` rollouts.
  * ``hybrid_v1`` — rule prior + small learned re-weighter.
"""

from __future__ import annotations

import importlib
import warnings


_SUBMODULES: tuple[str, ...] = (
    "mcts_v1",
    "physical_v1",
    "physical_v2",
    "physical_v3",
    "physical_v4",
    "physical_static_v1",
    "physical_orbit_v1",
    "physical_comet_v1",
    "random_v1",
    "sniper_v1",
    "sniper_v2",
    "hybrid_v1",
)


for _name in _SUBMODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except ImportError as e:
        # Some heuristic/search agents depend on the Kaggle environment
        # package, which is not needed for transformer_v2 pretraining.
        # Keep package import usable so `python -m
        # agents.transformer_v2.pretrain...` does not fail before the
        # learned stack is even imported.
        warnings.warn(f"heuristic.{_name} unavailable (missing deps): {e}")
