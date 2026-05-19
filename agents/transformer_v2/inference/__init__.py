"""Inference-time utilities for transformer_v2.

  * :mod:`target_ranker_scorer` — replay-time per-planet target-score
    extraction from a ``target_rank_best.pt`` ckpt. Drives the
    dashboard's side-by-side target-score replay window.
"""

from .target_ranker_scorer import load_target_ranker_stack, score_replay

__all__ = ["load_target_ranker_stack", "score_replay"]
