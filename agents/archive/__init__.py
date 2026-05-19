"""Archived agents — superseded but kept for inference & ablation.

These packages still register with ``agents.registry`` when imported,
so the dashboard / submission / play paths keep working. New
development happens in the active learned line at
``agents/transformer_v2/``.

Roster:
  * ``transformer_v1`` — previous transformer pipeline (cross-entity +
    pair-score / target-rank heads, action-decoder pretrain). Inference
    in ``app/server.py`` still loads ckpts from this package; the
    transformer-PPO entrypoint in ``run.py`` also still points here.
"""

try:
    from . import transformer_v1  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(f"archive.transformer_v1 unavailable (missing deps): {e}")
