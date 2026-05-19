"""Cross-entity aggregation layers for transformer_v1.

After the per-element encoders (``fleet_encoder``, ``planet_encoder``)
and the per-planet entity encoder (``entity_encoder``), this package
holds the layer that lets every entity token *see every other* —
self-attention across the set of planets / comets so that each token
ends up contextualized by the global game state.

See ``README.md`` for design + label rationale.
"""

from .cross_entity import CrossEntityAttention

__all__ = ["CrossEntityAttention"]
