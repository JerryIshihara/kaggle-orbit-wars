"""Cross-entity aggregation layers for transformer_v2.

After the per-element encoders (``fleet_encoder``, ``planet_encoder``)
and the per-planet entity encoder (``entity_encoder``), this package
holds the layer that lets every entity token *see every other* —
self-attention across the set of planets / comets so that each token
ends up contextualized by the global game state.

See ``README.md`` for design + label rationale.
"""

from .cross_entity import CrossEntityAttention
from .dual_role_attention import DualRoleAttention
from .joint_role_attention import JointRoleAttention
from .pair_head import PAIR_TYPE_EMBED_DIM, PAIR_TYPE_NUM_CLASSES, PairHead

__all__ = [
    "CrossEntityAttention",
    "DualRoleAttention",
    "JointRoleAttention",
    "PAIR_TYPE_EMBED_DIM",
    "PAIR_TYPE_NUM_CLASSES",
    "PairHead",
]
