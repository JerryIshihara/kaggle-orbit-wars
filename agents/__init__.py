from .registry import Agent, AgentSpec, list_agent_specs, list_agents, register

from . import mcts_v1  # noqa: F401
from . import physical_v1  # noqa: F401
from . import physical_v2  # noqa: F401
from . import physical_v3  # noqa: F401
from . import physical_v4  # noqa: F401
from . import random_v1  # noqa: F401
from . import sniper_v1  # noqa: F401
from . import hybrid_v1  # noqa: F401

try:
    from . import cnn_v1  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(f"cnn_v1 unavailable (missing deps): {e}")

__all__ = [
    "Agent",
    "AgentSpec",
    "list_agents",
    "list_agent_specs",
    "register",
]
