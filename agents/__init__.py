from .registry import Agent, AgentSpec, list_agent_specs, list_agents, register

from . import agent_physical_v1  # noqa: F401
from . import agent_physical_v2  # noqa: F401
from . import agent_physical_v3  # noqa: F401
from . import agent_random  # noqa: F401
from . import agent_sniper  # noqa: F401

try:
    from . import agent_cnn_v1  # noqa: F401
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
