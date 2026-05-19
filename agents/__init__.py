from .registry import Agent, AgentSpec, list_agent_specs, list_agents, register

from . import heuristic  # noqa: F401  — registers random/physical/sniper/mcts/hybrid

try:
    from . import transformer_v2  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(f"transformer_v2 unavailable (missing deps): {e}")

try:
    from .archive import transformer_v1  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(f"archive.transformer_v1 unavailable (missing deps): {e}")

__all__ = [
    "Agent",
    "AgentSpec",
    "list_agents",
    "list_agent_specs",
    "register",
]
