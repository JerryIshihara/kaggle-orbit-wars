from .registry import Agent, list_agents, register

from . import agent_random  # noqa: F401
from . import agent_sniper  # noqa: F401

__all__ = ["Agent", "list_agents", "register"]
