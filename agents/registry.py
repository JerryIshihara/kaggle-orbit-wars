from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

AgentFn = Callable[[dict], list]


@dataclass(frozen=True)
class AgentSpec:
    id: str
    fn: AgentFn
    description: str


_REGISTRY: dict[str, AgentSpec] = {}


def register(id: str, description: str):
    def decorator(fn: AgentFn) -> AgentFn:
        if id in _REGISTRY:
            raise ValueError(f"agent id {id!r} already registered")
        _REGISTRY[id] = AgentSpec(id=id, fn=fn, description=description)
        return fn

    return decorator


def list_agents() -> list[str]:
    return sorted(_REGISTRY)


def list_agent_specs() -> list[AgentSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


class Agent:
    def __init__(self, id: str):
        if id not in _REGISTRY:
            raise KeyError(f"unknown agent id {id!r}. available: {list_agents()}")
        spec = _REGISTRY[id]
        self.id = spec.id
        self.fn: AgentFn = spec.fn
        self.description = spec.description

    def __call__(self, obs):
        return self.fn(obs)

    def __repr__(self) -> str:
        return f"Agent(id={self.id!r})"
