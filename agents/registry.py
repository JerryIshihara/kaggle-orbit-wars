from __future__ import annotations

from typing import Callable

AgentFn = Callable[[dict], list]

_REGISTRY: dict[str, AgentFn] = {}


def register(id: str):
    def decorator(fn: AgentFn) -> AgentFn:
        if id in _REGISTRY:
            raise ValueError(f"agent id {id!r} already registered")
        _REGISTRY[id] = fn
        return fn

    return decorator


def list_agents() -> list[str]:
    return sorted(_REGISTRY)


class Agent:
    def __init__(self, id: str):
        if id not in _REGISTRY:
            raise KeyError(f"unknown agent id {id!r}. available: {list_agents()}")
        self.id = id
        self.fn: AgentFn = _REGISTRY[id]

    def __call__(self, obs):
        return self.fn(obs)

    def __repr__(self) -> str:
        return f"Agent(id={self.id!r})"
