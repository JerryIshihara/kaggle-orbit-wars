from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import agents as _agents

_REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_ROOT = _REPO_ROOT / "logs" / "replays"


def make_run_id(mode: str, agent_ids: list[str]) -> str:
    """Return a run id like ``play-20260421T013045-sniper-vs-random``."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    combo = "-vs-".join(agent_ids)
    return f"{mode}-{ts}-{combo}"


def compute_scores(env) -> list[list[int]]:
    """Per-step total ship count (planets + fleets) per player."""
    scores: list[list[int]] = []
    for step in env.steps:
        if not step:
            continue
        obs = step[0].observation
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
        n = len(step)
        per_player = [0] * n
        for p in planets:
            owner, ships = p[1], p[5]
            if 0 <= owner < n:
                per_player[owner] += ships
        for f in fleets:
            owner, ships = f[1], f[6]
            if 0 <= owner < n:
                per_player[owner] += ships
        scores.append(per_player)
    return scores


@dataclass
class MatchResult:
    agent_ids: list[str]
    env: Any
    scores: list[list[int]]
    rewards: list
    winner: int


def _validate_and_load(agent_ids: list[str]) -> list[_agents.Agent]:
    if len(agent_ids) not in (2, 4):
        raise ValueError(f"orbit_wars requires 2 or 4 agents, got {len(agent_ids)}")
    return [_agents.Agent(id=a) for a in agent_ids]


def run_match(
    agent_ids: list[str],
    seed: int | None = None,
    debug: bool = False,
) -> MatchResult:
    """Play one match and return env + scores + rewards + winner."""
    from kaggle_environments import make

    players = _validate_and_load(agent_ids)
    config: dict = {"seed": seed} if seed is not None else {}
    env = make("orbit_wars", configuration=config, debug=debug)
    env.run([p.fn for p in players])

    scores = compute_scores(env)
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    ranked = [r if r is not None else float("-inf") for r in rewards]
    winner = max(range(len(ranked)), key=lambda i: ranked[i])

    return MatchResult(
        agent_ids=agent_ids,
        env=env,
        scores=scores,
        rewards=rewards,
        winner=winner,
    )


def save_replay(env, path: Path | str) -> Path:
    """Write ``env.render(mode='html')`` to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(env.render(mode="html"))
    return path


def train_match(*args, **kwargs):
    """Placeholder — self-play training loop (not implemented)."""
    raise NotImplementedError("training mode is not implemented yet")
