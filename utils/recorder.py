from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def record_match(
    out_dir: Path,
    agent_ids: list[str],
    env: Any,
    scores: list[list[int]],
    rewards: list,
    winner: int,
) -> dict[str, Path]:
    """Write per-match artifacts to out_dir.

    Produces:
      moves.csv   — one row per fleet launch: step, player, from_planet, angle, ships
      scores.csv  — wide format: step, player_0, player_1, ...
      match.json  — match-level metadata (agents, rewards, winner, num_steps)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    moves_path = out_dir / "moves.csv"
    with moves_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "player", "from_planet", "angle", "ships"])
        for t, step in enumerate(env.steps):
            for i, state in enumerate(step):
                action = getattr(state, "action", None) or []
                for m in action:
                    if isinstance(m, (list, tuple)) and len(m) >= 3:
                        w.writerow([t, i, m[0], m[1], m[2]])
    paths["moves"] = moves_path

    scores_path = out_dir / "scores.csv"
    n = len(scores[0]) if scores else 0
    with scores_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"player_{i}" for i in range(n)])
        for t, row in enumerate(scores):
            w.writerow([t] + list(row))
    paths["scores"] = scores_path

    meta_path = out_dir / "match.json"
    meta_path.write_text(
        json.dumps(
            {
                "agents": agent_ids,
                "rewards": rewards,
                "winner": winner,
                "num_steps": len(scores),
            },
            indent=2,
        )
    )
    paths["match"] = meta_path

    return paths
