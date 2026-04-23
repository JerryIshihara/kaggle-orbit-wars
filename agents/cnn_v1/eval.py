"""Tournament evaluator — play N games against each opponent, report win rate."""

from __future__ import annotations

import sys
from typing import Iterable

from utils import run_match

DEFAULT_OPPONENTS = ["random_v1", "sniper_v1", "physical_v2"]


def evaluate_agent(
    agent_id: str,
    opponents: Iterable[str] = DEFAULT_OPPONENTS,
    games_per: int = 10,
    seed_start: int = 0,
    verbose: bool = True,
) -> dict[str, dict[str, int]]:
    """Play `games_per` games vs each opponent, alternating seats.

    Returns: ``{opponent: {"wins": w, "losses": l, "draws": d, "win_rate": p}}``.
    """
    results: dict[str, dict[str, int]] = {}
    for opp in opponents:
        wins = losses = draws = 0
        for g in range(games_per):
            agent_slot = g % 2  # alternate who is P0
            ids = [agent_id, opp] if agent_slot == 0 else [opp, agent_id]
            r = run_match(ids, seed=seed_start + g)
            if r.winner == agent_slot:
                wins += 1
            elif r.winner == 1 - agent_slot:
                losses += 1
            else:
                draws += 1
        total = max(1, wins + losses + draws)
        rec = {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / total}
        results[opp] = rec
        if verbose:
            print(
                f"  vs {opp:<14} {wins}-{losses}-{draws}  "
                f"({100 * rec['win_rate']:.0f}%)",
                file=sys.stderr,
            )
    return results
