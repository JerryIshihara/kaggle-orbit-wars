"""Play transformer_v2 vs physical_v4 for N games; save dashboard-readable
replays under ``data/runs/<run_id>/replays/`` with the train-time naming
convention so the dashboard's training panel picks them up.

Filename convention (matches ``app/server.py:_REPLAY_RE``):

    <outcome>_iter<NNN>_ep<MM>_<color>.html

where outcome ∈ {win, loss, draw} from transformer_v2's perspective,
iter is 000 (no train iteration in play mode), ep is the game index,
and color is the seat we played from (slot 0 = blue, 1 = orange).

Run:
    python scripts/play_transformer_v2_vs_physical_v4.py --num-games 5
    python scripts/play_transformer_v2_vs_physical_v4.py --num-games 5 --seeds 1729,42,1,7,100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import agents  # noqa: E402  — registers all agent ids
from utils.runner import run_match, save_replay  # noqa: E402
from utils.eval_seeds import SEEDS  # noqa: E402


def _outcome_for_seat(result, seat: int) -> str:
    """win / loss / draw from ``seat``'s perspective."""
    if result.winner == seat:
        return "win"
    # Tie if multiple players tie on max reward
    rewards = result.rewards
    if rewards and rewards[seat] is not None:
        max_r = max(r for r in rewards if r is not None)
        ties = sum(1 for r in rewards if r is not None and r == max_r)
        if rewards[seat] == max_r and ties > 1:
            return "draw"
    return "loss"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-games", type=int, default=5,
                    help="Number of games to play.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds. Default: first --num-games "
                         "entries from utils.eval_seeds.SEEDS (the stratified "
                         "panel).")
    ap.add_argument("--learner", default="transformer_v2",
                    help="Agent id to play as the learner.")
    ap.add_argument("--opponent", default="physical_v4",
                    help="Opponent agent id.")
    ap.add_argument("--seat", type=int, default=0, choices=(0, 1),
                    help="Which seat the learner plays from.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output run directory. Defaults to "
                         "data/runs/play_<learner>_vs_<opponent>_<TS>.")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(SEEDS[:args.num_games])
    if len(seeds) < args.num_games:
        seeds = (seeds * ((args.num_games // len(seeds)) + 1))[:args.num_games]
    seeds = seeds[:args.num_games]

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir or (
        REPO / "data" / "runs"
        / f"play_{args.learner}_vs_{args.opponent}_{ts}"
    )
    replay_dir = run_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    color = "blue" if args.seat == 0 else "orange"
    agent_slots = [args.opponent, args.opponent]
    agent_slots[args.seat] = args.learner

    print(f"learner: {args.learner} (seat {args.seat}, {color})")
    print(f"opponent: {args.opponent}")
    print(f"run dir:  {run_dir}")
    print(f"seeds:    {seeds}")
    print()

    wins = losses = draws = 0
    results = []
    t0 = time.time()
    for g, seed in enumerate(seeds, 1):
        t_game = time.time()
        result = run_match(agent_slots, seed=seed)
        outcome = _outcome_for_seat(result, args.seat)
        if outcome == "win":
            wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            losses += 1
        ep_tag = f"{g:02d}"
        name = f"{outcome}_iter000_ep{ep_tag}_{color}.html"
        path = save_replay(result.env, replay_dir / name)
        elapsed = time.time() - t_game
        learner_reward = result.rewards[args.seat]
        opp_reward = result.rewards[1 - args.seat]
        scoreboard = "  ".join(
            f"{aid}={r}" for aid, r in zip(result.agent_ids, result.rewards)
        )
        print(
            f"  game {g}/{args.num_games}  seed={seed}  {outcome:<5s}  "
            f"({scoreboard})  → {path.name}  [{elapsed:.1f}s]"
        )
        results.append(dict(
            game=g, seed=seed, outcome=outcome,
            learner_reward=learner_reward, opp_reward=opp_reward,
            replay=str(path),
        ))

    total = time.time() - t0
    print()
    print(f"summary: wins={wins}  draws={draws}  losses={losses}  "
          f"({wins}/{args.num_games}); total {total:.1f}s")
    print(f"replays under: {replay_dir}")


if __name__ == "__main__":
    main()
