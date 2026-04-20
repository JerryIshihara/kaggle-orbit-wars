from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import agents

REPLAY_ROOT = Path(__file__).resolve().parent / "replays"


def make_run_id(mode: str, agent_ids: list[str]) -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    combo = "-vs-".join(agent_ids)
    return f"{mode}-{ts}-{combo}"


def cmd_play(args: argparse.Namespace) -> int:
    from kaggle_environments import make

    if len(args.agents) not in (2, 4):
        print(
            f"error: orbit_wars requires 2 or 4 agents, got {len(args.agents)}",
            file=sys.stderr,
        )
        return 2

    try:
        players = [agents.Agent(id=aid) for aid in args.agents]
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    run_id = make_run_id("play", args.agents)
    run_dir = REPLAY_ROOT / "play" / run_id
    save_replays = not args.no_replay
    if save_replays:
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id: {run_id}")
    if save_replays:
        print(f"replays: {run_dir}")
    else:
        print("replays: disabled (--no-replay)")

    config: dict = {}
    if args.seed is not None:
        config["seed"] = args.seed

    pad = max(2, len(str(args.num_games)))
    for g in range(args.num_games):
        env = make("orbit_wars", configuration=config, debug=args.debug)
        env.run([p.fn for p in players])

        final = env.steps[-1]
        rewards = [s.reward if s.reward is not None else float("-inf") for s in final]
        winner = max(range(len(rewards)), key=lambda i: rewards[i])
        print(f"game {g + 1}/{args.num_games}:")
        for i, (p, s) in enumerate(zip(players, final)):
            marker = " <- winner" if i == winner else ""
            print(f"  player {i} ({p.id}): reward={s.reward} status={s.status}{marker}")

        if save_replays:
            game_num = str(g + 1).zfill(pad)
            replay_path = run_dir / f"game_{game_num}.html"
            replay_path.write_text(env.render(mode="html"))
            print(f"  replay: {replay_path}")

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    raise NotImplementedError("train mode is not implemented yet")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Orbit Wars runner")
    p.add_argument("--mode", choices=["play", "train"], required=True)
    p.add_argument(
        "--agents",
        nargs="+",
        help=f"agent IDs (2 or 4). available: {agents.list_agents()}",
    )
    p.add_argument("--num-games", type=int, default=1)
    p.add_argument(
        "--no-replay",
        action="store_true",
        help="skip writing HTML replays (faster for benchmarks)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "play":
        if not args.agents:
            print("error: --agents is required in play mode", file=sys.stderr)
            return 2
        return cmd_play(args)
    if args.mode == "train":
        return cmd_train(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
