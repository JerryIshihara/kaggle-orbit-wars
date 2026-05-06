from __future__ import annotations

import argparse
import sys

import agents
from utils import (
    REPLAY_ROOT,
    make_run_id,
    pack_agent,
    run_match,
    save_replay,
    submit_agent,
    train_match,
)


def cmd_play(args: argparse.Namespace) -> int:
    run_id = make_run_id("play", args.agents)
    run_dir = REPLAY_ROOT / "play" / run_id
    save = not args.no_replay
    if save:
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id: {run_id}")
    print(f"replays: {run_dir}" if save else "replays: disabled (--no-replay)")

    pad = max(2, len(str(args.num_games)))
    for g in range(args.num_games):
        try:
            result = run_match(args.agents, seed=args.seed, debug=args.debug)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except KeyError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        print(f"game {g + 1}/{args.num_games}:")
        for i, (aid, r, status) in enumerate(
            zip(result.agent_ids, result.rewards, [s.status for s in result.env.steps[-1]])
        ):
            marker = " <- winner" if i == result.winner else ""
            print(f"  player {i} ({aid}): reward={r} status={status}{marker}")

        if save:
            path = save_replay(result.env, run_dir / f"game_{str(g + 1).zfill(pad)}.html")
            print(f"  replay: {path}")

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import torch  # noqa: F401

    from agents.cnn_v1 import (
        collect_bc_dataset,
        default_dataset_path,
        evaluate_agent,
        train_bc,
        train_ppo,
    )

    stages = args.stages or ["ppo", "eval"]
    dataset = None

    if "collect" in stages:
        ds_path = default_dataset_path(args.teacher, args.collect_games)
        if ds_path.exists() and not args.force_collect:
            print(f"dataset exists: {ds_path} (use --force-collect to rebuild)")
            dataset = torch.load(ds_path)
        else:
            print(f"collecting BC dataset: {args.collect_games} games of "
                  f"{args.teacher} vs {args.opponent}")
            dataset = collect_bc_dataset(
                num_games=args.collect_games,
                teacher_id=args.teacher,
                opponent_id=args.opponent,
                save_to=ds_path,
            )
    if "bc" in stages:
        if dataset is None:
            ds_path = default_dataset_path(args.teacher, args.collect_games)
            if not ds_path.exists():
                print(f"error: no dataset at {ds_path}; run with --stages collect first",
                      file=sys.stderr)
                return 2
            dataset = torch.load(ds_path)
        print(f"training BC for {args.bc_epochs} epochs ({dataset['channels'].size(0)} samples)")
        train_bc(dataset, epochs=args.bc_epochs, batch_size=args.batch_size)
    if "ppo" in stages:
        print(f"training PPO for {args.ppo_iters} iterations x {args.ppo_episodes} eps")
        train_ppo(
            iterations=args.ppo_iters,
            episodes_per_iter=args.ppo_episodes,
            opponents=args.ppo_opponents,
            resume_from=None,  # uses the saved weights path by default
        )
    if "transformer-ppo" in stages:
        from agents.transformer_v1.ppo import train_ppo as train_transformer_ppo
        print(f"training transformer_v1 PPO: {args.ppo_iters} iters x {args.ppo_episodes} eps")
        ckpt = train_transformer_ppo(
            resume_action_pt=args.transformer_ppo_ckpt,
            iterations=args.ppo_iters,
            episodes_per_iter=args.ppo_episodes,
            seed_start=args.seed or 0,
        )
        print(f"transformer_v1 PPO checkpoint: {ckpt}")

    if "eval" in stages:
        print(f"evaluating cnn_v1 vs {args.eval_opponents} ({args.eval_games} games each)")
        evaluate_agent("cnn_v1", opponents=args.eval_opponents, games_per=args.eval_games)

    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    for aid in args.agents:
        r = pack_agent(aid, bundle=args.bundle)
        print(f"packed {aid}: {r['main']}")
        if r["bundle"]:
            print(f"  bundle: {r['bundle']}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    for aid in args.agents:
        submit_agent(aid, note=args.note, bundle=args.bundle, dry_run=args.dry_run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Orbit Wars runner")
    p.add_argument(
        "--mode", choices=["play", "train", "pack", "submit"], required=True
    )
    p.add_argument(
        "--agents",
        nargs="+",
        help=f"agent IDs. available: {agents.list_agents()}",
    )
    p.add_argument("--num-games", type=int, default=1)
    p.add_argument(
        "--no-replay",
        action="store_true",
        help="skip writing HTML replays (play mode)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--bundle", action="store_true", help="emit tar.gz (pack/submit)")
    p.add_argument("--note", default="", help="submission note (submit mode)")
    p.add_argument("--dry-run", action="store_true", help="submit mode: preview only")
    p.add_argument(
        "--stages",
        nargs="+",
        choices=["collect", "bc", "ppo", "eval", "transformer-ppo"],
        help="training stages to run (default: collect bc eval)",
    )
    p.add_argument("--teacher", default="sniper_v1", help="BC teacher agent")
    p.add_argument("--opponent", default="sniper_v1", help="BC data collection opponent")
    p.add_argument("--collect-games", type=int, default=30)
    p.add_argument("--force-collect", action="store_true")
    p.add_argument("--bc-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--ppo-iters", type=int, default=200)
    p.add_argument("--ppo-episodes", type=int, default=32)
    p.add_argument(
        "--ppo-opponents",
        nargs="*",
        default=[],
        help="external opponents to mix in (default: [] = pure self-play)",
    )
    p.add_argument(
        "--eval-opponents",
        nargs="+",
        default=["random_v1", "sniper_v1", "physical_v2"],
    )
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument(
        "--transformer-ppo-ckpt", default=None,
        help="Transformer BC checkpoint to resume from for transformer-ppo stage",
    )
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
    if args.mode == "pack":
        if not args.agents:
            print("error: --agents is required in pack mode", file=sys.stderr)
            return 2
        return cmd_pack(args)
    if args.mode == "submit":
        if not args.agents:
            print("error: --agents is required in submit mode", file=sys.stderr)
            return 2
        return cmd_submit(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
