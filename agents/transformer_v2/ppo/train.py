"""Multi-iteration PPO training loop (single machine, Phase 0).

Wraps the smoke runner's rollout + GAE + update in an outer loop, saves a
policy checkpoint per iteration, appends a JSONL row per iteration to a
train log.

This is the **Phase 0 single-machine path**. Distributed Phase 1+ (file-
mediated grad averaging across A and B), the eval gate, the self-play pool,
and the archive script are NOT in this file yet — they are spec'd in
``docs/PPO_TWO_CPU_PROTOCOL.md`` and ship in follow-up PRs. This file is
the smallest thing that actually trains: rollout N episodes per iter, do
one PPO update, save, log, repeat.

Usage::

    python -m agents.transformer_v2.ppo.train \\
        --ckpt data/runs/entity/<run>/entity_encoder_best.pt \\
        --run-id ppo_smoke_<ts> \\
        --iters 10 --episodes-per-iter 10 \\
        --opponent random_v1 \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from .actor_critic import PPOActorCritic
from .learner import PPOConfig, ppo_update_local
from .smoke import (
    _PPOWithL0,
    episodes_to_ppo,
    load_supervised,
    run_episode,
)


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path,
                         help="Supervised PairHead .pt to bootstrap from.")
    parser.add_argument("--run-id", required=True,
                         help="Run id; outputs go to data/runs/ppo/<run_id>/.")
    parser.add_argument("--iters", type=int, default=10,
                         help="Number of PPO iterations to run.")
    parser.add_argument("--episodes-per-iter", type=int, default=10)
    parser.add_argument("--opponent", default="random_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-heads", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.10)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--noop-logit-bias", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=100_000)
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=1,
                         help="Save a checkpoint every N iterations.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = repo_root / "data" / "runs" / "ppo" / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    config_path = run_dir / "config.json"

    # Persist the run config once at start.
    config_path.write_text(json.dumps({
        "started_at": _iso_now(),
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }, indent=2) + "\n")

    print(f"[train] run_dir={run_dir}", flush=True)
    print(f"[train] device={args.device} ckpt={args.ckpt}", flush=True)
    t0 = time.time()

    entity_model, fleet_enc, planet_enc, comet_enc, cfg = load_supervised(
        args.ckpt, args.device,
    )
    print(f"[train] loaded supervised ckpt in {time.time()-t0:.1f}s "
          f"(d_model={cfg.get('d_model')}, n_steps={cfg.get('n_steps')}, "
          f"conditioner_n_layers={cfg.get('conditioner_n_layers', 1)}, "
          f"head_n_layers={cfg.get('head_n_layers', 1)})", flush=True)

    policy = PPOActorCritic(entity_model, sigma=args.sigma).to(args.device)
    breakdown = policy.freeze_for_phase(0)
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[train] freeze_for_phase(0) → trainable={n_trainable:,} "
          f"by group: {breakdown}", flush=True)

    train_policy = _PPOWithL0(policy, planet_enc, fleet_enc, comet_enc).to(args.device)

    cfg_ppo = PPOConfig(
        clip=args.clip,
        target_kl=args.target_kl,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr_heads=args.lr_heads,
        lr_trunk=None,
        value_coef=args.value_coef,
        ent_coef=args.ent_coef,
        bc_coef=0.0,                      # no BC anchor in this smoke train
    )

    # Save v0 (the bootstrap state).
    torch.save({"policy": policy.state_dict(),
                "value_hidden": policy.value_head[0].out_features,
                "sigma": float(policy.sigma.item())},
               ckpt_dir / "policy_v0.pt")

    seed_cursor = args.seed_base
    for K in range(args.iters):
        t_iter = time.time()
        print(f"\n[train] === iter {K:03d} (vs {args.opponent}) ===", flush=True)

        # ---- Rollout ----
        episodes = []
        n_wins = 0
        total_steps = 0
        total_invalid = 0
        total_emitted = 0
        total_noop = 0
        t_roll = time.time()
        for i in range(args.episodes_per_iter):
            seed = seed_cursor
            seed_cursor += 1
            learner_seat = i % 2
            try:
                ep = run_episode(
                    policy=policy, planet_enc=planet_enc, fleet_enc=fleet_enc,
                    comet_enc=comet_enc, seed=seed, learner_seat=learner_seat,
                    opponent_id=args.opponent, device=args.device,
                    max_planets=args.max_planets, max_fleets=args.max_fleets,
                    sigma=args.sigma, noop_logit_bias=args.noop_logit_bias,
                )
            except Exception as e:
                print(f"[train] iter {K} ep {i} FAILED: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return 1
            episodes.append(ep)
            n_wins += int(ep.winner == learner_seat)
            total_steps += len(ep.steps)
            total_invalid += sum(s.invalid_launch for s in ep.steps)
            total_emitted += sum(s.emitted_launch for s in ep.steps)
            total_noop += sum(1 for s in ep.steps if s.action.source_id < 0)
        rollout_sec = time.time() - t_roll
        print(f"[train] rollout: {args.episodes_per_iter} eps, "
              f"{n_wins} wins, {total_steps} learner steps, "
              f"noop={total_noop} emit={total_emitted} inval={total_invalid} "
              f"({rollout_sec:.1f}s)", flush=True)

        # ---- GAE + minibatches ----
        t_gae = time.time()
        ep_objs, mbs = episodes_to_ppo(
            episodes, minibatch_size=args.minibatch_size, device=args.device,
        )
        gae_sec = time.time() - t_gae
        if not mbs:
            print(f"[train] iter {K}: no steps to train on — skipping update",
                  flush=True)
            continue

        # ---- Update ----
        t_upd = time.time()
        metrics = ppo_update_local(
            train_policy, ep_objs, mbs,
            bc_minibatch_source=lambda _n: None,
            cfg=cfg_ppo,
        )
        update_sec = time.time() - t_upd

        # Aggregate epoch metrics (last epoch wins for KL / clip_frac).
        last_epoch = metrics["epoch_metrics"][-1] if metrics["epoch_metrics"] else {}
        first_epoch = metrics["epoch_metrics"][0] if metrics["epoch_metrics"] else {}
        early = any(em.get("early_stopped") for em in metrics["epoch_metrics"])

        # Final reward (for diagnostics).
        mean_terminal_reward = (
            sum((ep.steps[-1].reward - sum(s.reward for s in ep.steps[:-1])
                  if len(ep.steps) > 1 else ep.steps[-1].reward)
                 for ep in episodes if ep.steps)
            / max(1, sum(1 for ep in episodes if ep.steps))
        )

        log_row = {
            "iter": K,
            "policy_version": K + 1,
            "timestamp": _iso_now(),
            "n_episodes": args.episodes_per_iter,
            "n_wins_vs_random": n_wins,
            "winrate": n_wins / args.episodes_per_iter,
            "n_learner_steps": total_steps,
            "n_noop_steps": total_noop,
            "n_emitted_launches": total_emitted,
            "n_invalid_launches": total_invalid,
            "invalid_rate_per_step": total_invalid / max(1, total_steps),
            "mean_terminal_reward": mean_terminal_reward,
            "n_minibatches": len(mbs),
            "epochs_ran": len(metrics["epoch_metrics"]),
            "early_stopped": early,
            "kl_first_epoch": first_epoch.get("avg_kl"),
            "kl_last_epoch": last_epoch.get("avg_kl"),
            "policy_loss_last": last_epoch.get("policy_loss"),
            "value_loss_last": last_epoch.get("value_loss"),
            "entropy_last": last_epoch.get("entropy"),
            "clip_frac_last": last_epoch.get("clip_frac"),
            "rollout_sec": rollout_sec,
            "gae_sec": gae_sec,
            "update_sec": update_sec,
            "iter_sec": time.time() - t_iter,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(log_row) + "\n")

        print(f"[train] iter {K:03d} done in {log_row['iter_sec']:.1f}s | "
              f"winrate={log_row['winrate']:.2f} "
              f"kl={log_row['kl_last_epoch']:.4f} "
              f"value_loss={log_row['value_loss_last']:.3f} "
              f"entropy={log_row['entropy_last']:.2f} "
              f"clip_frac={log_row['clip_frac_last']:.3f}"
              f"{' EARLY' if early else ''}", flush=True)

        # ---- Save ckpt ----
        if (K + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"policy_v{K+1}.pt"
            torch.save({
                "policy": policy.state_dict(),
                "value_hidden": policy.value_head[0].out_features,
                "sigma": float(policy.sigma.item()),
                "iter": K,
                "metrics": log_row,
            }, ckpt_path)

    total_wall = time.time() - t0
    print(f"\n[train] DONE — {args.iters} iters in {total_wall:.1f}s "
          f"({total_wall/max(1,args.iters):.1f}s/iter avg)", flush=True)
    print(f"[train] log: {log_path}", flush=True)
    print(f"[train] checkpoints: {ckpt_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
