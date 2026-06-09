"""Local eval: a PPO policy_vK.pt vs a heuristic opponent (default physical_v4).

Builds the actor architecture from the supervised base (--ckpt) + L0 encoders,
loads the PPO checkpoint's 'policy' state on top, then plays N seeds x both
seats with sampled actions and reports winrate. Env runs on CPU (Mac OK).

    python scripts/eval_ppo_vs_physical.py \
        --policy data/runs/ppo/.../policy_v5.pt \
        --ckpt   data/runs/.../joint_best.pt \
        --planet-run-dir data/runs/planet/specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep \
        --fleet-run-dir  data/runs/fleet/specialist_fleet_d256_40k_lr1e4_120ep \
        --comet-run-dir  data/runs/comet/fullpath_scalar_multitask_d256_40k_lr1e4_120ep \
        --num-seeds 6 --history-window 10 --opponent physical_v4
"""
import argparse
import time
from pathlib import Path

import torch

from agents.transformer_v2.ppo.smoke import load_supervised, run_episode
from agents.transformer_v2.ppo.actor_critic import PPOActorCritic
from agents.transformer_v2.ppo.sampler import Action


def greedy_single_target(pair_logits, frac_loc, sigma, *, pair_mask, source_mask,
                         noop_logit_bias=0.0):
    """Sampling-OFF version of sample_single_target: argmax target per owned
    source + sigmoid(loc) frac (the policy's mode). Same signature so it can be
    monkeypatched into run_episode's learner closure for a deterministic eval."""
    p = pair_logits.shape[0]
    device = pair_logits.device
    neg_inf = torch.full_like(pair_logits, float("-inf"))
    tgt_idx = torch.arange(p, device=device).clone()
    frac_raw = torch.zeros(p, dtype=torch.float32, device=device)
    n_launch = 0
    for s in source_mask.nonzero(as_tuple=False).flatten().tolist():
        row_valid = pair_mask[s].clone()
        row_valid[s] = True
        row_logits = torch.where(row_valid, pair_logits[s], neg_inf[s])
        ti = int(row_logits.argmax().item())
        tgt_idx[s] = ti
        if ti != s:
            n_launch += 1
            frac_raw[s] = torch.sigmoid(frac_loc[s, ti]).clamp(1e-4, 1 - 1e-4)
    z = torch.zeros((), device=device)
    nsrc = int(source_mask.sum().item())
    return Action(
        tgt_idx=tgt_idx, frac_raw=frac_raw, logprob=z, logprob_pair=z,
        logprob_frac=z, n_launch=n_launch,
        diagnostics={"n_valid_sources": nsrc, "n_launch": n_launch,
                     "n_hold": nsrc - n_launch},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, type=Path, help="PPO policy_vK.pt")
    p.add_argument("--ckpt", required=True, type=Path, help="supervised base (architecture)")
    p.add_argument("--planet-run-dir", type=Path, default=None)
    p.add_argument("--fleet-run-dir", type=Path, default=None)
    p.add_argument("--comet-run-dir", type=Path, default=None)
    p.add_argument("--opponent", default="physical_v4")
    p.add_argument("--num-seeds", type=int, default=6)
    p.add_argument("--seed-base", type=int, default=70001)
    p.add_argument("--num-players", type=int, default=2)
    p.add_argument("--history-window", type=int, default=10)
    p.add_argument("--max-planets", type=int, default=64)
    p.add_argument("--max-fleets", type=int, default=512)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--select-logit-bias", type=float, default=0.0,
                   help="Bernoulli selection logit bias — MUST match training")
    p.add_argument("--greedy", action="store_true",
                   help="turn sampling OFF: argmax target + sigmoid(loc) frac (policy mode)")
    p.add_argument("--reward-decomp", action="store_true", default=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.greedy:
        import agents.transformer_v2.ppo.smoke as _smoke_mod
        _smoke_mod.sample_single_target = greedy_single_target
        print("[eval] GREEDY mode: sampling OFF (argmax target + mean frac)", flush=True)

    t0 = time.time()
    entity_model, fleet_enc, planet_enc, comet_enc, cfg = load_supervised(
        args.ckpt, args.device,
        planet_run_dir=args.planet_run_dir,
        fleet_run_dir=args.fleet_run_dir,
        comet_run_dir=args.comet_run_dir,
    )
    print(f"[eval] base loaded in {time.time()-t0:.1f}s "
          f"(d_model={cfg.get('d_model')} n_steps={cfg.get('n_steps')})", flush=True)

    policy = PPOActorCritic(
        entity_model, sigma=args.sigma, critic_model=None,
        allow_debug_glob_critic=False, reward_decomp=args.reward_decomp, win_weight=0.7,
    ).to(args.device)
    st = torch.load(args.policy, map_location=args.device, weights_only=False)
    sd = st.get("policy", st)
    miss, unexp = policy.load_state_dict(sd, strict=False)
    policy.eval()
    print(f"[eval] loaded {args.policy.name} (iter={st.get('iter')}) "
          f"tensors={len(sd)} missing={len(miss)} unexpected={len(unexp)}", flush=True)

    seeds = [args.seed_base + i for i in range(args.num_seeds)]
    by_seat = {0: [0, 0], 1: [0, 0]}  # seat -> [wins, games]
    rows = []
    for seed in seeds:
        for seat in range(min(2, args.num_players)):
            t1 = time.time()
            ep = run_episode(
                policy=policy, planet_enc=planet_enc, fleet_enc=fleet_enc,
                comet_enc=comet_enc, seed=seed, learner_seat=seat,
                opponent_id=args.opponent, device=args.device,
                max_planets=args.max_planets, max_fleets=args.max_fleets,
                sigma=args.sigma, noop_logit_bias=0.0,
                select_logit_bias=args.select_logit_bias,
                history_window=args.history_window, num_players=args.num_players,
            )
            win = int(ep.winner == ep.learner_seat)
            by_seat[seat][0] += win
            by_seat[seat][1] += 1
            rows.append((seed, seat, win, ep.winner, len(ep.steps), time.time() - t1))
            print(f"[eval] seed={seed} seat={seat} -> {'WIN ' if win else 'loss'} "
                  f"(winner={ep.winner}, {len(ep.steps)} steps, {time.time()-t1:.1f}s)",
                  flush=True)

    tot_w = sum(by_seat[s][0] for s in by_seat)
    tot_g = sum(by_seat[s][1] for s in by_seat)
    mode = "greedy" if args.greedy else "sampled"
    print("\n=== RESULT vs %s (%dP, %s) ===" % (args.opponent, args.num_players, mode))
    for s in (0, 1):
        w, g = by_seat[s]
        if g:
            print("  seat %d: %d/%d = %.1f%%" % (s, w, g, 100.0 * w / g))
    print("  OVERALL: %d/%d = %.1f%%" % (tot_w, tot_g, 100.0 * tot_w / max(1, tot_g)))


if __name__ == "__main__":
    main()
