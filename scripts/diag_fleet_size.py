"""Diagnostic: multi-target allocation/fleet-size distribution for a PPO ckpt.

Runs a couple of games under the multi-target sampler and aggregates, per
acting step, the per-cell ship allocation (alloc_counts), # fired cells, source
ships, and held (self) ships — to see whether the N=S multinomial spreads ships
too thin across too many fired targets.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from agents.transformer_v2.ppo.smoke import load_supervised, run_episode
from agents.transformer_v2.ppo.actor_critic import PPOActorCritic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--planet-run-dir", type=Path)
    p.add_argument("--fleet-run-dir", type=Path)
    p.add_argument("--comet-run-dir", type=Path)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--select-logit-bias", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    em, fe, pe, ce, cfg = load_supervised(
        args.ckpt, args.device, planet_run_dir=args.planet_run_dir,
        fleet_run_dir=args.fleet_run_dir, comet_run_dir=args.comet_run_dir)
    policy = PPOActorCritic(em, sigma=0.35, critic_model=None,
                            allow_debug_glob_critic=False, reward_decomp=True,
                            win_weight=0.7).to(args.device)
    st = torch.load(args.policy, map_location=args.device, weights_only=False)
    policy.load_state_dict(st.get("policy", st), strict=False)
    policy.eval()
    print(f"[diag] loaded {args.policy.name} (iter={st.get('iter')})", flush=True)

    fleet, fired_per_step, held, src_ships = [], [], [], []
    for seed in range(70001, 70001 + args.seeds):
        for seat in range(2):
            ep = run_episode(
                policy=policy, planet_enc=pe, fleet_enc=fe, comet_enc=ce,
                seed=seed, learner_seat=seat, opponent_id="physical_v4",
                device=args.device, max_planets=64, max_fleets=512, sigma=0.35,
                noop_logit_bias=0.0, select_logit_bias=args.select_logit_bias,
                history_window=10, num_players=2)
            for s in ep.steps:
                a = getattr(s, "action", None)
                ac = getattr(a, "alloc_counts", None)
                if ac is None:
                    continue
                ac = ac.detach().cpu()
                vals = ac[ac > 0]
                if vals.numel():
                    fleet += vals.tolist()
                    fired_per_step.append(int((ac > 0).sum()))
                    src_ships.append(int(ac.sum()) + int(getattr(a, "self_counts").sum()))
                sc = getattr(a, "self_counts", None)
                if sc is not None:
                    held += sc[sc > 0].detach().cpu().tolist()
            print(f"[diag] seed={seed} seat={seat} steps={len(ep.steps)}", flush=True)

    f = np.array(fleet) if fleet else np.array([0])
    fp = np.array(fired_per_step) if fired_per_step else np.array([0])
    print("\n=== multi-target allocation (attempted, pre-plan_launch) ===")
    print(f"  launches (cells fired w/ ships): {len(fleet)}")
    print(f"  fleet size  mean={f.mean():.2f} median={np.median(f):.0f} "
          f"p10={np.percentile(f,10):.0f} p90={np.percentile(f,90):.0f} max={f.max()}")
    print(f"  fleet size  <2 ships: {100.0*(f<2).mean():.0f}%   <5 ships: {100.0*(f<5).mean():.0f}%")
    print(f"  fired cells/acting-step  mean={fp.mean():.1f} median={np.median(fp):.0f} max={fp.max()}")
    if src_ships:
        print(f"  source ships/acting-step mean={np.mean(src_ships):.1f}")
    if held:
        print(f"  held(self)/acting-step   mean={np.mean(held):.1f}")


if __name__ == "__main__":
    main()
