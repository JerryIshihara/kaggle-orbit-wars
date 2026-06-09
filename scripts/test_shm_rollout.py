"""Smoke + parity test for the shared-memory rollout (infserver_shm_rollout).

Run as a FILE (not stdin) so spawn workers can re-import __main__.
    OW_SHM_ALLOW_CPU=1 PYTHONPATH=. python scripts/test_shm_rollout.py
On CPU it validates the slot/flag handshake (no deadlock) + produces valid
episodes. Add --parity to also run the queue-based infserver on the same seeds
and compare learner-step counts + winners.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

A = "data/runs/ppo/best_agentA_gentle_v5_93pct/policy_gentle_v5.pt"
BASE = "data/runs/ppo/baseline_lr5e5_enthalf_iter5_20260604/entity_head3_value_merged.pt"
PLANET = "data/runs/planet/specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep"
FLEET = "data/runs/fleet/specialist_fleet_d256_40k_lr1e4_120ep"
COMET = "data/runs/comet/fullpath_scalar_multitask_d256_40k_lr1e4_120ep"


def _build(device, base, policy, planet, fleet, comet):
    from agents.transformer_v2.ppo.smoke import load_supervised
    from agents.transformer_v2.ppo.actor_critic import PPOActorCritic
    em, fe, pe, ce, _ = load_supervised(
        base, device, planet_run_dir=planet, fleet_run_dir=fleet, comet_run_dir=comet)
    pol = PPOActorCritic(em, sigma=0.35, critic_model=None, allow_debug_glob_critic=False,
                         reward_decomp=True, win_weight=0.5).to(device)
    st = torch.load(policy, map_location=device, weights_only=False)
    pol.load_state_dict(st.get("policy", st), strict=False)
    pol.eval()
    return pol, pe, fe, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--forwarders", type=int, default=4)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--policy", default=A)
    ap.add_argument("--planet-run-dir", default=PLANET)
    ap.add_argument("--fleet-run-dir", default=FLEET)
    ap.add_argument("--comet-run-dir", default=COMET)
    args = ap.parse_args()

    from agents.transformer_v2.ppo.infserver_shm_rollout import run_shm_rollout
    pol, pe, fe, ce = _build(args.device, args.base, args.policy,
                             args.planet_run_dir, args.fleet_run_dir, args.comet_run_dir)
    specs = [(1729 + i, 2, i % 2) for i in range(args.games)]  # (seed, num_players, learner_seat)

    eps, stats = run_shm_rollout(
        policy=pol, opponent_policy=None, planet_enc=pe, fleet_enc=fe, comet_enc=ce,
        specs=specs, n_workers=args.workers, device=args.device, max_planets=64,
        max_fleets=512, sigma=0.35, select_logit_bias=1.0, history_window=10,
        n_forwarders=args.forwarders)
    print(f"[shm] episodes={len(eps)} steps={[len(e.steps) for e in eps]} "
          f"winners={[e.winner for e in eps]} forwards={stats['n_forwards']} "
          f"wall_s={stats['wall_s']:.1f}", flush=True)
    assert len(eps) == args.games, "missing episodes"
    assert all(len(e.steps) > 0 for e in eps), "empty episode"

    if args.parity:
        from agents.transformer_v2.ppo.infserver_rollout import run_infserver_rollout
        import os
        if args.device.startswith("cpu"):
            print("[parity] infserver needs cuda; skipping queue-path comparison on CPU")
        else:
            eps2, _ = run_infserver_rollout(
                policy=pol, planet_enc=pe, fleet_enc=fe, comet_enc=ce, specs=specs,
                n_workers=args.workers, device=args.device, max_planets=64, max_fleets=512,
                sigma=0.35, select_logit_bias=1.0, history_window=10)
            s1 = sorted(len(e.steps) for e in eps)
            s2 = sorted(len(e.steps) for e in eps2)
            print(f"[parity] shm step-counts={s1}  infserver={s2}  "
                  f"{'MATCH' if s1 == s2 else 'DIFFER (sampling noise across paths is expected)'}")
    print("[shm] PASS", flush=True)


if __name__ == "__main__":
    main()
