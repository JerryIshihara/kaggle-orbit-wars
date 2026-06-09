"""Deploy-faithful eval: a PPO policy_vK.pt played through the *production*
runner (``inference_mode='threshold'``, ``logit_threshold=2.0``) vs physical_v4.

This is the number that actually matters for the competition. PPO trains the
SAMPLED multi-target distribution (Bernoulli-select + Multinomial alloc), but
the agent DEPLOYS the deterministic threshold rule (``pair_logits > 2.0`` cells
fire, sized by ``sigmoid(pair_frac)·ships``). This script measures the deployed
policy, so it tells us whether PPO preserved / improved / wrecked the heads the
runner actually reads — independent of the sampled training objective.

Mechanism: the PPO ckpt stores the EntityPretrainModel under an
``entity_model.`` key prefix (PPOActorCritic wraps it). We strip that prefix to
recover a runner-format ``{"model": sd, "config": cfg}`` ckpt (config taken from
the head=3 merged base), then load it through ``TransformerAgent.load`` with the
production threshold rule and play via ``utils.runner.run_match``.

    python scripts/eval_ppo_threshold_vs_physical.py \
        --policy /tmp/mt_eval/policy_v20.pt \
        --base   data/runs/ppo/baseline_lr5e5_enthalf_iter5_20260604/entity_head3_value_merged.pt \
        --num-seeds 6 --logit-threshold 2.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import agents  # noqa: E402  — registers all built-in agent ids (physical_v4, ...)
from agents import registry as _registry  # noqa: E402
from agents.transformer_v2.runner import TransformerAgent  # noqa: E402
from utils.runner import run_match  # noqa: E402


def build_runner_ckpt(policy_path: Path, base_path: Path, out_path: Path) -> dict:
    """Strip the ``entity_model.`` prefix off the PPO policy state-dict to
    recover the EntityPretrainModel weights, pair them with the base config,
    and write a runner-loadable ``{"model": sd, "config": cfg}`` ckpt."""
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    cfg = dict(base.get("config") or {})
    ppo = torch.load(policy_path, map_location="cpu", weights_only=False)
    psd = ppo.get("policy", ppo)
    prefix = "entity_model."
    model_sd = {
        k[len(prefix):]: v for k, v in psd.items() if k.startswith(prefix)
    }
    if not model_sd:
        raise SystemExit(f"no entity_model.* keys in {policy_path}")
    # Backfill any base-model keys the PPO ckpt did not carry (e.g. value heads
    # the runner ignores anyway) so load_state_dict(strict=False) stays clean.
    base_msd = base["model"]
    n_overlay = sum(1 for k in model_sd if k in base_msd)
    merged = dict(base_msd)
    merged.update(model_sd)
    torch.save({"model": merged, "config": cfg, "iter": ppo.get("iter")}, out_path)
    print(f"[build] {policy_path.name} iter={ppo.get('iter')}: "
          f"{len(model_sd)} entity_model tensors ({n_overlay} overlay the base) "
          f"-> {out_path}", flush=True)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, type=Path, help="PPO policy_vK.pt")
    ap.add_argument("--base", required=True, type=Path,
                    help="head=3 merged base ckpt (architecture + config)")
    ap.add_argument("--opponent", default="physical_v4",
                    help="registry opponent id (ignored if --opponent-policy set)")
    ap.add_argument("--opponent-policy", type=Path, default=None,
                    help="PPO policy_vK.pt to use as the opponent (threshold deploy) "
                         "instead of a registry agent — for agent-B-vs-agent-A eval.")
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--logit-threshold", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fleet-run-dir", type=Path, default=None,
                    help="L0 encoder dirs (default: repo data/runs/...; pass for pod ckpts/)")
    ap.add_argument("--planet-run-dir", type=Path, default=None)
    ap.add_argument("--comet-run-dir", type=Path, default=None)
    args = ap.parse_args()

    from utils.eval_seeds import SEEDS
    seeds = list(SEEDS[: args.num_seeds])

    tmp_ckpt = Path("/tmp/mt_eval") / f"runner_{args.policy.stem}.pt"
    tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)
    build_runner_ckpt(args.policy, args.base, tmp_ckpt)

    learner_id = f"_ppo_threshold_{args.policy.stem}"
    _singleton = {"agent": None}

    def _learner_fn(obs):
        if _singleton["agent"] is None:
            _singleton["agent"] = TransformerAgent.load(
                ckpt_path=tmp_ckpt, device=args.device,
                inference_mode="threshold", logit_threshold=args.logit_threshold,
                fleet_run_dir=args.fleet_run_dir,
                planet_run_dir=args.planet_run_dir,
                comet_run_dir=args.comet_run_dir,
            )
        return _singleton["agent"].act(obs)

    if learner_id not in _registry._REGISTRY:
        _registry.register(learner_id, "PPO threshold-deploy eval agent")(_learner_fn)

    # Opponent: either a registry id (physical_v4) or a second PPO ckpt (agent A)
    # built + registered the same threshold-deploy way as the learner.
    opponent_id = args.opponent
    if args.opponent_policy is not None:
        opp_ckpt = Path("/tmp/mt_eval") / f"runner_opp_{args.opponent_policy.stem}.pt"
        build_runner_ckpt(args.opponent_policy, args.base, opp_ckpt)
        opponent_id = f"_ppo_threshold_opp_{args.opponent_policy.stem}"
        _opp_singleton = {"agent": None}

        def _opponent_fn(obs):
            if _opp_singleton["agent"] is None:
                _opp_singleton["agent"] = TransformerAgent.load(
                    ckpt_path=opp_ckpt, device=args.device,
                    inference_mode="threshold", logit_threshold=args.logit_threshold,
                    fleet_run_dir=args.fleet_run_dir,
                    planet_run_dir=args.planet_run_dir,
                    comet_run_dir=args.comet_run_dir,
                )
            return _opp_singleton["agent"].act(obs)

        if opponent_id not in _registry._REGISTRY:
            _registry.register(opponent_id, "PPO threshold-deploy opponent")(_opponent_fn)

    print(f"learner:  {learner_id}  (threshold={args.logit_threshold})")
    print(f"opponent: {opponent_id}")
    print(f"seeds:    {seeds}  (both seats)\n", flush=True)

    by_seat = {0: [0, 0], 1: [0, 0]}
    t0 = time.time()
    for seed in seeds:
        for seat in (0, 1):
            slots = [opponent_id, opponent_id]
            slots[seat] = learner_id
            t1 = time.time()
            res = run_match(slots, seed=seed)
            win = int(res.winner == seat)
            by_seat[seat][0] += win
            by_seat[seat][1] += 1
            print(f"  seed={seed} seat={seat} -> {'WIN ' if win else 'loss'} "
                  f"(winner={res.winner}, rewards={res.rewards}) [{time.time()-t1:.1f}s]",
                  flush=True)

    tot_w = sum(by_seat[s][0] for s in by_seat)
    tot_g = sum(by_seat[s][1] for s in by_seat)
    _opp_label = args.opponent_policy.name if args.opponent_policy else args.opponent
    print(f"\n=== RESULT {args.policy.name} (threshold deploy) vs {_opp_label} ===")
    for s in (0, 1):
        w, g = by_seat[s]
        if g:
            print(f"  seat {s}: {w}/{g} = {100.0*w/g:.1f}%")
    print(f"  OVERALL: {tot_w}/{tot_g} = {100.0*tot_w/max(1,tot_g):.1f}%  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
