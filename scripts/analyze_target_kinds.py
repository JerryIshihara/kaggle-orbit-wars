"""Per-kind target-selection analysis of a PPO policy's DEPLOYED replays.

Wraps the runner's ``plan_launch`` call with a recorder and plays real
games (policy vs frozen baseline, both seats), then reports, per target
motion kind (static / orbital / comet):

  * how often the model CHOSE that kind (validated launches + refusals)
  * the validation ok-rate and the refusal-reason histogram
  * mean ships per validated launch

Run:
    .venv/bin/python scripts/analyze_target_kinds.py \
        --policy data/runs/ppo/.../policy_v0013_sigadv.pt --seeds 8
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=Path(
        "data/runs/joint/joint_mt_alloc_d256_T10_head3_20260610-164152/joint_best.pt"))
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    import torch
    from scripts.eval_ppo_threshold_vs_physical import build_runner_ckpt
    import agents.transformer_v2.runner as runner_mod
    import agents.physics_utils as pu
    import agents as agents_pkg
    from utils.runner import run_match

    tmp = Path("/tmp/ow_kinds")
    tmp.mkdir(exist_ok=True)
    rc_path = tmp / "runner_policy.pt"
    build_runner_ckpt(args.policy, args.base, rc_path)
    pol = torch.load(args.policy, map_location="cpu", weights_only=False)
    contract = (pol.get("ppo_trial") or {}).get("action_contract")
    if contract:
        rc = torch.load(rc_path, map_location="cpu", weights_only=False)
        rc["config"]["action_contract"] = contract
        if (pol.get("ppo_trial") or {}).get("select_k_max"):
            rc["config"]["select_k_max"] = int(pol["ppo_trial"]["select_k_max"])
        torch.save(rc, rc_path)

    # recorder: wrap the runner module's bound plan_launch name
    stats = {
        "chosen": Counter(), "ok": Counter(), "ships": defaultdict(list),
        "reasons": defaultdict(Counter),
    }
    orig_plan = runner_mod.plan_launch

    def recording_plan(from_planet, to_planet, **kw):
        launch = orig_plan(from_planet, to_planet, **kw)
        comet_ids = set(kw.get("comet_planet_ids") or [])
        kind = pu._target_motion_kind(
            to_planet, abs(float(kw.get("angular_velocity") or 0.0)), comet_ids)
        kind = {"static": "static", "orbital": "orbital", "comet": "comet"}[kind]
        stats["chosen"][kind] += 1
        if launch.ok:
            stats["ok"][kind] += 1
            stats["ships"][kind].append(int(launch.ships))
        else:
            stats["reasons"][kind][launch.reason.split("_vs_")[0]] += 1
        return launch

    runner_mod.plan_launch = recording_plan

    from agents import registry as _registry

    def _mk(name: str, ckpt: Path):
        holder: dict = {"agent": None}

        def _fn(obs):
            if holder["agent"] is None:
                holder["agent"] = runner_mod.TransformerAgent.load(
                    ckpt_path=ckpt, device="cpu")
            return holder["agent"].act(obs)

        if name not in _registry._REGISTRY:
            _registry.register(name, "target-kind analysis agent")(_fn)

    _mk("_kinds_pol", rc_path)
    _mk("_kinds_opp", args.base)
    pol_name, opp_name = "_kinds_pol", "_kinds_opp"

    wins = 0
    games = 0
    for i in range(args.seeds):
        seed = 7000 + i * 13
        for seat in (0, 1):
            ids = [pol_name, opp_name] if seat == 0 else [opp_name, pol_name]
            res = run_match(ids, seed=seed)
            games += 1
            if res.winner == seat:
                wins += 1
            print(f"  seed={seed} seat={seat} -> "
                  f"{'WIN' if res.winner == seat else 'loss'}", flush=True)

    print(f"\n===== target-kind selection ({games} deployed games, "
          f"policy {args.policy.name}) — wins {wins}/{games} =====")
    total = sum(stats["chosen"].values())
    print(f"{'kind':<9} {'chosen':>7} {'share':>7} {'ok':>6} {'ok%':>6} "
          f"{'avg ships':>10}  top refusals")
    for kind in ("static", "orbital", "comet"):
        n = stats["chosen"][kind]
        ok = stats["ok"][kind]
        ships = stats["ships"][kind]
        reasons = ", ".join(f"{k}x{v}" for k, v in
                            stats["reasons"][kind].most_common(3))
        print(f"{kind:<9} {n:>7} {n/max(1,total):>6.1%} {ok:>6} "
              f"{ok/max(1,n):>5.0%} "
              f"{(sum(ships)/len(ships)) if ships else float('nan'):>10.1f}  "
              f"{reasons}")


if __name__ == "__main__":
    main()
