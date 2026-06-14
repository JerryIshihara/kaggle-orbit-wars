"""Deploy eval for a RUNNER-FORMAT ckpt ({"model", "config"} — e.g. a joint
pretrain's joint_best.pt) played through the production runner vs a registry
agent. Unlike eval_ppo_threshold_vs_physical.py there is no PPO-prefix
stripping: the ckpt loads directly, and ``inference_mode`` defaults to the
runner's contract auto-selection (alloc_softmax for
bernoulli_select_multinomial_alloc_v2 ckpts, threshold otherwise).

    python scripts/eval_runner_ckpt_vs_agent.py \
        --ckpt data/runs/joint/<run>/joint_best.pt \
        --opponent physical_v4 --num-seeds 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import agents  # noqa: E402, F401 — registers built-in agent ids
from agents import registry as _registry  # noqa: E402
from agents.transformer_v2.runner import TransformerAgent  # noqa: E402
from utils.runner import run_match  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="runner-format ckpt (joint_best.pt / merged base)")
    ap.add_argument("--opponent", default="physical_v4",
                    help="registry agent id")
    ap.add_argument("--opponent-ckpt", type=Path, default=None,
                    help="runner-format ckpt as opponent instead of a registry "
                         "id; loads with ITS OWN contract auto-mode (so a v1 "
                         "ckpt plays threshold while a v2 ckpt plays "
                         "alloc_softmax)")
    ap.add_argument("--num-seeds", type=int, default=6,
                    help="stratified panel seeds; both seats each")
    ap.add_argument("--inference-mode", default=None,
                    help="override; default = auto from ckpt action_contract")
    ap.add_argument("--logit-threshold", type=float, default=2.0)
    ap.add_argument("--krank", type=int, default=0,
                    help="K>0 wraps the LEARNER in KRankAgent (simulate-then-"
                         "score over K candidates; needs a value-headed v4 "
                         "ckpt). N x M structure via OW_V4_N / OW_V4_M. "
                         "Gate-A/B: K in {1,4,8} vs plain sample.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from utils.eval_seeds import SEEDS
    seeds = list(SEEDS[: args.num_seeds])

    learner_id = f"_runner_eval_{args.ckpt.stem}"
    _singleton: dict = {"agent": None}

    def _learner_fn(obs):
        if _singleton["agent"] is None:
            ag = TransformerAgent.load(
                ckpt_path=args.ckpt, device=args.device,
                inference_mode=args.inference_mode,
                logit_threshold=args.logit_threshold,
            )
            if args.krank > 0:
                from agents.transformer_v3.krank import KRankAgent
                ag = KRankAgent(ag, k=args.krank)
            _singleton["agent"] = ag
        return _singleton["agent"].act(obs)

    if learner_id not in _registry._REGISTRY:
        _registry.register(learner_id, "runner-ckpt deploy eval agent")(_learner_fn)

    opponent_id = args.opponent
    if args.opponent_ckpt is not None:
        opponent_id = f"_runner_opp_{args.opponent_ckpt.stem}"
        _opp: dict = {"agent": None}

        def _opponent_fn(obs):
            if _opp["agent"] is None:
                _opp["agent"] = TransformerAgent.load(
                    ckpt_path=args.opponent_ckpt, device=args.device,
                    logit_threshold=args.logit_threshold,
                )
            return _opp["agent"].act(obs)

        if opponent_id not in _registry._REGISTRY:
            _registry.register(opponent_id, "runner-ckpt opponent")(_opponent_fn)

    print(f"learner:  {learner_id} (mode={args.inference_mode or 'auto'}, "
          f"thr={args.logit_threshold})")
    print(f"opponent: {opponent_id}")
    print(f"seeds:    {seeds} (both seats)\n", flush=True)

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
                  f"(winner={res.winner}) [{time.time() - t1:.1f}s]", flush=True)

    tot_w = sum(by_seat[s][0] for s in by_seat)
    tot_g = sum(by_seat[s][1] for s in by_seat)

    def _wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
        # Wilson 95% interval — well-behaved at small n / extreme rates,
        # unlike the normal approx. Panel n is small so report it always.
        if n == 0:
            return 0.0, 0.0
        p = w / n
        d = 1.0 + z * z / n
        c = p + z * z / (2 * n)
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
        return 100.0 * (c - h) / d, 100.0 * (c + h) / d

    print(f"\n=== RESULT {args.ckpt.name} vs {opponent_id} ===")
    for s in (0, 1):
        w, g = by_seat[s]
        if g:
            print(f"  seat {s}: {w}/{g} = {100.0 * w / g:.1f}%")
    lo, hi = _wilson(tot_w, tot_g)
    print(f"  OVERALL: {tot_w}/{tot_g} = {100.0 * tot_w / max(1, tot_g):.1f}%  "
          f"[95% CI {lo:.0f}-{hi:.0f}]  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
