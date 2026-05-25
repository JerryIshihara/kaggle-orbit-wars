"""Minimal CLI scaffold for the Phase 0 PPO loop.

Most heavy lifting (loading a supervised ckpt with the
``conditioner_n_layers >= 2`` / ``head_n_layers >= 3`` preconditions,
calibration eval, archive script invocation, eval split between machines)
is documented in the protocol but not yet wired here. This entrypoint
hard-fails on the most important precondition (PairHead depths) so the
training process can't silently start with a too-shallow ckpt.
"""

from __future__ import annotations

import argparse
import sys

from agents.transformer_v2.pretrain.entity_encoder import EntityPretrainModel

from .actor_critic import PPOActorCritic


def _check_pair_head_depths(model: EntityPretrainModel) -> None:
    """Refuse to start if the ckpt's PairHead depths are below the minimums.

    See ``docs/PPO_TWO_CPU_PROTOCOL.md`` → "PairHead bootstrap preconditions".
    """
    ph = model.pair_head
    issues: list[str] = []
    if getattr(ph, "conditioner_n_layers", 1) < 2:
        issues.append(
            f"FiLM conditioner has {ph.conditioner_n_layers} hidden layer(s); "
            "PPO requires conditioner_n_layers >= 2 (>= 3 Linears). "
            "Retrain supervised ckpt with --conditioner-n-layers 2."
        )
    if getattr(ph, "head_n_layers", 1) < 3:
        issues.append(
            f"PairHead action heads have head_n_layers={ph.head_n_layers}; "
            "PPO requires head_n_layers >= 3. "
            "Retrain supervised ckpt with --head-n-layers 3."
        )
    if issues:
        print("PPO refuses to start due to shallow PairHead ckpt:",
              file=sys.stderr)
        for msg in issues:
            print(f"  - {msg}", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PPO Phase 0 single-machine bring-up (transformer_v2).",
    )
    parser.add_argument("--init-ckpt", required=True,
                         help="Path to the supervised PairHead .pt to bootstrap from.")
    parser.add_argument("--run-id", required=True,
                         help="Run id; outputs go to data/runs/ppo/<run_id>/.")
    parser.add_argument("--episodes-per-iter", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--clip", type=float, default=0.10)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--lr-heads", type=float, default=1e-4)
    parser.add_argument("--lr-trunk", type=float, default=None,
                         help="None = trunk frozen (Phase 0).")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--bc-coef", type=float, default=0.05)
    parser.add_argument("--bc-target-weight", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.35,
                         help="Fixed sigma for the frac LogitNormal.")
    parser.add_argument("--sigma-schedule", choices=["constant", "linear", "cosine"],
                         default="constant")
    parser.add_argument("--noop-logit-bias", type=float, default=0.0,
                         help="Fixed bias on the NOOP slot in the source categorical.")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    # 1. Load supervised model + check depth preconditions.
    print(f"loading supervised ckpt: {args.init_ckpt}", file=sys.stderr)
    # NOTE: ckpt-loading wiring (constructor kwargs, state-dict shape match)
    # is project-specific — fill in once the supervised ckpt format is
    # finalized. The depth check below catches the most common failure.
    raise NotImplementedError(
        "CLI is a scaffold — wire the supervised ckpt loader, env adapter, "
        "and the rollout loop in agents/transformer_v2/ppo/rollout.py before "
        "running. See docs/PPO_TWO_CPU_PROTOCOL.md → 'Practical first run'."
    )


if __name__ == "__main__":
    sys.exit(main())
