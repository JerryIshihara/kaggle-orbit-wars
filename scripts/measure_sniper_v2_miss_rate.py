"""Run sniper_v2 against opponents locally and report miss rate.

A "miss" is any committed launch whose trajectory's first-hit (per the
same env-order collision sweep used by ``physics_utils.sniper`` against
the launch-time obs) isn't the planet sniper_v2 declared as the intended
target. By construction this should be zero — ``sniper`` already
validates the trajectory before sniper_v2 emits the move — so a non-zero
miss rate is a real bug to investigate (or a case of env physics drifting
from our predictions).

Run:
    python scripts/measure_sniper_v2_miss_rate.py \\
        --opponent random_v1 --num-games 5

Add ``--seat 1`` to flip which seat sniper_v2 plays. Pass
``--num-players 4`` for a 4p game; the rest of the slots get filled
with copies of ``--opponent`` unless ``--opponents`` (comma-list) is
given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import agents  # noqa: E402  — registers all agent ids
from agents.physics_utils import (  # noqa: E402
    P_ID, P_RADIUS, P_X, P_Y,
    _build_comet_lookup,
    _find_first_collision_dynamic,
    _infer_rotation_sign_raw,
    find_first_collision,
)
from agents.sniper_v2.agent import LAUNCH_LOG  # noqa: E402
from utils.runner import run_match  # noqa: E402


def _step_to_obs(env, step_idx: int):
    """Pick the observation that sniper_v2 saw when it issued the
    launches recorded at ``step_idx``. The launches were logged inside
    sniper_v2 *before* moves were applied, so the relevant obs is
    ``env.steps[step_idx][seat].observation`` for the seat that
    submitted them. We use seat 0's view since planets/fleets are
    seat-invariant.
    """
    if step_idx >= len(env.steps):
        return None
    step = env.steps[step_idx]
    if not step:
        return None
    return step[0].observation


def _verify_launch(
    src_pid: int,
    intended_target_pid: int,
    ships: int,
    angle: float,
    obs,
) -> tuple[bool, str]:
    """Return (hit, reason). Walk the trajectory the agent committed and
    compare its first-collision planet to the intended target."""
    planets = obs.get("planets") or []
    src = next((p for p in planets if int(p[P_ID]) == src_pid), None)
    if src is None:
        return False, "src_missing"
    angular_velocity = abs(float(obs.get("angular_velocity") or 0.0))
    comet_lookup = _build_comet_lookup(obs.get("comets") or [])
    if angular_velocity > 0.0 or comet_lookup:
        av_sign = _infer_rotation_sign_raw(planets, obs.get("initial_planets") or [])
        hit = _find_first_collision_dynamic(
            float(src[P_X]), float(src[P_Y]), float(src[P_RADIUS]),
            int(src[P_ID]), float(angle), int(ships), planets,
            angular_velocity=angular_velocity,
            av_signed=angular_velocity * av_sign,
            comet_lookup=comet_lookup,
            current_step=int(obs.get("step", 0) or 0),
        )
    else:
        hit = find_first_collision(
            float(src[P_X]), float(src[P_Y]), float(src[P_RADIUS]),
            int(src[P_ID]), float(angle), int(ships), planets,
        )
    if hit is None:
        return False, "no_collision"
    if hit["kind"] != "planet":
        return False, hit["kind"]
    actual = int(hit["planet"][P_ID])
    if actual != intended_target_pid:
        return False, f"wrong_planet_{actual}_vs_{intended_target_pid}"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", default="random_v1",
                        help="Default opponent id (used to fill empty seats).")
    parser.add_argument("--opponents", default=None,
                        help="Comma-separated list of opponent ids "
                             "(overrides --opponent for filler slots).")
    parser.add_argument("--num-players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--seat", type=int, default=0,
                        help="Seat sniper_v2 plays (0..num_players-1).")
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260)
    args = parser.parse_args()

    if args.opponents:
        opponent_ids = [s.strip() for s in args.opponents.split(",") if s.strip()]
    else:
        opponent_ids = [args.opponent] * (args.num_players - 1)
    if len(opponent_ids) != args.num_players - 1:
        parser.error(
            f"need {args.num_players - 1} opponents, got {len(opponent_ids)}"
        )

    print(f"sniper_v2 in seat {args.seat} of {args.num_players}; "
          f"opponents = {opponent_ids}")

    grand_total = 0
    grand_miss = 0
    miss_breakdown: dict[str, int] = {}

    for g in range(args.num_games):
        # Build seat order
        slot_agents: list[str] = list(opponent_ids)
        slot_agents.insert(args.seat, "sniper_v2")

        # sniper_v2 keeps a module-level launch log; clear it before
        # each new match so per-match accounting stays clean.
        LAUNCH_LOG.clear()

        result = run_match(slot_agents, seed=args.seed + g)
        env = result.env
        steps = env.steps

        # Verify each logged launch against the obs at that step.
        hit_count = 0
        miss_count = 0
        for entry in LAUNCH_LOG:
            step, src_pid, tgt_pid, ships, angle, _eta = entry
            obs = _step_to_obs(env, step)
            if obs is None:
                miss_count += 1
                miss_breakdown["obs_missing"] = miss_breakdown.get("obs_missing", 0) + 1
                continue
            ok, reason = _verify_launch(src_pid, tgt_pid, ships, angle, obs)
            if ok:
                hit_count += 1
            else:
                miss_count += 1
                miss_breakdown[reason] = miss_breakdown.get(reason, 0) + 1

        total = hit_count + miss_count
        grand_total += total
        grand_miss += miss_count
        rate = (miss_count / total) if total else 0.0
        winner_name = slot_agents[result.winner]
        sniper_won = "WIN" if result.winner == args.seat else "LOSS"
        print(
            f"  game {g + 1}/{args.num_games}: launches={total}  "
            f"hits={hit_count}  misses={miss_count}  miss_rate={rate:.2%}  "
            f"[{sniper_won}, winner={winner_name}, steps={len(steps)}]"
        )

    print()
    print(f"==== summary across {args.num_games} games ====")
    print(f"total launches: {grand_total}")
    print(f"total misses:   {grand_miss}")
    rate = (grand_miss / grand_total) if grand_total else 0.0
    print(f"overall miss rate: {rate:.4%}")
    if miss_breakdown:
        print("miss reasons:")
        for k, v in sorted(miss_breakdown.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<32s}  {v}")


if __name__ == "__main__":
    main()
