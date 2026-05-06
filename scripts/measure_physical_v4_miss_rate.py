"""Run physical_v4 against opponents locally and report miss rate.

Mirrors ``measure_sniper_v2_miss_rate.py``: every committed physical_v4
launch is appended to ``physical_v4.agent.LAUNCH_LOG`` as
``(step, src_pid, intended_target_pid, ships, angle, eta)``. After each
match we replay the per-step observation seen at launch time and walk
the same env-order collision sweep used by ``physics_utils.sniper`` to
check whether the trajectory's first hit is the intended target.

physical_v4 already gates each launch through ``validate_launch`` so
the expected miss rate is low-but-not-necessarily-zero — the validator
uses lead-aim physics that the env later resolves more precisely, so
edge cases (orbit drift over long ETAs, comet collisions, tangential
hits) can still slip through.

Run:
    python scripts/measure_physical_v4_miss_rate.py \\
        --opponent random_v1 --num-games 5

Pass ``--opponents random_v1,sniper_v2,physical_v3`` to set distinct
filler agents per slot, ``--num-players 4`` for 4-player, ``--seat`` to
flip which seat physical_v4 plays, and ``--out path.json`` to persist
the per-launch miss data for later analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import agents  # noqa: E402  — registers all agent ids
from agents.physical_v4.agent import LAUNCH_LOG  # noqa: E402
from agents.physics_utils import (  # noqa: E402
    P_ID, P_RADIUS, P_X, P_Y,
    _build_comet_lookup,
    _find_first_collision_dynamic,
    _infer_rotation_sign_raw,
    find_first_collision,
)
from utils.runner import run_match  # noqa: E402


def _miss_category(reason: str) -> str:
    """Collapse per-launch reason strings into a small set of buckets so
    the CLI summary stays readable. Pair-specific data (intended_pid,
    actual_pid) stays in the per-launch JSON output."""
    if reason.startswith("wrong_planet"):
        return "wrong_planet"
    return reason


def _step_to_obs(env, step_idx: int):
    """Observation that physical_v4 saw when it issued the launches
    recorded at ``step_idx``. Launches are appended inside the agent
    *before* moves are applied, so the relevant obs is
    ``env.steps[step_idx][seat].observation``. Planets/fleets are
    seat-invariant so seat 0's view is fine."""
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
) -> tuple[bool, str, int | None]:
    """Walk the trajectory the agent committed and compare its
    first-collision planet to the intended target.

    Returns (hit, reason, actual_pid_or_None)."""
    planets = obs.get("planets") or []
    src = next((p for p in planets if int(p[P_ID]) == src_pid), None)
    if src is None:
        return False, "src_missing", None
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
        return False, "no_collision", None
    if hit["kind"] != "planet":
        return False, hit["kind"], None
    actual = int(hit["planet"][P_ID])
    if actual != intended_target_pid:
        return False, f"wrong_planet_{actual}_vs_{intended_target_pid}", actual
    return True, "ok", actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", default="random_v1",
                        help="Default opponent id (used to fill empty seats).")
    parser.add_argument("--opponents", default=None,
                        help="Comma-separated list of opponent ids "
                             "(overrides --opponent for filler slots).")
    parser.add_argument("--num-players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--seat", type=int, default=0,
                        help="Seat physical_v4 plays (0..num_players-1).")
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to dump per-launch miss data as JSON.")
    args = parser.parse_args()

    if not 0 <= args.seat < args.num_players:
        parser.error(
            f"--seat {args.seat} out of range for --num-players {args.num_players}"
        )
    if args.opponents:
        opponent_ids = [s.strip() for s in args.opponents.split(",") if s.strip()]
    else:
        opponent_ids = [args.opponent] * (args.num_players - 1)
    if len(opponent_ids) != args.num_players - 1:
        parser.error(
            f"need {args.num_players - 1} opponents, got {len(opponent_ids)}"
        )

    print(f"physical_v4 in seat {args.seat} of {args.num_players}; "
          f"opponents = {opponent_ids}")

    grand_total = 0
    grand_miss = 0
    miss_breakdown: dict[str, int] = {}
    per_launch_records: list[dict] = []

    for g in range(args.num_games):
        slot_agents: list[str] = list(opponent_ids)
        slot_agents.insert(args.seat, "physical_v4")

        # LAUNCH_LOG is module-level, clear between matches.
        LAUNCH_LOG.clear()

        result = run_match(slot_agents, seed=args.seed + g)
        env = result.env
        steps = env.steps

        hit_count = 0
        miss_count = 0
        for entry in LAUNCH_LOG:
            step, src_pid, tgt_pid, ships, angle, eta = entry
            obs = _step_to_obs(env, step)
            if obs is None:
                miss_count += 1
                miss_breakdown["obs_missing"] = miss_breakdown.get("obs_missing", 0) + 1
                per_launch_records.append({
                    "game": g, "seed": args.seed + g, "step": step,
                    "src_pid": src_pid, "intended_target_pid": tgt_pid,
                    "ships": ships, "angle": angle, "eta": eta,
                    "hit": False, "reason": "obs_missing", "actual_pid": None,
                })
                continue
            ok, reason, actual_pid = _verify_launch(src_pid, tgt_pid, ships, angle, obs)
            if ok:
                hit_count += 1
            else:
                miss_count += 1
                category = _miss_category(reason)
                miss_breakdown[category] = miss_breakdown.get(category, 0) + 1
            per_launch_records.append({
                "game": g, "seed": args.seed + g, "step": step,
                "src_pid": src_pid, "intended_target_pid": tgt_pid,
                "ships": ships, "angle": angle, "eta": eta,
                "hit": ok, "reason": reason, "actual_pid": actual_pid,
            })

        total = hit_count + miss_count
        grand_total += total
        grand_miss += miss_count
        rate = (miss_count / total) if total else 0.0
        winner_name = slot_agents[result.winner]
        v4_won = "WIN" if result.winner == args.seat else "LOSS"
        print(
            f"  game {g + 1}/{args.num_games}: launches={total}  "
            f"hits={hit_count}  misses={miss_count}  miss_rate={rate:.2%}  "
            f"[{v4_won}, winner={winner_name}, steps={len(steps)}]"
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
            print(f"  {k:<40s}  {v}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "opponents": opponent_ids,
            "seat": args.seat,
            "num_players": args.num_players,
            "num_games": args.num_games,
            "seed": args.seed,
            "totals": {
                "launches": grand_total,
                "misses": grand_miss,
                "miss_rate": rate,
            },
            "miss_breakdown": miss_breakdown,
            "launches": per_launch_records,
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n[out] wrote {len(per_launch_records)} launches → {args.out}")


if __name__ == "__main__":
    main()
