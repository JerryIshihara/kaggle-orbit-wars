"""Audit shoot_*/plan_launch correctness across the 9 source × target motion
combinations: source ∈ {static, orbital, comet} × target ∈ {static, orbital, comet}.

For each combo we (a) generate a real env episode that exposes that source/target
pairing, (b) compute the predicted intercept via ``shoot_*`` and the validated
launch via ``plan_launch``, and (c) actually step the env forward until the
launched fleet either lands on the intended target, dies, or times out. We then
report:

  * predicted_eta vs actual_eta (turns until first-hit)
  * predicted_intercept_xy vs actual_target_xy_at_landing
  * ``ok`` agreement between plan_launch and the env outcome

Runs as: ``python scripts/audit_launch_9cases.py``.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from agents.physics_utils import (
    P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS,
    SUN_CX, SUN_CY, SUN_RADIUS, ROTATION_RADIUS_LIMIT,
    _is_orbiting_xy, _infer_rotation_sign_raw,
    shoot_static, shoot_orbit, shoot_comet,
    plan_launch,
)


def classify(planet, *, angular_velocity: float, comet_ids: set[int]) -> str:
    pid = int(planet[P_ID])
    if pid in comet_ids:
        return "comet"
    if _is_orbiting_xy(float(planet[P_X]), float(planet[P_Y]),
                      float(planet[P_RADIUS]), float(angular_velocity)):
        return "orbital"
    return "static"


def find_pair(planets, obs, src_kind: str, tgt_kind: str):
    av = float(obs.get("angular_velocity") or 0.0)
    comet_ids = set(obs.get("comet_planet_ids") or [])
    src_candidates = [p for p in planets
                      if classify(p, angular_velocity=av, comet_ids=comet_ids) == src_kind]
    tgt_candidates = [p for p in planets
                      if classify(p, angular_velocity=av, comet_ids=comet_ids) == tgt_kind]
    # Prefer pairs whose straight-line vector clears the sun.
    for src in src_candidates:
        for tgt in tgt_candidates:
            if int(src[P_ID]) == int(tgt[P_ID]):
                continue
            sx, sy = float(src[P_X]), float(src[P_Y])
            tx, ty = float(tgt[P_X]), float(tgt[P_Y])
            # quick sun-clearance test
            dx, dy = tx - sx, ty - sy
            l2 = dx*dx + dy*dy
            if l2 == 0:
                continue
            t = max(0.0, min(1.0, ((SUN_CX - sx) * dx + (SUN_CY - sy) * dy) / l2))
            cx = sx + t * dx
            cy = sy + t * dy
            if math.hypot(cx - SUN_CX, cy - SUN_CY) < SUN_RADIUS + 1.5:
                continue
            return src, tgt
    return None, None


def simulate(obs_initial, src_id: int, angle: float, ships: int, max_steps: int = 80):
    """Roll the env forward by replaying obs as state. Returns dict with
    {'landed_at': (x, y) | None, 'landed_pid': pid | None, 'eta': int | None,
     'reason': 'hit'|'sun'|'boundary'|'timeout'}.

    We rebuild a fresh env and reach an equivalent state by issuing the launch
    move on the first available turn. Since the env is random per seed, we use
    the same seed obs originally came from and step until ``obs_initial.step``
    using no-op moves, then submit the test launch.
    """
    raise NotImplementedError  # use the on-the-fly env approach below


def run_inline_test(env_seed: int, src_kind: str, tgt_kind: str, max_warmup: int = 80):
    """Run a fresh env, look for a (src, tgt) pair of the requested kinds in
    the first ``max_warmup`` turns. If found, issue a launch by player 0 from
    that source and step until the fleet resolves; compare predicted vs actual.
    """
    env = make("orbit_wars", configuration={"seed": env_seed}, debug=False)
    env.reset(num_agents=2)
    # Step forward with no-ops until we see the requested src/tgt kinds.
    for warmup in range(max_warmup):
        step = env.steps[-1]
        obs = step[0].observation
        planets = list(obs.get("planets") or [])
        # find_pair wants planets dict-like
        src, tgt = find_pair(planets, obs, src_kind, tgt_kind)
        if src is not None and tgt is not None and int(src[P_OWNER]) == -1:
            # Convert a neutral source into a player-0 source by mutating env
            # state (the env keeps planet/fleet lists in obs).
            src[P_OWNER] = 0
            src[P_SHIPS] = max(int(src[P_SHIPS]), 100)
        if src is not None and tgt is not None and int(src[P_OWNER]) == 0:
            break
        # Step with no-ops.
        env.step([[], []])
    else:
        return {"status": "no_pair", "src_kind": src_kind, "tgt_kind": tgt_kind}

    obs = env.steps[-1][0].observation
    av = abs(float(obs.get("angular_velocity") or 0.0))
    av_sign = _infer_rotation_sign_raw(
        list(obs.get("planets") or []),
        list(obs.get("initial_planets") or []),
    )
    current_step = int(obs.get("step", 0) or 0)
    comets = list(obs.get("comets") or [])
    comet_ids = set(obs.get("comet_planet_ids") or [])

    ships = 50
    if tgt_kind == "static":
        angle, pred_eta, _ = shoot_static(tuple(src), tuple(tgt), ships)
    elif tgt_kind == "orbital":
        angle, pred_eta, _ = shoot_orbit(tuple(src), tuple(tgt), ships, obs, av_sign=av_sign)
    else:
        angle, pred_eta, _ = shoot_comet(tuple(src), tuple(tgt), ships, obs)

    # plan_launch's verdict
    pl = plan_launch(
        tuple(src), tuple(tgt),
        planets=list(obs.get("planets") or []),
        fleets=list(obs.get("fleets") or []),
        player=0, angular_velocity=av, av_sign=av_sign,
        comet_planet_ids=comet_ids, comets=comets,
        fleet_ships=ships, current_step=current_step,
    )

    # Issue the move (player 0) and step.
    move = [[int(src[P_ID]), float(angle), int(ships)]]
    actual_landing = None
    actual_pid = None
    actual_eta = None
    actual_reason = "timeout"
    target_pos_at_landing = None
    fleet_id_to_watch = None
    fleet_was_alive_at_step = None
    expected_fleet_id_min = max((f[0] for f in (obs.get("fleets") or [])), default=-1) + 1
    env.step([move, []])
    # The newly launched fleet has id == prev next_fleet_id. We don't have direct
    # access from obs, so we infer by tracking new fleet ids after each step.
    for k in range(0, 80):
        cur_obs = env.steps[-1][0].observation
        fleets_now = list(cur_obs.get("fleets") or [])
        # Find the spawn fleet by minimum id >= expected_fleet_id_min.
        candidate_fleets = [f for f in fleets_now if int(f[0]) >= expected_fleet_id_min
                            and int(f[1]) == 0 and int(f[5]) == int(src[P_ID])]
        if candidate_fleets:
            fleet_id_to_watch = int(candidate_fleets[0][0])
            fleet_was_alive_at_step = int(cur_obs.get("step", 0) or 0)
        if fleet_id_to_watch is not None and not any(int(f[0]) == fleet_id_to_watch for f in fleets_now):
            # Fleet vanished — collision occurred at this step. Walk the planets
            # to find where the target ended up.
            actual_eta = int(cur_obs.get("step", 0) or 0) - current_step
            tgt_now = next((p for p in cur_obs.get("planets") or [] if int(p[P_ID]) == int(tgt[P_ID])), None)
            if tgt_now is None:
                actual_reason = "target_died"
            else:
                target_pos_at_landing = (float(tgt_now[P_X]), float(tgt_now[P_Y]))
                actual_pid = int(tgt[P_ID])
                actual_reason = "hit_or_died"
            break
        if env.done:
            break
        env.step([[], []])

    return {
        "status": "ran",
        "src_kind": src_kind, "tgt_kind": tgt_kind,
        "src_id": int(src[P_ID]), "tgt_id": int(tgt[P_ID]),
        "src_xy": (float(src[P_X]), float(src[P_Y])),
        "tgt_xy_at_launch": (float(tgt[P_X]), float(tgt[P_Y])),
        "predicted_angle": float(angle),
        "predicted_eta": float(pred_eta),
        "plan_launch_ok": bool(pl.ok),
        "plan_launch_reason": pl.reason,
        "plan_launch_actual_hit_id": pl.actual_hit_id,
        "actual_eta": actual_eta,
        "actual_reason": actual_reason,
        "tgt_xy_at_landing": target_pos_at_landing,
    }


def main():
    kinds = ["static", "orbital", "comet"]
    print(f"{'src':<8s} {'tgt':<8s} {'src_id':>6s} {'tgt_id':>6s}  "
          f"{'pred_eta':>9s} {'actual_eta':>11s}  {'plan_ok':>7s} {'reason':<35s} {'Δxy':>10s}")
    print("-" * 130)
    for src_kind in kinds:
        for tgt_kind in kinds:
            best = None
            for seed in range(1729, 1729 + 30):
                r = run_inline_test(seed, src_kind, tgt_kind)
                if r["status"] == "ran":
                    best = r
                    break
            if best is None:
                print(f"{src_kind:<8s} {tgt_kind:<8s} ── NO PAIR FOUND in 30 seeds")
                continue
            dxy_str = ""
            if best["tgt_xy_at_landing"] is not None and best["actual_eta"] is not None:
                tx, ty = best["tgt_xy_at_landing"]
                sx, sy = best["src_xy"]
                # fleet endpoint at landing using actual_eta (steps) and conservatively
                # using ships=50 → speed ≈ fleet_speed(50). The real fleet endpoint is
                # well-approximated by walking actual_eta steps along the launch ray.
                from agents.physics_utils import fleet_speed
                spd = fleet_speed(50)
                ax = sx + math.cos(best["predicted_angle"]) * best["actual_eta"] * spd
                ay = sy + math.sin(best["predicted_angle"]) * best["actual_eta"] * spd
                dxy = math.hypot(tx - ax, ty - ay)
                dxy_str = f"{dxy:.2f}"
            eta_str = "?" if best["actual_eta"] is None else f"{best['actual_eta']}"
            print(
                f"{best['src_kind']:<8s} {best['tgt_kind']:<8s} "
                f"{best['src_id']:>6d} {best['tgt_id']:>6d}  "
                f"{best['predicted_eta']:>9.2f} {eta_str:>11s}  "
                f"{str(best['plan_launch_ok']):>7s} "
                f"{best['plan_launch_reason']:<35s} "
                f"{dxy_str:>10s}"
            )


if __name__ == "__main__":
    main()
