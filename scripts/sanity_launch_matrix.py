"""Empirical sanity matrix for physics_utils.plan_launch vs the REAL env.

For every (source kind) x (target kind) combo — kinds: static / orbital /
comet — plan a launch with ``plan_launch`` and inject it into a live
kaggle_environments orbit_wars episode (idle opponent), then track the
fleet to its resolution and classify the outcome:

    HIT            fleet resolved at the intended target
    self_recapture fleet hit its own source (moving source swung into it)
    wrong_planet:X fleet hit some other planet X
    lost_void      fleet left the board or crossed the sun
    vanished_w_comet  fleet disappeared the same tick its comet target
                   expired (combat-at-expiry ambiguity)

Also records the planner's refusals (plan.ok=False reasons) and the
ETA error (actual ticks vs plan.eta) for hits.

Run:
    .venv/bin/python scripts/sanity_launch_matrix.py --seeds 24
    .venv/bin/python scripts/sanity_launch_matrix.py --seeds 8 --ships 31

Each combo trial re-drives a FRESH env to the probe step (idle agents) so
trials never contaminate each other.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.physics_utils import plan_launch  # noqa: E402

CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
# Probe steps: comets spawn at 50/150/250/350/450 and live ~tens of ticks.
PROBE_STEPS = (60, 160)
MAX_FLIGHT_TICKS = 80


def _mk_env(seed: int):
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(2)
    return env


def _obs(env):
    return env.state[0].observation


def _drive_idle(env, n_steps: int):
    for _ in range(n_steps):
        if env.done:
            return False
        env.step([[], []])
    return not env.done


def _kind(p, obs) -> str:
    if p[0] in set(obs.comet_planet_ids):
        return "comet"
    init = next((q for q in obs.initial_planets if q[0] == p[0]), None)
    if init is None or float(obs.angular_velocity) == 0.0:
        return "static"
    r = math.hypot(init[2] - CENTER, init[3] - CENTER)
    return "orbital" if r + p[4] < ROTATION_RADIUS_LIMIT else "static"


def _planet_kinds(obs) -> dict[int, str]:
    return {p[0]: _kind(p, obs) for p in obs.planets}


def _expected_ships(prev_planets, planets):
    """planet_id -> expected ships under production-only evolution."""
    out = {}
    for p in prev_planets:
        out[p[0]] = p[5] + (p[6] if p[1] != -1 else 0)
    return out


def _classify_resolution(env, fleet_id: int, src_id: int, tgt_id: int):
    """Step the env until the fleet disappears; classify what it hit."""
    prev = [list(p) for p in _obs(env).planets]
    for tick in range(1, MAX_FLIGHT_TICKS + 1):
        if env.done:
            return ("episode_end", tick)
        env.step([[], []])
        obs = _obs(env)
        alive = any(f[0] == fleet_id for f in obs.fleets)
        if alive:
            prev = [list(p) for p in obs.planets]
            continue
        # Fleet resolved THIS tick. Which planet deviated from production?
        expected = _expected_ships(prev, obs.planets)
        cur_ids = {p[0] for p in obs.planets}
        deviated = []
        for p in obs.planets:
            exp = expected.get(p[0])
            if exp is None:
                continue  # spawned this tick
            if p[5] != exp:
                deviated.append(p[0])
        gone = [pid for pid in expected if pid not in cur_ids]
        if tgt_id in deviated:
            return ("HIT", tick)
        if src_id in deviated:
            return ("self_recapture", tick)
        if deviated:
            return (f"wrong_planet:{deviated[0]}", tick)
        if tgt_id in gone:
            return ("vanished_w_comet", tick)
        return ("lost_void", tick)
    return ("still_flying", MAX_FLIGHT_TICKS)


def _pick(obs, kinds, kind: str, owner0: bool, min_ships: int = 0,
          exclude: set[int] = frozenset()):
    """Pick planets of `kind`; owner0=True restricts to player-0-owned."""
    out = []
    for p in obs.planets:
        if p[0] in exclude or kinds[p[0]] != kind:
            continue
        if owner0 and p[1] != 0:
            continue
        if p[5] < min_ships:
            continue
        out.append(p)
    return out


def run_matrix(seeds: list[int], ships_arms: list[int]) -> None:
    attempts = Counter()
    hits = Counter()
    refusals = defaultdict(Counter)
    outcomes = defaultdict(Counter)
    eta_errs = defaultdict(list)

    for seed in seeds:
        for probe in PROBE_STEPS:
            # Discover available combos at this (seed, probe) using one
            # scout env, then run each trial on its own fresh env.
            scout = _mk_env(seed)
            if not _drive_idle(scout, probe):
                continue
            sobs = _obs(scout)
            kinds = _planet_kinds(sobs)
            av_sign = 1 if float(sobs.angular_velocity) >= 0 else -1

            for src_kind in ("static", "orbital", "comet"):
                if ONLY_TGT and not any(
                        kinds[p[0]] == ONLY_TGT for p in sobs.planets):
                    break
                srcs = _pick(sobs, kinds, src_kind, owner0=True,
                             min_ships=max(ships_arms) + 5)
                if not srcs:
                    continue
                src = max(srcs, key=lambda p: p[5])
                for tgt_kind in ("static", "orbital", "comet"):
                    if ONLY_TGT and tgt_kind != ONLY_TGT:
                        continue
                    tgts = _pick(sobs, kinds, tgt_kind, owner0=False,
                                 exclude={src[0]})
                    if not tgts:
                        continue
                    # nearest target of this kind — representative geometry
                    tgt = min(tgts, key=lambda p: math.hypot(
                        p[2] - src[2], p[3] - src[3]))
                    for ships in ships_arms:
                        combo = (src_kind, tgt_kind, ships)
                        attempts[combo] += 1
                        env = _mk_env(seed)
                        if not _drive_idle(env, probe):
                            attempts[combo] -= 1
                            continue
                        obs = _obs(env)
                        live_src = next((p for p in obs.planets
                                         if p[0] == src[0]), None)
                        live_tgt = next((p for p in obs.planets
                                         if p[0] == tgt[0]), None)
                        if live_src is None or live_tgt is None \
                                or live_src[1] != 0 or live_src[5] < ships:
                            attempts[combo] -= 1
                            continue
                        plan = plan_launch(
                            live_src, live_tgt,
                            planets=obs.planets,
                            fleets=obs.fleets,
                            player=0,
                            angular_velocity=obs.angular_velocity,
                            av_sign=av_sign,
                            comet_planet_ids=set(obs.comet_planet_ids),
                            comets=getattr(obs, "comets", None),
                            fleet_ships=ships,
                            current_step=obs.step,
                        )
                        comet_dbg = ""
                        if tgt_kind == "comet":
                            grp = next(
                                (g for g in (getattr(obs, "comets", None) or [])
                                 if tgt[0] in g["planet_ids"]), None)
                            if grp is not None:
                                i = grp["planet_ids"].index(tgt[0])
                                remain = len(grp["paths"][i]) - grp["path_index"]
                                comet_dbg = (f" [comet seed={seed} probe={probe} "
                                             f"eta={plan.eta:.1f} remain={remain}t]")
                        if not plan.ok:
                            reason = plan.reason.split("_vs_")[0]
                            refusals[combo][reason] += 1
                            if comet_dbg:
                                print(f"    refuse {plan.reason}{comet_dbg}")
                            continue
                        fleet_id = obs.next_fleet_id
                        env.step([[[src[0], plan.angle, ships]], []])
                        # confirm the fleet actually spawned
                        if not any(f[0] == fleet_id for f in _obs(env).fleets):
                            # resolved within its first tick — classify now
                            outcome, tick = "first_tick_resolution", 0
                        else:
                            outcome, tick = _classify_resolution(
                                env, fleet_id, src[0], tgt[0])
                        outcomes[combo][outcome] += 1
                        if outcome == "HIT":
                            hits[combo] += 1
                            eta_errs[combo].append(tick - plan.eta)

    print(f"\n===== launch sanity matrix ({len(seeds)} seeds, probes "
          f"{PROBE_STEPS}, ships arms {ships_arms}) =====")
    print(f"{'src->tgt (ships)':<28} {'plans':>5} {'launched':>8} "
          f"{'hit%':>6} {'eta err':>8}  refusals / misses")
    all_combos = sorted(set(attempts) | set(refusals) | set(outcomes))
    for combo in all_combos:
        src_kind, tgt_kind, ships = combo
        n_att = attempts[combo]
        n_ref = sum(refusals[combo].values())
        n_launched = sum(outcomes[combo].values())
        n_hit = hits[combo]
        hit_pct = (100.0 * n_hit / n_launched) if n_launched else float("nan")
        errs = eta_errs[combo]
        eta_s = (f"{sum(errs)/len(errs):+.1f}t" if errs else "-")
        notes = []
        if refusals[combo]:
            notes.append("refused: " + ", ".join(
                f"{k}x{v}" for k, v in refusals[combo].most_common(3)))
        miss = {k: v for k, v in outcomes[combo].items() if k != "HIT"}
        if miss:
            notes.append("missed: " + ", ".join(
                f"{k}x{v}" for k, v in Counter(miss).most_common(3)))
        print(f"{src_kind+'->'+tgt_kind+f' ({ships})':<28} {n_att:>5} "
              f"{n_launched:>8} {hit_pct:>5.0f}% {eta_s:>8}  {'; '.join(notes)}")


ONLY_TGT: str | None = None


def main() -> None:
    global ONLY_TGT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--ships", type=int, nargs="*", default=[31, 200])
    ap.add_argument("--only", choices=("static", "orbital", "comet"),
                    default=None, help="restrict target kind")
    args = ap.parse_args()
    ONLY_TGT = args.only
    run_matrix([args.seed_base + i for i in range(args.seeds)], args.ships)


if __name__ == "__main__":
    main()
