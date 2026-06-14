"""Precompute the v5 Q head's DENSE doomed-launch label over the top-meta
replays, as a sidecar to the pair cache.

The Q head's anti-doomed gate needs, per legal (source, target) pair, a
``plan_launch`` reachable/doomed verdict. ``plan_launch`` needs RAW state
(planet geometry, comet trajectories, fleet ETAs, angular velocity) which the
*featurized* pair cache does not retain — so we recompute it here from the
replay JSONs, reusing the EXACT slot layout (slot index == enumerate order over
``obs.planets[:P]`` == the cache's ``pid_to_idx``) so the ``(P, P)`` doomed mask
lines up 1:1 with ``pair_labels`` / ``pair_valid``.

The masks (owned-launchable source × existing target) are built with the SAME
``legality_masks`` the PPO rollout uses, and the verdict with the SAME
``_dense_doomed_mask`` — so the pretrain doomed label is identical to the one
the rollout would emit for that state.

Output: a sidecar ``{"doomed": {(episode_id, turn): (P,P) bool}, ...}`` stored
only for acted snapshots with >=1 doomed pair (a missing key => all-reachable,
the safe default for the dense loss). The joint single-target pretrain loads
this and feeds ``compute_q_loss``'s ``doomed_mask``.

    python -m scripts.precompute_pair_doomed \
        --pair-cache-path data/datasets/_pair_cache/topmeta300_jun11_T6/topmeta300_p64_f512_acted.pt \
        --out data/datasets/_pair_cache/topmeta300_jun11_T6/topmeta300_pair_doomed.pt
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_pair_dataset_orbital_occle import (  # noqa: E402
    load_cache, _parse_seat_from_stem,
)
from agents.transformer_v2.ppo.smoke import (  # noqa: E402
    _dense_doomed_mask, compute_surplus, PHASE_TABLE, phase_of,
)
from agents.transformer_v2.ppo.sampler import legality_masks  # noqa: E402


def _log(msg: str) -> None:
    print(f"[doomed {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def doomed_for_obs(obs: dict, seat: int, P: int, turn: int) -> torch.Tensor:
    """(P, P) bool doomed mask for one acted snapshot, in cache slot space."""
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    raw_planets = obs.get("planets") or []
    raw_fleets = obs.get("fleets") or []
    _nb, defense_buffer, min_launch, _s, _fw, _et = PHASE_TABLE[phase_of(int(turn))]
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    enemy_fleets = [f for f in fleets if f.owner != seat and f.owner >= 0]

    planet_owner_rel = torch.full((P,), 99, dtype=torch.long)
    planet_surplus = torch.zeros(P, dtype=torch.float32)
    planet_exists = torch.zeros(P, dtype=torch.bool)
    slot_to_pid = [-1] * P
    # Slot == enumerate index over obs.planets[:P] == the cache's pid_to_idx
    # (build_pair_dataset enumerates planet_rows[:P] the same way), so the mask
    # lands on the same axis as pair_labels / pair_valid.
    for idx, planet in enumerate(planets[:P]):
        planet_exists[idx] = True
        slot_to_pid[idx] = int(planet.id)
        planet_owner_rel[idx] = 0 if int(planet.owner) == int(seat) else 1
        planet_surplus[idx] = float(
            compute_surplus(planet, enemy_fleets, defense_buffer))

    pair_mask, source_mask = legality_masks(
        planet_owner=planet_owner_rel, surplus=planet_surplus,
        planet_exists=planet_exists, min_launch=int(min_launch),
    )
    return _dense_doomed_mask(
        source_mask, pair_mask, slot_to_pid, planets, fleets,
        obs, int(seat), int(min_launch),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-cache-path", type=Path, required=True)
    ap.add_argument("--replay-dir", type=Path, default=REPO / "data" / "replays")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-planets", type=int, default=0,
                    help="0 => read from the cache config")
    ap.add_argument("--limit-replays", type=int, default=0,
                    help="cap replay files scanned (smoke); 0 = all")
    ap.add_argument("--shard", type=int, default=0,
                    help="this shard's index in [0, num-shards) — process only "
                         "replays where (index %% num-shards == shard). Run N "
                         "shards as independent processes (no in-proc "
                         "multiprocessing.spawn), then merge their sidecars.")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    _log(f"loading cache header from {args.pair_cache_path} ...")
    payload = load_cache(args.pair_cache_path)
    cfg = payload["config"]
    P = args.max_planets or int(cfg.get("max_planets", 64))
    keys = payload["keys"]
    acted = payload.get("acted_indices")
    if acted is None:
        acted = [i for i, s in enumerate(payload["snapshots"])
                 if bool(s["pair_labels"].any())]
    acted_keys = {keys[i] for i in acted}
    turns_by_ep: dict[str, set] = {}
    for (ep, t) in acted_keys:
        turns_by_ep.setdefault(ep, set()).add(int(t))
    players = list(cfg.get("players") or [])
    _log(f"P={P} | acted snapshots={len(acted_keys)} | episodes={len(turns_by_ep)} "
         f"| players={players}")

    replays: list[Path] = []
    for player in players:
        pdir = args.replay_dir / player
        if not pdir.is_dir():
            _log(f"WARN no replay dir for player {player!r} at {pdir}")
            continue
        replays += sorted(pdir.glob("*.json.gz"))
    if args.limit_replays:
        replays = replays[: args.limit_replays]
    if args.num_shards > 1:
        replays = [r for i, r in enumerate(replays)
                   if i % args.num_shards == args.shard]
        _log(f"shard {args.shard}/{args.num_shards}: {len(replays)} replay files")
    else:
        _log(f"{len(replays)} replay files to scan")

    store: dict[tuple, torch.Tensor] = {}
    n_proc = n_with = n_pairs = 0
    seen_eps: set = set()
    t0 = time.time()
    for ri, rp in enumerate(replays):
        stem = rp.name[: -len(".json.gz")]
        seat = _parse_seat_from_stem(stem)
        if seat is None:
            continue
        try:
            with gzip.open(rp, "rt") as fh:
                replay = json.load(fh)
        except Exception as e:  # noqa: BLE001
            _log(f"  {stem}: load error {e}; skip")
            continue
        ep = replay.get("id") or stem.split(".")[0]
        want = turns_by_ep.get(ep)
        if not want:
            continue
        seen_eps.add(ep)
        steps = replay.get("steps") or []
        for t in sorted(want):
            if t >= len(steps) or not steps[t]:
                continue
            seat_obj = steps[t][seat] if len(steps[t]) > seat else steps[t][0]
            obs = seat_obj.get("observation") if isinstance(seat_obj, dict) else None
            if not obs:
                continue
            try:
                doomed = doomed_for_obs(obs, seat, P, t)
            except Exception:  # noqa: BLE001  (a bad snapshot must not kill the pass)
                continue
            n_proc += 1
            nd = int(doomed.sum())
            if nd:
                store[(ep, t)] = doomed.to(torch.bool)
                n_with += 1
                n_pairs += nd
        if (ri + 1) % 20 == 0 or (ri + 1) == len(replays):
            dt = time.time() - t0
            _log(f"  {ri + 1}/{len(replays)} replays | acted-snaps {n_proc} | "
                 f"w/doomed {n_with} ({100 * n_with / max(1, n_proc):.0f}%) | "
                 f"doomed-pairs {n_pairs} | {n_proc / max(dt, 1e-3):.1f} snap/s")

    missing = set(turns_by_ep) - seen_eps
    _log(f"DONE: labeled {n_proc} acted snapshots | {n_with} have >=1 doomed "
         f"({100 * n_with / max(1, n_proc):.1f}%) | {n_pairs} doomed pairs total "
         f"| {len(missing)} episodes had no matching replay")
    out = {
        "doomed": store,
        "P": P,
        "n_acted_labeled": n_proc,
        "n_with_doomed": n_with,
        "pair_cache": str(args.pair_cache_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    _log(f"saved sidecar -> {args.out}  ({len(store)} non-empty entries)")


if __name__ == "__main__":
    main()
