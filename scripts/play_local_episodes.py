"""Play local matches and dump them in the Kaggle replay format.

Goal: bulk-up the encoder pretraining set with self-play / mixed-agent
games we control, in addition to the Kaggle-pulled replays under
``data/replays/<team>/``.

Each match is written to
``data/replays/local/<matchup>/<runid>_<n>_<winner>.json.gz`` so the
existing dataset builder (``scripts/build_encoder_dataset.py``) picks
them up via ``rglob("*.json.gz")`` exactly like real replays.

Default schedule produces 30 matches:
  * 15 × 2-player random_v1 vs physical_v4 (seat permutations)
  * 15 × 4-player {random_v1 ×2, physical_v4 ×2} (seat permutations)

Run:
    python scripts/play_local_episodes.py --num-2p 15 --num-4p 15
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.runner import run_match  # noqa: E402

REPLAY_OUT = REPO / "data" / "replays" / "local"


def _serialize_env(env, agent_ids: list[str], result) -> dict:
    """Build a JSON-serializable dict mirroring the Kaggle replay shape.

    The fields the dataset builder actually reads are ``steps`` (list of
    per-seat dicts with ``observation``) and the top-level ``id``. Other
    fields are filled in for fidelity but are not load-bearing for
    pretraining.
    """
    # ``env.toJSON()`` returns the canonical Kaggle replay dict but
    # python-side it can include non-JSON-safe fragments (numpy scalars
    # in observations). Round-trip through json to normalize.
    raw = env.toJSON()
    text = json.dumps(raw, default=lambda o: getattr(o, "tolist", lambda: str(o))())
    payload = json.loads(text)
    payload["id"] = result.run_id if hasattr(result, "run_id") else None
    payload["agent_ids"] = agent_ids
    payload["winner_seat"] = result.winner
    payload["scores_final"] = result.scores[-1] if result.scores else []
    return payload


def _matchup_dir(agent_ids: list[str]) -> str:
    """Stable, filesystem-safe matchup name; e.g. ``rnd-phy-2p``."""
    short = {"random_v1": "rnd", "physical_v4": "phy"}
    parts = [short.get(a, a) for a in agent_ids]
    n = len(agent_ids)
    return "_".join(parts) + f"_{n}p"


def _episode_id(seed: int, n_players: int, winner: int) -> str:
    """Mimic Kaggle's ``<eid>_<n>_<seat>`` stem so build_encoder_dataset
    treats locally generated games like real replays."""
    return f"local{seed:08d}_{n_players}_{winner}"


def play_and_save(
    agent_ids: list[str],
    *,
    seed: int,
    out_root: Path,
) -> Path:
    matchup = _matchup_dir(agent_ids)
    out_dir = out_root / matchup
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = run_match(agent_ids, seed=seed)
    payload = _serialize_env(result.env, agent_ids, result)
    eid = _episode_id(seed, len(agent_ids), result.winner)
    payload["id"] = eid
    out_path = out_dir / f"{eid}.json.gz"
    with gzip.open(out_path, "wt") as fh:
        json.dump(payload, fh)
    print(
        f"[match] {matchup}  seed={seed}  winner=p{result.winner}  "
        f"steps={len(result.env.steps)}  ({time.time()-t0:.1f}s)  → {out_path.name}"
    )
    return out_path


def schedule_2p(n: int, base_seed: int) -> list[tuple[list[str], int]]:
    """Alternate seat orders so neither agent always starts in seat 0."""
    schedule = []
    permutations = [["random_v1", "physical_v4"], ["physical_v4", "random_v1"]]
    for i in range(n):
        schedule.append((permutations[i % 2], base_seed + i))
    return schedule


def schedule_4p(n: int, base_seed: int) -> list[tuple[list[str], int]]:
    """Cycle through distinct seat orderings of {rnd, rnd, phy, phy}."""
    pool = ["random_v1", "random_v1", "physical_v4", "physical_v4"]
    perms = sorted({tuple(p) for p in itertools.permutations(pool)})
    schedule = []
    for i in range(n):
        schedule.append((list(perms[i % len(perms)]), base_seed + 1000 + i))
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-2p", type=int, default=15)
    parser.add_argument("--num-4p", type=int, default=15)
    parser.add_argument("--base-seed", type=int, default=10_000)
    parser.add_argument("--out-root", type=Path, default=REPLAY_OUT)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    sched = schedule_2p(args.num_2p, args.base_seed) + schedule_4p(
        args.num_4p, args.base_seed
    )
    print(
        f"[schedule] {len(sched)} matches "
        f"({args.num_2p}×2p + {args.num_4p}×4p) → {args.out_root}"
    )
    written = []
    t0 = time.time()
    for agents, seed in sched:
        try:
            path = play_and_save(agents, seed=seed, out_root=args.out_root)
            written.append(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] seed={seed} agents={agents}: {exc!r}")
    print(f"\n[done] wrote {len(written)} replays in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
