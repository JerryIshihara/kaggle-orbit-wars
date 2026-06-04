#!/usr/bin/env python3
"""Preview the dormant clockwise observation seat-swap layer.

This script does not modify any training/eval path. It applies
``ClockwiseSeatSwap`` to either a synthetic smoke observation or selected
seat streams from replay files and prints before/after owner counts. Multiple
replay files can be processed across CPU cores with ``--workers``.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_clockwise_seat_swap():
    path = ROOT / "agents" / "transformer_v2" / "featurizer" / "swap.py"
    spec = importlib.util.spec_from_file_location("_obs_swap", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import swap layer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ClockwiseSeatSwap


ClockwiseSeatSwap = _load_clockwise_seat_swap()


def _load_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _steps(payload: Any) -> list:
    if isinstance(payload, dict) and "steps" in payload:
        return payload["steps"]
    if isinstance(payload, list):
        return payload
    raise ValueError("expected a Kaggle replay payload with a top-level steps list")


def _infer_replay_num_players(steps: list) -> int | None:
    for step in steps:
        if isinstance(step, list) and step:
            return len(step)
    return None


def _owner_counts(obs: dict[str, Any]) -> dict[int, int]:
    c: Counter[int] = Counter()
    for p in obs.get("planets") or []:
        c[int(p[1])] += 1
    return dict(sorted(c.items()))


def _fleet_owner_counts(obs: dict[str, Any]) -> dict[int, int]:
    c: Counter[int] = Counter()
    for f in obs.get("fleets") or []:
        c[int(f[1])] += 1
    return dict(sorted(c.items()))


def _first_planet_xy(obs: dict[str, Any]) -> tuple[float, float] | None:
    planets = obs.get("planets") or []
    if not planets:
        return None
    row = planets[0]
    try:
        return (round(float(row[2]), 3), round(float(row[3]), 3))
    except (TypeError, ValueError, IndexError):
        return None


def _print_preview(obs: dict[str, Any], swapped: dict[str, Any], turn: int) -> None:
    print(_preview_line(obs, swapped, turn))


def _preview_line(obs: dict[str, Any], swapped: dict[str, Any], turn: int) -> str:
    return (
        f"turn={turn} player {obs.get('player')} -> {swapped.get('player')} | "
        f"planet owners {_owner_counts(obs)} -> {_owner_counts(swapped)} | "
        f"fleet owners {_fleet_owner_counts(obs)} -> {_fleet_owner_counts(swapped)} | "
        f"p0_xy {_first_planet_xy(obs)} -> {_first_planet_xy(swapped)}"
    )


def _synthetic_obs(step: int = 0) -> dict[str, Any]:
    return {
        "step": step,
        "player": 0,
        "planets": [
            [10, 0, 10.0, 10.0, 3.0, 50, 1],
            [11, 1, 90.0, 10.0, 3.0, 50, 1],
            [12, 2, 90.0, 90.0, 3.0, 50, 1],
            [13, 3, 10.0, 90.0, 3.0, 50, 1],
            [14, -1, 50.0, 50.0, 3.0, 20, 0],
        ],
        "initial_planets": [
            [10, 0, 10.0, 10.0, 3.0, 50, 1],
            [11, 1, 90.0, 10.0, 3.0, 50, 1],
            [12, 2, 90.0, 90.0, 3.0, 50, 1],
            [13, 3, 10.0, 90.0, 3.0, 50, 1],
        ],
        "fleets": [
            [100, 0, 20.0, 20.0, 0.0, 10, 12],
            [101, 3, 80.0, 80.0, 3.14, 13, 7],
        ],
        "comet_planet_ids": [],
        "comets": [],
    }


def _preview_replay_file(args: tuple[Path, int, int, int | None]) -> list[str]:
    """Worker entry point: preview one replay file, returning printable lines."""
    replay, seat, turns, num_players = args
    lines = [f"replay={replay}"]
    steps = _steps(_load_json(replay))
    resolved_num_players = (
        num_players if num_players is not None else _infer_replay_num_players(steps)
    )
    swap = ClockwiseSeatSwap(
        num_players=resolved_num_players,
    )
    map_printed = False
    for turn, step in enumerate(steps[: max(0, turns)]):
        if seat >= len(step):
            raise IndexError(f"seat {seat} missing at turn {turn} in {replay}")
        seat_row = step[seat]
        obs = seat_row.get("observation") if isinstance(seat_row, dict) else None
        if not obs:
            continue
        swapped = swap.apply(obs)
        if not map_printed and swap.owner_map is not None:
            lines.append(
                f"owner_map={swap.owner_map} rotation_radians={swap.rotation_radians}"
            )
            map_printed = True
        lines.append(_preview_line(obs, swapped, turn))
    return lines


def _resolve_workers(requested: int, n_jobs: int) -> int:
    if n_jobs <= 0:
        return 1
    if requested <= 0:
        requested = os.cpu_count() or 1
    return max(1, min(int(requested), n_jobs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--self-test", action="store_true",
                     help="preview a synthetic 4-player observation")
    src.add_argument("--replay", type=Path, nargs="+",
                     help="one or more Kaggle replay .json/.json.gz files to preview")
    p.add_argument("--seat", type=int, default=0,
                   help="replay seat stream to preview")
    p.add_argument("--turns", type=int, default=5,
                   help="number of replay turns to print")
    p.add_argument("--num-players", type=int, default=None,
                   help="explicit player count; otherwise infer from first obs")
    p.add_argument("--workers", type=int, default=1,
                   help="CPU process workers for multiple replay files; 0 uses all cores")
    args = p.parse_args()

    swap = ClockwiseSeatSwap(num_players=args.num_players)
    if args.self_test:
        obs0 = _synthetic_obs(step=0)
        swapped0 = swap.apply(obs0)
        print(f"owner_map={swap.owner_map} rotation_radians={swap.rotation_radians}")
        _print_preview(obs0, swapped0, 0)
        obs1 = _synthetic_obs(step=1)
        swapped1 = swap.apply(obs1)
        _print_preview(obs1, swapped1, 1)
        return

    jobs = [(path, args.seat, args.turns, args.num_players) for path in args.replay]
    workers = _resolve_workers(args.workers, len(jobs))
    if workers > 1:
        print(f"workers={workers} replays={len(jobs)}")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for lines in pool.map(_preview_replay_file, jobs, chunksize=1):
                print("\n".join(lines))
    else:
        for job in jobs:
            print("\n".join(_preview_replay_file(job)))


if __name__ == "__main__":
    main()
