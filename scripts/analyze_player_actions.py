"""Per-player launch statistics from action CSVs + replay observations.

For every player under ``data/replays/<player>/``, walks each replay
(``<stem>.json.gz``) plus its matching ``data/datasets/action/action_<stem>.csv``
and pools per-launch statistics. Source positions and target positions
come from ``obs.planets[idx]`` at the action's turn — featurized planet
CSVs don't expose raw coords, but the replay JSON does.

A "source set" is the set of distinct source planets a player used in
one replay; this script reports both within-replay cardinality and
per-source aggregate stats (how many launches per source, fleet-size
spread per source, etc.).

Per-player aggregates (printed + saved as JSON):

  Per-replay totals (one row per replay, aggregated over replays):
    - total_launches             (count of acted turns)
    - unique_sources             (cardinality of the source set)
    - unique_targets
    - max_launches_per_source    (most-used source in that replay)

  Per-launch distributions (pooled across all acted rows for the player):
    - src_tgt_distance           (Euclidean between source and target planets)
    - ships_sent                 (when present in CSV — only post-frac-featurizer runs)
    - frac_label                 (ships_sent / source_ships_before, ditto)

  Per-source within-replay distributions (one row per (replay, source)):
    - launches_per_source        (count)
    - ships_std_per_source       (std of ships across launches from that source)
    - dist_mean_per_source       (mean src→tgt distance from that source)

Run:
    python scripts/analyze_player_actions.py
    python scripts/analyze_player_actions.py --player Ebi --player flg
    python scripts/analyze_player_actions.py --out data/runs/player_action_stats.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPLAY_DIR = REPO / "data" / "replays"
ACTION_DIR = REPO / "data" / "datasets" / "action"

# Planet tuple field indices in obs.planets[i]. Mirrors physics_utils.
P_X = 2
P_Y = 3


def _summary(vals: list[float]) -> dict[str, float]:
    """Compute mean/median/std/min/max for one metric column. NaN
    stand-ins keep the JSON shape stable when a player has no rows
    contributing to a metric (e.g. kovi with no frac_label)."""
    if not vals:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "std": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _parse_seat(stem: str) -> int | None:
    """``<episode>_<n>_<seat>`` → ``seat`` (int). Returns None for stems
    that don't end in a digit (e.g. local-test files)."""
    try:
        return int(stem.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _read_action_rows(csv_path: Path) -> list[dict]:
    """Return acted-only rows with parsed numeric fields. Missing columns
    (kovi/Shun_PI/etc. predate the frac-label featurizer) are silently
    skipped — ``ships_sent`` / ``frac_label`` become None for those rows."""
    out: list[dict] = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("expert_acted") != "1":
                continue
            try:
                turn = int(row["turn"])
                src_idx = int(row["source_planet_idx"])
                tgt_idx = int(row["target_planet_idx"])
            except (KeyError, ValueError):
                continue
            if src_idx < 0 or tgt_idx < 0:
                continue
            ships_raw = row.get("ships_sent") or ""
            frac_raw = row.get("frac_label") or ""
            ships = int(ships_raw) if ships_raw else None
            try:
                frac = float(frac_raw) if frac_raw else None
            except ValueError:
                frac = None
            out.append({
                "turn": turn,
                "src": src_idx,
                "tgt": tgt_idx,
                "ships": ships,
                "frac": frac,
            })
    return out


def _replay_planet_xy(
    replay_path: Path,
    seat: int,
    turn: int,
    src_idx: int,
    tgt_idx: int,
    _step_cache: dict[Path, list],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return ``((sx, sy), (tx, ty))`` for the launch's source and
    target planet, read from the replay's per-turn observation. Uses a
    per-replay cache so we deserialize each gzip once."""
    steps = _step_cache.get(replay_path)
    if steps is None:
        try:
            with gzip.open(replay_path, "rt") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            _step_cache[replay_path] = []
            return None
        steps = payload.get("steps") or []
        _step_cache[replay_path] = steps
    if turn >= len(steps):
        return None
    step = steps[turn]
    if not step or seat >= len(step):
        return None
    seat_view = step[seat]
    obs = seat_view.get("observation") if isinstance(seat_view, dict) else None
    if not obs:
        return None
    planets = obs.get("planets") or []
    if src_idx >= len(planets) or tgt_idx >= len(planets):
        return None
    src = planets[src_idx]
    tgt = planets[tgt_idx]
    return (float(src[P_X]), float(src[P_Y])), (float(tgt[P_X]), float(tgt[P_Y]))


def analyze_player(player: str) -> dict | None:
    """Walk ``data/replays/<player>/*.json.gz`` + matching action CSVs
    and compute the pooled / per-replay / per-source stats described in
    the module docstring."""
    pdir = REPLAY_DIR / player
    if not pdir.is_dir():
        print(f"[{player}] no replay dir; skip", flush=True)
        return None
    replays = sorted(pdir.glob("*.json.gz"))
    if not replays:
        print(f"[{player}] 0 replays; skip", flush=True)
        return None
    print(f"[{player}] {len(replays)} replays", flush=True)

    # Per-replay totals
    total_launches: list[float] = []
    unique_sources: list[float] = []
    unique_targets: list[float] = []
    max_per_source: list[float] = []

    # Per-launch pooled
    dist: list[float] = []
    ships: list[float] = []
    fracs: list[float] = []

    # Per-(replay, source) within-replay
    per_source_n: list[float] = []
    per_source_ships_std: list[float] = []
    per_source_dist_mean: list[float] = []

    replays_with_rows = 0

    for r in replays:
        stem = r.name[: -len(".json.gz")]
        seat = _parse_seat(stem)
        if seat is None:
            continue
        action_csv = ACTION_DIR / f"action_{stem}.csv"
        if not action_csv.exists():
            continue
        rows = _read_action_rows(action_csv)
        if not rows:
            continue
        replays_with_rows += 1
        step_cache: dict = {}

        per_replay_dists: list[float] = []
        # source_idx → list of (ships_or_None, dist_or_None)
        by_source: dict[int, list[tuple[float | None, float | None]]] = defaultdict(list)
        sources_seen: set[int] = set()
        targets_seen: set[int] = set()

        for row in rows:
            sources_seen.add(row["src"])
            targets_seen.add(row["tgt"])
            xy = _replay_planet_xy(r, seat, row["turn"], row["src"], row["tgt"], step_cache)
            d = None
            if xy is not None:
                (sx, sy), (tx, ty) = xy
                d = math.hypot(tx - sx, ty - sy)
                dist.append(d)
                per_replay_dists.append(d)
            if row["ships"] is not None:
                ships.append(float(row["ships"]))
            if row["frac"] is not None and math.isfinite(row["frac"]):
                fracs.append(row["frac"])
            by_source[row["src"]].append((row["ships"], d))

        total_launches.append(len(rows))
        unique_sources.append(len(sources_seen))
        unique_targets.append(len(targets_seen))
        max_per_source.append(max((len(v) for v in by_source.values()), default=0))

        for src, entries in by_source.items():
            per_source_n.append(len(entries))
            src_ships = [s for s, _ in entries if s is not None]
            src_dists = [dd for _, dd in entries if dd is not None]
            if len(src_ships) >= 2:
                per_source_ships_std.append(statistics.pstdev(src_ships))
            if src_dists:
                per_source_dist_mean.append(statistics.fmean(src_dists))

    if replays_with_rows == 0:
        print(f"[{player}] no replays had acted action rows; skip", flush=True)
        return None

    return {
        "player": player,
        "replays": len(replays),
        "replays_with_action_rows": replays_with_rows,
        "per_replay": {
            "total_launches": _summary(total_launches),
            "unique_sources": _summary(unique_sources),
            "unique_targets": _summary(unique_targets),
            "max_launches_per_source": _summary(max_per_source),
        },
        "per_launch": {
            "src_tgt_distance": _summary(dist),
            "ships_sent": _summary(ships),
            "frac_label": _summary(fracs),
        },
        "per_source": {
            "launches_per_source": _summary(per_source_n),
            "ships_std_per_source": _summary(per_source_ships_std),
            "dist_mean_per_source": _summary(per_source_dist_mean),
        },
    }


def _print_table(reports: list[dict]) -> None:
    """Compact comparison table: one column per player, one row per metric."""
    if not reports:
        return
    headers = [r["player"] for r in reports]
    rows: list[tuple[str, list[str]]] = []
    rows.append(("replays", [str(r["replays"]) for r in reports]))
    rows.append((
        "replays w/ actions",
        [str(r["replays_with_action_rows"]) for r in reports],
    ))

    def cell(d: dict, key: str) -> str:
        m = d["mean"]
        sd = d["std"]
        n = d["n"]
        if not n or math.isnan(m):
            return "—"
        if key == "frac_label":
            return f"{m:.3f}±{sd:.3f}"
        if key in ("ships_sent", "ships_std_per_source"):
            return f"{m:.0f}±{sd:.0f}"
        return f"{m:.1f}±{sd:.1f}"

    def emit(label: str, group: str, key: str) -> None:
        rows.append((label, [cell(r[group][key], key) for r in reports]))

    emit("launches/replay", "per_replay", "total_launches")
    emit("unique_sources/replay", "per_replay", "unique_sources")
    emit("unique_targets/replay", "per_replay", "unique_targets")
    emit("max_launches/source/replay", "per_replay", "max_launches_per_source")
    emit("src→tgt distance", "per_launch", "src_tgt_distance")
    emit("ships_sent / launch", "per_launch", "ships_sent")
    emit("frac (ships/source)", "per_launch", "frac_label")
    emit("launches per source", "per_source", "launches_per_source")
    emit("ships std per source", "per_source", "ships_std_per_source")
    emit("dist mean per source", "per_source", "dist_mean_per_source")

    label_w = max(len(label) for label, _ in rows)
    col_w = max(len(h) for h in headers + sum(([c for c in cells] for _, cells in rows), []))
    col_w = max(col_w, 14)
    print()
    print(" " * label_w + "  " + "  ".join(f"{h:>{col_w}}" for h in headers))
    for label, cells in rows:
        print(f"{label:<{label_w}}  " + "  ".join(f"{c:>{col_w}}" for c in cells))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player", action="append",
                   help="Restrict to one or more players. Default: every "
                        "directory under data/replays/.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON path for the full per-player breakdown.")
    args = p.parse_args()

    if args.player:
        players = args.player
    else:
        players = sorted(
            d.name for d in REPLAY_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_") and d.name != "local"
        )

    reports: list[dict] = []
    for player in players:
        r = analyze_player(player)
        if r is not None:
            reports.append(r)

    _print_table(reports)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2))
        print(f"\n[out] wrote per-player JSON → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
