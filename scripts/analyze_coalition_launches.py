"""Per-player coalition analysis — multi-source-to-same-target launches.

For each player under ``data/replays/<player>/``, walks every replay
and detects turns where the player launched **two or more fleets at
the same target planet** in one turn ("coalitions" — what
``physical_v4``'s swarm logic is trying to mimic).

The existing action CSVs only record the *first* expert action per
turn, so they hide this signal. This script reads the raw replay JSONs
and reconstructs each launch by diffing ``obs[t].fleets`` against
``obs[t+1].fleets``: any new fleet owned by the player's seat is a
launch from ``F_FROM`` with target resolved via the same
:func:`_resolve_target` the featurizer uses.

For every (replay, turn, target) triple we count how many distinct
sources launched at that target on that turn — the "coalition size".
The per-player report aggregates that distribution.

Metrics reported per player:

  * launches_total              — every launch the seat made across all replays
  * launches_in_coalition       — launches that were part of a size≥2 group
  * coalition_share             — launches_in_coalition / launches_total
  * coalition_events            — distinct (replay, turn, target) groups with size≥2
  * coalition_size_*            — distribution over coalition sizes (mean, max, hist)
  * solo_launches               — single-source-on-target launches

A larger ``coalition_share`` means the player concentrates fleets on
shared targets that turn; a small share means launches are spread to
distinct targets each turn.

Run:
    python scripts/analyze_coalition_launches.py
    python scripts/analyze_coalition_launches.py --player Ebi --player Shun_PI
    python scripts/analyze_coalition_launches.py --out data/runs/coalition_stats.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agents.physics_utils import F_ID, F_FROM, F_OWNER  # noqa: E402
from agents.transformer_v1.featurizer.fleet_featurizer import _resolve_target  # noqa: E402

REPLAY_DIR = REPO / "data" / "replays"


def _parse_seat(stem: str) -> int | None:
    """Stem like ``75365996_2_0`` → seat ``0``."""
    try:
        return int(stem.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _diff_new_fleet_ids(prev_obs: dict, next_obs: dict) -> set[int]:
    """IDs that exist in ``next_obs.fleets`` but not in ``prev_obs.fleets``."""
    if not prev_obs:
        prev_ids: set[int] = set()
    else:
        prev_ids = {int(f[F_ID]) for f in (prev_obs.get("fleets") or [])}
    next_ids = {int(f[F_ID]) for f in (next_obs.get("fleets") or [])}
    return next_ids - prev_ids


def analyze_player(player: str) -> dict | None:
    """Pool launch coalitions across every replay belonging to ``player``."""
    pdir = REPLAY_DIR / player
    if not pdir.is_dir():
        return None
    replays = sorted(pdir.glob("*.json.gz"))
    if not replays:
        return None

    launches_total = 0
    solo_launches = 0
    coalition_launches = 0
    coalition_sizes: list[int] = []      # one entry per (replay, turn, tgt) group
    size_hist: Counter[int] = Counter()
    replays_processed = 0
    replays_skipped = 0

    for r in replays:
        stem = r.name[: -len(".json.gz")]
        seat = _parse_seat(stem)
        if seat is None:
            replays_skipped += 1
            continue
        try:
            with gzip.open(r, "rt") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            replays_skipped += 1
            continue
        steps = payload.get("steps") or []
        if not steps:
            replays_skipped += 1
            continue

        for t in range(len(steps) - 1):
            cur = steps[t]
            nxt = steps[t + 1]
            if not cur or not nxt or len(cur) <= seat or len(nxt) <= seat:
                continue
            cur_seat = cur[seat]
            nxt_seat = nxt[seat]
            cur_obs = cur_seat.get("observation") if isinstance(cur_seat, dict) else None
            nxt_obs = nxt_seat.get("observation") if isinstance(nxt_seat, dict) else None
            if not cur_obs or not nxt_obs:
                continue
            new_ids = _diff_new_fleet_ids(cur_obs, nxt_obs)
            if not new_ids:
                continue
            nxt_planets = nxt_obs.get("planets") or []
            # Group this turn's new fleets-by-target for this player only.
            target_to_sources: dict[int, set[int]] = defaultdict(set)
            for f in nxt_obs.get("fleets") or []:
                fid = int(f[F_ID])
                if fid not in new_ids:
                    continue
                if int(f[F_OWNER]) != seat:
                    continue
                tgt_pid, _eta, _planet = _resolve_target(f, nxt_planets)
                if tgt_pid is None:
                    continue
                target_to_sources[int(tgt_pid)].add(int(f[F_FROM]))
            for sources in target_to_sources.values():
                size = len(sources)
                launches_total += size
                size_hist[size] += 1
                coalition_sizes.append(size)
                if size == 1:
                    solo_launches += 1
                else:
                    coalition_launches += size
        replays_processed += 1

    if launches_total == 0:
        return None

    # Distribution stats on coalition-size groupings (one entry per group).
    mean_size = statistics.fmean(coalition_sizes) if coalition_sizes else 0.0
    max_size = max(coalition_sizes) if coalition_sizes else 0
    # Coalition size sweep: how many size-2 / 3 / 4+ events.
    hist = {str(k): int(v) for k, v in sorted(size_hist.items())}

    return {
        "player": player,
        "replays_processed": replays_processed,
        "replays_skipped": replays_skipped,
        "launches_total": launches_total,
        "solo_launches": solo_launches,
        "coalition_launches": coalition_launches,
        "coalition_share": coalition_launches / launches_total,
        "coalition_events": sum(v for k, v in size_hist.items() if k >= 2),
        "mean_coalition_size": float(mean_size),
        "max_coalition_size": int(max_size),
        "size_hist": hist,
    }


def _print_table(reports: list[dict]) -> None:
    if not reports:
        return
    headers = [r["player"] for r in reports]

    def pct(x: float) -> str:
        return f"{x * 100:5.1f}%"

    def hist_cell(h: dict[str, int]) -> str:
        """Compact 1/2/3/4/5+ histogram, e.g. '7402 / 312 / 14 / 0 / 0'."""
        buckets = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
        for k, v in h.items():
            ki = int(k)
            if ki <= 4:
                buckets[str(ki)] += int(v)
            else:
                buckets["5+"] += int(v)
        return " / ".join(f"{buckets[k]}" for k in ("1", "2", "3", "4", "5+"))

    rows = [
        ("replays processed", [str(r["replays_processed"]) for r in reports]),
        ("launches total", [f"{r['launches_total']:,}" for r in reports]),
        ("coalition events (size≥2)", [f"{r['coalition_events']:,}" for r in reports]),
        ("coalition share (of launches)", [pct(r["coalition_share"]) for r in reports]),
        ("mean coalition size", [f"{r['mean_coalition_size']:.2f}" for r in reports]),
        ("max coalition size", [str(r["max_coalition_size"]) for r in reports]),
        ("size 1 / 2 / 3 / 4 / 5+", [hist_cell(r["size_hist"]) for r in reports]),
    ]
    label_w = max(len(label) for label, _ in rows)
    col_w = max(
        max(len(h) for h in headers),
        max(max(len(c) for c in cells) for _, cells in rows),
        14,
    )
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
        print(f"[{player}] ...", flush=True)
        r = analyze_player(player)
        if r is not None:
            print(
                f"[{player}]   {r['launches_total']:,} launches, "
                f"{r['coalition_events']:,} coalitions (size≥2), "
                f"share={r['coalition_share']*100:.1f}%",
                flush=True,
            )
            reports.append(r)
        else:
            print(f"[{player}]   skipped (no replays or no launches)", flush=True)

    _print_table(reports)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2))
        print(f"\n[out] wrote per-player JSON → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
