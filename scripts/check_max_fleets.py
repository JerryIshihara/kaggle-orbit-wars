"""Walk every replay under data/replays/ and report the max fleet count
per episode. Replays are gzipped JSON in kaggle_environments format.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "replays"


def max_fleets_in_episode(path: Path) -> tuple[int, int, int]:
    """Return (max_fleets, step_at_max, total_steps) for one replay."""
    with gzip.open(path, "rb") as f:
        d = json.load(f)
    steps = d.get("steps") or []
    best = 0
    best_step = -1
    for t, step in enumerate(steps):
        if not step:
            continue
        # fleets list is identical across players; read from player 0.
        obs = step[0].get("observation") if isinstance(step[0], dict) else None
        if not obs:
            continue
        n = len(obs.get("fleets") or [])
        if n > best:
            best = n
            best_step = t
    return best, best_step, len(steps)


def main() -> None:
    files = sorted(DATA_DIR.rglob("*.json.gz"))
    print(f"scanning {len(files)} replay files under {DATA_DIR}")
    overall_max = 0
    overall_max_path: Path | None = None
    overall_max_step = -1
    rows: list[tuple[Path, int, int, int]] = []

    for p in files:
        try:
            mx, mx_step, n_steps = max_fleets_in_episode(p)
        except Exception as e:
            print(f"  skipped {p.relative_to(DATA_DIR.parent)}: {type(e).__name__}: {e}")
            continue
        rows.append((p, mx, mx_step, n_steps))
        if mx > overall_max:
            overall_max = mx
            overall_max_path = p
            overall_max_step = mx_step

    # Per-episode table, sorted by max desc, top 20.
    rows.sort(key=lambda r: -r[1])
    print()
    print(f"{'episode':<55s}  {'max':>4s}  {'@step':>5s}  {'steps':>5s}")
    print("-" * 78)
    for p, mx, mx_step, n_steps in rows[:20]:
        rel = p.relative_to(DATA_DIR.parent)
        print(f"{str(rel):<55s}  {mx:>4d}  {mx_step:>5d}  {n_steps:>5d}")
    if len(rows) > 20:
        print(f"... ({len(rows) - 20} more)")

    # Distribution.
    print()
    if rows:
        maxes = [r[1] for r in rows]
        maxes_sorted = sorted(maxes)
        n = len(maxes_sorted)

        def pct(p: float) -> int:
            return maxes_sorted[min(n - 1, int(p * (n - 1)))]

        print(f"distribution of max-fleets across {n} episodes:")
        print(f"  min   : {min(maxes)}")
        print(f"  p25   : {pct(0.25)}")
        print(f"  median: {pct(0.50)}")
        print(f"  p75   : {pct(0.75)}")
        print(f"  p95   : {pct(0.95)}")
        print(f"  p99   : {pct(0.99)}")
        print(f"  max   : {max(maxes)}")
        print(f"  mean  : {sum(maxes) / n:.1f}")

    if overall_max_path is not None:
        rel = overall_max_path.relative_to(DATA_DIR.parent)
        print()
        print(f"overall max: {overall_max} fleets at step {overall_max_step} in {rel}")


if __name__ == "__main__":
    main()
