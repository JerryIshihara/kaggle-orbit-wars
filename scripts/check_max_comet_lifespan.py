"""For each replay, report the longest comet lifespan in steps.

A "comet" here = a planet id that appears in ``obs['comet_planet_ids']``.
Comets come and go: a planet id can be a comet, vanish, and re-appear
later as a comet again — those count as two separate runs, NOT one.

Per episode we compute, for every planet id that was ever a comet, the
max length of its consecutive-step runs in ``comet_planet_ids``, then
take the max over all comet ids. That's the episode's longest comet
lifespan.

Output:
  * top-20 hottest episodes by max comet lifespan
  * distribution stats (min/p25/median/p75/p95/max/mean)
  * total comet runs encountered + average run length

Run from repo root:
    python scripts/check_max_comet_lifespan.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "replays"


def comet_runs(steps: list) -> list[tuple[int, int, int]]:
    """Return ``[(planet_id, start_step, end_step), ...]`` for every
    contiguous run of a planet id appearing in ``comet_planet_ids``.

    Both endpoints are inclusive. Runs are closed when a planet id
    drops out of the live set (or at the end of the episode).
    """
    runs: list[tuple[int, int, int]] = []
    starts: dict[int, int] = {}        # planet_id → start_step of current run
    prev_alive: set[int] = set()

    for t, step in enumerate(steps):
        if not step:
            continue
        obs = step[0].get("observation") if isinstance(step[0], dict) else None
        if not obs:
            continue
        alive = set(obs.get("comet_planet_ids") or [])

        # Comets that just appeared
        for pid in alive - prev_alive:
            starts[pid] = t
        # Comets that just disappeared — close their run at t-1
        for pid in prev_alive - alive:
            runs.append((pid, starts.pop(pid), t - 1))
        prev_alive = alive

    # Close out anything still alive at end-of-episode
    last_t = len(steps) - 1
    for pid, start in starts.items():
        runs.append((pid, start, last_t))

    return runs


def episode_stats(path: Path) -> tuple[int, int, int, int, int]:
    """Return (max_lifespan, max_pid, max_start_step, n_runs, n_steps)."""
    with gzip.open(path, "rb") as f:
        d = json.load(f)
    steps = d.get("steps") or []
    runs = comet_runs(steps)

    if not runs:
        return 0, -1, -1, 0, len(steps)

    best_pid, best_start, best_end = max(runs, key=lambda r: r[2] - r[1] + 1)
    best_len = best_end - best_start + 1
    return best_len, best_pid, best_start, len(runs), len(steps)


def _quantiles(vals: list[int]) -> dict[str, float]:
    s = sorted(vals)
    n = len(s)

    def pct(p: float) -> int:
        return s[min(n - 1, int(p * (n - 1)))]

    return {
        "min": s[0], "p25": pct(0.25), "median": pct(0.50),
        "p75": pct(0.75), "p95": pct(0.95), "max": s[-1],
        "mean": sum(s) / n,
    }


def main() -> None:
    files = sorted(DATA_DIR.rglob("*.json.gz"))
    print(f"scanning {len(files)} replay files under {DATA_DIR}")

    rows: list[tuple[Path, int, int, int, int, int]] = []
    total_runs = 0
    all_run_lens: list[int] = []

    for path in files:
        try:
            best_len, best_pid, best_start, n_runs, n_steps = episode_stats(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {path.relative_to(DATA_DIR.parent)}: "
                  f"{type(exc).__name__}: {exc}")
            continue
        rows.append((path, best_len, best_pid, best_start, n_runs, n_steps))
        total_runs += n_runs
        # Recompute all run lengths for the global distribution
        with gzip.open(path, "rb") as f:
            d = json.load(f)
        for _, s, e in comet_runs(d.get("steps") or []):
            all_run_lens.append(e - s + 1)

    if not rows:
        print("no episodes parsed")
        return

    # Top-20 hottest
    rows.sort(key=lambda r: -r[1])
    print()
    print(f"{'episode':<55s}  {'max':>4s}  {'pid':>4s}  {'start':>5s}  "
          f"{'runs':>5s}  {'steps':>5s}")
    print("-" * 90)
    for path, best, pid, start, n_runs, n_steps in rows[:20]:
        rel = path.relative_to(DATA_DIR.parent)
        print(f"{str(rel):<55s}  {best:>4d}  {pid:>4d}  {start:>5d}  "
              f"{n_runs:>5d}  {n_steps:>5d}")
    if len(rows) > 20:
        print(f"... ({len(rows) - 20} more)")

    # Per-episode max-lifespan distribution
    maxes = [r[1] for r in rows]
    qs = _quantiles(maxes)
    print()
    print("per-episode max comet lifespan (steps):")
    for k in ("min", "p25", "median", "p75", "p95", "max", "mean"):
        v = f"{qs[k]:.1f}" if k == "mean" else f"{qs[k]}"
        print(f"  {k:<8s}  {v}")

    # Run-length distribution across ALL comet runs in ALL episodes
    if all_run_lens:
        rqs = _quantiles(all_run_lens)
        print()
        print(f"all comet runs (n={len(all_run_lens)}) — length distribution:")
        for k in ("min", "p25", "median", "p75", "p95", "max", "mean"):
            v = f"{rqs[k]:.1f}" if k == "mean" else f"{rqs[k]}"
            print(f"  {k:<8s}  {v}")

    n_zero = sum(1 for r in rows if r[1] == 0)
    print()
    print(f"episodes with no comet at all: {n_zero}/{len(rows)}")


if __name__ == "__main__":
    main()
