"""For each replay, report the max inbound and max outbound fleet count
hitting/leaving any single planet at any moment.

  outbound[p]: count of in-flight fleets whose source planet id == p
  inbound[p]:  count of in-flight fleets whose straight-line trajectory
               first collides with planet p (computed via per-fleet ETA;
               minimum positive ETA across planets wins, or None if the
               fleet's trajectory misses every planet)

Both are per-step counts; we report the per-episode maximum across all
steps and all planets.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.physics_utils import _fleet_eta_to_planet, fleet_speed  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "replays"

# Indices into the env's planet/fleet tuples.
P_ID, P_X, P_Y, P_RADIUS = 0, 2, 3, 4
F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = 1, 2, 3, 4, 5, 6


def first_hit_planet_id(fleet, planets) -> int | None:
    """Return the planet_id whose disc the fleet's trajectory first crosses,
    or None if no planet is on the line."""
    best_eta = float("inf")
    best_pid: int | None = None
    for p in planets:
        eta = _fleet_eta_to_planet(fleet, p)
        if eta is None:
            continue
        if eta < best_eta:
            best_eta = eta
            best_pid = p[P_ID]
    return best_pid


def episode_fleet_pressure(path: Path) -> tuple[int, int, int, int, int]:
    """Return (max_inbound, max_outbound, max_inbound_step,
               max_outbound_step, n_steps)."""
    with gzip.open(path, "rb") as f:
        d = json.load(f)
    steps = d.get("steps") or []
    max_in = 0
    max_in_step = -1
    max_out = 0
    max_out_step = -1

    for t, step in enumerate(steps):
        if not step:
            continue
        obs = step[0].get("observation") if isinstance(step[0], dict) else None
        if not obs:
            continue
        fleets = obs.get("fleets") or []
        planets = obs.get("planets") or []
        if not fleets or not planets:
            continue

        outbound: Counter[int] = Counter()
        inbound: Counter[int] = Counter()
        for f in fleets:
            outbound[f[F_FROM]] += 1
            tgt = first_hit_planet_id(f, planets)
            if tgt is not None:
                inbound[tgt] += 1

        if outbound:
            mx_out = max(outbound.values())
            if mx_out > max_out:
                max_out = mx_out
                max_out_step = t
        if inbound:
            mx_in = max(inbound.values())
            if mx_in > max_in:
                max_in = mx_in
                max_in_step = t

    return max_in, max_out, max_in_step, max_out_step, len(steps)


def main() -> None:
    files = sorted(DATA_DIR.rglob("*.json.gz"))
    print(f"scanning {len(files)} replays under {DATA_DIR}")

    rows: list[tuple[Path, int, int, int, int, int]] = []
    for p in files:
        try:
            mi, mo, mi_t, mo_t, n = episode_fleet_pressure(p)
        except Exception as e:
            print(f"  skipped {p.relative_to(DATA_DIR.parent)}: {type(e).__name__}: {e}")
            continue
        rows.append((p, mi, mo, mi_t, mo_t, n))

    if not rows:
        print("no episodes parsed")
        return

    rows.sort(key=lambda r: -max(r[1], r[2]))

    # Top-20 hottest episodes by max(in, out)
    print()
    print(f"{'episode':<55s}  {'in':>4s}  {'@step':>5s}  {'out':>4s}  {'@step':>5s}  {'steps':>5s}")
    print("-" * 90)
    for p, mi, mo, mi_t, mo_t, n in rows[:20]:
        rel = p.relative_to(DATA_DIR.parent)
        print(f"{str(rel):<55s}  {mi:>4d}  {mi_t:>5d}  {mo:>4d}  {mo_t:>5d}  {n:>5d}")
    if len(rows) > 20:
        print(f"... ({len(rows) - 20} more)")

    # Distribution
    def stats(vals: list[int]) -> dict[str, float]:
        s = sorted(vals)
        n = len(s)
        def pct(p: float) -> int:
            return s[min(n - 1, int(p * (n - 1)))]
        return {
            "min": s[0], "p25": pct(0.25), "median": pct(0.50),
            "p75": pct(0.75), "p95": pct(0.95), "p99": pct(0.99),
            "max": s[-1], "mean": sum(s) / n,
        }

    ins = [r[1] for r in rows]
    outs = [r[2] for r in rows]
    print()
    print(f"{'stat':<8s}  {'inbound':>10s}  {'outbound':>10s}")
    print("-" * 36)
    si = stats(ins)
    so = stats(outs)
    for k in ("min", "p25", "median", "p75", "p95", "p99", "max", "mean"):
        v_in = f"{si[k]:.1f}" if k == "mean" else f"{si[k]}"
        v_out = f"{so[k]:.1f}" if k == "mean" else f"{so[k]}"
        print(f"{k:<8s}  {v_in:>10s}  {v_out:>10s}")


if __name__ == "__main__":
    main()
