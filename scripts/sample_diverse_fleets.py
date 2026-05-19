"""Sample N fleet rows balanced across (source_planet_id × heading_sector).

The fleet CSVs under ``data/datasets/fleet/`` contain ~5.5 M rows (one per
(fleet, turn) snapshot across 1128 episodes). Training the fleet encoder
on a random uniform sample biases toward common launch corridors and
heading bands. This script stratifies by **(source_planet_id, heading_sector)**
so the chosen 10 k rows cover diverse initial spots × diverse angles, then
writes the subset to one CSV that the fleet pretrainer can read directly
(via ``--data-dir``).

Usage:

    python -m scripts.sample_diverse_fleets \\
        --src-dir data/datasets/fleet \\
        --out-dir data/datasets/fleet_diverse_10k \\
        --n-rows 10000

The output dir holds a single ``fleet_diverse_10k.csv`` and a sidecar
``manifest.json`` recording the per-bucket sampling counts so the
selection is auditable / reproducible (with ``--seed``).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _index_rows(
    src_dir: Path,
    *,
    compute_will_hit_sun: bool = False,
) -> tuple[list[tuple[Path, int, int, int, int]], list[str]]:
    """Stream every fleet CSV and return ``[(path, row_idx, source_planet_id,
    heading_sector, label), ...]`` plus the canonical header field list.

    ``label`` is ``0`` when ``compute_will_hit_sun`` is False; otherwise it
    is the ray-vs-sun binary computed on the fly from the f000..f003
    feature columns (the existing CSVs don't carry the column).

    ``row_idx`` is the 0-based row position **after the header**, so the
    caller can re-open the file and skip ``row_idx`` lines to retrieve the
    original record byte-for-byte.
    """
    if compute_will_hit_sun:
        # Import lazily so the module still works in environments without
        # torch / our agent code in sys.path.
        from agents.transformer_v2.featurizer.fleet_featurizer import (
            BOARD, MAX_SPEED, _will_hit_sun,
        )
    csvs = sorted(src_dir.glob("fleet_*.csv"))
    if not csvs:
        raise RuntimeError(f"no fleet_*.csv under {src_dir}")
    header: list[str] | None = None
    rows: list[tuple[Path, int, int, int, int]] = []
    t0 = time.time()
    for i, p in enumerate(csvs):
        with p.open() as fh:
            reader = csv.DictReader(fh)
            if header is None:
                header = list(reader.fieldnames or [])
                for required in ("source_planet_id", "heading_sector"):
                    if required not in header:
                        raise RuntimeError(
                            f"{p} missing column {required!r}; expected schema "
                            f"includes {required}."
                        )
            for j, row in enumerate(reader):
                src_pid = row.get("source_planet_id")
                head = row.get("heading_sector")
                if src_pid is None or head is None:
                    continue
                try:
                    src_pid_i = int(src_pid)
                    head_i = int(head)
                except ValueError:
                    continue
                label = 0
                if compute_will_hit_sun:
                    try:
                        x = float(row["f000"]) * BOARD
                        y = float(row["f001"]) * BOARD
                        vx = float(row["f002"]) * MAX_SPEED
                        vy = float(row["f003"]) * MAX_SPEED
                    except (KeyError, ValueError):
                        continue
                    label = _will_hit_sun(x, y, vx, vy)
                rows.append((p, j, src_pid_i, head_i, label))
        if (i + 1) % 100 == 0:
            print(
                f"  [{i+1:4d}/{len(csvs)}] cumulative rows={len(rows):,} "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )
    assert header is not None
    print(
        f"indexed {len(rows):,} rows across {len(csvs)} files "
        f"in {time.time()-t0:.1f}s",
        flush=True,
    )
    return rows, header


def _stratified_sample(
    rows: list[tuple[Path, int, int, int, int]],
    n_target: int,
    rng: random.Random,
    *,
    only_label: int | None = None,
) -> tuple[list[tuple[Path, int]], dict[tuple[int, int], int]]:
    """Stratify by (source_planet_id, heading_sector), then sample with
    round-robin balance so under-represented buckets are not crushed by
    common corridors.

    ``only_label`` filters ``rows`` to entries whose 5th field (label)
    matches before bucketing. Passing ``None`` ignores the label.

    Returns the chosen ``(path, row_idx)`` list plus a per-bucket count
    dict for the manifest.
    """
    if only_label is not None:
        rows = [r for r in rows if r[4] == only_label]
    buckets: dict[tuple[int, int], list[tuple[Path, int]]] = defaultdict(list)
    for p, ridx, src_pid, head, _label in rows:
        buckets[(src_pid, head)].append((p, ridx))

    # Shuffle each bucket so round-robin draws are uniform within bucket.
    for k in buckets:
        rng.shuffle(buckets[k])

    chosen: list[tuple[Path, int]] = []
    counts: dict[tuple[int, int], int] = defaultdict(int)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    exhausted: set[tuple[int, int]] = set()
    while len(chosen) < n_target and len(exhausted) < len(keys):
        for k in keys:
            if k in exhausted:
                continue
            bucket = buckets[k]
            if not bucket:
                exhausted.add(k)
                continue
            chosen.append(bucket.pop())
            counts[k] += 1
            if len(chosen) >= n_target:
                break
    if len(chosen) < n_target:
        print(
            f"warning: only {len(chosen)} rows available across "
            f"{len(buckets)} buckets — target {n_target} not reachable",
            flush=True,
        )
    return chosen, dict(counts)


def _write_subset(
    chosen: list[tuple[Path, int]],
    header: list[str],
    out_csv: Path,
) -> None:
    """Re-read each chosen row and concatenate into one CSV."""
    # Group by file so we re-open each at most once.
    by_file: dict[Path, set[int]] = defaultdict(set)
    for p, ridx in chosen:
        by_file[p].add(ridx)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_csv.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=header)
        writer.writeheader()
        for p, idx_set in by_file.items():
            with p.open() as in_fh:
                reader = csv.DictReader(in_fh)
                for j, row in enumerate(reader):
                    if j in idx_set:
                        writer.writerow(row)
                        written += 1
    print(f"wrote {written:,} rows to {out_csv}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, default=Path("data/datasets/fleet"))
    ap.add_argument(
        "--out-dir", type=Path, default=Path("data/datasets/fleet_diverse_10k"),
    )
    ap.add_argument("--n-rows", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument(
        "--balance-will-hit-sun-frac",
        type=float,
        default=None,
        help=(
            "If set (e.g. 0.4), force the chosen rows to contain at least "
            "this fraction of will_hit_sun=1 positives. Positives + "
            "negatives are each stratified by (source_planet_id × "
            "heading_sector) before being mixed."
        ),
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    balance = args.balance_will_hit_sun_frac
    rows, header = _index_rows(
        args.src_dir, compute_will_hit_sun=balance is not None,
    )

    if balance is not None:
        if not (0.0 < balance < 1.0):
            raise ValueError(
                f"--balance-will-hit-sun-frac must be in (0, 1), got {balance}"
            )
        n_pos = int(round(args.n_rows * balance))
        n_neg = args.n_rows - n_pos
        n_pos_avail = sum(1 for r in rows if r[4] == 1)
        n_neg_avail = sum(1 for r in rows if r[4] == 0)
        print(
            f"target mix: positives={n_pos} negatives={n_neg} "
            f"(available pos={n_pos_avail} neg={n_neg_avail})",
            flush=True,
        )
        pos_rows, pos_counts = _stratified_sample(
            rows, n_pos, rng, only_label=1,
        )
        neg_rows, neg_counts = _stratified_sample(
            rows, n_neg, rng, only_label=0,
        )
        chosen = pos_rows + neg_rows
        rng.shuffle(chosen)
        counts = {**pos_counts, **neg_counts}  # buckets disjoint in label space anyway
        print(
            f"chose positives={len(pos_rows)} negatives={len(neg_rows)} "
            f"(actual pos frac={len(pos_rows)/max(1,len(chosen)):.3f})",
            flush=True,
        )
    else:
        chosen, counts = _stratified_sample(rows, args.n_rows, rng)

    n_buckets = len(counts)
    cs = sorted(counts.values(), reverse=True)
    print(
        f"chose {len(chosen):,} rows across {n_buckets} "
        f"(source_planet_id × heading_sector) buckets",
        flush=True,
    )
    if cs:
        print(
            f"  per-bucket counts — max={cs[0]} median={cs[len(cs)//2]} min={cs[-1]}",
            flush=True,
        )

    out_csv = args.out_dir / "fleet_diverse.csv"
    _write_subset(chosen, header, out_csv)

    manifest = {
        "src_dir": str(args.src_dir),
        "out_csv": str(out_csv),
        "n_rows": len(chosen),
        "n_buckets": n_buckets,
        "seed": args.seed,
        "balance_will_hit_sun_frac": balance,
        "bucket_counts": {
            f"src{p}_head{h}": c for (p, h), c in sorted(counts.items())
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest to {args.out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
