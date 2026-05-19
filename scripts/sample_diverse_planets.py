"""Sample N planet rows balanced across (is_comet × distance_to_sun_bucket ×
speed_bucket).

The planet CSVs under ``data/datasets/planet/`` contain ~7.6 M rows
(every planet/comet on every turn of 1110 episodes). Training the
planet encoder on a uniform random sample biases toward common
orbits and heading bands. This script stratifies by **(is_comet,
distance_to_sun_bucket, speed_bucket)** so the chosen 20 k rows cover
diverse entity types × radial bands × speed bands, then writes
train/val/test splits + ``manifest.json`` the planet pretrainer can
read directly (via ``--data-dir``).

Usage:

    python -m scripts.sample_diverse_planets \\
        --src-dir data/datasets/planet \\
        --out-dir data/datasets/planet_diverse_20k \\
        --n-rows 20000

The output dir holds three split CSVs (``planet_train.csv`` /
``planet_val.csv`` / ``planet_test.csv``) and a manifest. The manifest
file format matches what ``agents.transformer_v2.pretrain.planet_encoder``
expects.
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


STRATIFY_COLS = ("is_comet", "distance_to_sun_bucket", "speed_bucket")


def _index_rows(src_dir: Path) -> tuple[list[tuple[Path, int, tuple[int, ...]]], list[str]]:
    """Stream every planet CSV. Returns ``[(path, row_idx, bucket_key)]``
    plus the canonical header field list.
    """
    csvs = sorted(src_dir.glob("planet_*.csv"))
    if not csvs:
        raise RuntimeError(f"no planet_*.csv under {src_dir}")
    header: list[str] | None = None
    rows: list[tuple[Path, int, tuple[int, ...]]] = []
    t0 = time.time()
    for i, p in enumerate(csvs):
        with p.open() as fh:
            reader = csv.DictReader(fh)
            if header is None:
                header = list(reader.fieldnames or [])
                for required in STRATIFY_COLS:
                    if required not in header:
                        raise RuntimeError(
                            f"{p} missing column {required!r}; expected schema "
                            f"includes {required}."
                        )
            for j, row in enumerate(reader):
                try:
                    key = tuple(int(row[c]) for c in STRATIFY_COLS)
                except (KeyError, ValueError):
                    continue
                rows.append((p, j, key))
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
    rows: list[tuple[Path, int, tuple[int, ...]]],
    n_target: int,
    rng: random.Random,
) -> tuple[list[tuple[Path, int]], dict[tuple[int, ...], int]]:
    """Round-robin pull from each bucket until ``n_target`` rows are
    chosen or every bucket is empty.
    """
    buckets: dict[tuple[int, ...], list[tuple[Path, int]]] = defaultdict(list)
    for p, ridx, key in rows:
        buckets[key].append((p, ridx))
    for k in buckets:
        rng.shuffle(buckets[k])

    keys = list(buckets.keys())
    rng.shuffle(keys)
    chosen: list[tuple[Path, int]] = []
    counts: dict[tuple[int, ...], int] = defaultdict(int)
    exhausted: set[tuple[int, ...]] = set()
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
    ap.add_argument("--src-dir", type=Path, default=Path("data/datasets/planet"))
    ap.add_argument(
        "--out-dir", type=Path, default=Path("data/datasets/planet_diverse_20k"),
    )
    ap.add_argument("--n-rows", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument(
        "--split-fractions",
        type=str,
        default="0.8,0.1,0.1",
        help="train,val,test fractions, comma-separated.",
    )
    ap.add_argument(
        "--filter-is-comet",
        type=int,
        choices=(0, 1),
        default=None,
        help="If set, restrict the sampling pool to is_comet==value rows "
             "(0 = planets only, 1 = comets only). When set, the "
             "(is_comet, …) bucket key collapses to (…) so stratification "
             "happens within the remaining axes.",
    )
    args = ap.parse_args()

    fracs = tuple(float(x) for x in args.split_fractions.split(","))
    if len(fracs) != 3 or abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"--split-fractions must be 3 numbers summing to 1: {fracs}")

    rng = random.Random(args.seed)
    rows, header = _index_rows(args.src_dir)

    if args.filter_is_comet is not None:
        before = len(rows)
        # is_comet is the FIRST element of the bucket key (matches
        # STRATIFY_COLS = ("is_comet", "distance_to_sun_bucket",
        # "speed_bucket")).
        rows = [(p, ridx, key[1:]) for p, ridx, key in rows
                if key[0] == args.filter_is_comet]
        print(
            f"is_comet filter ({args.filter_is_comet}): kept {len(rows):,} "
            f"of {before:,} rows ({100*len(rows)/max(1,before):.1f}%)",
            flush=True,
        )

    chosen, counts = _stratified_sample(rows, args.n_rows, rng)

    n_buckets = len(counts)
    cs = sorted(counts.values(), reverse=True)
    print(
        f"chose {len(chosen):,} rows across {n_buckets} buckets "
        f"({' × '.join(STRATIFY_COLS)})",
        flush=True,
    )
    if cs:
        print(
            f"  per-bucket counts — max={cs[0]} median={cs[len(cs)//2]} min={cs[-1]}",
            flush=True,
        )

    # Single combined CSV first (audit trail) — then random-shuffle split.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = args.out_dir / "planet_diverse.csv"
    _write_subset(chosen, header, combined_csv)

    # Now reload + split. Cheaper than streaming source dir 3x.
    with combined_csv.open() as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)
    rng.shuffle(all_rows)
    n = len(all_rows)
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    splits = {
        "planet_train.csv": all_rows[:n_train],
        "planet_val.csv":   all_rows[n_train:n_train + n_val],
        "planet_test.csv":  all_rows[n_train + n_val:],
    }
    for name, subset in splits.items():
        with (args.out_dir / name).open("w", newline="") as out_fh:
            w = csv.DictWriter(out_fh, fieldnames=header)
            w.writeheader()
            for r in subset:
                w.writerow(r)
        print(f"wrote {name}: {len(subset)} rows", flush=True)

    manifest = {
        "train": ["planet_train.csv"],
        "val":   ["planet_val.csv"],
        "test":  ["planet_test.csv"],
        "src_dir": str(args.src_dir),
        "n_rows": n,
        "n_buckets": n_buckets,
        "seed": args.seed,
        "stratify_cols": list(STRATIFY_COLS),
        "split": {"train": n_train, "val": n_val, "test": n - n_train - n_val},
        "bucket_counts": {
            "_".join(f"{c}{v}" for c, v in zip(STRATIFY_COLS, k)): cnt
            for k, cnt in sorted(counts.items())
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest to {args.out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
