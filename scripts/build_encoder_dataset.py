"""Generate the per-fleet encoder pretraining dataset.

Picks N episodes that haven't been processed yet, runs
``save_episode_fleet_csv`` on each, and writes a ``manifest.json`` with
an episode-level train/val/test split. Episode-level split (not row-
level) prevents intra-game leakage: rows from turn t and t+1 of the
same match would otherwise both end up in train+val and trivially
inflate val accuracy.

Run from the repo root:

    python scripts/build_encoder_dataset.py --num-episodes 20

Re-running is idempotent on the CSV side (already-generated CSVs are
skipped) but always rewrites the manifest from the current set of
CSVs in ``data/encoders/``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agents.transformer_v1.featurizer import (  # noqa: E402
    DEFAULT_ENCODER_DATA_DIR,
    save_episode_fleet_csv,
)


def _episode_stem(path: Path) -> str:
    # ``75408674_2_0.json.gz`` → ``75408674_2_0``
    return path.name.split(".")[0]


def discover_replays(replay_root: Path) -> list[Path]:
    return sorted(replay_root.rglob("*.json.gz"))


def already_processed(out_dir: Path) -> set[str]:
    return {
        p.stem.removeprefix("fleet_")
        for p in out_dir.glob("fleet_*.csv")
    }


def write_manifest(
    out_dir: Path,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    csvs = sorted(out_dir.glob("fleet_*.csv"))
    rng = random.Random(seed)
    rng.shuffle(csvs)
    n = len(csvs)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_train = max(1, n - n_test - n_val)
    test = csvs[:n_test]
    val = csvs[n_test : n_test + n_val]
    train = csvs[n_test + n_val : n_test + n_val + n_train]
    manifest = {
        "train": [p.name for p in train],
        "val": [p.name for p in val],
        "test": [p.name for p in test],
        "split_seed": seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-dir", type=Path, default=REPO / "data" / "replays",
        help="Root containing replay .json.gz files (recursively).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_ENCODER_DATA_DIR,
        help="Where fleet CSVs and manifest land.",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=20,
        help="How many NEW episodes to process this run.",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15,
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.15,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_replays(args.replay_dir)
    done = already_processed(args.out_dir)
    todo = [p for p in candidates if _episode_stem(p) not in done][: args.num_episodes]

    print(
        f"replays found: {len(candidates)}; "
        f"already processed: {len(done)}; "
        f"will process: {len(todo)}"
    )
    for path in todo:
        save_episode_fleet_csv(path)

    manifest = write_manifest(
        args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(
        f"manifest: train={len(manifest['train'])} "
        f"val={len(manifest['val'])} test={len(manifest['test'])}"
    )


if __name__ == "__main__":
    main()
