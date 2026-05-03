"""Generate per-encoder pretraining datasets from raw replays.

Picks N episodes that haven't been processed yet, runs the corresponding
``save_episode_*_csv`` for each requested dataset, and writes a
``manifest.json`` (episode-level train/val/test split) per dataset.

Episode-level split (not row-level) prevents intra-game leakage: rows
from turn t and t+1 of the same match would otherwise both end up in
train+val and trivially inflate val accuracy.

Datasets:
  fleet         — per-fleet encoder pretrain CSV (per-(turn, fleet))
  planet        — per-planet encoder pretrain CSV (per-(turn, planet))
  entity        — per-(planet, player) inbound-fleet stats CSV
  cross_entity  — per-(turn, planet) spatial + tier-3 / tier-4 labels
  action        — snapshot-level expert-imitation labels (per-turn)

Run from the repo root:

    python scripts/build_encoder_dataset.py --num-episodes 20
    python scripts/build_encoder_dataset.py --num-episodes 5 \\
        --datasets fleet,planet,entity,cross_entity,action

Re-running is idempotent on the CSV side (already-generated CSVs are
skipped) but always rewrites each dataset's manifest from the current
set of CSVs in its output dir.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agents.transformer_v1.featurizer import (  # noqa: E402
    save_episode_action_csv,
    save_episode_cross_entity_csv,
    save_episode_entity_csv,
    save_episode_fleet_csv,
    save_episode_planet_csv,
)
from agents.transformer_v1.paths import (  # noqa: E402
    ACTION_DATASET_DIR,
    CROSS_ENTITY_DATASET_DIR,
    ENTITY_DATASET_DIR,
    FLEET_DATASET_DIR,
    PLANET_DATASET_DIR,
)


# Dataset-name → (out_dir, csv_filename_prefix, save_func).
DATASETS: dict[str, tuple[Path, str, Callable[..., Path]]] = {
    "fleet": (FLEET_DATASET_DIR, "fleet_", save_episode_fleet_csv),
    "planet": (PLANET_DATASET_DIR, "planet_", save_episode_planet_csv),
    "entity": (ENTITY_DATASET_DIR, "entity_", save_episode_entity_csv),
    "cross_entity": (
        CROSS_ENTITY_DATASET_DIR, "cross_entity_", save_episode_cross_entity_csv,
    ),
    "action": (ACTION_DATASET_DIR, "action_", save_episode_action_csv),
}


def _episode_stem(path: Path) -> str:
    # ``75408674_2_0.json.gz`` → ``75408674_2_0``
    return path.name.split(".")[0]


def discover_replays(replay_root: Path) -> list[Path]:
    return sorted(replay_root.rglob("*.json.gz"))


def already_processed(out_dir: Path, prefix: str) -> set[str]:
    return {
        p.stem.removeprefix(prefix)
        for p in out_dir.glob(f"{prefix}*.csv")
    }


def write_manifest(
    out_dir: Path,
    prefix: str,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    csvs = sorted(out_dir.glob(f"{prefix}*.csv"))
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


def _parse_datasets(spec: str) -> list[str]:
    items = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in items if s not in DATASETS]
    if unknown:
        raise SystemExit(
            f"unknown --datasets entries: {unknown} "
            f"(valid: {sorted(DATASETS.keys())})"
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-dir", type=Path, default=REPO / "data" / "replays",
        help="Root containing replay .json.gz files (recursively).",
    )
    parser.add_argument(
        "--datasets", type=str,
        default="fleet,planet,entity,cross_entity,action",
        help=(
            "Comma-separated list of datasets to build. "
            f"Valid: {sorted(DATASETS.keys())}."
        ),
    )
    parser.add_argument(
        "--num-episodes", type=int, default=20,
        help="How many NEW episodes to process this run (per dataset).",
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected = _parse_datasets(args.datasets)
    candidates = discover_replays(args.replay_dir)

    for name in selected:
        out_dir, prefix, save_func = DATASETS[name]
        out_dir.mkdir(parents=True, exist_ok=True)
        done = already_processed(out_dir, prefix)
        todo = [
            p for p in candidates
            if _episode_stem(p) not in done
        ][: args.num_episodes]
        print(
            f"[{name}] replays found: {len(candidates)}; "
            f"already processed: {len(done)}; "
            f"will process: {len(todo)}"
        )
        for path in todo:
            save_func(path)
        manifest = write_manifest(
            out_dir,
            prefix,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        print(
            f"[{name}] manifest: train={len(manifest['train'])} "
            f"val={len(manifest['val'])} test={len(manifest['test'])}"
        )


if __name__ == "__main__":
    main()
