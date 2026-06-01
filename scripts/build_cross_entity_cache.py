"""Pre-build the cross_entity dataset cache as a single ``.pt`` file.

Skips the 5-10 min CSV walk on Colab. Once built, training reads
snapshots via ``CachedCrossEntitySnapshotDataset`` in a few seconds.
The default output is tensorized (one tensor per field) so Colab can
load it with mmap instead of unpickling one dict per snapshot.

Usage:
  # Tiny smoke cache (~5-10 episodes for end-to-end flow tests)
  python -m scripts.build_cross_entity_cache \\
      --out /tmp/cross_entity_cache_tiny.pt \\
      --tiny

  # Full cache (all 1133 episodes)
  python -m scripts.build_cross_entity_cache \\
      --out /tmp/cross_entity_cache_full.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.transformer_v2.paths import CROSS_ENTITY_DATASET_DIR
from agents.transformer_v2.pretrain.cross_entity import build_cross_entity_cache


def _stems_from_manifest(per_split_cap: int | None = None) -> list[str]:
    """Pick a balanced subset across train/val/test using the manifest.

    Used by ``--tiny``: takes the first ``per_split_cap`` stems from
    each split so the resulting cache still produces non-empty splits
    after slicing inside ``train()``.
    """
    manifest = json.loads((CROSS_ENTITY_DATASET_DIR / "manifest.json").read_text())
    out: list[str] = []
    for split in ("train", "val", "test"):
        names = manifest.get(split, [])
        stems = [n.removeprefix("cross_entity_").removesuffix(".csv") for n in names]
        if per_split_cap is not None:
            stems = stems[:per_split_cap]
        out.extend(stems)
    return out


def _split_stems_from_manifest(
    *,
    tiny: bool = False,
    per_split_cap: int | None = None,
) -> dict[str, list[str]]:
    manifest = json.loads((CROSS_ENTITY_DATASET_DIR / "manifest.json").read_text())
    out: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        names = manifest.get(split, [])
        stems = [
            n.removeprefix("cross_entity_").removesuffix(".csv")
            for n in names
        ]
        if tiny:
            caps = {"train": 5, "val": 2, "test": 2}
            stems = stems[:caps[split]]
        elif per_split_cap is not None:
            stems = stems[:per_split_cap]
        out[split] = stems
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tiny", action="store_true",
        help="Build a small subset (5 train + 2 val + 2 test stems) for "
             "end-to-end flow smoke tests. Leave off for the full corpus.",
    )
    parser.add_argument(
        "--per-split-cap", type=int, default=None,
        help="Cap stems per split (default unlimited). Implied by --tiny "
             "as (5, 2, 2) when not set.",
    )
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=1024)
    parser.add_argument(
        "--legacy-format",
        action="store_true",
        help="Write the old list-of-snapshot-dicts cache format. Slower to "
             "load; kept only for debugging.",
    )
    args = parser.parse_args()

    stems: list[str] | None
    split_stems: dict[str, list[str]] | None = None
    if args.tiny:
        split_stems = _split_stems_from_manifest(tiny=True)
        stems = []
        for split in ("train", "val", "test"):
            stems.extend(split_stems[split])
        print(f"[tiny] {len(stems)} stems "
              f"(5 train + 2 val + 2 test from manifest)")
    elif args.per_split_cap is not None:
        split_stems = _split_stems_from_manifest(
            per_split_cap=args.per_split_cap,
        )
        stems = []
        for split in ("train", "val", "test"):
            stems.extend(split_stems[split])
        print(f"[capped] {len(stems)} stems "
              f"(cap={args.per_split_cap} per split)")
    else:
        stems = None  # full
        split_stems = _split_stems_from_manifest()

    build_cross_entity_cache(
        out_path=args.out,
        stems=stems,
        split_stems=split_stems,
        max_planets=args.max_planets,
        max_fleets=args.max_fleets,
        cache_format="legacy" if args.legacy_format else "tensorized",
    )


if __name__ == "__main__":
    main()
