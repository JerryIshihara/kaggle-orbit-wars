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
import csv
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agents.archive.transformer_v1.featurizer import (  # noqa: E402
    save_episode_action_csv,
    save_episode_cross_entity_csv,
    save_episode_entity_csv,
    save_episode_fleet_csv,
    save_episode_planet_csv,
)
from agents.archive.transformer_v1.paths import (  # noqa: E402
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


def _csv_max_turn(path: Path) -> int | None:
    """Cheap last-row read for ``turn`` column. Returns ``None`` if the
    file is empty / unreadable / has no ``turn`` column.

    Walks the entire file rather than slicing the tail because CSVs are
    not random-access; reading 30k rows is still sub-second per file.
    """
    try:
        with path.open() as fh:
            reader = csv.DictReader(fh)
            if "turn" not in (reader.fieldnames or []):
                return None
            last: int | None = None
            for row in reader:
                t = row.get("turn")
                if t is None or t == "":
                    continue
                try:
                    last = int(t)
                except ValueError:
                    continue
            return last
    except OSError:
        return None


def _replay_expected_max_turn(replay_path: Path, dataset_name: str) -> int | None:
    """The largest ``turn`` value we expect a complete CSV to carry.

    Dataset-specific:

      * ``fleet`` — the last turn at which any seat's observation has
        a non-empty ``fleets`` list. A game that ends with no in-flight
        fleets for its last 100+ turns genuinely has no rows to write
        past that point; flagging it as "incomplete" was a false
        positive in the original ``len(steps) - 5`` check (over-
        excluded 78 of Ebi's 434 stems that were all in fact correct).
      * Everything else (planet / entity / cross_entity / action) —
        these write one row per planet/snapshot regardless of fleet
        state, so ``len(steps) - K`` is the right floor. We keep a
        5-turn slack for any featurizer-side lookahead horizon.

    Returns ``-1`` for fleet-dataset replays that never had a fleet
    (i.e. completeness target is "empty CSV"); callers treat ``csv_max
    >= exp_max`` as complete.
    """
    try:
        with gzip.open(replay_path, "rt") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    steps = payload.get("steps") or []
    if not steps:
        return None
    if dataset_name == "fleet":
        last_with_fleets = -1
        for t, step in enumerate(steps):
            if not step or not isinstance(step[0], dict):
                continue
            obs = step[0].get("observation") or {}
            if obs.get("fleets"):
                last_with_fleets = t
        # 2-turn slack absorbs any off-by-one between the featurizer's
        # iteration end and the last fleet turn in the replay.
        return max(-1, last_with_fleets - 2)
    return max(0, len(steps) - 5)


def _action_mask_path(csv_path: Path) -> Path:
    """Companion ``_masks/<stem>.npz`` path for an action CSV.

    ``save_episode_action_csv`` writes both the CSV and its mask
    sidecar; the side-cache is what populates ``src_valid`` and
    ``tgt_valid`` in :class:`ActionSnapshotDataset`. Without the
    sidecar, training falls back to an owned-only mask derived at
    runtime from ``planet_features``, which works but loses the
    refined per-turn launchability signal the featurizer would have
    written.
    """
    stem = csv_path.stem.removeprefix("action_")
    return csv_path.parent / "_masks" / f"{stem}.npz"


def _is_complete_csv(
    csv_path: Path, replay_path: Path | None, dataset_name: str,
) -> bool:
    """A CSV is complete if its last ``turn`` is at least the replay's
    expected max turn. Missing replay or unparseable CSV → treat as
    incomplete so a fresh build re-emits it.

    Fleet dataset has a special case: when the replay never had any
    fleet at all (``exp_max == -1``), the expected CSV is empty, so
    a missing ``turn`` value on the CSV side (``csv_max is None``)
    is *complete*, not incomplete.

    Action dataset also requires its ``_masks/<stem>.npz`` sidecar to
    exist — ``save_episode_action_csv`` writes both atomically, so a
    CSV without its mask file is the result of an interrupted write
    (or, more commonly, an older build run from before the sidecar
    code shipped). Either way: incomplete, re-emit.
    """
    if replay_path is None or not replay_path.exists():
        # No replay file to compare against — keep the legacy behavior
        # (presence == done) so we don't gratuitously re-process when
        # the original replay has been moved.
        return True
    exp_max = _replay_expected_max_turn(replay_path, dataset_name)
    if exp_max is None:
        return True
    csv_max = _csv_max_turn(csv_path)
    if exp_max < 0:
        # Replay had no fleets ever; an empty CSV (csv_max=None) or
        # one with at most a header is complete.
        return True
    if csv_max is None:
        return False
    if csv_max < exp_max:
        return False
    # Action-dataset-specific: also require the _masks sidecar.
    if dataset_name == "action":
        if not _action_mask_path(csv_path).exists():
            return False
    return True


def already_processed(
    out_dir: Path,
    prefix: str,
    *,
    replay_by_stem: dict[str, Path] | None = None,
    dataset_name: str = "",
    strict: bool = False,
) -> set[str]:
    """Set of stems whose CSV exists under ``out_dir``.

    When ``strict=True`` and ``replay_by_stem`` is provided, each CSV is
    cross-checked against its replay's step count; CSVs that stopped
    short (interrupted writes — the bug that left fleet_75610892_4_2.csv
    at turn 409 while the replay ran through 438) are **excluded** from
    the "done" set so the build step re-processes them.

    **Action-dataset special case (non-strict path also).** The action
    featurizer writes a ``_masks/<stem>.npz`` sidecar alongside the
    CSV in the same call; older builds (pre-sidecar) left CSVs without
    masks, and the resulting training falls back to an owned-only mask
    that's correct but loses the per-turn launchability signal. To
    backfill those sidecars, a stem with an existing CSV but a missing
    mask file is treated as **not** processed — even in the default
    (non-strict) path — so a default ``--datasets action`` run rebuilds
    just the affected stems.
    """
    out: set[str] = set()
    for p in out_dir.glob(f"{prefix}*.csv"):
        stem = p.stem.removeprefix(prefix)
        if not strict:
            # Action sidecar guard runs even outside strict mode (see
            # docstring for rationale).
            if dataset_name == "action" and not _action_mask_path(p).exists():
                continue
            out.add(stem)
            continue
        replay = (replay_by_stem or {}).get(stem)
        if _is_complete_csv(p, replay, dataset_name):
            out.add(stem)
    return out


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
    parser.add_argument(
        "--strict-completeness", action="store_true",
        help=(
            "Cross-check every existing CSV's last turn against its "
            "replay's step count; treat CSVs that stopped short "
            "(interrupted writes) as not-yet-processed so they get "
            "regenerated. Off by default for back-compat with the "
            "old presence-only fast path."
        ),
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help=(
            "Don't build anything; just report per-dataset how many "
            "existing CSVs would be flagged as incomplete under "
            "--strict-completeness. Useful for diagnosing the "
            "fleet-vs-entity turn-count mismatch."
        ),
    )
    args = parser.parse_args()

    selected = _parse_datasets(args.datasets)
    candidates = discover_replays(args.replay_dir)
    replay_by_stem: dict[str, Path] = {
        _episode_stem(p): p for p in candidates
    }

    for name in selected:
        out_dir, prefix, save_func = DATASETS[name]
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.audit_only:
            existing = already_processed(out_dir, prefix)
            complete = already_processed(
                out_dir, prefix,
                replay_by_stem=replay_by_stem,
                dataset_name=name,
                strict=True,
            )
            incomplete = sorted(existing - complete)
            print(
                f"[{name}] existing={len(existing)} complete={len(complete)} "
                f"incomplete={len(incomplete)}"
            )
            if incomplete:
                # Surface a handful so the user can spot-check.
                preview = ", ".join(incomplete[:5])
                more = "" if len(incomplete) <= 5 else f" (+{len(incomplete) - 5} more)"
                print(f"[{name}]   first incomplete: {preview}{more}")
            continue
        done = already_processed(
            out_dir, prefix,
            replay_by_stem=replay_by_stem,
            dataset_name=name,
            strict=args.strict_completeness,
        )
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
