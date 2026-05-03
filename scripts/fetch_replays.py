"""Orchestrator: build a per-team manifest of replays to download.

For each `(team_name, submission_id)` pair, calls
`list_submission_episodes` and writes
`data/replays/<team_name>/_manifest.json` containing
`[{"eid": ..., "n": ..., "slot": ...}, ...]` ready for a pool of
`fetch_replays_worker.py` processes to consume.

Also prints suggested chunk assignments (`--indices 0,1,...`) for a
specified worker count so a calling agent / shell can dispatch workers
in parallel.

Usage:
    python scripts/fetch_replays.py \\
        --team Shun_PI:52018000 \\
        --team kovi:51987365 \\
        --workers 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "replays"


def _load_helper():
    path = REPO_ROOT / "utils" / "kaggle_episodes.py"
    spec = importlib.util.spec_from_file_location("kaggle_episodes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_manifest(team_name: str, submission_id: int, helper) -> Path:
    out_dir = DATA_ROOT / team_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resp = helper.list_submission_episodes(submission_id)
    episodes = resp["episodes"]

    manifest = []
    for ep in episodes:
        agents = ep["agents"]
        slot = next(
            i for i, a in enumerate(agents)
            if a["submissionId"] == submission_id
        )
        manifest.append({"eid": ep["id"], "n": len(agents), "slot": slot})

    manifest.sort(key=lambda e: e["eid"])
    manifest_path = out_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"[{team_name}] submission={submission_id}  "
        f"episodes={len(manifest)}  manifest={manifest_path}"
    )
    return manifest_path


def chunked_indices(n: int, k: int) -> list[list[int]]:
    """Split range(n) into k near-equal contiguous chunks."""
    base, extra = divmod(n, k)
    chunks, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return [c for c in chunks if c]


def build_team_manifests(team_specs: list[tuple[str, int]], workers: int) -> None:
    helper = _load_helper()
    for name, submission_id in team_specs:
        manifest_path = build_manifest(name, int(submission_id), helper)
        manifest = json.loads(manifest_path.read_text())
        chunks = chunked_indices(len(manifest), workers)
        out_dir = DATA_ROOT / name
        print(f"[{name}] worker assignments ({len(chunks)} chunks):")
        for i, chunk in enumerate(chunks):
            indices_csv = ",".join(str(x) for x in chunk)
            print(
                f"  worker {i:2d} ({len(chunk):3d} eps): "
                f"python scripts/fetch_replays_worker.py "
                f"--manifest {manifest_path} "
                f"--indices {indices_csv} "
                f"--out-dir {out_dir}"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--team", action="append", required=True,
        help="`<teamName>:<submissionId>` (repeatable).",
    )
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    team_specs = []
    for spec in args.team:
        name, sub = spec.split(":", 1)
        team_specs.append((name, int(sub)))
    build_team_manifests(team_specs, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
