"""Worker that fetches a chunk of episode replays.

Reads `(eid, n, slot)` triples from a manifest, fetches each replay via
`utils.kaggle_episodes.fetch_episode`, and writes
`<out_dir>/<eid>_<n>_<slot>.json`. Idempotent: skips files already present
above 1 KB so chunks can be re-run without redownloading.

Usage:
    python scripts/fetch_replays_worker.py \
        --manifest data/replays/<TEAM>/_manifest.json \
        --indices 0,1,2,3 \
        --out-dir data/replays/<TEAM>
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_helper():
    """Import utils/kaggle_episodes.py without triggering utils/__init__.py.

    `utils/__init__.py` chains into `agents`, which requires
    `kaggle_environments`; loading the helper file directly sidesteps that.
    """
    path = REPO_ROOT / "utils" / "kaggle_episodes.py"
    spec = importlib.util.spec_from_file_location("kaggle_episodes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--indices", required=True,
                   help="Comma-separated indices into the manifest list.")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    helper = _load_helper()

    ok, fail, skipped, failures = 0, 0, 0, []
    for i in indices:
        entry = manifest[i]
        eid, n, slot = entry["eid"], entry["n"], entry["slot"]
        out = args.out_dir / f"{eid}_{n}_{slot}.json.gz"
        if out.exists() and out.stat().st_size > 256:
            skipped += 1
            print(f"SKIP {eid}", flush=True)
            continue
        try:
            payload = helper.fetch_episode(eid)
            with gzip.open(out, "wt", compresslevel=9) as fh:
                json.dump(payload, fh)
            ok += 1
            print(f"OK {eid}", flush=True)
        except Exception as e:  # noqa: BLE001 — log and continue
            fail += 1
            failures.append(eid)
            print(f"FAIL {eid} {type(e).__name__}: {e}", flush=True)

    fail_csv = ",".join(str(x) for x in failures)
    print(
        f"DONE ok={ok} fail={fail} skipped={skipped} failures={fail_csv}",
        flush=True,
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
