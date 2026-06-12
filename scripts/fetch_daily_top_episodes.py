"""Selectively fetch top-team episodes from a Kaggle DAILY episode dataset.

Flow (honors "don't download all" — the daily is ~21 GB):
  1. page the daily dataset's file list (filenames ARE episode ids)
  2. batch-resolve episode metadata via the episodes API (get_episodes)
  3. keep episodes involving the requested teams; rank by (team priority,
     team won, recency); drop episodes already in data/replays
  4. download ONLY the selected files; store as
     data/replays/<team>/<episodeId>_<nplayers>_<seat>.json.gz
     (one file per selected team-seat — the featurizer's layout)

Run:
    .venv/bin/python scripts/fetch_daily_top_episodes.py \
        --day 2026-06-11 --take 100 \
        --teams "Jake Will,TonyK,213tubo,Harm Buisman,Felix M Neumann"
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.kaggle_episodes import get_episodes, get_leaderboard  # noqa: E402

KAGGLE = str(REPO_ROOT / ".venv" / "bin" / "kaggle")
REPLAYS = REPO_ROOT / "data" / "replays"


def _log(msg: str) -> None:
    print(f"[daily {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_all_files(slug: str) -> list[str]:
    names: list[str] = []
    token = None
    while True:
        cmd = [KAGGLE, "datasets", "files", slug, "--page-size", "200"]
        if token:
            cmd += ["--page-token", token]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        names += re.findall(r"^(\d+)\.json", out, flags=re.M)
        m = re.search(r"Next Page Token = (\S+)", out)
        token = m.group(1) if m else None
        _log(f"  listed {len(names)} files ...")
        if not token:
            return names


def existing_episode_ids() -> set[str]:
    ids = set()
    for p in REPLAYS.glob("*/*.json.gz"):
        ids.add(p.name.split("_")[0])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--take", type=int, default=100)
    ap.add_argument("--teams", required=True,
                    help="comma-separated team names, priority order")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--debug-first-batch", action="store_true")
    args = ap.parse_args()

    teams_priority = [t.strip() for t in args.teams.split(",")]
    slug = f"kaggle/orbit-wars-episodes-{args.day}"

    # name -> teamId from the live leaderboard (episode agents carry only
    # teamId; the ListEpisodes-by-ids response has no team-name table).
    lb = get_leaderboard("orbit-wars", top_n=30)
    id_by_name = {}
    for row in lb:
        nm = row.get("teamName") or row.get("name")
        tid = row.get("teamId") or row.get("id")
        if nm and tid:
            id_by_name[nm] = int(tid)
    missing = [t for t in teams_priority if t not in id_by_name]
    assert not missing, f"teams not on leaderboard top-30: {missing}"
    prio_by_tid = {id_by_name[t]: i for i, t in enumerate(teams_priority)}
    name_by_tid = {id_by_name[t]: t for t in teams_priority}
    _log(f"team ids: { {t: id_by_name[t] for t in teams_priority} }")

    _log(f"listing {slug} ...")
    files = list_all_files(slug)
    _log(f"{len(files)} episode files in the daily")

    have = existing_episode_ids()
    _log(f"{len(have)} episode ids already in data/replays (excluded)")

    candidates = []  # (priority, lost, -ep_id, ep_id, team, seat, nplayers)
    for i in range(0, len(files), args.batch):
        ids = [int(x) for x in files[i:i + args.batch]]
        resp = None
        for attempt, wait in enumerate((0, 15, 30, 60, 120, 240)):
            if wait:
                _log(f"  batch {i}: backing off {wait}s "
                     f"(attempt {attempt + 1})")
                time.sleep(wait)
            try:
                resp = get_episodes(ids)
                break
            except Exception as e:
                if "429" not in str(e) and attempt >= 1:
                    raise
        if resp is None:
            raise RuntimeError(f"batch {i}: exhausted retries")
        time.sleep(4)  # pace the API — ~50 batches per daily
        if args.debug_first_batch and i == 0:
            print(json.dumps(resp, indent=1)[:3000])
            return
        for ep in resp.get("episodes") or []:
            ep_id = ep.get("id") or ep.get("episodeId")
            agents = ep.get("agents") or []
            n = len(agents)
            # final reward per agent decides "won" (max reward)
            rewards = [a.get("reward") for a in agents]
            best = max((r for r in rewards if r is not None), default=None)
            for pos, a in enumerate(agents):
                tid = a.get("teamId")
                if tid not in prio_by_tid:
                    continue
                if str(ep_id) in have:
                    continue
                seat = int(a.get("index", pos))
                lost = 0 if (best is not None and rewards[pos] == best) else 1
                candidates.append((prio_by_tid[tid], lost, -int(ep_id),
                                   str(ep_id), name_by_tid[tid], seat, n))
        if (i // args.batch) % 5 == 0:
            _log(f"  scanned {min(i + args.batch, len(files))}/{len(files)} "
                 f"— {len(candidates)} top-team candidates")

    candidates.sort()
    picked = candidates[: args.take]
    _log(f"selected {len(picked)} of {len(candidates)} candidates")
    by_team: dict[str, int] = {}
    for c in picked:
        by_team[c[4]] = by_team.get(c[4], 0) + 1
    _log(f"  per team: {by_team}")

    tmp = Path("/tmp/ow_daily_pull")
    tmp.mkdir(exist_ok=True)
    n_done = 0
    for _, _, _, ep_id, team, seat, n in picked:
        dst_dir = REPLAYS / team
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{ep_id}_{n}_{seat}.json.gz"
        if dst.exists():
            continue
        raw = tmp / f"{ep_id}.json"
        if not raw.exists():
            r = subprocess.run(
                [KAGGLE, "datasets", "download", slug, "-f", f"{ep_id}.json",
                 "-p", str(tmp), "--unzip"],
                capture_output=True, text=True)
            if not raw.exists():
                _log(f"  ! download failed for {ep_id}: {r.stderr[-120:]}")
                continue
        with open(raw, "rb") as fh:
            payload = fh.read()
        with gzip.open(dst, "wb") as gz:
            gz.write(payload)
        n_done += 1
        if n_done % 10 == 0:
            _log(f"  stored {n_done}/{len(picked)}")
    _log(f"DONE: stored {n_done} new replays "
         f"({sum(1 for _ in REPLAYS.glob('*/*.json.gz'))} total on disk)")


if __name__ == "__main__":
    main()
