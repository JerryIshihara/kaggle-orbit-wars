"""Fetch replay manifests for the current top-N competition leaderboard.

The Kaggle REST leaderboard endpoint exposes ranks + teamIds + scores
but no submissionIds. We recover each team's current
``publicLeaderboardSubmissionId`` via the cross-section in any known
submission's ``ListEpisodes`` response — passing
``--anchor-submission-id`` toggles which submission we read that
cross-section from. The default anchor is Shun_PI (current #1, plays
nearly every other top entrant).

Usage:
    python scripts/fetch_replays_from_leaderboard.py --top 5 --workers 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fetch_replays import DATA_ROOT, build_team_manifests  # noqa: E402
from utils.kaggle_episodes import get_leaderboard, resolve_submission_ids  # noqa: E402

# Default anchor: Shun_PI submission 52018000 (current #1). Plays
# nearly every other top entrant. Override with --anchor-submission-id
# if that ever stops being true.
DEFAULT_ANCHOR_SUBMISSION_ID = 52018000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", type=str, default="orbit-wars")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--anchor-submission-id", type=int,
        default=DEFAULT_ANCHOR_SUBMISSION_ID,
        help="Submission id used to look up other teams' current "
             "submissionIds via its ListEpisodes cross-section. Pick a "
             "team that has played most leaderboard entrants. "
             f"(default: {DEFAULT_ANCHOR_SUBMISSION_ID})",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"[lb] querying top {args.top} for {args.competition!r} ...")
    rows = get_leaderboard(args.competition, top_n=args.top)
    if len(rows) < args.top:
        print(
            f"[lb] warning: leaderboard returned only {len(rows)} rows "
            f"(asked for {args.top})"
        )

    print(
        f"[lb] resolving submission ids via anchor "
        f"submissionId={args.anchor_submission_id} ..."
    )
    sub_map = resolve_submission_ids(
        [r["teamId"] for r in rows],
        anchor_submission_id=args.anchor_submission_id,
    )

    resolved: list[dict] = []
    missing: list[dict] = []
    for r in rows:
        sub_id = sub_map.get(r["teamId"])
        if sub_id is None:
            missing.append(r)
            continue
        resolved.append({**r, "submissionId": int(sub_id)})

    if missing:
        print(
            f"[lb] warning: could not resolve submissionIds for "
            f"{len(missing)} team(s) — they haven't played the anchor:"
        )
        for r in missing:
            print(f"  - rank={r['rank']:>2}  team={r['teamName']!r} "
                  f"(teamId={r['teamId']})")
        print(
            "[lb] hint: pass --anchor-submission-id pointing to a team "
            "that has played the missing entrants."
        )

    print(f"[lb] resolved {len(resolved)} team(s):")
    for r in resolved:
        print(
            f"  rank={r['rank']:>2}  team={r['teamName']:<25s}  "
            f"sub={r['submissionId']}"
        )

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_path = DATA_ROOT / f"_top{args.top}_{date.today():%Y%m%d}.json"
    snapshot_path.write_text(json.dumps(
        {
            "competition": args.competition,
            "rows": resolved,
            "missing": missing,
        },
        indent=2,
    ))
    print(f"[lb] snapshot → {snapshot_path}")

    team_specs: list[tuple[str, int]] = []
    skipped = 0
    for r in resolved:
        manifest_path = DATA_ROOT / r["teamName"] / "_manifest.json"
        if manifest_path.exists() and not args.force:
            print(f"[skip] {r['teamName']} already has {manifest_path}")
            skipped += 1
            continue
        team_specs.append((str(r["teamName"]), int(r["submissionId"])))

    if not team_specs:
        print(f"[lb] nothing to fetch ({skipped} teams skipped)")
        return 0

    print(f"[lb] building manifests for {len(team_specs)} team(s) ...")
    build_team_manifests(team_specs, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
