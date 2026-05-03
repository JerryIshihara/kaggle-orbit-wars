"""Fetch episode metadata and replays from Kaggle's internal EpisodeService.

Two endpoints (auth = HTTP basic with KAGGLE_USERNAME : KAGGLE_KEY):

  POST /api/i/competitions.EpisodeService/GetEpisodeReplay
       body: {"episodeId": <int>}
       returns: full replay (steps[], configuration, info, rewards, ...)

  POST /api/i/competitions.EpisodeService/ListEpisodes
       body: {"submissionId": <int>}    (or {"ids": [<episodeId>, ...]})
       returns: {"episodes": [...], "submissions": [...], "teams": [...]}

Episodes describe a single match between agent submissions; the replay
contains every per-step observation, action and reward. There is no
documented endpoint that lists *all* of a team's submissions, so to
collect every episode for a team you need each of its submission IDs.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_BASE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService"
# Public REST leaderboard endpoint used by the kaggle CLI. Returns
# {"submissions":[{teamId, teamName, score, submissionDate, ...}],
#  "nextPageToken": "..."} — 20 rows per page; paginate via nextPageToken
# query param. Does NOT include submissionId (resolve separately via
# `list_submission_episodes` cross-reference).
_LEADERBOARD_REST_URL = "https://www.kaggle.com/api/v1/competitions/{slug}/leaderboard/view"
# Kaggle's CDN returns 503 ("DNS cache overflow") for non-browser UAs like
# python-urllib's default — pin to a Mozilla UA so requests get through.
_USER_AGENT = "Mozilla/5.0"


def fetch_episode(
    episode_id: int,
    save_to: Path | str | None = None,
) -> dict[str, Any]:
    """Download the full replay for `episode_id`.

    Args:
      episode_id: numeric episode id (the `episodeId=` query param on a
        Kaggle leaderboard URL).
      save_to: optional path to also write the JSON to disk.

    Returns:
      Parsed replay JSON (top-level keys: configuration, specification,
      info, steps, rewards, statuses, ...).
    """
    payload = _post("GetEpisodeReplay", {"episodeId": int(episode_id)})
    if save_to is not None:
        path = Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return payload


def list_submission_episodes(submission_id: int) -> dict[str, Any]:
    """List every episode that includes the given submission.

    Returns the full ListEpisodes response: `episodes`, `submissions`,
    `teams`. Use `[e["id"] for e in resp["episodes"]]` for just the ids.
    """
    return _post("ListEpisodes", {"submissionId": int(submission_id)})


def get_episodes(episode_ids: list[int]) -> dict[str, Any]:
    """Fetch episode *metadata* (not replays) for a batch of ids."""
    return _post("ListEpisodes", {"ids": [int(i) for i in episode_ids]})


def get_leaderboard(slug: str, top_n: int = 5) -> list[dict[str, Any]]:
    """Return top leaderboard rows from Kaggle's public REST API.

    Each row carries at least ``{rank, teamId, teamName, score,
    submissionDate}``. The REST endpoint does NOT include
    ``submissionId``; resolve those via :func:`resolve_submission_ids`
    using a known anchor submission for any one team that has played
    against the targets.

    Walks paginated responses until ``top_n`` rows are accumulated. The
    endpoint returns 20 per page.
    """
    if top_n <= 0:
        return []
    rows: list[dict[str, Any]] = []
    next_token: str | None = None
    base = _LEADERBOARD_REST_URL.format(slug=slug)
    while len(rows) < top_n:
        url = base + (f"?pageToken={next_token}" if next_token else "")
        payload = _get_url(url)
        submissions = payload.get("submissions") or []
        if not isinstance(submissions, list) or not submissions:
            if not rows:
                raise RuntimeError(
                    f"Kaggle leaderboard endpoint returned no submissions "
                    f"for slug={slug!r}; verify the slug & endpoint shape"
                )
            break
        for entry in submissions:
            team_id = entry.get("teamId")
            team_name = entry.get("teamName") or entry.get("teamNameNullable")
            if team_id is None or team_name is None:
                continue
            score_str = entry.get("score") or entry.get("scoreNullable")
            try:
                score = float(score_str) if score_str is not None else None
            except (TypeError, ValueError):
                score = None
            rows.append({
                "rank": len(rows) + 1,
                "teamId": int(team_id),
                "teamName": str(team_name),
                "score": score,
                "submissionDate": entry.get("submissionDate"),
            })
            if len(rows) >= top_n:
                break
        next_token = payload.get("nextPageToken")
        if not next_token:
            break
    return rows[:top_n]


def resolve_submission_ids(
    team_ids: list[int],
    *,
    anchor_submission_id: int,
) -> dict[int, int]:
    """Map ``teamId → publicLeaderboardSubmissionId`` for the given teams.

    Strategy: pull the full episode list for ``anchor_submission_id``
    (any submission whose team has played against most top teams; e.g.,
    the current #1) — its ``teams`` array exposes
    ``publicLeaderboardSubmissionId`` for every team that has crossed
    paths with the anchor. Returns only teams found in the anchor's
    cross-section; missing teams are silently dropped (caller can
    detect & error).
    """
    resp = list_submission_episodes(int(anchor_submission_id))
    teams = resp.get("teams") or []
    by_id: dict[int, int] = {}
    for t in teams:
        tid = t.get("id")
        sub_id = t.get("publicLeaderboardSubmissionId")
        if tid is None or sub_id is None:
            continue
        by_id[int(tid)] = int(sub_id)
    return {tid: by_id[tid] for tid in team_ids if tid in by_id}


def _post(method: str, body: dict[str, Any]) -> dict[str, Any]:
    url = f"{_BASE_URL}/{method}"
    return _post_url(url, body)


def _post_url(url: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {_basic_auth()}",
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Kaggle request failed: HTTP {e.code} — {msg}") from e


def _get_url(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Basic {_basic_auth()}",
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Kaggle request failed: HTTP {e.code} — {msg}") from e


def _extract_competition_id(payload: Any) -> int | None:
    if isinstance(payload, dict):
        for key in ("competitionId", "id"):
            value = payload.get(key)
            if isinstance(value, int):
                return value
        for value in payload.values():
            found = _extract_competition_id(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _extract_competition_id(item)
            if found is not None:
                return found
    return None


def _extract_leaderboard_rows(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return

        direct_keys = {str(k) for k in node.keys()}
        anchor_keys = {
            "team", "teamId", "teamName",
            "submission", "submissionId", "latestSubmissionId", "currentSubmissionId",
            "rank", "score",
        }
        if direct_keys & anchor_keys:
            row = _try_make_leaderboard_row(node)
            if row is not None:
                key = (row["teamId"], row["submissionId"])
                if key not in seen:
                    rows.append(row)
                    seen.add(key)

        for value in node.values():
            walk(value)

    walk(payload)
    return rows


def _try_make_leaderboard_row(node: dict[str, Any]) -> dict[str, Any] | None:
    team_id = _find_team_id(node)
    team_name = _find_team_name(node)
    submission_id = _find_submission_id(node)
    if team_id is None or submission_id is None or not team_name:
        return None

    rank = _find_first_int(node, {"rank", "displayRank", "leaderboardRank"}) or 10**9
    score = _find_first_scalar(
        node,
        {"score", "displayScore", "publicScore", "privateScore"},
    )
    return {
        "rank": int(rank),
        "teamId": int(team_id),
        "teamName": str(team_name),
        "submissionId": int(submission_id),
        "score": score,
    }


def _find_first_int(node: Any, keys: set[str]) -> int | None:
    value = _find_first_scalar(node, keys)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _find_first_str(node: Any, keys: set[str]) -> str | None:
    value = _find_first_scalar(node, keys)
    return value if isinstance(value, str) else None


def _find_team_id(node: Any) -> int | None:
    value = _find_first_scalar(node, {"teamId", "team_id"})
    if value is not None:
        return _coerce_int(value)
    if isinstance(node, dict) and isinstance(node.get("team"), dict):
        return _coerce_int(_find_first_scalar(node["team"], {"teamId", "team_id", "id"}))
    return None


def _find_team_name(node: Any) -> str | None:
    value = _find_first_str(node, {"teamName", "team_name"})
    if value:
        return value
    if isinstance(node, dict) and isinstance(node.get("team"), dict):
        return _find_first_str(node["team"], {"teamName", "team_name", "name"})
    return None


def _find_submission_id(node: Any) -> int | None:
    value = _find_first_scalar(
        node,
        {"submissionId", "submission_id", "latestSubmissionId", "currentSubmissionId"},
    )
    if value is not None:
        return _coerce_int(value)
    if isinstance(node, dict) and isinstance(node.get("submission"), dict):
        return _coerce_int(_find_first_scalar(node["submission"], {"submissionId", "submission_id", "id"}))
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _find_first_scalar(node: Any, keys: set[str]) -> Any:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in keys and isinstance(value, (str, int, float)):
                return value
        for value in node.values():
            found = _find_first_scalar(value, keys)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_first_scalar(item, keys)
            if found is not None:
                return found
    return None


def _basic_auth() -> str:
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_KEY")
    if not (user and key):
        kj = Path.home() / ".kaggle" / "kaggle.json"
        if kj.exists():
            cfg = json.loads(kj.read_text())
            user = user or cfg.get("username")
            key = key or cfg.get("key")
    if not (user and key):
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME + KAGGLE_KEY env "
            "vars or populate ~/.kaggle/kaggle.json."
        )
    return base64.b64encode(f"{user}:{key}".encode()).decode()
