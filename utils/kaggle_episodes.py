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


def _post(method: str, body: dict[str, Any]) -> dict[str, Any]:
    url = f"{_BASE_URL}/{method}"
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
        raise RuntimeError(f"Kaggle {method} failed: HTTP {e.code} — {msg}") from e


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
