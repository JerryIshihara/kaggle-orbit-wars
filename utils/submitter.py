from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .packer import pack_agent

DEFAULT_COMPETITION = "orbit-wars"


def submit_agent(
    agent_id: str,
    note: str = "",
    bundle: bool = False,
    dry_run: bool = False,
    competition: str = DEFAULT_COMPETITION,
    kaggle_bin: str | None = None,
) -> dict:
    """Pack a registered agent and submit it to Kaggle.

    Message convention:
        "<agent_id> @ <git_sha>[-dirty][ — <note>]"

    Examples:
        "sniper @ abc1234"
        "sniper @ abc1234-dirty"
        "sniper @ abc1234 — bumped min-ships to 30"

    Rationale:
      - `agent_id` identifies which agent on the leaderboard.
      - `git_sha` makes submissions reproducible (check out this SHA to
        reproduce the packed artifact).
      - `-dirty` warns that uncommitted changes were included, so the
        artifact cannot be reconstructed from git alone.
      - `note` is optional human-readable context (what's new / what
        changed since the last submission).

    Args:
      agent_id: registered agent id (e.g. "sniper").
      note: free-text note appended after an em dash.
      bundle: if True, submit `submissions/<agent_id>.tar.gz`; else submit
        the bare `main.py`.
      dry_run: if True, print the kaggle command without executing.
      competition: kaggle competition slug.
      kaggle_bin: override path to the kaggle CLI. Falls back to
        `$KAGGLE_BIN`, then `which kaggle`.

    Returns:
      Dict with `submitted` (bool), `message`, `file`, and (when submitted)
      `stdout` from the kaggle CLI.

    Raises:
      RuntimeError if kaggle CLI is missing, creds are missing, or the
      submit command exits non-zero.
    """
    pack_result = pack_agent(agent_id, bundle=bundle)
    submit_path: Path = pack_result["bundle"] if bundle else pack_result["main"]  # type: ignore[assignment]

    sha = _git_sha()
    dirty = _git_is_dirty()
    message = _build_message(agent_id, sha, dirty, note)

    kaggle_bin = kaggle_bin or os.environ.get("KAGGLE_BIN") or shutil.which("kaggle")
    if not kaggle_bin:
        raise RuntimeError(
            "kaggle CLI not found. Install it (`pip install --user kaggle`) or "
            "set KAGGLE_BIN to its path."
        )

    cmd = [
        kaggle_bin,
        "competitions",
        "submit",
        competition,
        "-f",
        str(submit_path),
        "-m",
        message,
    ]

    print(f"[submit] file:    {submit_path}", file=sys.stderr)
    print(f"[submit] message: {message}", file=sys.stderr)
    if dry_run:
        print(f"[submit] dry run: {' '.join(cmd)}", file=sys.stderr)
        return {
            "submitted": False,
            "message": message,
            "file": str(submit_path),
            "cmd": cmd,
        }

    _check_creds()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "kaggle submit failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    stdout = result.stdout.strip()
    print(f"[submit] kaggle: {stdout}", file=sys.stderr)
    return {
        "submitted": True,
        "message": message,
        "file": str(submit_path),
        "stdout": stdout,
    }


def _build_message(agent_id: str, sha: str, dirty: bool, note: str) -> str:
    parts = [agent_id, "@", sha + ("-dirty" if dirty else "")]
    if note:
        parts.extend(["—", note])
    return " ".join(parts)


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _git_is_dirty() -> bool:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(r.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _check_creds() -> None:
    if "KAGGLE_USERNAME" in os.environ and (
        "KAGGLE_KEY" in os.environ or "KAGGLE_API_KEY" in os.environ
    ):
        if "KAGGLE_KEY" not in os.environ and "KAGGLE_API_KEY" in os.environ:
            os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_KEY"]
        return
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.exists():
        return
    raise RuntimeError(
        "Kaggle credentials not found. Set KAGGLE_USERNAME + KAGGLE_KEY env "
        "vars or populate ~/.kaggle/kaggle.json."
    )
