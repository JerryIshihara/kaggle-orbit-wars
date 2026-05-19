"""Discord rolling-pin notifier for PPO training runs.

Pure Python — no third-party deps. Uses only urllib.request + json + os.

Public surface:
    class DiscordNotifier
    def format_iter_status(metrics, *, run_dir, phase, opponent_mode, total_iters_in_run)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DISCORD_API = "https://discord.com/api/v10"
_USER_AGENT = "kaggle-orbit-wars-ppo (python urllib)"


def _make_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }


def _do_request(
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict | None = None,
    timeout: int = 10,
) -> dict | list | None:
    """Perform an HTTP request. Returns parsed JSON or None for 204/pin responses."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return None
        return json.loads(raw)


class DiscordNotifier:
    """Posts and PATCH-updates a single pinned Discord message per training run.

    Env vars consulted:
        DISCORD_TOKEN_CLAUDE  (preferred) or DISCORD_TOKEN
        DISCORD_GUILD_ID
        DISCORD_CHANNEL_NAME  (overrides channel_name kwarg)

    If any required env var is missing, or if the first HTTP call fails,
    the notifier becomes a silent no-op for the rest of the process.
    A single [notify] disabled (...) line is printed on init or first failure.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        channel_name: str = "claude-training",
        disabled: bool = False,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._channel_id: str | None = None
        self._message_id: str | None = None
        self._guild_id: str | None = None
        self._channel_name: str = channel_name
        self._token: str | None = None
        self._headers: dict[str, str] = {}
        self._disabled = disabled
        self._started = False

        if disabled:
            print("[notify] disabled (--no-discord flag)", flush=True)
            return

        # Read env vars
        token = os.environ.get("DISCORD_TOKEN_CLAUDE") or os.environ.get("DISCORD_TOKEN")
        guild_id = os.environ.get("DISCORD_GUILD_ID")
        channel_name_env = os.environ.get("DISCORD_CHANNEL_NAME")
        if channel_name_env:
            self._channel_name = channel_name_env

        missing = []
        if not token:
            missing.append("DISCORD_TOKEN_CLAUDE or DISCORD_TOKEN")
        if not guild_id:
            missing.append("DISCORD_GUILD_ID")

        if missing:
            print(f"[notify] disabled (missing env: {', '.join(missing)})", flush=True)
            self._disabled = True
            return

        self._token = token
        self._guild_id = guild_id
        self._headers = _make_headers(token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, header: str) -> None:
        """Look up channel, post initial message, pin it, persist IDs."""
        if self._disabled:
            return
        try:
            self._channel_id = self._lookup_channel()
            if self._channel_id is None:
                return  # _lookup_channel already disabled + printed

            # POST initial message
            msg = self._post_message(header)
            if msg is None:
                self._fail("start(): POST initial message returned None")
                return
            self._message_id = str(msg["id"])

            # PIN best-effort
            try:
                _do_request(
                    "PUT",
                    f"{_DISCORD_API}/channels/{self._channel_id}/pins/{self._message_id}",
                    headers=self._headers,
                )
            except Exception as e:
                print(f"[notify] pin failed (continuing without pin): {e}", flush=True)

            # Persist
            self._persist()
            self._started = True
        except Exception as e:
            self._fail(f"start(): {type(e).__name__}: {e}")

    def update(self, content: str) -> None:
        """PATCH the rolling pinned message with new content."""
        if self._disabled:
            return
        # Resume: load IDs from disk if start() wasn't called this process
        if not self._started:
            if not self._load_persisted():
                return
        if self._channel_id is None or self._message_id is None:
            return
        content = _truncate(content, 1900)
        try:
            _do_request(
                "PATCH",
                f"{_DISCORD_API}/channels/{self._channel_id}/messages/{self._message_id}",
                headers=self._headers,
                body={"content": content},
            )
        except Exception as e:
            self._fail(f"update(): {type(e).__name__}: {e}")

    def finish(self, summary: str) -> None:
        """POST a final summary message; unpin the rolling message."""
        if self._disabled:
            return
        if not self._started:
            if not self._load_persisted():
                return
        if self._channel_id is None:
            return
        summary = _truncate(summary, 1900)
        # POST final message (best-effort)
        try:
            self._post_message(summary)
        except Exception as e:
            print(f"[notify] finish() POST failed: {e}", flush=True)

        # Unpin (best-effort)
        if self._message_id is not None:
            try:
                _do_request(
                    "DELETE",
                    f"{_DISCORD_API}/channels/{self._channel_id}/pins/{self._message_id}",
                    headers=self._headers,
                )
            except Exception as e:
                print(f"[notify] unpin failed (non-fatal): {e}", flush=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lookup_channel(self) -> str | None:
        """GET guild channels, find by name. Returns channel_id or None (disables)."""
        try:
            channels = _do_request(
                "GET",
                f"{_DISCORD_API}/guilds/{self._guild_id}/channels",
                headers=self._headers,
            )
        except Exception as e:
            self._fail(f"channel lookup failed: {type(e).__name__}: {e}")
            return None

        if not isinstance(channels, list):
            self._fail("channel lookup: unexpected response (not a list)")
            return None

        text_channels = [c for c in channels if c.get("type") == 0]
        for c in text_channels:
            if c.get("name") == self._channel_name:
                return str(c["id"])

        available = ", ".join(c.get("name", "?") for c in text_channels)
        self._fail(
            f"channel '{self._channel_name}' not found. "
            f"Available text channels: [{available}]"
        )
        return None

    def _post_message(self, content: str) -> dict | None:
        content = _truncate(content, 1900)
        return _do_request(
            "POST",
            f"{_DISCORD_API}/channels/{self._channel_id}/messages",
            headers=self._headers,
            body={"content": content},
        )  # type: ignore[return-value]

    def _persist(self) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "channel_id": self._channel_id,
            "message_id": self._message_id,
            "guild_id": self._guild_id,
            "channel_name": self._channel_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (self._run_dir / "discord_message.json").write_text(json.dumps(payload))

    def _load_persisted(self) -> bool:
        """Load IDs from discord_message.json for resume. Returns True if loaded."""
        p = self._run_dir / "discord_message.json"
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self._channel_id = data.get("channel_id")
            self._message_id = data.get("message_id")
            self._guild_id = data.get("guild_id") or self._guild_id
            self._channel_name = data.get("channel_name", self._channel_name)
            self._started = True
            return True
        except Exception as e:
            print(f"[notify] could not load discord_message.json: {e}", flush=True)
            return False

    def _fail(self, reason: str) -> None:
        print(f"[notify] disabled ({reason})", flush=True)
        self._disabled = True


# ---------------------------------------------------------------------------
# Notifier factory (used by ppo.py to handle the --no-discord flag cleanly)
# ---------------------------------------------------------------------------

def make_notifier(
    run_dir: Path,
    *,
    disabled: bool = False,
    channel_name: str = "claude-training",
) -> "DiscordNotifier":
    """Return a DiscordNotifier (or a disabled no-op if ``disabled=True``)."""
    return DiscordNotifier(run_dir, channel_name=channel_name, disabled=disabled)


# ---------------------------------------------------------------------------
# Metric formatting
# ---------------------------------------------------------------------------

def _fmt_val(v: Any, fmt: str = ".3f", nan_str: str = "nan") -> str:
    if v is None:
        return nan_str
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return nan_str


def _fmt_eta(seconds: float | None) -> str:
    """Format seconds into human-readable ETA string (ASCII only)."""
    if seconds is None or seconds != seconds:  # None or NaN
        return "?"
    secs = int(max(0, seconds))
    if secs >= 3600:
        h = secs // 3600
        m = (secs % 3600) // 60
        return f"{h}h{m:02d}m"
    elif secs >= 60:
        m = secs // 60
        s = secs % 60
        return f"{m}m{s:02d}s"
    else:
        return f"{secs}s"


def _truncate(text: str, max_len: int = 1900) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_iter_status(
    metrics: dict,
    *,
    run_dir: Path,
    phase: str,
    opponent_mode: str,
    total_iters_in_run: int,
) -> str:
    """Build a Markdown code-block body for Discord update().

    Truncated to ~1900 chars.  Fields added by T-026:
      wins/losses/draws, main_ev, horizon_ev (10/20/50),
      frac_sample_std, episodes_short_pct.
    """
    it = metrics.get("iter", "?")
    wins = metrics.get("wins")
    losses = metrics.get("losses")
    draws = metrics.get("draws")
    wld = (
        f"{wins}/{losses}/{draws}"
        if wins is not None and losses is not None and draws is not None
        else "?/?/?"
    )
    wr = _fmt_val(metrics.get("winrate_vs_physical_v4"), ".3f")
    rew = _fmt_val(metrics.get("mean_reward"), "+.3f")
    ev = _fmt_val(
        metrics.get("main_explained_variance", metrics.get("explained_variance")),
        ".3f",
    )
    # Horizon EVs: h10 / h20 / h50
    evh_parts = [
        _fmt_val(metrics.get(f"explained_variance_h{k}"), ".2f")
        for k in (10, 20, 50)
    ]
    evh = "/".join(evh_parts)
    kl = _fmt_val(metrics.get("approx_kl"), ".4f")
    ent = _fmt_val(metrics.get("entropy"), ".3f")
    fls = _fmt_val(metrics.get("frac_log_std_exp"), ".3f")
    msz = _fmt_val(metrics.get("mean_sigmoid_z"), ".3f")
    fss = _fmt_val(metrics.get("frac_sample_std"), ".3f")
    eps_short = _fmt_val(metrics.get("episodes_short_pct"), ".1f")
    shp = _fmt_val(metrics.get("shaping_coef_eff"), ".4f")
    shaped_rew = _fmt_val(metrics.get("shaped_reward_mean"), "+.4f")
    dt = _fmt_val(metrics.get("iter_seconds"), ".1f")
    eta = _fmt_eta(metrics.get("eta_seconds_remaining"))
    run_dir_str = str(run_dir)

    lines = [
        f"**PPO Training Update** — phase `{phase}` | opp `{opponent_mode}`",
        f"",
        f"```",
        f"iter              : {it}/{total_iters_in_run}",
        f"wins/losses/draws : {wld}",
        f"mean_reward       : {rew}",
        f"winrate_phys_v4   : {wr}",
        f"main_ev           : {ev}",
        f"horizon_ev 10/20/50: {evh}",
        f"approx_kl         : {kl}",
        f"entropy           : {ent}",
        f"frac_log_std_exp  : {fls}",
        f"mean_sigmoid_z    : {msz}",
        f"frac_sample_std   : {fss}",
        f"episodes_short_pct: {eps_short}%",
        f"shaped_reward_mean: {shaped_rew}",
        f"shaping_coef_eff  : {shp}",
        f"iter_seconds      : {dt}s",
        f"ETA               : {eta}",
        f"run_dir           : {run_dir_str}",
        f"```",
    ]
    body = "\n".join(lines)
    return _truncate(body, 1900)
