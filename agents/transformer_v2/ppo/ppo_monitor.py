"""Read-only streaming dashboard for the distributed PPO state machine.

A passive watcher of the single authoritative ``STATE.json`` / Firestore doc that
the GCP rollout actor(s) and Colab learner ping-pong through. It changes nothing
— it only streams STATE and prints, so it is runnable from any VM or Colab,
since STATE is the one shared source of truth (see ``docs/PPO_STATE_MACHINE.md``).

What it shows, live:
  * a header line on every **phase change** (phase / iter / iters_total);
  * the **active side's** ``progress`` dict, compactly, whenever STATE changes
    (actor during ``rolling_out``, learner during ``training``); and
  * the **active side's** new ``log_tail`` lines — a continuous tail of the
    remote rollout / train stdout, streamed over the same GCS bus.

Rather than lean on ``stream()``'s single ``watch_side`` (which can't follow the
phase flip from actor->learner mid-run), this tracks ``log_n`` per side itself in
a small closure and prints whichever side is active for the current phase. Exits
when the run reaches ``done`` (or ``error``).

Usage::

    python -m agents.transformer_v2.ppo.ppo_monitor --state gs://.../ppo/<run_id>
    python -m agents.transformer_v2.ppo.ppo_monitor --state gs://.../ppo/<run_id>/STATE.json
    python -m agents.transformer_v2.ppo.ppo_monitor --state /tmp/run --poll-s 1.0
"""

from __future__ import annotations

import argparse
import sys

from . import ppo_state

# phase -> the side that is the live, sole writer during it (whose progress + log
# we tail). await_* / done / error have no single active worker.
_ACTIVE_SIDE = {
    ppo_state.PHASE_ROLLING_OUT: ppo_state.ACTOR,
    ppo_state.PHASE_TRAINING: ppo_state.LEARNER,
}


def _resolve_state_url(state: str) -> str:
    """Accept either a run prefix or a full STATE.json url; return the url."""
    s = state.rstrip("/")
    if s.startswith("fs://"):
        return s
    if s.endswith("/STATE.json") or s.endswith("STATE.json"):
        return s
    return ppo_state.state_url(s)


def _fmt_progress_lines(p: dict | None) -> list[str]:
    """Compact render of a progress dict, with pool workers split per core."""
    if not p:
        return ["(no progress yet)"]

    workers = p.get("workers") if isinstance(p.get("workers"), list) else []
    skip = {"workers"}
    preferred = [
        "state", "ep_done", "ep_total", "cur_step", "n_active",
        "games_done", "total_steps", "steps_per_s", "mean_step",
        "rss_mb", "peak_rss_mb", "avail_mb", "cpu_pct", "elapsed_s", "eta_s",
        "policy_version",
    ]
    keys = [k for k in preferred if k in p and k not in skip]
    keys.extend(k for k in sorted(p) if k not in skip and k not in keys)

    parts = []
    for k in keys:
        v = p[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    lines = ["  ".join(parts)]
    for r in sorted(workers, key=lambda x: x.get("slot", 0)):
        lines.append(
            f"core {r.get('slot')}: seed={r.get('seed')} "
            f"{r.get('num_players')}P step={r.get('step')} "
            f"score={r.get('score_my')}-{r.get('score_enemy_max')} "
            f"threads={r.get('threads')} games_done={r.get('games_done')}"
        )
    return lines


def _side_label(side: str | None) -> str:
    if side == ppo_state.ACTOR:
        return "ACTOR"
    if side == ppo_state.LEARNER:
        return "LEARNER(Colab)"
    return "—"


def _make_on_state():
    """Build the on_state callback with its own per-side log cursors + phase memo.

    Closure state:
      ``log_seen``  -> last log_n we have already printed, per side (dedup).
      ``last_phase``-> last phase header we printed (so we print a header only on
                       a real phase change, not on every progress bump).
    """
    log_seen = {ppo_state.ACTOR: 0, ppo_state.LEARNER: 0}
    actor_log_seen: dict[str, int] = {}
    last = {"phase": None}

    def on_state(st: dict) -> None:
        phase = st.get("phase")
        it = st.get("iter")
        total = st.get("iters_total")
        active = _ACTIVE_SIDE.get(phase)
        actors = st.get("actors") or {}
        actor_phase = phase == ppo_state.PHASE_AWAIT_ROLLOUT and bool(actors)

        # --- header on phase change ---
        if phase != last["phase"]:
            who = "ACTOR(s)" if actor_phase else _side_label(active)
            bar = "=" * 12
            print(
                f"\n{bar} phase: {phase}  |  iter {it}/{total}  |  active: {who} {bar}",
                flush=True,
            )
            msg = st.get("message")
            if msg:
                print(f"  message: {msg}", flush=True)
            last["phase"] = phase

        # --- active side's live progress (compact) ---
        if active is not None:
            blk = st.get(active) or {}
            hb = blk.get("heartbeat_ts")
            lines = _fmt_progress_lines(blk.get("progress"))
            label = f"  [{_side_label(active)} iter {it}] "
            for j, line in enumerate(lines):
                print((label if j == 0 else "      ") + line
                      + ((f"  hb={hb}") if (j == 0 and hb) else ""),
                      flush=True)

            # --- active side's new log_tail lines (dedup via log_n) ---
            log_n = int(blk.get("log_n", 0))
            seen = log_seen[active]
            if log_n > seen:
                tail = blk.get("log_tail") or []
                new = log_n - seen
                for line in tail[-min(new, len(tail)):]:
                    print(f"    | {line}", flush=True)
                log_seen[active] = log_n

        # Multi-actor rollouts intentionally keep phase=await_rollout until the
        # gated barrier completes, and publish live telemetry under actors[id].
        # Render those blocks explicitly; otherwise the monitor looks idle while
        # the pool is actually running.
        if actor_phase:
            for aid in sorted(actors):
                blk = actors.get(aid) or {}
                hb = blk.get("heartbeat_ts")
                status = blk.get("status")
                lines = _fmt_progress_lines(blk.get("progress"))
                label = f"  [ACTOR {aid} iter {it}] "
                for j, line in enumerate(lines):
                    suffix = ""
                    if j == 0:
                        suffix += f"  status={status}" if status else ""
                        suffix += f"  hb={hb}" if hb else ""
                    print((label if j == 0 else "      ") + line + suffix,
                          flush=True)

                log_n = int(blk.get("log_n", 0))
                seen = actor_log_seen.get(aid, 0)
                if log_n > seen:
                    tail = blk.get("log_tail") or []
                    new = log_n - seen
                    for line in tail[-min(new, len(tail)):]:
                        print(f"    | {aid}: {line}", flush=True)
                    actor_log_seen[aid] = log_n

    return on_state


def run_monitor(state: str, *, poll_s: float = 1.5) -> int:
    url = _resolve_state_url(state)
    print(f"[ppo_monitor] watching {url}  (poll {poll_s}s) — Ctrl-C to stop", flush=True)

    st0 = ppo_state.read_state(url)
    if st0 is None:
        print(
            f"[ppo_monitor] no STATE yet at {url} — waiting for the learner's "
            f"init cell to seed it…",
            flush=True,
        )

    on_state = _make_on_state()
    try:
        final = ppo_state.stream(
            url,
            until_phases={ppo_state.PHASE_DONE},
            poll_s=poll_s,
            on_state=on_state,
            stall_s=None,          # never time out a watcher; the peer may be idle
        )
    except KeyboardInterrupt:
        print("\n[ppo_monitor] stopped (Ctrl-C).", flush=True)
        return 0

    ph = final.get("phase")
    if ph == ppo_state.PHASE_ERROR:
        print(
            f"\n[ppo_monitor] run hit ERROR at iter {final.get('iter')}: "
            f"{final.get('message')}",
            flush=True,
        )
        return 1
    print(
        f"\n[ppo_monitor] run DONE — {final.get('iter')}/{final.get('iters_total')} "
        f"iters (run_id {final.get('run_id')}).",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only PPO state-machine dashboard.")
    ap.add_argument(
        "--state",
        required=True,
        help="Run prefix (gs://.../ppo/<run_id> or a local dir) OR the STATE.json url.",
    )
    ap.add_argument(
        "--poll-s",
        type=float,
        default=1.5,
        help="STATE poll interval in seconds (default 1.5).",
    )
    args = ap.parse_args()
    return run_monitor(args.state, poll_s=args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
