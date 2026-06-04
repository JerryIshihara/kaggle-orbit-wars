"""Stream a PPO trial state/log channel locally.

Examples:

  python -m agents.transformer_v2.ppo.stream_trial /tmp/ppo_trial_STATE.json
  python -m agents.transformer_v2.ppo.stream_trial gs://bucket/run/STATE.json
  python -m agents.transformer_v2.ppo.stream_trial fs://ppo_runs/run_id

For Firestore URLs this uses ppo_state.stream(), which is push-based via
DocumentReference.on_snapshot. Local/GCS URLs are polled.
"""

from __future__ import annotations

import argparse
import json

from . import ppo_state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", help="local path, gs://.../STATE.json, or fs://collection/doc")
    ap.add_argument("--side", default=ppo_state.LEARNER,
                    choices=(ppo_state.LEARNER, ppo_state.ACTOR))
    ap.add_argument("--poll-s", type=float, default=1.5)
    ap.add_argument("--stall-s", type=float, default=0.0)
    ap.add_argument("--show-progress", action="store_true",
                    help="print compact progress JSON whenever STATE changes")
    args = ap.parse_args()

    last_progress = None

    def on_state(st: dict) -> None:
        nonlocal last_progress
        if not args.show_progress:
            return
        progress = ((st.get(args.side) or {}).get("progress") or {})
        if progress == last_progress:
            return
        last_progress = progress
        print("[progress] " + json.dumps(progress, sort_keys=True), flush=True)

    def on_log(lines: list[str]) -> None:
        for line in lines:
            print(line, flush=True)

    final = ppo_state.stream(
        args.url,
        until_phases={ppo_state.PHASE_DONE, ppo_state.PHASE_ERROR},
        poll_s=args.poll_s,
        watch_side=args.side,
        on_state=on_state,
        on_log=on_log,
        stall_s=(args.stall_s or None),
    )
    print(f"[stream] final phase={final.get('phase')} iter={final.get('iter')}", flush=True)
    return 0 if final.get("phase") != ppo_state.PHASE_ERROR else 1


if __name__ == "__main__":
    raise SystemExit(main())
