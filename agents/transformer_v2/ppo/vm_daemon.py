"""PPO rollout poll-daemon — runs on the GCP VM, drives rollout_worker.

Watches ``gs://<bucket>/ppo/<run_id>/heads/`` for the newest ``policy_vK.heads.pt``
that has no matching ``rollouts/vK/<vm_id>/_DONE`` and, when one appears, invokes
``rollout_worker`` for that K. Colab's only job is to push the head-delta; the
daemon does the rest (no held SSH — resilient to Colab disconnects).

    python -m agents.transformer_v2.ppo.vm_daemon \
      --bucket gs://orbit-wars-shipping --run-id <run> --vm-id <vm_id> \
      --ckpt ckpts/entity/entity_encoder_best.pt \
      --fleet-run-dir ckpts/fleet --planet-run-dir ckpts/planet --comet-run-dir ckpts/comet \
      --base-version <base_id> --history-window 10 --episodes 16 --poll-s 15
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

from . import shards

_HEADS_RE = re.compile(r"policy_v(\d+)\.heads\.pt$")


def _latest_pending(bucket: str, run_id: str, vm_id: str) -> int | None:
    """Highest K with heads/policy_vK.heads.pt present but no rollouts/vK/<vm_id>/_DONE."""
    prefix = f"{bucket}/ppo/{run_id}"
    versions = []
    for obj in shards.gcs_list(f"{prefix}/heads/"):
        m = _HEADS_RE.search(obj)
        if m:
            versions.append(int(m.group(1)))
    for k in sorted(versions, reverse=True):
        if not shards.gcs_exists(f"{prefix}/rollouts/v{k}/{vm_id}/_DONE"):
            return k
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True, help="gs://orbit-wars-shipping")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--vm-id", required=True)
    ap.add_argument("--poll-s", type=float, default=15.0)
    # forwarded to rollout_worker
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fleet-run-dir", default=None)
    ap.add_argument("--planet-run-dir", default=None)
    ap.add_argument("--comet-run-dir", default=None)
    ap.add_argument("--base-version", default=None)
    ap.add_argument("--history-window", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--rollout-workers", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed-base", type=int, default=100000)
    ap.add_argument("--progress-every", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="Process one pending K then exit (testing).")
    args = ap.parse_args()
    prefix = f"{args.bucket}/ppo/{args.run_id}"
    print(f"[daemon] watching {prefix}/heads/ as vm={args.vm_id} (poll {args.poll_s}s)", flush=True)
    last_state = None

    while True:
        try:
            k = _latest_pending(args.bucket, args.run_id, args.vm_id)
        except Exception as e:
            print(f"[daemon] poll error: {e}", flush=True)
            time.sleep(args.poll_s)
            continue
        if k is None:
            if last_state != "idle":
                print("[daemon] no pending policy; idling", flush=True)
                last_state = "idle"
            time.sleep(args.poll_s)
            continue
        last_state = None
        heads_url = f"{prefix}/heads/policy_v{k}.heads.pt"
        out_url = f"{prefix}/rollouts/v{k}/{args.vm_id}/"
        print(f"[daemon] policy_v{k} delta detected → rollout start ({shards.utcnow_iso()})", flush=True)
        cmd = [sys.executable, "-u", "-m", "agents.transformer_v2.ppo.rollout_worker",
               "--ckpt", str(args.ckpt), "--policy-heads", heads_url,
               "--policy-version", str(k), "--history-window", str(args.history_window),
               "--episodes", str(args.episodes), "--rollout-workers", str(args.rollout_workers),
               "--device", args.device, "--out", out_url, "--vm-id", args.vm_id,
               "--seed-base", str(args.seed_base), "--progress-every", str(args.progress_every),
               "--verbose"]
        for flag, val in (("--fleet-run-dir", args.fleet_run_dir),
                          ("--planet-run-dir", args.planet_run_dir),
                          ("--comet-run-dir", args.comet_run_dir),
                          ("--base-version", args.base_version)):
            if val:
                cmd += [flag, str(val)]
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        dt = time.time() - t0
        if rc == 0:
            print(f"[daemon] policy_v{k} rollout DONE ({dt/60:.1f}m)", flush=True)
        else:
            # rollout failed without writing _DONE → it stays pending; back off so we
            # don't hot-loop on a persistent failure.
            print(f"[daemon] policy_v{k} rollout FAILED rc={rc} ({dt:.0f}s); backing off 60s", flush=True)
            time.sleep(60)
        if args.once:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
