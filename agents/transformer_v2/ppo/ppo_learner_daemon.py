"""PPO learner listen-loop — runs on the Colab GPU side of the distributed loop.

The learner half of the central-STATE machine in ``docs/PPO_STATE_MACHINE.md``.
It polls the authoritative ``STATE.json`` (GCS or local) and acts only on its
phase, ``await_train``: it claims the side, flips to ``training``, runs
``learner_step`` as a subprocess for iter ``K`` (consuming the actor's
``rollouts/vK/`` shards), streams the subprocess stdout — both to THIS console and
into ``STATE['learner']`` (live ``progress`` + a rolling log tail) so a watcher on
any console sees the training advance — and on success flips to the next phase
(``await_rollout K+1`` with ``model=v{K+1}``, or ``done`` on the last iter).

When it is the *actor's* turn (``await_rollout`` / ``rolling_out``) the loop
instead streams the actor side live, so the user watching the learner console
also sees the GCP rollout VM heartbeats — the same Firestore/GCS bus, no extra infra.

Invariants it upholds (see the doc):
  * artifacts before STATE — ``learner_step`` uploads ``policy_v{K+1}.pt`` + heads
    BEFORE this loop flips the phase, so the actor never sees a dangling model.
  * idempotent / resumable — if ``checkpoints/policy_v{K+1}.pt`` already exists it
    skips the (re)train and just flips the phase; output files are ground truth.

    python -m agents.transformer_v2.ppo.ppo_learner_daemon \
      --state gs://orbit-wars-shipping/ppo/<run_id> \
      --ckpt ckpts/entity/entity_encoder_best.pt \
      --fleet-run-dir ckpts/fleet --planet-run-dir ckpts/planet --comet-run-dir ckpts/comet \
      --device cuda --poll-s 10
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

from . import ppo_state, shards

# Matches one learner_step per-epoch line, e.g.
#   "[learner]   epoch 1: kl=0.009 policy_loss=-0.02 value_loss=0.04 entropy=2.1 clip_frac=0.08"
# (a trailing " EARLY-STOP" is tolerated). Best-effort: floats may be negative /
# scientific; missing fields just don't populate the parsed dict.
_EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+):\s+"
    r"kl=(?P<kl>[-+0-9.eE]+)\s+"
    r"policy_loss=(?P<policy_loss>[-+0-9.eE]+)\s+"
    r"value_loss=(?P<value_loss>[-+0-9.eE]+)\s+"
    r"entropy=(?P<entropy>[-+0-9.eE]+)\s+"
    r"clip_frac=(?P<clip_frac>[-+0-9.eE]+)"
)

_FLUSH_EVERY_S = 3.0   # how often we mirror stdout -> STATE['learner'] while training


def _log(msg: str) -> None:
    print(f"[learner-daemon {shards.utcnow_iso()}] {msg}", flush=True)


def _resolve_state_url(arg: str) -> str:
    """Resolve the ``--state`` arg into the STATE url to read/write.

    The control channel may be Firestore (``fs://<collection>/<doc>``) or a gs:///
    local STATE.json. We accept all three forms:
      * ``fs://...``       -> used VERBATIM (a Firestore doc has no /STATE.json child).
      * ``.../STATE.json`` -> verbatim (already the full url).
      * otherwise (a run prefix) -> ``ppo_state.state_url(arg)`` (append /STATE.json).
    """
    raw = str(arg)
    if raw.startswith("fs://"):
        return raw
    if raw.rstrip("/").endswith("STATE.json"):
        return raw
    return ppo_state.state_url(raw)


def _resolve_run_gcs(st: dict, url: str) -> str:
    """The gs:// run root for BIG files (checkpoints/heads/rollouts).

    Read from ``STATE['gcs_prefix']`` (set at init_state) — this is decoupled from
    the (possibly fs://) control-channel url. Back-compat: if STATE has no
    gcs_prefix AND the state url is itself gs://, derive it from the STATE.json
    parent dir (the old behavior). Otherwise fail loud."""
    run_gcs = ppo_state.gcs_prefix(st)
    if run_gcs is None and str(url).startswith("gs://"):
        run_gcs = str(url).rstrip("/")[: -len("/STATE.json")]
    if run_gcs is None:
        raise RuntimeError(
            "STATE has no gcs_prefix — set it via init_state(..., gcs_prefix=...)")
    return run_gcs.rstrip("/")


def _parse_epoch(line: str) -> dict | None:
    """Best-effort parse of a per-epoch learner_step line into a small float dict."""
    m = _EPOCH_RE.search(line)
    if not m:
        return None
    out: dict = {}
    for k, v in m.groupdict().items():
        try:
            out[k] = int(v) if k == "epoch" else float(v)
        except (TypeError, ValueError):
            pass
    return out or None


def _fmt_rollout_progress(progress: dict | None) -> list[str]:
    """Compact rollout progress lines, preserving per-core pool telemetry."""
    if not progress:
        return ["(no progress yet)"]
    workers = progress.get("workers") if isinstance(progress.get("workers"), list) else []
    keys = [
        "state", "ep_done", "ep_total", "cur_step", "n_active",
        "games_done", "total_steps", "steps_per_s", "mean_step",
        "rss_mb", "peak_rss_mb", "avail_mb", "cpu_pct", "elapsed_s", "eta_s",
        "policy_version",
    ]
    parts = []
    for k in keys:
        if k not in progress:
            continue
        v = progress[k]
        parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    lines = ["  ".join(parts) if parts else "(progress present)"]
    for row in sorted(workers, key=lambda r: r.get("slot", 0)):
        lines.append(
            f"core {row.get('slot')}: seed={row.get('seed')} "
            f"{row.get('num_players')}P step={row.get('step')} "
            f"score={row.get('score_my')}-{row.get('score_enemy_max')} "
            f"threads={row.get('threads')} games_done={row.get('games_done')}"
        )
    return lines


def _make_actor_stream_printer():
    """State-stream callback for rollout phases.

    With ``EXPECTED_ACTORS`` set, the actor daemon keeps phase=await_rollout until
    the gated barrier completes and writes telemetry under ``STATE['actors'][id]``.
    The older single-actor path writes ``STATE['actor']`` while phase=rolling_out.
    This callback renders both so Colab never looks idle while GCP VMs are rolling.
    """
    actor_log_seen: dict[str, int] = {}
    legacy_log_seen = {"n": 0}

    def _print_block(label: str, blk: dict, *, actor_id: str | None = None) -> None:
        progress = blk.get("progress")
        status = blk.get("status")
        hb = blk.get("heartbeat_ts")
        lines = _fmt_rollout_progress(progress)
        for j, line in enumerate(lines):
            suffix = ""
            if j == 0:
                suffix += f" status={status}" if status else ""
                suffix += f" hb={hb}" if hb else ""
            print((label if j == 0 else "    ") + line + suffix, flush=True)

        log_n = int(blk.get("log_n", 0))
        if actor_id is None:
            seen = legacy_log_seen["n"]
        else:
            seen = actor_log_seen.get(actor_id, 0)
        if log_n > seen:
            tail = blk.get("log_tail") or []
            new = log_n - seen
            for line in tail[-min(new, len(tail)):]:
                print(f"    | {line}", flush=True)
            if actor_id is None:
                legacy_log_seen["n"] = log_n
            else:
                actor_log_seen[actor_id] = log_n

    def on_state(st: dict) -> None:
        iter_ = st.get("iter")
        actors = st.get("actors") or {}
        if actors:
            for aid in sorted(actors):
                _print_block(f"[rollout {aid} iter {iter_}] ", actors.get(aid) or {},
                             actor_id=aid)
            return
        _print_block(f"[rollout actor iter {iter_}] ", st.get(ppo_state.ACTOR) or {})

    return on_state


def _build_learner_cmd(args, *, run_prefix: str, K: int, base_version: str,
                       shards_url: str, out_ckpt: str, out_heads: str) -> list[str]:
    """The exact ``learner_step`` subprocess invocation for iter ``K`` -> ``v{K+1}``.

    Mirrors notebooks/ppo_gpu_colab.ipynb §4 (ENC + HP), plus the run-prefix-derived
    in/out paths. ``--pair-cache-path`` / ``--bc-coef`` are appended only when a pair
    cache is configured (BC anchor on); otherwise BC stays off (the learner default).
    """
    cmd = [
        sys.executable, "-u", "-m", "agents.transformer_v2.ppo.learner_step",
        "--policy-ckpt", f"{run_prefix}/checkpoints/policy_v{K}.pt",
        "--shards", shards_url,
        "--out-ckpt", out_ckpt,
        "--out-heads", out_heads,
        "--train-log", f"{run_prefix}/train_log.jsonl",
        "--policy-version", str(K),
        "--base-version", base_version,
        "--device", args.device,
        "--ckpt", str(args.ckpt),
        # PPO hyperparams (notebook §4 HP list)
        "--minibatch-size", str(args.minibatch_size),
        "--epochs", str(args.epochs),
        "--clip", str(args.clip),
        "--target-kl", str(args.target_kl),
        "--lr-heads", str(args.lr_heads),
        "--value-coef", str(args.value_coef),
        "--ent-coef", str(args.ent_coef),
        "--sigma", str(args.sigma),
        # which encoder stages to UNFREEZE this run (forwarded to learner_step;
        # 0 = heads-only, the default). A sibling task added this flag to
        # learner_step.py; we just pass it through.
        "--freeze-phase", str(args.freeze_phase),
    ]
    # frozen-base encoder run dirs (notebook §4 ENC); only forward when given.
    for flag, val in (("--fleet-run-dir", args.fleet_run_dir),
                      ("--planet-run-dir", args.planet_run_dir),
                      ("--comet-run-dir", args.comet_run_dir)):
        if val:
            cmd += [flag, str(val)]
    if args.pair_cache_path:
        cmd += ["--pair-cache-path", str(args.pair_cache_path), "--bc-coef", str(args.bc_coef)]
    if args.lr_trunk is not None:
        cmd += ["--lr-trunk", str(args.lr_trunk)]
    return cmd


def _run_training(args, url: str, learner_id: str, *, run_prefix: str, K: int,
                  base_version: str, shards_url: str, out_ckpt: str, out_heads: str) -> int:
    """Run learner_step for iter K, streaming stdout to this console + STATE['learner'].

    Returns the subprocess exit code (0 == success). The caller owns the phase
    flip; this only does the work + the live mirror.
    """
    cmd = _build_learner_cmd(args, run_prefix=run_prefix, K=K, base_version=base_version,
                             shards_url=shards_url, out_ckpt=out_ckpt, out_heads=out_heads)
    _log("learner_step cmd: " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)

    pending: list[str] = []          # stdout lines accumulated since the last flush
    last_progress: dict | None = None  # most-recent successfully-parsed epoch dict
    last_flush = time.time()

    def flush() -> None:
        nonlocal pending, last_flush
        if not pending and last_progress is None:
            return
        ppo_state.set_progress(url, ppo_state.LEARNER, progress=last_progress,
                               log_lines=pending or None, who=learner_id)
        pending = []
        last_flush = time.time()

    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)            # local echo (THIS console)
        pending.append(line)
        parsed = _parse_epoch(line)
        if parsed is not None:
            last_progress = parsed         # keep the latest parsed epoch metrics
        if (time.time() - last_flush) >= _FLUSH_EVERY_S:
            flush()
    flush()                                # final flush after the stream closes
    return p.wait()


def _do_await_train(args, url: str, learner_id: str, st: dict) -> int | None:
    """Handle one ``await_train`` iteration. Returns an exit code to stop the loop,
    or ``None`` to continue polling."""
    K = int(st["iter"])
    base_version = args.base_version or st["model"]["base_version"]
    # BIG-file (gs://) paths come from STATE['gcs_prefix'], NOT the control-channel
    # url (which may be fs://). Resolved per-iter so a mid-run STATE edit is honored.
    run_gcs = _resolve_run_gcs(st, url)
    out_ckpt = f"{run_gcs}/checkpoints/policy_v{K + 1}.pt"
    out_heads = f"{run_gcs}/heads/policy_v{K + 1}.heads.pt"
    next_phase = (ppo_state.PHASE_DONE if (K + 1 >= int(st["iters_total"]))
                  else ppo_state.PHASE_AWAIT_ROLLOUT)
    new_model = ppo_state.model_block(run_gcs, K + 1, base_version)

    # Idempotent / resumable: the checkpoint is ground truth. If v{K+1} already
    # exists (a crashed/restarted learner), skip the retrain and just flip.
    if shards.gcs_exists(out_ckpt):
        _log(f"iter {K}: policy_v{K + 1}.pt already exists → skip train, flip to {next_phase}")
        ppo_state.transition(url, expect_phase=ppo_state.PHASE_AWAIT_TRAIN,
                             new_phase=next_phase, who=learner_id,
                             iter=K + 1, model=new_model, shards=None)
        return 0 if args.once else None

    # Claim the side (lease) then flip await_train -> training (we are now the owner).
    ppo_state.claim_side(url, ppo_state.LEARNER, who=learner_id, iter_=K)
    ppo_state.transition(url, expect_phase=ppo_state.PHASE_AWAIT_TRAIN,
                         new_phase=ppo_state.PHASE_TRAINING, who=learner_id)

    shards_url = st.get("shards") or f"{run_gcs}/rollouts/v{K}/"
    _log(f"iter {K}: TRAINING (shards={shards_url}, base={base_version}, device={args.device})")
    try:
        rc = _run_training(args, url, learner_id, run_prefix=run_gcs, K=K,
                           base_version=base_version, shards_url=shards_url,
                           out_ckpt=out_ckpt, out_heads=out_heads)
    except Exception as e:  # subprocess/streaming blew up -> surface as ERROR, halt.
        _log(f"iter {K}: training raised {e!r} → ERROR")
        ppo_state.transition(url, expect_phase=None, new_phase=ppo_state.PHASE_ERROR,
                             who=learner_id, message=f"train v{K} exception: {e}")
        return 1

    if rc == 0:
        _log(f"iter {K}: train OK → {next_phase} (iter={K + 1}, model=v{K + 1})")
        ppo_state.transition(url, expect_phase=ppo_state.PHASE_TRAINING,
                             new_phase=next_phase, who=learner_id,
                             iter=K + 1, model=new_model, shards=None)
        return 0 if args.once else None

    _log(f"iter {K}: learner_step exit {rc} → ERROR")
    ppo_state.transition(url, expect_phase=None, new_phase=ppo_state.PHASE_ERROR,
                         who=learner_id, message=f"train v{K} exit {rc}")
    return 1


def loop(args, url: str, learner_id: str) -> int:
    """The learner listen-loop. Returns a process exit code.

    Big-file (gs://) paths are resolved per-iteration from ``STATE['gcs_prefix']``
    (see ``_do_await_train`` / ``_resolve_run_gcs``), so the control-channel ``url``
    may be fs:// with no derivable parent dir."""
    _log(f"watching {url} as learner={learner_id} "
         f"(poll {args.poll_s}s{', once' if args.once else ''})")
    rollout_timeout = float(getattr(args, "rollout_timeout", 0.0) or 0.0)
    wait_since: dict = {}   # iter K -> ts we first saw await_rollout (for the timeout)
    actor_stream_printer = _make_actor_stream_printer()
    while True:
        st = ppo_state.read_state(url)
        if st is None:
            _log(f"no STATE at {url} yet; waiting")
            if args.once:
                return 0
            time.sleep(args.poll_s)
            continue

        phase = st.get("phase")

        # 1) terminal / past the end -> ensure DONE, exit 0.
        if phase == ppo_state.PHASE_DONE or int(st.get("iter", 0)) >= int(st["iters_total"]):
            if phase != ppo_state.PHASE_DONE:
                _log(f"iter {st.get('iter')} >= iters_total {st['iters_total']} → marking DONE")
                ppo_state.transition(url, expect_phase=None, new_phase=ppo_state.PHASE_DONE,
                                     who=learner_id)
            else:
                _log("phase=done → exiting")
            return 0

        # 2) error -> log + halt non-zero.
        if phase == ppo_state.PHASE_ERROR:
            _log(f"phase=error (message={st.get('message')!r}) → halting")
            return 1

        # 3) our turn.
        if phase == ppo_state.PHASE_AWAIT_TRAIN:
            rc = _do_await_train(args, url, learner_id, st)
            if rc is not None:
                return rc
            continue

        # 4) actor's turn (await_rollout / rolling_out): watch GCP rollout VMs live.
        if args.once:
            _log(f"phase={phase} (actor's turn) and --once → returning")
            time.sleep(min(args.poll_s, 1.0))
            return 0

        # 4a) optional rollout timeout: if we've been waiting in await_rollout for
        # iter K longer than --rollout-timeout, force-train on whatever shards have
        # landed under the PARENT rollouts/vK/ (the union written so far) rather
        # than block on a slow/dead actor's gated flip.
        if rollout_timeout > 0 and phase == ppo_state.PHASE_AWAIT_ROLLOUT:
            K = int(st["iter"])
            t0 = wait_since.setdefault(K, time.time())
            waited = time.time() - t0
            if waited >= rollout_timeout:
                run_gcs = _resolve_run_gcs(st, url)
                parent = f"{run_gcs}/rollouts/v{K}/"
                _log(f"ROLLOUT TIMEOUT: waited {waited:.0f}s >= {rollout_timeout:.0f}s "
                     f"in await_rollout iter {K} → forcing await_train on the "
                     f"shards landed so far ({parent}); some actors may be missing")
                try:
                    ppo_state.transition(
                        url, expect_phase=ppo_state.PHASE_AWAIT_ROLLOUT,
                        new_phase=ppo_state.PHASE_AWAIT_TRAIN, who=learner_id,
                        shards=parent, iter=K)
                except ppo_state.StatePhaseError:
                    _log("an actor flipped to await_train first; ok")
                wait_since.pop(K, None)
                continue

        _log(f"phase={phase} (actor's turn) → streaming rollout actors until await_train/done/error")
        try:
            ppo_state.stream(
                url,
                until_phases={ppo_state.PHASE_AWAIT_TRAIN, ppo_state.PHASE_DONE,
                              ppo_state.PHASE_ERROR},
                poll_s=args.poll_s,
                watch_side=None,
                on_state=actor_stream_printer,
                # bound the wait so the timeout check (4a) gets a turn; the stream
                # returns when STATE is unchanged this long (it raises TimeoutError,
                # which we swallow). Only armed when a timeout is configured.
                stall_s=(min(args.poll_s, rollout_timeout) if rollout_timeout > 0 else None),
            )
        except TimeoutError:
            pass  # timeout-armed re-poll: loop back to the 4a timeout check.
        except Exception as e:
            _log(f"stream(actor) raised {e!r}; re-polling")
            time.sleep(args.poll_s)
        # loop back: re-read STATE and act on the (likely) await_train.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True,
                    help="Run PREFIX (dir), full .../STATE.json url (gs:// or local), "
                         "or an fs://<collection>/<doc> Firestore control channel "
                         "(big-file gs:// paths come from STATE['gcs_prefix']).")
    ap.add_argument("--learner-id", default="colab", help="This learner's id (lease tag).")
    # frozen base + encoders forwarded to learner_step
    ap.add_argument("--ckpt", required=True, help="Frozen entity-encoder ckpt (local or gs://).")
    ap.add_argument("--planet-run-dir", default=None)
    ap.add_argument("--fleet-run-dir", default=None)
    ap.add_argument("--comet-run-dir", default=None)
    ap.add_argument("--device", default="cuda")
    # PPO hyperparams (notebook §4 defaults)
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--clip", type=float, default=0.10)
    ap.add_argument("--target-kl", type=float, default=0.01)
    ap.add_argument("--lr-heads", type=float, default=1e-4)
    ap.add_argument("--lr-trunk", type=float, default=None,
                    help="lr for unfrozen trunk/L3/L4 (default=lr_heads); set smaller "
                         "than lr_heads at --freeze-phase>=1. Passthrough to learner_step.")
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--pair-cache-path", default=None,
                    help="If set, BC anchor on: forwards --pair-cache-path + --bc-coef.")
    ap.add_argument("--bc-coef", type=float, default=0.0)
    ap.add_argument("--freeze-phase", type=int, default=0,
                    help="Encoder-unfreeze stage forwarded to learner_step "
                         "(0 = heads-only; default). Passthrough only.")
    ap.add_argument("--rollout-timeout", type=float, default=0.0,
                    help="If >0, and the loop has been waiting in await_rollout for "
                         "longer than this many seconds, train on whatever shards "
                         "exist under rollouts/vK/ (the union written so far). "
                         "0 = disabled (wait for the actors' gated flip).")
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--base-version", default=None,
                    help="Override the frozen-base id; else read STATE['model']['base_version'].")
    ap.add_argument("--once", action="store_true",
                    help="Do at most one training iteration then exit (notebook driver).")
    args = ap.parse_args()

    # --state may be a run prefix, a full .../STATE.json url, or an fs:// Firestore
    # control channel. Big-file (gs://) paths come from STATE['gcs_prefix'], NOT
    # this url's parent dir (which doesn't exist for fs://) — see _resolve_run_gcs.
    url = _resolve_state_url(args.state)

    try:
        return loop(args, url, args.learner_id)
    except KeyboardInterrupt:
        _log("interrupted → exiting")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
