"""PPO rollout-ACTOR daemon — runs on a GCP CPU VM, driven by the state machine.

This is the **actor** half of the distributed PPO ping-pong described in
``docs/PPO_STATE_MACHINE.md``. Instead of watching raw file presence (the older
``vm_daemon.py``), it polls the single authoritative ``STATE.json`` and acts only
on its phase: when STATE is ``await_rollout`` (model ``vK`` is ready), the VM claims
the actor side, downloads the ``policy_vK`` head-delta, runs ``rollout_worker`` for that
K, streams the worker's stdout + ``progress.json`` back into STATE (so Colab/the
user watch the rollout live over the same GCS bus), and on success flips the phase
to ``await_train`` with ``shards=rollouts/vK/<actor_id>/``. The learner (Colab) takes
it from there. No held SSH; resilient to disconnects; idempotent on restart.

Launched on each rollout VM via ``nohup``::

    nohup .venv/bin/python -u -m agents.transformer_v2.ppo.ppo_actor_daemon \
      --state fs://ppo_runs/<run_id> --actor-id c4-0 \
      --ckpt ckpts/entity/entity_encoder_best.pt \
      --fleet-run-dir ckpts/fleet --planet-run-dir ckpts/planet --comet-run-dir ckpts/comet \
      --device cpu --num-envs 8 --poll-s 10 > actor_daemon.log 2>&1 &

Invariants it upholds (see the doc): it uploads ALL artifacts (shards + ``_DONE``,
written by the worker) BEFORE flipping the phase, and it short-circuits on an
already-present ``_DONE`` — output files are ground truth, STATE is the hint.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import ppo_state, shards


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [actor-daemon] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# config resolution: STATE['config'] is authoritative, CLI is override/fallback
# --------------------------------------------------------------------------- #
def _cfg_get(state: dict, key: str, cli_val, default):
    """Pick a value: CLI override (if not None) wins, else STATE['config'][key],
    else the supplied default."""
    if cli_val is not None:
        return cli_val
    cfg = state.get("config") or {}
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


def _resolve_state_url(arg: str) -> str:
    """Resolve the ``--state`` arg into the STATE url to read/write.

    The control channel may be Firestore (``fs://<collection>/<doc>``) or a gs:///
    local STATE.json. We accept all three forms:
      * ``fs://...``      -> used VERBATIM (a Firestore doc has no /STATE.json child).
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


# --------------------------------------------------------------------------- #
# one rollout (the actor's work for a single iteration K)
# --------------------------------------------------------------------------- #
def _build_worker_cmd(
    *, ckpt, planet_run_dir, fleet_run_dir, comet_run_dir, policy_heads,
    base_version, policy_version, history_window, episodes, num_players,
    num_envs, max_planets, max_fleets, device, sigma, out_prefix, actor_id,
    seed_base, progress_every, pool_procs=0,
) -> list[str]:
    """The exact ``rollout_worker`` invocation (mirrors notebooks/ppo_gpu_colab.ipynb
    §4, single-worker on the rollout VM). ``sys.executable`` = this daemon's interpreter.

    ``pool_procs > 0`` appends ``--pool-procs N``: rollout_worker then runs the COW
    fork pool (one model load in that process, forked across N cores) instead of the
    serial/batched path — the preferred single-process actor path (one model in RAM,
    not N)."""
    cmd = [
        sys.executable, "-u", "-m", "agents.transformer_v2.ppo.rollout_worker",
        "--ckpt", str(ckpt),
        "--planet-run-dir", str(planet_run_dir),
        "--fleet-run-dir", str(fleet_run_dir),
        "--comet-run-dir", str(comet_run_dir),
        "--policy-heads", str(policy_heads),
        "--base-version", str(base_version),
        "--policy-version", str(policy_version),
        "--history-window", str(history_window),
        "--episodes", str(episodes),
        "--num-players", str(num_players),
        "--num-envs", str(num_envs),
        "--rollout-workers", "1",
        "--max-planets", str(max_planets),
        "--max-fleets", str(max_fleets),
        "--device", str(device),
        "--sigma", str(sigma),
        "--out", out_prefix,
        "--vm-id", str(actor_id),
        "--seed-base", str(seed_base),
        "--progress-every", str(progress_every),
        "--verbose",
    ]
    if pool_procs and int(pool_procs) > 0:
        cmd += ["--pool-procs", str(int(pool_procs))]
    return cmd


def _actor_index(state: dict, actor_id: str) -> tuple[int, int]:
    """``(index, n_actors)`` for this actor: its position in ``expected_actors``
    (0 if single-actor / not found) and the roster size N (1 if single-actor).
    Drives the distinct per-actor seed slice so actors roll non-overlapping
    games."""
    roster = ppo_state.expected_actors(state)
    if not roster:
        return 0, 1
    n = len(roster)
    idx = roster.index(actor_id) if actor_id in roster else 0
    return idx, n


def _partition_episodes(total: int, n_workers: int) -> list[int]:
    """Split ``total`` episodes across ``n_workers`` so the sum == total.

    ``per_worker = ceil(total / N)``; every worker gets ``per_worker`` except the
    last, which is clamped to the remainder (>= 0) so the sum is exactly ``total``.
    For the "1 game per core" launch (``--workers N --episodes N``) this yields N
    workers of 1 episode each."""
    n = max(1, int(n_workers))
    total = max(0, int(total))
    per = math.ceil(total / n) if total else 0
    counts = []
    remaining = total
    for _ in range(n):
        c = min(per, remaining)
        counts.append(c)
        remaining -= c
    return counts


def _worker_omp_env() -> dict:
    """A copy of ``os.environ`` pinned to 1 BLAS/OMP thread per process, so N
    parallel ``rollout_worker`` processes map onto N cores (one game per core)."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def _aggregate_worker_progress(progress_urls: list[str]) -> dict | None:
    """Roll up N per-worker ``progress.json`` snapshots into one actor-level dict:
    SUM ``ep_done``/``ep_total``/``n_active``, a representative (max) ``cur_step``/
    ``elapsed_s``/``eta_s``, and ``state='rolling'`` until every worker reports
    ``done``. Returns ``None`` if no worker has published progress yet."""
    snaps = [shards.read_progress(u) for u in progress_urls]
    snaps = [s for s in snaps if s]
    if not snaps:
        return None

    def _sum(key: str) -> float:
        return sum(float(s.get(key) or 0) for s in snaps)

    def _max(key: str) -> float:
        return max((float(s.get(key) or 0) for s in snaps), default=0.0)

    n_seen = len(snaps)
    all_done = (n_seen == len(progress_urls)
                and all(s.get("state") == "done" for s in snaps))
    return {
        "policy_version": snaps[0].get("policy_version"),
        "ep_done": int(_sum("ep_done")),
        "ep_total": int(_sum("ep_total")),
        "cur_step": int(_max("cur_step")),
        "n_active": int(_sum("n_active")),
        "state": "done" if all_done else "rolling",
        "rss_mb": round(_sum("rss_mb"), 1),
        "avail_mb": round(_max("avail_mb"), 1),
        "peak_rss_mb": round(_sum("peak_rss_mb"), 1),
        "cpu_pct": round(_sum("cpu_pct"), 1),
        "elapsed_s": round(_max("elapsed_s"), 1),
        "eta_s": round(_max("eta_s"), 1),
        "n_workers": len(progress_urls),
        "ts": shards.utcnow_iso(),
    }


def _run_fanned_rollout(url: str, state: dict, args, *, n_workers: int,
                        out_prefix: str, base_version, local_heads, policy_heads,
                        K, episodes, num_players, history_window, max_planets,
                        max_fleets, device, sigma, actor_id, seed_base,
                        progress_every, num_envs) -> int:
    """Fan-out path (``--workers N>1``): launch N ``rollout_worker`` PROCESSES in
    PARALLEL, each on its own core (1 thread/proc), each rolling a DISTINCT seed
    sub-slice into ``{out_prefix}w{j}/``. Periodically roll the per-worker
    ``progress.json`` up into ``state['actors'][actor_id]`` (via ``update_actor``)
    so watchers see actor-level progress; ``wait()`` for ALL processes. Returns 0
    iff every worker exited 0 (else the first nonzero rc — caller -> PHASE_ERROR).

    On success the per-worker ``_DONE``s live under ``w{j}/`` but the barrier
    (``all_actors_done``) checks ``{actor_id}/_DONE`` — so we write the actor-level
    ``_DONE`` marker at ``out_prefix`` here, AFTER all workers exit 0."""
    counts = _partition_episodes(episodes, n_workers)
    env = _worker_omp_env()

    procs: list[subprocess.Popen] = []
    progress_urls: list[str] = []
    worker_seed_base = seed_base   # actor slice base; each worker offsets by its eps
    for j in range(n_workers):
        w_eps = counts[j]
        w_out = f"{out_prefix}w{j}/"
        cmd = _build_worker_cmd(
            ckpt=args.ckpt, planet_run_dir=args.planet_run_dir,
            fleet_run_dir=args.fleet_run_dir, comet_run_dir=args.comet_run_dir,
            policy_heads=policy_heads, base_version=base_version,
            policy_version=K, history_window=history_window, episodes=w_eps,
            num_players=num_players, num_envs=num_envs, max_planets=max_planets,
            max_fleets=max_fleets, device=device, sigma=sigma, out_prefix=w_out,
            actor_id=actor_id, seed_base=worker_seed_base,
            progress_every=progress_every,
        )
        _log(f"  worker w{j}: episodes={w_eps} seed_base={worker_seed_base} "
             f"out={w_out}")
        _log("  launching rollout_worker:\n    " + " ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env=env)
        procs.append(p)
        progress_urls.append(w_out + "progress.json")
        # next worker's DISTINCT sub-slice starts after this worker's episodes
        worker_seed_base += w_eps

    # --- periodic actor-level rollup of the per-worker progress.json while the
    # processes run; capture a few recent stdout lines per beat (non-blocking is
    # awkward across N pipes, so we poll progress.json on a timer and drain stdout
    # opportunistically when a worker has exited). -------------------------------
    last_flush = 0.0

    def _rollup(status: str) -> None:
        prog = _aggregate_worker_progress(progress_urls)
        done_n = sum(1 for p in procs if p.poll() is not None)
        line = (f"[fanout] {actor_id}: {done_n}/{n_workers} workers exited"
                + (f"; ep_done={prog['ep_done']}/{prog['ep_total']}" if prog else ""))
        ppo_state.update_actor(url, actor_id, progress=prog, log_lines=[line],
                               status=status, who=actor_id)

    while any(p.poll() is None for p in procs):
        now = time.time()
        if (now - last_flush) >= max(0.1, progress_every):
            _rollup("rolling")
            last_flush = now
        time.sleep(min(0.5, max(0.05, progress_every)))

    # all processes have a returncode now; wait() to reap + collect rcs.
    rcs = [p.wait() for p in procs]
    # drain any leftover stdout so it lands in the daemon log.
    for j, p in enumerate(procs):
        if p.stdout is not None:
            for ln in p.stdout:
                _log(f"  w{j}| " + ln.rstrip("\n"))

    bad = [(j, rc) for j, rc in enumerate(rcs) if rc != 0]
    if bad:
        _rollup("error")
        j0, rc0 = bad[0]
        _log(f"worker w{j0} exited {rc0} (rcs={rcs}) -> actor error")
        return rc0

    # final actor-level rollup, then write the ACTOR-LEVEL _DONE marker at
    # out_prefix (the per-worker _DONEs live under w{j}/, but all_actors_done
    # checks {actor_id}/_DONE). A tiny empty marker file is enough; the learner's
    # _collect_shards rglobs the per-worker manifest.json under w{j}/.
    _rollup("done")
    with tempfile.NamedTemporaryFile("w", suffix="_DONE", delete=False) as fh:
        fh.write("")
        marker = fh.name
    try:
        shards.gcs_cp(marker, out_prefix + "_DONE")
    finally:
        try:
            Path(marker).unlink()
        except Exception:
            pass
    _log(f"all {n_workers} workers exited 0; wrote actor-level _DONE at "
         f"{out_prefix}_DONE")
    return 0


def _cleanup_after_rollout() -> None:
    """Free VM resources after a rollout slice so the NEXT iter starts clean:
    drop leftover rollout_worker temp dirs (local T=10 shards are ~1.5 GB each)
    and clear /dev/shm leftovers from the fork pool's ``file_system`` tensor
    sharing. The VM is dedicated to rollouts, so clearing /dev/shm wholesale is
    safe. Best-effort — cleanup must never fail an otherwise-successful rollout."""
    import os as _os
    import glob as _glob
    import shutil as _shutil
    n_dir = n_shm = 0
    for d in _glob.glob("/tmp/ppo_rollout_*"):
        _shutil.rmtree(d, ignore_errors=True)
        n_dir += 1
    if _os.path.isdir("/dev/shm"):
        for f in _glob.glob("/dev/shm/*"):
            try:
                if _os.path.isdir(f):
                    _shutil.rmtree(f, ignore_errors=True)
                else:
                    _os.unlink(f)
                n_shm += 1
            except OSError:
                pass
    _log(f"cleanup: dropped {n_dir} rollout tmpdir(s) + {n_shm} /dev/shm entr(ies) "
         f"— VM ready for the next rollout")


def _run_rollout(url: str, state: dict, args, workdir: Path, *,
                 do_phase_flip: bool = True) -> int:
    """Claim + roll out THIS actor's slice for iter K, streaming worker
    stdout/progress into STATE. Returns the process exit code. Raises on any setup
    error (caller turns it into a PHASE_ERROR transition).

    ``do_phase_flip``: single-actor flips ``await_rollout -> rolling_out`` here (it
    owns the phase). Multi-actor actors stay in ``await_rollout`` (no per-actor
    claim of the shared phase) and let the gated barrier flip to ``await_train``,
    so this passes ``do_phase_flip=False`` and uses ``update_actor`` for progress."""
    actor_id = args.actor_id
    run_gcs = _resolve_run_gcs(state, url)   # gs:// root for big files (from STATE)
    K = int(state["iter"])
    out_prefix = f"{run_gcs}/rollouts/v{K}/{actor_id}/"
    index, n_actors = _actor_index(state, actor_id)

    # config (STATE wins, CLI overrides/falls back)
    episodes = int(_cfg_get(state, "episodes", args.episodes, 16))
    num_players = _cfg_get(state, "num_players", args.num_players, "mix")
    history_window = int(_cfg_get(state, "history_window", args.history_window, 10))
    seed_base0 = int(_cfg_get(state, "seed_base", None, 100000))
    max_fleets = int(_cfg_get(state, "max_fleets", args.max_fleets, 512))
    sigma = float(_cfg_get(state, "sigma", args.sigma, 0.35))
    device = _cfg_get(state, "device", args.device, "cpu")
    max_planets = int(args.max_planets)
    progress_every = float(args.progress_every)
    n_workers = max(1, int(getattr(args, "workers", 1) or 1))
    pool_procs = max(0, int(getattr(args, "pool_procs", 0) or 0))
    # The COW pool is one process (one model load + COW share) so it is strictly
    # better on RAM than the --workers fan-out (N model loads). When set it WINS:
    # we launch a single rollout_worker with --pool-procs N --episodes <all games
    # for this actor> and skip the fan-out entirely.
    if pool_procs > 0 and n_workers > 1:
        _log(f"NOTE: --pool-procs={pool_procs} overrides --workers={n_workers} "
             f"(COW pool is one process, strictly less RAM than the fan-out).")
        n_workers = 1

    if do_phase_flip:
        # single-actor: claim the actor side + flip await_rollout -> rolling_out
        # (we are the sole owner of the phase).
        ppo_state.claim_side(url, ppo_state.ACTOR, who=actor_id, iter_=K)
        ppo_state.transition(url, expect_phase=ppo_state.PHASE_AWAIT_ROLLOUT,
                             new_phase=ppo_state.PHASE_ROLLING_OUT, who=actor_id)
    _log(f"actor={actor_id} idx={index}/{n_actors} rolling slice for iter {K} "
         f"(episodes={episodes} players={num_players} T={history_window} "
         f"num_envs={args.num_envs} workers={n_workers} pool_procs={pool_procs} "
         f"device={device}; phase_flip={do_phase_flip})")

    # download the head-delta for policy_vK
    model = state.get("model") or {}
    heads_src = model.get("heads")
    base_version = model.get("base_version")
    local_heads = workdir / f"policy_v{K}.heads.pt"
    _log(f"downloading model.heads {heads_src} -> {local_heads}")
    shards.gcs_cp(heads_src, str(local_heads))

    # DISTINCT, non-overlapping seed slice per actor per iter:
    #   seed_base_K = seed_base + K*(N*episodes) + index*episodes
    # (single-actor N=1,index=0 reduces to the original seed_base + K*episodes.)
    seed_base = seed_base0 + K * (n_actors * episodes) + index * episodes

    # --- WORKER FAN-OUT (--workers N>1): launch N rollout_worker PROCESSES in
    # PARALLEL (one game per core), each on its own seed sub-slice into w{j}/, and
    # write an actor-level _DONE after all exit 0. N==1 falls through to the
    # original single-worker path below (UNCHANGED). -------------------------- #
    if n_workers > 1:
        _log(f"fan-out: {n_workers} parallel workers under {out_prefix} "
             f"(actor seed_base={seed_base}, OMP/MKL/OPENBLAS pinned to 1 thread)")
        return _run_fanned_rollout(
            url, state, args, n_workers=n_workers, out_prefix=out_prefix,
            base_version=base_version, local_heads=local_heads,
            policy_heads=str(local_heads), K=K, episodes=episodes,
            num_players=num_players, history_window=history_window,
            max_planets=max_planets, max_fleets=max_fleets, device=device,
            sigma=sigma, actor_id=actor_id, seed_base=seed_base,
            progress_every=progress_every, num_envs=args.num_envs,
        )

    # Single rollout_worker process: the original serial/batched path when
    # pool_procs==0, or the COW fork pool (one model load, forked over pool_procs
    # cores) when pool_procs>0 — episodes = the actor's FULL game count either way.
    cmd = _build_worker_cmd(
        ckpt=args.ckpt, planet_run_dir=args.planet_run_dir,
        fleet_run_dir=args.fleet_run_dir, comet_run_dir=args.comet_run_dir,
        policy_heads=str(local_heads), base_version=base_version,
        policy_version=K, history_window=history_window, episodes=episodes,
        num_players=num_players, num_envs=args.num_envs, max_planets=max_planets,
        max_fleets=max_fleets, device=device, sigma=sigma, out_prefix=out_prefix,
        actor_id=actor_id, seed_base=seed_base, progress_every=progress_every,
        pool_procs=pool_procs,
    )
    _log("launching rollout_worker:\n  " + " ".join(cmd))

    progress_url = out_prefix + "progress.json"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)

    buf: list[str] = []
    last_flush = time.time()

    def _flush(final: bool = False) -> None:
        """Mirror the worker's latest progress.json + the new stdout lines into
        STATE so watchers tail the rollout live. Multi-actor uses ``update_actor``
        (per-actor sub-map, peers not clobbered); single-actor keeps the original
        ``set_progress`` on the shared ``actor`` block.

        The worker's progress.json is passed through VERBATIM, so for the
        ``--pool-procs`` path the ENRICHED dict — per-core ``workers:[...]`` rows
        (slot/seed/2P-4P/step/score/threads/games_done) plus the ``total_steps`` /
        ``games_done`` / ``steps_per_s`` / ``mean_step`` aggregates — lands intact
        under ``STATE.actor(s).<id>.progress``. It is a small list of small dicts, so
        the Firestore/STATE write carries it fine."""
        nonlocal buf, last_flush
        prog = shards.read_progress(progress_url)
        lines = buf
        buf = []
        if do_phase_flip:   # single-actor: shared actor block
            ppo_state.set_progress(url, ppo_state.ACTOR, progress=prog,
                                   log_lines=lines or None, who=actor_id)
        else:               # multi-actor: my own actors[id] sub-map only
            ppo_state.update_actor(url, actor_id, progress=prog,
                                   log_lines=lines or None,
                                   status=("done" if final else "rolling"),
                                   who=actor_id)
        last_flush = time.time()

    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\n")
        _log(line)
        buf.append(line)
        if (time.time() - last_flush) >= progress_every:
            _flush()
    # drain remaining buffered lines + the final progress snapshot
    _flush(final=True)

    return p.wait()


# --------------------------------------------------------------------------- #
# the listen loop
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True,
                    help="run PREFIX (gs://.../ppo/<run>), a full .../STATE.json, "
                         "or an fs://<collection>/<doc> Firestore control channel "
                         "(big-file gs:// paths come from STATE['gcs_prefix']).")
    ap.add_argument("--actor-id", default="gcp-actor")
    # forwarded to rollout_worker
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--planet-run-dir", default=None)
    ap.add_argument("--fleet-run-dir", default=None)
    ap.add_argument("--comet-run-dir", default=None)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1,
                    help="number of PARALLEL rollout_worker processes (1 game per "
                         "core). N==1 (default): a single worker writing its own "
                         "_DONE (unchanged). N>1: fan out N processes — each gets a "
                         "DISTINCT seed sub-slice + its own rollouts/vK/<id>/w{j}/ "
                         "out dir (OMP/MKL/OPENBLAS pinned to 1 thread each), and "
                         "the daemon writes the actor-level <id>/_DONE after all "
                         "exit 0. For '1 game per core': --workers N --episodes N "
                         "--num-envs 1. Ignored when --pool-procs>0 (the pool is "
                         "one process and strictly less RAM).")
    ap.add_argument("--pool-procs", type=int, default=0,
                    help="COW fork-pool rollout (PREFERRED): launch ONE "
                         "rollout_worker with --pool-procs N for ALL of this actor's "
                         "games — it loads the model ONCE and forks N workers that "
                         "share the weights copy-on-write (one model in RAM, not N) "
                         "and work-steal games across cores with a dynamic-OMP-for-"
                         "stragglers ramp. >0 replaces the --workers fan-out; 0 "
                         "(default) keeps the fan-out/single-worker path. CPU-only.")
    # config overrides (else taken from STATE['config'])
    ap.add_argument("--num-players", default=None,
                    help="2 / 4 / mix (default: STATE.config.num_players else 'mix').")
    ap.add_argument("--episodes", type=int, default=None,
                    help="default: STATE.config.episodes else 16.")
    ap.add_argument("--history-window", type=int, default=None,
                    help="default: STATE.config.history_window else 10.")
    ap.add_argument("--max-planets", type=int, default=64)
    ap.add_argument("--max-fleets", type=int, default=None,
                    help="default: STATE.config.max_fleets else 512.")
    ap.add_argument("--device", default=None,
                    help="default: STATE.config.device else 'cpu'.")
    ap.add_argument("--sigma", type=float, default=None,
                    help="default: STATE.config.sigma else 0.35.")
    ap.add_argument("--progress-every", type=float, default=5.0)
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--workdir", default=None,
                    help="scratch dir for the downloaded head-delta (default: a tempdir).")
    ap.add_argument("--once", action="store_true",
                    help="process a single iteration then exit (testing).")
    args = ap.parse_args()

    # resolve the STATE control-channel url (fs:// / .../STATE.json / run prefix).
    # Big-file (gs://) paths come from STATE['gcs_prefix'], resolved per-iter, NOT
    # from this url's parent dir (which doesn't exist for fs://).
    url = _resolve_state_url(args.state)

    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="ppo_actor_"))

    _log(f"START actor={args.actor_id} state={url} "
         f"poll={args.poll_s}s workdir={workdir}")

    while True:
        st = ppo_state.read_state(url)
        if st is None:
            _log(f"no STATE at {url}; retrying in {args.poll_s:.0f}s")
            time.sleep(args.poll_s)
            continue

        phase = st.get("phase")

        if phase == ppo_state.PHASE_DONE:
            _log("run done")
            return 0

        if phase == ppo_state.PHASE_ERROR:
            _log(f"run in ERROR: {st.get('message')!r}; exiting")
            return 1

        if phase == ppo_state.PHASE_AWAIT_ROLLOUT:
            K = int(st["iter"])
            run_gcs = _resolve_run_gcs(st, url)   # gs:// root for big files (from STATE)
            out_prefix = f"{run_gcs}/rollouts/v{K}/{args.actor_id}/"
            roster = ppo_state.expected_actors(st)   # None == single-actor

            # --- MY slice: roll it unless my own _DONE is already present. ----- #
            rolled = False   # did we actually run the worker (and thus, for the
                             # single-actor path, flip await_rollout->rolling_out)?
            if ppo_state.done_marker_exists(out_prefix):
                _log(f"_DONE already present for iter {K} (my slice "
                     f"{args.actor_id} done)")
            else:
                rolled = True
                try:
                    # single-actor owns the phase (flips await_rollout->rolling_out);
                    # multi-actor stays in await_rollout and only the gated barrier
                    # below flips to await_train.
                    rc = _run_rollout(url, st, args, workdir,
                                      do_phase_flip=(roster is None))
                except Exception as e:  # setup/streaming failure -> halt loud
                    import traceback
                    _log(f"rollout v{K} EXCEPTION: {e}\n{traceback.format_exc()}")
                    ppo_state.transition(
                        url, expect_phase=None, new_phase=ppo_state.PHASE_ERROR,
                        who=args.actor_id, message=f"rollout v{K} exception: {e}")
                    return 1
                if rc != 0:
                    _log(f"rollout v{K} exited {rc} -> ERROR")
                    ppo_state.transition(
                        url, expect_phase=None, new_phase=ppo_state.PHASE_ERROR,
                        who=args.actor_id, message=f"rollout v{K} exited {rc}")
                    return 1
                _log(f"rollout v{K} (slice {args.actor_id}) exited 0")
                _cleanup_after_rollout()   # free disk + /dev/shm before next iter

            # --- single-actor: flip -> await_train (shards = my dir). If we rolled
            # we're in rolling_out; on the idempotent skip we're still in
            # await_rollout — expect the phase we're actually in. ---------------- #
            if roster is None:
                expect = (ppo_state.PHASE_ROLLING_OUT if rolled
                          else ppo_state.PHASE_AWAIT_ROLLOUT)
                _log(f"single-actor -> flip {expect} -> await_train "
                     f"(shards={out_prefix})")
                ppo_state.transition(
                    url, expect_phase=expect,
                    new_phase=ppo_state.PHASE_AWAIT_TRAIN, who=args.actor_id,
                    shards=out_prefix, iter=K)
                if args.once:
                    return 0
                continue

            # --- multi-actor GATED FLIP: my slice is done; poll the barrier until
            # EVERY actor's _DONE exists, then ONE of us flips to await_train with
            # shards = the PARENT rollouts/vK/ (union of all actor subdirs). The
            # actor that finishes last (or any finisher post-completion) performs
            # the flip; an early finisher keeps polling here. -------------------- #
            parent = f"{run_gcs}/rollouts/v{K}/"
            while True:
                st2 = ppo_state.read_state(url)
                if st2 is None:
                    time.sleep(args.poll_s)
                    continue
                ph2 = st2.get("phase")
                # another actor (or the learner, on a crash-resume) already moved
                # us past await_rollout for this K -> nothing left to gate on.
                if ph2 != ppo_state.PHASE_AWAIT_ROLLOUT or int(st2.get("iter", -1)) != K:
                    _log(f"phase now {ph2} iter={st2.get('iter')} (someone flipped "
                         f"iter {K}); resuming main loop")
                    break
                if ppo_state.all_actors_done(run_gcs, K, roster):
                    _log(f"ALL {len(roster)} actors done for iter {K} -> "
                         f"flip await_rollout -> await_train (shards={parent})")
                    try:
                        ppo_state.transition(
                            url, expect_phase=ppo_state.PHASE_AWAIT_ROLLOUT,
                            new_phase=ppo_state.PHASE_AWAIT_TRAIN, who=args.actor_id,
                            shards=parent, iter=K)
                    except ppo_state.StatePhaseError:
                        # a peer flipped first between our read and write — fine.
                        _log("peer flipped to await_train first; ok")
                    break
                # peers not done yet. --once: one barrier check, then return
                # (stays in await_rollout). Daemon mode: sleep + re-poll so the
                # last finisher eventually flips.
                _log(f"my slice {args.actor_id} done; peers not done yet (iter {K})")
                if args.once:
                    return 0
                time.sleep(args.poll_s)

            if args.once:
                return 0
            continue

        # await_train / training / rolling_out (owned by a previous run / the peer)
        _log(f"phase={phase} iter={st.get('iter')} (not our turn); "
             f"sleeping {args.poll_s:.0f}s")
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
