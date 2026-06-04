"""PPO rollout actor — runs on a GCP CPU VM, pushes shards to GCS.

One invocation = one rollout iteration: load the frozen base (staged once on the
VM) + apply the <1 MB trainable head-delta for policy_vK, run N self-play games,
serialize each EpisodeBuffer to a shard, and upload shards + manifest + live
progress.json + rollout.log + metrics.json + a `_DONE` marker to GCS. Then exit.

Heavily instrumented (the VM is slow under T=10): a background telemetry thread
samples RAM (process RSS + system available + peak) and CPU every
``--progress-every`` seconds, prints a heartbeat, and republishes progress.json so
the rollout is watchable from Colab. See docs/PPO_TWO_CPU_PROTOCOL.md + the plan.

Example (driven by ppo_vm_daemon.sh):
  python -m agents.transformer_v2.ppo.rollout_worker \
    --ckpt ckpts/entity/entity_encoder_best.pt \
    --fleet-run-dir ckpts/fleet --planet-run-dir ckpts/planet --comet-run-dir ckpts/comet \
    --policy-heads gs://orbit-wars-shipping/ppo/<run>/heads/policy_v3.heads.pt \
    --base-version <entity_sha>:<critic_sha> --policy-version 3 --history-window 10 \
    --episodes 16 --rollout-workers 1 --device cpu \
    --out gs://orbit-wars-shipping/ppo/<run>/rollouts/v3/<vm_id>/ --vm-id <vm_id>
"""

from __future__ import annotations

import argparse
import os
import random
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

from . import shards
from .actor_critic import PPOActorCritic
from .smoke import load_supervised, make_opponent_closure, run_episode

try:
    import psutil  # optional; richer + cross-sampled RAM/CPU
    _PROC = psutil.Process()
except Exception:  # pragma: no cover - stdlib fallback
    psutil = None
    _PROC = None


# --------------------------------------------------------------------------- #
# System telemetry
# --------------------------------------------------------------------------- #
def _peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS.
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (ru / 1024.0) if sys.platform == "linux" else (ru / 1024.0 / 1024.0)


def _mem_sample() -> dict[str, float]:
    """Return {rss_mb, avail_mb, total_mb, peak_rss_mb, cpu_pct} best-effort."""
    out = {"peak_rss_mb": round(_peak_rss_mb(), 1)}
    if _PROC is not None:
        try:
            out["rss_mb"] = round(_PROC.memory_info().rss / 1024 / 1024, 1)
            vm = psutil.virtual_memory()
            out["avail_mb"] = round(vm.available / 1024 / 1024, 1)
            out["total_mb"] = round(vm.total / 1024 / 1024, 1)
            out["cpu_pct"] = round(_PROC.cpu_percent(interval=None), 1)
            return out
        except Exception:
            pass
    # stdlib fallback via /proc (Linux only)
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    out["rss_mb"] = round(int(ln.split()[1]) / 1024, 1)
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable:"):
                    out["avail_mb"] = round(int(ln.split()[1]) / 1024, 1)
                elif ln.startswith("MemTotal:"):
                    out["total_mb"] = round(int(ln.split()[1]) / 1024, 1)
    except Exception:
        pass
    return out


def _fmt_g(mb: float | None) -> str:
    return "?" if mb is None else f"{mb/1024:.1f}G"


class Telemetry:
    """Background heartbeat: sample RAM/CPU, print, republish progress.json."""

    def __init__(self, progress_url: str | None, every_s: float, ep_total: int,
                 policy_version: int, log):
        self.progress_url = progress_url
        self.every_s = max(1.0, float(every_s))
        self.ep_total = ep_total
        self.policy_version = policy_version
        self.log = log
        self.t0 = time.time()
        self.ep_done = 0
        self.cur_seed = -1
        self.cur_step = 0
        self.n_active = ep_total
        self.state = "starting"
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thr.start()

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=self.every_s + 2)
        self.publish(force=True)  # final snapshot

    def set(self, *, ep_done=None, cur_seed=None, state=None, cur_step=None, n_active=None):
        if ep_done is not None:
            self.ep_done = ep_done
        if cur_seed is not None:
            self.cur_seed = cur_seed
        if state is not None:
            self.state = state
        if cur_step is not None:
            self.cur_step = cur_step
        if n_active is not None:
            self.n_active = n_active

    def snapshot(self) -> dict:
        elapsed = time.time() - self.t0
        eta = (elapsed / self.ep_done * (self.ep_total - self.ep_done)
               if self.ep_done > 0 else None)
        mem = _mem_sample()
        return {
            "policy_version": self.policy_version,
            "ep_done": self.ep_done, "ep_total": self.ep_total,
            "cur_seed": self.cur_seed, "state": self.state,
            "cur_step": self.cur_step, "n_active": self.n_active,
            "elapsed_s": round(elapsed, 1),
            "eta_s": (round(eta, 1) if eta is not None else None),
            **mem, "ts": shards.utcnow_iso(),
        }

    def publish(self, force=False):
        snap = self.snapshot()
        if self.progress_url:
            shards.publish_progress(self.progress_url, snap)
        return snap

    def _loop(self):
        while not self._stop.is_set():
            snap = self.publish()
            self.log(f"[heartbeat] ep {snap['ep_done']}/{snap['ep_total']} "
                     f"step={snap['cur_step']} active={snap['n_active']} "
                     f"state={snap['state']} rss={_fmt_g(snap.get('rss_mb'))} "
                     f"peak={_fmt_g(snap.get('peak_rss_mb'))} cpu={snap.get('cpu_pct','?')}% "
                     f"(elapsed {snap['elapsed_s']:.0f}s)")
            self._stop.wait(self.every_s)


# --------------------------------------------------------------------------- #
# Per-core verbose progress publisher (the COW fork-pool path)
# --------------------------------------------------------------------------- #
class PoolProgressPublisher:
    """Background publisher for the ``--pool-procs`` path: every ``every_s`` it reads
    the shared ``worker_progress`` Manager dict (one row PER CORE, written by each
    fork worker's ``on_step`` hook) and writes an ENRICHED ``progress.json`` to the
    SAME url the daemon already mirrors into Firestore.

    The enriched dict keeps every top-level key the daemon/Colab already expect
    (``ep_done``/``ep_total``/``state``/``rss_mb``/``peak_rss_mb``/``elapsed_s``/...,
    via ``Telemetry.snapshot()``) and ADDS:
      * ``workers``: ``[{slot, game_idx, seed, num_players, step, score_my,
        score_enemy_max, threads, games_done}, ...]`` — one row per core, sorted by
        slot, so the monitor shows each core's live game.
      * aggregates: ``total_steps`` (Σ cores' current step), ``games_done`` (Σ cores'
        completed counts), ``steps_per_s`` (rolling, from the Σstep delta over wall),
        ``mean_step`` (mean of cores' current steps).

    It is the SOLE writer of progress.json during the pool rollout (the base
    Telemetry's own publish loop is left running for the heartbeat LOG line but its
    snapshot is folded in here, so the file the daemon reads always carries the
    per-core ``workers`` array)."""

    def __init__(self, progress_url, every_s, tele, worker_progress, log):
        self.progress_url = progress_url
        self.every_s = max(0.2, float(every_s))
        self.tele = tele
        self.worker_progress = worker_progress
        self.log = log
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._last_t = time.time()
        self._last_total_steps = 0
        self._steps_per_s = 0.0

    def start(self):
        self._thr.start()

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=self.every_s + 2)
        self.publish(force=True)   # final enriched snapshot

    def _rows(self) -> list[dict]:
        """A stable per-core list (sorted by slot) snapshotted from the Manager
        dict. Best-effort: a transient proxy hiccup yields an empty list rather than
        killing the publisher."""
        try:
            rows = [dict(v) for v in self.worker_progress.values() if v]
        except Exception:  # noqa: BLE001
            return []
        rows.sort(key=lambda r: r.get("slot", 0))
        return rows

    def snapshot(self) -> dict:
        base = self.tele.snapshot()       # all the existing top-level keys + mem
        rows = self._rows()
        total_steps = sum(int(r.get("step", 0)) for r in rows)
        games_done = sum(int(r.get("games_done", 0)) for r in rows)
        mean_step = (total_steps / len(rows)) if rows else 0
        now = time.time()
        dt = now - self._last_t
        if dt >= 0.5:
            # rolling steps/s from the Σ-current-step delta (monotone within a game;
            # resets dip when a game ends and its successor restarts at 0 — a rolling
            # rate is what the monitor wants, not a cumulative count).
            d = total_steps - self._last_total_steps
            self._steps_per_s = max(0.0, d) / dt if dt > 0 else 0.0
            self._last_t = now
            self._last_total_steps = total_steps
        base.update({
            "workers": rows,
            "n_workers": len(rows),
            "total_steps": total_steps,
            "games_done": games_done,
            "steps_per_s": round(self._steps_per_s, 2),
            "mean_step": round(mean_step, 1),
        })
        return base

    def publish(self, force=False) -> dict:
        snap = self.snapshot()
        if self.progress_url:
            shards.publish_progress(self.progress_url, snap)
        return snap

    def _loop(self):
        while not self._stop.is_set():
            snap = self.publish()
            rows = snap.get("workers") or []
            # one compact heartbeat line listing each core's live game
            cores = "  ".join(
                f"c{r.get('slot')}:{r.get('seed')}/{r.get('num_players')}P"
                f"st{r.get('step')} {r.get('score_my')}-{r.get('score_enemy_max')}"
                f" t{r.get('threads')}"
                for r in rows)
            self.log(f"[pool-progress] cores={len(rows)} "
                     f"total_steps={snap.get('total_steps')} "
                     f"games_done={snap.get('games_done')} "
                     f"steps/s={snap.get('steps_per_s')} "
                     f"peak={_fmt_g(snap.get('peak_rss_mb'))}"
                     + (f" | {cores}" if cores else ""))
            self._stop.wait(self.every_s)


# --------------------------------------------------------------------------- #
# Tee logger: stdout + a local rollout.log (periodically uploaded to GCS)
# --------------------------------------------------------------------------- #
class TeeLog:
    def __init__(self, local_path: Path):
        self.path = local_path
        self.fh = open(local_path, "w", buffering=1)

    def __call__(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


def _maybe_download(path: str, tmpdir: Path) -> str:
    """If ``path`` is a gs:// URL, download to ``tmpdir`` and return the local path."""
    if not path or not str(path).startswith("gs://"):
        return path
    local = tmpdir / Path(path).name
    shards.gcs_cp(path, local)
    return str(local)


def _episode_specs(n: int, seed_base: int, mode: str) -> list[tuple[int, int]]:
    """(seed, num_players) per episode. Draw BOTH the game seed and the player
    count from a per-call RNG seeded by ``seed_base`` so the rollout distribution
    stays UN-skewed. Why: with the ``--workers`` fan-out each worker rolls 1 game,
    so an index-based 'mix' (``i % 2``) would make every 1-episode worker pick the
    SAME player count -> an all-2P rollout. A seeded RNG makes each worker's draw
    independent (distinct seed_base -> distinct RNG stream) yet reproducible, so
    the aggregate over all workers/actors is a balanced 2P/4P mix with diverse,
    non-sequential seeds. mode '2'/'4' pins the player count; 'mix' randomizes it."""
    rng = random.Random(seed_base)
    specs: list[tuple[int, int]] = []
    for _ in range(n):
        num_players = rng.choice((2, 4)) if mode == "mix" else int(mode)
        specs.append((rng.randrange(2 ** 31), num_players))
    return specs


def _build_opponent(entity_model, cfg, policy, device, sigma):
    """Frozen self-play opponent = snapshot of the current learner (train.py:537-563)."""
    opp = PPOActorCritic(
        entity_model=type(entity_model)(
            d_model=cfg.get("d_model", 256),
            n_steps=cfg.get("n_steps", 6),
            d_pair=cfg.get("d_pair", cfg.get("d_model", 256)),
            entity_n_heads=cfg.get("entity_n_heads", 4),
            cross_n_heads=cfg.get("cross_n_heads", 8),
            cross_n_layers=cfg.get("cross_n_layers", 2),
            dual_n_heads=cfg.get("dual_n_heads", 4),
            conditioner_n_layers=cfg.get("conditioner_n_layers", 1),
            head_n_layers=cfg.get("head_n_layers", 1),
            # Match the learner's architecture exactly so the frozen opponent
            # does not get fresh RANDOM L3/L4 / consolidator modules.
            skip_l34=entity_model.skip_l34,
            with_consolidator=entity_model.consolidator is not None,
        ),
        sigma=sigma,
        allow_debug_glob_critic=True,
    ).to(device)
    opp.load_state_dict(policy.state_dict(), strict=False)
    opp.eval()
    for p in opp.parameters():
        p.requires_grad_(False)
    return opp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path, help="Supervised base (entity) ckpt.")
    ap.add_argument("--fleet-run-dir", type=Path, default=None)
    ap.add_argument("--planet-run-dir", type=Path, default=None)
    ap.add_argument("--comet-run-dir", type=Path, default=None)
    ap.add_argument("--policy-heads", type=str, default=None,
                    help="Trainable head-delta for policy_vK (local or gs://). None = base only.")
    ap.add_argument("--base-version", type=str, default=None, help="Expected base id the delta sits on.")
    ap.add_argument("--policy-version", type=int, required=True)
    ap.add_argument("--history-window", type=int, default=10, choices=(1, 10))
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--num-players", default="2", choices=("2", "4", "mix"),
                    help="players per game: 2, 4, or 'mix' (alternate 2P/4P, distinct seeds).")
    ap.add_argument("--rollout-workers", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=1,
                    help="batched rollout: run this many games in LOCKSTEP with ONE (B,...) GPU "
                         "forward per step (vectorized). 1 = serial run_episode (B=1).")
    ap.add_argument("--pool-procs", type=int, default=0,
                    help="COW fork-pool rollout: load the model ONCE in THIS process "
                         "then fork N workers that share the weights copy-on-write "
                         "(one model in RAM, not N) and work-steal games across cores "
                         "with a dynamic-OMP-for-stragglers ramp. >0 selects this path "
                         "over --num-envs/serial; 0 (default) keeps existing behavior. "
                         "CPU-only by construction.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--opponent", default="", help="'' = self-play snapshot; else a heuristic id.")
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--noop-logit-bias", type=float, default=0.0)
    ap.add_argument("--target-cap-k-max", type=int, default=4)
    ap.add_argument("--target-cap-lambda", type=float, default=0.001)
    ap.add_argument("--max-planets", type=int, default=64)
    ap.add_argument("--max-fleets", type=int, default=1024)
    ap.add_argument("--seed-base", type=int, default=100000)
    ap.add_argument("--out", required=True, help="GCS prefix gs://.../rollouts/vK/<vm_id>/")
    ap.add_argument("--vm-id", default="vm")
    ap.add_argument("--progress-every", type=float, default=5.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # FD-exhaustion fix (part 2): raise this process's soft open-files limit to the
    # hard max at the very start, BEFORE any torch/env/pool work. The COW fork pool
    # returns large EpisodeBuffer tensors whose shared-memory handles (plus the
    # kaggle env's per-game FDs) otherwise exhaust the 1024 default -> "[Errno 24]
    # Too many open files" ~16 games in. Best-effort; pool_rollout additionally
    # switches torch to the file_system sharing strategy (the primary fix) and
    # re-raises the limit inside each worker.
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        _fd_soft_new = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:  # noqa: BLE001
        _fd_soft_new = None

    if args.rollout_workers > 1:
        print(f"[rollout] NOTE: --rollout-workers={args.rollout_workers} but this VM worker runs "
              f"episodes SERIALLY (parallel rollout not implemented yet); using 1.", flush=True)

    out = args.out.rstrip("/") + "/"
    workdir = Path(tempfile.mkdtemp(prefix="ppo_rollout_"))
    log = TeeLog(workdir / "rollout.log")
    log_url = out + "rollout.log"
    progress_url = out + "progress.json"
    git = shards.git_sha()
    hw = args.history_window

    mem0 = _mem_sample()
    log(f"[rollout] START policy_v{args.policy_version} hw={hw} episodes={args.episodes} "
        f"players={args.num_players} num_envs={args.num_envs} pool_procs={args.pool_procs} "
        f"device={args.device} vm={args.vm_id} git={git}")
    log(f"[rollout] system: total={_fmt_g(mem0.get('total_mb'))} avail={_fmt_g(mem0.get('avail_mb'))} "
        f"cpus={os.cpu_count()} psutil={'yes' if psutil else 'no'}")
    log(f"[rollout] FD limit (RLIMIT_NOFILE) soft raised to hard max: soft={_fd_soft_new}")
    log(f"[rollout] ckpt={args.ckpt} heads={args.policy_heads}")

    # ---- load frozen base (staged on VM) + optional gs:// inputs ----
    heads_path = _maybe_download(args.policy_heads, workdir) if args.policy_heads else None

    t_load = time.time()
    entity_model, fleet_enc, planet_enc, comet_enc, cfg = load_supervised(
        args.ckpt, args.device,
        planet_run_dir=args.planet_run_dir, fleet_run_dir=args.fleet_run_dir,
        comet_run_dir=args.comet_run_dir,
    )
    # PPO critic = the value head inside PPOActorCritic (post-L2 player_state when
    # the actor has a PlayerConsolidator, else the glob value_head fallback). It
    # trains during PPO; there is no separate critic ckpt to load here.
    policy = PPOActorCritic(
        entity_model, sigma=args.sigma, allow_debug_glob_critic=True,
    ).to(args.device)

    # ---- apply the <1 MB trainable head-delta for policy_vK ----
    if heads_path:
        info = shards.load_trainable_delta(policy, heads_path, expect_base_version=args.base_version)
        log(f"[rollout] applied head-delta: {info['loaded_keys']} keys "
            f"(base={info['base_version']}); base loaded in {time.time()-t_load:.1f}s")
    else:
        log(f"[rollout] no head-delta (base only); loaded in {time.time()-t_load:.1f}s")
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    opp_policy = None
    if not args.opponent:
        opp_policy = _build_opponent(entity_model, cfg, policy, args.device, args.sigma)

    # ---- rollout (single worker for the trial; reuses run_episode) ----
    tele = Telemetry(progress_url, args.progress_every, args.episodes, args.policy_version, log)
    tele.set(state="rolling")
    tele.start()
    episode_specs = _episode_specs(args.episodes, args.seed_base, args.num_players)
    episodes = []
    ep_meta = []
    pool_pub = None        # set on the pool path -> owns progress.json (enriched)
    pool_mgr = None        # the Manager backing worker_progress (shut at the end)
    t_roll = time.time()
    try:
        if args.pool_procs > 0:
            # ---- COW FORK POOL: load model ONCE here, fork N workers that share
            # the weight pages copy-on-write (one model in RAM, not N), work-steal
            # games via imap_unordered, and dynamic-OMP the stragglers. The model is
            # already eval()+requires_grad_(False) above; run_pool_rollout re-asserts
            # that before forking. PER-CORE VERBOSE progress is streamed live: each
            # fork worker writes its slot row (seed/2P-4P/step/score/threads) into a
            # shared Manager dict via run_episode's on_step hook, and the
            # PoolProgressPublisher folds those rows + aggregates into progress.json
            # (the same file the daemon mirrors into Firestore). ---------------------
            import multiprocessing as _mp

            from . import pool_rollout
            specs = [(seed, n_players, ei % n_players)
                     for ei, (seed, n_players) in enumerate(episode_specs)]
            n2 = sum(1 for _, npl, _ in specs if npl == 2)
            n4 = sum(1 for _, npl, _ in specs if npl == 4)
            log(f"[rollout] POOL: {len(specs)} games ({n2}×2P, {n4}×4P) across "
                f"{args.pool_procs} fork workers (COW model share + dynamic-OMP "
                f"stragglers); one model load in this process.")
            log("[rollout] POOL: torch tensor-sharing strategy set to 'file_system' "
                "(FD-safe) + RLIMIT_NOFILE raised; per-core verbose progress ON.")
            tele.set(cur_seed=(specs[0][0] if specs else -1))
            # Shared per-core progress dict (fork ctx so children inherit it) + a
            # background publisher that writes the enriched progress.json with the
            # per-core ``workers`` array. The base Telemetry heartbeat THREAD is
            # stopped so the publisher is the sole writer of progress.json (it folds
            # tele.snapshot()'s top-level keys in), but ``tele`` stays alive for
            # state/snapshot.
            pool_mgr = _mp.get_context("fork").Manager()
            worker_progress = pool_mgr.dict()
            tele._stop.set()              # pause base publish loop (keep the object)
            pool_pub = PoolProgressPublisher(progress_url, args.progress_every, tele,
                                             worker_progress, log)
            pool_pub.start()
            try:
                buffers = pool_rollout.run_pool_rollout(
                    policy=policy, opponent_policy=opp_policy,
                    planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
                    specs=specs, n_procs=args.pool_procs, sigma=args.sigma,
                    max_planets=args.max_planets, max_fleets=args.max_fleets,
                    noop_logit_bias=args.noop_logit_bias,
                    history_window=args.history_window,
                    target_cap_k_max=args.target_cap_k_max,
                    target_cap_lambda=args.target_cap_lambda,
                    worker_progress=worker_progress,
                    progress_step_every=10,
                )
            finally:
                # Stop the publisher THREAD (writes a final enriched snapshot) but
                # KEEP pool_pub + the frozen worker rows alive: the end-of-main
                # state="done" write goes through pool_pub.publish() so the final
                # progress.json still carries the per-core ``workers`` array. The
                # Manager is shut at the very end of main().
                pool_pub.stop()
            pool_wall = time.time() - t_roll
            n_failed = sum(1 for b in buffers if b is None)
            if n_failed:
                log(f"[rollout] POOL: {n_failed}/{len(specs)} games FAILED "
                    f"(returned None) — dropping them.")
            # Feed the surviving EpisodeBuffers into the SAME downstream serialization
            # path: build episodes + ep_meta exactly like the serial/batched branches
            # (EpisodeBuffer carries no num_players, so pull seat/n_players from specs).
            for (seed, n_players, seat), ep in zip(specs, buffers):
                if ep is None:
                    continue
                episodes.append(ep)
                nsteps = len(ep.steps); won = int(ep.winner == seat)
                noop = sum(1 for s in ep.steps if s.action.n_launch == 0)
                meanv = (sum(s.value for s in ep.steps) / nsteps) if nsteps else 0.0
                ep_meta.append({"seed": seed, "seat": seat, "num_players": n_players,
                                "winner": ep.winner, "steps": nsteps, "won": won,
                                "wall_s": round(pool_wall / max(1, len(specs)), 2)})
                log(f"[rollout] {n_players}P seed={seed} seat={seat} steps={nsteps} "
                    f"noop={noop} winner={ep.winner} {'WIN' if won else 'lose'} "
                    f"meanV={meanv:.3f}")
            tele.set(ep_done=len(episodes), cur_step=0, n_active=0)
            log(f"[rollout] POOL: {len(episodes)}/{len(specs)} games OK "
                f"({pool_wall:.1f}s, {pool_wall / max(1, len(episodes)):.1f}s/game)")
            shards.gcs_cp(str(log.path), log_url)
        elif args.num_envs > 1:
            # ---- BATCHED: B games in lockstep, one (B,...) GPU forward per step ----
            from .batched_rollout import run_batched_episodes
            all_specs = [(seed, n_players, ei % n_players)
                         for ei, (seed, n_players) in enumerate(episode_specs)]
            log(f"[rollout] BATCHED: {len(all_specs)} games in chunks of {args.num_envs} "
                f"(vectorized — 1 forward/step over B)")
            for c0 in range(0, len(all_specs), args.num_envs):
                chunk = all_specs[c0:c0 + args.num_envs]
                tele.set(ep_done=len(episodes), cur_seed=chunk[0][0])
                t_c = time.time()
                _hb = [t_c]

                def _on_step(t, active, env_steps, _t0=t_c, _hb=_hb):
                    n_act = sum(active)
                    tele.set(cur_step=t + 1, n_active=n_act)
                    now = time.time()
                    if now - _hb[0] >= args.progress_every:
                        _hb[0] = now
                        # lockstep: every running env is at step t+1; only finished
                        # envs differ (frozen at their game length) -> list those.
                        done = "  ".join(f"env{ei}={env_steps[ei]}st"
                                         for ei in range(len(active)) if not active[ei])
                        m = _mem_sample()
                        msg = (f"[rollout] running {n_act}/{len(active)} envs @ step {t + 1} | "
                               f"rss={_fmt_g(m.get('rss_mb'))} cpu={m.get('cpu_pct', '?')}% "
                               f"({now - _t0:.0f}s)")
                        if done:
                            msg += f"\n           finished: {done}"
                        log(msg)
                eps = run_batched_episodes(
                    policy=policy, opponent_policy=opp_policy,
                    planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
                    specs=chunk, device=args.device, max_planets=args.max_planets,
                    max_fleets=args.max_fleets, sigma=args.sigma,
                    noop_logit_bias=args.noop_logit_bias,
                    target_cap_k_max=args.target_cap_k_max,
                    target_cap_lambda=args.target_cap_lambda,
                    history_window=args.history_window,
                    on_step=_on_step,
                )
                c_wall = time.time() - t_c
                for ep, (seed, n_players, seat) in zip(eps, chunk):
                    episodes.append(ep)
                    nsteps = len(ep.steps); won = int(ep.winner == seat)
                    noop = sum(1 for s in ep.steps if s.action.n_launch == 0)
                    meanv = (sum(s.value for s in ep.steps) / nsteps) if nsteps else 0.0
                    ep_meta.append({"seed": seed, "seat": seat, "num_players": n_players,
                                    "winner": ep.winner, "steps": nsteps, "won": won,
                                    "wall_s": round(c_wall / max(1, len(chunk)), 2)})
                    log(f"[rollout] {n_players}P seed={seed} seat={seat} steps={nsteps} noop={noop} "
                        f"winner={ep.winner} {'WIN' if won else 'lose'} meanV={meanv:.3f}")
                tele.set(ep_done=len(episodes))
                log(f"[rollout] chunk {c0 // args.num_envs + 1}: {len(episodes)}/{len(all_specs)} "
                    f"games ({c_wall:.1f}s, {c_wall / max(1, len(chunk)):.1f}s/game)")
                shards.gcs_cp(str(log.path), log_url)
        else:
            for ei, (seed, n_players) in enumerate(episode_specs):
                seat = ei % n_players              # rotate the learner through the seats
                tele.set(ep_done=ei, cur_seed=seed)
                t_ep = time.time()
                ep = run_episode(
                    policy=policy, planet_enc=planet_enc, fleet_enc=fleet_enc,
                    comet_enc=comet_enc, seed=seed, learner_seat=seat,
                    num_players=n_players,
                    opponent_policy=(opp_policy if not args.opponent else None),
                    opponent_id=(args.opponent or None),
                    device=args.device, max_planets=args.max_planets,
                    max_fleets=args.max_fleets, sigma=args.sigma,
                    noop_logit_bias=args.noop_logit_bias,
                    target_cap_k_max=args.target_cap_k_max,
                    target_cap_lambda=args.target_cap_lambda,
                    history_window=args.history_window,
                )
                episodes.append(ep)
                won = int(ep.winner == seat)
                nsteps = len(ep.steps)
                noop = sum(1 for s in ep.steps if s.action.n_launch == 0)
                meanv = (sum(s.value for s in ep.steps) / nsteps) if nsteps else 0.0
                wall = time.time() - t_ep
                ep_meta.append({"seed": seed, "seat": seat, "num_players": n_players,
                                "winner": ep.winner, "steps": nsteps, "won": won,
                                "wall_s": round(wall, 2)})
                done = ei + 1
                eta = (time.time() - t_roll) / done * (args.episodes - done)
                log(f"[rollout] ep {done}/{args.episodes} {n_players}P seed={seed} seat={seat} "
                    f"steps={nsteps} noop={noop} winner={ep.winner} {'WIN' if won else 'lose'} "
                    f"meanV={meanv:.3f} ({wall:.1f}s, eta {eta/60:.1f}m)")
                tele.set(ep_done=done)
                if (ei % 4) == 0:
                    shards.gcs_cp(str(log.path), log_url)  # periodic log upload
    except Exception as e:
        tele.set(state="error")
        tele.stop()
        log(f"[rollout] FAILED at episode: {e}")
        import traceback
        log(traceback.format_exc())
        shards.gcs_cp(str(log.path), log_url)
        return 1
    roll_s = time.time() - t_roll

    # ---- GAME STATUS summary (win rate, 2P/4P split, game-length spread) ----
    n_g = len(ep_meta) or 1
    n_w = sum(m["won"] for m in ep_meta)
    g2 = [m for m in ep_meta if m["num_players"] == 2]
    g4 = [m for m in ep_meta if m["num_players"] == 4]
    w2, w4 = sum(m["won"] for m in g2), sum(m["won"] for m in g4)
    n_draw = sum(1 for m in ep_meta if m["winner"] == -1)
    steps_all = [m["steps"] for m in ep_meta] or [0]
    split = []
    if g2:
        split.append(f"2P {w2}/{len(g2)}={w2/len(g2)*100:.0f}%")
    if g4:
        split.append(f"4P {w4}/{len(g4)}={w4/len(g4)*100:.0f}%")
    log("[rollout] ===== GAME STATUS =====")
    log(f"[rollout]   win rate (learner seat): {n_w}/{len(ep_meta)} = {n_w/n_g*100:.0f}%"
        + (f"   [{', '.join(split)}]" if split else "")
        + (f"   draws={n_draw}" if n_draw else ""))
    log(f"[rollout]   game length: mean {sum(steps_all)/len(steps_all):.0f}st  "
        f"min {min(steps_all)}st  max {max(steps_all)}st")

    tele.set(state="uploading")
    log(f"[rollout] all {args.episodes} episodes done in {roll_s/60:.1f}m; serializing shards ...")

    # ---- serialize + upload shards, manifest, metrics, log, _DONE ----
    entries = []
    pairs = []
    for i, ep in enumerate(episodes):
        sp = workdir / f"shard_{i:06d}.pt"
        entry = shards.save_shard(ep, sp, policy_version=args.policy_version,
                                  history_window=hw, git_sha_str=git, vm_id=args.vm_id)
        entries.append(entry)
        pairs.append((str(sp), out + entry["name"]))
    shards.gcs_cp_parallel(pairs, workers=8)
    for entry in entries:
        log(f"[upload] {entry['name']} {entry['bytes']/1024/1024:.2f} MB winner={entry['winner']}")

    man = workdir / "manifest.json"
    shards.write_manifest(man, entries, policy_version=args.policy_version,
                          history_window=hw, git_sha_str=git, vm_id=args.vm_id)
    shards.gcs_cp(str(man), out + "manifest.json")

    n_steps = sum(e["n_steps"] for e in entries)
    final_mem = _mem_sample()
    metrics = {
        "policy_version": args.policy_version, "vm_id": args.vm_id, "git_sha": git,
        "history_window": hw, "episodes": args.episodes, "n_steps": n_steps,
        "n_wins": sum(m["won"] for m in ep_meta),
        "total_wall_s": round(roll_s, 1),
        "mean_step_s": round(roll_s / max(1, n_steps), 4),
        "peak_rss_mb": final_mem.get("peak_rss_mb"),
        "cpu_time_s": round(resource.getrusage(resource.RUSAGE_SELF).ru_utime
                            + resource.getrusage(resource.RUSAGE_SELF).ru_stime, 1),
        "per_episode": ep_meta,
    }
    mpath = workdir / "metrics.json"
    mpath.write_text(__import__("json").dumps(metrics, indent=2))
    shards.gcs_cp(str(mpath), out + "metrics.json")

    tele.set(state="done")
    if pool_pub is not None:
        # Pool path: the publisher owns progress.json. Write the FINAL enriched
        # snapshot (state=done, with the frozen per-core ``workers`` rows + totals)
        # rather than tele.stop()'s plain snapshot, which would drop ``workers``.
        # tele._stop is already set; just emit the final enriched file, then shut
        # the Manager that backed the per-core dict.
        pool_pub.publish(force=True)
        try:
            if pool_mgr is not None:
                pool_mgr.shutdown()
        except Exception:
            pass
    else:
        tele.stop()
    log(f"[rollout] DONE v{args.policy_version}: {args.episodes} eps, {n_steps} steps, "
        f"{metrics['n_wins']} wins, peak_rss={_fmt_g(metrics['peak_rss_mb'])}, "
        f"wall={roll_s/60:.1f}m")
    shards.gcs_cp(str(log.path), log_url)
    log.close()
    # _DONE LAST — its appearance gates the learner's read (atomic completion).
    done_marker = workdir / "_DONE"
    done_marker.write_text("")
    shards.gcs_cp(str(done_marker), out + "_DONE")
    # CLEANUP: everything is durably on GCS now — drop the local workdir so the
    # T=10 shards (~1.5 GB each) don't pile up in /tmp across iterations and fill
    # the disk. Best-effort; the rollout already succeeded.
    try:
        import shutil as _sh
        _sh.rmtree(workdir, ignore_errors=True)
        print(f"[cleanup] removed local rollout workdir {workdir}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
