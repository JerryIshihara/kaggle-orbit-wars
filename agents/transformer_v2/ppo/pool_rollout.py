"""Multiprocessing-POOL rollout for the PPO actor (fork + copy-on-write).

Motivation
----------
The shipped ``--rollout-workers`` fan-out spawns N *separate* processes, each of
which independently runs ``load_supervised`` + builds a ``PPOActorCritic`` — that
is N model loads and N full RAM copies of the (entity + L0 specialist) weights.
On a slow CPU VM under T=10 the load alone is non-trivial, and N×RAM is the
binding constraint when you want one worker per core.

This module instead loads the model **once in the parent**, puts it into
inference mode (``eval()`` + ``requires_grad_(False)`` on every parameter so no
backward graph and no in-place grad writes ever touch the weight tensors), then
forks ``N`` workers with ``multiprocessing.get_context("fork")``. Because fork
gives each child a copy-on-write view of the parent's address space and we never
*write* to the model tensors after the fork, the weight pages stay physically
shared across all workers: total resident RAM is ≈ one model, not N.

Work is distributed with ``pool.imap_unordered`` over the list of game specs.
``imap_unordered`` hands the next spec to whichever worker is free, so a short
2P game does not block a core behind a long 4P game — this is the work-stealing
load balance. Each worker pins itself to a single core
(``torch.set_num_threads(1)``) and plays one full game via ``smoke.run_episode``
(agent forward + kaggle env step), returning its ``EpisodeBuffer``.

Dynamic-OMP for stragglers
--------------------------
``imap_unordered`` work-stealing balances *which* worker plays *which* game, but
once the queue drains the LAST few games run one-thread-each while N-k cores sit
idle — and the wall is the single longest game. To attack that tail we let the
surviving games grab the freed cores.

A shared counter ``remaining`` (``ctx.Value('i', len(specs))``) tracks games still
LEFT TO PLAY; each worker decrements it under its lock when it CLAIMS a game in
``_play_one``, and sizes that game's torch team to
``threads = max(1, n_procs // max(1, min(remaining, n_procs)))``: while
``remaining >= n_procs`` that is 1 (full oversubscription avoidance); once only
``k < n_procs`` games are left to play it climbs to ≈ ``n_procs / k``, so the ``k``
survivors fan across roughly all ``n_procs`` cores.

Two facts about OpenMP (ATen's parallel backend here) shape this: (1) a thread's
``set_num_threads`` only sizes the team that *that same thread* enters, so an
outside thread cannot grow a game ALREADY running — the worker MAIN thread sizes
its own game; and (2) under a fixed queue, work-stealing keeps every worker busy
until the queue empties, so a *completion* counter never drops below ``n_procs``
at a game START (idle workers would have drained the queue first) — hence we count
games-left-to-PLAY, claimed at start, which is what actually lets the final games
ramp. A per-worker daemon WATCHER thread (``daemon=True`` + stop ``Event``, dies
with the worker) reads the same counter every ~3 s and re-pins ITSELF, which on
process-global native-C10-pool builds additionally ramps the in-flight game (a
bonus OpenMP can't provide). No change to the game loop.

Caveats
-------
* fork + torch is only safe because the parent never started CUDA and we keep
  ``torch.set_num_threads(1)`` everywhere (an MKL/OpenMP thread pool that is live
  across a fork can deadlock the child). This is CPU-only by construction:
  workers always run on ``device="cpu"``.
* ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` in the parent does **not**
  include forked children's RSS; ``RUSAGE_CHILDREN`` aggregates the peak of
  *reaped* children, which the pool reaps on exit. The bench reports both plus a
  note that COW keeps the true total ≈ one model.

CLI bench::

    .venv/bin/python -m agents.transformer_v2.ppo.pool_rollout \\
        --ckpt /tmp/ppoB_ckpts/entity/entity_encoder_best.pt \\
        --planet-run-dir /tmp/ppoB_ckpts/planet \\
        --fleet-run-dir  /tmp/ppoB_ckpts/fleet \\
        --comet-run-dir  /tmp/ppoB_ckpts/comet \\
        --games 4 --procs 2 --history-window 1 --num-players mix
"""

from __future__ import annotations

import argparse
import multiprocessing
import resource
import sys
import threading
import time
from pathlib import Path

import torch

from . import smoke
from .actor_critic import PPOActorCritic
from .rollout_worker import _build_opponent, _episode_specs


# --------------------------------------------------------------------------- #
# Module globals — set in the PARENT before forking so children inherit them   #
# via copy-on-write. A forked child gets these already populated; it never     #
# re-loads the model. (Do NOT rely on these under a "spawn" start method —     #
# spawn re-imports the module fresh and they would be None.)                   #
# --------------------------------------------------------------------------- #
_POLICY: PPOActorCritic | None = None
_OPP: PPOActorCritic | None = None
_PLANET_ENC = None
_FLEET_ENC = None
_COMET_ENC = None
_KW: dict | None = None  # scalar kwargs forwarded to smoke.run_episode

# Dynamic-OMP state. ``_REMAINING`` is a multiprocessing.Value('i') shared across
# all fork workers (one physical counter = games still LEFT TO PLAY); ``_N_PROCS``
# is the pool width. They are PUBLISHED in the parent before the fork and re-bound
# inside each worker by the Pool initializer. ``_OMP_THREAD`` / ``_OMP_STOP`` are
# the worker-local watcher thread + its stop event (each worker has its own).
# ``_OMP_WANT`` is a 1-slot holder carrying the latest target thread count (kept in
# sync by both the claimer in _play_one and the watcher) — see the backend note.
_REMAINING = None      # multiprocessing.Value('i', len(specs)) | None
_N_PROCS: int = 1
_OMP_THREAD: threading.Thread | None = None
_OMP_STOP: threading.Event | None = None
_DYNAMIC_OMP: bool = False
_OMP_WANT: list = [1]  # single-element list; replacing [0] is atomic in CPython

# Per-core live progress (the verbose per-step chain). ``_WORKER_PROGRESS`` is a
# ``mp.Manager().dict()`` shared across all fork workers; each worker owns ONE slot
# id (0..n_procs-1, handed out round-robin by the Pool initializer from a shared
# counter) and writes its current game's live stats into ``_WORKER_PROGRESS[slot]``
# every ~throttle steps via the ``on_step`` hook. ``_SLOT_CTR`` is the shared
# counter the initializer draws the next slot from; ``_SLOT`` is THIS worker's slot.
# ``_GAMES_DONE`` counts games THIS worker has finished (for its per-core dict).
_WORKER_PROGRESS = None     # mp.Manager().dict() | None
_SLOT_CTR = None            # mp.Value('i', 0) — next slot to hand out | None
_SLOT: int = -1             # this worker's slot id (set in _pool_init)
_GAMES_DONE: int = 0        # this worker's completed-game count
_PROGRESS_STEP_EVERY: int = 10  # throttle: publish every Nth learner step


# --------------------------------------------------------------------------- #
# FD-exhaustion safety: raise the soft open-files limit to the hard max.        #
# --------------------------------------------------------------------------- #
def _raise_fd_limit() -> None:
    """Best-effort: bump RLIMIT_NOFILE soft -> hard. The 8-worker fork pool can
    exhaust the default 1024 FDs (large EpisodeBuffer tensors returned via
    imap_unordered hold shared-memory handles; the kaggle env may also leak per
    game). Called both here (pool worker initializer) and in rollout_worker.main().
    Never raises — if the platform refuses, we just keep the existing limit."""
    try:
        import resource as _r
        soft, hard = _r.getrlimit(_r.RLIMIT_NOFILE)
        _r.setrlimit(_r.RLIMIT_NOFILE, (hard, hard))
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Dynamic-OMP-for-stragglers: shared counter + per-worker DECIDER thread         #
# --------------------------------------------------------------------------- #
# Backend note (important): ATen's parallel backend here is OpenMP. With OpenMP,
# ``torch.set_num_threads`` writes the CALLING thread's ICV, and a parallel region
# is sized from the ICV of the thread that ENTERS it. Two consequences shape this
# design:
#   1. A *background* thread cannot resize the *main* thread's intra-op team — only
#      the thread that runs the game's torch ops can. So the worker MAIN thread must
#      be the one to call set_num_threads for its own game.
#   2. Under a fixed work queue, imap_unordered keeps every worker busy until the
#      queue empties, so a NEW game never starts while a COMPLETION counter is below
#      n_procs (idle workers would have drained the queue first). The actionable
#      counter is therefore "games still LEFT TO PLAY" (in-flight + queued),
#      decremented when a worker CLAIMS a game in ``_play_one`` — when only k <
#      n_procs games are left to play, n_procs-k workers are going idle and the
#      final k games each take ~n_procs/k cores (the tail grabs the freed cores).
# The per-worker daemon WATCHER thread reads the same counter and re-pins ITSELF; on
# builds whose backend is the process-global native C10 pool that also ramps the
# in-flight game (a bonus OpenMP can't give), and it keeps ``_OMP_WANT`` current.
def _omp_threads_for(remaining: int, n_procs: int) -> int:
    """Cores a game should claim given ``remaining`` games left to play and
    ``n_procs`` workers. ``remaining >= n_procs`` -> 1 (avoid oversubscription); as
    the tail shrinks to ``k`` games -> ≈ ``n_procs / k`` so the ``k`` survivors fan
    across all cores. Floor of 1; never exceeds ``n_procs``."""
    n_procs = max(1, int(n_procs))
    r = max(1, min(int(remaining), n_procs))
    return max(1, n_procs // r)


def _omp_watch_loop(remaining, n_procs: int, stop: threading.Event,
                    period_s: float = 3.0) -> None:
    """Daemon WATCHER body: every ``period_s`` read the shared ``remaining`` (games
    left to play) counter, compute ``_omp_threads_for(...)``, PUBLISH it into
    ``_OMP_WANT``, and call set_num_threads on THIS thread (so native-backend /
    process-global-pool builds ramp the in-flight game too — a no-op for OpenMP).
    Exits when ``stop`` is set (daemon -> dies with the process regardless).
    Best-effort: a transient error never kills the worker."""
    last = -1
    while not stop.is_set():
        try:
            r = int(remaining.value) if remaining is not None else n_procs
            want = _omp_threads_for(r, n_procs)
            if want != last:
                # Publish the target ONLY. The actual thread sizing happens at
                # game START in _play_one (safe: the main thread sizes its own
                # game before entering the forward). Calling set_num_threads from
                # THIS watcher thread mid-game runs concurrently with the main
                # thread's OpenMP forward and DEADLOCKS torch's thread pool (it
                # hung the rollout's tail games). So: telemetry-only watcher.
                _OMP_WANT[0] = want
                last = want
        except Exception:  # noqa: BLE001 — telemetry-grade; never kill the worker
            pass
        stop.wait(period_s)


def _pool_init(policy, opp, planet_enc, fleet_enc, comet_enc, kw,
               remaining, n_procs, dynamic_omp,
               worker_progress=None, slot_ctr=None, progress_step_every=10):
    """Pool initializer — runs ONCE per worker right after the fork.

    Re-binds the inherited model/encoders + the shared ``remaining`` (games-left-to-
    play) counter into THIS worker's module globals, pins it to a single core to
    start (1 game/core), and — when ``dynamic_omp`` — launches the daemon watcher
    thread that ramps the thread count up as the queue drains. Using an initializer
    (vs. relying purely on COW globals) guarantees one thread is spawned per worker
    process, not per task, and gives each worker its own stop Event.

    It also (a) raises this worker's soft FD limit to the hard max — the fork pool
    can otherwise exhaust the default 1024 across many returned tensors — and (b)
    claims THIS worker's per-core progress SLOT (0..n_procs-1) off the shared
    ``slot_ctr`` so its live game stats land in a stable ``worker_progress[slot]``."""
    global _POLICY, _OPP, _PLANET_ENC, _FLEET_ENC, _COMET_ENC, _KW
    global _REMAINING, _N_PROCS, _OMP_THREAD, _OMP_STOP, _DYNAMIC_OMP, _OMP_WANT
    global _WORKER_PROGRESS, _SLOT_CTR, _SLOT, _GAMES_DONE, _PROGRESS_STEP_EVERY
    # FD-exhaustion fix (part 2): raise soft RLIMIT_NOFILE -> hard in EACH worker.
    _raise_fd_limit()
    _POLICY = policy
    _OPP = opp
    _PLANET_ENC = planet_enc
    _FLEET_ENC = fleet_enc
    _COMET_ENC = comet_enc
    _KW = kw
    _REMAINING = remaining
    _N_PROCS = max(1, int(n_procs))
    _DYNAMIC_OMP = bool(dynamic_omp)
    _OMP_WANT = [1]
    # Per-core live progress: claim a stable slot id off the shared counter so this
    # worker always writes the SAME worker_progress[slot] (one row per core).
    _WORKER_PROGRESS = worker_progress
    _SLOT_CTR = slot_ctr
    _GAMES_DONE = 0
    _PROGRESS_STEP_EVERY = max(1, int(progress_step_every))
    _SLOT = -1
    if slot_ctr is not None:
        try:
            with slot_ctr.get_lock():
                _SLOT = int(slot_ctr.value)
                slot_ctr.value += 1
        except Exception:  # noqa: BLE001
            _SLOT = -1
    # Start single-threaded regardless: while the queue is full every worker should
    # own exactly one core. The watcher only ever raises the target later.
    torch.set_num_threads(1)
    if _DYNAMIC_OMP and _REMAINING is not None:
        _OMP_STOP = threading.Event()
        _OMP_THREAD = threading.Thread(
            target=_omp_watch_loop, args=(_REMAINING, _N_PROCS, _OMP_STOP),
            name="omp-straggler", daemon=True,
        )
        _OMP_THREAD.start()


def _to_inference(module: torch.nn.Module) -> torch.nn.Module:
    """Put a model into a fork-safe, COW-friendly inference state.

    ``eval()`` disables dropout / BN updates; ``requires_grad_(False)`` on every
    parameter means no autograd graph is built and — crucially — no optimizer or
    backward pass will ever write into these tensors, so the physical pages the
    children share via copy-on-write are never dirtied.
    """
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _play_one(item):
    """Pool worker body: play ONE full game on a single core.

    ``item`` is ``(idx, (seed, num_players, learner_seat))``. Reads the model +
    encoders from the module globals the parent populated before the fork.
    Returns ``(idx, EpisodeBuffer | None)`` — never raises, so one bad game does
    not poison the pool (imap_unordered would otherwise propagate the exception
    and tear the whole pool down).
    """
    idx, (seed, num_players, seat) = item
    # --- dynamic-OMP: size THIS game's torch team from how many games are still
    # left to PLAY (this one included), then claim this game off that counter. ----
    # Why "left to play" and not "still running": ATen's parallel backend is OpenMP,
    # where only the thread ENTERING a parallel region can size its team — an outside
    # thread cannot grow a game ALREADY running. And under a fixed work queue,
    # work-stealing keeps every worker busy until the queue empties, so a NEW game
    # never starts while a completion-counter sits below n_procs (idle workers would
    # have drained the queue first). The actionable, OpenMP-correct lever is thus to
    # size each game AS IT STARTS by the remaining-to-play count: while that is
    # >= n_procs every game starts on 1 core (no oversubscription); once only k <
    # n_procs games remain to play, the queue is draining and n_procs-k workers are
    # going idle, so those final k games each take ~n_procs/k cores — the tail grabs
    # the cores the finished workers freed. The watcher thread (the DECIDER) reads
    # the SAME counter and re-pins itself too, which additionally ramps the in-flight
    # game on builds whose backend is the process-global native C10 pool.
    # Fork-safe OMP policy: 1 thread per worker, ALWAYS. The COW pool forks its
    # workers, and GROWING the OpenMP team inside a forked process —
    # torch.set_num_threads(N>1) — deadlocks at the first parallel region (the
    # OpenMP runtime's thread pool is left in an undefined state by fork()). The
    # old dynamic-OMP straggler ramp did exactly that: every game that claimed a
    # low ``remaining`` count was sized to 2/4/8 threads and HUNG at step 0,
    # freezing the whole rollout (seen at 13/16, then at 4/8 with 8 games). One
    # game / one core / one thread is the only fork-safe policy here. ``remaining``
    # is kept draining for telemetry only; it no longer sizes any team.
    torch.set_num_threads(1)
    if _REMAINING is not None:
        with _REMAINING.get_lock():
            if _REMAINING.value > 0:
                _REMAINING.value -= 1

    # --- per-core live progress: publish THIS worker's slot row at game start, then
    # refresh it every ~_PROGRESS_STEP_EVERY learner steps via the on_step hook. The
    # row carries which core, which game (idx + seed + 2P/4P), the current step,
    # the live score, the worker's dynamic OMP thread count, and how many games this
    # worker has finished — so the monitor can show each core's live game. -------- #
    def _publish_row(
        step: int,
        score_my: int,
        score_enemy_max: int,
        *,
        status: str = "running",
    ) -> None:
        if _WORKER_PROGRESS is None or _SLOT < 0:
            return
        try:
            _WORKER_PROGRESS[_SLOT] = {
                "slot": _SLOT,
                "game_idx": int(idx),
                "seed": int(seed),
                "num_players": int(num_players),
                "step": int(step),
                "score_my": int(score_my),
                "score_enemy_max": int(score_enemy_max),
                "threads": int(torch.get_num_threads()),
                "games_done": int(_GAMES_DONE),
                "status": str(status),
            }
        except Exception:  # noqa: BLE001 — telemetry-grade; never kill the game
            pass

    def _on_step(step, npl, score_my, score_enemy_max):
        # Throttle: most steps are skipped (cheap), so the shared Manager dict write
        # stays infrequent. Always publish step 0 so the row appears as soon as a
        # game starts on this core.
        if step % _PROGRESS_STEP_EVERY == 0:
            _publish_row(step, score_my, score_enemy_max)

    _publish_row(0, 0, 0, status="running")   # row visible the instant this core claims the game
    try:
        # Self-play against the frozen snapshot when present; otherwise fall back
        # to a heuristic opponent. run_episode's guard wants EXACTLY one of
        # opponent_id / opponent_policy / opponent_fn to be non-None, and
        # opponent_id has no default (it is a required kw-only arg), so always
        # pass both keys with exactly one populated.
        if _OPP is not None:
            opp_kwargs = {"opponent_policy": _OPP, "opponent_id": None}
        else:
            opp_kwargs = {"opponent_policy": None, "opponent_id": "random_v1"}
        buf = smoke.run_episode(
            policy=_POLICY,
            planet_enc=_PLANET_ENC,
            fleet_enc=_FLEET_ENC,
            comet_enc=_COMET_ENC,
            seed=seed,
            learner_seat=seat,
            num_players=num_players,
            device="cpu",
            on_step=_on_step,
            **opp_kwargs,
            **_KW,
        )
        # This worker finished a game; bump its count and re-stamp the slot row so
        # ``games_done`` advances live and the final step is reflected.
        global _GAMES_DONE
        _GAMES_DONE += 1
        nsteps = len(buf.steps) if buf is not None else 0
        last = buf.steps[-1] if (buf is not None and buf.steps) else None
        _publish_row(nsteps,
                     int(getattr(last, "score_my", 0)),
                     int(getattr(last, "score_enemy_max", 0)),
                     status="done")
        return idx, buf
    except Exception as exc:  # noqa: BLE001 — isolate one bad game
        import traceback

        sys.stderr.write(
            f"[pool_rollout] game idx={idx} seed={seed} "
            f"({num_players}P seat={seat}) FAILED: {exc}\n"
        )
        traceback.print_exc()
        return idx, None


def run_pool_rollout(
    policy: PPOActorCritic,
    opponent_policy: PPOActorCritic | None,
    planet_enc,
    fleet_enc,
    comet_enc,
    specs: list[tuple[int, int, int]],
    n_procs: int,
    *,
    sigma: float,
    max_planets: int,
    max_fleets: int,
    noop_logit_bias: float = 0.0,
    select_logit_bias: float = 0.0,
    history_window: int = 10,
    target_cap_k_max: int = 4,
    target_cap_lambda: float = 0.0,
    dynamic_omp: bool = False,  # UNSAFE with the fork pool (see _play_one): growing
                                # the OMP team in a forked worker deadlocks step 0.
    worker_progress=None,
    progress_step_every: int = 10,
    on_episode_done=None,
) -> list:
    """Roll out ``len(specs)`` games across a fork pool of ``n_procs`` workers.

    Parameters
    ----------
    policy, opponent_policy:
        The learner ``PPOActorCritic`` and (optionally) its frozen self-play
        snapshot. Both are forced into inference mode here, BEFORE the fork, so
        their weight pages are shared copy-on-write across all workers.
    specs:
        ``list[(seed, num_players, learner_seat)]``. ``imap_unordered`` work-
        steals over this list so player-count imbalance load-balances itself.
    n_procs:
        Number of fork workers (≈ cores to use).
    dynamic_omp:
        When True (default) each worker runs a daemon thread that grows its torch
        thread count as games complete, so the stragglers absorb the freed cores
        and the tail wall shrinks. Set False to pin every worker to 1 thread for
        the whole run (the classic 1-game-per-core behaviour).
    worker_progress:
        Optional ``mp.Manager().dict()`` the caller owns so it can READ each core's
        live game (one row per slot 0..n_procs-1, written every
        ``progress_step_every`` learner steps via the ``run_episode`` on_step hook).
        When None a private Manager dict is created (the rollout still works, the
        caller just can't watch it). Pass one in to mirror per-core progress out.
    progress_step_every:
        Throttle — each worker republishes its slot row every Nth learner step
        (default 10) to bound the shared-dict write rate.
    on_episode_done:
        Optional parent-process callback ``(idx, EpisodeBuffer | None)`` invoked
        as soon as each game result arrives from ``imap_unordered``. This lets the
        trainer feed completed trajectories into a downstream packing/GAE-prep
        queue while other game workers are still stepping.

    Returns
    -------
    ``list[EpisodeBuffer | None]`` in the SAME order as ``specs`` (failed games
    are ``None`` — the caller decides whether to drop them). The count of
    failures is logged to stderr.
    """
    global _POLICY, _OPP, _PLANET_ENC, _FLEET_ENC, _COMET_ENC, _KW
    global _REMAINING, _N_PROCS, _DYNAMIC_OMP
    global _WORKER_PROGRESS, _SLOT_CTR, _PROGRESS_STEP_EVERY

    # ---- FD-exhaustion fix (part 1, the PRIMARY one): switch torch's tensor-
    # sharing strategy to ``file_system`` BEFORE creating the pool. The default
    # ``file_descriptor`` strategy backs every shared-memory tensor (the large
    # EpisodeBuffers each worker returns via imap_unordered -> pickle -> torch
    # reduction) with an OPEN FD; 16 T=10 games' worth of tensors exhaust the 1024
    # default and the pool dies with "[Errno 24] Too many open files".
    # ``file_system`` keys shared memory by NAME (no per-tensor FD), eliminating
    # the leak. Set in the PARENT so every forked child inherits it. Best-effort. -
    try:
        import torch.multiprocessing as tmp
        tmp.set_sharing_strategy("file_system")
    except Exception:  # noqa: BLE001
        pass
    # Also raise THIS (parent) process's soft FD limit, so the parent side of the
    # returned-tensor handles + the per-worker pipes never hit the 1024 wall.
    _raise_fd_limit()

    # Inference mode BEFORE forking -> COW pages stay clean / shared.
    _to_inference(policy)
    if opponent_policy is not None:
        _to_inference(opponent_policy)

    # Keep the PARENT single-threaded too: a live multi-thread MKL/OpenMP pool
    # in the parent can deadlock a forked child mid-allocation. CPU-only here.
    torch.set_num_threads(1)

    n_procs = max(1, min(int(n_procs), len(specs))) if specs else 1
    ctx = multiprocessing.get_context("fork")

    # Per-core live progress: a shared Manager dict keyed by worker SLOT (0..n_procs-
    # 1) + a shared counter the initializer hands slots out from (round-robin, one
    # per worker process). If the caller did not supply a dict, make a private one so
    # the on_step path is always safe to call.
    _progress_mgr = None
    if worker_progress is None:
        _progress_mgr = ctx.Manager()
        worker_progress = _progress_mgr.dict()
    slot_ctr = ctx.Value("i", 0)
    progress_step_every = max(1, int(progress_step_every))

    # Shared straggler counter: ONE physical int across all fork workers, seeded
    # to the total game count and decremented as games finish (in _play_one). The
    # workers' daemon threads poll it to decide how many cores to claim. Created
    # from the fork ctx so the children inherit the same shared-memory segment.
    kw = {
        "sigma": sigma,
        "max_planets": max_planets,
        "max_fleets": max_fleets,
        "noop_logit_bias": noop_logit_bias,
        "select_logit_bias": select_logit_bias,
        "history_window": history_window,
        "target_cap_k_max": target_cap_k_max,
        "target_cap_lambda": target_cap_lambda,
    }
    remaining = ctx.Value("i", len(specs)) if (dynamic_omp and specs) else None

    # Publish into the PARENT module globals too. This keeps the COW-inheritance
    # contract the docstring describes (a forked child already sees fully-built
    # objects) and lets the parent reset cleanly; the Pool initializer re-binds the
    # very same objects per worker (and is what guarantees one straggler thread per
    # worker process, not per task).
    _POLICY, _OPP = policy, opponent_policy
    _PLANET_ENC, _FLEET_ENC, _COMET_ENC = planet_enc, fleet_enc, comet_enc
    _KW = kw
    _REMAINING = remaining
    _N_PROCS = n_procs
    _DYNAMIC_OMP = bool(dynamic_omp)
    _WORKER_PROGRESS = worker_progress
    _SLOT_CTR = slot_ctr
    _PROGRESS_STEP_EVERY = progress_step_every

    results: dict[int, object] = {}
    try:
        with ctx.Pool(
            n_procs,
            initializer=_pool_init,
            initargs=(policy, opponent_policy, planet_enc, fleet_enc, comet_enc, kw,
                      remaining, n_procs, bool(dynamic_omp),
                      worker_progress, slot_ctr, progress_step_every),
        ) as pool:
            # imap_unordered: next spec goes to whichever worker is free -> work
            # stealing. We tag each spec with its index and re-sort at the end.
            for idx, buf in pool.imap_unordered(_play_one, list(enumerate(specs))):
                results[idx] = buf
                if on_episode_done is not None:
                    on_episode_done(idx, buf)
    finally:
        # Shut down a privately-created Manager (the caller's own dict is left for it
        # to drain). Best-effort — never mask a rollout result/exception.
        if _progress_mgr is not None:
            try:
                _progress_mgr.shutdown()
            except Exception:  # noqa: BLE001
                pass

    out = [results.get(i) for i in range(len(specs))]
    n_failed = sum(1 for b in out if b is None)
    if n_failed:
        sys.stderr.write(
            f"[pool_rollout] {n_failed}/{len(specs)} games failed (returned None)\n"
        )
    return out


# --------------------------------------------------------------------------- #
# RSS helpers (macOS: ru_maxrss is BYTES; Linux: KiB)                          #
# --------------------------------------------------------------------------- #
def _maxrss_mb(who: int) -> float:
    ru = resource.getrusage(who).ru_maxrss
    return (ru / 1024.0 / 1024.0) if sys.platform == "darwin" else (ru / 1024.0)


# --------------------------------------------------------------------------- #
# CLI bench                                                                    #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path, help="Supervised entity ckpt.")
    ap.add_argument("--planet-run-dir", type=Path, default=None)
    ap.add_argument("--fleet-run-dir", type=Path, default=None)
    ap.add_argument("--comet-run-dir", type=Path, default=None)
    ap.add_argument("--games", type=int, default=4, help="M: total games to play.")
    ap.add_argument("--procs", type=int, default=2, help="N: fork-pool workers.")
    ap.add_argument("--history-window", type=int, default=10)
    ap.add_argument("--num-players", default="mix", choices=("2", "4", "mix"))
    ap.add_argument("--seed-base", type=int, default=100_000)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--noop-logit-bias", type=float, default=0.0)
    ap.add_argument("--target-cap-k-max", type=int, default=4)
    ap.add_argument("--target-cap-lambda", type=float, default=0.0)
    ap.add_argument("--max-planets", type=int, default=64)
    ap.add_argument("--max-fleets", type=int, default=1024)
    ap.add_argument("--opponent", default="", help="'' = self-play snapshot; else heuristic id.")
    ap.add_argument("--no-dynamic-omp", action="store_true",
                    help="disable the dynamic-OMP-for-stragglers ramp (pin every "
                         "worker to 1 thread for the whole run).")
    args = ap.parse_args()

    # Single-thread the parent before doing any torch work, so the model load
    # and (more importantly) the fork inherit a quiescent thread state.
    torch.set_num_threads(1)

    t_load = time.time()
    entity_model, fleet_enc, planet_enc, comet_enc, cfg = smoke.load_supervised(
        args.ckpt,
        "cpu",
        planet_run_dir=args.planet_run_dir,
        fleet_run_dir=args.fleet_run_dir,
        comet_run_dir=args.comet_run_dir,
    )
    policy = PPOActorCritic(
        entity_model, sigma=args.sigma, allow_debug_glob_critic=True,
    ).to("cpu")
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    opp_policy = None
    if not args.opponent:
        opp_policy = _build_opponent(entity_model, cfg, policy, "cpu", args.sigma)
    load_s = time.time() - t_load
    print(
        f"[pool_rollout] loaded model ONCE in {load_s:.1f}s "
        f"(d_model={cfg.get('d_model')}, n_steps={cfg.get('n_steps')}); "
        f"opponent={'self-play snapshot' if opp_policy is not None else args.opponent}",
        flush=True,
    )

    # specs = [(seed, num_players, learner_seat)] — reuse the balanced-mix RNG
    # from rollout_worker, then rotate the learner through the seats.
    base_specs = _episode_specs(args.games, args.seed_base, args.num_players)
    specs = [(s, npl, i % npl) for i, (s, npl) in enumerate(base_specs)]
    n2 = sum(1 for _, npl, _ in specs if npl == 2)
    n4 = sum(1 for _, npl, _ in specs if npl == 4)
    print(
        f"[pool_rollout] {args.games} games ({n2}×2P, {n4}×4P) across "
        f"{args.procs} fork workers; history_window={args.history_window}",
        flush=True,
    )

    t_roll = time.time()
    buffers = run_pool_rollout(
        policy=policy,
        opponent_policy=opp_policy,
        planet_enc=planet_enc,
        fleet_enc=fleet_enc,
        comet_enc=comet_enc,
        specs=specs,
        n_procs=args.procs,
        sigma=args.sigma,
        max_planets=args.max_planets,
        max_fleets=args.max_fleets,
        noop_logit_bias=args.noop_logit_bias,
        history_window=args.history_window,
        target_cap_k_max=args.target_cap_k_max,
        target_cap_lambda=args.target_cap_lambda,
        dynamic_omp=not args.no_dynamic_omp,
    )
    roll_s = time.time() - t_roll

    n_failed = sum(1 for b in buffers if b is None)
    n_ok = len(buffers) - n_failed
    total_steps = sum(len(b.steps) for b in buffers if b is not None)

    print("[pool_rollout] ===== RESULT =====", flush=True)
    print(
        f"[pool_rollout] wall={roll_s:.1f}s  games_ok={n_ok}/{len(buffers)}  "
        f"failed={n_failed}  total_steps={total_steps}  "
        f"({roll_s / max(1, n_ok):.1f}s/game, {roll_s / max(1, total_steps):.3f}s/step)",
        flush=True,
    )
    print(f"[pool_rollout]   player split: 2P={n2}  4P={n4}", flush=True)
    for i, (buf, (seed, npl, seat)) in enumerate(zip(buffers, specs)):
        if buf is None:
            print(f"[pool_rollout]   game {i:02d} seed={seed} {npl}P seat={seat} FAILED", flush=True)
            continue
        nsteps = len(buf.steps)
        won = buf.winner == seat
        print(
            f"[pool_rollout]   game {i:02d} seed={seed} {npl}P seat={seat} "
            f"steps={nsteps} winner={buf.winner} {'WIN' if won else 'lose'}",
            flush=True,
        )

    parent_rss = _maxrss_mb(resource.RUSAGE_SELF)
    child_rss = _maxrss_mb(resource.RUSAGE_CHILDREN)
    print(
        f"[pool_rollout]   peak RSS: parent={parent_rss:.0f} MB  "
        f"children(peak-of-reaped)={child_rss:.0f} MB",
        flush=True,
    )
    print(
        "[pool_rollout]   NOTE: ru_maxrss is per-process and excludes the live "
        "COW sharing — because the model was loaded once and frozen before the "
        "fork, the true resident total stays ≈ one model + per-worker env "
        "scratch, NOT n_procs × model.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
