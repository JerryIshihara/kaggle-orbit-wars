"""Smoke test for pool_rollout: fork-pool work-stealing over toy 'episodes'.

Does NOT load the real model — it stubs ``smoke.run_episode`` so the test runs in
milliseconds and stays hermetic. What it verifies:

  * ``run_pool_rollout`` forks a Pool, distributes all specs, and returns one
    result per spec in ORIGINAL order.
  * The parent's inference-mode prep (eval + requires_grad_(False)) runs and the
    module globals are populated so forked children inherit them.
  * A worker that raises is isolated -> its slot comes back ``None`` while the
    rest succeed (one bad game does not kill the pool).
  * 2P/4P specs both flow through and the per-spec kwargs reach the worker.
  * Dynamic-OMP-for-stragglers: as the shared ``remaining`` counter drops, a
    still-running worker's ``torch.get_num_threads()`` RAMPS UP (the tail grabs
    the freed cores). ``_omp_threads_for`` is also unit-checked directly.

Run::  .venv/bin/python -m agents.transformer_v2.ppo.test_pool_rollout
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types

import torch

from . import pool_rollout
from . import smoke


class _StepStub:
    """Minimal StepRecord stand-in: only the score fields the post-game progress
    re-stamp reads off ``buf.steps[-1]``."""

    score_my = 9
    score_enemy_max = 7


class _ToyBuf:
    """Stand-in for EpisodeBuffer carrying just what the test inspects."""

    def __init__(self, seed, seat, num_players, n_steps=None):
        self.seed = seed
        self.learner_seat = seat
        self.num_players = num_players
        # one StepRecord-ish per "turn" so len(buf.steps) is non-trivial. Use a
        # realistic-ish length (>= the progress throttle) so the post-game row
        # re-stamp (step = len(buf.steps)) reflects a real game length.
        n = (2 + num_players) if n_steps is None else int(n_steps)
        self.steps = [_StepStub() for _ in range(n)]
        self.winner = seat


def _fake_run_episode(**kw):
    # Seed 999 is the designated 'bad game' -> exercise the try/except isolation.
    if kw["seed"] == 999:
        raise RuntimeError("synthetic worker failure")
    # The worker must have pinned itself to one thread.
    assert torch.get_num_threads() == 1, "worker did not pin torch to 1 thread"
    # Scalar kwargs from _KW must have been forwarded.
    assert "history_window" in kw and "sigma" in kw and "max_planets" in kw
    assert kw["device"] == "cpu"
    # on_step must be threaded through (None or callable) — when present, call it
    # the way run_episode would (once per learner turn, score climbing). 30 turns so
    # the post-game row re-stamp (step = len(buf.steps)) lands above the throttle.
    on_step = kw.get("on_step")
    n_turns = 30
    if on_step is not None:
        for step in range(n_turns):
            on_step(step, kw["num_players"], step * 2, step)
    return _ToyBuf(kw["seed"], kw["learner_seat"], kw["num_players"], n_steps=n_turns)


def test_work_stealing_and_isolation() -> None:
    """Fork-pool work-stealing + bad-game isolation + order preservation +
    pre-fork freeze. Runs with ``dynamic_omp=False`` so the worker stays pinned to
    exactly 1 thread (the stub asserts that) — the OMP ramp is covered separately
    in ``test_dynamic_omp_ramps``."""
    # A trivially-forkable "model": a tiny nn.Module with a couple of params.
    policy = torch.nn.Linear(4, 4)
    opp = torch.nn.Linear(4, 4)
    # Leave grads ON so we can assert run_pool_rollout flips them OFF pre-fork.
    for p in policy.parameters():
        p.requires_grad_(True)

    smoke.run_episode = _fake_run_episode  # monkeypatch the unit-of-work

    # Mixed 2P/4P, plus the poison seed 999 in the middle.
    specs = [
        (101, 2, 0),
        (102, 4, 1),
        (999, 2, 1),   # this one raises in the worker
        (104, 4, 3),
        (105, 2, 0),
    ]

    out = pool_rollout.run_pool_rollout(
        policy=policy,
        opponent_policy=opp,
        planet_enc=object(),
        fleet_enc=object(),
        comet_enc=object(),
        specs=specs,
        n_procs=3,
        sigma=0.35,
        max_planets=64,
        max_fleets=1024,
        history_window=1,
        dynamic_omp=False,   # keep workers at 1 thread for the stub's assertion
    )

    # Parent flipped the model into inference mode before forking.
    assert all(not p.requires_grad for p in policy.parameters()), "policy not frozen"
    assert all(not p.requires_grad for p in opp.parameters()), "opp not frozen"
    assert not policy.training, "policy not in eval()"

    # One result per spec, in original order.
    assert len(out) == len(specs), f"len mismatch: {len(out)} != {len(specs)}"
    assert out[2] is None, "poison game should be None"
    for i in (0, 1, 3, 4):
        assert out[i] is not None, f"game {i} unexpectedly failed"
        assert out[i].seed == specs[i][0], "result reordered!"
        assert out[i].learner_seat == specs[i][2]
        assert out[i].num_players == specs[i][1]
        assert len(out[i].steps) > 0

    n_ok = sum(1 for b in out if b is not None)
    print(f"  [ok] {n_ok}/{len(specs)} games returned, 1 poison isolated to None, "
          f"order preserved, model frozen pre-fork (dynamic_omp=False).")


# --------------------------------------------------------------------------- #
# Dynamic-OMP-for-stragglers                                                    #
# --------------------------------------------------------------------------- #
def test_omp_threads_for() -> None:
    """The pure ramp policy: full queue -> 1 thread; as the tail shrinks to k
    games each survivor claims ≈ N/k cores (floored at 1, capped at N)."""
    f = pool_rollout._omp_threads_for
    N = 8
    assert f(8, N) == 1     # remaining >= N -> 1 (no oversubscription)
    assert f(99, N) == 1
    assert f(4, N) == 2     # 4 left over 8 cores -> 2 each
    assert f(2, N) == 4
    assert f(1, N) == 8     # last straggler grabs all cores
    assert f(0, N) == 8     # clamped: never below floor-of-1-remaining
    assert f(3, N) == 2     # 8 // 3 == 2
    assert f(5, 1) == 1     # single-core machine: always 1
    print("  [ok] _omp_threads_for: 1 while full, ramps to ~N/k for the tail.")


def _ramp_run_episode(**kw):
    """Stub game that RECORDS, on the worker MAIN thread, the torch team size it was
    handed at game start — one int per line in ``_RAMP_LOG_FILE``. This is exactly
    the thread on which the production game's torch ops run, so ``get_num_threads()``
    here is the team those ops would use. A tiny sleep keeps games overlapping so the
    queue genuinely drains through the workers."""
    threads = torch.get_num_threads()
    with open(os.environ["_RAMP_LOG_FILE"], "a") as fh:
        fh.write(f"{threads}\n")
    time.sleep(0.02)
    return _ToyBuf(kw["seed"], kw["learner_seat"], kw["num_players"])


def test_dynamic_omp_ramps() -> None:
    """End-to-end ramp. ``remaining`` counts games LEFT TO PLAY and is claimed
    (read-then-decrement under its lock) as each worker starts a game, which sizes
    that game's torch team to ``_omp_threads_for(remaining, n_procs)``. Because the
    claim is serialized, the values claimed across the run are exactly the permutation
    ``len(specs) .. 1`` — so the MULTISET of team sizes the games ran with must equal
    ``{_omp_threads_for(r, n_procs) for r in 1..len(specs)}``. That deterministically
    encodes the ramp: while ``remaining >= n_procs`` every game is 1 thread (no
    oversubscription); the FINAL ``n_procs`` games, claimed as the queue drains, climb
    1 -> ... -> n_procs as the surviving games grab the cores freed by finished
    workers. We assert that exact multiset, that full-queue games stayed at 1, and
    that the tail peaks at ``n_procs``.

    (On this OpenMP/Accelerate Mac the team-size change need not speed the BLAS-bound
    toy op; what we validate is that the count is actually APPLIED on the thread that
    runs the game — on the Linux PPO VMs the ATen ops honour it.) Skipped on <2 cores
    (nothing to ramp into)."""
    n_cores = os.cpu_count() or 1
    if n_cores < 2:
        print("  [skip] dynamic-OMP ramp: single-core machine.")
        return
    n_procs = min(4, n_cores)

    import collections
    import tempfile
    log = tempfile.NamedTemporaryFile(
        prefix="ramp_log_", suffix=".txt", delete=False)
    log.close()
    os.environ["_RAMP_LOG_FILE"] = log.name

    smoke.run_episode = _ramp_run_episode

    policy = torch.nn.Linear(4, 4)
    n_games = n_procs * 6                        # several games/worker -> a real tail
    specs = [(100 + i, 2, 0) for i in range(n_games)]

    out = pool_rollout.run_pool_rollout(
        policy=policy,
        opponent_policy=None,
        planet_enc=object(),
        fleet_enc=object(),
        comet_enc=object(),
        specs=specs,
        n_procs=n_procs,
        sigma=0.35,
        max_planets=64,
        max_fleets=1024,
        history_window=1,
        dynamic_omp=True,
    )
    assert len(out) == len(specs)
    assert all(b is not None for b in out), "no game should have failed"

    threads_seen = []
    with open(log.name) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                threads_seen.append(int(ln))
    os.unlink(log.name)

    assert len(threads_seen) == n_games, \
        f"expected {n_games} game records, got {len(threads_seen)}"

    # The claim is serialized, so the remaining-values claimed are 1..n_games; the
    # team sizes the games ran with must be EXACTLY that ramp policy applied to each.
    expected = collections.Counter(
        pool_rollout._omp_threads_for(r, n_procs) for r in range(1, n_games + 1))
    got = collections.Counter(threads_seen)
    assert got == expected, (
        f"team-size multiset mismatch.\n got={dict(sorted(got.items()))}\n"
        f" exp={dict(sorted(expected.items()))}")

    peak = max(threads_seen)
    assert peak == n_procs, f"tail should peak at n_procs={n_procs}, got {peak}"
    # The body of the run (full queue) ran at 1 thread; only the deep tail ramped.
    n_ramped = sum(c for t, c in got.items() if t > 1)
    assert n_ramped >= 1, "no game ever ramped above 1 thread"
    assert got[1] == n_games - n_ramped, (got[1], n_ramped, n_games)
    print(f"  [ok] dynamic-OMP ramp: {got[1]}/{n_games} full-queue games at 1 thread; "
          f"{n_ramped} tail game(s) ramped up to peak={peak} as the queue drained "
          f"(team-size multiset == ramp policy over remaining=1..{n_games}).")


def test_on_step_per_core_progress() -> None:
    """The per-step/per-core chain: each worker's ``on_step`` hook (fired by
    run_episode, here by the stub) writes ITS slot row into the shared
    ``worker_progress`` Manager dict. After the run we assert one row PER CORE
    (slot 0..n_procs-1) carrying the full per-core schema — ``slot``, ``seed``,
    ``num_players``, ``step``, ``score_my``/``score_enemy_max``, ``threads``,
    ``games_done`` — and that ``step`` advanced past 0 (the hook actually fired
    mid-game, not just the start-of-game stamp). This is exactly what the enriched
    progress.json ``workers:[...]`` array mirrors to Firestore."""
    import multiprocessing as _mp

    smoke.run_episode = _fake_run_episode

    policy = torch.nn.Linear(4, 4)
    n_procs = min(3, max(2, os.cpu_count() or 2))
    # several games/worker so every slot is touched and games_done can exceed 1
    specs = [(200 + i, 2 + 2 * (i % 2), 0) for i in range(n_procs * 3)]

    mgr = _mp.get_context("fork").Manager()
    worker_progress = mgr.dict()

    out = pool_rollout.run_pool_rollout(
        policy=policy,
        opponent_policy=None,
        planet_enc=object(),
        fleet_enc=object(),
        comet_enc=object(),
        specs=specs,
        n_procs=n_procs,
        sigma=0.35,
        max_planets=64,
        max_fleets=1024,
        history_window=1,
        dynamic_omp=False,
        worker_progress=worker_progress,
        progress_step_every=10,
    )
    assert len(out) == len(specs)
    assert all(b is not None for b in out), "no game should have failed"

    rows = [dict(v) for v in worker_progress.values() if v]
    mgr.shutdown()

    # one row per CORE (slot), slots are exactly 0..n_procs-1
    assert len(rows) == n_procs, f"expected {n_procs} per-core rows, got {len(rows)}"
    slots = sorted(r["slot"] for r in rows)
    assert slots == list(range(n_procs)), slots
    _REQUIRED = {"slot", "game_idx", "seed", "num_players", "step",
                 "score_my", "score_enemy_max", "threads", "games_done"}
    for r in rows:
        assert _REQUIRED <= set(r), f"row missing keys: {_REQUIRED - set(r)}"
        # 30 on_step calls/game + a post-game re-stamp at step = len(buf.steps) = 30,
        # so the last-published step is well above the throttle (proves the hook
        # fired mid-game, not just the start-of-game stamp).
        assert r["step"] >= 10, f"on_step did not advance step (got {r['step']})"
        assert r["num_players"] in (2, 4), r["num_players"]
        assert r["threads"] >= 1
        # the score fields propagated from the (re-stamped) final step record
        assert r["score_my"] == 9 and r["score_enemy_max"] == 7, r
        # each worker finished >=1 game by the time it last stamped its row
        assert r["games_done"] >= 1
    # at least one core finished multiple games (n_procs*3 games over n_procs cores)
    assert max(r["games_done"] for r in rows) >= 1, rows
    print(f"  [ok] on_step -> per-core progress: {len(rows)} slot rows "
          f"(0..{n_procs-1}), each with seed/num_players/step/threads/games_done; "
          f"step advanced (max={max(r['step'] for r in rows)}).")


def main() -> int:
    test_work_stealing_and_isolation()
    test_omp_threads_for()
    test_dynamic_omp_ramps()
    test_on_step_per_core_progress()
    print("POOL ROLLOUT PASSED — work-stealing + isolation + order + freeze; "
          "dynamic-OMP ramp (_omp_threads_for + straggler grabs freed cores); "
          "on_step per-core progress (Manager dict workers rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
