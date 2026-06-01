"""ppo_actor_daemon — one mocked iteration over a LOCAL STATE.json.

The REAL rollout needs the supervised weights and is very slow, so the worker
subprocess is fully MOCKED: ``subprocess.Popen`` is replaced by a fake that prints
a couple of stdout lines AND writes a fake ``progress.json`` + an empty ``_DONE``
into the worker's ``--out`` prefix (exactly what the real worker does), then exits
0. STATE lives on the local filesystem (shards handles local paths), so no GCS and
no torch-heavy env is touched.

We assert the daemon drove the phase ping-pong ``await_rollout -> rolling_out ->
await_train`` for iter 0 and that it mirrored the worker's ``progress`` + streamed
stdout (``log_n``) into ``STATE['actor']``.

    python -m agents.transformer_v2.ppo.test_actor_daemon
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from agents.transformer_v2.ppo import ppo_actor_daemon, ppo_state, shards

_FAKE_PROGRESS = {
    "policy_version": 0, "ep_done": 16, "ep_total": 16, "cur_step": 120,
    "n_active": 0, "state": "done", "rss_mb": 812.3, "avail_mb": 5120.0,
    "peak_rss_mb": 933.1, "cpu_pct": 287.0, "elapsed_s": 42.0, "eta_s": 0.0,
    # ENRICHED pool fields: per-core ``workers`` rows + aggregates. The daemon must
    # mirror these VERBATIM into STATE.<actor>.progress (a small list of small
    # dicts), so the monitor can show each core's live game.
    "workers": [
        {"slot": 0, "game_idx": 2, "seed": 100002, "num_players": 2,
         "step": 212, "score_my": 5, "score_enemy_max": 3, "threads": 1,
         "games_done": 1},
        {"slot": 1, "game_idx": 3, "seed": 100003, "num_players": 4,
         "step": 188, "score_my": 4, "score_enemy_max": 6, "threads": 1,
         "games_done": 1},
    ],
    "n_workers": 2, "total_steps": 400, "games_done": 2,
    "steps_per_s": 7.5, "mean_step": 200.0,
    "ts": "2026-05-31T12:00:42Z",
}
_FAKE_LINES = [
    "[rollout] START policy_v0 hw=10 episodes=16 players=mix",
    "[rollout] DONE v0: 16 eps, 1234 steps, 9 wins",
]


class _FakeProc:
    """Stand-in for the rollout_worker subprocess: prints lines via .stdout, and
    on construction writes the worker's progress.json + _DONE into --out."""

    def __init__(self, cmd, **kwargs):
        # find the --out prefix the daemon passed
        out = cmd[cmd.index("--out") + 1]
        outdir = Path(out)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "progress.json").write_text(json.dumps(_FAKE_PROGRESS))
        (outdir / "_DONE").write_text("")          # worker writes this LAST
        self.returncode = 0
        # stdout is an iterator of newline-terminated lines (text mode, bufsize=1)
        self.stdout = iter([ln + "\n" for ln in _FAKE_LINES])

    def wait(self):
        return 0


def _make_fake_popen(seen_cmd: list):
    def _factory(cmd, **kwargs):
        seen_cmd.append(list(cmd))
        return _FakeProc(cmd, **kwargs)
    return _factory


class _FakeWorkerProc:
    """Fan-out worker stand-in: writes the per-worker ``w{j}/progress.json`` +
    ``w{j}/manifest.json`` + a fake shard into its ``--out`` (mirroring what the
    real rollout_worker writes under each worker subdir), then reports ``poll()``/
    ``wait()`` == 0. Unlike ``_FakeProc`` it does NOT write a ``_DONE`` — the
    DAEMON writes the actor-level ``{actor_id}/_DONE`` after all workers exit."""

    def __init__(self, cmd, **kwargs):
        out = cmd[cmd.index("--out") + 1]          # .../rollouts/vK/<id>/w{j}/
        eps = cmd[cmd.index("--episodes") + 1]
        seed = cmd[cmd.index("--seed-base") + 1]
        outdir = Path(out)
        outdir.mkdir(parents=True, exist_ok=True)
        prog = dict(_FAKE_PROGRESS)
        prog["ep_done"] = int(eps)
        prog["ep_total"] = int(eps)
        (outdir / "progress.json").write_text(json.dumps(prog))
        # a fake shard + manifest the learner's rglob('manifest.json') would find
        (outdir / f"shard_{seed}.pt").write_text("FAKE_SHARD")
        (outdir / "manifest.json").write_text(json.dumps(
            {"policy_version": 0, "shards": [{"name": f"shard_{seed}.pt"}]}))
        self.returncode = 0
        self.stdout = iter([f"[worker out={out} eps={eps} seed={seed}]\n"])

    def poll(self):
        return 0          # already "exited" 0 (synchronous fake)

    def wait(self):
        return 0


def _make_fake_worker_popen(seen_cmd: list, seen_env: list):
    def _factory(cmd, **kwargs):
        seen_cmd.append(list(cmd))
        seen_env.append(dict(kwargs.get("env") or {}))
        return _FakeWorkerProc(cmd, **kwargs)
    return _factory


def test_one_iteration(monkeypatch=None) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ppo_actor_test_"))
    run_prefix = str(tmp / "ppo" / "run_test")
    url = ppo_state.state_url(run_prefix)

    # seed a LOCAL STATE.json at await_rollout iter 0
    model = ppo_state.model_block(run_prefix, policy_version=0, base_version="basev0abc")
    # the actor downloads model['heads'] -> create that local file so gcs_cp works
    heads_local = Path(model["heads"])
    heads_local.parent.mkdir(parents=True, exist_ok=True)
    heads_local.write_text("FAKE_HEAD_DELTA")
    config = {"history_window": 10, "episodes": 16, "num_players": "mix",
              "seed_base": 100000, "device": "cpu", "max_fleets": 512, "sigma": 0.35}
    # gcs_prefix is the run root for BIG files; the daemon reads big-file (rollout)
    # paths from it, NOT from the state url's parent. Here it == run_prefix (local).
    ppo_state.init_state(url, run_id="run_test", iters_total=3, model=model,
                         config=config, gcs_prefix=run_prefix)

    # ---- record every transition (expect_phase -> new_phase) the daemon makes ----
    transitions: list[tuple] = []
    real_transition = ppo_state.transition

    def _spy_transition(u, *, expect_phase, new_phase, who, **fields):
        transitions.append((expect_phase, new_phase))
        return real_transition(u, expect_phase=expect_phase, new_phase=new_phase,
                               who=who, **fields)

    # ---- MOCK the worker subprocess (never run the real, slow rollout) ----
    seen_cmd: list = []
    fake_popen = _make_fake_popen(seen_cmd)

    if monkeypatch is not None:
        monkeypatch.setattr(ppo_actor_daemon.ppo_state, "transition", _spy_transition)
        monkeypatch.setattr(ppo_actor_daemon.subprocess, "Popen", fake_popen)
    else:
        ppo_actor_daemon.ppo_state.transition = _spy_transition
        ppo_actor_daemon.subprocess.Popen = fake_popen

    workdir = tmp / "work"
    argv = ["ppo_actor_daemon", "--state", run_prefix, "--actor-id", "machineB",
            "--ckpt", "ckpts/entity/x.pt", "--planet-run-dir", "ckpts/planet",
            "--fleet-run-dir", "ckpts/fleet", "--comet-run-dir", "ckpts/comet",
            "--num-envs", "4", "--poll-s", "0.1", "--progress-every", "0.0",
            "--workdir", str(workdir), "--once"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = ppo_actor_daemon.main()
    finally:
        sys.argv = old_argv
        if monkeypatch is None:
            ppo_actor_daemon.ppo_state.transition = real_transition

    # ---- assertions ----
    assert rc == 0, f"daemon returned {rc}"

    # phase ping-pong: claimed await_rollout->rolling_out, then rolling_out->await_train
    assert transitions == [
        (ppo_state.PHASE_AWAIT_ROLLOUT, ppo_state.PHASE_ROLLING_OUT),
        (ppo_state.PHASE_ROLLING_OUT, ppo_state.PHASE_AWAIT_TRAIN),
    ], transitions

    final = ppo_state.read_state(url)
    assert final["phase"] == ppo_state.PHASE_AWAIT_TRAIN, final["phase"]
    assert final["iter"] == 0, final["iter"]
    assert final["shards"] == f"{run_prefix}/rollouts/v0/machineB/", final["shards"]
    assert final["updated_by"] == "machineB"

    # the worker delta got downloaded into the workdir
    assert (workdir / "policy_v0.heads.pt").read_text() == "FAKE_HEAD_DELTA"

    # actor.progress mirrored from the worker progress.json
    actor = final["actor"]
    assert actor["id"] == "machineB" and actor["claimed_iter"] == 0
    assert actor["progress"] == _FAKE_PROGRESS, actor["progress"]
    # the ENRICHED per-core ``workers`` array + aggregates survived the STATE write
    # intact (this is what powers the per-core live monitor view).
    prog = actor["progress"]
    assert isinstance(prog.get("workers"), list) and len(prog["workers"]) == 2, prog
    w0 = prog["workers"][0]
    assert {"slot", "seed", "num_players", "step", "score_my",
            "score_enemy_max", "threads", "games_done"} <= set(w0), w0
    assert w0["slot"] == 0 and w0["step"] == 212 and w0["num_players"] == 2, w0
    assert prog["total_steps"] == 400 and prog["games_done"] == 2, prog
    assert prog["steps_per_s"] == 7.5 and prog["n_workers"] == 2, prog
    # streamed stdout mirrored: log_n counts both lines, tail carries them
    assert actor["log_n"] == len(_FAKE_LINES), actor["log_n"]
    assert any("DONE v0" in ln for ln in actor["log_tail"]), actor["log_tail"]
    assert actor["heartbeat_ts"]

    # the worker was invoked exactly once with the EXACT command shape we expect
    assert len(seen_cmd) == 1, seen_cmd
    cmd = seen_cmd[0]
    assert cmd[0] == sys.executable
    assert cmd[1:5] == ["-u", "-m", "agents.transformer_v2.ppo.rollout_worker", "--ckpt"]
    # spot-check the load-bearing flags
    def _flag(name):
        return cmd[cmd.index(name) + 1]
    assert _flag("--policy-version") == "0"
    assert _flag("--base-version") == "basev0abc"
    assert _flag("--policy-heads") == str(workdir / "policy_v0.heads.pt")
    # the rollout out-prefix is built under STATE['gcs_prefix'] (== run_prefix here),
    # NOT derived from the state url's parent dir
    gcs_prefix = ppo_state.gcs_prefix(final)
    assert gcs_prefix == run_prefix, gcs_prefix
    assert _flag("--out") == f"{gcs_prefix}/rollouts/v0/machineB/"
    assert _flag("--out").startswith(gcs_prefix + "/"), _flag("--out")
    assert _flag("--vm-id") == "machineB"
    assert _flag("--episodes") == "16"
    assert _flag("--num-players") == "mix"
    assert _flag("--history-window") == "10"
    assert _flag("--num-envs") == "4"
    assert _flag("--rollout-workers") == "1"
    assert _flag("--max-fleets") == "512"
    assert _flag("--sigma") == "0.35"
    assert _flag("--device") == "cpu"
    # seed_base = config.seed_base(100000) + K(0)*episodes(16) = 100000
    assert _flag("--seed-base") == "100000"
    assert "--verbose" in cmd

    print("  [ok] await_rollout -> rolling_out -> await_train; "
          f"actor.progress mirrored (ep_done={actor['progress']['ep_done']}), "
          f"log_n={actor['log_n']}; worker cmd verified")


def test_idempotent_skip(monkeypatch=None) -> None:
    """If _DONE already exists for iter K, the daemon skips the rollout and flips
    straight await_rollout -> await_train (output files are ground truth)."""
    tmp = Path(tempfile.mkdtemp(prefix="ppo_actor_idem_"))
    run_prefix = str(tmp / "ppo" / "run_idem")
    url = ppo_state.state_url(run_prefix)
    model = ppo_state.model_block(run_prefix, policy_version=0, base_version="b0")
    ppo_state.init_state(url, run_id="run_idem", iters_total=3, model=model,
                         config={"episodes": 16}, gcs_prefix=run_prefix)

    # pre-place the _DONE marker (a prior crashed/finished run)
    out_prefix = f"{run_prefix}/rollouts/v0/machineB/"
    Path(out_prefix).mkdir(parents=True, exist_ok=True)
    (Path(out_prefix) / "_DONE").write_text("")

    # Popen MUST NOT be called on the idempotent path
    called = {"n": 0}

    def _boom(cmd, **kwargs):
        called["n"] += 1
        raise AssertionError("rollout_worker should NOT run when _DONE exists")

    if monkeypatch is not None:
        monkeypatch.setattr(ppo_actor_daemon.subprocess, "Popen", _boom)
    else:
        ppo_actor_daemon.subprocess.Popen = _boom

    argv = ["ppo_actor_daemon", "--state", run_prefix, "--actor-id", "machineB",
            "--ckpt", "x.pt", "--poll-s", "0.1", "--once"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = ppo_actor_daemon.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert called["n"] == 0, "worker was launched despite existing _DONE"
    final = ppo_state.read_state(url)
    assert final["phase"] == ppo_state.PHASE_AWAIT_TRAIN
    assert final["shards"] == out_prefix
    print("  [ok] idempotent: existing _DONE -> skip rollout, flip to await_train")


def _run_actor_once(run_prefix: str, actor_id: str, popen_factory, workdir: Path):
    """Drive ONE ``--once`` iteration of the actor daemon for ``actor_id`` against
    the LOCAL STATE at ``run_prefix``, with ``subprocess.Popen`` mocked. Returns
    the daemon's exit code. (No spy on transition — the multi-actor test asserts on
    the resulting STATE, not the transition trace, since the flip is conditional.)"""
    saved = ppo_actor_daemon.subprocess.Popen
    ppo_actor_daemon.subprocess.Popen = popen_factory
    argv = ["ppo_actor_daemon", "--state", run_prefix, "--actor-id", actor_id,
            "--ckpt", "ckpts/entity/x.pt", "--planet-run-dir", "ckpts/planet",
            "--fleet-run-dir", "ckpts/fleet", "--comet-run-dir", "ckpts/comet",
            "--num-envs", "4", "--poll-s", "0.05", "--progress-every", "0.0",
            "--workdir", str(workdir), "--once"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        return ppo_actor_daemon.main()
    finally:
        sys.argv = old_argv
        ppo_actor_daemon.subprocess.Popen = saved


def test_multi_actor_gated_flip(monkeypatch=None) -> None:
    """N-actor barrier: with ``expected_actors=["a","b"]`` each actor rolls a
    DISTINCT seed slice into ``rollouts/v0/<id>/`` and the phase only flips
    ``await_rollout -> await_train`` once BOTH ``_DONE`` markers exist. We:

      1. run actor "a" (--once): it rolls its slice + writes its _DONE, then —
         because "b" has no _DONE — must STAY in await_rollout (NO flip);
      2. fabricate "b"'s _DONE, then run actor "a" again (--once): now the barrier
         is satisfied so it flips to await_train with shards = the PARENT
         rollouts/v0/ (the union of all actor subdirs the learner rglobs).

    We also assert the per-actor seed slices are non-overlapping (a: base+0,
    b: base+episodes) and that actor "a"'s live progress landed under
    STATE['actors']['a'] (not the shared 'actor' block, so peers don't clobber)."""
    tmp = Path(tempfile.mkdtemp(prefix="ppo_actor_multi_"))
    run_prefix = str(tmp / "ppo" / "run_multi")
    url = ppo_state.state_url(run_prefix)

    model = ppo_state.model_block(run_prefix, policy_version=0, base_version="basev0abc")
    heads_local = Path(model["heads"])
    heads_local.parent.mkdir(parents=True, exist_ok=True)
    heads_local.write_text("FAKE_HEAD_DELTA")
    config = {"history_window": 10, "episodes": 16, "num_players": "mix",
              "seed_base": 100000, "device": "cpu", "max_fleets": 512, "sigma": 0.35}
    ppo_state.init_state(url, run_id="run_multi", iters_total=3, model=model,
                         config=config, gcs_prefix=run_prefix,
                         expected_actors=["a", "b"])

    # sanity: the roster + accessor round-trip
    st0 = ppo_state.read_state(url)
    assert ppo_state.expected_actors(st0) == ["a", "b"], st0.get("expected_actors")

    workdir = tmp / "work"

    # ---- (1) actor "a" rolls its slice; "b" not done -> NO flip ----
    seen_a: list = []
    rc = _run_actor_once(run_prefix, "a", _make_fake_popen(seen_a), workdir)
    assert rc == 0, f"actor a returned {rc}"

    st = ppo_state.read_state(url)
    # a's _DONE exists, b's does NOT -> phase must remain await_rollout, iter 0
    assert ppo_state.done_marker_exists(f"{run_prefix}/rollouts/v0/a/"), "a _DONE missing"
    assert not ppo_state.done_marker_exists(f"{run_prefix}/rollouts/v0/b/"), "b _DONE leaked"
    assert not ppo_state.all_actors_done(run_prefix, 0, ["a", "b"]), "barrier should be OPEN"
    assert st["phase"] == ppo_state.PHASE_AWAIT_ROLLOUT, \
        f"actor a flipped despite peer b unfinished: {st['phase']}"
    assert st["iter"] == 0, st["iter"]
    assert st["shards"] is None, st["shards"]

    # a's progress went to the PER-ACTOR sub-map, not the shared 'actor' block
    assert "a" in (st.get("actors") or {}), st.get("actors")
    blk_a = st["actors"]["a"]
    assert blk_a.get("progress") == _FAKE_PROGRESS, blk_a.get("progress")
    assert blk_a.get("log_n") == len(_FAKE_LINES), blk_a
    assert any("DONE v0" in ln for ln in blk_a.get("log_tail") or []), blk_a
    assert blk_a.get("status") == "done", blk_a
    assert blk_a.get("heartbeat_ts")
    # the shared single-actor 'actor' block was NOT used (still empty / unclaimed)
    assert (st.get("actor") or {}).get("claimed_iter", -1) == -1, st.get("actor")

    # a's DISTINCT seed slice: index 0 of N=2 -> seed_base + 0*(N*ep) + 0*ep = 100000
    assert len(seen_a) == 1, seen_a
    cmd_a = seen_a[0]
    def _flag(cmd, name):
        return cmd[cmd.index(name) + 1]
    assert _flag(cmd_a, "--vm-id") == "a"
    assert _flag(cmd_a, "--out") == f"{run_prefix}/rollouts/v0/a/", _flag(cmd_a, "--out")
    assert _flag(cmd_a, "--seed-base") == "100000", _flag(cmd_a, "--seed-base")

    # ---- (1b) re-run actor "a" (its _DONE already there): still no flip, no rollout ----
    def _boom(cmd, **kwargs):
        raise AssertionError("rollout_worker should NOT re-run when a's _DONE exists")
    rc = _run_actor_once(run_prefix, "a", _boom, workdir)
    assert rc == 0
    st = ppo_state.read_state(url)
    assert st["phase"] == ppo_state.PHASE_AWAIT_ROLLOUT, \
        f"still must wait for b: {st['phase']}"

    # ---- (2) fabricate b's _DONE -> barrier satisfied -> next loop flips ----
    # (We let actor "b" roll its OWN slice here to also verify its distinct seed.)
    seen_b: list = []
    rc = _run_actor_once(run_prefix, "b", _make_fake_popen(seen_b), workdir)
    assert rc == 0, f"actor b returned {rc}"

    # b's DISTINCT seed slice: index 1 of N=2 -> seed_base + 0 + 1*ep = 100016
    assert len(seen_b) == 1, seen_b
    cmd_b = seen_b[0]
    assert _flag(cmd_b, "--vm-id") == "b"
    assert _flag(cmd_b, "--out") == f"{run_prefix}/rollouts/v0/b/", _flag(cmd_b, "--out")
    assert _flag(cmd_b, "--seed-base") == "100016", _flag(cmd_b, "--seed-base")
    # non-overlapping: a used [100000, +16), b used [100016, +16)
    assert _flag(cmd_a, "--seed-base") != _flag(cmd_b, "--seed-base")

    # now BOTH _DONE exist -> the barrier is closed and b (the last finisher)
    # performed the flip to await_train with shards = the PARENT rollouts/v0/
    assert ppo_state.all_actors_done(run_prefix, 0, ["a", "b"]), "barrier should be CLOSED"
    st = ppo_state.read_state(url)
    assert st["phase"] == ppo_state.PHASE_AWAIT_TRAIN, \
        f"both done -> should flip to await_train: {st['phase']}"
    assert st["iter"] == 0, st["iter"]
    # shards is the PARENT (union of all actors), NOT a single actor subdir, so the
    # learner's _collect_shards rglob picks up BOTH rollouts/v0/a/ and v0/b/.
    assert st["shards"] == f"{run_prefix}/rollouts/v0/", st["shards"]
    assert st["shards"].rstrip("/").endswith("/rollouts/v0"), st["shards"]
    assert "/a/" not in st["shards"] and "/b/" not in st["shards"], st["shards"]
    assert st["updated_by"] == "b", st["updated_by"]

    print("  [ok] multi-actor barrier: a's slice done but NO flip while b pending; "
          "after b's _DONE -> flip await_rollout->await_train shards=parent "
          f"(a seed=100000, b seed=100016, non-overlapping)")


def test_workers_fanout(monkeypatch=None) -> None:
    """``--workers 2`` fan-out (single-actor): the daemon launches TWO parallel
    ``rollout_worker`` PROCESSES, each on its OWN ``rollouts/v0/machineB/w{j}/`` out
    dir + a DISTINCT seed sub-slice, pins OMP/MKL/OPENBLAS to 1 thread per process,
    waits for BOTH, then writes the ACTOR-LEVEL ``machineB/_DONE`` itself (the
    per-worker _DONEs live under w{j}/, but the barrier checks ``<id>/_DONE``).
    With ``--episodes 2 --workers 2`` it's "1 game per core": w0 -> 1 ep @ seed
    base, w1 -> 1 ep @ base+1. We then confirm the phase flipped to await_train."""
    tmp = Path(tempfile.mkdtemp(prefix="ppo_actor_fanout_"))
    run_prefix = str(tmp / "ppo" / "run_fanout")
    url = ppo_state.state_url(run_prefix)

    model = ppo_state.model_block(run_prefix, policy_version=0, base_version="basev0abc")
    heads_local = Path(model["heads"])
    heads_local.parent.mkdir(parents=True, exist_ok=True)
    heads_local.write_text("FAKE_HEAD_DELTA")
    # episodes=2 + workers=2 -> 1 game per core
    config = {"history_window": 10, "episodes": 2, "num_players": "mix",
              "seed_base": 100000, "device": "cpu", "max_fleets": 512, "sigma": 0.35}
    ppo_state.init_state(url, run_id="run_fanout", iters_total=3, model=model,
                         config=config, gcs_prefix=run_prefix)

    transitions: list[tuple] = []
    real_transition = ppo_state.transition

    def _spy_transition(u, *, expect_phase, new_phase, who, **fields):
        transitions.append((expect_phase, new_phase))
        return real_transition(u, expect_phase=expect_phase, new_phase=new_phase,
                               who=who, **fields)

    seen_cmd: list = []
    seen_env: list = []
    fake_popen = _make_fake_worker_popen(seen_cmd, seen_env)

    if monkeypatch is not None:
        monkeypatch.setattr(ppo_actor_daemon.ppo_state, "transition", _spy_transition)
        monkeypatch.setattr(ppo_actor_daemon.subprocess, "Popen", fake_popen)
    else:
        ppo_actor_daemon.ppo_state.transition = _spy_transition
        ppo_actor_daemon.subprocess.Popen = fake_popen

    workdir = tmp / "work"
    argv = ["ppo_actor_daemon", "--state", run_prefix, "--actor-id", "machineB",
            "--ckpt", "ckpts/entity/x.pt", "--planet-run-dir", "ckpts/planet",
            "--fleet-run-dir", "ckpts/fleet", "--comet-run-dir", "ckpts/comet",
            "--num-envs", "1", "--workers", "2", "--poll-s", "0.1",
            "--progress-every", "0.0", "--workdir", str(workdir), "--once"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = ppo_actor_daemon.main()
    finally:
        sys.argv = old_argv
        if monkeypatch is None:
            ppo_actor_daemon.ppo_state.transition = real_transition
            ppo_actor_daemon.subprocess.Popen = _FakeProc  # restore harmless default

    assert rc == 0, f"daemon returned {rc}"

    # ---- exactly TWO worker PROCESSES were launched ----
    assert len(seen_cmd) == 2, f"expected 2 worker launches, got {len(seen_cmd)}"

    def _flag(cmd, name):
        return cmd[cmd.index(name) + 1]

    out0, out1 = _flag(seen_cmd[0], "--out"), _flag(seen_cmd[1], "--out")
    # distinct per-worker out dirs w0/ and w1/ under the actor's rollouts/v0/<id>/
    assert out0 == f"{run_prefix}/rollouts/v0/machineB/w0/", out0
    assert out1 == f"{run_prefix}/rollouts/v0/machineB/w1/", out1
    assert out0 != out1

    # distinct, non-overlapping seed sub-slices: base + j*per_worker_eps (per=1)
    s0, s1 = _flag(seen_cmd[0], "--seed-base"), _flag(seen_cmd[1], "--seed-base")
    assert s0 == "100000", s0          # actor seed_base (K=0, idx=0)
    assert s1 == "100001", s1          # + per_worker_eps (1)
    assert s0 != s1

    # 1 game per core: each worker rolls exactly 1 episode (ceil(2/2)=1)
    assert _flag(seen_cmd[0], "--episodes") == "1", _flag(seen_cmd[0], "--episodes")
    assert _flag(seen_cmd[1], "--episodes") == "1", _flag(seen_cmd[1], "--episodes")
    # both target the SAME policy/base/heads as the actor slice
    assert _flag(seen_cmd[0], "--policy-version") == "0"
    assert _flag(seen_cmd[0], "--base-version") == "basev0abc"
    assert _flag(seen_cmd[0], "--num-envs") == "1"

    # ---- each worker's env pinned to 1 thread (N procs -> N cores) ----
    for env in seen_env:
        assert env.get("OMP_NUM_THREADS") == "1", env.get("OMP_NUM_THREADS")
        assert env.get("MKL_NUM_THREADS") == "1", env.get("MKL_NUM_THREADS")
        assert env.get("OPENBLAS_NUM_THREADS") == "1", env.get("OPENBLAS_NUM_THREADS")

    # ---- both workers wrote their per-worker manifest+shard under w{j}/ ----
    for j, base in enumerate((out0, out1)):
        assert (Path(base) / "manifest.json").exists(), f"w{j} manifest missing"
        assert (Path(base) / "progress.json").exists(), f"w{j} progress missing"
        assert any(Path(base).glob("shard_*.pt")), f"w{j} shard missing"
        # the per-worker _DONE is NOT written by the worker fake (daemon owns it)
        assert not (Path(base) / "_DONE").exists(), f"w{j} should not write _DONE"

    # ---- the DAEMON wrote the ACTOR-LEVEL _DONE at rollouts/v0/<id>/_DONE ----
    actor_done = f"{run_prefix}/rollouts/v0/machineB/"
    assert ppo_state.done_marker_exists(actor_done), \
        "actor-level _DONE missing — barrier/all_actors_done would never close"
    assert (Path(actor_done) / "_DONE").exists()

    # ---- phase ping-pong: single-actor flips await_rollout->rolling_out->await_train ----
    assert transitions == [
        (ppo_state.PHASE_AWAIT_ROLLOUT, ppo_state.PHASE_ROLLING_OUT),
        (ppo_state.PHASE_ROLLING_OUT, ppo_state.PHASE_AWAIT_TRAIN),
    ], transitions
    final = ppo_state.read_state(url)
    assert final["phase"] == ppo_state.PHASE_AWAIT_TRAIN, final["phase"]
    assert final["iter"] == 0
    # single-actor flips with shards = its OWN dir (the learner rglobs w{j}/manifest)
    assert final["shards"] == f"{run_prefix}/rollouts/v0/machineB/", final["shards"]

    # ---- actor-level progress rollup landed under STATE['actors']['machineB'] ----
    blk = (final.get("actors") or {}).get("machineB") or {}
    assert blk.get("status") == "done", blk
    prog = blk.get("progress") or {}
    assert prog.get("ep_done") == 2, prog          # 1 + 1 summed across workers
    assert prog.get("ep_total") == 2, prog
    assert prog.get("n_workers") == 2, prog

    print("  [ok] --workers 2 fan-out: 2 parallel worker procs (w0/, w1/), "
          "distinct seeds (100000, 100001), 1 ep each, OMP=1 per proc; daemon "
          "wrote actor-level _DONE; flipped to await_train (rollup ep_done=2)")


def test_pool_procs_passthrough(monkeypatch=None) -> None:
    """``--pool-procs N`` (single-actor): the daemon launches ONE rollout_worker
    (the COW fork pool, one process) with ``--pool-procs N`` and ``--episodes`` =
    the actor's FULL game count — NO ``--workers`` fan-out (even though we pass
    ``--workers 4``, the pool overrides it: a single process, strictly less RAM).
    The single worker writes its own ``progress.json`` + ``_DONE`` (mocked by
    ``_FakeProc``), so the phase ping-pongs await_rollout->rolling_out->await_train
    just like the plain single-worker path."""
    tmp = Path(tempfile.mkdtemp(prefix="ppo_actor_pool_"))
    run_prefix = str(tmp / "ppo" / "run_pool")
    url = ppo_state.state_url(run_prefix)

    model = ppo_state.model_block(run_prefix, policy_version=0, base_version="basev0abc")
    heads_local = Path(model["heads"])
    heads_local.parent.mkdir(parents=True, exist_ok=True)
    heads_local.write_text("FAKE_HEAD_DELTA")
    config = {"history_window": 10, "episodes": 16, "num_players": "mix",
              "seed_base": 100000, "device": "cpu", "max_fleets": 512, "sigma": 0.35}
    ppo_state.init_state(url, run_id="run_pool", iters_total=3, model=model,
                         config=config, gcs_prefix=run_prefix)

    transitions: list[tuple] = []
    real_transition = ppo_state.transition

    def _spy_transition(u, *, expect_phase, new_phase, who, **fields):
        transitions.append((expect_phase, new_phase))
        return real_transition(u, expect_phase=expect_phase, new_phase=new_phase,
                               who=who, **fields)

    seen_cmd: list = []
    fake_popen = _make_fake_popen(seen_cmd)   # writes progress.json + _DONE in --out

    if monkeypatch is not None:
        monkeypatch.setattr(ppo_actor_daemon.ppo_state, "transition", _spy_transition)
        monkeypatch.setattr(ppo_actor_daemon.subprocess, "Popen", fake_popen)
    else:
        ppo_actor_daemon.ppo_state.transition = _spy_transition
        ppo_actor_daemon.subprocess.Popen = fake_popen

    workdir = tmp / "work"
    # pass BOTH --workers 4 AND --pool-procs 3: the pool must win (one process).
    argv = ["ppo_actor_daemon", "--state", run_prefix, "--actor-id", "machineB",
            "--ckpt", "ckpts/entity/x.pt", "--planet-run-dir", "ckpts/planet",
            "--fleet-run-dir", "ckpts/fleet", "--comet-run-dir", "ckpts/comet",
            "--num-envs", "1", "--workers", "4", "--pool-procs", "3",
            "--poll-s", "0.1", "--progress-every", "0.0",
            "--workdir", str(workdir), "--once"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = ppo_actor_daemon.main()
    finally:
        sys.argv = old_argv
        if monkeypatch is None:
            ppo_actor_daemon.ppo_state.transition = real_transition

    assert rc == 0, f"daemon returned {rc}"

    # EXACTLY ONE worker process — the pool, not a fan-out (no wN/ subdirs).
    assert len(seen_cmd) == 1, f"expected ONE pool worker, got {len(seen_cmd)}"
    cmd = seen_cmd[0]

    def _flag(name):
        return cmd[cmd.index(name) + 1]

    assert "--pool-procs" in cmd, "pool path must pass --pool-procs"
    assert _flag("--pool-procs") == "3", _flag("--pool-procs")
    # episodes = the actor's FULL game count (config.episodes), NOT a per-worker split
    assert _flag("--episodes") == "16", _flag("--episodes")
    # out is the actor dir itself (no w{j}/ fan-out subdir)
    assert _flag("--out") == f"{run_prefix}/rollouts/v0/machineB/", _flag("--out")
    assert "/w0/" not in _flag("--out") and "/w1/" not in _flag("--out")
    assert _flag("--policy-version") == "0"
    assert _flag("--base-version") == "basev0abc"
    assert _flag("--seed-base") == "100000"        # K=0, idx=0 -> base
    assert _flag("--num-envs") == "1"

    # phase ping-pong identical to the plain single-worker path
    assert transitions == [
        (ppo_state.PHASE_AWAIT_ROLLOUT, ppo_state.PHASE_ROLLING_OUT),
        (ppo_state.PHASE_ROLLING_OUT, ppo_state.PHASE_AWAIT_TRAIN),
    ], transitions
    final = ppo_state.read_state(url)
    assert final["phase"] == ppo_state.PHASE_AWAIT_TRAIN, final["phase"]
    assert final["iter"] == 0
    assert final["shards"] == f"{run_prefix}/rollouts/v0/machineB/", final["shards"]
    # the worker's progress.json was mirrored into the shared actor block, INCLUDING
    # the enriched per-core ``workers`` array (the whole point of the pool path).
    assert final["actor"]["progress"] == _FAKE_PROGRESS, final["actor"]["progress"]
    pool_prog = final["actor"]["progress"]
    assert isinstance(pool_prog.get("workers"), list) and pool_prog["workers"], pool_prog
    assert pool_prog["workers"][0]["seed"] == 100002, pool_prog["workers"][0]
    assert pool_prog.get("total_steps") == 400, pool_prog

    print("  [ok] --pool-procs 3 (overriding --workers 4): ONE pool worker, "
          "--episodes=16 (full actor slice), no w{j}/ fan-out; "
          "await_rollout->rolling_out->await_train, progress (+per-core workers[]) "
          "mirrored")


if __name__ == "__main__":
    test_one_iteration()
    test_idempotent_skip()
    test_multi_actor_gated_flip()
    test_workers_fanout()
    test_pool_procs_passthrough()
    print("ACTOR DAEMON PASSED — single-actor rollout + idempotent skip + "
          "multi-actor gated flip + --workers fan-out + --pool-procs passthrough")
