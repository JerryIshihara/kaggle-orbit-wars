"""Unit test for the PPO learner listen-loop (``ppo_learner_daemon``).

Drives ONE ``--once`` iteration of the loop against a LOCAL ``STATE.json`` (no GCS,
no GPU) with ``learner_step`` mocked out: ``subprocess.Popen`` is monkeypatched with
a fake that (a) prints a couple of the exact per-epoch lines ``learner_step`` emits
and (b) writes a fake ``out_ckpt`` file (the artifact-before-STATE handoff). The test
asserts the loop did the ``await_train -> training -> await_rollout`` ping-pong
(iter bumped 0->1, ``model.policy_version==1``) and mirrored the learner's parsed
``kl`` into ``STATE['learner'].progress`` + a non-empty log tail.

    python -m agents.transformer_v2.ppo.test_learner_daemon
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from agents.transformer_v2.ppo import ppo_learner_daemon as dmn
from agents.transformer_v2.ppo import ppo_state


# Two real learner_step per-epoch lines (matches the format string in
# learner_step.main: kl=%.4f policy_loss=%.4f value_loss=%.4f entropy=%.3f clip_frac=%.3f).
_FAKE_STDOUT = [
    "[learner] === iter 0 → v1 (device=cpu) ===",
    "[learner]   epoch 1: kl=0.009 policy_loss=-0.02 value_loss=0.04 entropy=2.1 clip_frac=0.08",
    "[learner]   epoch 2: kl=0.014 policy_loss=-0.05 value_loss=0.03 entropy=2.0 clip_frac=0.11",
    "[learner] DONE iter 0: winrate=0.5 | kl=0.014 ...",
]


class _FakePopen:
    """Stand-in for subprocess.Popen(learner_step): emits canned stdout lines and,
    as a side effect, writes the ``--out-ckpt`` path so the daemon's post-run
    idempotency view sees the artifact (the artifact-before-STATE handoff)."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.returncode = None
        # Honor the real call shape so a mismatch fails loud here, not silently.
        assert kwargs.get("stdout") is not None and kwargs.get("text") is True, kwargs
        # Parse --out-ckpt (and --out-heads) out of the built command and create them.
        out_ckpt = self._arg(cmd, "--out-ckpt")
        out_heads = self._arg(cmd, "--out-heads")
        for p in (out_ckpt, out_heads):
            if p:
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                Path(p).write_text("fake-ckpt")
        self.stdout = iter([ln + "\n" for ln in _FAKE_STDOUT])

    @staticmethod
    def _arg(cmd, flag):
        return cmd[cmd.index(flag) + 1] if flag in cmd else None

    def wait(self):
        self.returncode = 0
        return 0


def _args(*, state_url: str, run_prefix: str) -> SimpleNamespace:
    """A parsed-args namespace with the daemon defaults (only what loop() reads)."""
    return SimpleNamespace(
        state=run_prefix, learner_id="colab-test", ckpt="ckpts/entity/best.pt",
        planet_run_dir="ckpts/planet", fleet_run_dir="ckpts/fleet", comet_run_dir="ckpts/comet",
        device="cpu", minibatch_size=256, epochs=3, clip=0.10, target_kl=0.01, lr_heads=1e-4,
        lr_trunk=None,
        value_coef=0.5, ent_coef=0.01, sigma=0.35, pair_cache_path=None, bc_coef=0.0,
        freeze_phase=0, rollout_timeout=0.0,
        poll_s=0.01, base_version=None, once=True,
    )


def test_once_await_train_to_await_rollout(monkeypatch) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ppo_learner_daemon_"))
    run_prefix = str(tmp / "run")
    url = ppo_state.state_url(run_prefix)

    # --- seed STATE at await_rollout (init) then move it to await_train iter 0 ---
    # gcs_prefix is the run root for BIG files; the daemon reads big-file (ckpt/
    # heads/train_log/shards) paths from it, NOT from the state url's parent. Here
    # it == run_prefix (local), so the on-disk artifact assertions still hold.
    model0 = ppo_state.model_block(run_prefix, 0, "basev0abc123")
    ppo_state.init_state(url, run_id="test_run", iters_total=2, model=model0,
                         config={"history_window": 10}, gcs_prefix=run_prefix, message="")
    shards_url = f"{run_prefix}/rollouts/v0/"
    ppo_state.transition(url, expect_phase=ppo_state.PHASE_AWAIT_ROLLOUT,
                         new_phase=ppo_state.PHASE_AWAIT_TRAIN, who="machineB",
                         shards=shards_url, iter=0)

    # --- mock learner_step ---
    monkeypatch.setattr(dmn.subprocess, "Popen", _FakePopen)

    # --- run ONE --once iteration of the loop ---
    # loop() takes big-file paths from STATE['gcs_prefix'] (per-iter), so it no
    # longer needs a run_prefix arg — the control-channel url may even be fs://.
    args = _args(state_url=url, run_prefix=run_prefix)
    rc = dmn.loop(args, url, args.learner_id)
    assert rc == 0, f"loop returned {rc}, expected 0"

    # --- assert the ping-pong happened: await_train -> training -> await_rollout ---
    st = ppo_state.read_state(url)
    assert st is not None, "STATE vanished"
    assert st["phase"] == ppo_state.PHASE_AWAIT_ROLLOUT, st["phase"]
    assert st["iter"] == 1, st["iter"]
    assert st["model"]["policy_version"] == 1, st["model"]
    assert st["model"]["base_version"] == "basev0abc123", st["model"]
    # big-file (ckpt/heads) paths are built under STATE['gcs_prefix'] (== run_prefix
    # here), NOT derived from the state url's parent dir.
    gcs_prefix = ppo_state.gcs_prefix(st)
    assert gcs_prefix == run_prefix, gcs_prefix
    assert st["model"]["full"] == f"{gcs_prefix}/checkpoints/policy_v1.pt", st["model"]["full"]
    assert st["model"]["heads"] == f"{gcs_prefix}/heads/policy_v1.heads.pt", st["model"]["heads"]
    assert st["shards"] is None, "shards should be cleared on the flip to await_rollout"
    assert st["updated_by"] == "colab-test", st["updated_by"]

    # the real artifact was written by the (mocked) learner_step before the flip
    assert Path(f"{run_prefix}/checkpoints/policy_v1.pt").exists(), "out_ckpt not written"
    assert Path(f"{run_prefix}/heads/policy_v1.heads.pt").exists(), "out_heads not written"

    # --- assert the learner stream was mirrored into STATE['learner'] ---
    lrn = st["learner"]
    assert lrn["id"] == "colab-test", lrn          # claim_side stamped the lease
    assert lrn["claimed_iter"] == 0, lrn
    prog = lrn["progress"]
    assert prog is not None, "learner progress not mirrored"
    # last parsed epoch was epoch 2 with kl=0.014
    assert prog.get("epoch") == 2, prog
    assert abs(prog.get("kl", -1) - 0.014) < 1e-9, prog
    assert abs(prog.get("policy_loss", 0) - (-0.05)) < 1e-9, prog
    assert lrn.get("log_n", 0) >= len(_FAKE_STDOUT), lrn   # all stdout lines mirrored
    assert lrn.get("log_tail"), "log_tail empty"
    assert any("epoch 1" in ln for ln in lrn["log_tail"]), lrn["log_tail"]

    print("  [ok] await_train -> training -> await_rollout (iter 0->1, model v1); "
          f"learner.progress kl={prog['kl']} mirrored, log_n={lrn['log_n']}")


def test_build_learner_cmd_shape() -> None:
    """The built learner_step command carries the run-prefix paths + HP flags, and
    omits --pair-cache-path when no cache is configured (BC off)."""
    args = _args(state_url="x", run_prefix="/tmp/run")
    cmd = dmn._build_learner_cmd(
        args, run_prefix="gs://b/ppo/r", K=3, base_version="basev0abc123",
        shards_url="gs://b/ppo/r/rollouts/v3/",
        out_ckpt="gs://b/ppo/r/checkpoints/policy_v4.pt",
        out_heads="gs://b/ppo/r/heads/policy_v4.heads.pt")
    assert cmd[:4] == [dmn.sys.executable, "-u", "-m",
                       "agents.transformer_v2.ppo.learner_step"], cmd[:4]

    def val(flag):
        return cmd[cmd.index(flag) + 1]

    assert val("--policy-ckpt") == "gs://b/ppo/r/checkpoints/policy_v3.pt"
    assert val("--shards") == "gs://b/ppo/r/rollouts/v3/"
    assert val("--out-ckpt") == "gs://b/ppo/r/checkpoints/policy_v4.pt"
    assert val("--out-heads") == "gs://b/ppo/r/heads/policy_v4.heads.pt"
    assert val("--train-log") == "gs://b/ppo/r/train_log.jsonl"
    assert val("--policy-version") == "3"
    assert val("--base-version") == "basev0abc123"
    assert val("--device") == "cpu"
    assert val("--ckpt") == "ckpts/entity/best.pt"
    assert val("--minibatch-size") == "256" and val("--epochs") == "3"
    assert val("--clip") == "0.1" and val("--target-kl") == "0.01"
    assert val("--lr-heads") == "0.0001" and val("--value-coef") == "0.5"
    assert val("--ent-coef") == "0.01" and val("--sigma") == "0.35"
    assert val("--freeze-phase") == "0", "freeze-phase passthrough (default 0)"
    assert val("--fleet-run-dir") == "ckpts/fleet"
    assert val("--planet-run-dir") == "ckpts/planet"
    assert val("--comet-run-dir") == "ckpts/comet"
    assert "--pair-cache-path" not in cmd, "BC should be off without a cache"
    assert "--bc-coef" not in cmd

    # with a pair cache -> BC flags appended
    args.pair_cache_path = "/tmp/pair_cache.pt"
    args.bc_coef = 0.05
    cmd2 = dmn._build_learner_cmd(
        args, run_prefix="gs://b/ppo/r", K=3, base_version="b",
        shards_url="s", out_ckpt="oc", out_heads="oh")
    assert cmd2[cmd2.index("--pair-cache-path") + 1] == "/tmp/pair_cache.pt"
    assert cmd2[cmd2.index("--bc-coef") + 1] == "0.05"

    # --freeze-phase is forwarded with whatever value is set (not just the default)
    args.freeze_phase = 2
    cmd3 = dmn._build_learner_cmd(
        args, run_prefix="gs://b/ppo/r", K=3, base_version="b",
        shards_url="s", out_ckpt="oc", out_heads="oh")
    assert cmd3[cmd3.index("--freeze-phase") + 1] == "2", "freeze-phase passthrough"
    print("  [ok] _build_learner_cmd: run-prefix paths + HP flags; pair-cache gated on BC; "
          "freeze-phase forwarded")


# --- tiny pytest-free monkeypatch shim so this runs as a plain script too --- #
class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        self._undo = []


if __name__ == "__main__":
    mp = _MonkeyPatch()
    try:
        test_once_await_train_to_await_rollout(mp)
    finally:
        mp.undo()
    test_build_learner_cmd_shape()
    print("LEARNER DAEMON TEST PASSED — await_train→training→await_rollout + progress mirror")
