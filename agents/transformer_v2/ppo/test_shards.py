"""Step 0 of the distributed-PPO plan: shard serialization round-trip + trainable
delta + version checks. Pure-tensor + synthetic episodes — NO Linux env, NO GCS —
so it runs on macOS.

    python -m agents.transformer_v2.ppo.test_shards
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from agents.transformer_v2.ppo import shards
from agents.transformer_v2.ppo.sampler import MultiTargetAction
from agents.transformer_v2.ppo.smoke import EpisodeBuffer, StepRecord, episodes_to_ppo


def _make_action(P: int) -> MultiTargetAction:
    select_mask = (torch.rand(P, P) > 0.6)
    eye = torch.eye(P, dtype=torch.bool)
    select_mask = select_mask & ~eye                      # fired targets are off-diagonal
    alloc_counts = (torch.randint(0, 5, (P, P)) * select_mask.long())
    self_counts = torch.randint(0, 5, (P,))
    n_fired = int(select_mask.sum().item())
    return MultiTargetAction(
        select_mask=select_mask,
        alloc_counts=alloc_counts,
        self_counts=self_counts,
        logprob=torch.randn(()),
        logprob_select=torch.randn(()),
        logprob_alloc=torch.randn(()),
        n_terms=n_fired + 2,
        diagnostics={"n_fired_total": float(n_fired), "n_launch_sources": 2.0},
    )


def _make_step(P: int, F: int, dp: int, df: int, history: int) -> StepRecord:
    # T=1 -> model inputs are (P,...)/(F,...); T>1 -> (history,P,...)/(history,F,...).
    def shp(*rest):
        return tuple(rest) if history == 1 else (history, *rest)

    return StepRecord(
        planet_features=torch.randn(*shp(P, dp)),
        fleet_features=torch.randn(*shp(F, df)),
        fleet_target_idx=torch.randint(0, P, shp(F)),
        fleet_source_idx=torch.randint(0, P, shp(F)),
        fleet_owner_slot=torch.randint(0, 4, shp(F)),
        fleet_ships_log=torch.randn(*shp(F)),
        fleet_eta_norm=torch.randn(*shp(F)),
        fleet_mask=(torch.rand(*shp(F)) > 0.5),
        planet_mask=(torch.rand(*shp(P)) > 0.2),
        is_comet=(torch.rand(*shp(P)) > 0.7),
        pair_type_ids=torch.randint(0, 3, shp(P, P)),
        pair_mask=(torch.rand(P, P) > 0.5),
        source_mask=(torch.rand(P) > 0.3),
        action=_make_action(P),
        value=float(torch.randn(()).item()),
        reward=float(torch.randn(()).item()),
        done=0.0,
        invalid_launch=int(torch.randint(0, 3, (1,)).item()),
        emitted_launch=int(torch.randint(0, 2, (1,)).item()),
        n_selected_targets=int(torch.randint(0, 5, (1,)).item()),
        invalid_reasons=["surplus", "min_launch"],
        score_my=int(torch.randint(0, 100, (1,)).item()),
        score_enemy_max=int(torch.randint(0, 100, (1,)).item()),
    )


def _make_episode(seed: int, *, P, F, dp, df, history, n_steps) -> EpisodeBuffer:
    steps = [_make_step(P, F, dp, df, history) for _ in range(n_steps)]
    steps[-1].done = 1.0
    return EpisodeBuffer(seed=seed, learner_seat=seed % 2, steps=steps, winner=seed % 2)


def _assert_step_equal(a: StepRecord, b: StepRecord) -> None:
    for f in shards.STEP_TENSOR_FIELDS:
        assert torch.equal(getattr(a, f), getattr(b, f)), f"tensor field {f} differs"
    for f in shards.STEP_SCALAR_FIELDS:
        assert getattr(a, f) == getattr(b, f), f"scalar field {f} differs"
    assert a.invalid_reasons == b.invalid_reasons
    assert int(a.action.n_terms) == int(b.action.n_terms)
    assert torch.equal(a.action.select_mask, b.action.select_mask)
    assert torch.equal(a.action.alloc_counts, b.action.alloc_counts)
    assert torch.equal(a.action.self_counts, b.action.self_counts)
    for f in shards.ACTION_LOGPROB_FIELDS:
        assert float(getattr(a.action, f)) == float(getattr(b.action, f)), f
    assert a.action.diagnostics == b.action.diagnostics


def test_roundtrip_and_delta() -> None:
    torch.manual_seed(0)
    P, F, dp, df = 6, 8, 4, 4
    tmp = Path(tempfile.mkdtemp())

    # ---- 1. round-trip both T=1 and T=10 episodes, bit-identical ----
    for hist in (1, 10):
        ep = _make_episode(100 + hist, P=P, F=F, dp=dp, df=df, history=hist, n_steps=5)
        path = tmp / f"shard_h{hist}.pt"
        entry = shards.save_shard(ep, path, policy_version=7, history_window=hist,
                                  vm_id="test-vm")
        assert entry["n_steps"] == 5 and entry["seed"] == 100 + hist
        loaded = shards.load_shard(path, expect_policy_version=7,
                                   expect_contract=shards.POLICY_ACTION_CONTRACT,
                                   expect_history_window=hist)
        assert len(loaded.steps) == len(ep.steps)
        assert loaded.seed == ep.seed and loaded.winner == ep.winner
        for a, b in zip(ep.steps, loaded.steps):
            _assert_step_equal(a, b)
        # episodes_to_ppo consumes the loaded buffer (GAE + minibatch build)
        ep_objs, mbs = episodes_to_ppo([loaded], minibatch_size=3, device="cpu")
        assert len(mbs) >= 1 and len(ep_objs) == 1
        pf = mbs[0].feats["planet_features"]
        exp = (pf.shape[0], P, dp) if hist == 1 else (pf.shape[0], hist, P, dp)
        assert tuple(pf.shape) == exp, (pf.shape, exp)
        adv = torch.cat([m.adv for m in mbs])
        assert torch.isfinite(adv).all(), "advantages must be finite"
        print(f"  [ok] history={hist}: round-trip identical; "
              f"{len(mbs)} mb(s); planet_features {tuple(pf.shape)}; adv finite")

    # ---- 2. version check hard-fails on skew ----
    path = tmp / "shard_h1.pt"
    for kw in (dict(expect_policy_version=999),
               dict(expect_contract="wrong_v9"),
               dict(expect_history_window=99)):
        try:
            shards.load_shard(path, **kw)
            raise AssertionError(f"expected ShardVersionError for {kw}")
        except shards.ShardVersionError:
            pass
    print("  [ok] load_shard raises ShardVersionError on policy/contract/history skew")

    # ---- 3. trainable-only delta: only requires_grad keys, applies onto a base ----
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = nn.Linear(4, 4)   # backbone (frozen)
            self.head = nn.Linear(4, 2)      # trainable head

    m = Tiny()
    for p in m.frozen.parameters():
        p.requires_grad_(False)
    dpath = tmp / "policy_v1.heads.pt"
    info = shards.save_trainable_delta(m, dpath, base_version="base-abc123")
    payload = torch.load(dpath, map_location="cpu", weights_only=False)
    assert set(payload["keys"]) == {"head.weight", "head.bias"}, payload["keys"]
    assert info["n_keys"] == 2

    m2 = Tiny()
    with torch.no_grad():
        m2.head.weight.zero_()
        m2.head.bias.zero_()
    res = shards.load_trainable_delta(m2, dpath, expect_base_version="base-abc123")
    assert torch.equal(m2.head.weight, m.head.weight), "delta did not restore head"
    assert torch.equal(m2.head.bias, m.head.bias)
    assert res["missing_frozen"] >= 2, "frozen backbone keys should be 'missing' from the delta"
    try:
        shards.load_trainable_delta(m2, dpath, expect_base_version="WRONG")
        raise AssertionError("expected ShardVersionError on base mismatch")
    except shards.ShardVersionError:
        pass
    print(f"  [ok] trainable delta: {info['n_keys']} keys / {info['n_params']} params "
          f"({info['bytes']} B); restores head; base mismatch raises")


def test_rollout_history() -> None:
    """The T=10 history buffer: current-frame-last ordering + zero-pad early turns."""
    from agents.transformer_v2.history import HISTORY_OFFSETS
    from agents.transformer_v2.ppo.smoke import _RolloutHistory, _TEMPORAL_KEYS

    P = 6
    def frame(step):
        return {k: torch.full((P, 3), float(step)) for k in _TEMPORAL_KEYS}

    h = _RolloutHistory(window=10)
    for s in range(51):
        h.push(s, frame(s))
    pf = h.stack(50)["planet_features"]
    assert tuple(pf.shape) == (10, P, 3), pf.shape
    vals = [float(pf[i, 0, 0]) for i in range(10)]
    assert vals == [50 - off for off in HISTORY_OFFSETS], vals     # offsets 45..0 -> 5..50
    assert vals[-1] == 50.0                                         # current LAST

    h2 = _RolloutHistory(10)
    for s in range(11):
        h2.push(s, frame(s))
    ev = [float(h2.stack(10)["planet_features"][i, 0, 0]) for i in range(10)]
    assert ev[-1] == 10.0 and ev[:7] == [0.0] * 7, ev              # missing early -> zeros
    print("  [ok] _RolloutHistory: (10,...) stack, current-last, zero-pad early turns")


if __name__ == "__main__":
    test_roundtrip_and_delta()
    test_rollout_history()
    print("STEP 0 PASSED — shard round-trip + trainable delta + version + T=10 history")
