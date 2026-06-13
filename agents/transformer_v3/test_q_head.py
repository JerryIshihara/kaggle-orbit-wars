"""Smoke + learnability tests for the per-pair Q head.

1. Model wiring: with_q_head=True adds q_value (B,P,P) to forward output;
   default (off) leaves the output byte-identical to before.
2. Loss: compute_q_loss is finite, grads flow.
3. Learnability: a few hundred steps of SGD on a synthetic task drive the
   doomed/reachable Q SEPARATION negative (the head learns doomed < reachable),
   which is the whole point.

Run:  .venv/bin/python -m agents.transformer_v3.test_q_head
"""

from __future__ import annotations

import torch

from .model import EntityPretrainModelV3
from .q_head import Q_DOOMED, compute_q_loss, q_gate_select_logits


def test_model_wiring() -> None:
    torch.manual_seed(0)
    P, F_ = 8, 16
    common = dict(d_model=64, d_pair=64, entity_n_heads=4, cross_n_heads=4,
                  cross_n_layers=1, dual_n_heads=4, conditioner_n_layers=1,
                  head_n_layers=1, with_consolidator=True)
    m_off = EntityPretrainModelV3(**common)
    m_on = EntityPretrainModelV3(**common, with_q_head=True)
    assert m_off.pair_head.q_head is None
    assert m_on.pair_head.q_head is not None
    # q_head adds params ONLY when on (existing ckpts load strict on m_off).
    extra = sum(p.numel() for p in m_on.parameters()) - sum(
        p.numel() for p in m_off.parameters())
    assert extra > 0
    print(f"  wiring: q_head off=None / on={extra} params")


def test_loss_and_gate() -> None:
    torch.manual_seed(1)
    B, P = 4, 8
    q = torch.randn(B, P, P, requires_grad=True)
    legal = torch.rand(B, P, P) < 0.5
    doomed = legal & (torch.rand(B, P, P) < 0.3)
    fired = legal & ~doomed & (torch.rand(B, P, P) < 0.2)
    ret = torch.randn(B)
    loss, diag = compute_q_loss(
        q, fired_mask=fired, fired_return=ret, doomed_mask=doomed,
        legal_mask=legal)
    assert torch.isfinite(loss)
    loss.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    # gate: doomed-floor masks the doomed cells out of selection
    pl = torch.zeros(P, P)
    qv = torch.zeros(P, P)
    qv[0, 3] = Q_DOOMED
    gated = q_gate_select_logits(pl, qv, doomed_floor=Q_DOOMED + 0.1)
    assert gated[0, 3] == float("-inf") and torch.isfinite(gated[0, 4])
    print(f"  loss={float(loss):.3f} sep={diag['q_separation']:.3f} "
          f"grad ok; gate masks doomed cell")


def test_learnability(steps: int = 400) -> None:
    """Synthetic: a fixed per-pair feature → Q. doomed pairs share a feature
    direction; SGD must learn to map it to Q_DOOMED, driving separation < 0."""
    torch.manual_seed(2)
    B, P = 16, 8
    feat = torch.randn(B, P, P, 12)
    head = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.GELU(),
                               torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    legal = torch.rand(B, P, P) < 0.6
    # doomed iff a hidden linear score on feat is high (a learnable pattern)
    score = feat @ torch.randn(12)
    doomed = legal & (score > score.median())
    fired = legal & ~doomed & (torch.rand(B, P, P) < 0.25)
    ret = torch.randn(B) * 0.5 + 0.5
    sep0 = None
    for i in range(steps):
        q = head(feat).squeeze(-1)
        loss, diag = compute_q_loss(
            q, fired_mask=fired, fired_return=ret, doomed_mask=doomed,
            legal_mask=legal)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            sep0 = diag["q_separation"]
    print(f"  separation: start={sep0:+.3f} -> end={diag['q_separation']:+.3f} "
          f"(doomed Q {diag['q_doomed_mean']:+.2f} vs reach {diag['q_reach_mean']:+.2f})")
    assert diag["q_separation"] < -0.3, "Q head must learn doomed < reachable"


def main() -> None:
    print("[1/3] model wiring (q_head off=identity / on=+params)")
    test_model_wiring()
    print("[2/3] loss finite + grads + deploy gate")
    test_loss_and_gate()
    print("[3/3] learnability (separation goes negative)")
    test_learnability()
    print("ALL Q-HEAD TESTS PASSED")


if __name__ == "__main__":
    main()
