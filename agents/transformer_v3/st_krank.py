"""Single-target qrank deploy agent: cached-forward generation + batched
value scoring.

Per step:
  1. Forward the actor ONCE; decode DET (argmax) + N*M SAMPLED candidate
     action-sets off the SAME outputs (``act_candidates_st`` — N forwards -> 1,
     the cached-generation win). DET is always candidate 0.
  2. Apply each candidate -> successor state (deterministic env rules).
  3. Batch-score ALL successors with the value (win) head in ONE forward
     (``batched_value_forward`` — the batched-scoring win; ~free on GPU).
  4. Emit the highest win-prob candidate.

Vs the v4 ``KRankAgent`` this (a) uses the single-target decode (OW_ST_DECODE)
not the v4 Dirichlet, (b) caches the actor forward across candidates, and (c)
scores in a single batched pass instead of a per-candidate loop. N/M from
``OW_V4_N`` / ``OW_V4_M`` (else the constructor's).
"""
from __future__ import annotations

import os

import torch

from ..transformer_v2.runner import TransformerAgent
from .krank import KRankAgent  # _apply: deterministic successor state


class SingleTargetKRankAgent:
    def __init__(self, agent: TransformerAgent, n: int = 5, m: int = 5):
        if agent.model.value_heads is None:
            raise ValueError("qrank needs a value-headed ckpt (jst_best.pt)")
        self.agent = agent
        self.n, self.m = int(n), int(m)

    @classmethod
    def load(cls, ckpt_path: str, n: int = 5, m: int = 5,
             device: str = "cpu", **kw):
        return cls(TransformerAgent.load(ckpt_path=ckpt_path, device=device, **kw),
                   n=n, m=m)

    @torch.no_grad()
    def act(self, obs) -> list[list]:
        # Q-RANK (deploy default): one forward -> q_value; rank n*m sampled
        # select-sets by SUM-Q over launches (det = floor); size winner only.
        # No per-candidate forward/sizing -> fits the 1s CPU budget to ~1e6
        # draws. Q is per-pair, so this ranks target selection (sizing-blind).
        return self.agent.qrank_act(obs, int(os.environ.get("OW_V4_N", self.n))
                                    * int(os.environ.get("OW_V4_M", self.m)))

    @torch.no_grad()
    def act_value_sim(self, obs) -> list[list]:
        """OFFLINE-ONLY alt: simulate-then-score with the value head V(s'). Sees
        the resulting ships (Q-rank can't) but costs a forward PER candidate ->
        OOM/over-budget past ~5x5; never ships (CPU 1s deploy)."""
        n = int(os.environ.get("OW_V4_N", self.n))
        m = int(os.environ.get("OW_V4_M", self.m))
        cands = self.agent.act_candidates_st(obs, n * m)     # cand 0 = DET
        if len(cands) == 1:
            return cands[0]
        preds = self.agent.batched_value_forward(
            [KRankAgent._apply(obs, mv) for mv in cands])
        return cands[int(torch.argmax(torch.sigmoid(preds["win"][:, 0])).item())]
