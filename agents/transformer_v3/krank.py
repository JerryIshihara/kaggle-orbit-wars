"""K-sample critic-ranked deploy agent (simulate-then-score).

Wraps a value-headed v4 ``TransformerAgent``: per step, draw K candidate
action sets from the trained sampling decode, apply each to the current
obs with the env's deterministic action rules (debit garrisons, spawn
fleets at rim+0.1 along the emitted angle — no tick advance: the value
heads forecast FROM the post-action state), batch-score the successors
with the win + fwd heads, and emit the argmax set.

    from agents.transformer_v3.krank import KRankAgent
    ag = KRankAgent.load(ckpt_path="jointv4_best.pt", k=4)
    moves = ag.act(obs)

Mechanics and caveats:
  * candidate 0 is ALWAYS the deterministic mean-alloc set (expmatch
    select) — argmax can fall back to the deterministic policy;
    candidates 1..K-1 are sampled (OW_V3_DECODE=sample semantics).
  * scoring = w_win·σ(win[0]) + Σ_s Σ_h w_h·w_s·fwd[0, s, h] over
    horizons h ∈ {5, 10, 20} (defaults below; env-overridable).
  * the opponent's reply is unknown but IDENTICAL across candidates, so
    it cancels for ranking.
  * DEPLOY/GATES ONLY: sample-K-argmax is a greedier policy than the
    log-probs PPO records — never use it to generate training rollouts.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import torch

from ..transformer_v2.runner import TransformerAgent
from .value_v4 import RANKER_SIGNAL_W

#: per-horizon weights over fwd forecasts (near > far), keyed by horizon.
RANKER_HORIZON_W = {5: 0.5, 10: 0.3, 20: 0.2}
RANKER_WIN_W = 1.0


class KRankAgent:
    def __init__(self, agent: TransformerAgent, k: int = 4):
        assert agent.model.value_heads is not None, (
            "K-rank needs a value-headed ckpt (jointv4_best.pt) — the "
            "actor-only stage-B ckpt has no win/fwd heads")
        self.agent = agent
        self.k = int(os.environ.get("OW_V4_K", k))
        from ..transformer_v2.pretrain.value_signals import P1_FWD_HORIZONS
        self._h_idx = [(P1_FWD_HORIZONS.index(h), w)
                       for h, w in RANKER_HORIZON_W.items()]
        self._sig_w = torch.tensor(RANKER_SIGNAL_W)

    @classmethod
    def load(cls, ckpt_path: str, k: int = 4, device: str = "cpu", **kw):
        return cls(TransformerAgent.load(
            ckpt_path=ckpt_path, device=device, **kw), k=k)

    # ---- candidate generation: N select-samples × M alloc-draws -----
    def _realloc(self, moves: list[list], obs) -> list[list] | None:
        """Re-draw the ALLOCATION of one sampled move set from the learned
        Dirichlet, holding its SELECT (the src→tgt pair set) fixed — one
        extra candidate with ZERO extra forwards (head outputs stashed by
        the runner act()). Returns None when the ckpt has no α0 head."""
        ctx = getattr(self.agent, "_last_act_ctx", None)
        if not ctx or ctx.get("alloc_conc") is None or not moves:
            return None
        frac = ctx["pair_frac_raw"]
        conc = ctx["alloc_conc"]
        pid_to_idx = ctx["pid_to_idx"]
        get = obs.get if isinstance(obs, dict) else (
            lambda k, d=None: getattr(obs, k, d))
        ships_by_pid = {int(p[0]): int(p[5]) for p in (get("planets") or [])}
        from agents.heuristic.physical_v4.agent import PHASE_TABLE, phase_of
        min_launch = int(PHASE_TABLE[phase_of(int(get("step") or 0))][2])

        by_src: dict[int, list[int]] = {}
        for mi, (src_pid, _ang, _n) in enumerate(moves):
            by_src.setdefault(int(src_pid), []).append(mi)
        out = [list(m) for m in moves]
        for src_pid, mis in by_src.items():
            s = pid_to_idx.get(src_pid)
            n_ships = ships_by_pid.get(src_pid, 0)
            if s is None or n_ships <= 0:
                continue
            # NOTE: tgt slot per move is unknown from (pid, angle); the
            # Dirichlet over [fired…, self] only needs the FIRED CELLS'
            # frac logits — recover them by ranking: launches were emitted
            # per-source in fired order, share scale is what we re-draw.
            tgt_slots = [self._move_tgt_slot.get((id(moves), mi))
                         for mi in mis]
            if any(t is None for t in tgt_slots):
                return None
            alloc_logits = torch.cat([
                frac[s, torch.tensor(tgt_slots, dtype=torch.long)],
                frac[s, s].reshape(1),
            ])
            mean = torch.softmax(alloc_logits, dim=-1)
            a0 = conc[s].clamp(min=1e-3)
            x = torch.distributions.Dirichlet(
                (a0 * mean).clamp(min=1e-4)).sample()
            rem = max(0, n_ships - min_launch * len(mis))
            for j, mi in enumerate(mis):
                out[mi][2] = int(min_launch + round(float(x[j]) * rem))
        return out

    def _candidates(self, obs) -> list[list[list]]:
        """K = 1 deterministic (expmatch) + N sampled selects × M alloc
        draws each (first draw = the act() sample itself, M−1 re-draws via
        :meth:`_realloc` at zero forward cost). N/M from OW_V4_N / OW_V4_M
        (defaults keep K ≈ the old flat count: N = k−1, M = 1)."""
        n_sel = int(os.environ.get("OW_V4_N", max(1, self.k - 1)))
        m_alloc = int(os.environ.get("OW_V4_M", 1))
        sets: list[list[list]] = []
        prev = os.environ.get("OW_V3_DECODE")
        try:
            os.environ["OW_V3_DECODE"] = "expmatch"   # deterministic arm
            sets.append(self.agent.act(obs))
            os.environ["OW_V3_DECODE"] = "sample"
            for _ in range(n_sel):
                base = self.agent.act(obs)
                self._index_move_slots(base)
                sets.append(base)
                for _ in range(m_alloc - 1):
                    var = self._realloc(base, obs)
                    if var is not None:
                        sets.append(var)
        finally:
            if prev is None:
                os.environ.pop("OW_V3_DECODE", None)
            else:
                os.environ["OW_V3_DECODE"] = prev
        # dedup identical sets (common when few legal moves)
        uniq, seen = [], set()
        for s in sets:
            key = tuple(sorted((m[0], round(m[1], 4), m[2]) for m in s))
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    def _index_move_slots(self, moves: list[list]) -> None:
        """Map each move of a JUST-SAMPLED set to its target slot using the
        stashed pid→slot map (target pid is recoverable while the stash and
        the move set come from the same act() call)."""
        ctx = getattr(self.agent, "_last_act_ctx", None)
        self._move_tgt_slot = {}
        if not ctx:
            return
        tgt_pids = getattr(self.agent, "_last_move_tgt_pids", None)
        if tgt_pids is None or len(tgt_pids) != len(moves):
            return
        for mi, tpid in enumerate(tgt_pids):
            slot = ctx["pid_to_idx"].get(int(tpid))
            if slot is not None:
                self._move_tgt_slot[(id(moves), mi)] = int(slot)

    # ---- deterministic action application ---------------------------
    @staticmethod
    def _apply(obs, moves: list[list]) -> dict[str, Any]:
        get = obs.get if isinstance(obs, dict) else (
            lambda k, d=None: getattr(obs, k, d))
        post = {
            "planets": [list(p) for p in (get("planets") or [])],
            "fleets": [list(f) for f in (get("fleets") or [])],
        }
        for key in ("step", "angular_velocity", "initial_planets",
                    "comet_planet_ids", "comets", "next_fleet_id",
                    "player_id"):
            v = get(key)
            if v is not None:
                post[key] = copy.deepcopy(v) if key in (
                    "initial_planets", "comets") else v
        by_id = {int(p[0]): p for p in post["planets"]}
        fid = int(post.get("next_fleet_id") or 10_000_000)
        me = int(post.get("player_id") or 0)
        for from_id, angle, ships in moves:
            src = by_id.get(int(from_id))
            ships = int(ships)
            if src is None or src[5] < ships or ships <= 0:
                continue
            src[5] -= ships
            r = float(src[4]) + 0.1
            post["fleets"].append([
                fid, me,
                float(src[2]) + math.cos(angle) * r,
                float(src[3]) + math.sin(angle) * r,
                float(angle), int(from_id), ships,
            ])
            fid += 1
        post["next_fleet_id"] = fid
        return post

    # ---- scoring ------------------------------------------------------
    def _score(self, obs_post) -> float:
        preds = self.agent.value_forward(obs_post)
        win = torch.sigmoid(preds["win"][0, 0]).item()
        fwd = preds["fwd"][0, 0]                       # (S, NF)
        s = RANKER_WIN_W * win
        for h_i, w_h in self._h_idx:
            s += w_h * float((fwd[:, h_i] * self._sig_w.to(fwd.device)).sum())
        return s

    def act(self, obs) -> list[list]:
        cands = self._candidates(obs)
        if len(cands) == 1:
            return cands[0]
        best, best_s = cands[0], -math.inf
        for moves in cands:
            s = self._score(self._apply(obs, moves))
            if s > best_s:
                best, best_s = moves, s
        return best
