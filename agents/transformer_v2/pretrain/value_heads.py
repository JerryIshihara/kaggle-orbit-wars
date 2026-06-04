"""Value-pretrain heads (action-impact design).

Off the shared L2 (via ``feat = concat(glob, player_state)`` → ``value_trunk``):

  TIER 1 (objective)   win            (O,)        terminal win/loss, BCE
  TIER 2 (PPO value)   fwd            (O,5,5)     per-signal FUTURE LEVEL sᵢ(t+K),
                                                  5 P1 signals × P1_FWD_HORIZONS, Huber
  TIER 3 (aux)         back           (O,5,3)     backward Δ sᵢ(t)−sᵢ(t−K) (anti-shortcut), Huber
                       rank           (O,)        final-rank ordering, ListMLE
                       survives       (O,5)       alive at t+K, BCE

The 5 P1 signals (see value_signals.P1_SIGNAL_NAMES) are all relative-to-the-
strongest-enemy advantage shares in [0,1]; the heads predict their actual future
level. ``compute_value_pretrain_loss`` returns ``(total, per_head)`` with the
tiered weights and a per-head breakdown for verbose logging. Import-light (only
consolidator_heads + value_signals constants) — never imports cross_entity /
entity_encoder, so no cycle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .consolidator_heads import _build_mlp
from .value_signals import (
    N_P1_SIGNALS,
    P1_BACK_HORIZONS,
    P1_FWD_HORIZONS,
    P1_SIGNAL_NAMES,
)

N_FWD = len(P1_FWD_HORIZONS)
N_BACK = len(P1_BACK_HORIZONS)


def _zero_last_linear(m: nn.Module) -> None:
    last = m[-1] if isinstance(m, nn.Sequential) else m
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


class ValuePretrainHeads(nn.Module):
    def __init__(
        self, d_model: int, *,
        trunk_n_layers: int = 2, head_n_layers: int = 2, dropout: float = 0.0,
    ):
        super().__init__()
        feat = 2 * d_model
        self.value_trunk = _build_mlp(
            in_dim=feat, hidden=d_model, out_dim=d_model,
            n_layers=trunk_n_layers, dropout=dropout,
        )

        def mlp(out):
            return _build_mlp(in_dim=d_model, hidden=d_model, out_dim=out,
                              n_layers=head_n_layers, dropout=dropout)

        self.win_head = mlp(1)                                       # (O,)
        self.fwd_heads = nn.ModuleList([mlp(N_FWD) for _ in range(N_P1_SIGNALS)])
        self.back_heads = nn.ModuleList([mlp(N_BACK) for _ in range(N_P1_SIGNALS)])
        self.rank_head = mlp(1)
        self.surv_head = mlp(N_FWD)
        # Default init (no zero-init): regression/level heads start with small
        # predictions and the trunk receives gradient from step 1. Only win is
        # nudged neutral so V starts ~0.5 win-prob.
        _zero_last_linear(self.win_head)

    def forward(self, glob: torch.Tensor, player_state: torch.Tensor) -> dict:
        O = player_state.size(1)
        feat = torch.cat([glob.unsqueeze(1).expand(-1, O, -1), player_state], dim=-1)
        h = self.value_trunk(feat)                                  # (B,O,d)
        fwd = torch.stack([hd(h) for hd in self.fwd_heads], dim=2)  # (B,O,5,N_FWD)
        back = torch.stack([hd(h) for hd in self.back_heads], dim=2)  # (B,O,5,N_BACK)
        return {
            "win": self.win_head(h).squeeze(-1),       # (B,O)
            "fwd": fwd,                                  # (B,O,5,N_FWD)
            "back": back,                                # (B,O,5,N_BACK)
            "rank": self.rank_head(h).squeeze(-1),     # (B,O)
            "survives": self.surv_head(h),             # (B,O,N_FWD)
        }


# ---------- losses ----------
def _mmean(term: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(term.dtype)
    return (term * mask).sum() / mask.sum().clamp(min=1.0)


def _listmle(scores, rank, valid):
    neg = torch.finfo(scores.dtype).min
    isv = valid > 0.5
    s = scores.masked_fill(~isv, neg)
    order = torch.argsort(rank.float().masked_fill(~isv, float("inf")), dim=-1)
    s_sorted = torch.gather(s, 1, order)
    valid_sorted = torch.gather(isv.float(), 1, order)
    # log P(pick o_k | tail) = s[o_k] - logsumexp(s[o_k:])
    rev_lse = torch.logcumsumexp(s_sorted.flip(-1), dim=-1).flip(-1)
    ll = (s_sorted - rev_lse) * valid_sorted
    n_valid = valid.sum(-1)
    rows = n_valid >= 2
    if not bool(rows.any()):
        return scores.sum() * 0.0
    return -(ll.sum(-1)[rows] / n_valid[rows].clamp(min=1)).mean()


def compute_value_pretrain_loss(
    preds: dict, batch: dict, *,
    w_win: float = 1.0, w_fwd: float = 0.5, w_aux: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Tiered value-pretrain loss. Targets come from the cache's P1 labels:
      win:  final_rank==0     (B,O)
      fwd:  p1_future         (B,O,5,N_FWD)  + p1_valid (B,N_FWD)
      back: p1_back           (B,O,5,N_BACK) + valid_back (B,N_BACK)
      survives: survives_future (B,O,N_FWD)
      player_valid (B,O).
    """
    pv = batch["player_valid"].float()                              # (B,O)
    dev = preds["win"].device
    total = preds["win"].sum() * 0.0
    per: dict[str, float] = {}

    # TIER 1 — win
    z = (batch["final_rank"].long() == 0).float()
    bce = F.binary_cross_entropy_with_logits(preds["win"], z, reduction="none")
    L_win = _mmean(bce, pv)
    total = total + w_win * L_win
    per["win"] = float(L_win.detach())
    with torch.no_grad():
        acc = (((preds["win"] > 0).float() == z) * pv).sum() / pv.sum().clamp(min=1)
        per["win_acc"] = float(acc)

    # TIER 2 — forward signal levels (per-signal, masked by valid horizon)
    fut = batch["p1_future"]                                        # (B,O,5,N_FWD)
    vfwd = batch["p1_valid"].float()                                # (B,N_FWD)
    mask_f = pv[:, :, None, None] * vfwd[:, None, None, :]          # (B,O,1,N_FWD)->bcast
    err = F.smooth_l1_loss(preds["fwd"], fut, reduction="none")     # (B,O,5,N_FWD)
    L_fwd = _mmean(err, mask_f.expand_as(err))
    total = total + w_fwd * L_fwd
    per["fwd"] = float(L_fwd.detach())
    for si, name in enumerate(P1_SIGNAL_NAMES):
        per[f"fwd/{name}"] = float(_mmean(err[:, :, si], mask_f.expand_as(err)[:, :, si]).detach())

    # TIER 3 — aux: backward Δ (anti-shortcut), rank, survives
    if "p1_back" in batch:
        vback = batch.get("valid_back")
        mb = pv[:, :, None, None]
        if vback is not None:
            mb = mb * vback.float()[:, None, None, :]
        eb = F.smooth_l1_loss(preds["back"], batch["p1_back"], reduction="none")
        L_back = _mmean(eb, mb.expand_as(eb))
        total = total + w_aux * L_back
        per["aux/back"] = float(L_back.detach())
    L_rank = _listmle(preds["rank"], batch["final_rank"].long(), pv)
    total = total + w_aux * L_rank
    per["aux/rank"] = float(L_rank.detach())
    if "survives_future" in batch:
        sb = F.binary_cross_entropy_with_logits(
            preds["survives"], batch["survives_future"].float(), reduction="none")
        L_surv = _mmean(sb, (pv[:, :, None] * vfwd[:, None, :]).expand_as(sb))
        total = total + w_aux * L_surv
        per["aux/survives"] = float(L_surv.detach())

    return total, per


def format_value_per_head(per: dict[str, float], *, title: str = "value") -> str:
    lines = [f"    [{title}]"]
    order = ["win", "win_acc", "fwd"] + [f"fwd/{n}" for n in P1_SIGNAL_NAMES] \
        + ["aux/back", "aux/rank", "aux/survives"]
    for k in order:
        if k in per:
            lines.append(f"      {k:<22s} {per[k]:.4f}")
    return "\n".join(lines)
