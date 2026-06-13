"""Per-pair launch Q-value: training target + loss + deploy gate.

The Q head (``PairHead(with_q_head=True)`` → ``out["q_value"]`` shape (B,P,P))
scores each candidate launch s→t. Its purpose is the thing V(s) and the
inval-coef penalty cannot do (see the campaign notes): make DOOMED launches
(sun/boundary/comet-expired) visibly low-value to the actor at SELECTION time,
without waiting for the diluted, delayed, confounded ship-loss reward to
teach an unobservable physics constraint.

WHY ROLLOUT DATA, NOT PRETRAIN: top-meta experts never make doomed launches,
so a Q head fit on expert data has ZERO doomed examples and is blind to them
(OOD — exactly why we rejected it before). It MUST be trained on PPO-rollout
data, where exploration produces doomed launches.

TWO TARGETS, combined:

  1. DENSE reachability label (the anti-doomed signal). Every legal pair gets
     a plan_launch verdict — reachable or doomed — for FREE (we already
     compute it for the inval penalty). Doomed pairs are regressed to
     ``Q_DOOMED`` (a fixed low value); reachable-but-not-fired pairs are left
     UNSUPERVISED by this term (we don't know their strategic value, only that
     they're physically possible). This term gives the Q head dense supervision
     on the PHYSICS the actor's features don't encode.

  2. SPARSE TD return (the strategic signal). For each pair the policy actually
     FIRED, the Q target is the step's bootstrapped return ``r + γ·V(s')`` —
     the same return the value head fits, attributed to the launch. Sparse (one
     value shared across a step's fired pairs) but over many rollout steps the
     Q head learns which reachable targets correlate with winning.

Deploy use (``q_gate_select_logits``): bias the SELECT logits by the Q head —
fire toward high-Q targets, away from low-Q (doomed) ones. A learned, soft,
differentiable cousin of a hard reachability mask: it also expresses "risky"
and "strong target", not just "legal/illegal".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Regression target for plan_launch-doomed pairs (return units; well below a
#: typical winning return so doomed always ranks under any real target).
Q_DOOMED = -1.0
#: Weight of the dense doomed term relative to the sparse TD term.
Q_DOOMED_WEIGHT = 1.0


def compute_q_loss(
    q_value: torch.Tensor,        # (B, P, P) — Q head output
    *,
    fired_mask: torch.Tensor,     # (B, P, P) bool — pairs the policy launched this step
    fired_return: torch.Tensor,   # (B,) float — bootstrapped return r + γ·V(s') per step
    doomed_mask: torch.Tensor,    # (B, P, P) bool — plan_launch-doomed legal pairs
    legal_mask: torch.Tensor,     # (B, P, P) bool — legal (owned src, exists tgt) pairs
    huber_beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Q loss = sparse TD (fired pairs → step return) + dense doomed (→ Q_DOOMED).

    Both Huber. ``fired_mask`` and ``doomed_mask`` are disjoint by construction
    (a fired launch cleared validation; a doomed pair was refused) but the loss
    does not rely on it — TD supervises fired pairs, doomed supervises doomed
    pairs, and any overlap simply sums.
    """
    B, P, _ = q_value.shape
    device = q_value.device

    # --- sparse TD on fired pairs ---
    tgt_td = fired_return.view(B, 1, 1).expand_as(q_value)
    td_err = F.huber_loss(q_value, tgt_td, reduction="none", delta=huber_beta)
    n_fired = fired_mask.sum().clamp_min(1)
    loss_td = (td_err * fired_mask).sum() / n_fired

    # --- dense doomed label ---
    if doomed_mask.any():
        tgt_d = torch.full_like(q_value, Q_DOOMED)
        d_err = F.huber_loss(q_value, tgt_d, reduction="none", delta=huber_beta)
        n_doomed = doomed_mask.sum().clamp_min(1)
        loss_doomed = (d_err * doomed_mask).sum() / n_doomed
    else:
        loss_doomed = q_value.sum() * 0.0

    loss = loss_td + Q_DOOMED_WEIGHT * loss_doomed

    with torch.no_grad():
        # Separation: mean Q on doomed vs reachable legal pairs. A working Q
        # head drives this gap NEGATIVE (doomed ranks below reachable).
        reach = legal_mask & ~doomed_mask
        q_doomed_mean = (q_value[doomed_mask].mean().item()
                         if doomed_mask.any() else float("nan"))
        q_reach_mean = (q_value[reach].mean().item()
                        if reach.any() else float("nan"))
    return loss, {
        "q_loss_td": float(loss_td.detach()),
        "q_loss_doomed": float(loss_doomed.detach()),
        "q_doomed_mean": q_doomed_mean,
        "q_reach_mean": q_reach_mean,
        "q_separation": (q_doomed_mean - q_reach_mean
                         if q_reach_mean == q_reach_mean else float("nan")),
        "n_fired": int(fired_mask.sum()),
        "n_doomed": int(doomed_mask.sum()),
    }


def q_gate_select_logits(
    pair_logits: torch.Tensor,    # (P, P) or (B, P, P) — select head logits
    q_value: torch.Tensor,        # same shape — Q head
    *,
    weight: float = 1.0,
    doomed_floor: float | None = None,
) -> torch.Tensor:
    """Deploy gate: bias select logits toward high-Q targets.

    ``logits' = pair_logits + weight · (q_value − detach(max_t q_value))`` —
    the per-source max-shift keeps the strongest target unchanged and pushes
    everything below it down by its Q deficit, so low-Q (doomed) cells lose
    selection probability. ``doomed_floor`` (e.g. Q_DOOMED + 0.1) HARD-masks
    cells whose Q sits at the doomed level — a learned reachability mask. Off
    by default (soft bias only)."""
    q = q_value
    q_ref = q.max(dim=-1, keepdim=True).values.detach()
    out = pair_logits + weight * (q - q_ref)
    if doomed_floor is not None:
        out = out.masked_fill(q <= doomed_floor, float("-inf"))
    return out
