"""Teacher-forced allocation-multinomial labels for the
``bernoulli_select_multinomial_alloc_v2`` actor contract.

PPO samples actions in two stages (see
``agents/transformer_v2/ppo/sampler.py::sample_multi_target``):

  Stage 1 — Selection: per legal off-diagonal cell,
      ``fire[s,t] ~ Bernoulli(sigmoid(pair_logits[s,t]))``.
  Stage 2 — Allocation: per acting source, ONE multinomial over the fired
      targets plus a ``self``/HOLD category,
      ``counts ~ Multinomial(N_s, softmax([frac_loc[s, F_s], frac_loc[s, s]]))``
      with ``N_s`` = the source's current ship count. The counts ARE the
      launch sizes; the ``self`` count stays home. (v2: the HOLD logit is the
      frac head's own diagonal; v1 borrowed ``pair_logits[s, s]``.)

Pretraining the old way leaves exactly one input of that softmax untrained:
the diagonal HOLD logit ``frac_loc[s, s]`` (the diagonal is masked out of
both the whole-grid select BCE and the sigmoid-MSE frac loss). This module
builds the stage-2 supervision that closes that hole:

  * Stage 1 keeps the existing whole-grid select BCE (``pair_labels``) —
    nothing here touches it. Held rows supervise it with all-zero bits.
  * Stage 2 (this module): for each learner-owned source row with at least
    one expert launch, the empirical allocation distribution over
    ``[expert-fired targets ... , HOLD]``:

        target_share(s, t) = ships_sent(s, t) / N_s          (t in F_s)
        target_share(s, HOLD) = (N_s - sum_t ships_sent) / N_s

    trained by cross-entropy against
    ``log_softmax([frac_loc[s, F_s], frac_loc[s, s]])`` — the exact
    parameterization PPO samples from, teacher-forced on the expert's fired
    set. This supervises the relative frac logits AND the HOLD diagonal
    jointly. Held rows (no launch) have a degenerate one-category
    multinomial — no gradient — so hold-vs-launch stays entirely with
    stage 1, matching the contract's factorization.

``N_s`` is recovered from the planet-features log-ships channel (the same
path ``_pair_frac_targets_from_batch`` uses), so the kept-share label
inherits that channel's log/expm1 quantization. Rows whose summed sent
ships exceed the recovered ``N_s`` are clamped to HOLD share 0 and counted
in the stats.
"""

from __future__ import annotations

from typing import Counter as CounterT

import torch
import torch.nn.functional as F

from ..featurizer.fleet_featurizer import SHIPS_LOG_MAX
from .entity_encoder import (
    _PLANET_SHIPS_LOG_FEATURE_IDX,
    _current_planet_features,
    _owned_source_rows,
)

__all__ = [
    "build_alloc_targets",
    "alloc_multinomial_ce",
    "ALLOC_STAT_KEYS",
]

# Counter keys filled by build_alloc_targets when a stats Counter is passed.
ALLOC_STAT_KEYS: tuple[str, ...] = (
    "owned_rows",            # learner-owned present source rows
    "acted_rows",            # owned rows with >=1 positive (label & valid) cell
    "supervised_rows",       # rows that produce an allocation target
    "fired_cells",           # expert (s, t) cells inside supervised rows
    "dropped_rows_ships0",   # acted rows whose positives ALL carry ships == 0
    "dropped_rows_no_src",   # acted rows with recovered N_s < 1
    "dropped_cells_ships0",  # positive cells with ships == 0 inside kept rows
    "overflow_rows",         # rows where sum(sent) > N_s (HOLD clamped to 0)
    "acted_flag_no_window",  # is_source_this_turn rows with no in-grid positive
)


def _source_ships_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Recover integer pre-launch source ships per planet slot, (B, P).

    Same channel + transform as ``_pair_frac_targets_from_batch``; rounded
    because the multinomial N is an integer ship count.
    """
    planet_features = _current_planet_features(batch["planet_features"])
    ships_log = planet_features[..., _PLANET_SHIPS_LOG_FEATURE_IDX]
    return torch.expm1(ships_log.clamp(min=0.0) * SHIPS_LOG_MAX).round()


def build_alloc_targets(
    batch: dict[str, torch.Tensor],
    *,
    stats: CounterT[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build stage-2 multinomial targets from a pair-cache batch.

    Returns ``(row_mask, target)``:

      row_mask (B, P) bool — rows carrying a supervised allocation
          multinomial: learner-owned, >=1 expert fired cell with ships > 0,
          and recovered ``N_s >= 1``.
      target (B, P, P+1) float — empirical shares. ``target[b, s, t]`` for
          ``t < P`` is ``sent(s, t) / total`` on the expert's fired cells
          (0 elsewhere); ``target[b, s, P]`` is the HOLD share
          ``kept / total``. Masked rows sum to 1; other rows are all 0.

    ``total = max(N_s, sum(sent))`` — when the log-channel ``N_s``
    under-recovers the true pre-launch count, HOLD clamps to 0 instead of
    going negative (counted as ``overflow_rows`` in ``stats``).
    """
    pair_labels = batch["pair_labels"].bool()
    pair_valid = batch["pair_valid"].bool()
    ships = batch["pair_ships"].float()
    squeeze = pair_labels.dim() == 2
    if squeeze:  # single snapshot -> fake batch axis
        pair_labels, pair_valid, ships = (
            x.unsqueeze(0) for x in (pair_labels, pair_valid, ships)
        )
        batch = {
            **batch,
            "pair_labels": pair_labels,
            "pair_valid": pair_valid,
            "pair_ships": ships,
        }

    B, P, _ = pair_labels.shape
    pos = pair_labels & pair_valid                       # (B, P, P)
    fired = pos & (ships > 0)                            # expert cells with sizes
    owned = _owned_source_rows(batch, pair_valid)        # (B, P)

    n_src = _source_ships_from_batch(batch)              # (B, P) float (integers)
    sent = (ships * fired).sum(dim=-1)                   # (B, P)
    row_mask = owned & fired.any(dim=-1) & (n_src >= 1.0)

    total = torch.maximum(n_src, sent).clamp(min=1.0)    # (B, P)
    kept = total - sent                                  # >= 0 by construction

    target = torch.zeros(B, P, P + 1, dtype=torch.float32, device=ships.device)
    target[..., :P] = torch.where(fired, ships, 0.0) / total.unsqueeze(-1)
    target[..., P] = kept / total
    target = target * row_mask.unsqueeze(-1)

    if stats is not None:
        acted = owned & pos.any(dim=-1)
        stats["owned_rows"] += int(owned.sum())
        stats["acted_rows"] += int(acted.sum())
        stats["supervised_rows"] += int(row_mask.sum())
        stats["fired_cells"] += int((fired & row_mask.unsqueeze(-1)).sum())
        stats["dropped_rows_ships0"] += int((acted & ~fired.any(dim=-1)).sum())
        stats["dropped_rows_no_src"] += int(
            (acted & fired.any(dim=-1) & (n_src < 1.0)).sum()
        )
        stats["dropped_cells_ships0"] += int(
            (pos & ~fired & row_mask.unsqueeze(-1)).sum()
        )
        stats["overflow_rows"] += int((row_mask & (sent > n_src)).sum())
        if "is_source_this_turn" in batch:
            acted_flag = batch["is_source_this_turn"] > 0.5
            if squeeze and acted_flag.dim() == 1:
                acted_flag = acted_flag.unsqueeze(0)
            stats["acted_flag_no_window"] += int(
                (acted_flag & owned & ~pos.any(dim=-1)).sum()
            )

    if squeeze:
        return row_mask.squeeze(0), target.squeeze(0)
    return row_mask, target


def alloc_multinomial_ce(
    frac_loc: torch.Tensor,           # (B, P, P) — raw pair_frac logits (diagonal = HOLD)
    batch: dict[str, torch.Tensor],
    *,
    stats: CounterT[str] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage-2 cross-entropy in the exact PPO allocation parameterization.

    Per supervised row ``s``: ``CE(target, log_softmax([frac_loc[s, F_s],
    frac_loc[s, s]]))`` with ``F_s`` teacher-forced to the expert's fired
    cells (v2: the self/HOLD logit is the frac head's own diagonal — the
    select head is not involved, so select and alloc gradients stay
    decoupled). Mean over supervised rows (uniform row weight — a 600-ship
    turn counts the same as a 6-ship turn, mirroring how the select BCE
    averages cells). Returns ``(loss, diagnostics)``; loss is 0 (grad-free)
    when the batch has no supervised row.
    """
    pair_labels = batch["pair_labels"].bool()
    pair_valid = batch["pair_valid"].bool()
    ships = batch["pair_ships"].float()
    fired = pair_labels & pair_valid & (ships > 0)        # (B, P, P)

    row_mask, target = build_alloc_targets(batch, stats=stats)
    if not bool(row_mask.any()):
        zero = frac_loc.new_zeros(())
        return zero, {"alloc_ce": float("nan"), "alloc_rows": 0,
                      "hold_share_pred": float("nan"),
                      "hold_share_label": float("nan"),
                      "hold_mae": float("nan")}

    neg_inf = torch.finfo(frac_loc.dtype).min
    alloc_logits = frac_loc.masked_fill(~fired, neg_inf)  # (B, P, P)
    diag = frac_loc.diagonal(dim1=-2, dim2=-1)            # (B, P) HOLD logit
    full = torch.cat([alloc_logits, diag.unsqueeze(-1)], dim=-1)  # (B, P, P+1)
    logp = F.log_softmax(full, dim=-1)

    # target == 0 exactly where logp may be -inf (non-fired cells); a HOLD
    # share of exactly 0 (expert launched everything) also lands here.
    ce_terms = torch.where(target > 0, -target * logp, torch.zeros_like(logp))
    ce_row = ce_terms.sum(dim=-1)                          # (B, P)
    loss = ce_row[row_mask].mean()

    with torch.no_grad():
        hold_pred = logp[..., -1].exp()[row_mask]          # model HOLD share
        hold_label = target[..., -1][row_mask]
        diagnostics = {
            "alloc_ce": float(loss.detach()),
            "alloc_rows": int(row_mask.sum()),
            "hold_share_pred": float(hold_pred.mean()),
            "hold_share_label": float(hold_label.mean()),
            "hold_mae": float((hold_pred - hold_label).abs().mean()),
        }
    return loss, diagnostics
