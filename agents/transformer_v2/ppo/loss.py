"""PPO losses + factorized BC anchor.

action_logprob:
    Recomputes ``logp_source + logp_targets + logp_frac`` at update time
    from the stored ``(source_id, target_bits, frac_raw)`` and the current
    policy outputs. Matches the sampler's source ``row_max`` formula and
    the per-row Bernoulli factorization.

factorized_bc:
    BC anchor that matches the ``source_multi_target_v1`` action contract.
    Splits into ``bc_source`` (CE over source choices including NOOP) +
    ``bc_target`` (Bernoulli BCE on the expert's source row only). See
    protocol "BC anchor" + "Known problems" #4.

ppo_minibatch_loss:
    Standard PPO clipped policy loss + MSE value loss + entropy +
    factorized BC. Returns ``(loss, diagnostics_dict)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Bernoulli, Categorical, Normal
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Minibatch dataclass                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class PPOMinibatch:
    """One PPO minibatch's stored tensors.

    Shapes (B = minibatch size, P = max planet slots):
      feats           dict of input tensors for the model forward
      pair_mask       (B, P, P) bool — exact mask used at sample time
      source_mask     (B, P) bool
      source_id_plus  (B,) long ∈ {0..P}; 0 = NOOP, else 1+source_idx
      target_bits     (B, P) bool
      frac_raw        (B, P) float
      old_logp        (B,) float (sum of source + target + frac logprobs)
      adv             (B,) float (normalized GAE)
      returns         (B,) float (GAE + value)
      noop_logit_bias scalar; saved in shard so update is replay-faithful
    """

    feats: dict[str, torch.Tensor]
    pair_mask: torch.Tensor
    source_mask: torch.Tensor
    source_id_plus: torch.Tensor
    target_bits: torch.Tensor
    frac_raw: torch.Tensor
    old_logp: torch.Tensor
    adv: torch.Tensor
    returns: torch.Tensor
    noop_logit_bias: float = 0.0

    @property
    def size(self) -> int:
        return int(self.source_id_plus.shape[0])


@dataclass
class BCMinibatch:
    """Factorized BC anchor minibatch (sampled from the supervised pair_cache).

    source_mask: (B, P) bool — legal source rows under the PPO source
        categorical approximation. This is intentionally stricter than
        pair_mask.any(dim=2): source rows must be learner-owned, otherwise
        the BC source classifier is trained on impossible PPO source choices.
    expert_source_idx_or_noop: (B,) long ∈ {0..P}. 0 means "no expert
        launched from any source this snapshot" (NOOP); k>0 means the
        expert launched from source (k-1). For coalition turns with
        multiple expert sources, one canonical source is picked at cache
        build time (e.g., the source with the largest single launch).
    expert_target_bits: (B, P, P) bool — per-source row of expert target
        bits. Multi-positive on coalition turns; the BC target loss only
        consumes the row matching ``expert_source_idx_or_noop``.
    """

    feats: dict[str, torch.Tensor]
    pair_mask: torch.Tensor                  # (B, P, P) bool
    source_mask: torch.Tensor                # (B, P) bool
    expert_source_idx_or_noop: torch.Tensor  # (B,) long
    expert_target_bits: torch.Tensor         # (B, P, P) bool


# --------------------------------------------------------------------------- #
# action_logprob                                                              #
# --------------------------------------------------------------------------- #
def action_logprob(
    pair_logits: torch.Tensor,        # (B, P, P)
    frac_loc: torch.Tensor,            # (B, P, P)
    sigma: torch.Tensor,               # scalar
    pair_mask: torch.Tensor,           # (B, P, P) bool
    source_mask: torch.Tensor,         # (B, P) bool
    source_id_plus: torch.Tensor,      # (B,) long ∈ {0..P}; 0 = NOOP
    target_bits: torch.Tensor,         # (B, P) bool
    frac_raw: torch.Tensor,            # (B, P) float
    *,
    noop_logit_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the action's logprob under the CURRENT policy.

    Matches :func:`sample_source_multi_target` exactly. Used for the PPO ratio.

    Returns ``(logp, n_terms)`` where ``logp`` is the summed action logprob and
    ``n_terms`` is the per-sample count of summed logprob components (1 source +
    valid target Bernoulli cells + selected-target frac terms). ``n_terms`` lets
    the caller normalize the KL to a PER-COMPONENT scale — the raw KL is a sum
    over ~30 sub-actions, so a scalar ``target_kl`` is otherwise ~30× too tight.
    """
    if pair_logits.dim() != 3:
        raise ValueError("expected pair_logits (B, P, P)")
    B, P, _ = pair_logits.shape
    device = pair_logits.device

    neg_inf = torch.full_like(pair_logits, float("-inf"))
    masked_pairs = torch.where(pair_mask, pair_logits, neg_inf)

    # 1. Source logprob (Categorical over [NOOP, P sources]; row max scoring).
    source_logits = masked_pairs.max(dim=2).values                   # (B, P)
    source_logits = torch.where(
        source_mask, source_logits,
        torch.full_like(source_logits, float("-inf")),
    )
    noop_col = torch.full((B, 1), float(noop_logit_bias), device=device)
    cat_logits = torch.cat([noop_col, source_logits], dim=1)         # (B, P+1)
    src_dist = Categorical(logits=cat_logits)
    logp_source = src_dist.log_prob(source_id_plus)                  # (B,)

    # 2 + 3. Target Bernoulli + frac logit-normal on the chosen source's row.
    #    Vectorize over B by gathering the chosen row per sample. For NOOP
    #    samples (source_id_plus == 0), the target and frac contributions
    #    are 0.
    acted = source_id_plus > 0                                        # (B,)
    logp_targets = torch.zeros(B, device=device)
    logp_frac = torch.zeros(B, device=device)
    n_terms = torch.ones(B, device=device)   # summed-logp component count (source=1)

    if acted.any():
        # Gather the chosen source's row for each acted sample.
        src_idx = (source_id_plus - 1).clamp_min(0).long()           # (B,)
        # Use gather: pair_logits[b, src_idx[b], :] -> (B, P) per sample
        row_logits = pair_logits.gather(
            1, src_idx[:, None, None].expand(B, 1, P),
        ).squeeze(1)                                                  # (B, P)
        row_mask = pair_mask.gather(
            1, src_idx[:, None, None].expand(B, 1, P),
        ).squeeze(1)                                                  # (B, P) bool
        row_frac_loc = frac_loc.gather(
            1, src_idx[:, None, None].expand(B, 1, P),
        ).squeeze(1)                                                  # (B, P)

        # 2. Bernoulli logprob on valid cells only.
        row_logits_masked = torch.where(
            row_mask, row_logits, torch.full_like(row_logits, float("-inf")),
        )
        tgt_dist = Bernoulli(logits=row_logits_masked)
        per_cell_logp = tgt_dist.log_prob(target_bits.float())        # (B, P)
        per_cell_logp = torch.where(
            row_mask, per_cell_logp, torch.zeros_like(per_cell_logp),
        )
        logp_t = per_cell_logp.sum(dim=1)                             # (B,)
        logp_targets = torch.where(acted, logp_t, logp_targets)

        # 3. Frac logit-normal logprob on SELECTED targets only.
        #    Skip cells where target_bits=False (their stored frac_raw is 0).
        sel = target_bits & row_mask                                  # (B, P) bool
        frac_safe = frac_raw.clamp(1e-4, 1 - 1e-4)
        z = torch.logit(frac_safe)
        # Normal log_prob (B, P); zero out unselected cells before summing.
        normal_logp = Normal(row_frac_loc, sigma).log_prob(z)         # (B, P)
        # Jacobian: log p(sigmoid(z)) = log p(z) - log(frac) - log(1 - frac).
        jacobian = -torch.log(frac_safe) - torch.log1p(-frac_safe)    # (B, P)
        per_target_frac_logp = normal_logp + jacobian
        per_target_frac_logp = torch.where(
            sel, per_target_frac_logp, torch.zeros_like(per_target_frac_logp),
        )
        logp_f = per_target_frac_logp.sum(dim=1)                      # (B,)
        logp_frac = torch.where(acted, logp_f, logp_frac)

        # Per-sample component count: valid target Bernoulli cells + selected
        # frac terms (the source's 1 is already in n_terms). Used to normalize
        # the PPO KL to a per-component scale.
        acted_n = row_mask.float().sum(dim=1) + sel.float().sum(dim=1)  # (B,)
        n_terms = n_terms + torch.where(acted, acted_n, torch.zeros_like(acted_n))

    return logp_source + logp_targets + logp_frac, n_terms


# --------------------------------------------------------------------------- #
# Entropy (for the entropy bonus term)                                        #
# --------------------------------------------------------------------------- #
def source_target_entropy(
    pair_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    noop_logit_bias: float = 0.0,
) -> torch.Tensor:
    """Categorical entropy on the source choice (NOOP + P) plus per-row
    Bernoulli entropy weighted by P(choose that source).

    The frac logit-normal has fixed sigma, so its entropy is constant and
    contributes only a state-independent additive constant — we drop it
    from the bonus.
    """
    B, P, _ = pair_logits.shape
    device = pair_logits.device
    neg_inf = torch.full_like(pair_logits, float("-inf"))
    masked = torch.where(pair_mask, pair_logits, neg_inf)
    source_logits = masked.max(dim=2).values                          # (B, P)
    source_logits = torch.where(
        source_mask, source_logits,
        torch.full_like(source_logits, float("-inf")),
    )
    noop_col = torch.full((B, 1), float(noop_logit_bias), device=device)
    cat_logits = torch.cat([noop_col, source_logits], dim=1)
    src_dist = Categorical(logits=cat_logits)
    src_entropy = src_dist.entropy()                                  # (B,)

    # Per-row Bernoulli entropy weighted by P(source=s).
    src_probs = src_dist.probs                                         # (B, P+1)
    src_probs_no_noop = src_probs[:, 1:]                              # (B, P)
    # For each (B, s), the Bernoulli entropy summed over valid target cells.
    masked_rows = torch.where(
        pair_mask, pair_logits, torch.full_like(pair_logits, float("-inf")),
    )
    row_dist = Bernoulli(logits=masked_rows)
    # Bernoulli entropy is 0 at logit=+-inf; valid cells contribute.
    per_cell_ent = torch.where(
        pair_mask, row_dist.entropy(), torch.zeros_like(masked_rows),
    )                                                                  # (B, P, P)
    row_ent = per_cell_ent.sum(dim=2)                                 # (B, P)
    weighted_row_ent = (src_probs_no_noop * row_ent).sum(dim=1)       # (B,)

    return src_entropy + weighted_row_ent


# --------------------------------------------------------------------------- #
# Factorized BC anchor                                                        #
# --------------------------------------------------------------------------- #
def factorized_bc(
    pair_logits: torch.Tensor,         # (B, P, P)
    pair_mask: torch.Tensor,            # (B, P, P) bool
    source_mask: torch.Tensor,           # (B, P) bool
    expert_src_idx: torch.Tensor,       # (B,) long ∈ {0..P}; 0 = no expert launch
    expert_target_bits: torch.Tensor,   # (B, P, P) bool
    *,
    source_weight: float = 1.0,
    target_weight: float = 1.0,
    noop_logit_bias: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """BC anchor matching ``source_multi_target_v1``.

    bc_source: CE over (P+1) source choices (NOOP + P sources), where the
               source score is the SAME ``row_max`` formula the actor uses.
               ``source_mask`` must match the PPO source legality mask
               approximation; using only ``pair_mask.any(dim=2)`` trains
               impossible non-learner-owned sources as categorical choices.
    bc_target: Bernoulli BCE on the EXPERT's source row only — does not
               bleed across rows, so it cannot fight the source categorical.

    Returns ``(bc_loss, diagnostics)`` where diagnostics carries the two
    sub-losses and quick accuracy metrics for the train log.
    """
    B, P, _ = pair_logits.shape
    device = pair_logits.device
    neg_inf = torch.full_like(pair_logits, float("-inf"))
    masked_pairs = torch.where(pair_mask, pair_logits, neg_inf)
    src_scores = masked_pairs.max(dim=2).values                       # (B, P)
    src_scores = torch.where(
        source_mask & torch.isfinite(src_scores), src_scores,
        torch.full_like(src_scores, float("-inf")),
    )
    noop_col = torch.full((B, 1), float(noop_logit_bias), device=device)
    src_with_noop = torch.cat([noop_col, src_scores], dim=1)          # (B, P+1)
    bc_source = F.cross_entropy(src_with_noop, expert_src_idx)

    # Diagnostics: top-1 accuracy on the source classifier.
    src_top1 = src_with_noop.argmax(dim=1)
    src_top1_acc = (src_top1 == expert_src_idx).float().mean()

    bc_target = torch.zeros((), device=device)
    target_row_recall = torch.zeros((), device=device)
    acted = expert_src_idx > 0
    if acted.any():
        idx = (expert_src_idx[acted] - 1).long()                      # (B_acted,)
        # Gather the expert's row from pair_logits, pair_mask, expert_target_bits.
        pl_acted = pair_logits[acted]
        pm_acted = pair_mask[acted]
        et_acted = expert_target_bits[acted]
        rows_logits = pl_acted.gather(
            1, idx[:, None, None].expand(-1, 1, P),
        ).squeeze(1)                                                   # (B_acted, P)
        rows_mask = pm_acted.gather(
            1, idx[:, None, None].expand(-1, 1, P),
        ).squeeze(1)
        rows_label = et_acted.gather(
            1, idx[:, None, None].expand(-1, 1, P),
        ).squeeze(1).float()                                           # (B_acted, P)

        # Masked Bernoulli BCE. Use BCEWithLogits per cell, mask, divide by
        # valid-cell count for a stable per-sample scale.
        per_cell = F.binary_cross_entropy_with_logits(
            rows_logits, rows_label, reduction="none",
        )                                                              # (B_acted, P)
        per_cell = torch.where(rows_mask, per_cell, torch.zeros_like(per_cell))
        n_valid = rows_mask.float().sum(dim=1).clamp_min(1.0)
        per_sample = per_cell.sum(dim=1) / n_valid
        bc_target = per_sample.mean()

        # Diagnostic: per-row recall at top-k = number of expert positives.
        with torch.no_grad():
            k = rows_label.long().sum(dim=1).clamp_min(1)
            # Soft top-k: for each row, recall = fraction of expert positives
            # that fall in the top-k of (masked) logits.
            row_logits_masked = torch.where(
                rows_mask, rows_logits, torch.full_like(rows_logits, float("-inf")),
            )
            recalls = []
            for i in range(rows_logits.shape[0]):
                ki = int(k[i].item())
                topk = row_logits_masked[i].topk(min(ki, P)).indices
                hit = rows_label[i][topk].sum().item()
                pos = max(1, int(rows_label[i].sum().item()))
                recalls.append(hit / pos)
            target_row_recall = torch.tensor(
                sum(recalls) / max(1, len(recalls)), device=device,
            )

    bc_loss = float(source_weight) * bc_source + float(target_weight) * bc_target
    diagnostics = {
        "bc_source_loss": float(bc_source.detach().item()),
        "bc_target_loss": float(bc_target.detach().item()),
        "bc_source_top1_acc": float(src_top1_acc.detach().item()),
        "bc_target_row_recall_at_k": float(target_row_recall.detach().item()),
    }
    return bc_loss, diagnostics


# --------------------------------------------------------------------------- #
# PPO minibatch loss                                                          #
# --------------------------------------------------------------------------- #
def ppo_minibatch_loss(
    policy,                                # PPOActorCritic
    mb: PPOMinibatch,
    bc_mb: BCMinibatch | None,
    *,
    clip: float = 0.10,
    value_coef: float = 0.5,
    ent_coef: float = 0.01,
    bc_coef: float = 0.05,
    bc_target_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One PPO minibatch loss.

    Returns ``(total_loss, diagnostics)``. ``approx_kl`` is included in the
    diagnostics so the outer loop can early-stop the PPO epoch when
    ``approx_kl > 1.5 * target_kl``.
    """
    out = policy(**mb.feats)
    sigma = out["sigma"]

    new_logp, n_terms = action_logprob(
        pair_logits=out["pair_logits"],
        frac_loc=out["frac_loc"],
        sigma=sigma,
        pair_mask=mb.pair_mask,
        source_mask=mb.source_mask,
        source_id_plus=mb.source_id_plus,
        target_bits=mb.target_bits,
        frac_raw=mb.frac_raw,
        noop_logit_bias=mb.noop_logit_bias,
    )

    ratio = torch.exp(new_logp - mb.old_logp)
    unclipped = ratio * mb.adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb.adv
    policy_loss = -(torch.minimum(unclipped, clipped)).mean()

    value_loss = F.mse_loss(out["value"], mb.returns)

    entropy = source_target_entropy(
        out["pair_logits"], mb.pair_mask, mb.source_mask,
        noop_logit_bias=mb.noop_logit_bias,
    ).mean()

    bc_loss = torch.zeros((), device=out["value"].device)
    bc_diag: dict[str, float] = {}
    if bc_mb is not None and bc_coef > 0:
        bc_out = policy(**bc_mb.feats)
        bc_loss, bc_diag = factorized_bc(
            pair_logits=bc_out["pair_logits"],
            pair_mask=bc_mb.pair_mask,
            source_mask=bc_mb.source_mask,
            expert_src_idx=bc_mb.expert_source_idx_or_noop,
            expert_target_bits=bc_mb.expert_target_bits,
            target_weight=bc_target_weight,
            noop_logit_bias=mb.noop_logit_bias,
        )

    total = (
        policy_loss
        + value_coef * value_loss
        - ent_coef * entropy
        + bc_coef * bc_loss
    )

    with torch.no_grad():
        # PER-COMPONENT KL: divide the summed-over-sub-actions logp diff by the
        # action's component count so target_kl is a sensible per-decision scale
        # (raw KL summed over ~30 targets made target_kl=0.01 fire after 1 epoch).
        approx_kl = ((mb.old_logp - new_logp) / n_terms.clamp_min(1.0)).mean().item()
        clip_frac = (
            ((ratio - 1.0).abs() > clip).float().mean().item()
        )

    diagnostics = {
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "bc_loss": float(bc_loss.detach().item()),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "ratio_mean": float(ratio.mean().detach().item()),
        **bc_diag,
    }
    return total, diagnostics
