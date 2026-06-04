"""PPO losses + single-target BC anchor.

action_logprob:
    Recomputes ``logp_pair + logp_frac`` at update time from the stored
    ``(tgt_idx, frac_raw)`` and the current policy outputs, matching
    :func:`agents.transformer_v2.ppo.sampler.sample_single_target` exactly.
    Each OWNED source row is one Categorical over its legal targets + the
    diagonal HOLD slot; launching rows add a LogitNormal frac term. Used for
    the PPO ratio (ratio == 1 for an unchanged policy).

single_target_bc:
    BC anchor matching the ``single_target_per_source_v1`` action contract +
    the pretrain ``_pair_single_target_ce`` labeling: per owned source row, a
    Categorical CE whose label is the diagonal (hold) when the expert held,
    else the expert's dominant target column.

ppo_minibatch_loss:
    Standard PPO clipped policy loss + MSE value loss + entropy +
    single-target BC. Returns ``(loss, diagnostics_dict)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical, Normal
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Minibatch dataclass                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class PPOMinibatch:
    """One PPO minibatch's stored tensors.

    Shapes (B = minibatch size, P = max planet slots):
      feats           dict of input tensors for the model forward
      pair_mask       (B, P, P) bool — exact off-diagonal legal pairs at sample time
      source_mask     (B, P) bool — owned/legal source rows (each one acted)
      tgt_idx         (B, P) long — chosen target col per source; s == hold/NOOP
      frac_raw        (B, P) float — clamped sigmoid launch fraction per launching source
      old_logp        (B,) float (sum of per-source Categorical + launching frac logprobs)
      adv             (B,) float (normalized GAE)
      returns         (B,) float (GAE + value)
      noop_logit_bias scalar; carried for shard compatibility (unused by the
                      single-target contract — the diagonal is the hold slot)
    """

    feats: dict[str, torch.Tensor]
    pair_mask: torch.Tensor
    source_mask: torch.Tensor
    tgt_idx: torch.Tensor
    frac_raw: torch.Tensor
    old_logp: torch.Tensor
    adv: torch.Tensor
    returns: torch.Tensor
    noop_logit_bias: float = 0.0

    @property
    def size(self) -> int:
        return int(self.tgt_idx.shape[0])


@dataclass
class BCMinibatch:
    """Single-target BC anchor minibatch (sampled from the supervised pair_cache).

    source_mask: (B, P) bool — learner-owned present source rows. Each one is
        a single-target Categorical CE example (hold vs which target).
    expert_tgt_idx: (B, P) long — per owned source row, the expert's label:
        the diagonal ``s`` when the expert held, else the dominant target
        column (largest ``pair_ships``). Mirrors pretrain
        ``_pair_single_target_ce``'s per-row label.
    """

    feats: dict[str, torch.Tensor]
    pair_mask: torch.Tensor                  # (B, P, P) bool — off-diagonal legal pairs
    source_mask: torch.Tensor                # (B, P) bool
    expert_tgt_idx: torch.Tensor             # (B, P) long — per-row hold(=s) / target col


# --------------------------------------------------------------------------- #
# Per-row valid-column mask (legal targets + diagonal hold on source rows)     #
# --------------------------------------------------------------------------- #
def _row_col_valid(
    pair_mask: torch.Tensor,         # (B, P, P) bool — off-diagonal legal pairs
    source_mask: torch.Tensor,       # (B, P) bool — owned/legal source rows
) -> torch.Tensor:
    """Return ``(B, P, P)`` bool: per source row, the columns that form its
    Categorical support — the legal off-diagonal targets PLUS the diagonal
    HOLD slot on owned source rows. Mirrors the sampler's ``row_valid``.
    """
    B, P, _ = pair_mask.shape
    device = pair_mask.device
    eye = torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0)  # (1,P,P)
    diag_valid = eye & source_mask.unsqueeze(2)                       # diagonal on source rows
    return pair_mask | diag_valid                                     # (B,P,P)


# --------------------------------------------------------------------------- #
# action_logprob                                                              #
# --------------------------------------------------------------------------- #
def action_logprob(
    pair_logits: torch.Tensor,        # (B, P, P)
    frac_loc: torch.Tensor,            # (B, P, P)
    sigma: torch.Tensor,               # scalar
    pair_mask: torch.Tensor,           # (B, P, P) bool — off-diagonal legal pairs
    source_mask: torch.Tensor,         # (B, P) bool — owned/legal source rows
    tgt_idx: torch.Tensor,             # (B, P) long — chosen target col per source (s == hold)
    frac_raw: torch.Tensor,            # (B, P) float — clamped sigmoid per launching source
    *,
    noop_logit_bias: float = 0.0,      # unused; kept for call-site compatibility
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the action's logprob under the CURRENT policy.

    Matches :func:`sample_single_target` exactly. Used for the PPO ratio.

    Per OWNED source row ``s`` we form a Categorical over ``pair_logits[b, s,
    :]`` masked to its legal targets + the diagonal HOLD slot, score
    ``tgt_idx[b, s]``, and add a LogitNormal frac term for launching rows
    (``tgt_idx != s``). Non-source rows contribute nothing.

    Returns ``(logp, n_terms)`` where ``logp`` is the summed action logprob and
    ``n_terms`` is the per-sample count of summed logprob components (one per
    acting source row + one per launching frac term). ``n_terms`` lets the
    caller normalize the KL to a PER-COMPONENT scale (the raw KL is a sum over
    all acting sources, so a scalar ``target_kl`` is otherwise too tight).
    """
    if pair_logits.dim() != 3:
        raise ValueError("expected pair_logits (B, P, P)")
    B, P, _ = pair_logits.shape
    device = pair_logits.device

    col_valid = _row_col_valid(pair_mask, source_mask)               # (B,P,P)
    row_logits = torch.where(
        col_valid, pair_logits, torch.full_like(pair_logits, float("-inf")),
    )                                                                # (B,P,P)

    # Per-row Categorical logprob on the chosen target column. Categorical over
    # the last dim treats each (B, P) row independently; rows with no valid
    # column (non-source rows) get a uniform -inf logit row, which would make
    # log_prob NaN — so we only KEEP the contribution on source rows below.
    # Guard: give non-source rows a finite dummy logit so the distribution is
    # constructible, then mask their logp to 0.
    safe_logits = torch.where(
        source_mask.unsqueeze(2),
        row_logits,
        torch.zeros_like(row_logits),    # uniform over P -> valid, masked out after
    )
    dist = Categorical(logits=safe_logits)
    logp_rows = dist.log_prob(tgt_idx)                               # (B,P)
    logp_rows = torch.where(source_mask, logp_rows, torch.zeros_like(logp_rows))
    logp_pair = logp_rows.sum(dim=1)                                 # (B,)

    # Frac LogitNormal on LAUNCHING rows only (tgt_idx != s and source row).
    arange_p = torch.arange(P, device=device).expand(B, P)          # (B,P)
    launch = source_mask & (tgt_idx != arange_p)                    # (B,P)
    chosen_loc = frac_loc.gather(2, tgt_idx.clamp(0, P - 1).unsqueeze(2)).squeeze(2)  # (B,P)
    frac_safe = frac_raw.clamp(1e-4, 1 - 1e-4)
    z = torch.logit(frac_safe)
    normal_logp = Normal(chosen_loc, sigma).log_prob(z)             # (B,P)
    jacobian = -torch.log(frac_safe) - torch.log1p(-frac_safe)      # (B,P)
    per_row_frac = normal_logp + jacobian
    per_row_frac = torch.where(launch, per_row_frac, torch.zeros_like(per_row_frac))
    logp_frac = per_row_frac.sum(dim=1)                             # (B,)

    # Per-sample component count: one per acting source row + one per launching
    # frac term. Used to normalize the PPO KL to a per-component scale.
    n_terms = source_mask.float().sum(dim=1) + launch.float().sum(dim=1)  # (B,)
    n_terms = n_terms.clamp_min(1.0)

    return logp_pair + logp_frac, n_terms


# --------------------------------------------------------------------------- #
# Entropy (for the entropy bonus term)                                        #
# --------------------------------------------------------------------------- #
def source_target_entropy(
    pair_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    noop_logit_bias: float = 0.0,      # unused; kept for call-site compatibility
) -> torch.Tensor:
    """Sum over owned source rows of the per-row Categorical entropy.

    Each owned source row is one Categorical over its legal targets + the
    diagonal HOLD slot, exactly as sampled. The frac logit-normal has fixed
    sigma, so its entropy is a state-independent constant and is dropped from
    the bonus.
    """
    B, P, _ = pair_logits.shape
    col_valid = _row_col_valid(pair_mask, source_mask)               # (B,P,P)
    row_logits = torch.where(
        col_valid, pair_logits, torch.full_like(pair_logits, float("-inf")),
    )
    safe_logits = torch.where(
        source_mask.unsqueeze(2),
        row_logits,
        torch.zeros_like(row_logits),
    )
    per_row_ent = Categorical(logits=safe_logits).entropy()          # (B,P)
    per_row_ent = torch.where(source_mask, per_row_ent, torch.zeros_like(per_row_ent))
    return per_row_ent.sum(dim=1)                                    # (B,)


# --------------------------------------------------------------------------- #
# Single-target BC anchor                                                      #
# --------------------------------------------------------------------------- #
def single_target_bc(
    pair_logits: torch.Tensor,         # (B, P, P)
    pair_mask: torch.Tensor,            # (B, P, P) bool — off-diagonal legal pairs
    source_mask: torch.Tensor,           # (B, P) bool — owned/legal source rows
    expert_tgt_idx: torch.Tensor,       # (B, P) long — per-row hold(=s) / target col
    *,
    launch_weight: float = 1.0,
    noop_logit_bias: float = 0.0,       # unused; kept for call-site compatibility
) -> tuple[torch.Tensor, dict[str, float]]:
    """BC anchor matching ``single_target_per_source_v1`` + pretrain
    ``_pair_single_target_ce``.

    For each owned source row, a Categorical CE over its legal targets + the
    diagonal HOLD slot, with the expert's per-row label (hold == ``s``, else
    the dominant target column). Optionally up-weight launching rows
    (``label != s``) so the usually-dominant hold rows don't drown the launch
    signal — the single-target analogue of the old per-cell BCE pos_weight.

    Returns ``(bc_loss, diagnostics)``.
    """
    B, P, _ = pair_logits.shape
    device = pair_logits.device

    col_valid = _row_col_valid(pair_mask, source_mask)               # (B,P,P)
    row_logits = pair_logits.masked_fill(~col_valid, float("-inf"))  # (B,P,P)

    src = source_mask                                                 # (B,P)
    if not bool(src.any()):
        z = torch.zeros((), device=device)
        return z, {
            "bc_loss": 0.0, "bc_acc": float("nan"),
            "bc_launch_acc": float("nan"), "bc_hold_acc": float("nan"),
        }

    rl = row_logits[src]                                             # (N,P)
    lab = expert_tgt_idx[src]                                        # (N,)
    bs = torch.nonzero(src, as_tuple=False)                         # (N,2) [b,s]
    src_of_row = bs[:, 1]                                            # (N,)
    launch_sel = lab != src_of_row                                  # (N,) launch vs hold
    hold_sel = ~launch_sel

    if launch_weight != 1.0:
        per_row = F.cross_entropy(rl, lab, reduction="none")        # (N,)
        w = torch.where(
            launch_sel,
            torch.full_like(per_row, float(launch_weight)),
            torch.ones_like(per_row),
        )
        bc_loss = (per_row * w).sum() / w.sum().clamp(min=1.0)
    else:
        bc_loss = F.cross_entropy(rl, lab)

    with torch.no_grad():
        pred = rl.argmax(dim=1)
        bc_acc = (pred == lab).float().mean()
        n_launch = int(launch_sel.sum())
        n_hold = int(hold_sel.sum())
        bc_launch_acc = (
            (pred[launch_sel] == lab[launch_sel]).float().mean()
            if n_launch else torch.tensor(float("nan"), device=device)
        )
        bc_hold_acc = (
            (pred[hold_sel] == lab[hold_sel]).float().mean()
            if n_hold else torch.tensor(float("nan"), device=device)
        )

    diagnostics = {
        "bc_loss": float(bc_loss.detach().item()),
        "bc_acc": float(bc_acc.detach().item()),
        "bc_launch_acc": float(bc_launch_acc.detach().item()),
        "bc_hold_acc": float(bc_hold_acc.detach().item()),
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

    # NaN-guard the actor head outputs. Degenerate samples — e.g. a learner
    # eliminated mid-game (0 planets → all-masked perception/attention) — can
    # yield NaN ``pair_logits``/``frac_loc``. Those frames have no acting
    # source rows, so their pair/frac logp is masked to 0 by ``source_mask``/
    # ``launch`` downstream; but ``Normal(loc=NaN)`` validates and RAISES
    # before that masking runs. Zero out the non-finite entries.
    _n_guarded = 0
    for _k in ("pair_logits", "frac_loc"):
        _v = out.get(_k)
        if _v is not None and not torch.isfinite(_v).all():
            _n_guarded += int((~torch.isfinite(_v)).any(dim=tuple(range(1, _v.dim()))).sum())
            out[_k] = torch.nan_to_num(_v, nan=0.0, posinf=0.0, neginf=0.0)

    new_logp, n_terms = action_logprob(
        pair_logits=out["pair_logits"],
        frac_loc=out["frac_loc"],
        sigma=sigma,
        pair_mask=mb.pair_mask,
        source_mask=mb.source_mask,
        tgt_idx=mb.tgt_idx,
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
        bc_loss, bc_diag = single_target_bc(
            pair_logits=bc_out["pair_logits"],
            pair_mask=bc_mb.pair_mask,
            source_mask=bc_mb.source_mask,
            expert_tgt_idx=bc_mb.expert_tgt_idx,
            launch_weight=bc_target_weight,
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
        # action's component count so target_kl is a sensible per-decision scale.
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
        "nan_guarded": _n_guarded,
        **bc_diag,
    }
    return total, diagnostics
