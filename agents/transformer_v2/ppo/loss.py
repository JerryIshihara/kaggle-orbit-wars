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
      old_logp        (B,) float — stored action logprob (sum of sub-action logprobs)
      adv             (B,) float (normalized GAE)
      returns         (B,) float (GAE + value)
      noop_logit_bias scalar; carried for shard compatibility (unused by either
                      contract — the diagonal is the hold/self slot)

    Action fields — ONE contract's tensors are populated, the other left None.
    The loss detects which by field presence (``select_mask is not None`` ->
    ``bernoulli_select_multinomial_alloc_v2``; else ``single_target_per_source_v1``).

      single_target_per_source_v1:
        tgt_idx       (B, P) long — chosen target col per source; s == hold/NOOP
        frac_raw      (B, P) float — clamped sigmoid launch fraction per launching source
      bernoulli_select_multinomial_alloc_v2:
        select_mask   (B, P, P) bool — fired off-diagonal targets per owned source
        alloc_counts  (B, P, P) long — ships routed to each fired target
        self_counts   (B, P) long — ships held on each owned source (self category)
    """

    feats: dict[str, torch.Tensor]
    pair_mask: torch.Tensor
    source_mask: torch.Tensor
    old_logp: torch.Tensor
    adv: torch.Tensor
    returns: torch.Tensor
    # single_target_per_source_v1 action fields (None under the multi-target contract)
    tgt_idx: torch.Tensor | None = None
    frac_raw: torch.Tensor | None = None
    # bernoulli_select_multinomial_alloc_v2 action fields (None under single-target)
    select_mask: torch.Tensor | None = None
    alloc_counts: torch.Tensor | None = None
    self_counts: torch.Tensor | None = None
    # bounded_k_select_multinomial_alloc_v3 action fields (None otherwise)
    select_counts: torch.Tensor | None = None   # (B, P, P+1) long — draws incl. self token
    alloc_extras: torch.Tensor | None = None    # (B, P, P) long — extras beyond the floor
    self_extras: torch.Tensor | None = None     # (B, P) long — extras held on acting rows
    noop_logit_bias: float = 0.0

    @property
    def is_bounded_k(self) -> bool:
        return self.select_counts is not None

    @property
    def is_multi_target(self) -> bool:
        return self.select_mask is not None

    @property
    def size(self) -> int:
        if self.select_counts is not None:
            return int(self.select_counts.shape[0])
        if self.select_mask is not None:
            return int(self.select_mask.shape[0])
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


# =========================================================================== #
# bernoulli_select_multinomial_alloc_v2 logprob + entropy                     #
# =========================================================================== #
def action_logprob_multi(
    pair_logits: torch.Tensor,        # (B, P, P)
    frac_loc: torch.Tensor,            # (B, P, P)
    *,
    pair_mask: torch.Tensor,           # (B, P, P) bool — legal off-diagonal targets
    source_mask: torch.Tensor,         # (B, P) bool — owned/legal source rows
    act,                               # MultiTargetAction-shaped minibatch fields
    temperature: float = 1.0,
    select_logit_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the ``bernoulli_select_multinomial_alloc_v2`` logprob under the
    CURRENT policy. Recomputes the SAME quantity the sampler stored, so the PPO
    ratio is 1 for an unchanged policy.

    ``act`` is anything carrying ``select_mask (B,P,P) bool``, ``alloc_counts
    (B,P,P) long`` and ``self_counts (B,P) long`` (a ``MultiTargetAction`` or
    the minibatch itself).

    selection = ``-BCE_with_logits(pair_logits[s, legal]/tau - select_logit_bias,
        fired_bits, reduction='sum')`` summed over legal cells of owned source
        rows. ``select_logit_bias`` MUST equal the value
        :func:`agents.transformer_v2.ppo.sampler.sample_multi_target` sampled
        with (it shifts only the selection Bernoulli logit) or the PPO ratio
        desyncs.
    allocation = ``Σ counts · log_softmax([frac_loc[s, F_s], frac_loc[s, s]])``
        over ``F_s ∪ self`` per owned source (UNCHANGED by the bias; v2 — the
        self/HOLD logit is the frac head's diagonal, not ``pair_logits[s, s]``).

    Returns ``(logp, n_terms)`` where ``n_terms`` counts the legal Bernoulli
    cells across owned rows plus one per acting source (its multinomial),
    matching the sampler — used to normalize the PPO KL per-component.
    """
    if pair_logits.dim() != 3:
        raise ValueError("expected pair_logits (B, P, P)")
    B, P, _ = pair_logits.shape
    device = pair_logits.device
    tau = float(temperature)
    sel_bias = float(select_logit_bias)

    select_mask = act.select_mask.bool()                            # (B,P,P)
    alloc_counts = act.alloc_counts.to(pair_logits.dtype)           # (B,P,P)
    self_counts = act.self_counts.to(pair_logits.dtype)             # (B,P)

    # ----- Stage 1: selection (Bernoulli over ALL legal cells) -----
    # Legal cells are the off-diagonal pair_mask cells on owned source rows.
    # The selection logit carries the SAME bias the sampler used (alloc below
    # uses raw frac_loc / diagonal logits, so it is untouched by the bias).
    legal = pair_mask & source_mask.unsqueeze(2)                    # (B,P,P)
    sel_logits = pair_logits / tau - sel_bias                       # (B,P,P)
    fired_bits = select_mask.to(pair_logits.dtype)                  # (B,P,P) in {0,1}
    # Per-cell Bernoulli log-likelihood = -BCE_with_logits; mask to legal cells.
    bce = F.binary_cross_entropy_with_logits(
        sel_logits, fired_bits, reduction="none",
    )                                                               # (B,P,P)
    bce = torch.where(legal, bce, torch.zeros_like(bce))
    logp_select = -bce.sum(dim=(1, 2))                              # (B,)

    # ----- Stage 2: allocation (Σ counts · log softmax over F_s ∪ self) -----
    # Build a per-row category set [all legal targets ... , self] with masked
    # log-softmax. The not-fired legal targets carry count 0, so including them
    # in the softmax support is WRONG — the sampler's softmax is over F_s ∪ self
    # only. So we mask the alloc logits to the FIRED targets + the diagonal self.
    # v2: the self/HOLD logit IS the frac head's diagonal, so ``frac_loc / tau``
    # already carries every category — fired targets off-diagonal, self on the
    # diagonal. Support per (b, s) row: fired targets (select_mask) OR the
    # diagonal self; one masked log-softmax covers F_s ∪ self in one shot.
    alloc_logit_mat = frac_loc / tau                                # (B,P,P)
    eye = torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0)  # (1,P,P)
    support = select_mask | (eye & source_mask.unsqueeze(2))        # (B,P,P)
    masked_logits = torch.where(
        support, alloc_logit_mat, torch.full_like(alloc_logit_mat, float("-inf")),
    )
    # Rows with empty support (non-owned, or owned with zero fired AND not the
    # diagonal — can't happen since self is always in support on owned rows)
    # would be all -inf -> NaN log-softmax. Guard owned rows always have self;
    # give the remaining all--inf rows a finite dummy, masked out by counts==0.
    row_has_support = support.any(dim=2, keepdim=True)              # (B,P,1)
    masked_logits = torch.where(
        row_has_support, masked_logits, torch.zeros_like(masked_logits),
    )
    log_probs = torch.log_softmax(masked_logits, dim=2)            # (B,P,P)
    log_probs = torch.where(
        torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs),
    )
    # counts on targets (alloc_counts) live off-diagonal; counts on self live in
    # self_counts and must be scored against the diagonal log-prob.
    counts_mat = alloc_counts.clone()
    counts_mat = counts_mat - counts_mat * eye.to(counts_mat.dtype)  # zero any diagonal
    counts_mat = counts_mat + (self_counts.unsqueeze(2) * eye.to(counts_mat.dtype))
    logp_alloc = (counts_mat * log_probs).sum(dim=(1, 2))          # (B,)

    # ----- per-sample component count (matches the sampler's n_terms) -----
    n_bernoulli = legal.float().sum(dim=(1, 2))                    # (B,) legal cells
    n_alloc = source_mask.float().sum(dim=1)                       # (B,) one per acting source
    n_terms = (n_bernoulli + n_alloc).clamp_min(1.0)              # (B,)

    return logp_select + logp_alloc, n_terms


def multi_target_entropy(
    pair_logits: torch.Tensor,        # (B, P, P)
    frac_loc: torch.Tensor,            # (B, P, P)
    *,
    pair_mask: torch.Tensor,           # (B, P, P) bool — legal off-diagonal targets
    source_mask: torch.Tensor,         # (B, P) bool — owned/legal source rows
    act,                               # MultiTargetAction-shaped minibatch fields
    temperature: float = 1.0,
    select_logit_bias: float = 0.0,
) -> torch.Tensor:
    """Entropy bonus for ``bernoulli_select_multinomial_alloc_v2``.

    ``Σ_{legal cells} Bernoulli-entropy(pair_logits)`` (selection) ``+ Σ_{owned
    sources} categorical-entropy(softmax(alloc over F_s ∪ self))`` (allocation).
    The allocation categorical's support is the SAMPLED ``F_s ∪ self`` (read
    from ``act.select_mask``), so the entropy matches the per-sample multinomial
    the sampler drew. ``select_logit_bias`` shifts the selection Bernoulli logit
    (same value as the sampler) so ``ent_select`` reflects the biased fire
    probability; the allocation entropy is UNCHANGED by the bias. Returns
    ``(B,)``.
    """
    B, P, _ = pair_logits.shape
    device = pair_logits.device
    tau = float(temperature)
    sel_bias = float(select_logit_bias)

    # ---- selection: sum of per-legal-cell Bernoulli entropy ----
    legal = pair_mask & source_mask.unsqueeze(2)                    # (B,P,P)
    sel_logits = pair_logits / tau - sel_bias
    pr = torch.sigmoid(sel_logits)
    # H(Bernoulli) = -p·log p - (1-p)·log(1-p); use logsigmoid for stability.
    log_p = F.logsigmoid(sel_logits)
    log_1mp = F.logsigmoid(-sel_logits)
    ber_ent = -(pr * log_p + (1.0 - pr) * log_1mp)                 # (B,P,P)
    ber_ent = torch.where(legal, ber_ent, torch.zeros_like(ber_ent))
    ent_select = ber_ent.sum(dim=(1, 2))                           # (B,)

    # ---- allocation: categorical entropy over sampled F_s ∪ self per owned row ----
    select_mask = act.select_mask.bool()
    eye = torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0)
    # v2: self/HOLD logit = frac head's diagonal; frac_loc already carries it.
    alloc_logit_mat = frac_loc / tau                               # (B,P,P)
    support = select_mask | (eye & source_mask.unsqueeze(2))       # (B,P,P)
    masked_logits = torch.where(
        support, alloc_logit_mat, torch.full_like(alloc_logit_mat, float("-inf")),
    )
    row_has_support = support.any(dim=2, keepdim=True)
    masked_logits = torch.where(
        row_has_support, masked_logits, torch.zeros_like(masked_logits),
    )
    log_probs = torch.log_softmax(masked_logits, dim=2)
    p_probs = log_probs.exp()
    cat_ent = -(p_probs * torch.where(
        torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs),
    )).sum(dim=2)                                                  # (B,P)
    cat_ent = torch.where(source_mask, cat_ent, torch.zeros_like(cat_ent))
    ent_alloc = cat_ent.sum(dim=1)                                 # (B,)

    return ent_select + ent_alloc


# --------------------------------------------------------------------------- #
# bounded_k_select_multinomial_alloc_v3 logprob + entropy                      #
# --------------------------------------------------------------------------- #
def _bounded_k_log_softmaxes(
    pair_logits: torch.Tensor,        # (B, P, P)
    frac_loc: torch.Tensor,            # (B, P, P)
    pair_mask: torch.Tensor,           # (B, P, P) bool
    source_mask: torch.Tensor,         # (B, P) bool
    select_counts: torch.Tensor,       # (B, P, P+1) long
    tau: float,
    self_bias: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared select/alloc masked log-softmaxes for the v3 contract.

    Returns ``(logp_sel (B,P,P+1), logp_al (B,P,P+1), drew (B,P), has_fired
    (B,P))``. Non-finite entries are zeroed — counts are 0 there, so they
    contribute nothing to ``sum counts*logp`` and nothing to entropy terms
    that multiply by the (zero) probability.
    """
    B, P, _ = pair_logits.shape
    legal = pair_mask & source_mask.unsqueeze(2)                    # (B,P,P)
    drew = select_counts.sum(dim=-1) > 0                            # (B,P) rows that sampled
    neg_inf = float("-inf")

    sel_self = pair_logits.diagonal(dim1=1, dim2=2) + self_bias     # (B,P)
    sel = torch.cat([pair_logits, sel_self.unsqueeze(-1)], dim=-1) / tau
    sel_support = torch.cat([legal, drew.unsqueeze(-1)], dim=-1)    # self on drawing rows
    sel_m = torch.where(sel_support, sel, torch.full_like(sel, neg_inf))
    row_ok = sel_support.any(dim=-1, keepdim=True)
    sel_m = torch.where(row_ok, sel_m, torch.zeros_like(sel_m))
    logp_sel = torch.log_softmax(sel_m, dim=-1)
    logp_sel = torch.where(torch.isfinite(logp_sel), logp_sel,
                           torch.zeros_like(logp_sel))

    fired = select_counts[..., :P] >= 1                             # (B,P,P)
    has_fired = fired.any(dim=-1)                                   # (B,P)
    al_self = frac_loc.diagonal(dim1=1, dim2=2)                     # (B,P)
    al = torch.cat([frac_loc, al_self.unsqueeze(-1)], dim=-1) / tau
    al_support = torch.cat([fired, has_fired.unsqueeze(-1)], dim=-1)
    al_m = torch.where(al_support, al, torch.full_like(al, neg_inf))
    row_ok2 = al_support.any(dim=-1, keepdim=True)
    al_m = torch.where(row_ok2, al_m, torch.zeros_like(al_m))
    logp_al = torch.log_softmax(al_m, dim=-1)
    logp_al = torch.where(torch.isfinite(logp_al), logp_al,
                          torch.zeros_like(logp_al))
    return logp_sel, logp_al, drew, has_fired


def action_logprob_bounded_k(
    pair_logits: torch.Tensor,        # (B, P, P)
    frac_loc: torch.Tensor,            # (B, P, P)
    *,
    pair_mask: torch.Tensor,           # (B, P, P) bool
    source_mask: torch.Tensor,         # (B, P) bool
    act,                               # carries select_counts / alloc_extras / self_extras
    temperature: float = 1.0,
    self_logit_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the ``bounded_k_select_multinomial_alloc_v3`` logprob.

    selection = ``sum draws*log_softmax([pair_logits[s, legal],
        pair_logits[s, s]+bias])`` per drawing row;
    allocation = ``sum extras*log_softmax([frac_loc[s, F_s], frac_loc[s, s]])``
        per row with >= 1 fired target.
    Both coefficient-free, matching :func:`sampler.sample_bounded_k` exactly.
    Returns ``(logp (B,), n_terms (B,))`` — n_terms = drawing rows + acting
    rows, matching the sampler's count.
    """
    if pair_logits.dim() != 3:
        raise ValueError("expected pair_logits (B, P, P)")
    tau = float(temperature)
    select_counts = act.select_counts.to(pair_logits.dtype)
    logp_sel, logp_al, drew, has_fired = _bounded_k_log_softmaxes(
        pair_logits, frac_loc, pair_mask, source_mask,
        act.select_counts, tau, float(self_logit_bias),
    )
    logp_select = (select_counts * logp_sel).sum(dim=(1, 2))

    extras_full = torch.cat([
        act.alloc_extras.to(pair_logits.dtype),
        act.self_extras.to(pair_logits.dtype).unsqueeze(-1),
    ], dim=-1)                                                      # (B,P,P+1)
    logp_alloc = (extras_full * logp_al).sum(dim=(1, 2))

    n_terms = (drew.float().sum(dim=1)
               + has_fired.float().sum(dim=1)).clamp_min(1.0)
    return logp_select + logp_alloc, n_terms


def bounded_k_entropy(
    pair_logits: torch.Tensor,
    frac_loc: torch.Tensor,
    *,
    pair_mask: torch.Tensor,
    source_mask: torch.Tensor,
    act,
    temperature: float = 1.0,
    self_logit_bias: float = 0.0,
) -> torch.Tensor:
    """Entropy bonus for v3: categorical entropy of the select softmax per
    drawing row + categorical entropy of the alloc softmax per acting row
    (underlying distributions, not count-scaled — same convention as v2)."""
    tau = float(temperature)
    logp_sel, logp_al, drew, has_fired = _bounded_k_log_softmaxes(
        pair_logits, frac_loc, pair_mask, source_mask,
        act.select_counts, tau, float(self_logit_bias),
    )
    ent_sel = -(logp_sel.exp() * logp_sel).sum(dim=-1)              # (B,P)
    ent_sel = torch.where(drew, ent_sel, torch.zeros_like(ent_sel))
    ent_al = -(logp_al.exp() * logp_al).sum(dim=-1)                 # (B,P)
    ent_al = torch.where(has_fired, ent_al, torch.zeros_like(ent_al))
    return ent_sel.sum(dim=1) + ent_al.sum(dim=1)


@torch.no_grad()
def bounded_k_ent_breakdown(
    pair_logits: torch.Tensor,
    frac_loc: torch.Tensor,
    *,
    pair_mask: torch.Tensor,
    source_mask: torch.Tensor,
    act,
    temperature: float = 1.0,
    self_logit_bias: float = 0.0,
    spike_p: float = 0.10,
) -> dict[str, float]:
    """Per-stage entropy decomposition for the v3 contract (diagnostic).

    Splits the combined entropy bonus into its two stages and characterizes
    each distribution's SHAPE on this minibatch:

      ent/sel_ent   mean select-softmax entropy per drawing row (nats)
      ent/sel_ppl   exp(H) — effective # of select modes (targets + self)
      ent/sel_top1  mean max-probability token
      ent/sel_spikes mean # tokens with p > ``spike_p`` (the "bumps")
      ent/sel_pself mean probability mass on the self token
      ent/al_*      same four for the allocation softmax over [fired, self]
      ent/al_hold   mean HOLD share of the alloc softmax

    Computed on the SAMPLED supports (drawing rows / acting rows), matching
    bounded_k_entropy's terms, so sel_ent/al_ent average to the same scale
    the entropy bonus sums.
    """
    tau = float(temperature)
    B, P, _ = pair_logits.shape
    logp_sel, logp_al, drew, has_fired = _bounded_k_log_softmaxes(
        pair_logits, frac_loc, pair_mask, source_mask,
        act.select_counts, tau, float(self_logit_bias),
    )
    # Rebuild the true supports: _bounded_k_log_softmaxes zeroes masked
    # logp entries, which exp() to a bogus p=1 — probabilities must be
    # zeroed outside the support before any shape statistic.
    legal = pair_mask & source_mask.unsqueeze(2)
    sel_support = torch.cat([legal, drew.unsqueeze(-1)], dim=-1)
    fired = act.select_counts[..., :P] >= 1
    al_support = torch.cat([fired, has_fired.unsqueeze(-1)], dim=-1)
    out: dict[str, float] = {}

    def _stats(logp, support, rows, prefix):
        if not bool(rows.any()):
            for k in ("ent", "ppl", "top1", "spikes"):
                out[f"ent/{prefix}_{k}"] = float("nan")
            return None
        pr = torch.where(support, logp.exp(), torch.zeros_like(logp))
        ent = -(pr * torch.where(support, logp, torch.zeros_like(logp))
                ).sum(dim=-1)                              # (B,P)
        ent_r = ent[rows]
        pr_r = pr[rows]                                    # (N,P+1)
        out[f"ent/{prefix}_ent"] = float(ent_r.mean())
        out[f"ent/{prefix}_ppl"] = float(ent_r.exp().mean())
        out[f"ent/{prefix}_top1"] = float(pr_r.max(dim=-1).values.mean())
        out[f"ent/{prefix}_spikes"] = float(
            (pr_r > spike_p).sum(dim=-1).float().mean())
        return pr_r

    pr_sel = _stats(logp_sel, sel_support, drew, "sel")
    if pr_sel is not None:
        out["ent/sel_pself"] = float(pr_sel[:, -1].mean())
    pr_al = _stats(logp_al, al_support, has_fired, "al")
    if pr_al is not None:
        out["ent/al_hold"] = float(pr_al[:, -1].mean())
    return out


# --------------------------------------------------------------------------- #
# Entropy SHAPE diagnostics (flat vs. a few bumps)                            #
# --------------------------------------------------------------------------- #
def source_target_dist_shape(
    pair_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    topk: int = 8,
) -> dict[str, float]:
    """SHAPE of the per-row target Categorical, averaged over owned source rows.

    Tells a *flat* high-entropy distribution (near-uniform over many legal
    targets — the pretrained structure collapsing) apart from one with a few
    *bumps* (the policy hedging between a handful of good targets — benign).
    Reuses the exact masked-Categorical construction of
    :func:`source_target_entropy`, so the entropy matches the bonus term.

    Diagnostic only (``no_grad``). Keys (all means over owned rows):
      ``ent/norm``        H / log(K)   ~1.0 = flat,  <1 = peaked
      ``ent/perplexity``  exp(H)       effective # targets (Shannon)
      ``ent/collision``   1 / Σ p²     effective # bumps   (Rényi-2)
      ``ent/K``           # legal options (HOLD + legal targets) per row
      ``ent/top1``        top-1 prob
      ``ent/top3``        top-3 cumulative prob
      ``ent/p0..p{k-1}``  sorted-descending mean prob profile (eyeball the shape)
    """
    B, P, _ = pair_logits.shape
    col_valid = _row_col_valid(pair_mask, source_mask)               # (B,P,P)
    row_logits = torch.where(
        col_valid, pair_logits, torch.full_like(pair_logits, float("-inf")),
    )
    safe_logits = torch.where(
        source_mask.unsqueeze(2), row_logits, torch.zeros_like(row_logits),
    )
    sm = source_mask
    with torch.no_grad():
        logp = torch.log_softmax(safe_logits, dim=-1)                # (B,P,P)
        p = logp.exp()
        # entropy in nats; 0·(-inf) on illegal cols -> 0 via the finite guard
        H = -(p * torch.where(torch.isfinite(logp), logp,
                              torch.zeros_like(logp))).sum(-1)        # (B,P)
        K = col_valid.sum(-1).float()                                # (B,P) legal opts
        collision = 1.0 / p.pow(2).sum(-1).clamp_min(1e-12)          # (B,P) Rényi-2
        kk = max(1, min(topk, P))
        top = p.topk(kk, dim=-1).values                             # (B,P,kk) desc
        top1 = top[..., 0]
        top3 = top[..., :min(3, kk)].sum(-1)

        n_rows = sm.float().sum().clamp_min(1.0)

        def _m(x: torch.Tensor) -> float:
            return float((torch.where(sm, x, torch.zeros_like(x)).sum()
                          / n_rows).item())

        # H/log(K) is undefined at K=1 (forced HOLD); average only over rows
        # that actually have a choice (>=2 legal options).
        multi = sm & (col_valid.sum(-1) >= 2)
        n_multi = multi.float().sum().clamp_min(1.0)
        norm = H / K.clamp_min(2).log()
        ent_norm = float((torch.where(multi, norm, torch.zeros_like(norm)).sum()
                          / n_multi).item())

        profile = (top * sm.unsqueeze(-1).float()).sum(dim=(0, 1)) / n_rows  # (kk,)

        out = {
            "ent/norm": ent_norm,
            "ent/perplexity": _m(H.exp()),
            "ent/collision": _m(collision),
            "ent/K": _m(K),
            "ent/top1": _m(top1),
            "ent/top3": _m(top3),
        }
        for i, v in enumerate(profile.tolist()):
            out[f"ent/p{i}"] = float(v)
    return out


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
    collect_shape: bool = False,
    select_logit_bias: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One PPO minibatch loss.

    Returns ``(total_loss, diagnostics)``. ``approx_kl`` is included in the
    diagnostics so the outer loop can early-stop the PPO epoch when
    ``approx_kl > 1.5 * target_kl``.

    ``select_logit_bias`` (multi-target contract only) MUST equal the value the
    rollout sampler used so the recomputed selection logprob/entropy match the
    stored action (else the PPO ratio silently desyncs). Ignored under the
    single-target contract.
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

    # Contract switch: detect by action-field presence on the minibatch.
    # ``select_counts is not None`` -> bounded_k_select_multinomial_alloc_v3;
    # ``select_mask is not None`` -> bernoulli_select_multinomial_alloc_v2.
    if getattr(mb, "select_counts", None) is not None:
        # v3 reuses the select_logit_bias knob as the SELF-token bias (same
        # sign convention: positive -> fires less).
        new_logp, n_terms = action_logprob_bounded_k(
            pair_logits=out["pair_logits"],
            frac_loc=out["frac_loc"],
            pair_mask=mb.pair_mask,
            source_mask=mb.source_mask,
            act=mb,
            self_logit_bias=select_logit_bias,
        )
    elif getattr(mb, "select_mask", None) is not None:
        new_logp, n_terms = action_logprob_multi(
            pair_logits=out["pair_logits"],
            frac_loc=out["frac_loc"],
            pair_mask=mb.pair_mask,
            source_mask=mb.source_mask,
            act=mb,
            select_logit_bias=select_logit_bias,
        )
    else:
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

    if getattr(mb, "select_counts", None) is not None:
        entropy = (bounded_k_entropy(
            out["pair_logits"], out["frac_loc"],
            pair_mask=mb.pair_mask, source_mask=mb.source_mask, act=mb,
            self_logit_bias=select_logit_bias,
        ) / n_terms).mean()
    elif getattr(mb, "select_mask", None) is not None:
        # Per-component normalization (÷ n_terms), matching the KL normalization,
        # so ent_coef has a comparable effect to the single-target contract. The
        # raw multi-target entropy is a sum over ~all legal Bernoulli cells (~40),
        # which otherwise lets the entropy bonus dominate the loss (ent_coef·40).
        entropy = (multi_target_entropy(
            out["pair_logits"], out["frac_loc"],
            pair_mask=mb.pair_mask, source_mask=mb.source_mask, act=mb,
            select_logit_bias=select_logit_bias,
        ) / n_terms).mean()
    else:
        entropy = source_target_entropy(
            out["pair_logits"], mb.pair_mask, mb.source_mask,
            noop_logit_bias=mb.noop_logit_bias,
        ).mean()

    # Diagnostic: is a high entropy FLAT (uniform collapse) or a few BUMPS
    # (benign hedging)? Cheap, no_grad; gated so it runs on a subsample only.
    # Wrapped: a diagnostic failure must never kill a long training run.
    shape_diag: dict[str, float] = {}
    if collect_shape and getattr(mb, "select_counts", None) is not None:
        try:
            shape_diag = bounded_k_ent_breakdown(
                out["pair_logits"], out["frac_loc"],
                pair_mask=mb.pair_mask, source_mask=mb.source_mask, act=mb,
                self_logit_bias=select_logit_bias,
            )
        except Exception:
            shape_diag = {}
    elif collect_shape and getattr(mb, "select_mask", None) is None:
        # The per-row Categorical shape diagnostic is single-target-specific
        # (the multi-target contract has no single per-source Categorical).
        try:
            shape_diag = source_target_dist_shape(
                out["pair_logits"], mb.pair_mask, mb.source_mask)
        except Exception:
            shape_diag = {}

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
        **shape_diag,
        **bc_diag,
    }
    return total, diagnostics
