"""Action sampling and env projection for ``single_target_per_source_v1``.

This matches the pretrained actor (``entity_encoder._pair_single_target_ce``)
and the runner ``single_target`` inference mode. One snapshot's actor decisions:

  1. Per OWNED source row ``s`` (``source_mask[s]`` true) we draw ONE
     ``Categorical`` over the P target columns INCLUDING the diagonal
     ``[s, s]`` (the HOLD/NOOP slot). The valid columns of row ``s`` are
     ``pair_mask[s]`` with index ``s`` forced True.
       * diagonal wins (``tgt_idx[s] == s``)  -> planet ``s`` HOLDS (NOOP),
       * column ``t != s``                    -> launch ``s -> t``.
     Sources act INDEPENDENTLY (one Categorical each), unlike the old
     ``source_multi_target_v1`` single-source + per-target Bernoulli.
  2. ``frac_raw[s]`` — for each LAUNCHING source, ``LogitNormal(loc=
     pair_frac[s, tgt_idx[s]], sigma)``. sigma is fixed (CLI hyperparameter).

The stored ``frac_raw`` is the numerically-clamped sigmoid; the launch-side
ship rounding is applied only at env projection. PPO recomputes the logprob
from the stored value, so the two must agree exactly (ratio == 1 for an
unchanged policy).

Ship budgeting is PER-SOURCE: source ``s`` launches ``round(frac_raw[s] *
surplus_of_s)`` ships (gated by ``min_launch``), mirroring the runner
``single_target`` semantics. There is NO shared-budget split across sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Multinomial, Normal


# --------------------------------------------------------------------------- #
# Legality                                                                    #
# --------------------------------------------------------------------------- #
def legality_masks(
    planet_owner: torch.Tensor,      # (P,) long, learner-relative; 0 = me
    surplus: torch.Tensor,            # (P,) float, ships available to launch
    planet_exists: torch.Tensor,      # (P,) bool, real (vs padded) planet
    *,
    min_launch: int,
    learner_owner_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pair_mask, source_mask)`` for one snapshot.

    pair_mask (P, P) bool: a (source, target) cell is valid when source is
        learner-owned with surplus >= min_launch, target exists, and s != t.

    source_mask (P,) bool: row reduction of pair_mask (does this source
        have any valid target?). Used to select which rows act: every such
        source draws one Categorical over its legal targets + the diagonal
        HOLD slot.
    """
    if planet_owner.dim() != 1 or surplus.dim() != 1 or planet_exists.dim() != 1:
        raise ValueError("planet_owner, surplus, planet_exists must be 1-D")
    p = int(planet_exists.shape[0])
    own = planet_owner == learner_owner_id                   # (P,)
    has_ships = surplus >= float(min_launch)                  # (P,)
    src_ok = own & has_ships & planet_exists                  # (P,)
    tgt_ok = planet_exists                                     # (P,)
    pair_ok = src_ok.unsqueeze(1) & tgt_ok.unsqueeze(0)       # (P, P)
    eye = torch.eye(p, dtype=torch.bool, device=pair_ok.device)
    pair_ok = pair_ok & ~eye
    source_mask = pair_ok.any(dim=1)                          # (P,)
    return pair_ok, source_mask


# --------------------------------------------------------------------------- #
# Action dataclass                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    """One snapshot's sampled action + per-component logprobs.

    ``tgt_idx (P,) long``: per source the chosen target column. ``tgt_idx[s]
        == s`` means HOLD/NOOP; ``tgt_idx[s] != s`` means launch ``s ->
        tgt_idx[s]``. Non-acting rows (``~source_mask``) stay at ``s`` (hold)
        and contribute no logprob.
    ``frac_raw (P,) float``: clamped sigmoid launch fraction per launching
        source; zeros on hold / non-acting rows.
    """

    tgt_idx: torch.Tensor           # (P,) long — chosen target col per source (s == hold)
    frac_raw: torch.Tensor          # (P,) float — clamped sigmoid for launching sources
    logprob: torch.Tensor           # scalar — logp_pair + logp_frac
    logprob_pair: torch.Tensor      # scalar — sum of per-source Categorical logprobs
    logprob_frac: torch.Tensor      # scalar — sum of launching-source frac logprobs
    n_launch: int                   # number of launching sources (tgt != s)
    diagnostics: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Frac logit-normal logprob (shared with loss.action_logprob)                  #
# --------------------------------------------------------------------------- #
def _frac_logp(loc: torch.Tensor, sigma: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """log p(sigmoid(z)) for raw == clamped sigmoid(z), z ~ Normal(loc, sigma)."""
    raw = raw.clamp(1e-4, 1 - 1e-4)
    return (
        Normal(loc, sigma).log_prob(torch.logit(raw))
        - torch.log(raw)
        - torch.log1p(-raw)
    )


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #
def sample_single_target(
    pair_logits: torch.Tensor,        # (P, P) — actor logits for one snapshot
    frac_loc: torch.Tensor,            # (P, P) — actor frac means (pair_frac raw logit)
    sigma: torch.Tensor | float,       # () scalar — fixed frac stddev
    *,
    pair_mask: torch.Tensor,           # (P, P) bool
    source_mask: torch.Tensor,         # (P,) bool
    noop_logit_bias: float = 0.0,      # unused; kept for call-site compatibility
) -> Action:
    """Draw one ``single_target_per_source_v1`` action.

    Each owned source (``source_mask[s]``) draws one Categorical over its
    legal targets PLUS the diagonal HOLD slot. The diagonal logit is the
    actor's own ``pair_logits[s, s]`` (the calibrated hold logit a
    single-target-trained model produces); ``noop_logit_bias`` is accepted
    only for backwards-compatible call sites and is NOT applied (the contract
    has no separate NOOP logit).
    """
    if pair_logits.dim() != 2:
        raise ValueError("sample is per-snapshot; expected pair_logits (P, P)")
    p = pair_logits.shape[0]
    device = pair_logits.device
    sigma_t = sigma if torch.is_tensor(sigma) else torch.tensor(float(sigma), device=device)

    neg_inf = torch.full_like(pair_logits, float("-inf"))
    diag = torch.arange(p, device=device)

    tgt_idx = diag.clone()                                    # default: all hold (tgt == s)
    frac_raw = torch.zeros(p, dtype=torch.float32, device=device)
    logp_pair = torch.zeros((), device=device)
    logp_frac = torch.zeros((), device=device)
    n_launch = 0

    src_rows = source_mask.nonzero(as_tuple=False).flatten().tolist()
    for s in src_rows:
        # Valid columns for row s: legal off-diagonal targets + the diagonal hold.
        row_valid = pair_mask[s].clone()
        row_valid[s] = True
        row_logits = torch.where(row_valid, pair_logits[s], neg_inf[s])
        dist = Categorical(logits=row_logits)
        t = dist.sample()
        logp_pair = logp_pair + dist.log_prob(t)
        ti = int(t.item())
        tgt_idx[s] = ti
        if ti != s:
            n_launch += 1
            loc = frac_loc[s, ti]
            z = Normal(loc, sigma_t).sample()
            raw = torch.sigmoid(z).clamp(1e-4, 1 - 1e-4)
            frac_raw[s] = raw
            logp_frac = logp_frac + _frac_logp(loc, sigma_t, raw)

    diagnostics = {
        "n_valid_sources": int(source_mask.sum().item()),
        "n_launch": int(n_launch),
        "n_hold": int(source_mask.sum().item()) - int(n_launch),
    }
    return Action(
        tgt_idx=tgt_idx,
        frac_raw=frac_raw,
        logprob=logp_pair + logp_frac,
        logprob_pair=logp_pair,
        logprob_frac=logp_frac,
        n_launch=int(n_launch),
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# Env projection                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ProjectionResult:
    env_actions: list[list[int]]      # list of [from_planet_id, angle, ships]
    ok: bool                           # False if any launching source was dropped
    n_emitted: int
    n_invalid: int
    invalid_reasons: list[str]


def project_to_env(
    action: Action,
    *,
    source_mask: torch.Tensor,        # (P,) bool — which rows are owned/legal sources
    surplus: torch.Tensor,            # (P,) float/int — per-source ships available to launch
    source_planet_ids: list[int],     # (P,) slot -> planet id (-1 if padded)
    target_planet_ids: list[int],     # (P,) slot -> planet id (-1 if padded)
    min_launch: int,
    plan_launch_fn,                   # (src_id, tgt_id, ships) -> object with .ok, .angle, .reason
) -> ProjectionResult:
    """Project a sampled :class:`Action` into env-shaped launch commands.

    PER-SOURCE budget: each launching source ``s`` (``source_mask[s]`` and
    ``tgt_idx[s] != s``) emits ``round(frac_raw[s] * surplus[s])`` ships,
    gated below ``min_launch``. There is NO shared-budget normalization
    across sources — this matches the runner ``single_target`` semantics.

    NEVER resample on invalid — the original sampled action stays in the
    buffer with its original logprob; an invalid launch just doesn't fire and
    the rollout earns an ``invalid_launch_penalty``.
    """
    tgt_idx = action.tgt_idx
    env_actions: list[list[int]] = []
    n_invalid = 0
    n_emitted = 0
    invalid_reasons: list[str] = []

    src_rows = source_mask.nonzero(as_tuple=False).flatten().tolist()
    for s in src_rows:
        t = int(tgt_idx[s].item())
        if t == s:
            continue                                # hold / NOOP
        src_pid = source_planet_ids[s]
        tgt_pid = target_planet_ids[t]
        if src_pid is None or src_pid < 0 or tgt_pid is None or tgt_pid < 0:
            n_invalid += 1
            invalid_reasons.append("pad_slot")
            continue
        budget = max(int(surplus[s].item()), 0)
        ships = int(round(float(action.frac_raw[s].item()) * budget))
        if ships < int(min_launch):
            n_invalid += 1
            invalid_reasons.append("min_launch")
            continue
        launch = plan_launch_fn(int(src_pid), int(tgt_pid), int(ships))
        if not getattr(launch, "ok", False):
            n_invalid += 1
            invalid_reasons.append(str(getattr(launch, "reason", "unknown")))
            continue
        env_actions.append([int(src_pid), float(launch.angle), int(ships)])
        n_emitted += 1

    return ProjectionResult(
        env_actions=env_actions,
        ok=(n_invalid == 0),
        n_emitted=n_emitted,
        n_invalid=n_invalid,
        invalid_reasons=invalid_reasons,
    )


# =========================================================================== #
# bernoulli_select_multinomial_alloc_v2                                       #
# =========================================================================== #
# Per OWNED source row ``s`` (``source_mask[s]``), with ``pair_mask[s]`` = legal
# off-diagonal targets:
#
#   Stage 1 — Selection (independent per-cell Bernoulli): for each legal
#     off-diagonal target ``t``, ``fire[s,t] ~ Bernoulli(sigmoid(pair_logits
#     [s,t]))``. The fired set ``F_s = {t : fired}``. The selection logprob is
#     the sum over ALL legal cells (fired AND not-fired) of the Bernoulli
#     log-likelihood, i.e. ``-BCE_with_logits(pair_logits[s, legal], fired_bits,
#     reduction='sum')``.
#
#   Stage 2 — Allocation (one Multinomial over [F_s ∪ self], N = source ships):
#     a categorical over the fired targets PLUS a ``self`` (keep/hold) category.
#     Category logits, in a FIXED order (fired targets in ascending column index,
#     then ``self`` last):
#         target t ∈ F_s : ``frac_loc[s, t]``    (REUSE the existing pair_frac raw
#                                                  logit — no new head)
#         self           : ``frac_loc[s, s]``    (the frac head's diagonal = HOLD)
#     ``probs = softmax([frac_loc[s, F_s], frac_loc[s, s]])`` and ``counts ~
#     Multinomial(S, probs)`` with ``S = source_ships[s]``. ``ships[s,t] =
#     counts[t]`` for ``t ∈ F_s``; ``held[s] = counts[self]``. So each of the S
#     ships is routed individually and the counts ARE the launch sizes. The
#     allocation logprob is ``Σ_{F_s ∪ self} counts · log(probs)`` (the
#     multinomial coefficient is dropped — it depends only on the stored counts
#     and cancels in the PPO ratio; the recompute uses the SAME Σ c·log p form).
#
# v2 (2026-06-11): the HOLD logit moved from ``pair_logits[s, s]`` (v1) to
# ``frac_loc[s, s]`` — the whole allocation softmax now lives in ONE head at
# one output scale, and select (pair_head) / alloc (pair_frac_head) gradients
# decouple at the head level. v1's slot existed to inherit the single-target
# NOOP diagonal; current warm-starts are multi-target pretrains whose
# diagonals were never supervised, so nothing is lost. ``pair_logits``'
# diagonal is now fully unused. Still NO new parameters: any prior checkpoint
# loads via ``load_state_dict(strict=False)`` with 0 missing / 0 unexpected.
@dataclass
class MultiTargetAction:
    """One snapshot's ``bernoulli_select_multinomial_alloc_v2`` action.

    ``select_mask (P, P) bool``: per owned source row, the fired off-diagonal
        targets ``F_s`` (a subset of ``pair_mask[s]``). Rows for non-owned /
        non-acting sources are all-False.
    ``alloc_counts (P, P) long``: ships routed to each fired target,
        ``alloc_counts[s, t] = counts[t]`` for ``t ∈ F_s``; 0 elsewhere.
    ``self_counts (P,) long``: ships kept/held on each owned source
        (the ``self`` multinomial category); 0 on non-acting rows.
    ``logprob`` scalar: ``logprob_select + logprob_alloc``.
    ``logprob_select`` scalar: sum over owned rows of the all-legal-cell
        Bernoulli log-likelihood.
    ``logprob_alloc`` scalar: sum over owned rows of ``Σ counts · log(probs)``
        over ``F_s ∪ self``.
    ``n_terms`` int: number of summed logprob components — one per legal
        Bernoulli cell across all owned rows, plus one per acting source (its
        allocation multinomial). Used to normalize the PPO KL to a
        per-component scale, exactly as the single-target path does.
    """

    select_mask: torch.Tensor       # (P, P) bool — fired targets per owned source
    alloc_counts: torch.Tensor      # (P, P) long — ships per fired target
    self_counts: torch.Tensor       # (P,) long — ships held on each owned source
    logprob: torch.Tensor           # scalar — select + alloc
    logprob_select: torch.Tensor    # scalar — sum of per-cell Bernoulli logliks
    logprob_alloc: torch.Tensor     # scalar — sum of per-source Σ c·log p
    n_terms: int                    # Bernoulli cells + one per acting source
    diagnostics: dict[str, float] = field(default_factory=dict)


def sample_multi_target(
    pair_logits: torch.Tensor,        # (P, P) — actor logits for one snapshot
    frac_loc: torch.Tensor,            # (P, P) — actor pair_frac raw logit (alloc logits)
    source_ships: torch.Tensor,        # (P,) long — current ship count per source slot
    *,
    pair_mask: torch.Tensor,           # (P, P) bool — legal off-diagonal targets
    source_mask: torch.Tensor,         # (P,) bool — owned/legal source rows
    temperature: float = 1.0,
    select_logit_bias: float = 0.0,
) -> MultiTargetAction:
    """Draw one ``bernoulli_select_multinomial_alloc_v2`` action.

    A per-owned-source Python loop (mirrors :func:`sample_single_target`). The
    stored logprobs are recomputed EXACTLY by
    :func:`agents.transformer_v2.ppo.loss.action_logprob_multi`, so the PPO
    ratio is 1 for an unchanged policy. ``temperature`` divides BOTH the
    selection logits and the allocation logits (a global softmax/sigmoid
    temperature); pass 1.0 for the contract default.

    ``select_logit_bias`` shifts ONLY the per-cell selection Bernoulli logit:
    ``fire[s,t] ~ Bernoulli(sigmoid(pair_logits[s,t]/tau - select_logit_bias))``.
    A positive bias fires fewer (more confident) targets — directly reducing
    over-firing. b=2.0 mirrors the runner's ``pair_logits > 2.0`` threshold. The
    ALLOCATION multinomial is UNCHANGED; the bias touches the selection only.
    The same value MUST be passed to
    :func:`agents.transformer_v2.ppo.loss.action_logprob_multi` and
    :func:`agents.transformer_v2.ppo.loss.multi_target_entropy` at update time
    or the recomputed logprob desyncs the PPO ratio.
    """
    if pair_logits.dim() != 2:
        raise ValueError("sample is per-snapshot; expected pair_logits (P, P)")
    p = pair_logits.shape[0]
    device = pair_logits.device
    tau = float(temperature)
    sel_bias = float(select_logit_bias)

    select_mask = torch.zeros((p, p), dtype=torch.bool, device=device)
    alloc_counts = torch.zeros((p, p), dtype=torch.long, device=device)
    self_counts = torch.zeros(p, dtype=torch.long, device=device)
    logp_select = torch.zeros((), device=device)
    logp_alloc = torch.zeros((), device=device)
    n_terms = 0
    n_launch_sources = 0
    n_fired_total = 0
    n_ships_launched = 0

    src_rows = source_mask.nonzero(as_tuple=False).flatten().tolist()
    for s in src_rows:
        legal_cols = pair_mask[s].nonzero(as_tuple=False).flatten()
        n_legal = int(legal_cols.numel())
        if n_legal == 0:
            continue  # owned but no legal target (should not happen given source_mask)

        # --- Stage 1: independent per-cell Bernoulli over legal targets ---
        # select_logit_bias shifts the fire probability DOWN (positive bias ->
        # fewer fires); the recompute applies the SAME shift before sigmoid.
        sel_logits = pair_logits[s, legal_cols] / tau - sel_bias     # (n_legal,)
        probs1 = torch.sigmoid(sel_logits)
        fired_bits = torch.bernoulli(probs1)                          # (n_legal,) in {0,1}
        # Selection logprob = sum of Bernoulli log-likelihood over ALL legal
        # cells (fired AND not-fired). -BCE_with_logits(reduction='sum') is
        # exactly Σ [b·log σ(x) + (1-b)·log(1-σ(x))].
        logp_select = logp_select - F.binary_cross_entropy_with_logits(
            sel_logits, fired_bits, reduction="sum",
        )
        n_terms += n_legal

        fired_local = fired_bits.bool()                               # (n_legal,)
        fired_cols = legal_cols[fired_local]                          # (n_fired,) ascending
        select_mask[s, fired_cols] = True
        n_fired = int(fired_cols.numel())
        n_fired_total += n_fired

        # --- Stage 2: one Multinomial over [F_s ∪ self], N = source ships ---
        # Category order (FIXED, must match the recompute): fired targets in
        # ascending column index, then ``self`` last.
        ship_n = int(source_ships[s].item())
        alloc_logits = torch.cat([
            frac_loc[s, fired_cols],                                  # (n_fired,)
            frac_loc[s, s].reshape(1),                                # self (frac diagonal)
        ]) / tau                                                      # (n_fired + 1,)
        log_probs2 = torch.log_softmax(alloc_logits, dim=-1)          # (n_fired + 1,)
        if ship_n > 0:
            counts = Multinomial(
                total_count=ship_n, logits=alloc_logits,
            ).sample()                                                # (n_fired + 1,) float
        else:
            counts = torch.zeros_like(alloc_logits)
        # Allocation logprob = Σ counts · log(probs) over F_s ∪ self (drop the
        # multinomial coefficient; it cancels in the PPO ratio).
        logp_alloc = logp_alloc + (counts * log_probs2).sum()
        n_terms += 1  # one allocation multinomial per acting source

        counts_long = counts.round().long()
        if n_fired > 0:
            alloc_counts[s, fired_cols] = counts_long[:n_fired]
            n_launch_sources += 1
        self_counts[s] = counts_long[n_fired]
        n_ships_launched += int(counts_long[:n_fired].sum().item())

    diagnostics = {
        "n_valid_sources": int(source_mask.sum().item()),
        "n_launch_sources": int(n_launch_sources),
        "n_fired_total": int(n_fired_total),
        "n_ships_launched": int(n_ships_launched),
    }
    return MultiTargetAction(
        select_mask=select_mask,
        alloc_counts=alloc_counts,
        self_counts=self_counts,
        logprob=logp_select + logp_alloc,
        logprob_select=logp_select,
        logprob_alloc=logp_alloc,
        n_terms=int(n_terms),
        diagnostics=diagnostics,
    )


def project_multi_target_to_env(
    action: MultiTargetAction,
    *,
    source_mask: torch.Tensor,        # (P,) bool — owned/legal source rows
    source_planet_ids: list[int],     # (P,) slot -> planet id (-1 if padded)
    target_planet_ids: list[int],     # (P,) slot -> planet id (-1 if padded)
    min_launch: int,
    plan_launch_fn,                   # (src_id, tgt_id, ships) -> obj with .ok, .angle, .reason
) -> ProjectionResult:
    """Project a :class:`MultiTargetAction` into env-shaped launch commands.

    Each fired cell ``(s, t)`` with ``alloc_counts[s, t] > 0`` emits a launch of
    ``ships = alloc_counts[s, t]`` (the counts ARE the launch sizes — N = S was
    already routed in the multinomial). Cells below ``min_launch`` are dropped;
    held ships (``self_counts``) stay home. Invalid launches are NOT resampled —
    the stored action keeps its original logprob and the rollout earns the
    existing invalid penalty.
    """
    env_actions: list[list[int]] = []
    n_invalid = 0
    n_emitted = 0
    invalid_reasons: list[str] = []

    src_rows = source_mask.nonzero(as_tuple=False).flatten().tolist()
    for s in src_rows:
        fired_cols = action.select_mask[s].nonzero(as_tuple=False).flatten().tolist()
        for t in fired_cols:
            ships = int(action.alloc_counts[s, t].item())
            if ships < int(min_launch):
                n_invalid += 1
                invalid_reasons.append("min_launch")
                continue
            src_pid = source_planet_ids[s]
            tgt_pid = target_planet_ids[t]
            if src_pid is None or src_pid < 0 or tgt_pid is None or tgt_pid < 0:
                n_invalid += 1
                invalid_reasons.append("pad_slot")
                continue
            launch = plan_launch_fn(int(src_pid), int(tgt_pid), int(ships))
            if not getattr(launch, "ok", False):
                n_invalid += 1
                invalid_reasons.append(str(getattr(launch, "reason", "unknown")))
                continue
            env_actions.append([int(src_pid), float(launch.angle), int(ships)])
            n_emitted += 1

    return ProjectionResult(
        env_actions=env_actions,
        ok=(n_invalid == 0),
        n_emitted=n_emitted,
        n_invalid=n_invalid,
        invalid_reasons=invalid_reasons,
    )


# =========================================================================== #
# bounded_k_select_multinomial_alloc_v3                                       #
# =========================================================================== #
# Per OWNED source row ``s``:
#
#   Stage 1 — Selection (ONE multinomial of k draws): tokens = the legal
#     off-diagonal targets PLUS a ``self`` null token (logit = the select
#     head's diagonal ``pair_logits[s, s]`` + ``self_logit_bias``; positive
#     bias -> more self draws -> fires less, same direction as v2's
#     select_logit_bias).  ``k = min(k_max, N_s // min_launch)`` is a
#     DETERMINISTIC function of state (no logprob term).  ``draws ~
#     Multinomial(k, softmax(logits/tau))``.  Fired set F_s = target tokens
#     with >= 1 draw; duplicates merge (stronger preference); all-self = hold.
#     By construction |F_s| <= k <= N_s/min_launch: the floor is ALWAYS
#     feasible — projection-level min_launch drops are retired.
#
#   Stage 2 — Floor + allocation of the REMAINDER: each fired target is
#     pre-assigned ``min_launch`` (deterministic given F_s — no logprob term);
#     ``extras ~ Multinomial(N_s - min_launch*|F_s|, softmax([frac_loc[s, F_s],
#     frac_loc[s, s]]/tau))``.  Launch size = min_launch + extras_t; the self
#     slot's extras stay home.  All-hold rows have no alloc multinomial.
#
# The action stores DRAW COUNTS and EXTRAS (not final sizes), so the logprob
# recompute needs no phase-dependent min_launch; projection adds the floor.
# Both stages' logprobs use the coefficient-free ``sum counts*log p`` form
# (stored counts are constants -> coefficients cancel in the PPO ratio).
@dataclass
class BoundedKAction:
    """One snapshot's ``bounded_k_select_multinomial_alloc_v3`` action.

    ``select_counts (P, P+1) long``: per owned row, draws per token —
        columns 0..P-1 = target columns, column P = the ``self`` token.
        Row sum = k for owned rows, 0 elsewhere.
    ``alloc_extras (P, P) long``: extra ships (beyond the min_launch floor)
        per fired target; 0 on non-fired cells.
    ``self_extras (P,) long``: extra ships allocated to the self slot on
        rows with >= 1 fired target (held ships beyond any floor); 0 on
        all-hold rows (those keep everything home implicitly).
    ``logprob / logprob_select / logprob_alloc / n_terms``: as in
        :class:`MultiTargetAction`; n_terms = one select multinomial per
        owned row + one alloc multinomial per row with >= 1 fired target.
    """

    select_counts: torch.Tensor     # (P, P+1) long
    alloc_extras: torch.Tensor      # (P, P) long
    self_extras: torch.Tensor       # (P,) long
    logprob: torch.Tensor           # scalar
    logprob_select: torch.Tensor    # scalar
    logprob_alloc: torch.Tensor     # scalar
    n_terms: int
    diagnostics: dict[str, float] = field(default_factory=dict)


def sample_bounded_k(
    pair_logits: torch.Tensor,        # (P, P) — select head (diag = self token)
    frac_loc: torch.Tensor,            # (P, P) — alloc head (diag = HOLD extras)
    source_ships: torch.Tensor,        # (P,) long — current ship count per source
    *,
    pair_mask: torch.Tensor,           # (P, P) bool — legal off-diagonal targets
    source_mask: torch.Tensor,         # (P,) bool — owned/legal source rows
    min_launch: int,
    k_max: int = 3,
    temperature: float = 1.0,
    self_logit_bias: float = 0.0,
) -> BoundedKAction:
    """Draw one ``bounded_k_select_multinomial_alloc_v3`` action.

    Recomputed EXACTLY by ``loss.action_logprob_bounded_k`` (same tau /
    self_logit_bias MUST be passed there or the PPO ratio desyncs).
    """
    if pair_logits.dim() != 2:
        raise ValueError("sample is per-snapshot; expected pair_logits (P, P)")
    p = pair_logits.shape[0]
    device = pair_logits.device
    tau = float(temperature)
    m = int(min_launch)

    select_counts = torch.zeros((p, p + 1), dtype=torch.long, device=device)
    alloc_extras = torch.zeros((p, p), dtype=torch.long, device=device)
    self_extras = torch.zeros(p, dtype=torch.long, device=device)
    logp_select = torch.zeros((), device=device)
    logp_alloc = torch.zeros((), device=device)
    n_terms = 0
    n_launch_sources = 0
    n_fired_total = 0
    n_ships_launched = 0
    k_sum = 0

    src_rows = source_mask.nonzero(as_tuple=False).flatten().tolist()
    for s in src_rows:
        legal_cols = pair_mask[s].nonzero(as_tuple=False).flatten()
        n_legal = int(legal_cols.numel())
        if n_legal == 0:
            continue
        ship_n = int(source_ships[s].item())
        k = min(int(k_max), ship_n // max(1, m))
        if k <= 0:
            continue  # cannot floor even one launch; legality should preclude
        k_sum += k

        # --- Stage 1: one multinomial of k draws over [legal targets, self] ---
        sel_logits = torch.cat([
            pair_logits[s, legal_cols],
            (pair_logits[s, s] + self_logit_bias).reshape(1),
        ]) / tau                                                   # (n_legal+1,)
        log_probs1 = torch.log_softmax(sel_logits, dim=-1)
        draws = Multinomial(total_count=k, logits=sel_logits).sample()
        logp_select = logp_select + (draws * log_probs1).sum()
        n_terms += 1
        draws_long = draws.round().long()
        select_counts[s, legal_cols] = draws_long[:n_legal]
        select_counts[s, p] = draws_long[n_legal]

        fired_local = draws_long[:n_legal] >= 1
        fired_cols = legal_cols[fired_local]                       # ascending
        n_fired = int(fired_cols.numel())
        n_fired_total += n_fired
        if n_fired == 0:
            continue  # all-self: full hold, no alloc multinomial

        # --- Stage 2: floor min_launch per fired target, multinomial extras ---
        rem = ship_n - m * n_fired                                 # >= 0 by k-cap
        alloc_logits = torch.cat([
            frac_loc[s, fired_cols],
            frac_loc[s, s].reshape(1),
        ]) / tau                                                   # (n_fired+1,)
        log_probs2 = torch.log_softmax(alloc_logits, dim=-1)
        if rem > 0:
            extras = Multinomial(total_count=rem, logits=alloc_logits).sample()
        else:
            extras = torch.zeros_like(alloc_logits)
        logp_alloc = logp_alloc + (extras * log_probs2).sum()
        n_terms += 1
        extras_long = extras.round().long()
        alloc_extras[s, fired_cols] = extras_long[:n_fired]
        self_extras[s] = extras_long[n_fired]
        n_launch_sources += 1
        n_ships_launched += m * n_fired + int(extras_long[:n_fired].sum().item())

    diagnostics = {
        "n_valid_sources": int(source_mask.sum().item()),
        "n_launch_sources": int(n_launch_sources),
        "n_fired_total": int(n_fired_total),
        "n_ships_launched": int(n_ships_launched),
        "k_mean": (k_sum / max(1, len(src_rows))),
    }
    return BoundedKAction(
        select_counts=select_counts,
        alloc_extras=alloc_extras,
        self_extras=self_extras,
        logprob=logp_select + logp_alloc,
        logprob_select=logp_select,
        logprob_alloc=logp_alloc,
        n_terms=int(n_terms),
        diagnostics=diagnostics,
    )


def project_bounded_k_to_env(
    action: BoundedKAction,
    *,
    source_mask: torch.Tensor,        # (P,) bool
    source_planet_ids: list[int],
    target_planet_ids: list[int],
    min_launch: int,
    plan_launch_fn,
) -> ProjectionResult:
    """Project a :class:`BoundedKAction` into env launches.

    Launch size = ``min_launch + alloc_extras[s, t]`` for every fired cell —
    every launch clears the floor BY CONSTRUCTION, so the only remaining
    invalids are trajectory-level (``plan_launch`` rejections: sun/wrong-
    planet/boundary) and pad slots.
    """
    env_actions: list[list[int]] = []
    n_invalid = 0
    n_emitted = 0
    invalid_reasons: list[str] = []
    p = action.alloc_extras.shape[0]

    src_rows = source_mask.nonzero(as_tuple=False).flatten().tolist()
    for s in src_rows:
        fired_cols = (action.select_counts[s, :p] >= 1).nonzero(
            as_tuple=False).flatten().tolist()
        for t in fired_cols:
            ships = int(min_launch) + int(action.alloc_extras[s, t].item())
            src_pid = source_planet_ids[s]
            tgt_pid = target_planet_ids[t]
            if src_pid is None or src_pid < 0 or tgt_pid is None or tgt_pid < 0:
                n_invalid += 1
                invalid_reasons.append("pad_slot")
                continue
            launch = plan_launch_fn(int(src_pid), int(tgt_pid), int(ships))
            if not getattr(launch, "ok", False):
                n_invalid += 1
                invalid_reasons.append(str(getattr(launch, "reason", "unknown")))
                continue
            env_actions.append([int(src_pid), float(launch.angle), int(ships)])
            n_emitted += 1

    return ProjectionResult(
        env_actions=env_actions,
        ok=(n_invalid == 0),
        n_emitted=n_emitted,
        n_invalid=n_invalid,
        invalid_reasons=invalid_reasons,
    )
