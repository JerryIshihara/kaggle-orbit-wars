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
from torch.distributions import Categorical, Normal


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
