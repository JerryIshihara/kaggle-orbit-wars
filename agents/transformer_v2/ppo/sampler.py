"""Action sampling and env projection for ``source_multi_target_v1``.

One snapshot's actor decisions:

  1. ``source_id``    — Categorical with a fixed-logit NOOP entry, scored
                        per source by ``max_j(pair_logits[s, j])`` over the
                        valid pair mask (row max, not logsumexp — see
                        protocol "Known problems" #1 + "Resolutions" #1).
  2. ``target_bits``  — Bernoulli per valid target on the chosen source's row.
  3. ``frac_raw[j]``  — LogitNormal(loc=pair_frac[s, j], sigma) for every
                        selected target. sigma is fixed (CLI hyperparameter).

The stored ``frac_raw`` is the numerically-clamped sigmoid; the launch-side
clamp ``[0.02, 1.00]`` is applied only at env projection. PPO recomputes
logprob from the stored value, so the two must agree exactly.

NOOP semantics: NOOP is the fixed-logit source option (``source_id = -1``).
If a real source samples an empty target set, the rollout earns a small
``empty_target_set`` penalty so the source categorical learns to prefer
NOOP in those states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.distributions import Bernoulli, Categorical, Normal


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
        have any valid target?). Used as a mask on the source categorical so
        sources with no valid target get -inf logit.
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
    """One snapshot's sampled action + per-component logprobs."""

    source_id: int                  # -1 = NOOP, else 0..P-1
    target_bits: torch.Tensor       # (P,) bool — selected targets on the chosen source's row
    frac_raw: torch.Tensor          # (P,) float — clamped sigmoid for selected targets; zeros elsewhere
    logprob: torch.Tensor           # scalar — logp_source + logp_targets + logp_frac
    logprob_source: torch.Tensor    # scalar
    logprob_targets: torch.Tensor   # scalar (0.0 if NOOP)
    logprob_frac: torch.Tensor      # scalar (0.0 if NOOP or no targets)
    empty_target_set: bool          # True if a real source was chosen but no targets fired
    diagnostics: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #
def sample_source_multi_target(
    pair_logits: torch.Tensor,        # (P, P) — actor logits for one snapshot
    frac_loc: torch.Tensor,            # (P, P) — actor frac means (pair_frac raw logit)
    sigma: torch.Tensor | float,       # () scalar — fixed frac stddev
    *,
    pair_mask: torch.Tensor,           # (P, P) bool
    source_mask: torch.Tensor,         # (P,) bool
    noop_logit_bias: float = 0.0,
) -> Action:
    """Draw one ``source_multi_target_v1`` action.

    Source logits use ``row_max`` (not logsumexp) to avoid an
    n-valid-targets count bias on the source pick — see protocol
    "Known problems" #1.
    """
    if pair_logits.dim() != 2:
        raise ValueError("sample is per-snapshot; expected pair_logits (P, P)")
    p = pair_logits.shape[0]
    sigma_t = sigma if torch.is_tensor(sigma) else torch.tensor(float(sigma),
                                                                 device=pair_logits.device)

    masked_pairs = torch.where(pair_mask, pair_logits, torch.full_like(pair_logits, float("-inf")))

    # 1. Source categorical with fixed-logit NOOP at index 0.
    #    row_max over valid targets — see "Resolutions" #1.
    source_logits = masked_pairs.max(dim=1).values                          # (P,)
    source_logits = torch.where(
        source_mask, source_logits,
        torch.full_like(source_logits, float("-inf")),
    )
    src_dist = Categorical(logits=torch.cat([
        torch.tensor([float(noop_logit_bias)], device=pair_logits.device),
        source_logits,
    ]))
    src_plus = src_dist.sample()
    logp_source = src_dist.log_prob(src_plus)

    if int(src_plus.item()) == 0:
        # NOOP — no target / frac decisions.
        zeros_p = torch.zeros(p, dtype=torch.float32, device=pair_logits.device)
        bits = torch.zeros(p, dtype=torch.bool, device=pair_logits.device)
        zero = torch.zeros((), device=pair_logits.device)
        return Action(
            source_id=-1, target_bits=bits, frac_raw=zeros_p,
            logprob=logp_source, logprob_source=logp_source,
            logprob_targets=zero, logprob_frac=zero,
            empty_target_set=False,
            diagnostics={"n_valid_sources": int(source_mask.sum().item())},
        )

    src = int(src_plus.item()) - 1

    # 2. Target Bernoulli over the chosen source's row (valid cells only).
    valid_tgt_row = pair_mask[src]                                           # (P,)
    row_logits = torch.where(
        valid_tgt_row, pair_logits[src],
        torch.full_like(pair_logits[src], float("-inf")),
    )
    target_dist = Bernoulli(logits=row_logits)
    # Sample raw bits but force invalid cells off. The Bernoulli logprob is
    # masked to valid cells only — invalid cells contribute 0 to logp.
    raw_bits = target_dist.sample().bool()
    target_bits = raw_bits & valid_tgt_row
    log_probs_per_cell = target_dist.log_prob(target_bits.float())           # (P,)
    logp_targets = torch.where(
        valid_tgt_row, log_probs_per_cell, torch.zeros_like(log_probs_per_cell),
    ).sum()

    # 3. Frac sample per selected target — independent logit-normals.
    frac_raw = torch.zeros(p, dtype=torch.float32, device=pair_logits.device)
    logp_frac = torch.zeros((), device=pair_logits.device)
    selected = target_bits.nonzero(as_tuple=False).flatten()
    for tgt in selected.tolist():
        loc = frac_loc[src, tgt]
        z = Normal(loc, sigma_t).sample()
        # Numerical clamp only — match PSEUDOCODE / protocol.
        raw = torch.sigmoid(z).clamp(1e-4, 1 - 1e-4)
        frac_raw[tgt] = raw
        # log p(sigmoid(z)) = log p(z) - log|d sigmoid(z) / dz|
        logp = (
            Normal(loc, sigma_t).log_prob(torch.logit(raw))
            - torch.log(raw)
            - torch.log1p(-raw)
        )
        logp_frac = logp_frac + logp

    empty = bool((target_bits.sum() == 0).item())
    diagnostics = {
        "n_valid_sources": int(source_mask.sum().item()),
        "n_valid_targets_in_row": int(valid_tgt_row.sum().item()),
        "n_selected_targets": int(target_bits.sum().item()),
        "noop_sampled": False,
        "empty_target_set": empty,
    }
    return Action(
        source_id=src, target_bits=target_bits, frac_raw=frac_raw,
        logprob=logp_source + logp_targets + logp_frac,
        logprob_source=logp_source,
        logprob_targets=logp_targets,
        logprob_frac=logp_frac,
        empty_target_set=empty,
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# Env projection                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ProjectionResult:
    env_actions: list[list[int]]      # list of [from_planet_id, angle, ships]
    ok: bool                           # False if any selected target was dropped
    n_emitted: int
    n_invalid: int
    invalid_reasons: list[str]


def project_to_env(
    action: Action,
    *,
    source_planet_id: int,
    source_ships: int,
    surplus: int,
    min_launch: int,
    plan_launch_fn,                  # (src_id, tgt_id, ships) -> object with .ok, .angle, .reason
    target_planet_ids: list[int],
) -> ProjectionResult:
    """Project a sampled :class:`Action` into env-shaped launch commands.

    Allocation: selected target ``frac_raw`` values are normalized into a
    shared source budget — they are weights, not absolute fractions. The
    semantic ``frac_raw`` stored in the shard is what PPO recomputes logprob
    on; the budgeted ship count is only used for the env call.

    NEVER resample on invalid — the original sampled action stays in the
    buffer with its original logprob; the invalid target just doesn't fire
    and the rollout earns an ``invalid_launch_penalty``.
    """
    if action.source_id < 0:
        return ProjectionResult(env_actions=[], ok=True, n_emitted=0,
                                 n_invalid=0, invalid_reasons=[])
    if action.target_bits.sum().item() == 0:
        # empty_target_set — no launch, but not an "invalid" per-target case.
        return ProjectionResult(env_actions=[], ok=True, n_emitted=0,
                                 n_invalid=0, invalid_reasons=[])

    selected_idx = action.target_bits.nonzero(as_tuple=False).flatten().tolist()
    raws = torch.stack([action.frac_raw[i] for i in selected_idx])
    weights = raws / raws.sum().clamp_min(1e-8)
    budget = max(int(surplus), 0)

    env_actions: list[list[int]] = []
    n_invalid = 0
    invalid_reasons: list[str] = []

    # Allocate ship budget across selected targets using stable rounding.
    # Largest-remainder method preserves the total within +/- 1 ship.
    raw_alloc = (weights * float(budget)).tolist()
    floored = [int(x) for x in raw_alloc]
    remainder = budget - sum(floored)
    fracs = [(raw_alloc[i] - floored[i], i) for i in range(len(floored))]
    fracs.sort(reverse=True)
    for r, _ in fracs[:max(0, remainder)]:
        # nudge top-remainder cells up by 1 ship
        idx_in_selected = next(i for f, i in fracs if f == r)
        floored[idx_in_selected] += 1

    for slot_i, sel_i in enumerate(selected_idx):
        ships = floored[slot_i]
        if ships < min_launch:
            n_invalid += 1
            invalid_reasons.append("min_launch")
            continue
        tgt_planet_id = target_planet_ids[sel_i]
        launch = plan_launch_fn(source_planet_id, tgt_planet_id, ships)
        if not getattr(launch, "ok", False):
            n_invalid += 1
            invalid_reasons.append(str(getattr(launch, "reason", "unknown")))
            continue
        env_actions.append([source_planet_id, float(launch.angle), int(ships)])

    return ProjectionResult(
        env_actions=env_actions,
        ok=(n_invalid == 0),
        n_emitted=len(env_actions),
        n_invalid=n_invalid,
        invalid_reasons=invalid_reasons,
    )
