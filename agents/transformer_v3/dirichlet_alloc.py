"""Contract v4 allocation: Dirichlet shares with a learned concentration.

Parameterization (migration-safe by construction):

    mean_i = softmax(frac_loc[s, F_s ∪ {s}])     — the EXISTING v2/v3 softmax
    α_0(s) = ALPHA_MAX · sigmoid(conc(s))        — NEW per-source scalar head
    α_i    = α_0(s) · mean_i                     — Dirichlet(α) over [fired…, HOLD]

At init (or with the conc head removed) the mean equals the current
contract's allocation distribution exactly, so v3-trained frac heads warm-
start bit-for-bit; only the concentration scalar is new. α_0 is the
actor's LEARNED confidence: its supervision comes from expert consistency
— contexts where the experts always make the same split push α_0 up,
contexts with varied splits force it down to explain the spread. No
explicit confidence labels exist or are needed.

Pretrain loss = Dirichlet NLL on ε-smoothed expert shares. Smoothing is
MANDATORY: top-meta experts commit ~98% of ships to one target, so raw
shares sit on the simplex boundary where the Dirichlet density diverges.

Deploy/PPO decodes (task #17, not in this module): mean (α_i/α_0 — the
deterministic arm ≡ today's alloc_softmax sizing), sample (Dirichlet
draw), α_0-gated.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..transformer_v2.pretrain.alloc_labels import build_alloc_targets

ALPHA_MAX = 100.0     # α_0 ceiling — keeps NLL finite on decisive experts.
                      # Was 200: stage-B detonated at ep17-19 (satfrac 0.46
                      # then α_0 collapse) and stage-D pressed 0.53 by ep3;
                      # halving the cap + a slower conc-head LR group are
                      # the stability fixes.
SHARE_EPS = 1e-3      # simplex smoothing for boundary expert shares

ACTION_CONTRACT_V4 = "bounded_k_select_dirichlet_alloc_v4"


class AllocConcentrationHead(nn.Module):
    """Per-source α_0 off the L4 source tokens: (B, P, d) -> (B, P)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        # Start mid-range (α_0 ≈ ALPHA_MAX/2 at zero input would be
        # sigmoid(0)=0.5): bias the final layer to sigmoid≈0.25 so early
        # training starts diffuse-ish and earns confidence.
        with torch.no_grad():
            self.net[-1].bias.fill_(-1.1)

    def forward(self, source_joint: torch.Tensor) -> torch.Tensor:
        return ALPHA_MAX * torch.sigmoid(self.net(source_joint).squeeze(-1))


def _smooth_shares(shares: torch.Tensor) -> torch.Tensor:
    """Pull boundary shares inside the open simplex (renormalized)."""
    k = shares.shape[-1]
    sm = shares.clamp(min=SHARE_EPS)
    return sm / sm.sum(-1, keepdim=True).clamp(min=k * SHARE_EPS)


def dirichlet_alloc_nll(
    frac_loc: torch.Tensor,        # (B, P, P) raw frac head
    alloc_conc: torch.Tensor,      # (B, P) α_0 per source
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dirichlet NLL over acting source rows.

    Reuses :func:`build_alloc_targets` (the v2/v3 label builder): per
    acting source row it yields the expert's empirical share vector over
    ``[fired targets…, HOLD]`` and the row/cell masks in the SAME
    parameterization the frac head softmaxes — so the mean term of this
    loss supervises exactly what the multinomial CE supervised, plus the
    concentration.
    """
    row_mask, shares = build_alloc_targets(batch)   # (B,P), (B,P,P+1)
    if not row_mask.any():
        zero = frac_loc.sum() * 0.0
        return zero, {"empty": 1.0}
    # Support = the expert's fired cells (share > 0 by construction) plus
    # HOLD — the same cells the v2/v3 multinomial CE softmaxes over.
    support = shares > 0
    support[..., -1] = True

    B, P, _ = frac_loc.shape
    diag = torch.arange(P, device=frac_loc.device)
    hold_logit = frac_loc[:, diag, diag].unsqueeze(-1)          # (B, P, 1)
    logits = torch.cat([frac_loc, hold_logit], dim=-1)          # (B, P, P+1)
    neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~support, neg)
    mean = torch.softmax(logits, dim=-1)                         # (B, P, P+1)

    alpha = alloc_conc.unsqueeze(-1) * mean                      # (B, P, P+1)
    # Smooth on the support only, then renormalize within support.
    sm_raw = torch.where(support, shares.clamp(min=0.0),
                         torch.zeros_like(shares))
    sm = torch.where(support, sm_raw.clamp(min=SHARE_EPS),
                     torch.zeros_like(sm_raw))
    sm = sm / sm.sum(-1, keepdim=True).clamp(min=SHARE_EPS)

    # Dirichlet log-density over support cells of acting rows:
    #   log Γ(α0) − Σ_sup log Γ(αi) + Σ_sup (αi − 1)·log xi
    a = alpha[row_mask]                                          # (N, P+1)
    x = sm[row_mask]
    sup = support[row_mask]
    a_sup = torch.where(sup, a, torch.zeros_like(a))
    a0 = a_sup.sum(-1)                                           # (N,)
    logp = (
        torch.lgamma(a0.clamp(min=1e-6))
        - torch.where(sup, torch.lgamma(a.clamp(min=1e-6)),
                      torch.zeros_like(a)).sum(-1)
        + torch.where(sup, (a - 1.0) * x.clamp(min=1e-12).log(),
                      torch.zeros_like(a)).sum(-1)
    )
    nll = -logp.mean()

    with torch.no_grad():
        share_err = (
            torch.where(sup, mean[row_mask] - x, torch.zeros_like(x))
            .abs().sum(-1).mean()
        )
        conc = alloc_conc[row_mask]
    return nll, {
        "alloc_nll": float(nll.detach()),
        "alpha0_mean": float(conc.mean()),
        "alpha0_p90": float(conc.quantile(0.9)) if conc.numel() > 1 else float(conc.mean()),
        "share_l1": float(share_err),
        "alpha0_satfrac": float((conc > 0.95 * ALPHA_MAX).float().mean()),
    }
