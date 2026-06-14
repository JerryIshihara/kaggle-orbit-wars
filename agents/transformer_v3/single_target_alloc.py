"""Single-target contract: softmax select + LOGIT-NORMAL sigmoid alloc with a
learned per-source CONFIDENCE (sigma).

Motivated by the meta analysis (2026-06-14): the top-3 players (Jake Will,
Hober Malloc, TonyK) are 100% SINGLE-TARGET per source — each launching planet
commits ALL its force to ONE target. The v4 bounded-k Dirichlet contract let
the actor split a source across 2-3 targets (our agent: 27% multi-target, the
most of anyone), diluting fleet commitment — which the win-signal probe prices
at ~-33% win prob. Reverting to single-target full-commitment matches the meta.

The two heads (UNCHANGED neural modules, reinterpreted):
  SELECT  pair_logits  -> per-source Categorical over [targets ... HOLD diag]
          (one target). Trained by `_pair_single_target_ce` (CE on the
          expert's dominant target; the meta data is 100% single-target so the
          label is unambiguous).
  ALLOC   pair_frac[s,t*] = the LOC of LogitNormal; the launch fraction is
          f = sigmoid(Normal(loc, sigma)). `FracSigmaHead` predicts a per-source
          log-sigma = the CONFIDENCE (small sigma = decisive commitment). At
          deploy/PPO the fraction is SAMPLED from this LogitNormal — the
          sampling-with-confidence the contract needs (sigma replaces the old
          hardcoded --sigma, exactly as alpha0 did for Dirichlet).

Pretrain loss = select CE + LogitNormal NLL of the expert's observed fraction
(teaches loc AND sigma jointly: contexts where experts always send the same
fraction push sigma down; varied fractions force it up). No explicit
confidence labels — supervision is the expert fraction's consistency, same
principle as the Dirichlet alpha0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

#: log-sigma clamp: sigma in [~0.05, ~2.7]. Below, the LogitNormal collapses to
#: a point mass (no exploration); above, near-uniform fractions.
LOG_SIGMA_MIN = -3.0
LOG_SIGMA_MAX = 1.0
FRAC_EPS = 1e-4   # keep logit(f) finite on boundary fractions
_LOG_2PI = math.log(2.0 * math.pi)


class FracSigmaHead(nn.Module):
    """Per-source LogitNormal log-sigma off the L4 source tokens: (B,P,d)->(B,P).

    Mirrors AllocConcentrationHead. Bias-init to a MODERATE spread
    (log_sigma ~ -0.7 -> sigma ~ 0.5) so early training samples explore, then
    earns confidence (smaller sigma) where experts are decisive."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(-0.7)

    def forward(self, source_joint: torch.Tensor) -> torch.Tensor:
        raw = self.net(source_joint).squeeze(-1)                  # (B, P)
        return raw.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)


def logit_normal_nll(
    loc: torch.Tensor,            # (N,) — pair_frac at the chosen cell (the mean logit)
    log_sigma: torch.Tensor,      # (N,) — per-source confidence
    frac: torch.Tensor,           # (N,) — expert observed fraction in (0,1)
) -> torch.Tensor:
    """NLL of ``f`` under ``f = sigmoid(Normal(loc, sigma))`` (per element).

    -log p(f) = log_sigma + 0.5·log(2π) + (logit f − loc)² / (2 σ²)
                + log f + log(1−f)
    The last two terms are the sigmoid change-of-variables Jacobian; they don't
    depend on (loc, sigma) so they don't affect the gradient, but are kept so
    the reported NLL is a true log-density (comparable across runs)."""
    f = frac.clamp(FRAC_EPS, 1.0 - FRAC_EPS)
    z = torch.logit(f)
    sigma = log_sigma.exp()
    quad = (z - loc) ** 2 / (2.0 * sigma * sigma)
    jac = torch.log(f) + torch.log1p(-f)
    return log_sigma + 0.5 * _LOG_2PI + quad + jac


def single_target_alloc_nll(
    pair_frac: torch.Tensor,      # (B, P, P) — alloc loc head
    frac_sigma_logs: torch.Tensor,  # (B, P) — per-source log-sigma (FracSigmaHead)
    chosen_mask: torch.Tensor,    # (B, P, P) bool — the single chosen target per acted row
    frac_targets: torch.Tensor,   # (B, P, P) — expert fraction of source ships per cell
) -> tuple[torch.Tensor, dict[str, float]]:
    """LogitNormal NLL over the chosen (source -> single target) launches.

    Only acted rows with a chosen off-diagonal target contribute (the select
    CE handles HOLD). loc = pair_frac at the chosen cell; sigma from the source
    row's FracSigmaHead output; target = the expert's observed fraction.
    """
    if not chosen_mask.any():
        return pair_frac.sum() * 0.0, {"empty": 1.0}
    B, P, _ = pair_frac.shape
    bi, si, ti = chosen_mask.nonzero(as_tuple=True)       # (N,)
    loc = pair_frac[bi, si, ti]
    log_sigma = frac_sigma_logs[bi, si]
    frac = frac_targets[bi, si, ti].clamp(FRAC_EPS, 1.0 - FRAC_EPS)
    nll = logit_normal_nll(loc, log_sigma, frac)
    nll_mean = nll.mean()
    with torch.no_grad():
        sigma = log_sigma.exp()
        mean_frac_pred = torch.sigmoid(loc)
        frac_mae = (mean_frac_pred - frac).abs().mean()
    return nll_mean, {
        "alloc_nll": float(nll_mean.detach()),
        "sigma_mean": float(sigma.mean()),
        "sigma_p10": float(sigma.quantile(0.1)) if sigma.numel() > 1 else float(sigma.mean()),
        "frac_mae": float(frac_mae),       # mean |sigmoid(loc) − expert frac|
        "n_launches": int(chosen_mask.sum()),
    }


def sample_frac(loc: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Draw the launch fraction from the LogitNormal (deploy/PPO sampling).
    ``f = sigmoid(loc + sigma·ε)``, ε~N(0,1). Confidence = small sigma -> tight
    around sigmoid(loc)."""
    sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX).exp()
    z = loc + sigma * torch.randn_like(loc)
    return torch.sigmoid(z).clamp(FRAC_EPS, 1.0 - FRAC_EPS)
