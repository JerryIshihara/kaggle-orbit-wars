"""Logprob PARITY TEST for ``bernoulli_select_multinomial_alloc_v2``.

This is the correctness gate for the PPO action-contract migration: a prior
contract mismatch caused PPO divergence, so the stored-vs-recomputed logprob
parity is the definition of done.

    .venv/bin/python -m agents.transformer_v2.ppo.test_multi_target_parity

Or via pytest:

    .venv/bin/python -m pytest agents/transformer_v2/ppo/test_multi_target_parity.py -q

Checks:
  (a) PARITY: ``sample_multi_target`` stores a logprob; rebuilding the action
      and calling ``action_logprob_multi`` with the SAME logits recovers it to
      < 1e-4 (selection, allocation, and total — separately). Over several
      random P and seeds.
  (b) FEASIBILITY: ``Σ_t alloc_counts[s,·] + self_counts[s] == source_ships[s]``
      for every owned source (the multinomial conserves N = S); non-owned rows
      produce nothing.
  (c) SMOKE: one synthetic minibatch with the new fields on a tiny CPU
      ``PPOActorCritic`` runs one ``ppo_minibatch_loss`` forward -> finite loss,
      finite approx_kl, finite entropy.
"""

from __future__ import annotations

import torch

from .loss import (
    PPOMinibatch,
    action_logprob_multi,
    multi_target_entropy,
    ppo_minibatch_loss,
)
from .sampler import legality_masks, sample_multi_target


# --------------------------------------------------------------------------- #
# Synthetic legality / logits                                                 #
# --------------------------------------------------------------------------- #
def _random_setup(P: int, seed: int, *, dtype=torch.float32, device: str = "cpu"):
    """Random (pair_logits, frac_loc, pair_mask, source_mask, source_ships).

    The masks are built via the real ``legality_masks`` from a random owner /
    surplus / exists draw so they obey the off-diagonal + owned-source-row
    invariants the sampler assumes. ``dtype`` controls the logit precision —
    the exact-math parity gate uses float64 (sampler + recompute run at the same
    precision on the same RNG draws, so only the logprob arithmetic differs).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    pair_logits = torch.randn(P, P, generator=g, device=device).to(dtype)
    frac_loc = torch.randn(P, P, generator=g, device=device).to(dtype)

    # Random ownership / surplus / existence -> real legality masks.
    owner = torch.where(
        torch.rand(P, generator=g, device=device) < 0.5,
        torch.zeros(P, dtype=torch.long, device=device),   # me
        torch.ones(P, dtype=torch.long, device=device),    # enemy
    )
    exists = torch.rand(P, generator=g, device=device) > 0.15
    surplus = (torch.rand(P, generator=g, device=device) * 60.0)  # 0..60 ships
    min_launch = 3
    pair_mask, source_mask = legality_masks(
        planet_owner=owner, surplus=surplus, planet_exists=exists,
        min_launch=min_launch,
    )
    # Per-source ship count (>= surplus so it's a plausible integer count).
    source_ships = (surplus + torch.randint(0, 40, (P,), generator=g, device=device)
                    ).round().long()
    return pair_logits, frac_loc, pair_mask, source_mask, source_ships


# --------------------------------------------------------------------------- #
# (a) PARITY + (b) FEASIBILITY                                                 #
# --------------------------------------------------------------------------- #
def _check_one(P: int, seed: int, *, dtype=torch.float64,
               select_logit_bias: float = 0.0) -> dict[str, float]:
    """Sample once and recompute under the SAME logits at ``dtype``.

    The exact-math gate runs at float64 (Δ ~ 1e-12). A separate float32 pass
    (the rollout/train precision) is checked with a RELATIVE tolerance because
    the allocation logprob is ``Σ counts·log p`` over many ships — its magnitude
    can reach hundreds, so float32's ~7-digit precision gives an absolute Δ up
    to ~1e-4 that is pure rounding (and cancels in the PPO ratio, where old_logp
    and new_logp are both float32).

    ``select_logit_bias`` shifts the selection Bernoulli logit; it is passed
    IDENTICALLY to the sampler and to ``action_logprob_multi`` /
    ``multi_target_entropy``. A non-zero bias must still recover the stored
    logprob exactly — a bias applied inconsistently between sample and recompute
    desyncs the PPO ratio, so this is the regression the gate guards.
    """
    torch.manual_seed(seed)
    pair_logits, frac_loc, pair_mask, source_mask, source_ships = _random_setup(
        P, seed, dtype=dtype,
    )

    act = sample_multi_target(
        pair_logits, frac_loc, source_ships,
        pair_mask=pair_mask, source_mask=source_mask,
        select_logit_bias=select_logit_bias,
    )

    # ---- (b) feasibility: counts conserve N = source_ships on owned rows ----
    row_target_sum = act.alloc_counts.sum(dim=1)             # (P,)
    total = row_target_sum + act.self_counts                # (P,)
    owned = source_mask
    if bool(owned.any()):
        assert torch.equal(total[owned], source_ships[owned]), (
            "multinomial must conserve N=S on owned rows:\n"
            f"  total={total[owned].tolist()}\n  S={source_ships[owned].tolist()}"
        )
    # Non-owned rows must produce nothing.
    not_owned = ~owned
    assert int(act.alloc_counts[not_owned].sum().item()) == 0, "non-owned row launched"
    assert int(act.self_counts[not_owned].sum().item()) == 0, "non-owned row held ships"
    # Fired cells must be a subset of legal off-diagonal targets on owned rows.
    assert bool((act.select_mask & ~pair_mask).sum() == 0), "fired an illegal cell"
    # alloc_counts are nonzero only on fired cells.
    assert bool((act.alloc_counts > 0).logical_and(~act.select_mask).sum() == 0), (
        "alloc_counts nonzero off a fired cell"
    )

    # ---- (a) parity: recompute logprob from the SAME logits ----
    # action_logprob_multi expects batched (B,P,P); add a batch dim of 1 and wrap
    # the action fields in a minibatch-shaped namespace.
    class _Act:
        select_mask = act.select_mask.unsqueeze(0)
        alloc_counts = act.alloc_counts.unsqueeze(0)
        self_counts = act.self_counts.unsqueeze(0)

    logp, n_terms = action_logprob_multi(
        pair_logits=pair_logits.unsqueeze(0),
        frac_loc=frac_loc.unsqueeze(0),
        pair_mask=pair_mask.unsqueeze(0),
        source_mask=source_mask.unsqueeze(0),
        act=_Act(),
        select_logit_bias=select_logit_bias,
    )
    recomputed_total = float(logp[0].item())

    # Recompute selection / allocation separately by zeroing the other branch's
    # logits is awkward; instead re-derive each from the same masked math the
    # function uses, to compare against the sampler's stored split.
    # Selection-only recompute: drop allocation by passing zero counts.
    class _ActSelOnly:
        select_mask = act.select_mask.unsqueeze(0)
        alloc_counts = torch.zeros_like(act.alloc_counts).unsqueeze(0)
        self_counts = torch.zeros_like(act.self_counts).unsqueeze(0)

    logp_sel_only, _ = action_logprob_multi(
        pair_logits=pair_logits.unsqueeze(0),
        frac_loc=frac_loc.unsqueeze(0),
        pair_mask=pair_mask.unsqueeze(0),
        source_mask=source_mask.unsqueeze(0),
        act=_ActSelOnly(),
        select_logit_bias=select_logit_bias,
    )
    recomputed_select = float(logp_sel_only[0].item())
    recomputed_alloc = recomputed_total - recomputed_select

    stored_total = float(act.logprob.item())
    stored_select = float(act.logprob_select.item())
    stored_alloc = float(act.logprob_alloc.item())

    d_total = abs(stored_total - recomputed_total)
    d_select = abs(stored_select - recomputed_select)
    d_alloc = abs(stored_alloc - recomputed_alloc)

    if dtype == torch.float64:
        # Exact-math gate: identical arithmetic -> agreement at float64 epsilon.
        tol = 1e-6
        assert d_select < tol, f"P={P} seed={seed} selection Δ={d_select:.2e}"
        assert d_alloc < tol, f"P={P} seed={seed} allocation Δ={d_alloc:.2e}"
        assert d_total < tol, f"P={P} seed={seed} total Δ={d_total:.2e}"
    else:
        # float32 runtime gate: absolute Δ < 1e-4 OR relative Δ < 1e-5 (the
        # large-magnitude allocation sum rounds at ~float32 epsilon).
        def _ok(d, mag):
            return d < 1e-4 or d < 1e-5 * max(abs(mag), 1.0)
        assert _ok(d_select, stored_select), f"P={P} seed={seed} sel Δ={d_select:.2e}"
        assert _ok(d_alloc, stored_alloc), f"P={P} seed={seed} alloc Δ={d_alloc:.2e}"
        assert _ok(d_total, stored_total), f"P={P} seed={seed} total Δ={d_total:.2e}"

    # n_terms must match the sampler's stored component count. The recompute
    # applies the documented clamp_min(1.0) safety floor (avoids div-by-zero in
    # the per-component KL), so compare against max(stored, 1).
    assert int(n_terms[0].item()) == max(int(act.n_terms), 1), (
        f"n_terms mismatch: recompute={int(n_terms[0].item())} "
        f"stored={act.n_terms} (clamped {max(int(act.n_terms), 1)})"
    )

    # entropy is finite and non-negative.
    ent = multi_target_entropy(
        pair_logits.unsqueeze(0), frac_loc.unsqueeze(0),
        pair_mask=pair_mask.unsqueeze(0), source_mask=source_mask.unsqueeze(0),
        act=_Act(),
        select_logit_bias=select_logit_bias,
    )
    assert torch.isfinite(ent).all() and float(ent[0].item()) >= -1e-6, "bad entropy"

    return {"d_total": d_total, "d_select": d_select, "d_alloc": d_alloc,
            "n_owned": int(owned.sum().item()),
            "n_fired": int(act.select_mask.sum().item())}


def test_parity_and_feasibility() -> None:
    # Exercise BOTH the contract default (bias 0.0) and a non-zero selection
    # logit bias (2.0 — mirrors the runner's pair_logits>2.0 fire threshold).
    # A bias applied inconsistently between sample and recompute silently
    # desyncs the PPO ratio, so the non-zero pass is the regression gate.
    for select_logit_bias in (0.0, 2.0):
        for dtype, label in ((torch.float64, "float64 exact-math gate"),
                             (torch.float32, "float32 runtime gate")):
            max_d = 0.0
            n_cases = 0
            for P in (3, 6, 9, 16, 24):
                for seed in range(8):
                    r = _check_one(P, seed=1000 * P + seed, dtype=dtype,
                                   select_logit_bias=select_logit_bias)
                    max_d = max(max_d, r["d_total"], r["d_select"], r["d_alloc"])
                    n_cases += 1
            print(f"  [ok] bias={select_logit_bias:.1f} {label}: "
                  f"parity+feasibility over {n_cases} cases — "
                  f"max |Δlogp| = {max_d:.3e} (max over selection/allocation/total)")


# --------------------------------------------------------------------------- #
# (c) ppo_minibatch_loss smoke                                                 #
# --------------------------------------------------------------------------- #
class _StubPolicy(torch.nn.Module):
    """Minimal stand-in for ``_PPOWithL0(PPOActorCritic)``.

    ``ppo_minibatch_loss`` only consumes ``out["value"]``, ``out["pair_logits"]``,
    ``out["frac_loc"]`` and ``out["sigma"]`` from the policy forward. A tiny
    learnable map from the per-planet features to those heads is enough to drive
    the loss / KL / entropy math (the exact transformer internals are validated
    elsewhere). Keeping a real ``PPOActorCritic`` forward here would require the
    on-disk L0 specialist ckpts; this isolates the changed loss branch on CPU.

    The forward is parameterized so a fresh policy reproduces the rollout logits
    deterministically — we seed it identically when sampling old_logp, so the
    PPO ratio starts at exactly 1 (the unchanged-policy invariant).
    """

    def __init__(self, P: int, d_in: int, sigma: float = 0.35):
        super().__init__()
        self.P = P
        self.pair = torch.nn.Linear(d_in, P)       # (..,P,d)->(..,P,P) row logits
        self.frac = torch.nn.Linear(d_in, P)        # (..,P,d)->(..,P,P) alloc logits
        self.val = torch.nn.Linear(d_in, 1)
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    def forward(self, planet_features, fleet_features=None, planet_mask=None,
                is_comet=None, pair_type_ids=None, routing=None, phi=None):
        x = planet_features
        if x.dim() == 4:           # (B,T,P,d) -> use current frame
            x = x[:, -1]
        pair_logits = self.pair(x)                  # (B,P,P)
        frac_loc = self.frac(x)                     # (B,P,P)
        value = self.val(x).squeeze(-1).mean(dim=1)  # (B,)
        return {
            "value": value,
            "pair_logits": pair_logits,
            "frac_loc": frac_loc,
            "sigma": self.sigma,
        }


def _run_minibatch_smoke(select_logit_bias: float) -> None:
    B, P, d_in = 4, 6, 12
    torch.manual_seed(11)
    policy = _StubPolicy(P, d_in, sigma=0.35)
    policy.eval()

    g = torch.Generator().manual_seed(7)
    planet_features = torch.randn(B, P, d_in, generator=g)
    feats = {
        "planet_features": planet_features,
        "fleet_features": torch.zeros(B, 1, 1),
        "planet_mask": torch.ones(B, P, dtype=torch.bool),
        "is_comet": torch.zeros(B, P, dtype=torch.bool),
        "pair_type_ids": torch.zeros(B, P, P, dtype=torch.long),
        "routing": {},
        "phi": torch.zeros(B),
    }

    # Forward once to get the rollout-time pair_logits / frac_loc, then sample a
    # self-consistent multi-target action per row so old_logp == new_logp at the
    # first PPO step (ratio == 1). The sampler and the loss recompute MUST use
    # the SAME ``select_logit_bias`` for the ratio to start at exactly 1.
    with torch.no_grad():
        out0 = policy(**feats)
        pair_logits = out0["pair_logits"]
        frac_loc = out0["frac_loc"]

    select_mask = torch.zeros(B, P, P, dtype=torch.bool)
    alloc_counts = torch.zeros(B, P, P, dtype=torch.long)
    self_counts = torch.zeros(B, P, dtype=torch.long)
    pair_mask_b = torch.zeros(B, P, P, dtype=torch.bool)
    source_mask_b = torch.zeros(B, P, dtype=torch.bool)
    old_logp = torch.zeros(B)
    for b in range(B):
        owner = torch.where(torch.rand(P, generator=g) < 0.5,
                            torch.zeros(P, dtype=torch.long),
                            torch.ones(P, dtype=torch.long))
        exists = torch.rand(P, generator=g) > 0.15
        surplus = torch.rand(P, generator=g) * 50.0
        pm, sm = legality_masks(planet_owner=owner, surplus=surplus,
                                planet_exists=exists, min_launch=3)
        ssh = (surplus + 10).round().long()
        a = sample_multi_target(pair_logits[b], frac_loc[b], ssh,
                                pair_mask=pm, source_mask=sm,
                                select_logit_bias=select_logit_bias)
        select_mask[b] = a.select_mask
        alloc_counts[b] = a.alloc_counts
        self_counts[b] = a.self_counts
        pair_mask_b[b] = pm
        source_mask_b[b] = sm
        old_logp[b] = a.logprob

    mb = PPOMinibatch(
        feats=feats,
        pair_mask=pair_mask_b,
        source_mask=source_mask_b,
        old_logp=old_logp,
        adv=torch.randn(B),
        returns=torch.randn(B),
        select_mask=select_mask,
        alloc_counts=alloc_counts,
        self_counts=self_counts,
    )
    assert mb.is_multi_target and mb.size == B

    loss, diag = ppo_minibatch_loss(
        policy, mb, None,
        clip=0.10, value_coef=0.5, ent_coef=0.01, bc_coef=0.0,
        collect_shape=True,
        select_logit_bias=select_logit_bias,
    )
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    for k in ("approx_kl", "entropy", "policy_loss", "value_loss", "ratio_mean"):
        assert k in diag and (diag[k] == diag[k]) and abs(diag[k]) < 1e9, (
            f"bad diag[{k}] = {diag.get(k)}"
        )
    # Unchanged-policy invariant: ratio == 1 and approx_kl == 0 at the first step
    # (holds for ANY bias as long as sampler and recompute share it).
    assert abs(diag["ratio_mean"] - 1.0) < 1e-4, f"ratio_mean={diag['ratio_mean']}"
    assert abs(diag["approx_kl"]) < 1e-4, f"approx_kl={diag['approx_kl']}"
    print(f"  [ok] ppo_minibatch_loss smoke bias={select_logit_bias:.1f} — "
          f"loss={float(loss.detach()):.4f} "
          f"approx_kl={diag['approx_kl']:.4e} entropy={diag['entropy']:.4f} "
          f"policy_loss={diag['policy_loss']:.4f} value_loss={diag['value_loss']:.4f} "
          f"ratio_mean={diag['ratio_mean']:.6f}")


def test_ppo_minibatch_loss_smoke() -> None:
    # Finite loss + ratio==1 invariant at both the default and a non-zero bias.
    for select_logit_bias in (0.0, 2.0):
        _run_minibatch_smoke(select_logit_bias)


def test_bounded_k_v3_parity() -> None:
    """bounded_k_select_multinomial_alloc_v3: stored logprob == recomputed
    (coefficient-free sum counts*log p both sides), n_terms matched,
    floor feasibility + exact ship-budget conservation on acting rows."""
    import torch as _t
    from .sampler import sample_bounded_k, legality_masks as _lm
    from .loss import action_logprob_bounded_k

    P, m = 8, 5
    worst = 0.0
    for case in range(40):
        g = _t.Generator().manual_seed(case)
        pair_logits = _t.randn(P, P, generator=g)
        frac_loc = _t.randn(P, P, generator=g)
        owner = _t.randint(0, 3, (P,), generator=g)
        surplus = _t.randint(0, 60, (P,)).float()
        exists = _t.rand(P, generator=g) > 0.15
        pm, sm = _lm(owner, surplus, exists, min_launch=m)
        ships = (surplus + _t.randint(0, 40, (P,)).float()).long()
        for bias in (0.0, 1.0):
            act = sample_bounded_k(
                pair_logits, frac_loc, ships, pair_mask=pm, source_mask=sm,
                min_launch=m, k_max=4, self_logit_bias=bias)

            class _MB:  # minimal act-carrier
                pass
            mb = _MB()
            mb.select_counts = act.select_counts.unsqueeze(0)
            mb.alloc_extras = act.alloc_extras.unsqueeze(0)
            mb.self_extras = act.self_extras.unsqueeze(0)
            logp, n_terms = action_logprob_bounded_k(
                pair_logits.unsqueeze(0), frac_loc.unsqueeze(0),
                pair_mask=pm.unsqueeze(0), source_mask=sm.unsqueeze(0),
                act=mb, self_logit_bias=bias)
            worst = max(worst, abs(float(logp[0]) - float(act.logprob)))
            assert abs(float(n_terms[0]) - max(1, act.n_terms)) == 0
            fired = act.select_counts[:, :P] >= 1
            assert (m + act.alloc_extras[fired] >= m).all()
            spent = m * fired.sum(1) + act.alloc_extras.sum(1) + act.self_extras
            has_f = fired.any(1)
            assert (spent[has_f] == ships[has_f]).all()
    assert worst < 1e-4, worst
    print(f"  [ok] bounded_k v3 parity over 80 cases — max |dlogp| = {worst:.3e}")



if __name__ == "__main__":
    test_parity_and_feasibility()
    test_ppo_minibatch_loss_smoke()
    test_bounded_k_v3_parity()
    print("MULTI-TARGET PARITY PASSED — "
          "stored logprob == recomputed logprob (< 1e-4); "
          "feasibility holds; ppo_minibatch_loss finite; v3 bounded-k exact")
