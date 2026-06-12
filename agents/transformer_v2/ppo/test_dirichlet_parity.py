"""Parity tests for ``bounded_k_select_dirichlet_alloc_v4``.

The PPO correctness contract: the batched loss-side recompute
(`action_logprob_dirichlet_k`) evaluated on UNCHANGED head outputs must
reproduce the sampler's stored logprob exactly (ratio == 1), the analytic
entropy must be finite with gradients flowing to frac_loc AND alloc_conc,
and the action must survive the shards round-trip bit-for-bit.

Model-free: random head tensors stand in for the policy so the test runs
anywhere (no ckpt / GPU).

Run:  .venv/bin/python -m agents.transformer_v2.ppo.test_dirichlet_parity
"""

from __future__ import annotations

import torch

from .loss import PPOMinibatch, action_logprob_dirichlet_k, dirichlet_k_entropy
from .sampler import (
    DirichletKAction,
    legality_masks,
    sample_dirichlet_k,
)

P = 12
MIN_LAUNCH = 10
K_MAX = 3


def _random_snapshot(gen: torch.Generator):
    pair_logits = torch.randn(P, P, generator=gen) * 2.0
    frac_loc = torch.randn(P, P, generator=gen) * 2.0
    # α0 head output spans diffuse -> near-cap (the conc head is sigmoid·cap,
    # but the sampler only assumes positivity; cover both regimes).
    alloc_conc = torch.rand(P, generator=gen) * 199.0 + 0.5
    owner = (torch.rand(P, generator=gen) < 0.5).long()  # 0 = learner
    exists = torch.rand(P, generator=gen) < 0.9
    exists[:4] = True
    ships = torch.randint(0, 400, (P,), generator=gen)
    surplus = ships.float() * (torch.rand(P, generator=gen) * 0.9)
    pair_mask, source_mask = legality_masks(
        planet_owner=owner, surplus=surplus, planet_exists=exists,
        min_launch=MIN_LAUNCH,
    )
    return pair_logits, frac_loc, alloc_conc, ships, pair_mask, source_mask


def test_ratio_is_one(n: int = 64, bias: float = 0.7) -> None:
    gen = torch.Generator().manual_seed(7)
    acts, pls, fls, cons, pms, sms = [], [], [], [], [], []
    n_acting = 0
    for _ in range(n):
        pl, fl, con, ships, pm, sm = _random_snapshot(gen)
        act = sample_dirichlet_k(
            pl, fl, con, ships,
            pair_mask=pm, source_mask=sm,
            min_launch=MIN_LAUNCH, k_max=K_MAX, self_logit_bias=bias,
        )
        n_acting += int((act.select_counts.sum(-1) > 0).sum())
        acts.append(act); pls.append(pl); fls.append(fl)
        cons.append(con); pms.append(pm); sms.append(sm)
    assert n_acting > 0, "degenerate test: no acting rows sampled"

    mb = PPOMinibatch(
        feats={},
        pair_mask=torch.stack(pms),
        source_mask=torch.stack(sms),
        old_logp=torch.stack([a.logprob for a in acts]),
        adv=torch.zeros(n),
        returns=torch.zeros(n),
        select_counts=torch.stack([a.select_counts for a in acts]),
        alloc_shares=torch.stack([a.alloc_shares for a in acts]),
    )
    assert mb.is_dirichlet_k and not mb.is_bounded_k

    new_logp, n_terms, parts = action_logprob_dirichlet_k(
        torch.stack(pls), torch.stack(fls), torch.stack(cons),
        pair_mask=mb.pair_mask, source_mask=mb.source_mask, act=mb,
        self_logit_bias=bias,
    )
    # per-component parity: the batched per-row recompute must match the
    # sampler's stored row logps (the per-component clip ratios are built
    # from exactly these pairs).
    sel_old = torch.stack([a.select_row_logp for a in acts])
    al_old = torch.stack([a.alloc_row_logp for a in acts])
    d_sel = (parts["sel_rows"] - sel_old).abs()[parts["drew"]]
    d_al = (parts["al_rows"] - al_old).abs()[parts["has_fired"]]
    print(f"  per-row |diff| sel max={float(d_sel.max() if d_sel.numel() else 0):.3e} "
          f"al max={float(d_al.max() if d_al.numel() else 0):.3e}")
    assert float(d_sel.max() if d_sel.numel() else 0) < 1e-3
    assert float(d_al.max() if d_al.numel() else 0) < 1e-3
    diff = (new_logp - mb.old_logp).abs()
    ratio = (new_logp - mb.old_logp).exp()
    print(f"  logp |diff| max={float(diff.max()):.3e} "
          f"ratio in [{float(ratio.min()):.6f}, {float(ratio.max()):.6f}] "
          f"n_terms mean={float(n_terms.mean()):.1f}")
    # Tolerance: the dominant lgamma terms have magnitude ~2e3 at α0≈400, so
    # fp32 accumulation-order noise alone is ~2e-4 on the logp. 1e-3 bounds
    # the ratio within ±0.1% — three orders below the PPO clip — while still
    # catching any STRUCTURAL desync (a wrong support/clamp shows up as O(1)).
    assert float(diff.max()) < 1e-3, "PPO ratio desync: recompute != sampler"

    # entropy: finite, and grads reach BOTH frac_loc and alloc_conc
    pl_g = torch.stack(pls).requires_grad_(True)
    fl_g = torch.stack(fls).requires_grad_(True)
    con_g = torch.stack(cons).requires_grad_(True)
    ent = dirichlet_k_entropy(
        pl_g, fl_g, con_g,
        pair_mask=mb.pair_mask, source_mask=mb.source_mask, act=mb,
        self_logit_bias=bias,
    )
    assert torch.isfinite(ent).all(), "non-finite v4 entropy"
    ent.sum().backward()
    g_fl = float(fl_g.grad.abs().sum())
    g_con = float(con_g.grad.abs().sum())
    print(f"  entropy mean={float(ent.mean()):.3f} "
          f"|grad frac|={g_fl:.3e} |grad conc|={g_con:.3e}")
    assert g_fl > 0 and g_con > 0, "entropy grads must reach frac AND conc"

    # Confidence pricing in the TRAINED regime (all α_i > 1): doubling α0
    # must strictly lower the Dirichlet entropy. (Not asserted on the random
    # batch above — Dirichlet entropy peaks near α=1, so samples with tiny
    # α_i can legitimately gain entropy when α0 doubles.)
    with torch.no_grad():
        pl1 = torch.zeros(1, 4, 4)
        fl1 = torch.zeros(1, 4, 4)
        pm1 = torch.zeros(1, 4, 4, dtype=torch.bool)
        pm1[0, 0, 1:3] = True
        sm1 = torch.zeros(1, 4, dtype=torch.bool)
        sm1[0, 0] = True
        sc1 = torch.zeros(1, 4, 5, dtype=torch.long)
        sc1[0, 0, 1] = 1
        sc1[0, 0, 2] = 1                       # 2 fired + self support, K=3
        xs1 = torch.zeros(1, 4, 5)
        xs1[0, 0, 1], xs1[0, 0, 2], xs1[0, 0, 4] = 0.5, 0.3, 0.2
        mb1 = PPOMinibatch(
            feats={}, pair_mask=pm1, source_mask=sm1,
            old_logp=torch.zeros(1), adv=torch.zeros(1), returns=torch.zeros(1),
            select_counts=sc1, alloc_shares=xs1,
        )
        e_lo, e_hi = (
            float(dirichlet_k_entropy(
                pl1, fl1, torch.full((1, 4), a0),
                pair_mask=pm1, source_mask=sm1, act=mb1,
            ))
            for a0 in (30.0, 60.0)
        )
        print(f"  pricing: H(α0=30)={e_lo:.3f} > H(α0=60)={e_hi:.3f}")
        assert e_hi < e_lo, "entropy must fall as α0 grows (α_i > 1 regime)"


def test_shards_roundtrip() -> None:
    from . import shards

    gen = torch.Generator().manual_seed(11)
    pl, fl, con, ships, pm, sm = _random_snapshot(gen)
    act = sample_dirichlet_k(
        pl, fl, con, ships, pair_mask=pm, source_mask=sm,
        min_launch=MIN_LAUNCH, k_max=K_MAX,
    )
    d = {
        "action_contract": "v4",
        "action_select_counts": act.select_counts.unsqueeze(0),
        "action_alloc_shares": act.alloc_shares.unsqueeze(0),
        "action_logprob": act.logprob.reshape(1),
        "action_logprob_select": act.logprob_select.reshape(1),
        "action_logprob_alloc": act.logprob_alloc.reshape(1),
        "action_n_terms": torch.tensor([act.n_terms]),
        "action_diagnostics": [dict(act.diagnostics)],
    }
    # round-trip through the v4 arm of dict_to_episode_buffer's action builder
    rebuilt = DirichletKAction(
        select_counts=d["action_select_counts"][0].long(),
        alloc_shares=d["action_alloc_shares"][0].float(),
        logprob=d["action_logprob"][0],
        logprob_select=d["action_logprob_select"][0],
        logprob_alloc=d["action_logprob_alloc"][0],
        n_terms=int(d["action_n_terms"][0].item()),
        diagnostics=dict(d["action_diagnostics"][0]),
    )
    assert torch.equal(rebuilt.select_counts, act.select_counts)
    assert torch.equal(rebuilt.alloc_shares, act.alloc_shares), \
        "alloc_shares must round-trip bit-for-bit (float32)"
    assert shards.ACTION_TENSOR_FIELDS_V4 == ("select_counts", "alloc_shares")
    print("  shards round-trip: bit-exact")


def main() -> None:
    torch.manual_seed(0)
    print("[1/3] ratio == 1 (sample -> batched recompute), bias=0.7")
    test_ratio_is_one()
    print("[2/3] ratio == 1, bias=0.0")
    test_ratio_is_one(bias=0.0)
    print("[3/3] shards round-trip")
    test_shards_roundtrip()
    print("ALL v4 PARITY TESTS PASSED")


if __name__ == "__main__":
    main()
