"""bernoulli_select_multinomial_alloc_v2 pretrain labels — exact-share and
loss-wiring tests over a tiny synthetic snapshot.

Layout (P=6, planet 5 is padding):
  row 0  owned, N=100, launches 0->2 (30), 0->3 (50), plus a ships=0
         positive 0->4  -> shares .30/.50, HOLD .20; ships-0 cell dropped
  row 1  owned, N=50, holds            -> select-only row, no alloc target
  row 2  NOT owned, has a positive     -> excluded by the owned mask
  row 3  owned, N=10, launch 3->1 (30) -> overflow: share 1.0, HOLD 0.0
  row 4  owned, N=80, only a ships=0 positive -> row dropped
"""

from __future__ import annotations

import math
from collections import Counter

import torch

from agents.transformer_v2.featurizer.fleet_featurizer import SHIPS_LOG_MAX
from agents.transformer_v2.pretrain.alloc_labels import (
    alloc_multinomial_ce,
    build_alloc_targets,
)
from agents.transformer_v2.pretrain.entity_encoder import (
    _PLANET_OWNER_START_IDX,
    _PLANET_SHIPS_LOG_FEATURE_IDX,
    compute_multi_loss,
)

P = 6
D = _PLANET_SHIPS_LOG_FEATURE_IDX + 4


def _snapshot() -> dict[str, torch.Tensor]:
    pf = torch.zeros(P, D)
    mask = torch.tensor([True, True, True, True, True, False])
    owned_rows = (0, 1, 3, 4)
    for s in owned_rows:
        pf[s, _PLANET_OWNER_START_IDX] = 1.0
    for s, n in ((0, 100), (1, 50), (2, 60), (3, 10), (4, 80)):
        pf[s, _PLANET_SHIPS_LOG_FEATURE_IDX] = math.log1p(n) / SHIPS_LOG_MAX

    labels = torch.zeros(P, P, dtype=torch.bool)
    ships = torch.zeros(P, P, dtype=torch.int32)
    labels[0, 2] = True; ships[0, 2] = 30
    labels[0, 3] = True; ships[0, 3] = 50
    labels[0, 4] = True  # ships stays 0 — dropped cell
    labels[2, 4] = True; ships[2, 4] = 10   # unowned row
    labels[3, 1] = True; ships[3, 1] = 30   # overflow (N=10)
    labels[4, 2] = True                      # ships=0 only — dropped row

    valid = mask.unsqueeze(1) & mask.unsqueeze(0)
    valid.fill_diagonal_(False)
    return {
        "pair_labels": labels,
        "pair_valid": valid,
        "pair_ships": ships,
        "planet_features": pf,
        "planet_mask": mask,
    }


def test_build_alloc_targets_exact_shares():
    stats: Counter[str] = Counter()
    row_mask, target = build_alloc_targets(_snapshot(), stats=stats)

    assert row_mask.tolist() == [True, False, False, True, False, False]
    assert target.shape == (P, P + 1)

    torch.testing.assert_close(target[0, 2], torch.tensor(0.30), atol=1e-5, rtol=0)
    torch.testing.assert_close(target[0, 3], torch.tensor(0.50), atol=1e-5, rtol=0)
    assert target[0, 4] == 0.0                       # ships=0 cell carries no share
    torch.testing.assert_close(target[0, P], torch.tensor(0.20), atol=1e-5, rtol=0)
    torch.testing.assert_close(target[0].sum(), torch.tensor(1.0), atol=1e-5, rtol=0)

    # Overflow row: sent (30) > recovered N (10) -> renormalize, HOLD = 0.
    torch.testing.assert_close(target[3, 1], torch.tensor(1.0), atol=1e-5, rtol=0)
    assert target[3, P] == 0.0

    assert target[1].sum() == 0.0                    # held row: no alloc target
    assert target[2].sum() == 0.0                    # unowned row excluded
    assert target[4].sum() == 0.0                    # all-ships-0 row dropped

    assert stats["owned_rows"] == 4
    assert stats["acted_rows"] == 3                  # rows 0, 3, 4 (owned & positive)
    assert stats["supervised_rows"] == 2             # rows 0, 3
    assert stats["fired_cells"] == 3                 # (0,2) (0,3) (3,1)
    assert stats["dropped_rows_ships0"] == 1         # row 4
    assert stats["dropped_cells_ships0"] == 1        # (0,4)
    assert stats["overflow_rows"] == 1               # row 3
    assert stats["dropped_rows_no_src"] == 0


def test_batched_matches_single():
    snap = _snapshot()
    batch = {k: torch.stack([v, v]) for k, v in snap.items()}
    row_mask_b, target_b = build_alloc_targets(batch)
    row_mask_s, target_s = build_alloc_targets(snap)
    assert row_mask_b.shape == (2, P)
    torch.testing.assert_close(target_b[0], target_s)
    torch.testing.assert_close(target_b[1], target_s)


def test_alloc_ce_matches_manual_and_grads_reach_diagonal():
    snap = _snapshot()
    batch = {k: v.unsqueeze(0) for k, v in snap.items()}
    torch.manual_seed(0)
    frac_loc = torch.randn(1, P, P, requires_grad=True)

    loss, diag = alloc_multinomial_ce(frac_loc, batch)

    # Manual CE, row 0: softmax over [frac(0,2), frac(0,3), frac-diag(0,0)].
    lp0 = torch.log_softmax(
        torch.stack([frac_loc[0, 0, 2], frac_loc[0, 0, 3], frac_loc[0, 0, 0]]),
        dim=0,
    )
    ce0 = -(0.30 * lp0[0] + 0.50 * lp0[1] + 0.20 * lp0[2])
    # Row 3: softmax over [frac(3,1), frac-diag(3,3)], target [1.0, 0.0].
    lp3 = torch.log_softmax(
        torch.stack([frac_loc[0, 3, 1], frac_loc[0, 3, 3]]), dim=0,
    )
    ce3 = -lp3[0]
    torch.testing.assert_close(loss, (ce0 + ce3) / 2, atol=1e-5, rtol=0)
    assert diag["alloc_rows"] == 2
    assert 0.0 <= diag["hold_share_label"] <= 1.0

    loss.backward()
    # The HOLD diagonal (frac head, v2) of supervised rows gets gradient —
    # the whole point.
    assert frac_loc.grad[0, 0, 0].abs() > 0
    assert frac_loc.grad[0, 3, 3].abs() > 0
    # Unsupervised rows' diagonals stay untouched by this loss.
    assert frac_loc.grad[0, 1, 1] == 0
    assert frac_loc.grad[0, 4, 4] == 0
    # frac_loc gradient lands exactly on the expert fired cells.
    assert frac_loc.grad[0, 0, 2].abs() > 0
    assert frac_loc.grad[0, 0, 3].abs() > 0
    assert frac_loc.grad[0, 0, 4] == 0               # ships=0 cell: no gradient
    assert frac_loc.grad[0, 2, 4] == 0               # unowned row: no gradient


def test_compute_multi_loss_multinomial_alloc_branch():
    snap = _snapshot()
    batch = {k: v.unsqueeze(0) for k, v in snap.items()}
    torch.manual_seed(1)
    preds = {
        "pair_logits": torch.randn(1, P, P, requires_grad=True),
        "pair_frac": torch.randn(1, P, P, requires_grad=True),
    }
    total, per_head = compute_multi_loss(
        preds, batch, multinomial_alloc=True, pair_pos_weight=10.0,
    )
    assert torch.isfinite(total)
    assert "pair_frac" in per_head and math.isfinite(per_head["pair_frac"])
    assert "hold_share_label" in per_head and "hold_mae" in per_head
    # Legacy source-categorical alignment terms must be skipped.
    assert "ppo_source" not in per_head and "ppo_target" not in per_head
    total.backward()
    assert preds["pair_logits"].grad is not None        # select BCE
    assert preds["pair_frac"].grad is not None          # alloc CE
    # v2 decoupling: the alloc CE must NOT reach the select head — its
    # diagonal (dead in this contract) stays gradient-free.
    diag_grad = preds["pair_logits"].grad.diagonal(dim1=-2, dim2=-1)
    assert diag_grad.abs().sum() == 0
    # The frac head's diagonal (HOLD) on supervised rows DOES get gradient.
    assert preds["pair_frac"].grad[0, 0, 0].abs() > 0

    try:
        compute_multi_loss(preds, batch, single_target=True, multinomial_alloc=True)
    except ValueError as e:
        assert "mutually exclusive" in str(e)
    else:
        raise AssertionError("single_target + multinomial_alloc must raise")


def test_empty_batch_returns_gradfree_zero():
    snap = _snapshot()
    snap = {**snap, "pair_labels": torch.zeros(P, P, dtype=torch.bool)}
    batch = {k: v.unsqueeze(0) for k, v in snap.items()}
    frac_loc = torch.randn(1, P, P, requires_grad=True)
    loss, diag = alloc_multinomial_ce(frac_loc, batch)
    assert float(loss) == 0.0 and not loss.requires_grad
    assert diag["alloc_rows"] == 0
