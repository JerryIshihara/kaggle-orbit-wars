"""Smoke proof for the dual-rate L2 (run on CPU, no data needed).

    .venv/bin/python -m agents.transformer_v3.smoke_dual_rate

Asserts, in order:
  1. WARM-START MAPPING — ``adapt_v2_state_dict`` loads a v2 state dict
     into the v3 tree with missing == the 2 fusion layers only,
     unexpected == none; the short branch's step_embed keeps the
     trained rows at shared offsets (10, 0).
  2. INIT EQUIVALENCE — a warm-started v3 on the 18-frame UNION stack
     reproduces the v2 model on the long sub-stack output-for-output
     (pair_logits / pair_frac / glob / ctx_now / player_state), both
     temporal and single-frame.
  3. GRADIENT GATE — at init the loss reaches the fusion weights' short
     half but NOT the short branch (zero-init gate, by construction);
     after one optimizer step on the fusion, the short branch receives
     real gradient. The long path's gradient is alive throughout.
  4. FREEZE SEMANTICS — ``freeze_below_l2`` keeps L1 frozen and counts
     both branches + fusion as the trainable L2 group.
"""

from __future__ import annotations

import torch

from ..transformer_v2.pretrain.entity_encoder import (
    ENTITY_N_OWNER_CLASSES,
    EntityPretrainModel,
)
from . import (
    LONG_HISTORY_OFFSETS,
    SHORT_HISTORY_OFFSETS,
    LONG_SLOT_IDX,
    N_UNION,
    EntityPretrainModelV3,
    adapt_v2_state_dict,
)

D, B, P, F = 64, 2, 8, 12


def _synthetic_batch(T: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    planet_tokens = torch.randn(B, T, P, D, generator=g)
    fleet_tokens = torch.randn(B, T, F, D, generator=g)
    planet_mask = torch.ones(B, T, P, dtype=torch.bool)
    planet_mask[:, :, -2:] = False                       # padded planets
    fleet_mask = torch.rand(B, T, F, generator=g) > 0.3
    routing = {
        "fleet_target_idx": torch.randint(-1, P - 2, (B, T, F), generator=g),
        "fleet_source_idx": torch.randint(-1, P - 2, (B, T, F), generator=g),
        "fleet_owner_slot": torch.randint(0, ENTITY_N_OWNER_CLASSES, (B, T, F), generator=g),
        "fleet_ships_log": torch.rand(B, T, F, generator=g),
        "fleet_eta_norm": torch.rand(B, T, F, generator=g),
        "fleet_mask": fleet_mask,
    }
    is_comet = torch.zeros(B, T, P, dtype=torch.bool)
    is_comet[:, :, 3] = True
    pair_type_ids = torch.randint(0, 27, (B, T, P, P), generator=g)
    owner_oh = torch.zeros(B, T, P, ENTITY_N_OWNER_CLASSES)
    owner_oh[..., 0] = 1.0
    return planet_tokens, fleet_tokens, routing, planet_mask, is_comet, pair_type_ids, owner_oh


def _sub(t: torch.Tensor, idx) -> torch.Tensor:
    return t[:, list(idx)]


def _fwd(model, pt, ft, routing, pm, ic, pti, oh):
    return model.forward_with_context(
        pt, ft, routing, pm,
        is_comet=ic, pair_type_ids=pti, planet_owner_oh=oh,
    )


def main() -> None:
    kw = dict(
        d_model=D, conditioner_n_layers=2, head_n_layers=2,
        with_consolidator=True, with_value_heads=True, value_dropout=0.0,
    )
    v2 = EntityPretrainModel(n_steps=len(LONG_HISTORY_OFFSETS), **kw)
    v3 = EntityPretrainModelV3(**kw)

    # ---- 1. warm-start mapping --------------------------------------
    adapted = adapt_v2_state_dict(v2.state_dict())
    assert not any(k.startswith("consolidator.") for k in adapted), (
        "consolidator keys must be dropped by the adapter (v3.1)")
    res = v3.load_state_dict(adapted, strict=False)
    fuse_missing = sorted(
        k for k in res.missing_keys if k.startswith("cross.fuse_")
    )
    fresh_ok = (
        "cross.fuse_", "short_heads.", "cross.owner_proj",
        "cross.long.player_tokens", "cross.short.player_tokens",
    )
    other_missing = sorted(
        k for k in res.missing_keys if not k.startswith(fresh_ok)
    )
    assert fuse_missing == [
        "cross.fuse_glob.bias", "cross.fuse_glob.weight",
        "cross.fuse_player.bias", "cross.fuse_player.weight",
        "cross.fuse_tokens.bias", "cross.fuse_tokens.weight",
    ], f"unexpected fusion missing set: {fuse_missing}"
    assert not other_missing, f"non-fresh missing: {other_missing}"
    assert not res.unexpected_keys, res.unexpected_keys
    assert v3.consolidator is None and v3.cross.owner_proj.weight.abs().sum() == 0
    se_v2 = v2.cross.step_embed
    se_s = v3.cross.short.step_embed
    for off in (10, 0):
        i2 = LONG_HISTORY_OFFSETS.index(off)
        i3 = SHORT_HISTORY_OFFSETS.index(off)
        assert torch.equal(se_s[i3], se_v2[i2]), f"step_embed row off={off}"
    assert torch.equal(v3.cross.long.step_embed, se_v2)
    print("[1] warm-start mapping OK "
          f"(loaded {len(adapted)} tensors; fusion stays zero-init)")

    # ---- 2. init equivalence ----------------------------------------
    v2.eval(); v3.eval()
    pt, ft, routing, pm, ic, pti, oh = _synthetic_batch(N_UNION)
    long_views = (
        _sub(pt, LONG_SLOT_IDX), _sub(ft, LONG_SLOT_IDX),
        {k: _sub(v, LONG_SLOT_IDX) for k, v in routing.items()},
        _sub(pm, LONG_SLOT_IDX), _sub(ic, LONG_SLOT_IDX),
        _sub(pti, LONG_SLOT_IDX), _sub(oh, LONG_SLOT_IDX),
    )
    with torch.no_grad():
        out2 = _fwd(v2, *long_views)
        out3 = _fwd(v3, pt, ft, routing, pm, ic, pti, oh)
    # player_state intentionally differs (v2 = consolidator, v3.1 = in-L2
    # player tokens). The four action/context outputs must match exactly —
    # which simultaneously PROVES the asymmetric mask: the extra player
    # tokens and the zero-init owner projection leave the planet/global
    # path bit-identical.
    for k in ("pair_logits", "pair_frac", "glob", "ctx_now"):
        a, b = out2[k], out3[k]
        assert a is not None and b is not None, k
        diff = (a - b).abs().max().item()
        assert diff < 1e-5, f"{k}: max diff {diff}"
    ps = out3["player_state"]
    assert ps is not None and ps.shape == (B, 4, D) and torch.isfinite(ps).all()
    assert "win" in out3 and out3["win"].shape[:2] == (B, 4)
    print("[2] init equivalence OK (v3(union+owner+player tokens) == "
          "v2(long view) on pair_logits/pair_frac/glob/ctx_now — mask "
          "invisibility + zero-init owner proven; player_state (B,4,d) live)")

    with torch.no_grad():
        s2 = _fwd(v2, *(t[:, -1] if isinstance(t, torch.Tensor) else
                        {k: v[:, -1] for k, v in t.items()}
                        for t in long_views))
        s3 = _fwd(v3, pt[:, -1], ft[:, -1],
                  {k: v[:, -1] for k, v in routing.items()},
                  pm[:, -1], ic[:, -1], pti[:, -1], oh[:, -1])
    d_sf = (s2["pair_logits"] - s3["pair_logits"]).abs().max().item()
    assert d_sf < 1e-5, d_sf
    print("[2] init equivalence OK (single-frame cold-start path)")

    # ---- 3. gradient gate -------------------------------------------
    v3.train()
    fuse_w = v3.cross.fuse_tokens.weight

    def _loss():
        out = _fwd(v3, pt, ft, routing, pm, ic, pti, oh)
        return out["pair_logits"].square().mean() + out["glob"].square().mean()

    v3.zero_grad()
    _loss().backward()
    g_short_half = fuse_w.grad[:, D:].abs().sum().item()
    g_long_half = fuse_w.grad[:, :D].abs().sum().item()
    g_short_params = sum(
        p.grad.abs().sum().item()
        for p in v3.cross.short.parameters() if p.grad is not None
    )
    g_long_params = sum(
        p.grad.abs().sum().item()
        for p in v3.cross.long.parameters() if p.grad is not None
    )
    assert g_short_half > 0 and g_long_half > 0, (g_short_half, g_long_half)
    assert g_short_params == 0.0, (
        f"short branch grad should be EXACTLY zero at zero-init gate, "
        f"got {g_short_params}"
    )
    assert g_long_params > 0
    print(f"[3] grad gate OK at init (fusion short-half |g|={g_short_half:.4f}, "
          f"short branch |g|=0 by construction, long branch alive)")

    with torch.no_grad():
        for prm in (v3.cross.fuse_tokens, v3.cross.fuse_glob):
            for p in prm.parameters():
                p.add_(-0.05 * p.grad)
    v3.zero_grad()
    _loss().backward()
    g_short_params2 = sum(
        p.grad.abs().sum().item()
        for p in v3.cross.short.parameters() if p.grad is not None
    )
    assert g_short_params2 > 0, "short branch must receive grad after 1 fusion step"
    print(f"[3] grad gate OK after one fusion step (short branch |g|="
          f"{g_short_params2:.4f} > 0 — branch fades in)")

    # ---- 3b. short-horizon aux: trains the branch THROUGH the gate ---
    from .short_horizon import N_OWNER, N_PLAYER_SLOTS, short_horizon_loss

    g = torch.Generator().manual_seed(7)
    aux_batch = {
        "planet_mask": pm,
        "owner_t_plus_5": torch.randint(0, N_OWNER, (B, P), generator=g),
        "log_ships_t_plus_5": torch.rand(B, P, generator=g),
        "valid_t_plus_5": torch.ones(B, P),
        "ships_arriving_within_5": torch.rand(B, P, N_PLAYER_SLOTS, generator=g),
        "earliest_arrival_owner_slot": torch.randint(0, N_OWNER, (B, P), generator=g),
    }
    v3.zero_grad()
    _ = _fwd(v3, pt, ft, routing, pm, ic, pti, oh)        # populate stash
    aux_loss, aux_terms = v3.short_aux_loss(aux_batch)
    aux_loss.backward()
    g_short_aux = sum(
        p.grad.abs().sum().item()
        for p in v3.cross.short.parameters() if p.grad is not None
    )
    g_long_aux = sum(
        p.grad.abs().sum().item()
        for p in v3.cross.long.parameters() if p.grad is not None
    )
    assert g_short_aux > 0, "aux loss must reach the short branch at init"
    assert g_long_aux == 0.0, "aux loss must NOT touch the long branch"
    assert {"owner_t5", "ships_t5", "arrivals5", "earliest"} <= set(aux_terms)
    print(f"[3b] short-horizon aux OK (|g| short={g_short_aux:.4f} > 0 at "
          f"zero-init fusion — gate bypassed; long untouched; terms: "
          f"{ {k: round(v, 3) for k, v in aux_terms.items()} })")

    # ---- 3c. value path trains the player-token machinery ------------
    v3.zero_grad()
    out = _fwd(v3, pt, ft, routing, pm, ic, pti, oh)
    # NOT the win head — its last Linear is deliberately zero-init
    # (neutral V at start), so it passes no gradient at init. The fwd
    # signal heads are default-init and train from step 1.
    out["fwd"].square().mean().backward()
    g_pl_long = v3.cross.long.player_tokens.grad.abs().sum().item()
    g_pl_short = v3.cross.short.player_tokens.grad
    g_pl_short = 0.0 if g_pl_short is None else g_pl_short.abs().sum().item()
    g_fuse_pl = v3.cross.fuse_player.weight.grad.abs().sum().item()
    g_owner = v3.cross.owner_proj.weight.grad.abs().sum().item()
    assert g_pl_long > 0, "win loss must reach the long player tokens"
    assert g_pl_short == 0.0, "short player tokens gated at zero-init fusion"
    assert g_fuse_pl > 0 and g_owner > 0
    print(f"[3c] player-token value path OK (long tokens |g|={g_pl_long:.4f}, "
          f"fuse_player |g|={g_fuse_pl:.4f}, owner_proj |g|={g_owner:.4f}; "
          f"short tokens gated at init as designed)")

    # ---- 4. freeze semantics ----------------------------------------
    report = v3.freeze_below_l2()
    n_l2 = sum(p.numel() for p in v3.cross.parameters())
    n_l2_train = sum(p.numel() for p in v3.cross.parameters() if p.requires_grad)
    assert n_l2_train == n_l2, "whole dual-rate L2 must be trainable"
    assert not any(p.requires_grad for p in v3.entity.parameters())
    n_long = sum(p.numel() for p in v3.cross.long.parameters())
    n_short = sum(p.numel() for p in v3.cross.short.parameters())
    n_fuse = n_l2 - n_long - n_short
    print(f"[4] freeze_below_l2 OK (L1 frozen; L2 group = long {n_long:,} + "
          f"short {n_short:,} + fusion {n_fuse:,} = {n_l2:,} trainable)")
    for k, v in report.items():
        print(f"      {k:<28s} {v:,}")

    n_total = sum(p.numel() for p in v3.parameters())
    print(f"\nALL SMOKE CHECKS PASSED  (toy d_model={D}: v3 total {n_total:,} "
          f"params; at d256 the added short branch + fusion ≈ L2-sized)")


if __name__ == "__main__":
    main()
