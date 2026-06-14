"""Stage-B JOINT pretrain: the SINGLE-TARGET actor + value heads + the v5 Q
head + the LogitNormal confidence (frac_sigma) head, all trained together.

This is ``single_target_pretrain`` (actor) fused with ``joint_v4_pretrain``'s
value side, plus the v5 Q head's DENSE doomed term — the user-ordered "q and
value heads too, also the sampling head (confidence)" so the produced
checkpoint can deploy in BOTH det mode and qrank (sample + Q-gate) mode.

Heads trained (off the shared v3 dual-rate-L2 trunk; L0/L1 frozen):
  * SELECT      pair_logits   — per-source softmax over [targets ... HOLD diag]
                                (single-target CE; meta is 100% single-target).
  * ALLOC loc   pair_frac     — LogitNormal location; sized at deploy by
                                sigmoid(loc) (det) or sample_frac (qrank).
  * CONFIDENCE  frac_sigma    — per-source LogitNormal log-sigma (FracSigmaHead);
                                the sampling head, trained by the alloc NLL.
  * VALUE       win/fwd/inbound + contrast — the pruned ``value_v4`` task set,
                                on the t+Δ paired CROSS cache (warm-start strips
                                value heads -> fresh -> trained here).
  * Q           q_value (B,P,P) — per-pair launch value. PRETRAIN target:
                  - DENSE doomed: every plan_launch-DOOMED legal pair -> Q_DOOMED
                    (the anti-doomed gate). Label precomputed by
                    ``scripts/precompute_pair_doomed`` (sidecar, aligned 1:1 to
                    the pair cache's slots) since plan_launch needs raw state the
                    featurized cache drops.
                  - POSITIVE anchor: the expert's fired pairs -> +1 (a constant
                    Monte-Carlo win proxy; top-meta experts mostly win, and the
                    real bootstrapped TD return only exists in PPO rollout). This
                    just establishes reachable/fired > doomed so the deploy
                    q-gate ranks correctly; PPO refines the strategic signal.

    python -m agents.transformer_v3.joint_single_target_pretrain \
        --out-dir data/runs/joint/jst_v5_<tag> \
        --pair-cache-path  .../topmeta300_p64_f512_acted.pt \
        --cross-cache-path .../cross_topmeta300_cap150.pt \
        --doomed-sidecar   .../topmeta300_pair_doomed.pt \
        --fleet-run-dir ... --planet-run-dir ... --comet-run-dir ... \
        --warm-start joint_warm_merged.pt

Stamps ``action_contract = single_target_per_source_logitnormal_v5`` +
``with_frac_sigma = True`` + ``with_value_heads = True`` + ``with_q_head = True``
so the runner builds every head on load.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..transformer_v2.pretrain.joint_pretrain import (
    _forward_context_from_batch,
    _load_encoders,
    _split_value_by_game,
)
from ..transformer_v2.pretrain.cross_entity import CachedCrossEntitySnapshotDataset
from ..transformer_v2.pretrain.entity_encoder import (
    _pair_single_target_ce,
    _pair_frac_targets_from_batch,
)
from .model import EntityPretrainModelV3, adapt_v2_state_dict
from .history import UNION_HISTORY_OFFSETS
from .single_target_alloc import single_target_alloc_nll
from .single_target_pretrain import _eval_metrics as _eval_actor_metrics
from .joint_v4_pretrain import PairedValueDataset, _split_b, _cycle
from .value_v4 import PlayerInboundAux, compute_value_v4_loss
from .q_head import compute_q_loss
from .l2_pretrain import _episode_split


def _log(msg: str) -> None:
    print(f"[jst-v5 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class DoomedPairDataset(Dataset):
    """Pair-cache subset that attaches the precomputed ``pair_doomed`` (P,P)
    bool per item, keyed by the snapshot's (episode_id, turn). Missing key ->
    all-reachable (the safe default for the dense Q loss)."""

    def __init__(self, base, indices, doomed: dict, P: int):
        self.base = base
        self.indices = list(indices)
        self.doomed = doomed
        self.P = P

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        item = self.base[idx]
        ep, t = self.base.keys[idx]
        dm = self.doomed.get((ep, int(t)))
        item["pair_doomed"] = (dm if dm is not None
                               else torch.zeros(self.P, self.P, dtype=torch.bool))
        return item


def _actor_q_losses(model, encoders, batch, *, launch_weight, alloc_weight,
                    short_aux_weight, q_coef, q_reach_anchor):
    """Single-target actor (select CE + LogitNormal alloc + short aux) + the
    v5 Q dense loss, all on ONE forward of the acted pair batch.

    Mirrors ``single_target_pretrain._step_losses`` and appends the Q term:
    fired (expert) pairs -> ``q_reach_anchor``, doomed pairs -> ``Q_DOOMED``."""
    out = _forward_context_from_batch(
        model, encoders[0], encoders[1], encoders[2], batch)
    pair_labels = batch["pair_labels"].bool()
    pair_valid = batch["pair_valid"].bool()

    ce, chosen, ce_diag = _pair_single_target_ce(
        out["pair_logits"], batch, pair_labels, pair_valid,
        launch_weight=launch_weight)
    total = ce
    terms = {f"sel/{k}": v for k, v in ce_diag.items()}

    if "pair_ships" in batch:
        ships = batch["pair_ships"].float()
        frac_targets = _pair_frac_targets_from_batch(batch, ships)
        nll, alloc_diag = single_target_alloc_nll(
            out["pair_frac"], frac_sigma_logs=out["frac_sigma_log"],
            chosen_mask=chosen, frac_targets=frac_targets)
        total = total + alloc_weight * nll
        terms.update({f"alloc/{k}": v for k, v in alloc_diag.items()})

    if short_aux_weight > 0 and model.short_heads is not None:
        aux, aux_terms = model.short_aux_loss(batch)
        total = total + short_aux_weight * aux
        terms.update({f"sh5/{k}": v for k, v in aux_terms.items()})

    if q_coef > 0 and out.get("q_value") is not None and "pair_doomed" in batch:
        B = out["q_value"].shape[0]
        fired_return = torch.full((B,), float(q_reach_anchor),
                                  device=out["q_value"].device)
        q_loss, q_diag = compute_q_loss(
            out["q_value"], fired_mask=pair_labels, fired_return=fired_return,
            doomed_mask=batch["pair_doomed"].bool(), legal_mask=pair_valid)
        total = total + q_coef * q_loss
        terms.update({f"q/{k}": v for k, v in q_diag.items()})

    return total, terms


def _grp(groups, mods, lr, name, *, exclude_ids=frozenset()):
    params = []
    for m in mods:
        if m is None:
            continue
        for p in m.parameters():
            if id(p) in exclude_ids:
                continue
            p.requires_grad_(True)
            params.append(p)
    if params:
        groups.append({"params": params, "lr": lr})
        _log(f"  lr group {name:<7s} lr={lr:g}  "
             f"{sum(p.numel() for p in params):,} params")
    return params


def train_joint_single_target(args) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.build_pair_dataset_orbital_occle import CachedPairDataset

    _log(f"device={device}  loading L0 specialists ...")
    encoders = _load_encoders(
        Path(args.fleet_run_dir), Path(args.planet_run_dir),
        Path(args.comet_run_dir), device=device, expected_d_model=args.d_model)
    for enc in encoders:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()

    warm_ck = torch.load(args.warm_start, map_location=device, weights_only=False)
    wcfg = warm_ck.get("config", {})
    model = EntityPretrainModelV3(
        d_model=args.d_model,
        conditioner_n_layers=int(wcfg.get("conditioner_n_layers", 3)),
        head_n_layers=int(wcfg.get("head_n_layers", 3)),
        with_consolidator=True, with_value_heads=True,
        value_dropout=args.value_dropout, with_short_aux=True,
        with_frac_sigma=True, with_q_head=True,
    ).to(device)
    inbound_head = PlayerInboundAux(args.d_model).to(device)
    assert model.frac_sigma_head is not None, "frac_sigma flag did not wire"
    assert model.pair_head.q_head is not None, "q_head flag did not wire"

    sd = {k: v for k, v in warm_ck["model"].items()
          if not k.startswith("value_heads.")}
    res = model.load_state_dict(adapt_v2_state_dict(sd), strict=False)
    fresh_ok = ("value_heads.", "frac_sigma_head.", "pair_head.q_head.",
                "short_heads.", "cross.fuse_player", "cross.owner_proj",
                "cross.long.player_tokens", "cross.short.player_tokens")
    bad = [k for k in res.missing_keys if not k.startswith(fresh_ok)]
    assert not bad, f"backbone skew vs warm ckpt: {bad[:6]}"
    for tag in ("frac_sigma_head.", "pair_head.q_head.", "value_heads."):
        assert any(k.startswith(tag) for k in res.missing_keys), \
            f"{tag} expected fresh-init (missing from warm ckpt)"
    _log(f"warm-start {args.warm_start}: {len(sd)} tensors; fresh = "
         f"{sorted({k.split('.')[0] for k in res.missing_keys})}")
    del warm_ck, sd

    # ---- LR groups (L0/L1 frozen). q_head + frac_sigma get OWN groups ----
    for p in model.parameters():
        p.requires_grad_(False)
    groups: list = []
    vh = model.value_heads
    q_ids = {id(p) for p in model.pair_head.q_head.parameters()}
    _grp(groups, [vh.value_trunk, vh.win_head, vh.fwd_heads, inbound_head],
         args.lr_value, "value")
    # action = pair_head (EXCLUDING the fresh q_head) + short aux heads.
    _grp(groups, [model.pair_head, model.short_heads], args.lr_action, "action",
         exclude_ids=q_ids)
    # q_head: fresh, OWN fast group (clean dense doomed label).
    _grp(groups, [model.pair_head.q_head], args.lr_q, "q")
    # frac_sigma: fresh confidence head, OWN slow group (alloc NLL is touchy).
    _grp(groups, [model.frac_sigma_head], args.lr_sigma, "sigma")
    _grp(groups, [model.dual_role, model.joint_role], args.lr_l34, "l34")
    _grp(groups, [model.cross], args.lr_l2, "l2")
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    # ---- doomed sidecar ----
    _log(f"loading doomed sidecar {args.doomed_sidecar} ...")
    doomed_blob = torch.load(args.doomed_sidecar, weights_only=False)
    doomed = doomed_blob["doomed"]
    _log(f"  {len(doomed)} acted snapshots carry >=1 doomed pair "
         f"(P={doomed_blob.get('P')})")

    # ---- ACTOR data (pair cache, acted rows + doomed) ----
    pair_ds = CachedPairDataset(args.pair_cache_path)
    if tuple(pair_ds.history_offsets) != UNION_HISTORY_OFFSETS:
        pair_ds.history_offsets = UNION_HISTORY_OFFSETS
    P = int(pair_ds.config.get("max_planets", 64))
    acted = list(pair_ds.acted_indices)
    a_tr, a_va, _, _ = _episode_split(
        [pair_ds.keys[i] for i in acted], args.val_frac, args.seed)
    act_tr = DoomedPairDataset(pair_ds, [acted[i] for i in a_tr], doomed, P)
    act_va = DoomedPairDataset(pair_ds, [acted[i] for i in a_va], doomed, P)
    act_loader = DataLoader(act_tr, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)
    act_va_loader = DataLoader(act_va, batch_size=args.batch_size,
                               num_workers=args.num_workers)

    # ---- VALUE data (cross cache, t+Δ paired, by-game holdout) ----
    cross_ds = CachedCrossEntitySnapshotDataset(
        args.cross_cache_path, history_offsets=UNION_HISTORY_OFFSETS)
    v_tr, v_va = _split_value_by_game(cross_ds)
    val_loader = DataLoader(
        PairedValueDataset(cross_ds, list(v_tr.indices), seed=args.seed),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True)
    ho_loader = (DataLoader(
        PairedValueDataset(cross_ds, list(v_va.indices), seed=args.seed + 1),
        batch_size=args.batch_size, num_workers=args.num_workers)
        if v_va is not None else None)
    n_steps = max(len(act_loader), len(val_loader))
    _log(f"actor: {len(act_loader)} tr / {len(act_va_loader)} va batches | "
         f"value: {len(val_loader)} batches (paired) | "
         f"holdout {len(ho_loader) if ho_loader else 0} | "
         f"{n_steps} interleaved steps/ep")

    config = dict(wcfg)
    config.update({
        "stage": "joint_single_target_v5",
        "action_contract": "single_target_per_source_logitnormal_v5",
        "arch": "dual_rate_l2_v3", "d_model": args.d_model,
        "with_value_heads": True, "with_frac_sigma": True, "with_q_head": True,
        "with_short_aux": True, "value_dropout": args.value_dropout,
        "launch_weight": args.launch_weight, "alloc_weight": args.alloc_weight,
        "q_coef": args.q_coef, "q_reach_anchor": args.q_reach_anchor,
        "value_coef": args.value_coef,
        "pair_cache_path": str(args.pair_cache_path),
        "cross_cache_path": str(args.cross_cache_path),
        "doomed_sidecar": str(args.doomed_sidecar),
        "warm_start": str(args.warm_start),
    })

    best = math.inf
    log: list[dict] = []

    def _save(path: Path, epoch: int) -> None:
        torch.save({"model": model.state_dict(),
                    "inbound_head": inbound_head.state_dict(),
                    "epoch": epoch, "config": config}, path)

    for epoch in range(1, args.epochs + 1):
        model.train(); inbound_head.train()
        t0, run, cnt = time.time(), {}, {}
        a_it, v_it = _cycle(act_loader), _cycle(val_loader)
        for i in range(1, n_steps + 1):
            ab = {k: v.to(device) for k, v in next(a_it).items()}
            a_loss, a_terms = _actor_q_losses(
                model, encoders, ab, launch_weight=args.launch_weight,
                alloc_weight=args.alloc_weight,
                short_aux_weight=args.short_aux_weight,
                q_coef=args.q_coef, q_reach_anchor=args.q_reach_anchor)

            vb_all = {k: v.to(device) for k, v in next(v_it).items()}
            va_b, vb_b = _split_b(vb_all)
            out_a = _forward_context_from_batch(
                model, encoders[0], encoders[1], encoders[2], va_b)
            out_b = _forward_context_from_batch(
                model, encoders[0], encoders[1], encoders[2], vb_b)
            v_loss, v_terms = compute_value_v4_loss(
                out_a, va_b, inbound_head=inbound_head, out_b=out_b, batch_b=vb_b)

            loss = a_loss + args.value_coef * v_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for g in groups for p in g["params"]), args.grad_clip)
            opt.step()
            for k, v in {**{f"a/{k}": v for k, v in a_terms.items()},
                         **{f"v/{k}": v for k, v in v_terms.items()}}.items():
                if isinstance(v, float) and math.isnan(v):
                    continue
                run[k] = run.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
            if args.progress_every and i % args.progress_every == 0:
                _log(f"ep {epoch} step {i}/{n_steps} loss={loss.item():.4f} "
                     f"a_sel={a_terms.get('sel/pair_ce', float('nan')):.3f} "
                     f"q_sep={a_terms.get('q/q_separation', float('nan')):+.3f} "
                     f"({time.time()-t0:.0f}s)")
        tr = {k: run[k] / cnt[k] for k in run}

        # ---- eval: actor (held-out acted) + value (holdout) ----
        am = _eval_actor_metrics(
            model, encoders, act_va_loader, device,
            launch_weight=args.launch_weight, alloc_weight=args.alloc_weight,
            short_aux_weight=args.short_aux_weight)
        vm = _value_holdout(model, encoders, inbound_head, ho_loader, device)

        actor_total = am.get("total", float("inf"))
        _log(f"epoch {epoch} done {time.time()-t0:.0f}s | "
             f"ACTOR val_total={actor_total:+.3f} select_acc={am.get('select_acc', float('nan')):.3f} "
             f"launch_recall={am.get('launch_recall', float('nan')):.3f} "
             f"launch_acc={am.get('launch_acc', float('nan')):.3f} "
             f"frac_mae={am.get('frac_mae', float('nan')):.3f} "
             f"sigma={am.get('sigma_mean', float('nan')):.3f} | "
             f"Q sep(train)={tr.get('a/q/q_separation', float('nan')):+.3f} "
             f"(doomed {tr.get('a/q/q_doomed_mean', float('nan')):+.2f} vs "
             f"reach {tr.get('a/q/q_reach_mean', float('nan')):+.2f}) | "
             f"VALUE win_acc={vm.get('win_acc', float('nan')):.3f}")
        log.append({"epoch": epoch, "train": tr, "actor_val": am, "value_ho": vm})
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
        _save(out_dir / "jst_last.pt", epoch)
        if actor_total < best:
            best = actor_total
            _save(out_dir / "jst_best.pt", epoch)
            _log(f"  new best (actor val_total={best:+.4f}) -> jst_best.pt")

    return out_dir / "jst_best.pt"


@torch.no_grad()
def _value_holdout(model, encoders, inbound_head, ho_loader, device) -> dict:
    if ho_loader is None:
        return {}
    model.eval(); inbound_head.eval()
    run, cnt = {}, {}
    for vb_all in ho_loader:
        vb_all = {k: v.to(device) for k, v in vb_all.items()}
        va_b, vb_b = _split_b(vb_all)
        out_a = _forward_context_from_batch(
            model, encoders[0], encoders[1], encoders[2], va_b)
        out_b = _forward_context_from_batch(
            model, encoders[0], encoders[1], encoders[2], vb_b)
        hl, ht = compute_value_v4_loss(
            out_a, va_b, inbound_head=inbound_head, out_b=out_b, batch_b=vb_b)
        ht["total"] = float(hl)
        for k, v in ht.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            run[k] = run.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
    return {k: run[k] / cnt[k] for k in run}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pair-cache-path", type=Path, required=True)
    p.add_argument("--cross-cache-path", type=Path, required=True)
    p.add_argument("--doomed-sidecar", type=Path, required=True)
    p.add_argument("--fleet-run-dir", type=Path, required=True)
    p.add_argument("--planet-run-dir", type=Path, required=True)
    p.add_argument("--comet-run-dir", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, required=True)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr-value", type=float, default=1e-4)
    p.add_argument("--lr-action", type=float, default=1e-4)
    p.add_argument("--lr-q", type=float, default=1e-4,
                   help="fresh Q head — own FAST group (clean dense label)")
    p.add_argument("--lr-sigma", type=float, default=5e-6,
                   help="fresh frac_sigma confidence head — own SLOW group")
    p.add_argument("--lr-l34", type=float, default=5e-5)
    p.add_argument("--lr-l2", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-coef", type=float, default=1.0)
    p.add_argument("--value-dropout", type=float, default=0.1)
    p.add_argument("--launch-weight", type=float, default=6.0)
    p.add_argument("--alloc-weight", type=float, default=1.0)
    p.add_argument("--short-aux-weight", type=float, default=0.5)
    p.add_argument("--q-coef", type=float, default=1.0)
    p.add_argument("--q-reach-anchor", type=float, default=1.0,
                   help="constant positive Q target for expert-fired pairs "
                        "(MC win proxy; > Q_DOOMED so the gate ranks doomed low)")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    train_joint_single_target(p.parse_args())


if __name__ == "__main__":
    main()
