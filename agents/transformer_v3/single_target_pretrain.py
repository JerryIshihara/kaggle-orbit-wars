"""Stage-B pretrain: the SINGLE-TARGET actor on top of the stage-A dual L2.

    python -m agents.transformer_v3.single_target_pretrain \
        --out-dir data/runs/joint/single_target_v5_<tag> \
        --pair-cache-path .../pair_cache.pt \
        --fleet-run-dir ... --planet-run-dir ... --comet-run-dir ... \
        --warm-start joint_warm_merged.pt \
        --lr-heads 1e-4 --lr-l34 5e-5 --lr-l2 1e-5 --lr-sigma 5e-6

This is a near-exact STRUCTURAL MIRROR of ``actor_pretrain.py`` (same
warm-start, same per-part LR groups + freeze, same top-meta pair-cache
DataLoader, same episode train/val split, same save/val cadence). It differs
ONLY in the action contract and the loss/metrics.

Composition (user-ordered, identical backbone to the v4 actor):
  * warm start = stage-A merged ckpt (pretrained dual L2 + player tokens,
    L3/L4 + PairHead from the v3dual joint run); the NEW ``frac_sigma_head``
    is fresh-init (its OWN slow LR group, like the v4 ``alloc_conc_head``).
  * L3/L4 INCLUDED and trained; VALUE HEADS EXCLUDED for now.
  * NEW actor design (contract v5, ``single_target_per_source_logitnormal_v5``):
    the top meta is 100% single-target per source (one launching planet commits
    ALL its force to ONE target). SELECT becomes a per-source softmax over the
    target columns INCLUDING the diagonal HOLD (one target or hold); ALLOC
    becomes a LogitNormal-sigmoid with a learned per-source CONFIDENCE (sigma)
    off the ``FracSigmaHead``. The v4 bounded-k Dirichlet that let a source split
    across 2-3 targets is reverted — it diluted commitment vs the meta.
  * pretrain tasks: single-target select CE (``_pair_single_target_ce``,
    optional launch up-weight) + LogitNormal alloc NLL on the expert's observed
    fraction-of-source ships (``single_target_alloc_nll`` — teaches the loc AND
    the sigma jointly) + the t+5 short-branch aux.
  * per-part learning rates: heads (PairHead + short aux) / L3+L4 / L2
    (pretrained — gentle) / frac_sigma head (slow, its own group). L0/L1 frozen.

Trains on ACTED rows (launch turns) like every actor pretrain. The saved
``single_target_best.pt`` stamps
``action_contract = single_target_per_source_logitnormal_v5`` +
``with_frac_sigma = True``; PPO/runner decode support for v5 is a later task.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ..transformer_v2.pretrain.joint_pretrain import _forward_context_from_batch
from ..transformer_v2.pretrain.joint_pretrain import _load_encoders
from ..transformer_v2.pretrain.entity_encoder import (
    _pair_single_target_ce,
    _pair_frac_targets_from_batch,
)
from .model import EntityPretrainModelV3, adapt_v2_state_dict
from .history import UNION_HISTORY_OFFSETS
from .single_target_alloc import single_target_alloc_nll
from .l2_pretrain import _episode_split


def _log(msg: str) -> None:
    # Verbose timestamped logging — this repo MANDATES it for long runs so
    # silence never looks like a hang. Wall-clock HH:MM:SS prefix; intra-epoch
    # progress + per-epoch summaries both stream through here.
    print(f"[sgl-tgt {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _step_losses(model, encoders, batch, *, launch_weight, alloc_weight,
                 short_aux_weight):
    """Single-target select CE + LogitNormal alloc NLL (+ t+5 short aux).

    ``out`` carries ``pair_logits``/``pair_frac`` and — because the model is
    built ``with_frac_sigma=True`` — ``frac_sigma_log`` (B,P), the per-source
    LogitNormal log-sigma (the learned alloc confidence).

    Returns ``(total_loss, terms)`` where ``terms`` is a flat scalar dict for
    logging (namespaced ``sel/`` select, ``alloc/`` allocation, ``sh5/`` short
    aux). ``terms`` carries the raw select counts (``sel/n_*``) so the val pass
    can aggregate recall/accuracy across batches without re-running.
    """
    out = _forward_context_from_batch(
        model, encoders[0], encoders[1], encoders[2], batch)

    pair_labels = batch["pair_labels"].bool()
    pair_valid = batch["pair_valid"].bool()

    # 1) SELECT — per-source single-target categorical (diagonal = HOLD/NOOP).
    #    ``chosen`` (B,P,P) bool marks the single chosen off-diagonal target per
    #    acted launch row; the frac NLL is supervised ONLY on those cells (a hold
    #    trains no frac). Mirrors compute_multi_loss(single_target=True).
    ce, chosen, ce_diag = _pair_single_target_ce(
        out["pair_logits"], batch, pair_labels, pair_valid,
        launch_weight=launch_weight,
    )
    total = ce
    terms = {f"sel/{k}": v for k, v in ce_diag.items()}

    # 2) ALLOC — LogitNormal NLL of the expert observed fraction-of-source ships
    #    on the chosen cells. ``frac_targets`` = ships_sent / source_ships (the
    #    existing helper); the NLL teaches both the loc (pair_frac) and the sigma
    #    (FracSigmaHead confidence). Skip gracefully if pair_ships is absent.
    if "pair_ships" in batch:
        ships = batch["pair_ships"].float()
        frac_targets = _pair_frac_targets_from_batch(batch, ships)
        nll, alloc_diag = single_target_alloc_nll(
            out["pair_frac"], frac_sigma_logs=out["frac_sigma_log"],
            chosen_mask=chosen, frac_targets=frac_targets,
        )
        total = total + alloc_weight * nll
        terms.update({f"alloc/{k}": v for k, v in alloc_diag.items()})
    else:
        terms["alloc/alloc_nll"] = float("nan")

    # 3) t+5 short-branch aux (direct supervision for the dual-L2 short branch).
    if short_aux_weight > 0 and model.short_heads is not None:
        aux, aux_terms = model.short_aux_loss(batch)
        total = total + short_aux_weight * aux
        terms.update({f"sh5/{k}": v for k, v in aux_terms.items()})

    return total, terms


@torch.no_grad()
def _eval_metrics(model, encoders, loader, device, *, launch_weight,
                  alloc_weight, short_aux_weight) -> dict[str, float]:
    """Full val pass: mean losses + select recall/accuracy + alloc calibration.

    Select metrics aggregate the RAW per-batch counts from
    ``_pair_single_target_ce`` (so recall/accuracy are exact dataset-level
    ratios, not means-of-means):
      * ``select_acc``    — argmax(row incl HOLD) == expert chosen cell, all rows
      * ``launch_recall`` — of expert-LAUNCH rows, frac where argmax is OFF the
        diagonal (decided to launch at all; the over-holding diagnostic)
      * ``launch_acc``    — of expert-LAUNCH rows, frac hitting the EXACT target
      * ``hold_precision``/``hold_recall`` — for the HOLD (diagonal) class
    Alloc calibration aggregates ``frac_mae`` and the sigma stats (sigma should
    fall as the contract sharpens).
    """
    model.eval()
    run, cnt = {}, {}
    # Raw select counts (summed across batches).
    n_rows = n_correct = 0
    n_noop = n_noop_correct = 0
    n_launch = n_launch_correct = n_launch_recalled = 0
    # HOLD precision needs predicted-NOOP totals; recover from the row counts:
    #   pred==NOOP&label==NOOP = n_noop_correct (true positives for HOLD)
    #   we also need predicted-NOOP on launch rows (false positives for HOLD).
    n_launch_pred_noop = 0
    # Alloc calibration (weighted by launch count per batch).
    frac_mae_sum = 0.0
    sigma_mean_sum = 0.0
    sigma_p10_min = math.inf
    n_alloc = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = _forward_context_from_batch(
            model, encoders[0], encoders[1], encoders[2], batch)
        pair_labels = batch["pair_labels"].bool()
        pair_valid = batch["pair_valid"].bool()
        ce, chosen, st = _pair_single_target_ce(
            out["pair_logits"], batch, pair_labels, pair_valid,
            launch_weight=launch_weight,
        )

        # ---- total loss (for save-best) reuses _step_losses on this out ----
        vtotal = ce
        terms = {f"sel/{k}": v for k, v in st.items()}
        if "pair_ships" in batch:
            ships = batch["pair_ships"].float()
            frac_targets = _pair_frac_targets_from_batch(batch, ships)
            nll, alloc_diag = single_target_alloc_nll(
                out["pair_frac"], frac_sigma_logs=out["frac_sigma_log"],
                chosen_mask=chosen, frac_targets=frac_targets,
            )
            vtotal = vtotal + alloc_weight * nll
            terms.update({f"alloc/{k}": v for k, v in alloc_diag.items()})
            nl = int(alloc_diag.get("n_launches", 0))
            if nl:
                frac_mae_sum += alloc_diag["frac_mae"] * nl
                sigma_mean_sum += alloc_diag["sigma_mean"] * nl
                sigma_p10_min = min(sigma_p10_min, alloc_diag["sigma_p10"])
                n_alloc += nl
        if short_aux_weight > 0 and model.short_heads is not None:
            aux, aux_terms = model.short_aux_loss(batch)
            vtotal = vtotal + short_aux_weight * aux
            terms.update({f"sh5/{k}": v for k, v in aux_terms.items()})
        terms["total"] = float(vtotal)

        for k, v in terms.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            run[k] = run.get(k, 0.0) + v
            cnt[k] = cnt.get(k, 0) + 1

        # ---- raw select counts ----
        n_rows += st["n_rows"]
        n_correct += st["n_correct"]
        n_noop += st["n_noop"]
        n_noop_correct += st["n_noop_correct"]
        n_launch += st["n_launch"]
        n_launch_correct += st["n_launch_correct"]
        n_launch_recalled += st["n_launch_recalled"]
        # launch rows predicted as NOOP = launch rows that were NOT recalled.
        n_launch_pred_noop += st["n_launch"] - st["n_launch_recalled"]

    metrics = {k: run[k] / cnt[k] for k in run}
    metrics["select_acc"] = n_correct / max(1, n_rows)
    metrics["launch_recall"] = (
        n_launch_recalled / n_launch if n_launch else float("nan"))
    metrics["launch_acc"] = (
        n_launch_correct / n_launch if n_launch else float("nan"))
    # HOLD class precision/recall (diagonal = NOOP).
    #   recall    = TP / (TP + FN) = noop_correct / n_noop
    #   precision = TP / (TP + FP); FP = launch rows the model called NOOP.
    pred_noop = n_noop_correct + n_launch_pred_noop
    metrics["hold_recall"] = (
        n_noop_correct / n_noop if n_noop else float("nan"))
    metrics["hold_precision"] = (
        n_noop_correct / pred_noop if pred_noop else float("nan"))
    metrics["frac_mae"] = frac_mae_sum / n_alloc if n_alloc else float("nan")
    metrics["sigma_mean"] = sigma_mean_sum / n_alloc if n_alloc else float("nan")
    metrics["sigma_p10"] = sigma_p10_min if n_alloc else float("nan")
    metrics["n_rows"] = float(n_rows)
    metrics["n_launch"] = float(n_launch)
    return metrics


def train_single_target(args) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.build_pair_dataset_orbital_occle import CachedPairDataset

    _log(f"device={device}  loading L0 specialists ...")
    encoders = _load_encoders(
        Path(args.fleet_run_dir), Path(args.planet_run_dir),
        Path(args.comet_run_dir), device=device, expected_d_model=args.d_model,
    )
    for enc in encoders:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()

    warm_ck = torch.load(args.warm_start, map_location=device,
                         weights_only=False)
    wcfg = warm_ck.get("config", {})
    model = EntityPretrainModelV3(
        d_model=args.d_model,
        conditioner_n_layers=int(wcfg.get("conditioner_n_layers", 3)),
        head_n_layers=int(wcfg.get("head_n_layers", 3)),
        with_consolidator=True, with_value_heads=False,
        with_short_aux=True, with_frac_sigma=True,
    ).to(device)
    assert model.frac_sigma_head is not None, "with_frac_sigma flag did not wire"
    sd = {k: v for k, v in warm_ck["model"].items()
          if not k.startswith("value_heads.")}
    res = model.load_state_dict(adapt_v2_state_dict(sd), strict=False)
    # frac_sigma_head is the ONLY new module vs the v4 actor warm path (which
    # adds alloc_conc_head); everything else loads via the adapter.
    fresh_ok = ("frac_sigma_head.", "short_heads.", "cross.fuse_player",
                "cross.owner_proj", "cross.long.player_tokens",
                "cross.short.player_tokens")
    bad = [k for k in res.missing_keys if not k.startswith(fresh_ok)]
    assert not bad, f"backbone skew vs warm ckpt: {bad[:6]}"
    assert any(k.startswith("frac_sigma_head.") for k in res.missing_keys), (
        "frac_sigma_head expected fresh-init (missing from warm ckpt)")
    _log(f"warm-start {args.warm_start}: {len(sd)} tensors; fresh = "
         f"{sorted({k.split('.')[0] for k in res.missing_keys})} "
         f"unexpected={len(res.unexpected_keys)}")
    del warm_ck, sd

    # ---- per-part LR groups (L0/L1 frozen) --------------------------
    for p in model.parameters():
        p.requires_grad_(False)
    groups = []

    def _grp(mods, lr, name):
        params = []
        for m in mods:
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad_(True)
                params.append(p)
        if params:
            groups.append({"params": params, "lr": lr})
            _log(f"  lr group {name:<6s} lr={lr:g}  "
                 f"{sum(p.numel() for p in params):,} params")

    # frac_sigma head is its OWN slow group (mirrors actor_pretrain's conc head,
    # which sat in the heads group at lr_heads; here the user asked for the fresh
    # confidence head to learn gently on its own slow rate).
    _grp([model.pair_head, model.short_heads], args.lr_heads, "heads")
    _grp([model.dual_role, model.joint_role], args.lr_l34, "l34")
    _grp([model.cross], args.lr_l2, "l2")
    _grp([model.frac_sigma_head], args.lr_sigma, "sigma")
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    ds = CachedPairDataset(args.pair_cache_path)
    if tuple(ds.history_offsets) != UNION_HISTORY_OFFSETS:
        _log(f"restack {tuple(ds.history_offsets)} -> UNION "
             f"(T={len(UNION_HISTORY_OFFSETS)})")
        ds.history_offsets = UNION_HISTORY_OFFSETS
    acted = list(getattr(ds, "acted_indices", []) or [])
    assert acted, "pair cache lacks acted_indices"
    tr_idx, va_idx, n_tr_ep, n_va_ep = _episode_split(
        [ds.keys[i] for i in acted], args.val_frac, args.seed)
    tr_rows = [acted[i] for i in tr_idx]
    va_rows = [acted[i] for i in va_idx]
    _log(f"{len(ds)} snapshots, {len(acted)} ACTED | episode split: "
         f"train {len(tr_rows)}/{n_tr_ep} eps, val {len(va_rows)}/{n_va_ep} eps")
    tr_loader = DataLoader(Subset(ds, tr_rows), batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers,
                           drop_last=True)
    va_loader = DataLoader(Subset(ds, va_rows), batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)

    config = {
        "stage": "single_target_v5", "d_model": args.d_model,
        "lr_heads": args.lr_heads, "lr_l34": args.lr_l34, "lr_l2": args.lr_l2,
        "lr_sigma": args.lr_sigma,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "launch_weight": args.launch_weight,
        "alloc_weight": args.alloc_weight,
        "short_aux_weight": args.short_aux_weight,
        "warm_start": str(args.warm_start),
        "pair_cache_path": str(args.pair_cache_path),
        "conditioner_n_layers": int(wcfg.get("conditioner_n_layers", 3)),
        "head_n_layers": int(wcfg.get("head_n_layers", 3)),
        "d_pair": args.d_model,
        "entity_n_heads": 8, "cross_n_heads": 8, "cross_n_layers": 2,
        "dual_n_heads": 8, "skip_l34": False,
        "with_consolidator": False, "with_value_heads": False,
        # Contract stamp — the runner/PPO detect the single-target LogitNormal
        # contract (vs the v4 Dirichlet) off these keys.
        "with_frac_sigma": True,
        "action_contract": "single_target_per_source_logitnormal_v5",
        "max_planets": int((getattr(ds, "config", {}) or {}).get("max_planets", 64)),
        "max_fleets": int((getattr(ds, "config", {}) or {}).get("max_fleets", 512)),
    }
    config.update(model.config_extra)
    # config_extra would only inject the v4 Dirichlet contract keys if
    # with_alloc_conc; with_frac_sigma it injects none, so our stamp stands.
    config["action_contract"] = "single_target_per_source_logitnormal_v5"
    config["with_frac_sigma"] = True

    best_val = math.inf
    log: list[dict] = []

    def _save(path: Path, epoch: int) -> None:
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "config": config}, path)

    _log(f"starting {args.epochs} epochs | launch_weight={args.launch_weight} "
         f"alloc_weight={args.alloc_weight} short_aux={args.short_aux_weight}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, cnt = time.time(), {}, {}
        for i, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, terms = _step_losses(
                model, encoders, batch,
                launch_weight=args.launch_weight,
                alloc_weight=args.alloc_weight,
                short_aux_weight=args.short_aux_weight,
            )
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in terms.items():
                if isinstance(v, float) and math.isnan(v):
                    continue
                run[k] = run.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
            if args.progress_every and i % args.progress_every == 0:
                el = time.time() - t0
                _log(f"ep {epoch} step {i}/{len(tr_loader)} "
                     f"loss={loss.item():.4f} "
                     f"sel_ce={terms.get('sel/pair_ce', float('nan')):.4f} "
                     f"alloc_nll={terms.get('alloc/alloc_nll', float('nan')):.3f} "
                     f"({el:.0f}s, {i / max(el, 1e-6):.1f} it/s)")
        tr = {k: run[k] / cnt[k] for k in run}

        va = _eval_metrics(
            model, encoders, va_loader, device,
            launch_weight=args.launch_weight,
            alloc_weight=args.alloc_weight,
            short_aux_weight=args.short_aux_weight,
        )

        _log(f"epoch {epoch} done in {time.time()-t0:.0f}s | "
             f"val_total={va['total']:.4f} sel_ce="
             f"{va.get('sel/pair_ce', float('nan')):.4f} alloc_nll="
             f"{va.get('alloc/alloc_nll', float('nan')):.3f} | "
             f"sh5 owner={va.get('sh5/owner_acc', float('nan')):.3f}")
        # Compact metric table: select recall/accuracy + alloc calibration.
        _log("  SELECT  "
             f"acc={va['select_acc']:.3f}  "
             f"launch_recall={va['launch_recall']:.3f}  "
             f"launch_acc={va['launch_acc']:.3f}  "
             f"hold_prec={va['hold_precision']:.3f}  "
             f"hold_recall={va['hold_recall']:.3f}  "
             f"(rows={int(va['n_rows'])}, launches={int(va['n_launch'])})")
        _log("  ALLOC   "
             f"frac_mae={va['frac_mae']:.3f}  "
             f"sigma_mean={va['sigma_mean']:.3f}  "
             f"sigma_p10={va['sigma_p10']:.3f}")
        log.append({"epoch": epoch, "train": tr, "val": va})
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
        _save(out_dir / "single_target_last.pt", epoch)
        if va["total"] < best_val:
            best_val = va["total"]
            _save(out_dir / "single_target_best.pt", epoch)
            _log(f"  new best (val_total={best_val:.4f}) -> single_target_best.pt")

    _log(f"done. best val_total={best_val:.4f}")
    return out_dir / "single_target_best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pair-cache-path", type=Path, required=True)
    p.add_argument("--fleet-run-dir", type=Path, required=True)
    p.add_argument("--planet-run-dir", type=Path, required=True)
    p.add_argument("--comet-run-dir", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, required=True)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr-heads", type=float, default=1e-4)
    p.add_argument("--lr-l34", type=float, default=5e-5)
    p.add_argument("--lr-l2", type=float, default=1e-5)
    p.add_argument("--lr-sigma", type=float, default=5e-6,
                   help="slow LR for the fresh FracSigmaHead (the confidence "
                        "head); mirrors actor_pretrain's slow conc-head rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--launch-weight", type=float, default=1.0,
                   help="up-weight launch rows in the select CE so the "
                        "(dominant) HOLD rows don't drown the launch signal")
    p.add_argument("--alloc-weight", type=float, default=1.0)
    p.add_argument("--short-aux-weight", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    train_single_target(p.parse_args())


if __name__ == "__main__":
    main()
