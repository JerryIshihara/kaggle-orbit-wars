"""Stage-B pretrain: the ACTOR on top of the stage-A dual L2.

    python -m agents.transformer_v3.actor_pretrain \
        --out-dir data/runs/joint/actor_v4_<tag> \
        --pair-cache-path .../pair_cache.pt \
        --fleet-run-dir ... --planet-run-dir ... --comet-run-dir ... \
        --warm-start joint_warm_merged.pt \
        --lr-heads 1e-4 --lr-l34 5e-5 --lr-l2 1e-5

Composition (user-ordered):
  * warm start = stage-A merged ckpt (pretrained dual L2 + player tokens,
    L3/L4 + PairHead from the v3dual joint run)
  * L3/L4 INCLUDED and trained; VALUE HEADS EXCLUDED for now
  * NEW actor design (contract v4): select stays the bounded-k multinomial
    (k_max=3); allocation becomes Dirichlet — mean = the existing frac
    softmax, α0 = new per-source concentration head (learned confidence)
  * pretrain tasks: select whole-grid BCE (pos_weight 600) + Dirichlet
    alloc NLL on ε-smoothed expert shares + the t+5 short-branch aux
  * per-part learning rates: heads (PairHead + α0 head + short aux heads)
    / L3+L4 / L2 (pretrained — gentle). L0/L1 frozen.

Trains on ACTED rows (launch turns) like every actor pretrain. The saved
``actor_best.pt`` stamps ``action_contract = v4`` + ``select_k_max = 3``;
PPO/runner decode support for v4 is task #17.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ..transformer_v2.pretrain.joint_pretrain import (
    _forward_context_from_batch,
    _load_encoders,
)
from ..transformer_v2.pretrain.entity_encoder import compute_multi_loss
from .model import EntityPretrainModelV3, adapt_v2_state_dict
from .history import UNION_HISTORY_OFFSETS
from .dirichlet_alloc import dirichlet_alloc_nll
from .short_horizon import short_horizon_loss
from .l2_pretrain import _episode_split, _log as _l2log


def _log(msg: str) -> None:
    print(f"[actor {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _step_losses(model, encoders, batch, *, pair_pos_weight, alloc_weight,
                 short_aux_weight):
    out = _forward_context_from_batch(
        model, encoders[0], encoders[1], encoders[2], batch)
    # select BCE only (alloc_weight=0 disables the multinomial CE term —
    # the Dirichlet NLL below replaces it).
    sel_loss, sel_terms = compute_multi_loss(
        out, batch, multinomial_alloc=True,
        pair_pos_weight=pair_pos_weight, alloc_weight=0.0,
    )
    nll, dir_terms = dirichlet_alloc_nll(
        out["pair_frac"], out["alloc_conc"], batch)
    total = sel_loss + alloc_weight * nll
    terms = {f"sel/{k}": v for k, v in sel_terms.items()
             if not k.startswith("alloc")}
    terms.update({f"dir/{k}": v for k, v in dir_terms.items()})
    if short_aux_weight > 0 and model.short_heads is not None:
        aux, aux_terms = model.short_aux_loss(batch)
        total = total + short_aux_weight * aux
        terms.update({f"sh5/{k}": v for k, v in aux_terms.items()})
    return total, terms


def train_actor(args) -> Path:
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
        with_short_aux=True, with_alloc_conc=True,
    ).to(device)
    sd = {k: v for k, v in warm_ck["model"].items()
          if not k.startswith("value_heads.")}
    res = model.load_state_dict(adapt_v2_state_dict(sd), strict=False)
    fresh_ok = ("alloc_conc_head.", "short_heads.", "cross.fuse_player",
                "cross.owner_proj", "cross.long.player_tokens",
                "cross.short.player_tokens")
    bad = [k for k in res.missing_keys if not k.startswith(fresh_ok)]
    assert not bad, f"backbone skew vs warm ckpt: {bad[:6]}"
    _log(f"warm-start {args.warm_start}: {len(sd)} tensors; fresh = "
         f"{[k.split('.')[0] for k in res.missing_keys][:3]}… "
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

    _grp([model.pair_head, model.alloc_conc_head, model.short_heads],
         args.lr_heads, "heads")
    _grp([model.dual_role, model.joint_role], args.lr_l34, "l34")
    _grp([model.cross], args.lr_l2, "l2")
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
        "stage": "actor_v4", "d_model": args.d_model,
        "lr_heads": args.lr_heads, "lr_l34": args.lr_l34, "lr_l2": args.lr_l2,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "pair_pos_weight": args.pair_pos_weight,
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
        "max_planets": int((getattr(ds, "config", {}) or {}).get("max_planets", 64)),
        "max_fleets": int((getattr(ds, "config", {}) or {}).get("max_fleets", 512)),
    }
    config.update(model.config_extra)

    best_val = math.inf
    log: list[dict] = []

    def _save(path: Path, epoch: int) -> None:
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "config": config}, path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, cnt = time.time(), {}, {}
        for i, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, terms = _step_losses(
                model, encoders, batch,
                pair_pos_weight=args.pair_pos_weight,
                alloc_weight=args.alloc_weight,
                short_aux_weight=args.short_aux_weight,
            )
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in terms.items():
                if isinstance(v, float) and math.isnan(v):
                    continue
                run[k] = run.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
            if args.progress_every and i % args.progress_every == 0:
                _log(f"ep {epoch} step {i}/{len(tr_loader)} "
                     f"loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
        tr = {k: run[k] / cnt[k] for k in run}

        model.eval()
        vrun, vcnt = {}, {}
        with torch.no_grad():
            for batch in va_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                vloss, vterms = _step_losses(
                    model, encoders, batch,
                    pair_pos_weight=args.pair_pos_weight,
                    alloc_weight=args.alloc_weight,
                    short_aux_weight=args.short_aux_weight,
                )
                vterms["total"] = float(vloss)
                for k, v in vterms.items():
                    if isinstance(v, float) and math.isnan(v):
                        continue
                    vrun[k] = vrun.get(k, 0.0) + v; vcnt[k] = vcnt.get(k, 0) + 1
        va = {k: vrun[k] / vcnt[k] for k in vrun}

        _log(f"epoch {epoch} done in {time.time()-t0:.0f}s | "
             f"val_total={va['total']:.4f} | dir: nll="
             f"{va.get('dir/alloc_nll', float('nan')):.3f} "
             f"a0={va.get('dir/alpha0_mean', float('nan')):.1f} "
             f"sat={va.get('dir/alpha0_satfrac', float('nan')):.2f} "
             f"shareL1={va.get('dir/share_l1', float('nan')):.3f} | "
             f"sh5 owner={va.get('sh5/owner_acc', float('nan')):.3f}")
        for grp in ("sel/", "dir/", "sh5/"):
            row = "  ".join(f"{k.split('/', 1)[1]}={va[k]:.4f}"
                            for k in sorted(va) if k.startswith(grp))
            _log(f"  val {grp:<4s} {row}")
        log.append({"epoch": epoch, "train": tr, "val": va})
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
        _save(out_dir / "actor_last.pt", epoch)
        if va["total"] < best_val:
            best_val = va["total"]
            _save(out_dir / "actor_best.pt", epoch)
            _log(f"  new best (val_total={best_val:.4f}) -> actor_best.pt")

    return out_dir / "actor_best.pt"


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
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pair-pos-weight", type=float, default=600.0)
    p.add_argument("--alloc-weight", type=float, default=1.0)
    p.add_argument("--short-aux-weight", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    train_actor(p.parse_args())


if __name__ == "__main__":
    main()
