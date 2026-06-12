"""Stage-C pretrain: the VALUE heads on the frozen v4 actor backbone.

    python -m agents.transformer_v3.value_pretrain \
        --out-dir data/runs/joint/value_v4_<tag> \
        --cross-cache-path .../cross_topmeta300_cap150_p1.pt \
        --fleet-run-dir ... --planet-run-dir ... --comet-run-dir ... \
        --warm-start actor_best.pt

"Previous value heads, same tasks" (user-ordered): the exact
``ValuePretrainHeads`` + ``compute_value_pretrain_loss`` tiers (win BCE
1.0 / fwd Huber 0.5 / back+rank+survives 0.1) — unchanged — except the
``player_state`` they read now comes from the in-L2 player CLS tokens
(v3.1) instead of the deleted consolidator. Backbone (L2/L3/L4/PairHead/
α0) FROZEN at the stage-B actor weights: the actor's behavior is
untouched, the value heads learn to read it. By-game holdout logging
(win-memorization gap) as in every value pretrain.

Output ``value_best.pt`` = the full v4 agent (actor + value heads): the
PPO critic warm start AND the simulate-then-score ranker's scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..transformer_v2.pretrain.joint_pretrain import (
    _forward_context_from_batch,
    _load_encoders,
    _split_value_by_game,
)
from ..transformer_v2.pretrain.value_heads import compute_value_pretrain_loss
from ..transformer_v2.pretrain.cross_entity import (
    CachedCrossEntitySnapshotDataset,
)
from .model import EntityPretrainModelV3, adapt_v2_state_dict
from .history import UNION_HISTORY_OFFSETS


def _log(msg: str) -> None:
    print(f"[value {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_value(args) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        with_consolidator=True, with_value_heads=True,
        value_dropout=args.value_dropout,
        with_short_aux=bool(wcfg.get("with_short_aux", True)),
        with_alloc_conc=bool(wcfg.get("with_alloc_conc", True)),
    ).to(device)
    sd = {k: v for k, v in warm_ck["model"].items()
          if not k.startswith("value_heads.")}
    res = model.load_state_dict(adapt_v2_state_dict(sd), strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("value_heads.")]
    assert not bad, f"backbone skew vs actor ckpt: {bad[:6]}"
    assert not res.unexpected_keys, res.unexpected_keys[:6]
    _log(f"warm-start {args.warm_start}: {len(sd)} actor tensors loaded; "
         f"value heads fresh ({len(res.missing_keys)} tensors)")
    del warm_ck, sd

    # Freeze the entire actor; train value heads only.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.value_heads.parameters():
        p.requires_grad_(True)
    params = list(model.value_heads.parameters())
    _log(f"trainable: value heads {sum(p.numel() for p in params):,} "
         f"(backbone frozen at stage-B actor weights)")
    opt = torch.optim.AdamW(params, lr=args.lr,
                            weight_decay=args.weight_decay)

    ds = CachedCrossEntitySnapshotDataset(
        args.cross_cache_path, history_offsets=UNION_HISTORY_OFFSETS,
    )
    for col in ("p1_now", "p1_future", "p1_valid", "p1_back",
                "valid_back", "survives_future"):
        assert col in ds.columns, f"cross cache missing P1 col {col} — run " \
                                  f"scripts.add_p1_value_labels first"
    train_ds, val_ds = _split_value_by_game(ds)
    _log(f"{len(ds)} snapshots | by-game split: train {len(train_ds)}, "
         f"holdout {len(val_ds) if val_ds is not None else 0}")
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, drop_last=True)
    va_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
                 if val_ds is not None and len(val_ds) > 0 else None)

    config = dict(wcfg)
    config.update({
        "stage": "value_v4", "with_value_heads": True,
        "value_lr": args.lr, "value_dropout": args.value_dropout,
        "cross_cache_path": str(args.cross_cache_path),
        "value_warm_start": str(args.warm_start),
    })

    best_val = math.inf
    log: list[dict] = []

    def _save(path: Path, epoch: int) -> None:
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "config": config}, path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # frozen modules stay eval-consistent; dropout lives in value heads
        t0, run, cnt = time.time(), {}, {}
        for i, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = _forward_context_from_batch(
                model, encoders[0], encoders[1], encoders[2], batch)
            loss, terms = compute_value_pretrain_loss(out, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in terms.items():
                if isinstance(v, float) and math.isnan(v):
                    continue
                run[k] = run.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
            if args.progress_every and i % args.progress_every == 0:
                _log(f"ep {epoch} step {i}/{len(tr_loader)} "
                     f"loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
        tr = {k: run[k] / cnt[k] for k in run}

        va = {}
        if va_loader is not None:
            model.eval()
            vrun, vcnt = {}, {}
            with torch.no_grad():
                for batch in va_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = _forward_context_from_batch(
                        model, encoders[0], encoders[1], encoders[2], batch)
                    vloss, vterms = compute_value_pretrain_loss(out, batch)
                    vterms["total"] = float(vloss)
                    for k, v in vterms.items():
                        if isinstance(v, float) and math.isnan(v):
                            continue
                        vrun[k] = vrun.get(k, 0.0) + v
                        vcnt[k] = vcnt.get(k, 0) + 1
            va = {k: vrun[k] / vcnt[k] for k in vrun}

        _log(f"epoch {epoch} done in {time.time()-t0:.0f}s | "
             f"train win_acc={tr.get('win_acc', float('nan')):.3f} | "
             f"HOLDOUT total={va.get('total', float('nan')):.4f} "
             f"win_acc={va.get('win_acc', float('nan')):.3f} "
             f"fwd={va.get('fwd', float('nan')):.4f} "
             f"(memorization gap = train-holdout win_acc "
             f"{tr.get('win_acc', 0) - va.get('win_acc', 0):+.3f})")
        log.append({"epoch": epoch, "train": tr, "holdout": va})
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
        _save(out_dir / "value_last.pt", epoch)
        crit = va.get("total", tr.get("total", math.inf))
        if crit < best_val:
            best_val = crit
            _save(out_dir / "value_best.pt", epoch)
            _log(f"  new best (holdout total={best_val:.4f}) -> value_best.pt")

    return out_dir / "value_best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cross-cache-path", type=Path, required=True)
    p.add_argument("--fleet-run-dir", type=Path, required=True)
    p.add_argument("--planet-run-dir", type=Path, required=True)
    p.add_argument("--comet-run-dir", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, required=True,
                   help="stage-B actor_best.pt (backbone frozen from it)")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    train_value(p.parse_args())


if __name__ == "__main__":
    main()
