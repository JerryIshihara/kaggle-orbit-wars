"""Stage-A pretrain: dual-rate L2 ONLY (perception), no later parts.

    python -m agents.transformer_v3.l2_pretrain \
        --out-dir data/runs/joint/l2only_<tag> \
        --pair-cache-path .../topmeta_jun10_p64_f512_acted.pt \
        --fleet-run-dir ... --planet-run-dir ... --comet-run-dir ... \
        --warm-start /path/to/joint_best.pt   # v2 OR v3-shaped

Trains exactly: both L2 branches, the three fusions, owner projection,
player tokens, and the ``DualL2AuxHeads`` — on the planet/player/global
short+long-term forecast tasks (``l2_aux.py``). L0/L1 frozen as always;
L3/L4/PairHead/value heads are NOT built (``skip_l34``, no value heads).

Uses ALL snapshots (not just acted turns — perception tasks don't care who
acted; ~2.5x more rows than the joint stage). Split by EPISODE so val
measures generalization. The output checkpoint is a drop-in warm start for
the later joint stage (same module names; missing parts initialize there).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ..transformer_v2.pretrain.joint_pretrain import _load_encoders
from .model import EntityPretrainModelV3, adapt_v2_state_dict
from .history import UNION_HISTORY_OFFSETS
from .l2_aux import DualL2AuxHeads, dual_l2_aux_loss, L2_AUX_LABEL_KEYS


def _log(msg: str) -> None:
    print(f"[l2pre {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _episode_split(keys: list, val_frac: float, seed: int):
    eps = sorted({ep for ep, _t in keys})
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(eps), generator=g).tolist()
    n_val = max(1, int(len(eps) * val_frac))
    val_eps = {eps[i] for i in order[:n_val]}
    tr_idx = [i for i, (ep, _t) in enumerate(keys) if ep not in val_eps]
    va_idx = [i for i, (ep, _t) in enumerate(keys) if ep in val_eps]
    return tr_idx, va_idx, len(eps) - n_val, n_val


def _forward_l2(model, encoders, batch, device):
    """L0 → L1 → dual L2 with owner threading; returns fused outputs.

    Mirrors joint_pretrain._forward_context_from_batch's L0 stage but
    stops at the fused L2 readouts — PairHead (the expensive P×P cell
    MLP) and everything above never run in this stage.
    """
    from ..transformer_v2.pretrain.joint_pretrain import (
        _build_entity_self_tokens,
        _FLEET_ROUTING_KEYS,
    )
    from ..transformer_v2.pretrain.entity_encoder import (
        ENTITY_N_OWNER_CLASSES,
        _PLANET_OWNER_START_IDX,
    )
    fleet_enc, planet_enc, comet_enc = encoders
    with torch.no_grad():
        planet_tok = planet_enc(batch["planet_features"])
        comet_tok = comet_enc(batch["comet_features"])
        fleet_tok = fleet_enc(batch["fleet_features"])
    entity_self = _build_entity_self_tokens(
        planet_tok, comet_tok, batch["is_comet"])
    routing = {k: batch[k] for k in _FLEET_ROUTING_KEYS}
    owner_oh = batch["planet_features"][
        ..., _PLANET_OWNER_START_IDX:
        _PLANET_OWNER_START_IDX + ENTITY_N_OWNER_CLASSES
    ]
    planet_mask = batch["planet_mask"]
    B, T, P, d = entity_self.shape
    F_ = fleet_tok.shape[2]
    entity_tokens = model.entity(
        entity_self.reshape(B * T, P, d),
        fleet_tok.reshape(B * T, F_, d),
        routing["fleet_target_idx"].reshape(B * T, F_),
        routing["fleet_source_idx"].reshape(B * T, F_),
        routing["fleet_owner_slot"].reshape(B * T, F_),
        routing["fleet_ships_log"].reshape(B * T, F_),
        routing["fleet_eta_norm"].reshape(B * T, F_),
        routing["fleet_mask"].reshape(B * T, F_),
        planet_mask=planet_mask.reshape(B * T, P),
    ).reshape(B, T, P, d)
    ctx_full, glob = model.cross(entity_tokens, planet_mask, owner_oh=owner_oh)
    return (
        ctx_full[:, -1],
        model.cross.last_player_state,
        glob,
        planet_mask[:, -1],
    )


def train_l2(args) -> Path:
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

    # Mirror the warm ckpt's head/conditioner depth so PairHead keys map
    # 1:1 even though stage A never runs it (a depth mismatch shows up as
    # confusing 'missing' film keys).
    warm_ck = None
    cond_n, head_n = 1, 1
    if args.warm_start:
        warm_ck = torch.load(args.warm_start, map_location=device,
                             weights_only=False)
        wcfg = warm_ck.get("config", {})
        cond_n = int(wcfg.get("conditioner_n_layers", 1))
        head_n = int(wcfg.get("head_n_layers", 1))
    model = EntityPretrainModelV3(
        d_model=args.d_model, skip_l34=True,
        conditioner_n_layers=cond_n, head_n_layers=head_n,
        with_consolidator=True, with_value_heads=False,
        with_short_aux=False,
    ).to(device)
    aux = DualL2AuxHeads(args.d_model).to(device)

    if warm_ck is not None:
        sd = {k: v for k, v in warm_ck.get("model", warm_ck).items()
              if not k.startswith(("value_heads.", "short_heads."))}
        res = model.load_state_dict(adapt_v2_state_dict(sd), strict=False)
        fresh_ok = ("cross.fuse_player", "cross.owner_proj",
                    "cross.long.player_tokens", "cross.short.player_tokens")
        bad = [k for k in res.missing_keys if not k.startswith(fresh_ok)]
        _log(f"warm-start {args.warm_start}: loaded {len(sd)} tensors "
             f"(cond{cond_n}/head{head_n}; missing={len(res.missing_keys)} "
             f"bad={bad[:4]} unexpected={len(res.unexpected_keys)})")
        assert not bad, f"backbone skew vs warm ckpt: {bad[:6]}"
        del warm_ck, sd

    # Freeze EVERYTHING except the dual L2; aux heads train alongside.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.cross.parameters():
        p.requires_grad_(True)
    params = [p for p in model.cross.parameters()] + list(aux.parameters())
    n_tr = sum(p.numel() for p in params)
    _log(f"trainable: dual-L2 {sum(p.numel() for p in model.cross.parameters()):,} "
         f"+ aux heads {sum(p.numel() for p in aux.parameters()):,} = {n_tr:,}")

    ds = CachedPairDataset(args.pair_cache_path)
    if tuple(ds.history_offsets) != UNION_HISTORY_OFFSETS:
        _log(f"restack {tuple(ds.history_offsets)} -> UNION "
             f"(T={len(UNION_HISTORY_OFFSETS)})")
        ds.history_offsets = UNION_HISTORY_OFFSETS
    probe = ds[0]["planet_features"]
    assert probe.shape[0] == len(UNION_HISTORY_OFFSETS), probe.shape
    for k in L2_AUX_LABEL_KEYS:
        assert k in ds[0], f"pair cache lacks {k}"
    tr_idx, va_idx, n_tr_ep, n_va_ep = _episode_split(
        ds.keys, args.val_frac, args.seed)
    _log(f"{len(ds)} snapshots (ALL turns) | split by episode: "
         f"train {len(tr_idx)} rows/{n_tr_ep} eps, val {len(va_idx)} rows/"
         f"{n_va_ep} eps")
    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers,
                           drop_last=True)
    va_loader = DataLoader(Subset(ds, va_idx), batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    best_val = math.inf
    log: list[dict] = []
    config = {
        "stage": "l2_only", "d_model": args.d_model, "lr": args.lr,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "pair_cache_path": str(args.pair_cache_path),
        "all_snapshots": True,
    }
    config.update(model.config_extra)

    def _save(path: Path, epoch: int) -> None:
        torch.save({
            "model": model.state_dict(), "aux": aux.state_dict(),
            "epoch": epoch, "config": config,
        }, path)

    for epoch in range(1, args.epochs + 1):
        model.train(); aux.train()
        t0, run, cnt = time.time(), {}, {}
        for i, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx, ps, gl, pm_now = _forward_l2(model, encoders, batch, device)
            loss, mets = dual_l2_aux_loss(aux(ctx, ps, gl), batch, pm_now)
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in mets.items():
                run[k] = run.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
            if args.progress_every and i % args.progress_every == 0:
                _log(f"ep {epoch} step {i}/{len(tr_loader)} "
                     f"loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
        tr_mets = {k: run[k] / cnt[k] for k in run}

        model.eval(); aux.eval()
        vrun, vcnt = {}, {}
        with torch.no_grad():
            for batch in va_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                ctx, ps, gl, pm_now = _forward_l2(model, encoders, batch, device)
                vloss, vmets = dual_l2_aux_loss(aux(ctx, ps, gl), batch, pm_now)
                vmets["total"] = vloss.item()
                for k, v in vmets.items():
                    vrun[k] = vrun.get(k, 0.0) + v; vcnt[k] = vcnt.get(k, 0) + 1
        va_mets = {k: vrun[k] / vcnt[k] for k in vrun}

        _log(f"epoch {epoch} done in {time.time()-t0:.0f}s | "
             f"val_total={va_mets['total']:.4f} | "
             f"val owner5_acc={va_mets.get('p/owner5_acc', float('nan')):.3f} "
             f"owner10_acc={va_mets.get('p/owner10_acc', float('nan')):.3f} "
             f"earliest_acc={va_mets.get('p/earliest_acc', float('nan')):.3f}")
        for group in ("p/", "pl/", "g/"):
            row = "  ".join(f"{k.split('/')[1]}={va_mets[k]:.4f}"
                            for k in sorted(va_mets) if k.startswith(group)
                            and not k.endswith("_acc"))
            _log(f"  val {group:<3s} {row}")
        log.append({"epoch": epoch, "train": tr_mets, "val": va_mets})
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
        _save(out_dir / "l2_last.pt", epoch)
        if va_mets["total"] < best_val:
            best_val = va_mets["total"]
            _save(out_dir / "l2_best.pt", epoch)
            _log(f"  new best (val_total={best_val:.4f}) -> l2_best.pt")

    return out_dir / "l2_best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pair-cache-path", type=Path, required=True)
    p.add_argument("--fleet-run-dir", type=Path, required=True)
    p.add_argument("--planet-run-dir", type=Path, required=True)
    p.add_argument("--comet-run-dir", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, default=None,
                   help="v2 or v3-shaped ckpt (auto-adapted)")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    train_l2(p.parse_args())


if __name__ == "__main__":
    main()
