"""Stage: pretrain the ACTION-SET TOKEN (self-supervised, NO outcome).

Trunk (L0/L1 frozen, L2/L3/L4 small-LR) → source_joint/target_joint/glob/
player → gather the turn's fired launches into a set → ActionSetEncoder →
two joint tasks through the shared SetReconHead:
  B) AUTOENCODER  : full set → set_token → reconstruct EVERY owned source's
                    target/HOLD.
  A) MASKED DENOISE: mask a random subset of fired launches → predict the
                    MASKED sources' targets from the rest + the state.
Label for both = the expert per-source assignment (target or HOLD). The Q_set
value head (outcome-supervised) is a LATER stage on top of this token.

    python -m agents.transformer_v3.set_token_pretrain \
        --out-dir data/runs/joint/set_token_<tag> \
        --pair-cache-path .../pair_cache.pt \
        --fleet-run-dir ... --planet-run-dir ... --comet-run-dir ... \
        --warm-start <trunk ckpt>
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
    _forward_context_from_batch, _load_encoders)
from ..transformer_v2.pretrain.entity_encoder import (
    _pair_frac_targets_from_batch, _owned_source_rows)
from ..transformer_v2.featurizer.fleet_featurizer import SHIPS_LOG_MAX
from .model import EntityPretrainModelV3, adapt_v2_state_dict
from .history import UNION_HISTORY_OFFSETS
from .l2_pretrain import _episode_split
from .action_set_encoder import (
    ActionSetEncoder, SetReconHead, FracEmbed, set_pretrain_loss,
    SharedQTrunk, SetValueHeads)


def _log(m: str) -> None:
    print(f"[set-tok {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _gather_launches(out, batch, frac_embed, ships_embed, d_frac, d_ships):
    """Build padded launch tokens for each turn's fired set.

    Launch token = source_joint[s] ⊕ target_joint[t] ⊕ frac_embed(f) ⊕
    ships_embed(log1p(ships_sent)). frac = size-INVARIANT signal; log-ships =
    the ABSOLUTE sample-specific force (so equal-frac/diff-garrison launches get
    different Q — the fleet-size fix). source/target_joint already carry the
    garrison + the frac-head's latent context.

    Returns launch_tok (B,Kmax, 2d+d_frac+d_ships), pad_mask, src_slot, owned,
    fired_src."""
    sj = out["source_joint"]                       # (B,P,d)
    tj = out["target_joint"]                       # (B,P,d)
    B, P, d = sj.shape
    device = sj.device
    pair_labels = batch["pair_labels"].bool()
    owned = _owned_source_rows(batch, batch["pair_valid"].bool())   # (B,P)
    fired_src = owned & pair_labels.any(-1)                          # (B,P)
    tgt_idx = pair_labels.float().argmax(-1)                         # (B,P)
    ships = batch["pair_ships"].float() if "pair_ships" in batch else None
    frac = (_pair_frac_targets_from_batch(batch, ships)
            if ships is not None else torch.zeros(B, P, P, device=device))
    Ks = fired_src.sum(-1)
    Kmax = int(Ks.max().clamp(min=1).item())
    launch = torch.zeros(B, Kmax, 2 * d + d_frac + d_ships, device=device)
    src_slot = torch.full((B, Kmax), -1, dtype=torch.long, device=device)
    pad = torch.ones(B, Kmax, dtype=torch.bool, device=device)
    for b in range(B):
        srcs = fired_src[b].nonzero(as_tuple=False).flatten()
        for i, s in enumerate(srcs.tolist()):
            t = int(tgt_idx[b, s])
            fe = frac_embed(frac[b, s, t].view(1)).squeeze(0)
            sh = ships[b, s, t] if ships is not None else torch.zeros((), device=device)
            ls = (torch.log1p(sh) / SHIPS_LOG_MAX).clamp(0.0, 1.0)
            se = ships_embed(ls.view(1)).squeeze(0)
            launch[b, i] = torch.cat([sj[b, s], tj[b, t], fe, se])
            src_slot[b, i] = s
            pad[b, i] = False
    return launch, pad, src_slot, owned, fired_src


def _step(model, set_enc, recon, frac_embed, ships_embed, encoders, batch, *,
          d_frac, d_ships, mask_ratio):
    out = _forward_context_from_batch(
        model, encoders[0], encoders[1], encoders[2], batch)
    pair_labels = batch["pair_labels"].bool()
    sj, tj = out["source_joint"], out["target_joint"]
    glob = out["glob"]
    player = out["player_state"][:, 0]                  # learner-relative slot 0
    launch, pad, src_slot, owned, fired = _gather_launches(
        out, batch, frac_embed, ships_embed, d_frac, d_ships)

    # B) autoencoder — full set → reconstruct ALL owned rows
    set_b, _ = set_enc(launch, pad)
    pair_b, hold_b = recon(set_b, sj, tj, glob, player)
    loss_b, d_b = set_pretrain_loss(pair_b, hold_b, pair_labels, owned, owned)

    # A) masked denoise — mask a subset of fired launches, predict their rows
    nonpad = ~pad
    rnd = torch.rand_like(launch[..., 0])
    mask_sel = nonpad & (rnd < mask_ratio)
    # guarantee >=1 masked per row that has any fired launch
    none = nonpad.any(1) & ~mask_sel.any(1)
    if bool(none.any()):
        first = nonpad.float().argmax(1)
        mask_sel[none, first[none]] = True
    pred_rows = torch.zeros_like(owned)
    bi, ki = mask_sel.nonzero(as_tuple=True)
    sl = src_slot[bi, ki]
    pred_rows[bi, sl] = True
    set_a, _ = set_enc(launch, pad, mask_sel)
    pair_a, hold_a = recon(set_a, sj, tj, glob, player)
    loss_a, d_a = set_pretrain_loss(pair_a, hold_a, pair_labels, owned, pred_rows)

    total = loss_b + loss_a
    terms = {f"ae/{k}": v for k, v in d_b.items()}
    terms.update({f"mask/{k}": v for k, v in d_a.items()})
    terms["total"] = float(total.detach())
    return total, terms


def train_set_token(args) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from scripts.build_pair_dataset_orbital_occle import CachedPairDataset

    _log(f"device={device}  loading L0 ...")
    encoders = _load_encoders(
        Path(args.fleet_run_dir), Path(args.planet_run_dir),
        Path(args.comet_run_dir), device=device, expected_d_model=args.d_model)
    for enc in encoders:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()

    warm = torch.load(args.warm_start, map_location=device, weights_only=False)
    wcfg = warm.get("config", {})
    model = EntityPretrainModelV3(
        d_model=args.d_model,
        conditioner_n_layers=int(wcfg.get("conditioner_n_layers", 3)),
        head_n_layers=int(wcfg.get("head_n_layers", 3)),
        with_consolidator=True, with_value_heads=False, with_short_aux=True,
    ).to(device)
    sd = {k: v for k, v in warm["model"].items() if not k.startswith("value_heads.")}
    model.load_state_dict(adapt_v2_state_dict(sd), strict=False)
    _log(f"warm-start {args.warm_start}: trunk loaded")
    del warm, sd

    d_frac, d_ships = args.d_frac, args.d_ships
    frac_embed = FracEmbed(d_frac).to(device)               # size-invariant
    ships_embed = FracEmbed(d_ships).to(device)             # absolute force (log-ships)
    set_enc = ActionSetEncoder(args.d_model, 2 * args.d_model + d_frac + d_ships,
                               n_layers=args.set_layers).to(device)
    recon = SetReconHead(args.d_model).to(device)
    # Shared Q-trunk + value/forecast heads — baked in NOW (saved fresh) so the
    # ckpt is forward-compatible; TRAINED in the next stage (Q_set outcome + aux
    # forecasts on the doubled good/bad cache). Not in this stage's LR groups.
    q_trunk = SharedQTrunk(args.d_model).to(device)
    set_value_heads = SetValueHeads(args.d_model).to(device)

    # ---- LR groups: set modules fast; L2/L3/L4 SMALL; L0/L1 + heads frozen ----
    for p in model.parameters():
        p.requires_grad_(False)
    groups = []

    def _grp(mods, lr, name):
        ps = [p for m in mods if m is not None for p in m.parameters()]
        for p in ps:
            p.requires_grad_(True)
        if ps:
            groups.append({"params": ps, "lr": lr})
            _log(f"  lr {name:<7s} lr={lr:g}  {sum(p.numel() for p in ps):,}")

    _grp([set_enc, recon, frac_embed, ships_embed], args.lr_set, "set")
    _grp([model.dual_role, model.joint_role], args.lr_l34, "l34")
    _grp([model.cross], args.lr_l2, "l2")
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    pair_ds = CachedPairDataset(args.pair_cache_path)
    if tuple(pair_ds.history_offsets) != UNION_HISTORY_OFFSETS:
        pair_ds.history_offsets = UNION_HISTORY_OFFSETS
    acted = list(pair_ds.acted_indices)
    a_tr, a_va, _, _ = _episode_split(
        [pair_ds.keys[i] for i in acted], args.val_frac, args.seed)
    tr_loader = DataLoader(Subset(pair_ds, [acted[i] for i in a_tr]),
                           batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, drop_last=True)
    va_loader = DataLoader(Subset(pair_ds, [acted[i] for i in a_va]),
                           batch_size=args.batch_size, num_workers=args.num_workers)
    _log(f"train {len(tr_loader)} / val {len(va_loader)} batches")

    config = dict(wcfg)
    config.update({"stage": "set_token_pretrain", "set_layers": args.set_layers,
                   "d_frac": d_frac, "mask_ratio": args.mask_ratio,
                   "tasks": "autoencoder+masked_denoise"})
    best = math.inf
    log = []

    def _save(path, ep):
        torch.save({"model": model.state_dict(), "set_enc": set_enc.state_dict(),
                    "recon": recon.state_dict(), "frac_embed": frac_embed.state_dict(),
                    "ships_embed": ships_embed.state_dict(),
                    "q_trunk": q_trunk.state_dict(),           # fresh stub (next stage)
                    "set_value_heads": set_value_heads.state_dict(),
                    "epoch": ep, "config": config}, path)

    for epoch in range(1, args.epochs + 1):
        model.train(); set_enc.train(); recon.train()
        frac_embed.train(); ships_embed.train()
        t0, run, cnt = time.time(), {}, {}
        for i, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, terms = _step(model, set_enc, recon, frac_embed, ships_embed,
                                 encoders, batch, d_frac=d_frac, d_ships=d_ships,
                                 mask_ratio=args.mask_ratio)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for g in groups for p in g["params"]), args.grad_clip)
            opt.step()
            for k, v in terms.items():
                if isinstance(v, float) and not math.isnan(v):
                    run[k] = run.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
            if args.progress_every and i % args.progress_every == 0:
                _log(f"ep{epoch} {i}/{len(tr_loader)} loss={loss.item():.3f} "
                     f"ae_acc={terms.get('ae/acc', float('nan')):.3f} "
                     f"mask_acc={terms.get('mask/acc', float('nan')):.3f} "
                     f"({time.time()-t0:.0f}s)")
        tr = {k: run[k] / cnt[k] for k in run}

        model.eval(); set_enc.eval(); recon.eval()
        frac_embed.eval(); ships_embed.eval()
        vrun, vcnt = {}, {}
        with torch.no_grad():
            for batch in va_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, terms = _step(model, set_enc, recon, frac_embed, ships_embed,
                                 encoders, batch, d_frac=d_frac, d_ships=d_ships,
                                 mask_ratio=args.mask_ratio)
                for k, v in terms.items():
                    if isinstance(v, float) and not math.isnan(v):
                        vrun[k] = vrun.get(k, 0.0) + v; vcnt[k] = vcnt.get(k, 0) + 1
        va = {k: vrun[k] / vcnt[k] for k in vrun}
        _log(f"epoch {epoch} {time.time()-t0:.0f}s | VAL total={va.get('total', float('nan')):.3f} "
             f"AE acc={va.get('ae/acc', float('nan')):.3f} launch={va.get('ae/launch_acc', float('nan')):.3f} "
             f"| MASK acc={va.get('mask/acc', float('nan')):.3f} launch={va.get('mask/launch_acc', float('nan')):.3f}")
        log.append({"epoch": epoch, "train": tr, "val": va})
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
        _save(out_dir / "set_token_last.pt", epoch)
        if va.get("total", math.inf) < best:
            best = va["total"]; _save(out_dir / "set_token_best.pt", epoch)
            _log(f"  new best (val total={best:.4f})")
    return out_dir / "set_token_best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pair-cache-path", type=Path, required=True)
    p.add_argument("--fleet-run-dir", type=Path, required=True)
    p.add_argument("--planet-run-dir", type=Path, required=True)
    p.add_argument("--comet-run-dir", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, required=True)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-frac", type=int, default=32)
    p.add_argument("--d-ships", type=int, default=32,
                   help="embed dim for log1p(ships_sent) — the ABSOLUTE force")
    p.add_argument("--set-layers", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr-set", type=float, default=1e-4)
    p.add_argument("--lr-l34", type=float, default=1e-5,
                   help="SMALL — gently adapt the trunk's role attention")
    p.add_argument("--lr-l2", type=float, default=1e-5, help="SMALL — dual L2")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    train_set_token(p.parse_args())


if __name__ == "__main__":
    main()
