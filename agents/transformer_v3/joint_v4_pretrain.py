"""Stage-D pretrain: v4 actor + pruned value heads, jointly, four LR tiers.

    python -m agents.transformer_v3.joint_v4_pretrain \
        --out-dir data/runs/joint/jointv4_<tag> \
        --pair-cache-path .../topmeta300_p64_f512_acted.pt \
        --cross-cache-path .../cross_topmeta300_cap150.pt \
        --fleet-run-dir ... --planet-run-dir ... --comet-run-dir ... \
        --warm-start actor_best.pt

User-ordered composition:
  * ACTION side keeps training (select BCE + Dirichlet NLL + α0 confidence
    + t+5 short aux) — at a LOWER lr than the value side, alongside L2/L3/L4
    (slow backbone, fast value heads):
        value heads (win/fwd + inbound aux)   lr 1e-4
        action heads (PairHead + α0 + sh5)    lr 2e-5
        L3 + L4                               lr 1e-5
        dual L2                               lr 5e-6
  * VALUE side = the pruned v4 task set (``value_v4``): win + fwd kept;
    back/rank/survives REMOVED (no loss term); ADDED temporal contrast
    (sibling snapshots, comparator monotonicity for the sample-K ranker)
    and the per-slot inbound aux (fleet-timing into player_state).
  * by-game holdout each epoch (win-memorization gap + contrast_acc).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ..transformer_v2.pretrain.joint_pretrain import (
    _forward_context_from_batch,
    _load_encoders,
    _split_value_by_game,
)
from ..transformer_v2.pretrain.cross_entity import (
    CachedCrossEntitySnapshotDataset,
)
from .model import EntityPretrainModelV3, adapt_v2_state_dict
from .history import UNION_HISTORY_OFFSETS
from .actor_pretrain import _step_losses as _action_step_losses
from .l2_pretrain import _episode_split
from .value_v4 import PlayerInboundAux, compute_value_v4_loss


def _log(msg: str) -> None:
    print(f"[jointv4 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class PairedValueDataset(Dataset):
    """Yields (snapshot, sibling-at-t+Δ) merged dicts for the contrast task.

    Sibling keys are prefixed ``b::``. Δ drawn from ``deltas``; falls back
    to the largest available sibling (or itself at episode end — the
    contrast loss then sees label≈0.5 noise on a handful of rows).
    """

    def __init__(self, base: CachedCrossEntitySnapshotDataset,
                 indices: list[int], deltas=(5, 10, 20), seed: int = 0):
        self.base = base
        self.indices = list(indices)
        self.deltas = tuple(deltas)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        ep, t = self.base.keys[idx]
        for d in self.rng.sample(self.deltas, len(self.deltas)):
            j = self.base._key_to_idx.get((ep, t + d))
            if j is not None:
                break
        else:
            j = idx
        a = self.base[idx]
        b = self.base[j]
        out = dict(a)
        out.update({f"b::{k}": v for k, v in b.items()})
        return out


def _split_b(batch: dict):
    a = {k: v for k, v in batch.items() if not k.startswith("b::")}
    b = {k[3:]: v for k, v in batch.items() if k.startswith("b::")}
    return a, b


def _cycle(loader):
    while True:
        for x in loader:
            yield x


def train_joint_v4(args) -> Path:
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
        with_consolidator=True, with_value_heads=True,
        value_dropout=args.value_dropout,
        with_short_aux=True, with_alloc_conc=True,
    ).to(device)
    inbound_head = PlayerInboundAux(args.d_model).to(device)
    sd = {k: v for k, v in warm_ck["model"].items()
          if not k.startswith("value_heads.")}
    res = model.load_state_dict(adapt_v2_state_dict(sd), strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("value_heads.")]
    assert not bad, f"backbone skew vs actor ckpt: {bad[:6]}"
    _log(f"warm-start {args.warm_start}: {len(sd)} actor tensors; value "
         f"heads + inbound aux fresh")
    del warm_ck, sd

    # ---- four LR tiers (L0/L1 frozen; dead value heads excluded) -----
    for p in model.parameters():
        p.requires_grad_(False)
    groups = []

    def _grp(named, lr, name):
        params = []
        for m in named:
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad_(True)
                params.append(p)
        if params:
            groups.append({"params": params, "lr": lr})
            _log(f"  lr group {name:<7s} lr={lr:g}  "
                 f"{sum(p.numel() for p in params):,} params")

    vh = model.value_heads
    _grp([vh.value_trunk, vh.win_head, vh.fwd_heads, inbound_head],
         args.lr_value, "value")
    # back/rank/survives heads: NO loss, NO optimizer — dead weights.
    _grp([model.pair_head, model.alloc_conc_head, model.short_heads],
         args.lr_action, "action")
    _grp([model.dual_role, model.joint_role], args.lr_l34, "l34")
    _grp([model.cross], args.lr_l2, "l2")
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    # ---- data ---------------------------------------------------------
    pair_ds = CachedPairDataset(args.pair_cache_path)
    if tuple(pair_ds.history_offsets) != UNION_HISTORY_OFFSETS:
        pair_ds.history_offsets = UNION_HISTORY_OFFSETS
    acted = list(pair_ds.acted_indices)
    a_tr, a_va, _, _ = _episode_split(
        [pair_ds.keys[i] for i in acted], args.val_frac, args.seed)
    act_loader = DataLoader(
        torch.utils.data.Subset(pair_ds, [acted[i] for i in a_tr]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True)

    cross_ds = CachedCrossEntitySnapshotDataset(
        args.cross_cache_path, history_offsets=UNION_HISTORY_OFFSETS)
    v_tr, v_va = _split_value_by_game(cross_ds)
    val_pair_ds = PairedValueDataset(cross_ds, list(v_tr.indices),
                                     seed=args.seed)
    val_loader = DataLoader(val_pair_ds, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            drop_last=True)
    ho_pair_ds = (PairedValueDataset(cross_ds, list(v_va.indices),
                                     seed=args.seed + 1)
                  if v_va is not None else None)
    ho_loader = (DataLoader(ho_pair_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)
                 if ho_pair_ds is not None else None)
    n_steps = max(len(act_loader), len(val_loader))
    _log(f"action: {len(act_loader)} batches/ep | value: {len(val_loader)} "
         f"batches/ep (paired) | holdout {len(ho_pair_ds) if ho_pair_ds else 0} rows | "
         f"{n_steps} interleaved steps/epoch")

    config = dict(wcfg)
    config.update({
        "stage": "joint_v4", "with_value_heads": True,
        "lr_value": args.lr_value, "lr_action": args.lr_action,
        "lr_l34": args.lr_l34, "lr_l2": args.lr_l2,
        "value_tasks": "win,fwd,contrast,inbound (back/rank/survives removed)",
        "pair_cache_path": str(args.pair_cache_path),
        "cross_cache_path": str(args.cross_cache_path),
        "value_warm_start": str(args.warm_start),
    })

    best = math.inf
    log: list[dict] = []

    def _save(path: Path, epoch: int) -> None:
        torch.save({
            "model": model.state_dict(),
            "inbound_head": inbound_head.state_dict(),
            "epoch": epoch, "config": config,
        }, path)

    for epoch in range(1, args.epochs + 1):
        model.train(); inbound_head.train()
        t0, run, cnt = time.time(), {}, {}
        a_it, v_it = _cycle(act_loader), _cycle(val_loader)
        for i in range(1, n_steps + 1):
            ab = {k: v.to(device) for k, v in next(a_it).items()}
            a_loss, a_terms = _action_step_losses(
                model, encoders, ab,
                pair_pos_weight=args.pair_pos_weight,
                alloc_weight=args.alloc_weight,
                short_aux_weight=args.short_aux_weight,
            )
            vb_all = {k: v.to(device) for k, v in next(v_it).items()}
            va_b, vb_b = _split_b(vb_all)
            out_a = _forward_context_from_batch(
                model, encoders[0], encoders[1], encoders[2], va_b)
            out_b = _forward_context_from_batch(
                model, encoders[0], encoders[1], encoders[2], vb_b)
            v_loss, v_terms = compute_value_v4_loss(
                out_a, va_b, inbound_head=inbound_head,
                out_b=out_b, batch_b=vb_b,
            )
            loss = a_loss + args.value_coef * v_loss
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in {**{f"a/{k}": v for k, v in a_terms.items()},
                         **{f"v/{k}": v for k, v in v_terms.items()}}.items():
                if isinstance(v, float) and math.isnan(v):
                    continue
                run[k] = run.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
            if args.progress_every and i % args.progress_every == 0:
                _log(f"ep {epoch} step {i}/{n_steps} loss={loss.item():.4f} "
                     f"({time.time()-t0:.0f}s)")
        tr = {k: run[k] / cnt[k] for k in run}

        ho = {}
        if ho_loader is not None:
            model.eval(); inbound_head.eval()
            hrun, hcnt = {}, {}
            with torch.no_grad():
                for vb_all in ho_loader:
                    vb_all = {k: v.to(device) for k, v in vb_all.items()}
                    va_b, vb_b = _split_b(vb_all)
                    out_a = _forward_context_from_batch(
                        model, encoders[0], encoders[1], encoders[2], va_b)
                    out_b = _forward_context_from_batch(
                        model, encoders[0], encoders[1], encoders[2], vb_b)
                    hl, ht = compute_value_v4_loss(
                        out_a, va_b, inbound_head=inbound_head,
                        out_b=out_b, batch_b=vb_b)
                    ht["total"] = float(hl)
                    for k, v in ht.items():
                        if isinstance(v, float) and math.isnan(v):
                            continue
                        hrun[k] = hrun.get(k, 0.0) + v
                        hcnt[k] = hcnt.get(k, 0) + 1
            ho = {k: hrun[k] / hcnt[k] for k in hrun}

        _log(f"epoch {epoch} done in {time.time()-t0:.0f}s | "
             f"a: nll={tr.get('a/dir/alloc_nll', float('nan')):+.3f} "
             f"sat={tr.get('a/dir/alpha0_satfrac', float('nan')):.2f} | "
             f"v train win_acc={tr.get('v/win_acc', float('nan')):.3f} | "
             f"HOLDOUT win_acc={ho.get('win_acc', float('nan')):.3f} "
             f"contrast_acc={ho.get('contrast_acc', float('nan')):.3f} "
             f"(gap {tr.get('v/win_acc', 0) - ho.get('win_acc', 0):+.3f})")
        log.append({"epoch": epoch, "train": tr, "holdout": ho})
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
        _save(out_dir / "jointv4_last.pt", epoch)
        crit = ho.get("total", tr.get("v/win", math.inf))
        if crit < best:
            best = crit
            _save(out_dir / "jointv4_best.pt", epoch)
            _log(f"  new best (holdout total={best:.4f}) -> jointv4_best.pt")

    return out_dir / "jointv4_best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pair-cache-path", type=Path, required=True)
    p.add_argument("--cross-cache-path", type=Path, required=True)
    p.add_argument("--fleet-run-dir", type=Path, required=True)
    p.add_argument("--planet-run-dir", type=Path, required=True)
    p.add_argument("--comet-run-dir", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, required=True)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr-value", type=float, default=1e-4)
    p.add_argument("--lr-action", type=float, default=2e-5)
    p.add_argument("--lr-l34", type=float, default=1e-5)
    p.add_argument("--lr-l2", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-coef", type=float, default=1.0)
    p.add_argument("--value-dropout", type=float, default=0.1)
    p.add_argument("--pair-pos-weight", type=float, default=600.0)
    p.add_argument("--alloc-weight", type=float, default=1.0)
    p.add_argument("--short-aux-weight", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    train_joint_v4(p.parse_args())


if __name__ == "__main__":
    main()
