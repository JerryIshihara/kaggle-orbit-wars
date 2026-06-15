"""Stage 3: train the set-level value head Q_set (+ shared Q-trunk) on the
episode OUTCOME. Warm-starts the set-token ckpt (trunk + set encoder + the
forward-compat q_trunk/head stubs).

Q_set(s, A) = value of committing the turn's action SET A, regressed (Huber) to
the episode MC return (+1 win / -1 loss) from the outcome sidecar. Accuracy
matters: it's the ENERGY the (source × target × size) Q-guided diffusion search
optimizes, so a LARGE good/bad dataset is the point. No doomed label (set-Q uses
the outcome, not the per-pair plan_launch verdict).

    python -m agents.transformer_v3.qset_pretrain \
        --out-dir data/runs/joint/qset_<tag> \
        --pair-cache-path .../pair_cache.pt --outcome-sidecar .../outcome.pt \
        --fleet-run-dir ... --planet-run-dir ... --comet-run-dir ... \
        --warm-start <set_token ckpt>
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..transformer_v2.pretrain.joint_pretrain import (
    _forward_context_from_batch, _load_encoders)
from .model import EntityPretrainModelV3, adapt_v2_state_dict
from .history import UNION_HISTORY_OFFSETS
from .l2_pretrain import _episode_split
from .action_set_encoder import (
    ActionSetEncoder, FracEmbed, SharedQTrunk, SetValueHeads)
from .set_token_pretrain import _gather_launches


def _log(m: str) -> None:
    print(f"[qset {time.strftime('%H:%M:%S')}] {m}", flush=True)


class OutcomePairDataset(Dataset):
    """Pair-cache subset that attaches the episode outcome (+1/-1) per item,
    keyed by (episode_id, turn). Drops snapshots with no outcome label."""

    def __init__(self, base, indices, outcome: dict, forecast: dict | None = None):
        self.base = base
        self.outcome = outcome
        self.forecast = forecast or {}
        self.indices = [i for i in indices
                        if (base.keys[i][0], int(base.keys[i][1])) in outcome]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        item = self.base[idx]
        ep, t = self.base.keys[idx]
        key = (ep, int(t))
        item["outcome"] = torch.tensor(float(self.outcome[key]), dtype=torch.float32)
        fc = self.forecast.get(key)
        if fc is not None:                                  # aux 5×5 forecast labels
            item["p1_future"] = fc["p1_future"].float()
            item["p1_valid"] = fc["p1_valid"].float()
        return item


def _qset_step(model, set_enc, frac_embed, ships_embed, q_trunk, vh, encoders,
               batch, *, d_frac, d_ships, aux_weight=0.0):
    out = _forward_context_from_batch(
        model, encoders[0], encoders[1], encoders[2], batch)
    glob = out["glob"]
    player = out["player_state"][:, 0]
    launch, pad, _src, _owned, _fired = _gather_launches(
        out, batch, frac_embed, ships_embed, d_frac, d_ships)
    set_token, _ = set_enc(launch, pad)                    # full set
    q_emb = q_trunk(set_token, glob, player)
    heads = vh(q_emb)
    q_set = heads["q_set"]                                 # (B,)
    y = batch["outcome"]                                   # (B,) +1/-1
    loss = F.huber_loss(q_set, y, delta=1.0)
    terms = {"huber": float(loss.detach())}
    # AUX FORECAST — action-conditioned 5 signals × 5 horizons → realized t+h
    if aux_weight > 0 and "p1_future" in batch:
        tgt = batch["p1_future"]                           # (B,5,5)
        vmask = batch["p1_valid"].view(-1, 1, tgt.shape[-1])  # (B,1,5) per-horizon
        hub = F.huber_loss(heads["aux_fwd"], tgt, delta=0.25, reduction="none")
        aux = (hub * vmask).sum() / vmask.expand_as(hub).sum().clamp(min=1)
        loss = loss + aux_weight * aux
        terms["aux_fwd"] = float(aux.detach())
    with torch.no_grad():
        qd = q_set.detach()
        sign_acc = ((qd.sign() == y.sign()) | (y == 0)).float().mean().item()
        corr = _corr(qd, y)
    terms.update({"sign_acc": sign_acc, "corr": corr, "q_mean": float(qd.mean()),
                  "q_win": float(qd[y > 0].mean()) if (y > 0).any() else float("nan"),
                  "q_loss": float(qd[y < 0].mean()) if (y < 0).any() else float("nan")})
    return loss, terms


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm()).clamp(min=1e-6)
    return float((a * b).sum() / d)


def train_qset(args) -> Path:
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
    d_frac = int(wcfg.get("d_frac", args.d_frac))
    d_ships = int(wcfg.get("d_ships", args.d_ships))
    model = EntityPretrainModelV3(
        d_model=args.d_model,
        conditioner_n_layers=int(wcfg.get("conditioner_n_layers", 3)),
        head_n_layers=int(wcfg.get("head_n_layers", 3)),
        with_consolidator=True, with_value_heads=False, with_short_aux=True).to(device)
    model.load_state_dict(adapt_v2_state_dict(
        {k: v for k, v in warm["model"].items()
         if not k.startswith("value_heads.")}), strict=False)
    set_enc = ActionSetEncoder(args.d_model, 2 * args.d_model + d_frac + d_ships,
                               n_layers=int(wcfg.get("set_layers", 2))).to(device)
    frac_embed = FracEmbed(d_frac).to(device)
    ships_embed = FracEmbed(d_ships).to(device)
    q_trunk = SharedQTrunk(args.d_model).to(device)
    vh = SetValueHeads(args.d_model).to(device)
    # load the set-token stage's set encoder / embeds / (fresh) q_trunk+heads
    for mod, key in ((set_enc, "set_enc"), (frac_embed, "frac_embed"),
                     (ships_embed, "ships_embed"), (q_trunk, "q_trunk"),
                     (vh, "set_value_heads")):
        if key in warm:
            mod.load_state_dict(warm[key])
    _log(f"warm-start {args.warm_start}: trunk + set-token modules loaded "
         f"(q_trunk/heads {'loaded' if 'q_trunk' in warm else 'FRESH'})")
    del warm

    for p in model.parameters():
        p.requires_grad_(False)
    groups = []

    def _grp(mods, lr, name):
        ps = [p for m in mods for p in m.parameters()]
        for p in ps:
            p.requires_grad_(True)
        if ps:
            groups.append({"params": ps, "lr": lr})
            _log(f"  lr {name:<6s} lr={lr:g}  {sum(p.numel() for p in ps):,}")

    _grp([q_trunk, vh], args.lr_q, "q")                    # value heads — fast
    _grp([set_enc, frac_embed, ships_embed], args.lr_set, "set")  # fine-tune
    _grp([model.dual_role, model.joint_role], args.lr_l34, "l34")  # tiny
    _grp([model.cross], args.lr_l2, "l2")                  # tiny
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    _log(f"loading outcome sidecar {args.outcome_sidecar} ...")
    outcome = torch.load(args.outcome_sidecar, weights_only=False)["outcome"]
    forecast = (torch.load(args.forecast_sidecar, weights_only=False)["forecast"]
                if args.forecast_sidecar else None)
    _log(f"  {len(outcome)} outcome labels" + (
        f" | {len(forecast)} forecast labels — aux 5×5 ON (w={args.aux_weight})"
        if forecast else " — aux OFF"))
    pair_ds = CachedPairDataset(args.pair_cache_path)
    if tuple(pair_ds.history_offsets) != UNION_HISTORY_OFFSETS:
        pair_ds.history_offsets = UNION_HISTORY_OFFSETS
    acted = list(pair_ds.acted_indices)
    a_tr, a_va, _, _ = _episode_split(
        [pair_ds.keys[i] for i in acted], args.val_frac, args.seed)
    tr = OutcomePairDataset(pair_ds, [acted[i] for i in a_tr], outcome, forecast)
    va = OutcomePairDataset(pair_ds, [acted[i] for i in a_va], outcome, forecast)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, drop_last=True)
    va_loader = DataLoader(va, batch_size=args.batch_size, num_workers=args.num_workers)
    _log(f"train {len(tr)} / val {len(va)} labeled snapshots")

    config = dict(wcfg); config.update({"stage": "qset_pretrain",
                                        "d_frac": d_frac, "d_ships": d_ships})
    best = -math.inf; logj = []

    def _save(path, ep):
        torch.save({"model": model.state_dict(), "set_enc": set_enc.state_dict(),
                    "frac_embed": frac_embed.state_dict(),
                    "ships_embed": ships_embed.state_dict(),
                    "q_trunk": q_trunk.state_dict(),
                    "set_value_heads": vh.state_dict(),
                    "epoch": ep, "config": config}, path)

    mods = (model, set_enc, frac_embed, ships_embed, q_trunk, vh)
    for epoch in range(1, args.epochs + 1):
        for m in mods:
            m.train()
        t0, run, cnt = time.time(), {}, {}
        for i, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, terms = _qset_step(model, set_enc, frac_embed, ships_embed,
                                     q_trunk, vh, encoders, batch,
                                     d_frac=d_frac, d_ships=d_ships,
                                     aux_weight=args.aux_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for g in groups for p in g["params"]), args.grad_clip)
            opt.step()
            for k, v in terms.items():
                if isinstance(v, float) and not math.isnan(v):
                    run[k] = run.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
            if args.progress_every and i % args.progress_every == 0:
                _log(f"ep{epoch} {i}/{len(tr_loader)} huber={terms['huber']:.3f} "
                     f"sign_acc={terms['sign_acc']:.3f} corr={terms['corr']:+.3f} "
                     f"(w {terms['q_win']:+.2f} / l {terms['q_loss']:+.2f}) "
                     f"({time.time()-t0:.0f}s)")
        for m in mods:
            m.eval()
        vrun, vcnt = {}, {}
        with torch.no_grad():
            for batch in va_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, terms = _qset_step(model, set_enc, frac_embed, ships_embed,
                                      q_trunk, vh, encoders, batch,
                                      d_frac=d_frac, d_ships=d_ships,
                                      aux_weight=args.aux_weight)
                for k, v in terms.items():
                    if isinstance(v, float) and not math.isnan(v):
                        vrun[k] = vrun.get(k, 0.0) + v; vcnt[k] = vcnt.get(k, 0) + 1
        vm = {k: vrun[k] / vcnt[k] for k in vrun}
        _log(f"epoch {epoch} {time.time()-t0:.0f}s | VAL sign_acc={vm.get('sign_acc', float('nan')):.3f} "
             f"corr={vm.get('corr', float('nan')):+.3f} huber={vm.get('huber', float('nan')):.3f} "
             f"| Q win {vm.get('q_win', float('nan')):+.2f} vs loss {vm.get('q_loss', float('nan')):+.2f}")
        logj.append({"epoch": epoch, "val": vm})
        (out_dir / "log.json").write_text(json.dumps(logj, indent=1))
        _save(out_dir / "qset_last.pt", epoch)
        if vm.get("sign_acc", -math.inf) > best:
            best = vm["sign_acc"]; _save(out_dir / "qset_best.pt", epoch)
            _log(f"  new best (val sign_acc={best:.4f})")
    return out_dir / "qset_best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pair-cache-path", type=Path, required=True)
    p.add_argument("--outcome-sidecar", type=Path, required=True)
    p.add_argument("--forecast-sidecar", type=Path, default=None,
                   help="aux 5×5 signal-forecast labels (cross-cache join); aux OFF if absent")
    p.add_argument("--aux-weight", type=float, default=0.25)
    p.add_argument("--fleet-run-dir", type=Path, required=True)
    p.add_argument("--planet-run-dir", type=Path, required=True)
    p.add_argument("--comet-run-dir", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, required=True)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-frac", type=int, default=32)
    p.add_argument("--d-ships", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr-q", type=float, default=1e-4)
    p.add_argument("--lr-set", type=float, default=2e-5,
                   help="fine-tune the set encoder toward value structure")
    p.add_argument("--lr-l34", type=float, default=5e-6)
    p.add_argument("--lr-l2", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    train_qset(p.parse_args())


if __name__ == "__main__":
    main()
