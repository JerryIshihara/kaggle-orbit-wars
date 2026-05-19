"""Probe the frozen encoder representation.

For a chosen run's frozen encoder + cross stack, precompute per-planet
``(glob, ctx, entity)`` tensors over a slice of the dataset, then train
small **linear probes** that try to reconstruct interpretable per-
planet labels straight from those representations:

  * inbound ships from each owner slot (0/1/2/3) within horizon h=10
  * total inbound ships (sum across slots)
  * count of friendly / enemy planets within radius R (already
    normalized in the snapshot)
  * distance to nearest enemy planet (normalized)

Each probe is a single ``Linear(d → 1)`` trained with MSE. We report
on val:

  * ``mae`` of the probe
  * ``mae_baseline`` of a constant predictor (the per-label train mean)
  * ``r2`` against the val labels (1 − mse / var)
  * ``rel_mae`` = probe_mae / baseline_mae  (lower is better, < 1.0
                  means the probe beats the trivial mean)

We probe both ``ctx`` (post-cross-attention, what the heads see) and
``entity_now`` (pre-cross-attention) to localize where information
lives:

  * ctx easy & entity easy   → input feature is preserved end-to-end
  * ctx hard & entity easy   → cross-attention destroyed the signal
  * ctx hard & entity hard   → never reached the encoder output
  * ctx easy & entity hard   → cross attention SYNTHESIZED the info
                               (rare, would indicate genuine new feature)

Run locally — fast: a 5 000-snapshot subset cost ≈ 7 min on CPU,
dominated by the encoder forward pass. The probes themselves train
in seconds.

Usage:
    python scripts/probe_encoder_representation.py \\
        --ckpt data/runs/pair_score/target_only_v2_Ebi_full_<TS>/pair_score_best.pt \\
        --max-rows 5000 --player Ebi
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from agents.archive.transformer_v1.aggregator import CrossEntityAttention
from agents.archive.transformer_v1.encoder import (
    FleetEncoder, PlanetEncoder, PlanetEntityEncoder,
)
from agents.archive.transformer_v1.pretrain.cross_entity import _entity_tokens_per_step
from agents.archive.transformer_v1.pretrain.pair_score import (
    PairScoreHead,
    PairScoreStack,
    TargetHead,
    FracHead,
    prepare_dataset,
    acted_only_indices,
)


def _build_stack_from_ckpt(ckpt_path: Path, device: str) -> PairScoreStack:
    """Reconstruct the frozen stack from a saved pair_score ckpt.

    Replicates the loader in :class:`agents.archive.transformer_v1.runner.TransformerAgent.load`
    but stops short of doing any inference plumbing — we only need
    ``stack.cross`` / ``stack.entity_encoder`` etc. for the probe.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    d_model = int(cfg.get("d_model", 64))
    hidden = int(cfg.get("hidden", 128))

    fenc = FleetEncoder(d_model=d_model); fenc.load_state_dict(ckpt["fleet_encoder"])
    penc = PlanetEncoder(d_model=d_model); penc.load_state_dict(ckpt["planet_encoder"])
    eenc = PlanetEntityEncoder(d_model=d_model); eenc.load_state_dict(ckpt["entity_encoder"])
    cross = CrossEntityAttention(d_model=d_model); cross.load_state_dict(ckpt["cross"])

    pair_head = PairScoreHead(d_model=d_model, hidden=hidden)
    pair_head.load_state_dict(ckpt["pair_score_head"])

    frac_head = None
    if "frac_head" in ckpt:
        frac_head = FracHead(d_model=d_model, hidden=int(cfg.get("frac_hidden", hidden)))
        frac_head.load_state_dict(ckpt["frac_head"])

    target_head = None
    if "target_head" in ckpt:
        target_head = TargetHead(
            d_model=d_model,
            hidden=int(cfg.get("target_hidden", hidden)),
            use_entity=bool(cfg.get("target_use_entity", False)),
            num_layers=int(cfg.get("target_num_layers", 2)),
        )
        target_head.load_state_dict(ckpt["target_head"])

    stack = PairScoreStack(
        fleet_encoder=fenc, planet_encoder=penc, entity_encoder=eenc,
        cross=cross, pair_score_head=pair_head,
        frac_head=frac_head, target_head=target_head,
    ).to(device).eval()
    for p in stack.parameters():
        p.requires_grad_(False)
    return stack


@torch.no_grad()
def _precompute(
    stack: PairScoreStack,
    dataset,
    indices: list[int],
    *,
    device: str,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Single encoder forward per snapshot. Returns per-snapshot CPU
    tensors: ``ctx`` (P, d), ``entity`` (P, d), ``planet_mask`` (P,)
    + the per-planet labels we want to probe."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)
    out: list[dict[str, torch.Tensor]] = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch,
            stack.fleet_encoder, stack.planet_encoder, stack.entity_encoder,
        )
        ctx, glob = stack.cross(entity_tokens, entity_mask)
        ctx_now = ctx[:, -1] if ctx.dim() == 4 else ctx
        entity_now = (
            entity_tokens[:, -1] if entity_tokens.dim() == 4 else entity_tokens
        )
        B = ctx_now.shape[0]
        # Some snapshot tensors carry the (T, P, …) history shape after
        # n_history wrapping. Reduce to the current-step view (last
        # along T) so per-planet labels align with ctx_now (B, P, d).
        def _now(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[:, -1] if tensor.dim() >= 3 and tensor.shape[1] == 3 else tensor

        mask_b = _now(batch["planet_mask"])
        arr_b = _now(batch["ships_arriving_within_10"])
        n_friend_b = _now(batch["n_friendly_within_R_norm"])
        n_enemy_b = _now(batch["n_enemy_within_R_norm"])
        near_enemy_b = _now(batch["nearest_enemy_dist_norm"])
        for i in range(B):
            mask = mask_b[i].bool().cpu()
            arr_h10 = arr_b[i].cpu()                                   # (P, 4)
            out.append({
                "ctx": ctx_now[i].cpu(),
                "entity": entity_now[i].cpu(),
                "mask": mask,
                "labels": {
                    "inbound_slot0_h10": arr_h10[:, 0],
                    "inbound_slot1_h10": arr_h10[:, 1],
                    "inbound_slot2_h10": arr_h10[:, 2],
                    "inbound_slot3_h10": arr_h10[:, 3],
                    "inbound_total_h10": arr_h10.sum(-1),
                    "n_friendly_R": n_friend_b[i].cpu(),
                    "n_enemy_R": n_enemy_b[i].cpu(),
                    "nearest_enemy_dist": near_enemy_b[i].cpu(),
                },
            })
        if bi % 10 == 0:
            done = (bi + 1) * batch_size
            elapsed = time.time() - t0
            print(f"  [precompute] {min(done, len(subset))}/{len(subset)} "
                  f"({elapsed:.0f}s, {elapsed / max(1, done):.3f}s/snap)",
                  flush=True)
    return out


def _flatten(cached: list[dict], feat_key: str, label_key: str
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack one feature tensor + one label across all snapshots, keeping
    only real planet slots (per ``planet_mask``)."""
    xs, ys = [], []
    for snap in cached:
        m = snap["mask"]
        if not m.any():
            continue
        xs.append(snap[feat_key][m])
        ys.append(snap["labels"][label_key][m])
    if not xs:
        return torch.empty(0), torch.empty(0)
    return torch.cat(xs, 0), torch.cat(ys, 0)


def _fit_linear_probe(
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor,
    *,
    epochs: int = 20,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> dict[str, float]:
    """Fit ``Linear(d → 1)`` with AdamW + MSE. Report MAE / R² on val,
    plus a constant-mean baseline for context."""
    d = x_train.shape[-1]
    probe = nn.Linear(d, 1)
    optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        # Full-batch fit — N is at most ~5000 × ~16 valid planets ≈ 80k
        # rows × d=64 ≈ 20 MB. Fits comfortably.
        pred = probe(x_train).squeeze(-1)
        loss = F.mse_loss(pred, y_train)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        pred_val = probe(x_val).squeeze(-1)
        mae = float((pred_val - y_val).abs().mean())
        mean = float(y_train.mean())
        baseline_mae = float((y_val - mean).abs().mean())
        var = float(((y_val - y_val.mean()) ** 2).mean())
        mse = float(((pred_val - y_val) ** 2).mean())
        r2 = 1.0 - mse / max(var, 1e-9)
    return {
        "mae": mae,
        "mae_baseline": baseline_mae,
        "rel_mae": mae / max(baseline_mae, 1e-9),
        "r2": r2,
        "var": var,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Saved pair_score ckpt to probe.")
    p.add_argument("--player", default="Ebi",
                   help="Player whose action CSVs anchor the dataset slice.")
    p.add_argument("--max-rows", type=int, default=5000,
                   help="How many acted snapshots to precompute.")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON output path for the probe results.")
    args = p.parse_args()

    print(f"[probe] ckpt: {args.ckpt}", flush=True)
    stack = _build_stack_from_ckpt(args.ckpt, args.device)
    print(f"[probe] stack loaded; frac_head={stack.frac_head is not None}, "
          f"target_head={stack.target_head is not None}", flush=True)

    dataset = prepare_dataset(
        player=args.player, filter_mode="all",
        max_planets=64, max_fleets=1024, n_history=3,
        cache_dir=None,
    )
    acted = acted_only_indices(dataset)
    if args.max_rows is not None:
        acted = acted[: args.max_rows]
    n_val = max(1, int(round(len(acted) * args.val_frac)))
    train_idx = acted[:-n_val]
    val_idx = acted[-n_val:]
    print(f"[probe] {len(train_idx)} train + {len(val_idx)} val acted snapshots", flush=True)

    print("[probe] running encoder forward over train ...", flush=True)
    train_cached = _precompute(stack, dataset, train_idx,
                               device=args.device, batch_size=args.batch_size)
    print("[probe] running encoder forward over val ...", flush=True)
    val_cached = _precompute(stack, dataset, val_idx,
                             device=args.device, batch_size=args.batch_size)

    label_keys = [
        "inbound_slot0_h10",
        "inbound_slot1_h10",
        "inbound_slot2_h10",
        "inbound_slot3_h10",
        "inbound_total_h10",
        "n_friendly_R",
        "n_enemy_R",
        "nearest_enemy_dist",
    ]
    feat_keys = ["ctx", "entity"]

    results: dict[str, dict[str, dict]] = {}
    print()
    print(f"  {'label':<24s}  {'feat':<7s}  "
          f"{'mae':>9s}  {'baseline':>9s}  {'rel':>7s}  {'r2':>7s}")
    print("  " + "-" * 75)
    for label in label_keys:
        results[label] = {}
        for feat in feat_keys:
            x_tr, y_tr = _flatten(train_cached, feat, label)
            x_va, y_va = _flatten(val_cached, feat, label)
            r = _fit_linear_probe(x_tr, y_tr, x_va, y_va)
            results[label][feat] = r
            print(f"  {label:<24s}  {feat:<7s}  "
                  f"{r['mae']:>9.4f}  {r['mae_baseline']:>9.4f}  "
                  f"{r['rel_mae']:>7.3f}  {r['r2']:>7.3f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\n[probe] wrote results → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
