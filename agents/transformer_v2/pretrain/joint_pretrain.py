"""Joint action + value pretraining over ONE shared L2 backbone.

The action head (PairHead, single-target contract) and the explicit value heads
(win / rank / score / current-state / momentum / near-future is_ahead) branch
from the SAME ``EntityPretrainModel`` L2 — the action branch via L3/L4, the
value branch via the PlayerConsolidator — so a combined loss co-adapts the one
L2 (and L3/L4/consolidator/heads) while L0+L1 stay frozen ("L2~ unfreeze").

Two caches feed the two losses (the action labels live in the pair cache, the
value labels in the cross-entity cache); each batch is L0-encoded the same way
and run through ``forward_with_context``, which emits both head groups. The
value loss uses a large near-future decay so the value/critic representation
focuses on near-future return (see ``cross_entity.VALUE_FWD_GAMMA``).

Data note: the pair cache is T=6 and the cross-entity cache is T=10. A shared L2
built with ``n_steps=10`` consumes both via ``step_embed[-T:]``, but for clean
temporal semantics rebuild the two caches with matched ``history_offsets``
(or build one unified cache carrying both label sets). This is the documented
prerequisite for a production joint run; the loss/step/loop below are
cache-agnostic.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .entity_encoder import (
    ENTITY_N_OWNER_CLASSES,
    EntityPretrainModel,
    _PLANET_OWNER_START_IDX,
    _build_entity_self_tokens,
    _load_encoders,
    build_pair_type_ids,
    compute_multi_loss,
)

_FLEET_ROUTING_KEYS = (
    "fleet_target_idx", "fleet_source_idx", "fleet_owner_slot",
    "fleet_ships_log", "fleet_eta_norm", "fleet_mask",
)


def _forward_context_from_batch(
    model: EntityPretrainModel,
    fleet_enc: nn.Module,
    planet_enc: nn.Module,
    comet_enc: nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """L0-encode a batch (pair OR cross-entity) and run ``forward_with_context``.

    Both caches expose the same L0 input keys, so one builder serves both. The
    L0 specialists are frozen, so they run under ``no_grad`` for speed; gradient
    still flows from L2 up (the entity_self tokens are leaves into L1/L2).
    """
    with torch.no_grad():
        planet_tok = planet_enc(batch["planet_features"])
        comet_tok = comet_enc(batch["comet_features"])
        fleet_tok = fleet_enc(batch["fleet_features"])
    entity_self = _build_entity_self_tokens(planet_tok, comet_tok, batch["is_comet"])
    routing = {k: batch[k] for k in _FLEET_ROUTING_KEYS}
    # Learner-relative owner one-hot slice for the consolidator's owner cue.
    owner_oh = batch["planet_features"][
        ..., _PLANET_OWNER_START_IDX:_PLANET_OWNER_START_IDX + ENTITY_N_OWNER_CLASSES
    ]
    return model.forward_with_context(
        entity_self, fleet_tok, routing, batch["planet_mask"],
        is_comet=batch["is_comet"],
        pair_type_ids=build_pair_type_ids(
            batch["planet_features"], batch["planet_mask"],
        ),
        planet_owner_oh=owner_oh,
    )


def joint_train_step(
    model: EntityPretrainModel,
    encoders: tuple[nn.Module, nn.Module, nn.Module],
    action_batch: dict[str, torch.Tensor] | None,
    value_batch: dict[str, torch.Tensor] | None,
    *,
    launch_weight: float = 1.0,
    fwd_gamma: float = 0.9,
    value_coef: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined action + value loss over the shared L2.

    ``total = action_loss + value_coef * value_loss`` where the action loss is
    the single-target PairHead objective (``compute_multi_loss(single_target=
    True)``) on the pair-cache batch, and the value loss is
    ``compute_loss_value_pretrain`` (with near-future decay ``fwd_gamma``) on the
    cross-entity batch. Either batch may be ``None`` (e.g. when one loader is
    exhausted), in which case only the present side contributes. Per-head losses
    are returned namespaced ``act/<head>`` and ``val/<head>``.
    """
    from .cross_entity import compute_loss_value_pretrain  # local: avoid import cycle

    fleet_enc, planet_enc, comet_enc = encoders
    total: torch.Tensor | None = None
    per_head: dict[str, float] = {}

    if action_batch is not None:
        act_out = _forward_context_from_batch(
            model, fleet_enc, planet_enc, comet_enc, action_batch,
        )
        act_loss, act_terms = compute_multi_loss(
            act_out, action_batch,
            single_target=True,
            single_target_launch_weight=launch_weight,
        )
        total = act_loss if total is None else total + act_loss
        for k, v in act_terms.items():
            per_head[f"act/{k}"] = v

    if value_batch is not None:
        val_out = _forward_context_from_batch(
            model, fleet_enc, planet_enc, comet_enc, value_batch,
        )
        val_loss, val_terms = compute_loss_value_pretrain(
            val_out, value_batch, fwd_gamma=fwd_gamma,
        )
        scaled = value_coef * val_loss
        total = scaled if total is None else total + scaled
        for k, v in val_terms.items():
            per_head[f"val/{k}"] = v

    if total is None:
        raise ValueError("joint_train_step needs at least one of action/value batch")
    return total, per_head


def format_joint_per_head(
    per_head: dict[str, float],
    *,
    title: str = "per-head",
) -> str:
    """Verbose grouped per-head table: action heads then value heads."""
    act = {k[4:]: v for k, v in per_head.items() if k.startswith("act/")}
    val = {k[4:]: v for k, v in per_head.items() if k.startswith("val/")}
    other = {k: v for k, v in per_head.items() if not k.startswith(("act/", "val/"))}
    lines = [f"    [{title}]"]

    def _block(name: str, d: dict[str, float]) -> None:
        if not d:
            return
        lines.append(f"      {name}:")
        for k in sorted(d):
            v = d[k]
            cell = "  nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
            lines.append(f"        {k:<28s} {cell}")

    _block("action", act)
    _block("value", val)
    _block("other", other)
    return "\n".join(lines)


def _cycle_zip(action_loader, value_loader):
    """Yield (action_batch, value_batch) for max(len) steps, cycling the shorter
    loader so every step trains both branches (the longer loader sets the epoch
    length; the shorter repeats)."""
    la = len(action_loader)
    lv = len(value_loader)
    n = max(la, lv)
    a_it = iter(action_loader) if la == n else itertools.cycle(action_loader)
    v_it = iter(value_loader) if lv == n else itertools.cycle(value_loader)
    for _ in range(n):
        yield next(a_it), next(v_it)


def train_joint(
    *,
    out_dir: Path,
    fleet_run_dir: Path,
    planet_run_dir: Path,
    comet_run_dir: Path,
    pair_cache_path: Path,
    cross_cache_path: Path,
    d_model: int = 256,
    n_steps: int = 10,
    batch_size: int = 16,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    launch_weight: float = 1.0,
    fwd_gamma: float = 0.9,
    value_coef: float = 1.0,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 0,
    progress_every: int = 50,
) -> Path:
    """Joint action+value pretrain loop with L2+ unfreeze + verbose per-head logs.

    Builds the combined ``EntityPretrainModel`` (consolidator + value heads),
    freezes L0+L1, unfreezes L2 and up, and trains the action head on the pair
    cache and the value heads on the cross-entity cache each step, sharing L2.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    from scripts.build_pair_dataset_orbital_occle import CachedPairDataset
    from .cross_entity import CachedCrossEntitySnapshotDataset

    print(f"[joint] device={device} d_model={d_model} n_steps={n_steps}", flush=True)
    print(f"[joint] loading L0 specialists ...", flush=True)
    fleet_enc, planet_enc, comet_enc = _load_encoders(
        fleet_run_dir, planet_run_dir, comet_run_dir,
        device=device, expected_d_model=d_model,
    )
    for enc in (fleet_enc, planet_enc, comet_enc):
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()

    model = EntityPretrainModel(
        d_model=d_model, n_steps=n_steps,
        with_consolidator=True, with_value_heads=True,
    ).to(device)
    report = model.freeze_below_l2()
    print("[joint] freeze_below_l2 (L0+L1 frozen, L2+ trainable):", flush=True)
    for k, v in report.items():
        print(f"    {k:<28s} {v:,}", flush=True)

    print(f"[joint] action cache: {pair_cache_path}", flush=True)
    action_ds = CachedPairDataset(pair_cache_path)
    print(f"[joint] value  cache: {cross_cache_path}", flush=True)
    value_ds = CachedCrossEntitySnapshotDataset(cross_cache_path)
    action_loader = DataLoader(
        action_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    value_loader = DataLoader(
        value_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    print(
        f"[joint] action batches/epoch={len(action_loader)}  "
        f"value batches/epoch={len(value_loader)}  "
        f"steps/epoch={max(len(action_loader), len(value_loader))}",
        flush=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    n_trainable = sum(p.numel() for p in params)
    print(f"[joint] trainable params: {n_trainable:,}", flush=True)

    best_path = out_dir / "joint_best.pt"
    last_path = out_dir / "joint_last.pt"
    log: list[dict[str, Any]] = []
    config = {
        "d_model": d_model, "n_steps": n_steps, "batch_size": batch_size,
        "epochs": epochs, "lr": lr, "launch_weight": launch_weight,
        "fwd_gamma": fwd_gamma, "value_coef": value_coef,
        "action_contract": "single_target_per_source_v1",
        "pair_cache_path": str(pair_cache_path),
        "cross_cache_path": str(cross_cache_path),
    }

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_total = 0.0
        running: dict[str, float] = {}
        counts: dict[str, int] = {}
        n_steps_done = 0
        for action_batch, value_batch in _cycle_zip(action_loader, value_loader):
            action_batch = {k: v.to(device) for k, v in action_batch.items()}
            value_batch = {k: v.to(device) for k, v in value_batch.items()}
            total_loss, per_head = joint_train_step(
                model, (fleet_enc, planet_enc, comet_enc),
                action_batch, value_batch,
                launch_weight=launch_weight, fwd_gamma=fwd_gamma,
                value_coef=value_coef,
            )
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            running_total += float(total_loss.detach())
            for k, v in per_head.items():
                if isinstance(v, float) and math.isnan(v):
                    continue
                running[k] = running.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
            n_steps_done += 1
            if progress_every and n_steps_done % progress_every == 0:
                print(
                    f"    [ep {epoch} step {n_steps_done}] "
                    f"running_total={running_total / n_steps_done:.4f} "
                    f"({round(time.time() - t0, 1)}s)",
                    flush=True,
                )

        mean_total = running_total / max(1, n_steps_done)
        mean_per_head = {k: running[k] / max(1, counts[k]) for k in running}
        elapsed = round(time.time() - t0, 1)
        print(
            f"[ep {epoch:>2}/{epochs}] joint_total={mean_total:.4f} "
            f"({n_steps_done} steps, {elapsed}s)",
            flush=True,
        )
        print(format_joint_per_head(mean_per_head, title=f"ep {epoch} train"), flush=True)

        entry = {"epoch": epoch, "joint_total": mean_total,
                 "per_head": mean_per_head, "elapsed_s": elapsed}
        log.append(entry)
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, last_path)
        if epoch == 1 or mean_total <= min(e["joint_total"] for e in log):
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, best_path)
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    print(f"[joint] done. outputs in {out_dir}", flush=True)
    return best_path


def main() -> None:
    import argparse

    from .cross_entity import VALUE_FWD_GAMMA

    p = argparse.ArgumentParser(
        description="Joint action+value pretrain over one shared L2 "
        "(action head + value heads, L2~ unfreeze, near-future-decayed reward, "
        "verbose per-head logging).",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fleet-run-dir", type=Path, required=True)
    p.add_argument("--planet-run-dir", type=Path, required=True)
    p.add_argument("--comet-run-dir", type=Path, required=True)
    p.add_argument("--pair-cache-path", type=Path, required=True,
                   help="action labels (pair cache .pt)")
    p.add_argument("--cross-cache-path", type=Path, required=True,
                   help="value labels (cross-entity cache .pt)")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=10,
                   help="L2 history length; 10 consumes both T=6 and T=10 caches")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--launch-weight", type=float, default=1.0,
                   help="up-weight launching source rows in the action CE "
                        "(single-target NOOP-imbalance control)")
    p.add_argument("--fwd-gamma", type=float, default=VALUE_FWD_GAMMA,
                   help="per-turn discount on the future is_ahead reward; "
                        "large decay (small gamma) -> near-future focus")
    p.add_argument("--value-coef", type=float, default=1.0,
                   help="weight on the value loss relative to the action loss")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    args = p.parse_args()

    train_joint(
        out_dir=args.out_dir,
        fleet_run_dir=args.fleet_run_dir,
        planet_run_dir=args.planet_run_dir,
        comet_run_dir=args.comet_run_dir,
        pair_cache_path=args.pair_cache_path,
        cross_cache_path=args.cross_cache_path,
        d_model=args.d_model,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        launch_weight=args.launch_weight,
        fwd_gamma=args.fwd_gamma,
        value_coef=args.value_coef,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
