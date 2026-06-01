"""Stage A: supervised pretraining of the PlayerConsolidator.

Freezes everything below the Consolidator (L0 specialists, L1 entity
encoder, L2 cross-entity attention, plus L3/L4/PairHead which the
Consolidator does not touch) and trains:

    self.entity_model.consolidator
    + CurrentStateHead (lives only in this training run, attached fresh)

against per-player current-state labels derived from the same snapshot
that the Consolidator sees. This is the v0 head bundle — just
``CurrentStateHead``. Future heads (FutureStateHead, OutcomeHead,
MatchupHead, AffordanceHead) plug into this loop without restructuring
once their labels are available.

Reuses ``CachedPairDataset`` from the existing pair cache so we don't
need a new dataset for Stage A. The pair labels in the cache are
ignored here — we read only the planet/fleet feature tensors and
derive per-player labels at batch time via
``consolidator_heads.compute_current_state_labels``.

Run:
    python -m agents.transformer_v2.pretrain.train_consolidator \\
        --init-from-entity-ckpt path/to/entity_encoder_best.pt \\
        --pair-cache-path path/to/pair_cache.pt \\
        --out-dir data/runs/consolidator/<tag>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from .consolidator_heads import (
    CurrentStateHead,
    compute_current_state_labels,
    current_state_loss,
)
from .entity_encoder import (
    DATASETS_ROOT,
    EntityPretrainModel,
    _PLANET_OWNER_NEUTRAL_IDX,
    _PLANET_OWNER_START_IDX,
    _build_entity_self_tokens,
    _load_encoders,
    build_pair_type_ids,
)
from ..paths import FLEET_RUNS_DIR, PLANET_RUNS_DIR, RUNS_ROOT

# Default L0 specialist run dirs. Mirrors the constants ``runner.py``
# uses, but imported from ``paths.py`` so this module can be loaded
# inside the Colab bundle (which does not ship ``agents/heuristic``,
# which ``runner.py`` transitively imports).
DEFAULT_FLEET_RUN_DIR = FLEET_RUNS_DIR / "specialist_fleet_d256_40k_lr1e4_120ep"
DEFAULT_PLANET_RUN_DIR = (
    PLANET_RUNS_DIR / "specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep"
)
DEFAULT_COMET_RUN_DIR = (
    RUNS_ROOT / "comet" / "fullpath_scalar_multitask_d256_40k_lr1e4_120ep"
)


def _freeze_all_but_consolidator(model: EntityPretrainModel) -> dict[str, int]:
    """Freeze every entity-model parameter except the Consolidator's.

    Returns a small dict of param counts for logging: ``{group:
    n_params}``. The Consolidator is the only group with trainable
    parameters; everything else is frozen.
    """
    trainable = 0
    frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("consolidator."):
            p.requires_grad_(True)
            trainable += p.numel()
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    return {"consolidator_trainable": trainable, "rest_frozen": frozen}


def _forward_through_l2(
    *,
    model: EntityPretrainModel,
    planet_enc, fleet_enc, comet_enc,
    batch: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    """Run L0 (frozen) + L1+L2 + PlayerConsolidator. Returns the dict
    from ``forward_with_context`` including ``player_state``.

    L0 forward is wrapped in ``torch.no_grad`` because the L0
    specialists are also frozen — no need to build a graph for them.
    """
    with torch.no_grad():
        planet_tok = planet_enc(batch["planet_features"])
        comet_tok = comet_enc(batch["comet_features"])
        fleet_tok = fleet_enc(batch["fleet_features"])
        entity_self = _build_entity_self_tokens(
            planet_tok, comet_tok, batch["is_comet"],
        )

    routing = {
        "fleet_target_idx": batch["fleet_target_idx"],
        "fleet_source_idx": batch["fleet_source_idx"],
        "fleet_owner_slot": batch["fleet_owner_slot"],
        "fleet_ships_log": batch["fleet_ships_log"],
        "fleet_eta_norm": batch["fleet_eta_norm"],
        "fleet_mask": batch["fleet_mask"],
    }
    # PlayerConsolidator needs the 5-dim learner-relative owner one-hot.
    owner_oh = batch["planet_features"][
        ..., _PLANET_OWNER_START_IDX:_PLANET_OWNER_NEUTRAL_IDX + 1
    ]
    return model.forward_with_context(
        entity_self, fleet_tok, routing, batch["planet_mask"],
        is_comet=batch["is_comet"],
        pair_type_ids=build_pair_type_ids(
            batch["planet_features"], batch["planet_mask"],
        ),
        planet_owner_oh=owner_oh,
    )


def _epoch_pass(
    *,
    model: EntityPretrainModel,
    head: CurrentStateHead,
    planet_enc, fleet_enc, comet_enc,
    loader: DataLoader,
    device: str,
    train: bool,
    opt: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one epoch over ``loader`` and return mean losses."""
    if train:
        model.train()
        head.train()
        assert opt is not None
    else:
        model.eval()
        head.eval()

    running: dict[str, float] = defaultdict(float)
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = _forward_through_l2(
                model=model,
                planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
                batch=batch, device=device,
            )
            preds = head(out["player_state"])
            labels = compute_current_state_labels(
                planet_features=batch["planet_features"],
                planet_mask=batch["planet_mask"],
                fleet_features=batch["fleet_features"],
                fleet_mask=batch["fleet_mask"],
            )
            losses = current_state_loss(preds, labels)
            if train:
                opt.zero_grad()
                losses["loss"].backward()
                opt.step()
            for k, v in losses.items():
                running[k] += float(v.detach() if torch.is_tensor(v) else v)
            n_batches += 1

    return {k: v / max(1, n_batches) for k, v in running.items()}


def train_consolidator(
    *,
    out_dir: Path,
    fleet_run_dir: Path,
    planet_run_dir: Path,
    comet_run_dir: Path,
    init_from_entity_ckpt: Path,
    pair_cache_path: Path,
    d_model: int = 256,
    cross_n_layers: int = 2,
    conditioner_n_layers: int = 3,
    head_n_layers: int = 3,
    state_head_trunk_n_layers: int = 2,
    state_head_n_layers: int = 2,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    eval_every: int = 1,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 0,
) -> Path:
    """Stage A training entry point. Returns path to best ckpt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # ---- Load pair cache ----
    from scripts.build_pair_dataset_orbital_occle import CachedPairDataset

    if not pair_cache_path.exists():
        raise FileNotFoundError(f"pair cache not found: {pair_cache_path}")
    print(f"[stage-A] loading pair cache from {pair_cache_path}", flush=True)
    full_ds = CachedPairDataset(pair_cache_path)
    print(
        f"  cache: player={full_ds.config.get('player')!r}  "
        f"history_offsets={full_ds.config.get('history_offsets')}  "
        f"max_planets={full_ds.config.get('max_planets')}  "
        f"snapshots={len(full_ds):,}", flush=True,
    )
    n_steps = (
        len(full_ds.history_offsets) if full_ds.history_offsets else 1
    )

    # Split by episode so a snapshot from train cannot leak into val/test.
    episode_ids = sorted({ep for ep, _t in full_ds.keys})
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(episode_ids), generator=rng).tolist()
    episode_ids = [episode_ids[i] for i in perm]
    n_test = int(len(episode_ids) * test_frac)
    n_val = int(len(episode_ids) * val_frac)
    test_eps = set(episode_ids[:n_test])
    val_eps = set(episode_ids[n_test:n_test + n_val])
    train_eps = set(episode_ids[n_test + n_val:])

    # Stage A doesn't filter to acted rows — every snapshot has valid
    # per-player current-state labels regardless of whether anyone acted.
    split_idxs: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, (ep, _t) in enumerate(full_ds.keys):
        if ep in train_eps:
            split_idxs["train"].append(i)
        elif ep in val_eps:
            split_idxs["val"].append(i)
        elif ep in test_eps:
            split_idxs["test"].append(i)
    splits = {name: Subset(full_ds, idxs) for name, idxs in split_idxs.items()}
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(splits[split]):,} snapshots", flush=True)

    train_loader = DataLoader(
        splits["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        splits["val"], batch_size=batch_size, num_workers=num_workers,
    )
    test_loader = DataLoader(
        splits["test"], batch_size=batch_size, num_workers=num_workers,
    )

    # ---- Frozen L0 specialists ----
    fleet_enc, planet_enc, comet_enc = _load_encoders(
        fleet_run_dir, planet_run_dir, comet_run_dir,
        device=device, expected_d_model=d_model,
    )

    # ---- Entity model + warm-start ----
    model = EntityPretrainModel(
        d_model=d_model, n_steps=n_steps,
        cross_n_layers=cross_n_layers,
        conditioner_n_layers=conditioner_n_layers,
        head_n_layers=head_n_layers,
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  entity-model params: {n_total:,}", flush=True)

    if not init_from_entity_ckpt.exists():
        raise FileNotFoundError(
            f"--init-from-entity-ckpt: {init_from_entity_ckpt} not found"
        )
    print(f"[stage-A] warm-start from {init_from_entity_ckpt}", flush=True)
    ckpt = torch.load(init_from_entity_ckpt, map_location=device, weights_only=False)
    load = model.load_state_dict(ckpt["model"], strict=False)
    # Expected: every Consolidator key is missing (it's a new module not
    # in the source ckpt). Anything else missing is a real warning.
    missing_consolidator = [k for k in load.missing_keys
                             if k.startswith("consolidator.")]
    missing_other = [k for k in load.missing_keys
                      if not k.startswith("consolidator.")]
    print(
        f"  warm-start: missing={len(load.missing_keys)} "
        f"(consolidator={len(missing_consolidator)}, "
        f"other={len(missing_other)}), "
        f"unexpected={len(load.unexpected_keys)}",
        flush=True,
    )
    if missing_other:
        print(
            f"  WARNING: {len(missing_other)} non-consolidator keys at "
            f"random init: {missing_other[:5]}", flush=True,
        )

    counts = _freeze_all_but_consolidator(model)
    print(
        f"[stage-A] consolidator trainable: "
        f"{counts['consolidator_trainable']:,}  "
        f"rest frozen: {counts['rest_frozen']:,}", flush=True,
    )

    # ---- Head ----
    head = CurrentStateHead(
        d_model=d_model,
        trunk_n_layers=state_head_trunk_n_layers,
        head_n_layers=state_head_n_layers,
    ).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(
        f"  CurrentStateHead params: {n_head:,} "
        f"(trunk_n_layers={state_head_trunk_n_layers}, "
        f"head_n_layers={state_head_n_layers})", flush=True,
    )

    # ---- Optimizer ----
    params = (
        [p for p in model.consolidator.parameters() if p.requires_grad]
        + list(head.parameters())
    )
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    config = {
        "d_model": d_model, "n_steps": n_steps,
        "cross_n_layers": cross_n_layers,
        "conditioner_n_layers": conditioner_n_layers,
        "head_n_layers": head_n_layers,
        "state_head_trunk_n_layers": state_head_trunk_n_layers,
        "state_head_n_layers": state_head_n_layers,
        "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": epochs,
        "pair_cache_path": str(pair_cache_path),
        "init_from_entity_ckpt": str(init_from_entity_ckpt),
        "fleet_run_dir": str(fleet_run_dir),
        "planet_run_dir": str(planet_run_dir),
        "comet_run_dir": str(comet_run_dir),
        "stage": "A_current_state_only",
    }

    log: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "consolidator_best.pt"
    last_path = out_dir / "consolidator_last.pt"
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch_pass(
            model=model, head=head,
            planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
            loader=train_loader, device=device, train=True, opt=opt,
        )
        elapsed = round(time.time() - t0, 2)
        entry: dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "elapsed_s": elapsed,
        }
        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = _epoch_pass(
                model=model, head=head,
                planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
                loader=val_loader, device=device, train=False,
            )
            entry["val"] = val_metrics
            print(
                f"[ep {epoch:>2}/{epochs}]  "
                f"train_loss={train_metrics['loss']:.4f} "
                f"(cont={train_metrics['cont_loss']:.4f} "
                f"alive={train_metrics['alive_loss']:.4f} "
                f"rank={train_metrics['rank_loss']:.4f})  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"({elapsed}s)", flush=True,
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save({
                    "consolidator": model.consolidator.state_dict(),
                    "current_state_head": head.state_dict(),
                    "epoch": epoch,
                    "config": config,
                }, best_path)
        else:
            print(
                f"[ep {epoch:>2}/{epochs}]  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"(no val; {elapsed}s)", flush=True,
            )

        log.append(entry)
        torch.save({
            "consolidator": model.consolidator.state_dict(),
            "current_state_head": head.state_dict(),
            "epoch": epoch, "config": config,
        }, last_path)
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    # ---- Test with best ----
    print("\n[stage-A] evaluating best on test ...", flush=True)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.consolidator.load_state_dict(ckpt["consolidator"])
    head.load_state_dict(ckpt["current_state_head"])
    test_metrics = _epoch_pass(
        model=model, head=head,
        planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
        loader=test_loader, device=device, train=False,
    )
    print(
        f"  test_loss={test_metrics['loss']:.4f} "
        f"(cont={test_metrics['cont_loss']:.4f} "
        f"alive={test_metrics['alive_loss']:.4f} "
        f"rank={test_metrics['rank_loss']:.4f})", flush=True,
    )
    (out_dir / "test_summary.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"\n[stage-A] best ckpt: {best_path}", flush=True)
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--fleet-run-dir", type=Path, default=DEFAULT_FLEET_RUN_DIR,
    )
    parser.add_argument(
        "--planet-run-dir", type=Path, default=DEFAULT_PLANET_RUN_DIR,
    )
    parser.add_argument(
        "--comet-run-dir", type=Path, default=DEFAULT_COMET_RUN_DIR,
    )
    parser.add_argument(
        "--init-from-entity-ckpt", type=Path, required=True,
        help="Pre-trained entity_encoder_best.pt to warm-start L1-L4 + "
             "PairHead from. Consolidator stays at random init regardless.",
    )
    parser.add_argument(
        "--pair-cache-path", type=Path, required=True,
        help="Pair cache .pt to source snapshots from. Pair labels are "
             "ignored; only feature/mask tensors are used.",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--cross-n-layers", type=int, default=2)
    parser.add_argument("--conditioner-n-layers", type=int, default=3)
    parser.add_argument("--head-n-layers", type=int, default=3)
    parser.add_argument(
        "--state-head-trunk-n-layers", type=int, default=2,
        help="CurrentStateHead shared-trunk depth (Linears). Default 2.",
    )
    parser.add_argument(
        "--state-head-n-layers", type=int, default=2,
        help="Per-task MLP depth inside CurrentStateHead "
             "(continuous/alive/rank). Floor 2.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_consolidator(
        out_dir=args.out_dir,
        fleet_run_dir=args.fleet_run_dir,
        planet_run_dir=args.planet_run_dir,
        comet_run_dir=args.comet_run_dir,
        init_from_entity_ckpt=args.init_from_entity_ckpt,
        pair_cache_path=args.pair_cache_path,
        d_model=args.d_model,
        cross_n_layers=args.cross_n_layers,
        conditioner_n_layers=args.conditioner_n_layers,
        head_n_layers=args.head_n_layers,
        state_head_trunk_n_layers=args.state_head_trunk_n_layers,
        state_head_n_layers=args.state_head_n_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
