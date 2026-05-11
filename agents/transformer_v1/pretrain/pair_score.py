"""Encoder-freeze + single PairScoreHead smoke test.

Goal of this stage: answer one question.

    Can the current frozen encoder representation support direct expert
    ``(source, target)`` pair prediction from replay?

If the answer is "yes" (val pair top-1 clearly beats random-valid), we
layer NOOP / frac / value / PPO back on top in subsequent stages. If
"no" (the head can't even overfit a tiny subset), we know to fix the
encoder or the labels/masking before re-trying any policy work.

This file is intentionally minimal:

  * one trainable module — :class:`PairScoreHead` (a 2-layer MLP).
  * encoders frozen via ``requires_grad_(False)`` and ``.eval()``.
  * one loss — joint cross-entropy on flattened ``(P×P)`` pair logits.
  * one expert — ``--filter winner`` keeps only rows where the CSV's
    learner-slot matches the replay's ``winner_seat`` (proxy for "one
    strong expert" in the absence of per-replay agent metadata).
  * one dataset class reused — :class:`ActionSnapshotDataset` already
    emits ``source_planet_idx`` / ``target_planet_idx`` /
    ``src_valid`` / ``tgt_valid`` per snapshot.

Run from the repo root:

    python -m agents.transformer_v1.pretrain.pair_score \\
        --encoder-ckpt data/runs/action/<run>/action_best.pt \\
        --filter winner --max-rows 50 --overfit \\
        --epochs 200 --device cpu \\
        --out-dir data/runs/pair_score/$(date +%Y%m%d-%H%M%S)

For the small-real split (Experiment 2):

    python -m agents.transformer_v1.pretrain.pair_score \\
        --encoder-ckpt data/runs/action/<run>/action_best.pt \\
        --filter winner --max-rows 5000 \\
        --batch-size 64 --epochs 10 --lr 1e-3 --device cuda \\
        --out-dir data/runs/pair_score/$(date +%Y%m%d-%H%M%S)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..aggregator import CrossEntityAttention
from ..encoder.entity_encoder import PlanetEntityEncoder
from ..encoder.fleet_encoder import FleetEncoder
from ..encoder.planet_encoder import PlanetEncoder
from ..paths import (
    ACTION_DATASET_DIR,
    CROSS_ENTITY_DATASET_DIR,
    ENTITY_DATASET_DIR,
    FLEET_DATASET_DIR,
    PLANET_DATASET_DIR,
)
from .cross_entity import _entity_tokens_per_step
from .expert_action import ActionSnapshotDataset


# ---------- Model ----------
class PairScoreHead(nn.Module):
    """One MLP scoring every ``(source_i, target_j)`` pair.

    Pair feature per (i, j):
        h_ij = [ glob ‖ ctx_i ‖ ctx_j ‖ ctx_i ⊙ ctx_j ]   (4·d)

    Score:
        s_ij = MLP(h_ij)                                   (1)

    Output ``pair_logits`` has shape ``(B, P, P)``; the ``[i, j]`` cell
    is the pre-softmax score for "expert launches from i, aiming at j".
    Invalid pairs (per ``src_valid × tgt_valid``) are masked to ``-inf``.
    """

    def __init__(self, d_model: int = 64, hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)

    def forward(
        self,
        glob: torch.Tensor,                          # (B, d)
        ctx: torch.Tensor,                           # (B, P, d)
        src_valid: torch.Tensor | None = None,        # (B, P) bool
        tgt_valid: torch.Tensor | None = None,        # (B, P) bool
    ) -> torch.Tensor:                                # (B, P, P)
        B, P, d = ctx.shape
        if d != self.d_model:
            raise ValueError(
                f"ctx d={d} but head built for d_model={self.d_model}"
            )
        glob_b = glob.view(B, 1, 1, d).expand(B, P, P, d)
        src = ctx.unsqueeze(2).expand(B, P, P, d)
        tgt = ctx.unsqueeze(1).expand(B, P, P, d)
        had = src * tgt
        feat = torch.cat([glob_b, src, tgt, had], dim=-1)        # (B,P,P,4d)
        scores = self.mlp(feat).squeeze(-1)                       # (B,P,P)
        if src_valid is not None and tgt_valid is not None:
            pair_valid = src_valid.unsqueeze(2) & tgt_valid.unsqueeze(1)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~pair_valid, neg_inf)
        return scores


class PairScoreStack(nn.Module):
    """Frozen encoders + trainable :class:`PairScoreHead`.

    Forward expects the same batch dict as :class:`ActionSnapshotDataset`
    emits — `_entity_tokens_per_step` handles the (B,T,P,...) history
    layout transparently.
    """

    def __init__(
        self,
        *,
        fleet_encoder: FleetEncoder,
        planet_encoder: PlanetEncoder,
        entity_encoder: PlanetEntityEncoder,
        cross: CrossEntityAttention,
        pair_score_head: PairScoreHead,
    ):
        super().__init__()
        self.fleet_encoder = fleet_encoder
        self.planet_encoder = planet_encoder
        self.entity_encoder = entity_encoder
        self.cross = cross
        self.pair_score_head = pair_score_head

    # Encoder modules accessible by short name. Keep this as the
    # authoritative ordering — CLI arg parsing + checkpoint save/load
    # walk the same names.
    ENCODER_MODULES: tuple[str, ...] = (
        "fleet_encoder",
        "planet_encoder",
        "entity_encoder",
        "cross",
    )

    def freeze_encoders(self) -> None:
        """Backwards-compat alias: freeze every encoder."""
        self.set_freeze_state(unfrozen=())

    def set_freeze_state(self, unfrozen: tuple[str, ...] | list[str] | set[str]) -> None:
        """Set each encoder's train/grad mode based on whether its name
        appears in ``unfrozen``. The head is always trainable.

        ``unfrozen`` may contain ``"cross"``, ``"entity"`` (= entity_encoder),
        ``"planet"`` (= planet_encoder), or ``"fleet"`` (= fleet_encoder).
        Anything else raises.
        """
        canon = self._canonicalize(unfrozen)
        for name in self.ENCODER_MODULES:
            module = getattr(self, name)
            if name in canon:
                module.train()
                for p in module.parameters():
                    p.requires_grad_(True)
            else:
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)

    @classmethod
    def _canonicalize(cls, names) -> set[str]:
        """Map user-facing aliases (e.g. ``entity`` → ``entity_encoder``)
        and validate against ``ENCODER_MODULES``.
        """
        aliases = {
            "fleet": "fleet_encoder",
            "planet": "planet_encoder",
            "entity": "entity_encoder",
            "cross": "cross",
        }
        out: set[str] = set()
        for n in names or ():
            n = n.strip()
            if not n:
                continue
            if n in cls.ENCODER_MODULES:
                out.add(n)
            elif n in aliases:
                out.add(aliases[n])
            elif n in ("head", "pair_score_head"):
                # Always trainable; silently accepted.
                continue
            else:
                raise ValueError(
                    f"unknown unfreeze target {n!r}. "
                    f"valid: {sorted(set(aliases) | set(cls.ENCODER_MODULES))}"
                )
        return out

    def trainable_module_state(self, unfrozen: set[str]) -> dict[str, dict]:
        """Collect state-dicts for the head + every unfrozen encoder.

        Used by the checkpoint writer so an unfreeze-cross run can be
        chained with another (e.g. unfreeze-cross-and-entity) by passing
        its ``pair_score_best.pt`` as the next run's ``--init-from``.
        """
        out: dict[str, dict] = {"pair_score_head": self.pair_score_head.state_dict()}
        for name in self.ENCODER_MODULES:
            if name in unfrozen:
                out[name] = getattr(self, name).state_dict()
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch,
            self.fleet_encoder,
            self.planet_encoder,
            self.entity_encoder,
        )
        ctx, glob = self.cross(entity_tokens, entity_mask)
        # Single-step input is shape (B, P, d); multi-step is (B, T, P, d).
        ctx_now = ctx[:, -1] if ctx.dim() == 4 else ctx
        if entity_mask.dim() == 3:
            mask_now = entity_mask[:, -1]
        else:
            mask_now = entity_mask

        P = ctx_now.shape[1]
        src_valid = batch.get("src_valid")
        tgt_valid = batch.get("tgt_valid")
        if src_valid is None:
            src_valid = mask_now
        if tgt_valid is None:
            tgt_valid = mask_now
        # Dataset masks may have been allocated wider than the encoder's
        # planet axis; clip so shapes match.
        src_valid = src_valid[..., :P].bool().clone()
        tgt_valid = tgt_valid[..., :P].bool().clone()
        mask_now = mask_now[..., :P].bool()

        # Older action CSV packs do not have the optional `_masks/*.npz`
        # side cache, which leaves src/tgt masks all-False. Fall back to
        # the current real-planet mask for those rows so pair CE remains
        # trainable, then force-include the supervised pair labels.
        if mask_now.shape == src_valid.shape:
            src_empty = ~src_valid.any(dim=-1)
            tgt_empty = ~tgt_valid.any(dim=-1)
            fallback = src_empty | tgt_empty
            src_valid[fallback] = mask_now[fallback]
            tgt_valid[fallback] = mask_now[fallback]

        src_idx = batch.get("source_planet_idx")
        tgt_idx = batch.get("target_planet_idx")
        if src_idx is not None:
            src_idx = src_idx.to(src_valid.device).long()
            rows = torch.nonzero((src_idx >= 0) & (src_idx < P), as_tuple=True)[0]
            if rows.numel() > 0:
                src_valid[rows, src_idx[rows]] = True
        if tgt_idx is not None:
            tgt_idx = tgt_idx.to(tgt_valid.device).long()
            rows = torch.nonzero((tgt_idx >= 0) & (tgt_idx < P), as_tuple=True)[0]
            if rows.numel() > 0:
                tgt_valid[rows, tgt_idx[rows]] = True

        pair_logits = self.pair_score_head(glob, ctx_now, src_valid, tgt_valid)
        return {
            "pair_logits": pair_logits,
            "_ctx_now": ctx_now,
            "_glob": glob,
        }


# ---------- Loss + metrics ----------
def compute_pair_score_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint pair CE on rows where the expert acted.

    Returns (loss, metrics) where metrics contains top-1/3/5 accuracies,
    induced source/target accuracy, and the count of valid rows.
    """
    pair_logits = preds["pair_logits"]                               # (B, P, P)
    src_idx = batch["source_planet_idx"].to(pair_logits.device)       # (B,) Long
    tgt_idx = batch["target_planet_idx"].to(pair_logits.device)       # (B,)
    B, P, _ = pair_logits.shape

    valid = (src_idx >= 0) & (tgt_idx >= 0) & (src_idx < P) & (tgt_idx < P)
    n_valid = int(valid.sum().item())
    metrics: dict[str, float] = {
        "top1": 0.0, "top3": 0.0, "top5": 0.0,
        "src_top1": 0.0, "tgt_top1": 0.0,
        "n_valid": float(n_valid),
    }
    if n_valid == 0:
        # Zero-grad sentinel — keeps the optimizer step a no-op without
        # branching the train loop.
        return pair_logits.sum() * 0.0, metrics

    pl = pair_logits[valid]                                          # (Nv, P, P)
    si = src_idx[valid]
    ti = tgt_idx[valid]
    flat = pl.reshape(pl.shape[0], P * P)                            # (Nv, P*P)
    y = si * P + ti                                                  # (Nv,)
    loss = F.cross_entropy(flat, y)

    with torch.no_grad():
        k = min(5, P * P)
        top_ids = flat.topk(k, dim=-1).indices                       # (Nv, k)
        match = top_ids == y.unsqueeze(-1)
        metrics["top1"] = float(match[:, 0].float().mean().item())
        metrics["top3"] = float(match[:, : min(3, k)].any(-1).float().mean().item())
        metrics["top5"] = float(match.any(-1).float().mean().item())
        pred = flat.argmax(-1)
        metrics["src_top1"] = float((pred // P == si).float().mean().item())
        metrics["tgt_top1"] = float((pred % P == ti).float().mean().item())
    return loss, metrics


@torch.no_grad()
def random_valid_baseline(
    batch: dict[str, torch.Tensor], device: torch.device,
) -> float:
    """Top-1 accuracy of picking a uniformly-random valid pair.

    Computed once per batch as ``1 / n_valid_pairs`` averaged over rows.
    """
    src_idx = batch["source_planet_idx"].to(device)
    tgt_idx = batch["target_planet_idx"].to(device)
    src_valid = batch["src_valid"].to(device).bool()
    tgt_valid = batch["tgt_valid"].to(device).bool()
    valid = (src_idx >= 0) & (tgt_idx >= 0)
    if not valid.any():
        return 0.0
    pair_valid = src_valid.unsqueeze(2) & tgt_valid.unsqueeze(1)     # (B,P,P)
    n_pairs = pair_valid[valid].reshape(int(valid.sum()), -1).sum(-1).clamp(min=1)
    return float((1.0 / n_pairs.float()).mean().item())


# ---------- Encoder-only ckpt loader ----------
def load_frozen_encoder_stack(
    ckpt_path: str | Path,
    *,
    d_model: int = 64,
    device: str = "cpu",
) -> tuple[FleetEncoder, PlanetEncoder, PlanetEntityEncoder, CrossEntityAttention]:
    """Build empty encoders and load only the encoder + cross weights
    from a previously-saved action checkpoint.

    The pre-deletion ``_save_checkpoint`` wrote keys
    ``cross``, ``fleet_encoder``, ``planet_encoder``, ``entity_encoder``,
    ``action_decoder``, ``global_decoder``. This loader takes the first
    four and ignores the rest.
    """
    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"unexpected ckpt format at {ckpt_path}: {type(ckpt)}")
    for k in ("fleet_encoder", "planet_encoder", "entity_encoder", "cross"):
        if k not in ckpt:
            raise KeyError(
                f"{ckpt_path} is missing '{k}' state-dict key — "
                "this loader expects an action-stage checkpoint."
            )

    fenc = FleetEncoder(d_model=d_model)
    fenc.load_state_dict(ckpt["fleet_encoder"])
    penc = PlanetEncoder(d_model=d_model)
    penc.load_state_dict(ckpt["planet_encoder"])
    eenc = PlanetEntityEncoder(d_model=d_model)
    eenc.load_state_dict(ckpt["entity_encoder"])
    cross = CrossEntityAttention(d_model=d_model)
    cross.load_state_dict(ckpt["cross"])

    for m in (fenc, penc, eenc, cross):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return fenc, penc, eenc, cross


# ---------- Dataset helpers ----------
def _csv_winner_slot_match(action_csv_path: Path) -> bool:
    """``True`` iff the CSV's learner-slot equals the replay's
    ``winner_seat`` for the first row.

    Filename convention: ``action_<replay>_<num_players>_<learner_slot>.csv``.
    Each CSV's rows share the same ``learner_slot``, so reading any row
    gives the answer.
    """
    try:
        learner_slot = int(action_csv_path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return False
    try:
        with action_csv_path.open() as fh:
            row = next(csv.DictReader(fh), None)
    except OSError:
        return False
    if row is None:
        return False
    try:
        winner_seat = int(row["winner_seat"])
    except (KeyError, ValueError):
        return False
    return winner_seat == learner_slot


def player_replay_stems(
    replay_dir: Path,
    player: str,
) -> set[str]:
    """Return the set of replay-stem strings (e.g. ``75365996_2_0``) under
    ``<replay_dir>/<player>/``.

    Layout is ``data/replays/<player>/<replay_id>_<num_players>_<seat>.json.gz``.
    The directory name is the player whose perspective the replay is from;
    the replay JSON's ``info.TeamNames[seat]`` matches that player.

    Action CSV filenames share the same stem with an ``action_`` prefix,
    so the returned set can be used to filter
    :func:`discover_action_csvs` outputs to one player.
    """
    pdir = Path(replay_dir) / player
    if not pdir.is_dir():
        raise FileNotFoundError(
            f"no replay directory for player={player!r} at {pdir}"
        )
    stems: set[str] = set()
    for p in pdir.iterdir():
        if not p.is_file():
            continue
        # ``75365996_2_0.json.gz`` → ``75365996_2_0``
        name = p.name
        if name.endswith(".json.gz"):
            stems.add(name[: -len(".json.gz")])
        elif name.endswith(".json"):
            stems.add(name[: -len(".json")])
    return stems


def discover_action_csvs(
    action_dir: Path,
    *,
    filter_mode: str,                  # "winner" or "all"
    player: str | None = None,
    replay_dir: Path | None = None,
) -> list[Path]:
    """Find action CSVs, optionally restricted to one player and/or
    winner-only perspectives.

    ``filter_mode``:
      * ``all``    — every CSV under ``action_dir``.
      * ``winner`` — keep only CSVs whose learner-slot equals
                     ``winner_seat`` (proxy for "perspective player won").

    ``player`` (optional): restrict to CSVs whose stem matches a replay
    under ``<replay_dir>/<player>/``. Composes with ``filter_mode`` —
    ``player='kovi', filter_mode='winner'`` keeps only kovi's actions in
    replays kovi won.
    """
    csvs = sorted(action_dir.glob("action_*.csv"))

    if player is not None:
        if replay_dir is None:
            raise ValueError("player= requires replay_dir=")
        keep_stems = player_replay_stems(replay_dir, player)
        csvs = [p for p in csvs if p.stem.removeprefix("action_") in keep_stems]

    if filter_mode == "all":
        return csvs
    if filter_mode == "winner":
        return [p for p in csvs if _csv_winner_slot_match(p)]
    raise ValueError(f"unknown filter_mode={filter_mode!r}")


def acted_only_indices(dataset: ActionSnapshotDataset) -> list[int]:
    """Return dataset indices where ``expert_acted > 0.5``."""
    out: list[int] = []
    for i in range(len(dataset)):
        snap = dataset.snapshots[i]
        if float(snap["expert_acted"].item()) > 0.5:
            out.append(i)
    return out


# ---------- Training loop ----------
def _train_one_epoch(
    stack: PairScoreStack,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    unfrozen: set[str],
) -> dict[str, float]:
    stack.train()
    stack.set_freeze_state(unfrozen)  # frozen modules stay in eval mode
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = stack(batch)
        loss, metrics = compute_pair_score_loss(preds, batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        sums["loss"] = sums.get("loss", 0.0) + float(loss.item())
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in sums.items()}


@torch.no_grad()
def _evaluate(
    stack: PairScoreStack,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    stack.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = stack(batch)
        loss, metrics = compute_pair_score_loss(preds, batch)
        sums["loss"] = sums.get("loss", 0.0) + float(loss.item())
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        sums["random_valid_top1"] = (
            sums.get("random_valid_top1", 0.0) + random_valid_baseline(batch, device)
        )
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in sums.items()}


def train_pair_score(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. Resolve unfrozen modules ----
    unfreeze_list: list[str] = []
    if getattr(args, "unfreeze", None):
        unfreeze_list = [s.strip() for s in args.unfreeze.split(",") if s.strip()]
    unfrozen = PairScoreStack._canonicalize(unfreeze_list)
    if unfrozen:
        print(f"[pair_score] unfreezing {sorted(unfrozen)}", flush=True)

    # ---- 1. Load + freeze encoders ----
    fenc, penc, eenc, cross = load_frozen_encoder_stack(
        args.encoder_ckpt, d_model=args.d_model, device=str(device),
    )

    # ---- 2. Build dataset ----
    action_csvs = discover_action_csvs(
        Path(args.action_dir),
        filter_mode=args.filter,
        player=args.player,
        replay_dir=Path(args.replay_dir) if args.player else None,
    )
    if not action_csvs:
        raise SystemExit(
            f"no action CSVs found under {args.action_dir} "
            f"with player={args.player!r} filter={args.filter!r}"
        )
    print(
        f"[pair_score] {len(action_csvs)} action CSVs "
        f"(player={args.player or 'any'}, filter={args.filter!r})",
        flush=True,
    )

    # Match planet/fleet/entity/cross_entity CSVs by stem.
    def _other(stems: list[str], dir_path: Path, prefix: str) -> list[Path]:
        return [dir_path / f"{prefix}{s}.csv" for s in stems
                if (dir_path / f"{prefix}{s}.csv").exists()]
    stems = [p.stem.removeprefix("action_") for p in action_csvs]
    planet_csvs = _other(stems, Path(args.planet_dir), "planet_")
    fleet_csvs = _other(stems, Path(args.fleet_dir), "fleet_")
    entity_csvs = _other(stems, Path(args.entity_dir), "entity_")
    cross_csvs = _other(stems, Path(args.cross_entity_dir), "cross_entity_")

    dataset = ActionSnapshotDataset(
        planet_csv_paths=planet_csvs,
        fleet_csv_paths=fleet_csvs,
        entity_csv_paths=entity_csvs,
        cross_entity_csv_paths=cross_csvs,
        action_csv_paths=action_csvs,
        max_planets=args.max_planets,
        max_fleets=args.max_fleets,
        n_history=args.n_history,
    )
    print(f"[pair_score] dataset size: {len(dataset)} snapshots", flush=True)

    # ---- 3. Filter to acted rows; cap at --max-rows ----
    acted_idx = acted_only_indices(dataset)
    print(f"[pair_score] acted rows: {len(acted_idx)}", flush=True)
    if args.max_rows is not None:
        acted_idx = acted_idx[: args.max_rows]
        print(f"[pair_score] capped to {len(acted_idx)} rows", flush=True)

    if args.overfit:
        train_idx = acted_idx
        val_idx = acted_idx        # train==val on purpose for tiny-overfit
    else:
        n_val = max(1, int(round(len(acted_idx) * args.val_frac)))
        train_idx = acted_idx[:-n_val]
        val_idx = acted_idx[-n_val:]
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    print(f"[pair_score] train={len(train_set)} val={len(val_set)}", flush=True)

    # ---- 4. Build pair head + stack ----
    head = PairScoreHead(d_model=args.d_model, hidden=args.hidden).to(device)
    stack = PairScoreStack(
        fleet_encoder=fenc,
        planet_encoder=penc,
        entity_encoder=eenc,
        cross=cross,
        pair_score_head=head,
    ).to(device)

    # ---- 4b. Optional resume from a prior pair_score_best.pt ----
    init_from = getattr(args, "init_from", None)
    if init_from:
        prior = torch.load(Path(init_from), map_location=device, weights_only=False)
        if "pair_score_head" not in prior:
            raise SystemExit(
                f"--init-from {init_from} has no 'pair_score_head' key — "
                "expected output of a prior pair_score run."
            )
        head.load_state_dict(prior["pair_score_head"])
        # If the prior run unfroze any encoders, prefer those weights —
        # otherwise the encoder state from --encoder-ckpt is kept.
        for name in PairScoreStack.ENCODER_MODULES:
            if name in prior:
                getattr(stack, name).load_state_dict(prior[name])
                print(f"[pair_score] init-from: loaded {name} state", flush=True)
        print(f"[pair_score] init-from: loaded pair_score_head from "
              f"{init_from} (epoch={prior.get('epoch')})", flush=True)

    stack.set_freeze_state(unfrozen)

    # AdamW over every parameter that's actually trainable now (head +
    # any unfrozen encoders). filter is needed because frozen params
    # have requires_grad=False.
    trainable_params = [p for p in stack.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    n_train_params = sum(p.numel() for p in trainable_params)
    print(f"[pair_score] trainable params: {n_train_params:,}", flush=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
    )

    # ---- 5. Train + log ----
    best_val_loss = float("inf")
    best_path = out_dir / "pair_score_best.pt"
    last_path = out_dir / "pair_score_last.pt"
    log_path = out_dir / "log.json"
    log_entries: list[dict[str, Any]] = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_metrics = _train_one_epoch(stack, train_loader, optim, device, unfrozen)
        val_metrics = _evaluate(stack, val_loader, device)
        elapsed = time.time() - t0
        log = {
            "epoch": epoch,
            "elapsed_s": round(elapsed, 1),
            "epoch_s": round(time.time() - t_epoch, 1),
            "train": train_metrics,
            "val": val_metrics,
        }
        log_entries.append(log)
        log_path.write_text(json.dumps(log_entries, indent=2))
        print(
            f"[pair_score] ep={epoch:3d} "
            f"tr_loss={train_metrics.get('loss', 0):.4f} "
            f"tr_top1={train_metrics.get('top1', 0):.3f}  |  "
            f"val_loss={val_metrics.get('loss', 0):.4f} "
            f"val_top1={val_metrics.get('top1', 0):.3f} "
            f"val_top3={val_metrics.get('top3', 0):.3f} "
            f"val_top5={val_metrics.get('top5', 0):.3f} "
            f"rand={val_metrics.get('random_valid_top1', 0):.3f} "
            f"dt={time.time() - t_epoch:.1f}s",
            flush=True,
        )

        # Save last + best. Include any unfrozen encoder state so a
        # follow-up run can resume via --init-from.
        ckpt_payload: dict = {
            "epoch": epoch,
            "encoder_ckpt": str(args.encoder_ckpt),
            "init_from": str(init_from) if init_from else None,
            "unfrozen": sorted(unfrozen),
            "config": {
                "d_model": args.d_model,
                "hidden": args.hidden,
                "max_planets": args.max_planets,
                "max_fleets": args.max_fleets,
                "n_history": args.n_history,
            },
            "metrics": {"train": train_metrics, "val": val_metrics},
        }
        ckpt_payload.update(stack.trainable_module_state(unfrozen))
        torch.save(ckpt_payload, last_path)
        if val_metrics.get("loss", float("inf")) < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(torch.load(last_path, weights_only=False), best_path)

    print(f"[pair_score] done. best_val_loss={best_val_loss:.4f} "
          f"ckpts: {best_path.name}, {last_path.name}", flush=True)
    return best_path


# ---------- CLI ----------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder-ckpt", type=Path, required=True,
                   help="Path to a stack ckpt with fleet/planet/entity/cross state dicts.")
    p.add_argument("--action-dir", type=Path, default=ACTION_DATASET_DIR)
    p.add_argument("--planet-dir", type=Path, default=PLANET_DATASET_DIR)
    p.add_argument("--fleet-dir", type=Path, default=FLEET_DATASET_DIR)
    p.add_argument("--entity-dir", type=Path, default=ENTITY_DATASET_DIR)
    p.add_argument("--cross-entity-dir", type=Path, default=CROSS_ENTITY_DATASET_DIR)
    p.add_argument("--filter", choices=["winner", "all"], default="all",
                   help="winner = keep only CSVs whose learner_slot == winner_seat. "
                        "Composes with --player.")
    p.add_argument("--player", default=None,
                   help="Restrict to one player's replays (e.g. 'kovi', "
                        "'Shun_PI', 'Orbital Occle'). Replay tree is "
                        "<replay-dir>/<player>/<stem>.json.gz.")
    p.add_argument("--replay-dir", type=Path,
                   default=Path("data/replays"),
                   help="Root containing per-player replay subdirs.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap on acted rows (after filter). None = no cap.")
    p.add_argument("--overfit", action="store_true",
                   help="Use train==val (tiny-overfit Experiment 1).")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--max-planets", type=int, default=64)
    p.add_argument("--max-fleets", type=int, default=256)
    p.add_argument("--n-history", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--unfreeze", default=None,
                   help="Comma-separated encoder modules to thaw alongside "
                        "the head. Names: cross, entity (=entity_encoder), "
                        "planet (=planet_encoder), fleet (=fleet_encoder). "
                        "Example: --unfreeze cross,entity")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Resume from a prior pair_score_best.pt. Loads the "
                        "head and any previously-thawed encoder state. "
                        "Encoders not in the prior file fall back to the "
                        "ones in --encoder-ckpt.")
    args = p.parse_args()
    train_pair_score(args)


if __name__ == "__main__":
    main()
