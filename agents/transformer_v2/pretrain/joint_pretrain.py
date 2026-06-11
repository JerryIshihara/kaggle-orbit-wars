"""Joint action + value pretraining over ONE shared L2 backbone.

The action head (PairHead, single-target contract) and the explicit value heads
(win + 5 relative P1 signals at future horizons + anti-shortcut aux) branch
from the SAME ``EntityPretrainModel`` L2 — the action branch via L3/L4, the
value branch via the PlayerConsolidator — so a combined loss co-adapts the one
L2 (and L3/L4/consolidator/heads) while L0+L1 stay frozen ("L2~ unfreeze").

Two caches feed the two losses (the action labels live in the pair cache, the
value labels in the cross-entity cache); each batch is L0-encoded the same way
and run through ``forward_with_context``, which emits both head groups. The
value heads predict the ACTUAL FUTURE LEVEL sᵢ(t+K) of each P1 signal at
horizons {5,10,15,20,50} (see ``value_signals`` / ``value_heads``); the diffs
that drive PBRS shaping are derived from those levels at PPO time.

Data note: the pair cache is T=6 and the cross-entity cache is T=10. A shared L2
built with ``n_steps=10`` consumes both via ``step_embed[-T:]``, but for clean
temporal semantics rebuild the two caches with matched ``history_offsets``
(or build one unified cache carrying both label sets). This is the documented
prerequisite for a production joint run; the loss/step/loop below are
cache-agnostic.
"""

from __future__ import annotations

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
    value_coef: float = 1.0,
    multinomial_alloc: bool = False,
    pair_pos_weight: float = 600.0,
    alloc_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined action + value loss over the shared L2.

    ``total = action_loss + value_coef * value_loss`` where the action loss is
    the PairHead objective on the pair-cache batch — by default the
    single-target contract (``compute_multi_loss(single_target=True)``), or
    with ``multinomial_alloc=True`` the
    ``bernoulli_select_multinomial_alloc_v2`` contract (whole-grid select BCE
    with ``pair_pos_weight`` + stage-2 allocation CE that supervises
    ``frac_loc`` and the HOLD diagonal in PPO's exact softmax) — and the value
    loss is the action-impact ``compute_value_pretrain_loss`` (win tier-1 + 5
    future-level signal heads tier-2 + aux tier-3) on the cross-entity batch.
    Either batch may be ``None`` (loader exhaustion) — only the present side
    contributes. Per-head losses are returned namespaced ``act/<head>`` and
    ``val/<head>``.
    """
    from .value_heads import compute_value_pretrain_loss

    fleet_enc, planet_enc, comet_enc = encoders
    total: torch.Tensor | None = None
    per_head: dict[str, float] = {}

    if action_batch is not None:
        act_out = _forward_context_from_batch(
            model, fleet_enc, planet_enc, comet_enc, action_batch,
        )
        if multinomial_alloc:
            act_loss, act_terms = compute_multi_loss(
                act_out, action_batch,
                multinomial_alloc=True,
                pair_pos_weight=pair_pos_weight,
                alloc_weight=alloc_weight,
            )
        else:
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
        val_loss, val_terms = compute_value_pretrain_loss(val_out, value_batch)
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
    """Yield (action_batch, value_batch) for max(len) steps, restarting the
    shorter loader's iterator when it is exhausted so every step trains both
    branches (the longer loader sets the epoch length; the shorter repeats).

    Uses explicit ``iter()`` reset on ``StopIteration`` rather than
    ``itertools.cycle``: ``cycle`` caches every batch it has yielded, which (a)
    pins a full epoch of collated tensors in RAM for the whole run and (b)
    replays them in the IDENTICAL order every cycle (no re-shuffle). Re-iterating
    a shuffled ``DataLoader`` reshuffles each pass and holds only the current
    batch."""
    n = max(len(action_loader), len(value_loader))
    a_it = iter(action_loader)
    v_it = iter(value_loader)
    for _ in range(n):
        try:
            a = next(a_it)
        except StopIteration:
            a_it = iter(action_loader)
            a = next(a_it)
        try:
            v = next(v_it)
        except StopIteration:
            v_it = iter(value_loader)
            v = next(v_it)
        yield a, v


def _split_value_by_game(value_ds):
    """Split the cross-entity value dataset into (train, val) Subsets BY GAME,
    using the cache's stored ``split_stems`` (train/val/test stem lists) +
    ``episode_to_stem``. Snapshots whose game-stem is in the val split go to the
    held-out set; everything not-in-val (train + test stems, or all games if no
    split metadata) is used for training. Returns ``(train_subset, val_subset)``;
    ``val_subset`` is ``None`` when the cache carries no split metadata."""
    from torch.utils.data import Subset

    cfg = getattr(value_ds, "config", {}) or {}
    split = cfg.get("split_stems")
    e2s = cfg.get("episode_to_stem")
    keys = value_ds.keys
    if not split or not e2s:
        print("[joint] WARNING: cross cache has no split_stems/episode_to_stem — "
              "training value heads on ALL games with NO held-out eval (win_acc "
              "cannot be trusted; rebuild the cache to enable the holdout).",
              flush=True)
        return value_ds, None

    val_stems = set(split.get("val", []))
    val_idx, train_idx = [], []
    for i, (ep, _t) in enumerate(keys):
        (val_idx if e2s.get(ep) in val_stems else train_idx).append(i)
    n_train_games = len({e2s.get(keys[i][0]) for i in train_idx})
    n_val_games = len({e2s.get(keys[i][0]) for i in val_idx})
    print(f"[joint] value split BY GAME: train={len(train_idx)} snaps / "
          f"{n_train_games} games | holdout(val)={len(val_idx)} snaps / "
          f"{n_val_games} games", flush=True)
    return Subset(value_ds, train_idx), Subset(value_ds, val_idx)


@torch.no_grad()
def _value_holdout_eval(model, encoders, val_loader, device,
                        max_batches: int = 64) -> dict[str, float]:
    """Held-out value-head eval over whole games the model never trained on.
    Returns mean per-head metrics namespaced ``holdout/<head>`` (e.g.
    ``holdout/win``, ``holdout/win_acc``, ``holdout/fwd``) — distinct from the
    training-time ``val/<head>`` keys, where 'val' means VALUE-branch, not
    validation. Caps at ``max_batches`` (the loader is shuffled, so this is a
    representative sample across all holdout games — win is per-game-constant,
    so cross-game coverage is what matters) to keep per-epoch eval cheap even
    when the holdout set is large. Dropout off via ``model.eval()``; caller
    restores ``train()``."""
    from .value_heads import compute_value_pretrain_loss

    fleet_enc, planet_enc, comet_enc = encoders
    model.eval()
    agg: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for bi, batch in enumerate(val_loader):
        if bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = _forward_context_from_batch(
            model, fleet_enc, planet_enc, comet_enc, batch)
        _loss, per = compute_value_pretrain_loss(out, batch)
        for k, v in per.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            agg[k] = agg.get(k, 0.0) + v
            cnt[k] = cnt.get(k, 0) + 1
    return {f"holdout/{k}": agg[k] / max(1, cnt[k]) for k in agg}


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
    conditioner_n_layers: int = 1,
    head_n_layers: int = 1,
    batch_size: int = 16,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    launch_weight: float = 1.0,
    value_coef: float = 1.0,
    value_dropout: float = 0.1,
    multinomial_alloc: bool = False,
    pair_pos_weight: float = 600.0,
    alloc_weight: float = 1.0,
    warm_start: Path | None = None,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 0,
    progress_every: int = 50,
    # --- arch injection points (transformer_v3 driver) ------------------
    # ``model_cls``: alternative EntityPretrainModel subclass (same ctor
    # kwargs). ``history_offsets``: override the temporal window both
    # caches are restacked to (default: this package's HISTORY_OFFSETS).
    # ``warm_start_adapter``: callable(sd) -> sd applied to the checkpoint
    # state dict before load_state_dict (e.g. v2->v3 L2 key mapping).
    model_cls: type | None = None,
    history_offsets: tuple[int, ...] | None = None,
    warm_start_adapter=None,
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

    model = (model_cls or EntityPretrainModel)(
        d_model=d_model, n_steps=n_steps,
        conditioner_n_layers=conditioner_n_layers,
        head_n_layers=head_n_layers,
        with_consolidator=True, with_value_heads=True,
        value_dropout=value_dropout,
    ).to(device)
    if model_cls is not None:
        print(f"[joint] model_cls={model_cls.__name__} "
              f"arch={getattr(model, 'ARCH', 'v2')}", flush=True)
    if warm_start is not None:
        ck = torch.load(warm_start, map_location=device, weights_only=False)
        sd = ck.get("model", ck)
        # Warm-start the shared backbone + action head (PairHead/L2/L3/L4/
        # consolidator) from the previous run; the NEW value heads stay fresh.
        # DROP any old ``value_heads.*`` keys first: the previous design's value
        # heads share names with the new ones at DIFFERENT shapes, and
        # ``load_state_dict(strict=False)`` still hard-errors on a shape mismatch
        # for a shared key (it only tolerates missing/unexpected). Filtering them
        # out guarantees the new value heads initialize fresh.
        n_drop = sum(1 for k in sd if k.startswith("value_heads."))
        sd = {k: v for k, v in sd.items() if not k.startswith("value_heads.")}
        if warm_start_adapter is not None:
            n_in = len(sd)
            sd = warm_start_adapter(sd)
            print(f"[joint] warm_start_adapter: {n_in} -> {len(sd)} tensors",
                  flush=True)
        res = model.load_state_dict(sd, strict=False)
        non_value_missing = [k for k in res.missing_keys
                             if not k.startswith("value_heads.")]
        print(f"[joint] warm-started from {warm_start}: loaded {len(sd)} backbone/"
              f"action tensors; dropped {n_drop} old value_heads.* (re-init fresh); "
              f"missing={len(res.missing_keys)} (non-value={len(non_value_missing)}) "
              f"unexpected={len(res.unexpected_keys)}", flush=True)
        if non_value_missing:
            print(f"[joint]   WARNING non-value missing keys (backbone skew?): "
                  f"{non_value_missing[:6]}", flush=True)
    report = model.freeze_below_l2()
    print("[joint] freeze_below_l2 (L0+L1 frozen, L2+ trainable):", flush=True)
    for k, v in report.items():
        print(f"    {k:<28s} {v:,}", flush=True)

    print(f"[joint] action cache: {pair_cache_path}", flush=True)
    action_full = CachedPairDataset(pair_cache_path)
    # --- force the MODEL's temporal contract onto the pair cache ---
    # The deployed stack is T=10 strided (history.HISTORY_OFFSETS); legacy pair
    # caches were built T=6-consecutive but store EVERY turn as single frames
    # and stack lazily at __getitem__, so restacking is a metadata override,
    # not a rebuild. This also matches the cross-entity (value) cache, closing
    # the documented T=6/T=10 mismatch.
    from ..history import HISTORY_OFFSETS as _DEFAULT_OFFSETS
    HISTORY_OFFSETS = (
        tuple(history_offsets) if history_offsets is not None
        else _DEFAULT_OFFSETS
    )
    if tuple(action_full.history_offsets) != HISTORY_OFFSETS:
        print(f"[joint]   pair cache stored history_offsets="
              f"{tuple(action_full.history_offsets)} -> OVERRIDING to model "
              f"HISTORY_OFFSETS={HISTORY_OFFSETS} (T={len(HISTORY_OFFSETS)})",
              flush=True)
        action_full.history_offsets = HISTORY_OFFSETS
    # Context-coverage probe: strided lookback only works if the cache kept
    # per-turn context frames. Old acted-only caches (no stored context) would
    # silently stack zero frames — refuse those.
    _key_set = set(action_full.keys)
    _acted_probe = list(getattr(action_full, "acted_indices", []) or [])
    _sample = _acted_probe[:: max(1, len(_acted_probe) // 500)] or [0]
    _full = sum(
        all((ep, t - off) in _key_set for off in HISTORY_OFFSETS
            if off <= t)
        for ep, t in (action_full.keys[i] for i in _sample)
    )
    _cov = _full / max(1, len(_sample))
    print(f"[joint]   strided-T{len(HISTORY_OFFSETS)} context coverage: "
          f"{_full}/{len(_sample)} sampled acted rows have every in-game "
          f"lookback frame ({_cov:.0%}; pre-game offsets zero-fill like the "
          f"deploy deque)", flush=True)
    if _cov < 0.5:
        raise ValueError(
            "pair cache lacks per-turn context frames for strided T=10 "
            "stacking — rebuild it with the current builder (which stores "
            "every turn), e.g. the Ebi_T6 cache layout.",
        )
    _probe_item = action_full[_acted_probe[0] if _acted_probe else 0]
    _pf = _probe_item["planet_features"]
    assert _pf.dim() == 3 and _pf.shape[0] == len(HISTORY_OFFSETS), (
        f"expected stacked planet_features (T={len(HISTORY_OFFSETS)}, P, D), "
        f"got {tuple(_pf.shape)}"
    )
    print(f"[joint]   probe item planet_features={tuple(_pf.shape)} "
          f"(T axis verified)", flush=True)
    # Train the action head ONLY on acted (launch) turns — non-acted snapshots
    # are kept in the cache solely as T-history context. Iterating the whole
    # cache (all snapshots) floods the single-target CE with pure-NOOP turns
    # and drives the actor to over-hold; restricting to acted_indices keeps the
    # launch:hold balance close to the experts'. (entity_encoder.train does the
    # same via train_row_indices=acted_indices.)
    acted = list(getattr(action_full, "acted_indices", []) or [])
    action_ds = torch.utils.data.Subset(action_full, acted) if acted else action_full
    print(
        f"[joint]   {len(action_full)} snapshots; training on "
        f"{len(action_ds)} ACTED/launch turns "
        f"({100 * len(action_ds) / max(1, len(action_full)):.0f}% acted)",
        flush=True,
    )
    print(f"[joint] value  cache: {cross_cache_path}", flush=True)
    value_ds = CachedCrossEntitySnapshotDataset(
        cross_cache_path, history_offsets=HISTORY_OFFSETS,
    )
    # --- game-level train/val split for the value branch ---
    # The win head's label is a per-game CONSTANT, so a snapshot-level split
    # leaks (sibling snapshots of the same game land in both sides) and the head
    # memorizes game identity. Split by GAME using the cache's `split_stems`, and
    # hold the val games out entirely so `val/win_acc` measures real generalization.
    from torch.utils.data import Subset
    value_train_ds, value_val_ds = _split_value_by_game(value_ds)
    action_loader = DataLoader(
        action_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    value_loader = DataLoader(
        value_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    value_val_loader = (
        DataLoader(value_val_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=False)
        if value_val_ds is not None and len(value_val_ds) > 0 else None
    )
    print(
        f"[joint] action batches/epoch={len(action_loader)}  "
        f"value(train) batches/epoch={len(value_loader)}  "
        f"value(holdout) batches={len(value_val_loader) if value_val_loader else 0}  "
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
    cache_cfg = getattr(action_full, "config", {}) or {}
    config = {
        "d_model": d_model, "n_steps": n_steps, "batch_size": batch_size,
        "epochs": epochs, "lr": lr, "launch_weight": launch_weight,
        "value_coef": value_coef, "value_dropout": value_dropout,
        "multinomial_alloc": multinomial_alloc,
        "pair_pos_weight": pair_pos_weight, "alloc_weight": alloc_weight,
        "warm_start": str(warm_start) if warm_start else None,
        "action_contract": (
            "bernoulli_select_multinomial_alloc_v2" if multinomial_alloc
            else "single_target_per_source_v1"
        ),
        "pair_history_offsets": list(action_full.history_offsets),
        "pair_cache_path": str(pair_cache_path),
        "cross_cache_path": str(cross_cache_path),
        # Architecture (matches the EntityPretrainModel(...) built above) so
        # runner.TransformerAgent.load() reconstructs the stack correctly
        # instead of falling back to the legacy d_pair=128 / 4-head defaults.
        "d_pair": d_model,
        "entity_n_heads": 8, "cross_n_heads": 8, "cross_n_layers": 2,
        "dual_n_heads": 8,
        "conditioner_n_layers": conditioner_n_layers,
        "head_n_layers": head_n_layers,
        "skip_l34": False, "with_consolidator": True, "with_value_heads": True,
        "max_planets": int(cache_cfg.get("max_planets", 64)),
        "max_fleets": int(cache_cfg.get("max_fleets", 1024)),
    }
    # Arch-specific self-description (e.g. transformer_v3 dual-rate L2
    # stamps arch + union/long/short offset tuples) — lets the runner and
    # PPO loaders reconstruct non-default architectures from the ckpt.
    config.update(getattr(model, "config_extra", {}) or {})
    print(f"[joint] action_contract={config['action_contract']}  "
          f"max_planets={config['max_planets']} max_fleets={config['max_fleets']} "
          f"(from pair cache)", flush=True)

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
                launch_weight=launch_weight,
                value_coef=value_coef,
                multinomial_alloc=multinomial_alloc,
                pair_pos_weight=pair_pos_weight,
                alloc_weight=alloc_weight,
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

        # --- held-out value eval on whole games never seen in training ---
        val_metrics: dict[str, float] = {}
        if value_val_loader is not None:
            val_metrics = _value_holdout_eval(
                model, (fleet_enc, planet_enc, comet_enc), value_val_loader, device)
            model.train()  # restore (eval() was set inside)
            print(format_joint_per_head(val_metrics, title=f"ep {epoch} HOLDOUT (val games)"),
                  flush=True)
            tr_wa = mean_per_head.get("val/win_acc")
            ho_wa = val_metrics.get("holdout/win_acc")
            if tr_wa is not None and ho_wa is not None:
                # The memorization tell: train win_acc → 1.0 while held-out stays
                # near the 0.5–0.67 base rate ⇒ the win head is memorizing games.
                print(f"    [win memorization gap] train_win_acc={tr_wa:.3f}  "
                      f"holdout_win_acc={ho_wa:.3f}  gap={tr_wa - ho_wa:+.3f}",
                      flush=True)

        entry = {"epoch": epoch, "joint_total": mean_total,
                 "per_head": mean_per_head, "holdout": val_metrics,
                 "elapsed_s": elapsed}
        log.append(entry)
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, last_path)
        if epoch == 1 or mean_total <= min(e["joint_total"] for e in log):
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, best_path)
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    print(f"[joint] done. outputs in {out_dir}", flush=True)
    return best_path


def main() -> None:
    import argparse

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
    p.add_argument("--conditioner-n-layers", type=int, default=1,
                   help="FiLM conditioner hidden layers (deep-heads design: 2)")
    p.add_argument("--head-n-layers", type=int, default=1,
                   help="per-head decoder MLP depth (deep-heads design: 3)")
    p.add_argument("--launch-weight", type=float, default=1.0,
                   help="up-weight launching source rows in the action CE "
                        "(single-target NOOP-imbalance control)")
    p.add_argument("--multinomial-alloc", action="store_true",
                   help="train the bernoulli_select_multinomial_alloc_v2 "
                        "contract: whole-grid select BCE + allocation-"
                        "multinomial CE over [frac_loc, HOLD diagonal] — the "
                        "contract PPO samples (replaces --launch-weight's "
                        "single-target CE)")
    p.add_argument("--pair-pos-weight", type=float, default=600.0,
                   help="positive-cell weight of the whole-grid select BCE "
                        "(multinomial-alloc contract only)")
    p.add_argument("--alloc-weight", type=float, default=1.0,
                   help="weight of the allocation-multinomial CE term "
                        "(multinomial-alloc contract only)")
    p.add_argument("--value-coef", type=float, default=1.0,
                   help="weight on the value loss relative to the action loss")
    p.add_argument("--value-dropout", type=float, default=0.1,
                   help="dropout on the value trunk/heads ONLY (decoupled from "
                        "the action backbone) — regularizes the win head against "
                        "memorizing the small set of training games")
    p.add_argument("--warm-start", type=Path, default=None,
                   help="previous joint_best.pt to warm-start the shared "
                        "backbone + action head from (strict=False; the new "
                        "value heads stay freshly initialized)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--arch", choices=("v2", "v3"), default="v2",
                   help="v3 = transformer_v3 dual-rate L2 (long T=10@5 + "
                        "short T=10@2 branches, zero-init fusion); both "
                        "caches are restacked to the 18-frame union and a "
                        "v2 --warm-start is key-mapped onto both branches")
    args = p.parse_args()

    arch_kwargs: dict[str, Any] = {}
    if args.arch == "v3":
        from ...transformer_v3 import (
            EntityPretrainModelV3,
            UNION_HISTORY_OFFSETS,
            adapt_v2_state_dict,
        )
        arch_kwargs = dict(
            model_cls=EntityPretrainModelV3,
            history_offsets=UNION_HISTORY_OFFSETS,
            warm_start_adapter=adapt_v2_state_dict,
        )

    train_joint(
        out_dir=args.out_dir,
        fleet_run_dir=args.fleet_run_dir,
        planet_run_dir=args.planet_run_dir,
        comet_run_dir=args.comet_run_dir,
        pair_cache_path=args.pair_cache_path,
        cross_cache_path=args.cross_cache_path,
        d_model=args.d_model,
        n_steps=args.n_steps,
        conditioner_n_layers=args.conditioner_n_layers,
        head_n_layers=args.head_n_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        launch_weight=args.launch_weight,
        value_coef=args.value_coef,
        value_dropout=args.value_dropout,
        multinomial_alloc=args.multinomial_alloc,
        pair_pos_weight=args.pair_pos_weight,
        alloc_weight=args.alloc_weight,
        warm_start=args.warm_start,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        progress_every=args.progress_every,
        **arch_kwargs,
    )


if __name__ == "__main__":
    main()
