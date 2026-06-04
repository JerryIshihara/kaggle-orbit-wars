"""Pair-cache sampler for the single-target BC anchor.

Wraps the existing ``CachedPairDataset`` (from
``scripts/build_pair_dataset_orbital_occle.py``) and produces
:class:`agents.transformer_v2.ppo.loss.BCMinibatch` samples on demand.

The cache stores per-snapshot:
  * model inputs (planet_features, fleet_features, routing tensors,
    planet_mask, fleet_mask, is_comet) — T=6 history-stacked
  * ``pair_labels (P, P) bool`` — expert source→target pair set at
    the snapshot's current step (multi-positive on coalition turns)
  * ``pair_valid (P, P) bool`` — same as ``planet_mask × planet_mask``
    minus the diagonal

For the single-target BC (matching pretrain ``_pair_single_target_ce``):
  * ``source_mask`` is derived from current-frame planet owner features so
    the source CE never trains on non-learner-owned rows.
  * ``expert_tgt_idx (P,)`` is the per owned source row label:
        diagonal ``s``                       if the row held (no positive),
        argmax_t(pair_ships[s, t]) over its   otherwise (the dominant target
        off-diagonal positives                column; ties -> lowest index).
    Held rows supervise the diagonal HOLD slot; launching rows supervise the
    chosen target column.

``pair_type_ids`` is computed on the fly via
:func:`build_pair_type_ids` since the cache doesn't store it.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import torch

from agents.transformer_v2.pretrain.entity_encoder import (
    _PLANET_OWNER_START_IDX,
    _current_planet_features,
    build_pair_type_ids,
)

from .loss import BCMinibatch


class BCCacheSampler:
    """Lazy sampler over a pair_cache .pt for the factorized BC anchor.

    Loads the cache once at construction; ``sample(size)`` returns a
    fresh random minibatch each call. Snapshots are accessed lazily
    through ``CachedPairDataset.__getitem__`` — no per-sample disk reads
    after the initial load.
    """

    def __init__(self, cache_path: Path, *, device: str = "cpu",
                  acted_only: bool = True):
        # Imported lazily because the dataset class lives under
        # scripts/, which isn't a package on disk.
        from scripts.build_pair_dataset_orbital_occle import CachedPairDataset

        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"pair cache not found: {self.cache_path}")
        self.device = device
        t0 = time.time()
        print(f"[bc] loading pair cache from {self.cache_path} ...", flush=True)
        self.dataset = CachedPairDataset(self.cache_path)
        load_sec = time.time() - t0
        # Filter to acted snapshots (any expert positive) by default —
        # the BC source-categorical needs both NOOP and acted samples,
        # but a cache without ACTING snapshots starves the target loss.
        self.indices: list[int] = list(range(len(self.dataset)))
        if acted_only:
            acted = getattr(self.dataset, "acted_indices", None)
            if acted is not None:
                self.indices = list(acted)
            else:
                self.indices = [
                    i for i in self.indices
                    if bool(self.dataset.snapshots[i]["pair_labels"].any())
                ]
        print(f"[bc] available: {len(self.indices)} snapshots "
              f"(acted_only={acted_only}, load_sec={load_sec:.1f})", flush=True)

    def __len__(self) -> int:
        return len(self.indices)

    def sample(self, size: int) -> BCMinibatch | None:
        if not self.indices or size <= 0:
            return None
        idxs = random.choices(self.indices, k=size)
        items = [self.dataset[i] for i in idxs]

        # Stack the model-input tensors. The cache returns history-
        # stacked features (T=6 axis already present); we keep that.
        def _stack(key: str) -> torch.Tensor:
            return torch.stack([it[key] for it in items])

        planet_features = _stack("planet_features").to(self.device)
        fleet_features = _stack("fleet_features").to(self.device)
        planet_mask = _stack("planet_mask").bool().to(self.device)
        is_comet = _stack("is_comet").bool().to(self.device)
        routing = {
            "fleet_target_idx": _stack("fleet_target_idx").to(self.device),
            "fleet_source_idx": _stack("fleet_source_idx").to(self.device),
            "fleet_owner_slot": _stack("fleet_owner_slot").to(self.device),
            "fleet_ships_log": _stack("fleet_ships_log").to(self.device),
            "fleet_eta_norm": _stack("fleet_eta_norm").to(self.device),
            "fleet_mask": _stack("fleet_mask").bool().to(self.device),
        }
        # Current-frame planet_features for build_pair_type_ids — strip T axis
        # if history-stacked.
        pf_for_type = (
            planet_features[:, -1] if planet_features.dim() == 4
            else planet_features
        )
        planet_mask_now = (
            planet_mask[:, -1] if planet_mask.dim() == 3 else planet_mask
        )
        pair_type_ids = build_pair_type_ids(pf_for_type, planet_mask_now).to(self.device)

        # Expert labels — current-step (P, P) per snapshot.
        pair_labels = torch.stack([it["pair_labels"].bool() for it in items]).to(self.device)
        pair_valid = torch.stack([it["pair_valid"].bool() for it in items]).to(self.device)

        # PPO source-categorical legality approximation. The rollout source
        # mask also checks launch surplus, but the cache has no inbound-threat
        # surplus calculation. Owner legality is the important invariant here:
        # non-learner-owned rows are impossible source choices in PPO.
        pf_now = _current_planet_features(planet_features)
        source_mask = (
            (pf_now[..., _PLANET_OWNER_START_IDX] > 0.5)
            & planet_mask_now
            & pair_valid.any(dim=2)
        )

        # expert_tgt_idx: per owned source row, the single-target label
        # (mirrors pretrain _pair_single_target_ce):
        #   diagonal s                            if the row held (no positive),
        #   argmax_t(pair_ships[s, t]) over its    otherwise (dominant target;
        #   off-diagonal positives                 ties -> lowest index).
        B, P, _ = pair_labels.shape
        pos = pair_labels & pair_valid                               # (B, P, P)
        row_has_pos = pos.any(dim=2)                                 # (B, P)
        if "pair_ships" in items[0]:
            score = torch.stack([
                it["pair_ships"].float() for it in items
            ]).to(self.device)
        else:
            score = pair_labels.float()
        score = score.masked_fill(~pos, -1.0)
        dom_t = score.argmax(dim=2)                                  # (B, P)
        diag_idx = torch.arange(P, device=self.device).expand(B, P)  # (B, P)
        expert_tgt_idx = torch.where(row_has_pos, dom_t, diag_idx).long()  # (B, P)

        feats: dict[str, Any] = {
            "planet_features": planet_features,
            "fleet_features": fleet_features,
            "planet_mask": planet_mask,
            "is_comet": is_comet,
            "pair_type_ids": pair_type_ids,
            "routing": routing,
        }
        return BCMinibatch(
            feats=feats,
            pair_mask=pair_valid,
            source_mask=source_mask,
            expert_tgt_idx=expert_tgt_idx,
        )
