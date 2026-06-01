"""Pair-cache sampler for the factorized BC anchor.

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

For the factorized BC:
  * ``expert_target_bits = pair_labels`` (the full P×P matrix; the loss
    masks to the chosen source's row at compute time).
  * ``source_mask`` is derived from current-frame planet owner features so
    the source categorical never trains on non-learner-owned rows.
  * ``expert_source_idx_or_noop`` is derived per snapshot:
        0  if no positives (NOOP — expert took no launch action)
        1 + argmax_s(pair_ships[s, :].sum())  otherwise (the source
        with the largest launched ship total among learner-owned rows,
        canonical for coalition turns)

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

        # expert_source_idx_or_noop:
        #   0 if no legal positives (snapshot is NOOP for PPO source CE),
        #   1 + argmax_s(total launched ships from legal source s) otherwise.
        if "pair_ships" in items[0]:
            pair_ships = torch.stack([
                it["pair_ships"].float() for it in items
            ]).to(self.device)
            row_scores = pair_ships.sum(dim=2)
        else:
            row_scores = pair_labels.float().sum(dim=2)
        row_scores = row_scores.masked_fill(~source_mask, 0.0)       # (B, P)
        has_any = row_scores.sum(dim=1) > 0                          # (B,)
        argmax_src = row_scores.argmax(dim=1)                        # (B,)
        expert_src = torch.where(
            has_any, argmax_src + 1, torch.zeros_like(argmax_src),
        ).long()

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
            expert_source_idx_or_noop=expert_src,
            expert_target_bits=pair_labels,
        )
