"""Live observation → batch-dict featurization for inference.

Bridges the training CSV pipeline and the runtime agent. Calls
:func:`featurize_planets` and :func:`featurize_fleets` on a single
``obs`` and returns the same ``batch`` dict shape that
:meth:`EntitySnapshotDataset._build_snapshot` produces (B=1, no T axis
— :class:`CrossEntityAttention` auto-promotes single-step input).

Why a separate file: the training-side CSV writers live next to their
schema constants in fleet/planet featurizers. The inference path needs
the same per-fleet routing tensors (``fleet_target_idx``,
``fleet_source_idx``, ``fleet_owner_slot``, ``fleet_ships_log``,
``fleet_eta_norm``) that are derived from CSV columns at training time
— here we derive them from the ``FleetFeaturizer`` records returned by
:func:`featurize_fleets`. Keeping that re-derivation here means the
training schema can evolve without breaking the agent.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .fleet_featurizer import FLEET_RAW_DIM, FleetTracker, SHIPS_LOG_MAX, featurize_fleets
from .planet_featurizer import PLANET_RAW_DIM, featurize_planets


def featurize_observation(
    obs: Any,
    *,
    learner_slot: int,
    tracker: FleetTracker,
    num_players: int = 4,
    max_planets: int = 64,
    max_fleets: int = 256,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
    """Build the batch dict + planet_id → planet_idx map for one obs.

    Returns a tuple ``(batch, pid_to_idx)``. ``batch`` is fed straight
    into :class:`ActionTrainStack.forward`. ``pid_to_idx`` is needed by
    the agent to translate a model-predicted planet index back to the
    real ``planet_id`` for action-emission.

    The fleet feature vector's ETA-norm slot is hard-coded to index 10 —
    matches what :meth:`EntitySnapshotDataset._build_snapshot` reads
    out of the fleet CSV. If
    :meth:`fleet_featurizer.FleetFeaturizer.to_vector` ever reorders
    its output, update both call sites in lock-step.
    """
    p_features, p_mask, p_records = featurize_planets(
        obs,
        learner_slot=learner_slot,
        num_players=num_players,
        max_entities=max_planets,
    )
    f_features, f_mask, f_records = featurize_fleets(
        obs,
        learner_slot=learner_slot,
        tracker=tracker,
        num_players=num_players,
        max_fleets=max_fleets,
    )

    # featurize_planets / featurize_fleets pad to max_* internally, so
    # the returned tensors are already shape (max, RAW_DIM) / (max,).
    if p_features.shape != (max_planets, PLANET_RAW_DIM):
        raise ValueError(
            f"planet feature shape mismatch: got {tuple(p_features.shape)}, "
            f"expected ({max_planets}, {PLANET_RAW_DIM})"
        )
    if f_features.shape != (max_fleets, FLEET_RAW_DIM):
        raise ValueError(
            f"fleet feature shape mismatch: got {tuple(f_features.shape)}, "
            f"expected ({max_fleets}, {FLEET_RAW_DIM})"
        )

    pid_to_idx: dict[int, int] = {
        rec.planet_id: i for i, rec in enumerate(p_records)
    }

    fleet_target_idx = torch.full((max_fleets,), -1, dtype=torch.long)
    fleet_source_idx = torch.full((max_fleets,), -1, dtype=torch.long)
    fleet_owner_slot = torch.zeros(max_fleets, dtype=torch.long)
    fleet_ships_log = torch.zeros(max_fleets, dtype=torch.float32)
    fleet_eta_norm = torch.ones(max_fleets, dtype=torch.float32)

    for i, rec in enumerate(f_records):
        if i >= max_fleets:
            break
        if rec.target_planet_id is not None and rec.target_planet_id in pid_to_idx:
            fleet_target_idx[i] = pid_to_idx[rec.target_planet_id]
        if rec.source_planet_id in pid_to_idx:
            fleet_source_idx[i] = pid_to_idx[rec.source_planet_id]
        if 0 <= rec.owner_id < num_players:
            fleet_owner_slot[i] = (rec.owner_id - learner_slot) % num_players
        fleet_ships_log[i] = math.log1p(max(0, rec.ships)) / SHIPS_LOG_MAX
        # Slot 10 of the to_vector output carries eta_norm; mirror what
        # EntitySnapshotDataset._build_snapshot reads from the CSV.
        fleet_eta_norm[i] = float(f_features[i, 10])

    batch: dict[str, torch.Tensor] = {
        "planet_features": p_features.unsqueeze(0).to(device),
        "planet_mask": p_mask.bool().unsqueeze(0).to(device),
        "fleet_features": f_features.unsqueeze(0).to(device),
        "fleet_mask": f_mask.bool().unsqueeze(0).to(device),
        "fleet_target_idx": fleet_target_idx.unsqueeze(0).to(device),
        "fleet_source_idx": fleet_source_idx.unsqueeze(0).to(device),
        "fleet_owner_slot": fleet_owner_slot.unsqueeze(0).to(device),
        "fleet_ships_log": fleet_ships_log.unsqueeze(0).to(device),
        "fleet_eta_norm": fleet_eta_norm.unsqueeze(0).to(device),
    }
    return batch, pid_to_idx
