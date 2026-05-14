"""Per-turn target-logit scoring from a saved replay.

Loads a ``target_rank_best.pt`` checkpoint (the new two-stage attention
ranker — see ``agents/transformer_v1/pretrain/target_rank.py``), walks
every turn of a Kaggle Orbit Wars replay, and emits the ranker's
per-planet target logits + softmax probabilities. Designed for the
dashboard's side-by-side target-score visualization.

The replay viewer expects each turn's payload to carry:

  * ``turn`` — int
  * ``planets`` — list of ``{id, x, y, owner, ships, score, prob,
    target_valid}`` per real planet visible in the observation.

History (the encoder's ``n_history`` rolling window) is reconstructed
turn-by-turn so the encoder sees the same input shape it did during
training.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import torch

from ..aggregator import CrossEntityAttention
from ..encoder import FleetEncoder, PlanetEncoder, PlanetEntityEncoder
from ..featurizer import FleetTracker
from ..featurizer.inference import featurize_observation
from ..pretrain.target_rank import (
    C_AGG,
    PFI_OWNER_SELF,
    TargetRanker,
    TargetRankerStack,
)


def load_target_ranker_stack(
    ckpt_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TargetRankerStack, dict]:
    """Reconstruct the full stack from a saved ``target_rank_best.pt``.

    Returns ``(stack, config)`` where ``config`` is the saved config
    dict so callers can read e.g. ``max_planets`` / ``max_fleets`` /
    ``n_history`` without guessing.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    for k in ("fleet_encoder", "planet_encoder", "entity_encoder", "cross",
              "target_ranker"):
        if k not in ckpt:
            raise KeyError(
                f"{ckpt_path} missing '{k}' state — not a target_rank ckpt."
            )
    cfg = ckpt.get("config") or {}
    d_model = int(cfg.get("d_model", 64))
    d_rank = int(cfg.get("d_rank", 128))
    n_heads = int(cfg.get("n_heads", 4))
    head_hidden = int(cfg.get("head_hidden", 128))
    head_num_layers = int(cfg.get("head_num_layers", 3))

    fenc = FleetEncoder(d_model=d_model); fenc.load_state_dict(ckpt["fleet_encoder"])
    penc = PlanetEncoder(d_model=d_model); penc.load_state_dict(ckpt["planet_encoder"])
    eenc = PlanetEntityEncoder(d_model=d_model); eenc.load_state_dict(ckpt["entity_encoder"])
    cross = CrossEntityAttention(d_model=d_model); cross.load_state_dict(ckpt["cross"])
    ranker = TargetRanker(
        d_model=d_model, c_agg=C_AGG, d_rank=d_rank, n_heads=n_heads,
        mlp_hidden=head_hidden, mlp_layers=head_num_layers,
    )
    ranker.load_state_dict(ckpt["target_ranker"])

    stack = TargetRankerStack(
        fleet_encoder=fenc, planet_encoder=penc, entity_encoder=eenc,
        cross=cross, target_ranker=ranker,
    ).to(device).eval()
    for p in stack.parameters():
        p.requires_grad_(False)
    return stack, cfg


def _stack_history(
    history: deque[dict[str, torch.Tensor]],
    n_history: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack a deque of per-turn batches into the (B, T, …) shape the
    encoder expects. Pads with zeros at the front when the deque is
    shorter than ``n_history`` (cold-start frames).

    Only the encoder-input keys are stacked; per-turn label tensors
    (``ships_arriving_within_10`` etc.) stay current-step-only to
    match the dataset's ``_STACK_KEYS`` contract.
    """
    STACK_KEYS = (
        "planet_features", "planet_mask",
        "fleet_features", "fleet_mask",
        "fleet_target_idx", "fleet_source_idx",
        "fleet_owner_slot", "fleet_ships_log", "fleet_eta_norm",
    )
    cur = history[-1]
    out: dict[str, torch.Tensor] = {}
    for k, v in cur.items():
        if k in STACK_KEYS and v.dim() >= 2:
            # Build (T, …) by walking the deque oldest-first; pad with
            # zeros for missing past frames.
            stack: list[torch.Tensor] = []
            for off in range(n_history - 1, -1, -1):
                idx = len(history) - 1 - off
                if idx < 0:
                    stack.append(torch.zeros_like(v[0] if v.dim() >= 1 else v))
                else:
                    stack.append(history[idx][k][0])    # drop the B=1 axis
            out[k] = torch.stack(stack, dim=0).unsqueeze(0).to(device)  # (1, T, ...)
        else:
            # Current-step tensor — already shape (B=1, ...).
            out[k] = v.to(device)
    return out


def _ensure_label_tensors(batch: dict[str, torch.Tensor], P: int) -> None:
    """Stub-in zero tensors for the per-turn labels the ranker reads.

    ``featurize_observation`` is the training-CSV → inference bridge
    and doesn't compute the ``ships_arriving_within_10`` /
    ``n_friendly_within_R_norm`` / ``n_enemy_within_R_norm`` /
    ``nearest_enemy_dist_norm`` labels — those come from
    :class:`CrossEntitySnapshotDataset` during training. At inference
    we don't need their exact values for the encoders to run; the
    ranker's ``target_scalars`` MLP simply reads zeros (and any
    structure the model learned around them is gracefully degraded).

    A follow-up could plug these in via a runtime cross-entity feature
    extractor for a slightly stronger inference signal.
    """
    if "ships_arriving_within_10" not in batch:
        batch["ships_arriving_within_10"] = torch.zeros(1, P, 4)
    if "n_friendly_within_R_norm" not in batch:
        batch["n_friendly_within_R_norm"] = torch.zeros(1, P)
    if "n_enemy_within_R_norm" not in batch:
        batch["n_enemy_within_R_norm"] = torch.zeros(1, P)
    if "nearest_enemy_dist_norm" not in batch:
        batch["nearest_enemy_dist_norm"] = torch.zeros(1, P)
    # Stub src/tgt_valid as all-False; TargetRankerStack._build_masks
    # will fall back to the owned-only / real-planet masks.
    if "src_valid" not in batch:
        batch["src_valid"] = torch.zeros(1, P, dtype=torch.bool)
    if "tgt_valid" not in batch:
        batch["tgt_valid"] = torch.zeros(1, P, dtype=torch.bool)


def score_replay(
    replay_steps: list[list[Any]],
    *,
    ckpt_path: str | Path,
    learner_slot: int = 0,
    num_players: int = 4,
    device: str | torch.device = "cpu",
) -> list[dict]:
    """Walk a replay's ``env.steps`` from the learner's perspective and
    emit per-turn target-logit payloads.

    ``replay_steps`` is the list returned by ``env.toJSON()['steps']``;
    each step is a list of seat observations.

    Returns a list of dicts:

        {
          "turn": int,
          "planets": [
            {
              "id": int,           planet_id from the obs
              "x": float, "y": float,
              "owner": int,        seat owner at this turn
              "ships": int,
              "logit": float,      raw target_logit
              "prob": float,       softmax over valid candidates
              "target_valid": bool
            },
            ...
          ],
        }
    """
    stack, cfg = load_target_ranker_stack(ckpt_path, device=device)
    n_history = int(cfg.get("n_history", 3))
    max_planets = int(cfg.get("max_planets", 64))
    max_fleets = int(cfg.get("max_fleets", 1024))

    tracker = FleetTracker()
    history: deque[dict[str, torch.Tensor]] = deque(maxlen=n_history)
    device_t = torch.device(device)

    out_steps: list[dict] = []

    for t, step in enumerate(replay_steps):
        if not step or len(step) <= learner_slot:
            continue
        seat = step[learner_slot]
        obs = seat.get("observation") if isinstance(seat, dict) else None
        if obs is None:
            continue

        # Featurize current obs into a (B=1) batch dict.
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=learner_slot,
            tracker=tracker,
            num_players=num_players,
            max_planets=max_planets,
            max_fleets=max_fleets,
            device=device,
        )
        _ensure_label_tensors(batch, max_planets)
        history.append(batch)

        stacked = _stack_history(history, n_history, device_t)

        with torch.no_grad():
            target_logits, tgt_valid = stack(stacked)         # (1, P), (1, P) bool

        logits = target_logits[0]                              # (P,)
        valid = tgt_valid[0].bool()                            # (P,)

        # Softmax restricted to valid candidates.
        masked = logits.clone()
        masked[~valid] = float("-inf")
        if valid.any():
            probs = torch.softmax(masked, dim=-1)
        else:
            probs = torch.zeros_like(logits)

        # Read positions/owners directly from the obs (more reliable than
        # back-decoding from planet_features which are normalized).
        planets_obs = obs.get("planets") or []
        payload_planets = []
        for p in planets_obs:
            pid = int(p[0])
            idx = pid_to_idx.get(pid)
            if idx is None or idx >= max_planets:
                continue
            payload_planets.append({
                "id": pid,
                "x": float(p[2]),
                "y": float(p[3]),
                "owner": int(p[1]),
                "ships": int(p[5]),
                "logit": float(logits[idx].item()),
                "prob": float(probs[idx].item()),
                "target_valid": bool(valid[idx].item()),
            })
        out_steps.append({"turn": t, "planets": payload_planets})

    return out_steps
