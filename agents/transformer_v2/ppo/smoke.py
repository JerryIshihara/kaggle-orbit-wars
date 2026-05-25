"""PPO smoke runner — N episodes vs random_v1, one update, no BC anchor.

Bypasses the PairHead-depth precondition (this is a development smoke test,
not a production run). Existing supervised ckpts use `head_n_layers = 1` and
`conditioner_n_layers = 1`; the smoke run validates the rollout / GAE / PPO
update math end-to-end even though the actor heads are shallower than the
spec requires for a real PPO run.

Usage::

    python -m agents.transformer_v2.ppo.smoke \\
        --episodes 10 \\
        --ckpt data/runs/entity/.../entity_encoder_best.pt \\
        --opponent random_v1 \\
        --device mps

What this validates:

  * The full forward through PPOActorCritic (incl. value_head on glob).
  * `sample_source_multi_target` produces well-typed Actions.
  * `project_to_env` emits valid env action triples that `plan_launch` accepts.
  * Episodes terminate; per-step rewards are computed from score deltas.
  * `compute_advantages` runs on real rollouts (not synthetic).
  * `ppo_update_local` applies a gradient step on `value_head` + the two
    PairHead output heads.

Out of scope for the smoke: BC anchor (no pair cache load), distributed
training, eval gate, archive.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from agents.heuristic.physical_v4.agent import (
    PHASE_TABLE,
    compute_surplus,
    phase_of,
)
from agents.transformer_v2.featurizer.fleet_featurizer import FleetTracker
from agents.transformer_v2.featurizer.inference import featurize_observation
from agents.transformer_v2.pretrain.entity_encoder import (
    EntityPretrainModel,
    _build_entity_self_tokens,
    _load_encoders,
    build_pair_type_ids,
)
from agents.transformer_v2.runner import (
    DEFAULT_COMET_RUN_DIR,
    DEFAULT_FLEET_RUN_DIR,
    DEFAULT_PLANET_RUN_DIR,
)

from .actor_critic import PPOActorCritic
from .gae import Episode, compute_advantages
from .learner import PPOConfig, ppo_update_local
from .loss import PPOMinibatch
from .sampler import Action, legality_masks, sample_source_multi_target


# --------------------------------------------------------------------------- #
# Per-step buffer                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class StepRecord:
    """One learner-turn record."""

    planet_features: torch.Tensor
    fleet_features: torch.Tensor
    fleet_target_idx: torch.Tensor
    fleet_source_idx: torch.Tensor
    fleet_owner_slot: torch.Tensor
    fleet_ships_log: torch.Tensor
    fleet_eta_norm: torch.Tensor
    fleet_mask: torch.Tensor
    planet_mask: torch.Tensor
    is_comet: torch.Tensor
    pair_type_ids: torch.Tensor
    # Rollout-time computed:
    pair_mask: torch.Tensor                # (P, P) bool
    source_mask: torch.Tensor               # (P,) bool
    action: Action
    value: float
    # Rewards / dones are filled later from env score history.
    reward: float = 0.0
    done: float = 0.0
    invalid_launch: int = 0
    emitted_launch: int = 0
    n_selected_targets: int = 0            # count of target_bits True; for soft cap
    invalid_reasons: list = None           # list[str] — per-target failure reasons
    score_my: int = 0           # total ships of learner at end of THIS turn
    score_enemy_max: int = 0

    def __post_init__(self):
        if self.invalid_reasons is None:
            self.invalid_reasons = []


@dataclass
class EpisodeBuffer:
    seed: int
    learner_seat: int
    steps: list[StepRecord] = field(default_factory=list)
    winner: int | None = None


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #
def load_supervised(
    ckpt_path: Path,
    device: str,
    planet_run_dir: Path | None = None,
    fleet_run_dir: Path | None = None,
    comet_run_dir: Path | None = None,
) -> tuple[EntityPretrainModel, Any, Any, Any, dict]:
    """Mirror TransformerAgent.load — but return raw pieces for PPO wrapping."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    d_model = int(cfg.get("d_model", 256))
    n_steps = int(cfg.get("n_steps", 6))
    d_pair = int(cfg["d_pair"]) if "d_pair" in cfg else 128
    entity_n_heads = int(cfg.get("entity_n_heads", 4))
    cross_n_heads = int(cfg.get("cross_n_heads", 8))
    cross_n_layers = int(cfg.get("cross_n_layers", 2))
    dual_n_heads = int(cfg.get("dual_n_heads", 4))
    conditioner_n_layers = int(cfg.get("conditioner_n_layers", 1))
    head_n_layers = int(cfg.get("head_n_layers", 1))

    fleet_enc, planet_enc, comet_enc = _load_encoders(
        fleet_run_dir or DEFAULT_FLEET_RUN_DIR,
        planet_run_dir or DEFAULT_PLANET_RUN_DIR,
        comet_run_dir or DEFAULT_COMET_RUN_DIR,
        device=device,
    )

    model = EntityPretrainModel(
        d_model=d_model, n_steps=n_steps, d_pair=d_pair,
        entity_n_heads=entity_n_heads,
        cross_n_heads=cross_n_heads,
        cross_n_layers=cross_n_layers,
        dual_n_heads=dual_n_heads,
        conditioner_n_layers=conditioner_n_layers,
        head_n_layers=head_n_layers,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.to(device), fleet_enc, planet_enc, comet_enc, cfg


# --------------------------------------------------------------------------- #
# Rollout episode via env.run                                                 #
# --------------------------------------------------------------------------- #
def make_opponent_closure(
    *,
    opponent_policy: PPOActorCritic,
    planet_enc: Any,
    fleet_enc: Any,
    comet_enc: Any,
    opponent_slot: int,
    device: str,
    max_planets: int,
    max_fleets: int,
    sigma: float,
    noop_logit_bias: float,
    num_players: int = 2,
):
    """Build an env-compatible opponent fn that samples from a FROZEN
    PPOActorCritic using the same source_multi_target_v1 contract as the
    learner. No buffering — opponent moves are not tracked for PPO.

    Use this for self-play: load a frozen snapshot of the learner (or any
    other PPOActorCritic), wrap it here, pass the returned callable as the
    opponent_fn to run_episode.
    """
    tracker = FleetTracker()
    comet_input_dim = comet_enc.input_dim
    opponent_policy.eval()

    def opp_fn(obs):
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=opponent_slot,
            tracker=tracker,
            num_players=num_players,
            max_planets=max_planets,
            max_fleets=max_fleets,
            device=device,
        )
        B, P, _ = batch["planet_features"].shape
        comet_features = torch.zeros(
            (B, P, comet_input_dim),
            device=device,
            dtype=batch["planet_features"].dtype,
        )
        comet_features[..., :18] = batch["planet_features"][..., :18]
        is_comet = batch["planet_features"][..., 0] > 0.5

        with torch.inference_mode():
            planet_tok = planet_enc(batch["planet_features"])
            comet_tok = comet_enc(comet_features)
            fleet_tok = fleet_enc(batch["fleet_features"])
            entity_self = _build_entity_self_tokens(planet_tok, comet_tok, is_comet)
            routing = {
                "fleet_target_idx": batch["fleet_target_idx"],
                "fleet_source_idx": batch["fleet_source_idx"],
                "fleet_owner_slot": batch["fleet_owner_slot"],
                "fleet_ships_log": batch["fleet_ships_log"],
                "fleet_eta_norm": batch["fleet_eta_norm"],
                "fleet_mask": batch["fleet_mask"],
            }
            pair_type_ids = build_pair_type_ids(
                batch["planet_features"], batch["planet_mask"],
            )
            out = opponent_policy(
                entity_self, fleet_tok, routing, batch["planet_mask"],
                is_comet=is_comet, pair_type_ids=pair_type_ids,
            )
            pair_logits = out["pair_logits"][0].detach().cpu()
            frac_loc = out["frac_loc"][0].detach().cpu()
            sigma_val = float(out["sigma"].item())

        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        step = int(get("step") or 0)
        _nb, defense_buffer, min_launch, _s, _fw, _et = PHASE_TABLE[phase_of(step)]

        planet_owner_rel = torch.full((P,), 99, dtype=torch.long)
        planet_surplus = torch.zeros(P, dtype=torch.float32)
        planet_exists = torch.zeros(P, dtype=torch.bool)
        slot_to_pid = [-1] * P
        planets: list = []
        if raw_planets:
            from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
            planets = [Planet(*p) for p in raw_planets]
            fleets = [Fleet(*f) for f in raw_fleets]
            enemy_fleets = [f for f in fleets if f.owner != opponent_slot and f.owner >= 0]
            for planet in planets:
                pid = int(planet.id)
                if pid in pid_to_idx:
                    idx = pid_to_idx[pid]
                    planet_exists[idx] = True
                    slot_to_pid[idx] = pid
                    planet_owner_rel[idx] = 0 if int(planet.owner) == opponent_slot else 1
                    surplus = compute_surplus(planet, enemy_fleets, defense_buffer)
                    planet_surplus[idx] = float(surplus)

        pair_mask, source_mask = legality_masks(
            planet_owner=planet_owner_rel,
            surplus=planet_surplus,
            planet_exists=planet_exists,
            min_launch=int(min_launch),
        )
        action = sample_source_multi_target(
            pair_logits, frac_loc, sigma_val,
            pair_mask=pair_mask, source_mask=source_mask,
            noop_logit_bias=noop_logit_bias,
        )

        env_moves: list[list[float]] = []
        if action.source_id < 0 or action.target_bits.sum().item() == 0:
            return env_moves
        src_idx = action.source_id
        src_pid = slot_to_pid[src_idx]
        src_planet = next((p for p in planets if int(p.id) == src_pid), None)
        if src_planet is None:
            return env_moves
        surplus_b = int(planet_surplus[src_idx].item())
        budget = max(0, surplus_b)
        selected = action.target_bits.nonzero(as_tuple=False).flatten().tolist()
        raws = torch.stack([action.frac_raw[i] for i in selected])
        weights = (raws / raws.sum().clamp_min(1e-8)).tolist()
        raw_alloc = [w * budget for w in weights]
        floored = [int(x) for x in raw_alloc]
        remainder = budget - sum(floored)
        frac_rem = sorted(
            ((raw_alloc[i] - floored[i], i) for i in range(len(floored))),
            reverse=True,
        )
        for r, idx in frac_rem[:max(0, remainder)]:
            floored[idx] += 1
        for slot_i, sel_i in enumerate(selected):
            ships = floored[slot_i]
            if ships < int(min_launch):
                continue
            tgt_pid = slot_to_pid[sel_i]
            if tgt_pid < 0:
                continue
            tgt_planet = next((p for p in planets if int(p.id) == tgt_pid), None)
            if tgt_planet is None:
                continue
            dx = tgt_planet.x - src_planet.x
            dy = tgt_planet.y - src_planet.y
            angle = math.atan2(dy, dx)
            env_moves.append([int(src_pid), float(angle), int(ships)])
        return env_moves

    return opp_fn


def _make_learner_closure(
    *,
    policy: PPOActorCritic,
    planet_enc: Any,
    fleet_enc: Any,
    comet_enc: Any,
    learner_slot: int,
    device: str,
    buffer: EpisodeBuffer,
    max_planets: int,
    max_fleets: int,
    sigma: float,
    noop_logit_bias: float,
):
    """Build an env.run-compatible agent fn that samples policy actions and
    accumulates per-step records into ``buffer``."""
    tracker = FleetTracker()
    comet_input_dim = comet_enc.input_dim
    num_players = 2

    def agent_fn(obs):
        # 1. Featurize observation to model inputs.
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=learner_slot,
            tracker=tracker,
            num_players=num_players,
            max_planets=max_planets,
            max_fleets=max_fleets,
            device=device,
        )
        B, P, _ = batch["planet_features"].shape

        # 2. Comet features stub + L0 specialists forward.
        comet_features = torch.zeros(
            (B, P, comet_input_dim),
            device=device,
            dtype=batch["planet_features"].dtype,
        )
        comet_features[..., :18] = batch["planet_features"][..., :18]
        is_comet = batch["planet_features"][..., 0] > 0.5

        with torch.no_grad():
            planet_tok = planet_enc(batch["planet_features"])
            comet_tok = comet_enc(comet_features)
            fleet_tok = fleet_enc(batch["fleet_features"])
        entity_self = _build_entity_self_tokens(planet_tok, comet_tok, is_comet)

        routing = {
            "fleet_target_idx": batch["fleet_target_idx"],
            "fleet_source_idx": batch["fleet_source_idx"],
            "fleet_owner_slot": batch["fleet_owner_slot"],
            "fleet_ships_log": batch["fleet_ships_log"],
            "fleet_eta_norm": batch["fleet_eta_norm"],
            "fleet_mask": batch["fleet_mask"],
        }
        pair_type_ids = build_pair_type_ids(
            batch["planet_features"], batch["planet_mask"],
        )

        # 3. Policy forward (no grad — rollout-time).
        with torch.inference_mode():
            out = policy(
                entity_self, fleet_tok, routing, batch["planet_mask"],
                is_comet=is_comet, pair_type_ids=pair_type_ids,
            )
            pair_logits = out["pair_logits"][0].detach().cpu()        # (P, P)
            frac_loc = out["frac_loc"][0].detach().cpu()              # (P, P)
            value = float(out["value"][0].item())
            sigma_val = float(out["sigma"].item())

        # 4. Build legality masks from the obs.
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        step = int(get("step") or 0)
        _nb, defense_buffer, min_launch, _s, _fw, _et = PHASE_TABLE[phase_of(step)]

        planet_owner_rel = torch.full((P,), 99, dtype=torch.long)
        planet_surplus = torch.zeros(P, dtype=torch.float32)
        planet_exists = torch.zeros(P, dtype=torch.bool)
        slot_to_pid = [-1] * P

        if raw_planets:
            from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
            planets = [Planet(*p) for p in raw_planets]
            fleets = [Fleet(*f) for f in raw_fleets]
            enemy_fleets = [f for f in fleets if f.owner != learner_slot and f.owner >= 0]
            for planet in planets:
                pid = int(planet.id)
                if pid in pid_to_idx:
                    idx = pid_to_idx[pid]
                    planet_exists[idx] = True
                    slot_to_pid[idx] = pid
                    owner_rel = 0 if int(planet.owner) == learner_slot else 1
                    planet_owner_rel[idx] = owner_rel
                    surplus = compute_surplus(planet, enemy_fleets, defense_buffer)
                    planet_surplus[idx] = float(surplus)

        pair_mask, source_mask = legality_masks(
            planet_owner=planet_owner_rel,
            surplus=planet_surplus,
            planet_exists=planet_exists,
            min_launch=int(min_launch),
        )

        # 5. Sample action.
        action = sample_source_multi_target(
            pair_logits, frac_loc, sigma_val,
            pair_mask=pair_mask,
            source_mask=source_mask,
            noop_logit_bias=noop_logit_bias,
        )

        # 6. Project to env moves (with allocation).
        env_moves: list[list[float]] = []
        n_invalid = 0
        n_emitted = 0
        invalid_reasons: list[str] = []
        if action.source_id >= 0 and action.target_bits.sum().item() > 0:
            src_idx = action.source_id
            src_pid = slot_to_pid[src_idx]
            src_planet = next(
                (p for p in planets if int(p.id) == src_pid), None,
            ) if raw_planets else None
            if src_planet is not None:
                surplus_b = int(planet_surplus[src_idx].item())
                budget = max(0, surplus_b)
                selected = action.target_bits.nonzero(as_tuple=False).flatten().tolist()
                raws = torch.stack([action.frac_raw[i] for i in selected])
                weights = (raws / raws.sum().clamp_min(1e-8)).tolist()
                # Largest-remainder allocation.
                raw_alloc = [w * budget for w in weights]
                floored = [int(x) for x in raw_alloc]
                remainder = budget - sum(floored)
                frac_rem = sorted(
                    ((raw_alloc[i] - floored[i], i) for i in range(len(floored))),
                    reverse=True,
                )
                for r, idx in frac_rem[:max(0, remainder)]:
                    floored[idx] += 1
                for slot_i, sel_i in enumerate(selected):
                    ships = floored[slot_i]
                    if ships < int(min_launch):
                        n_invalid += 1
                        invalid_reasons.append("min_launch")
                        continue
                    tgt_pid = slot_to_pid[sel_i]
                    if tgt_pid < 0:
                        n_invalid += 1
                        invalid_reasons.append("pad_slot")
                        continue
                    tgt_planet = next(
                        (p for p in planets if int(p.id) == tgt_pid), None,
                    )
                    if tgt_planet is None:
                        n_invalid += 1
                        invalid_reasons.append("no_planet")
                        continue
                    # Compute aim angle.
                    dx = tgt_planet.x - src_planet.x
                    dy = tgt_planet.y - src_planet.y
                    angle = math.atan2(dy, dx)
                    env_moves.append([int(src_pid), float(angle), int(ships)])
                    n_emitted += 1

        # 7. Compute per-step score (for reward shaping later).
        score_my = sum(p.ships for p in planets if int(p.owner) == learner_slot) if raw_planets else 0
        score_my += sum(f.ships for f in fleets if int(f.owner) == learner_slot) if raw_fleets else 0
        score_enemy_max = 0
        if raw_planets:
            for seat in range(num_players):
                if seat == learner_slot:
                    continue
                s = sum(p.ships for p in planets if int(p.owner) == seat)
                s += sum(f.ships for f in fleets if int(f.owner) == seat)
                score_enemy_max = max(score_enemy_max, s)

        # 8. Buffer the step.
        record = StepRecord(
            planet_features=batch["planet_features"][0].detach().cpu(),
            fleet_features=batch["fleet_features"][0].detach().cpu(),
            fleet_target_idx=batch["fleet_target_idx"][0].detach().cpu(),
            fleet_source_idx=batch["fleet_source_idx"][0].detach().cpu(),
            fleet_owner_slot=batch["fleet_owner_slot"][0].detach().cpu(),
            fleet_ships_log=batch["fleet_ships_log"][0].detach().cpu(),
            fleet_eta_norm=batch["fleet_eta_norm"][0].detach().cpu(),
            fleet_mask=batch["fleet_mask"][0].detach().cpu(),
            planet_mask=batch["planet_mask"][0].detach().cpu(),
            is_comet=is_comet[0].detach().cpu(),
            pair_type_ids=pair_type_ids[0].detach().cpu(),
            pair_mask=pair_mask,
            source_mask=source_mask,
            action=action,
            value=value,
            invalid_launch=n_invalid,
            emitted_launch=n_emitted,
            n_selected_targets=int(action.target_bits.sum().item()),
            invalid_reasons=list(invalid_reasons),
            score_my=score_my,
            score_enemy_max=score_enemy_max,
        )
        buffer.steps.append(record)
        return env_moves

    return agent_fn


def run_episode(
    *,
    policy: PPOActorCritic,
    planet_enc: Any,
    fleet_enc: Any,
    comet_enc: Any,
    seed: int,
    learner_seat: int,
    opponent_id: str | None,
    opponent_fn=None,
    device: str,
    max_planets: int,
    max_fleets: int,
    sigma: float,
    noop_logit_bias: float,
    # Soft cap on target count — penalises excess(k - k_max)^2 per step.
    target_cap_k_max: int = 4,
    target_cap_lambda: float = 0.0,
) -> EpisodeBuffer:
    """Roll out one learner episode.

    Opponent selection (exactly one of):
      * ``opponent_fn``: an env-compatible callable ``fn(obs) -> moves``,
        used as-is. For self-play, pass the result of
        :func:`make_opponent_closure` here.
      * ``opponent_id``: a registered heuristic agent id (random_v1, …),
        looked up via ``agents.Agent(id=...)``.
    """
    from kaggle_environments import make
    import agents as _agents

    buffer = EpisodeBuffer(seed=seed, learner_seat=learner_seat)

    learner_fn = _make_learner_closure(
        policy=policy, planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
        learner_slot=learner_seat, device=device, buffer=buffer,
        max_planets=max_planets, max_fleets=max_fleets,
        sigma=sigma, noop_logit_bias=noop_logit_bias,
    )
    if opponent_fn is None and opponent_id is None:
        raise ValueError("must pass exactly one of opponent_fn / opponent_id")
    if opponent_fn is not None and opponent_id is not None:
        raise ValueError("pass only one of opponent_fn / opponent_id")
    opp_fn = opponent_fn if opponent_fn is not None else _agents.Agent(id=opponent_id).fn

    fns = [None, None]
    fns[learner_seat] = learner_fn
    fns[1 - learner_seat] = opp_fn

    env = make("orbit_wars", configuration={"seed": seed} if seed is not None else {})
    env.run(fns)

    # Determine winner from final rewards.
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    ranked = [r if r is not None else float("-inf") for r in rewards]
    buffer.winner = int(max(range(len(ranked)), key=lambda i: ranked[i]))

    # Compute per-step rewards: potential-based shaping + terminal +/- 1.
    # Soft cap on excess targets per source ("Known problems" #3 mitigation):
    # subtract `target_cap_lambda * max(0, k - k_max)^2`. NEVER truncates the
    # sampled action — the penalty just shapes future logit distributions.
    # Pull per-step scores from the buffered records (already computed at
    # turn time, so they are the LEARNER-turn snapshots).
    prev_potential = 0.0
    for i, step in enumerate(buffer.steps):
        potential = float(step.score_my) - float(step.score_enemy_max)
        dense = (potential - prev_potential) / 200.0
        dense = max(-0.02, min(0.02, dense))
        excess = max(0, step.n_selected_targets - int(target_cap_k_max))
        cap_penalty = float(target_cap_lambda) * (excess * excess)
        r = dense - 0.01 * float(step.invalid_launch) - cap_penalty
        prev_potential = potential
        step.reward = r
        step.done = 0.0

    if buffer.steps:
        buffer.steps[-1].done = 1.0
        terminal = 1.0 if buffer.winner == learner_seat else (-1.0 if buffer.winner != -1 else 0.0)
        buffer.steps[-1].reward += terminal

    return buffer


# --------------------------------------------------------------------------- #
# Buffer → PPOMinibatch tensor packing                                        #
# --------------------------------------------------------------------------- #
def episodes_to_ppo(
    episodes: list[EpisodeBuffer], *, minibatch_size: int, device: str,
) -> tuple[list[Episode], list[PPOMinibatch]]:
    """Convert per-step records into Episode (for GAE) and PPOMinibatch lists.

    All tensors moved to ``device`` for the update step.
    """
    # 1. Build Episode objects (values/rewards/dones).
    ep_objs: list[Episode] = []
    for ep in episodes:
        if not ep.steps:
            continue
        ep_objs.append(Episode(
            values=torch.tensor([s.value for s in ep.steps], dtype=torch.float32),
            rewards=torch.tensor([s.reward for s in ep.steps], dtype=torch.float32),
            dones=torch.tensor([s.done for s in ep.steps], dtype=torch.float32),
        ))
    compute_advantages(ep_objs, normalize=True)

    # 2. Flatten all steps into a single training set, paired with advantages.
    flat_steps: list[StepRecord] = []
    flat_adv: list[float] = []
    flat_ret: list[float] = []
    for ep_obj, ep in zip(ep_objs, [e for e in episodes if e.steps]):
        adv = ep_obj.advantages.tolist()
        ret = ep_obj.returns.tolist()
        for s, a, r in zip(ep.steps, adv, ret):
            flat_steps.append(s)
            flat_adv.append(a)
            flat_ret.append(r)

    # 3. Stack tensors and chunk into minibatches.
    N = len(flat_steps)
    if N == 0:
        return ep_objs, []

    def _stack(attr: str) -> torch.Tensor:
        return torch.stack([getattr(s, attr) for s in flat_steps])

    feats = {
        "planet_features": _stack("planet_features").to(device),
        "fleet_features": _stack("fleet_features").to(device),
        "planet_mask": _stack("planet_mask").to(device),
        "is_comet": _stack("is_comet").to(device),
        "pair_type_ids": _stack("pair_type_ids").to(device),
    }
    # Routing dict — also stacked.
    routing_keys = ("fleet_target_idx", "fleet_source_idx", "fleet_owner_slot",
                    "fleet_ships_log", "fleet_eta_norm", "fleet_mask")
    feats["routing"] = {k: _stack(k).to(device) for k in routing_keys}

    pair_mask = _stack("pair_mask").to(device)
    source_mask = _stack("source_mask").to(device)
    target_bits = torch.stack([s.action.target_bits for s in flat_steps]).to(device)
    frac_raw = torch.stack([s.action.frac_raw for s in flat_steps]).to(device)

    # source_id_plus: 0 = NOOP, else 1+source_idx.
    src_plus = torch.tensor(
        [0 if s.action.source_id < 0 else 1 + s.action.source_id for s in flat_steps],
        dtype=torch.long, device=device,
    )

    old_logp = torch.tensor(
        [float(s.action.logprob.item()) for s in flat_steps],
        dtype=torch.float32, device=device,
    )
    adv = torch.tensor(flat_adv, dtype=torch.float32, device=device)
    ret = torch.tensor(flat_ret, dtype=torch.float32, device=device)

    # Chunk into minibatches.
    minibatches: list[PPOMinibatch] = []
    for start in range(0, N, minibatch_size):
        end = min(start + minibatch_size, N)
        idx = slice(start, end)
        # Slice the routing dict.
        feats_slice = {k: (v[idx] if isinstance(v, torch.Tensor) else
                            {kk: vv[idx] for kk, vv in v.items()})
                         for k, v in feats.items()}
        # Note: PPOActorCritic.forward expects planet_tokens / fleet_tokens
        # (post-L0). The smoke runner does NOT pre-compute L0 outputs into
        # the shard — it stores raw features and re-runs L0 at update time.
        # So we wrap the model's __call__ with a sub-wrapper that runs L0
        # first. The minibatch carries the raw-feature feats dict; the
        # train-time forward (see _PPOWithL0 below) handles L0.
        minibatches.append(PPOMinibatch(
            feats=feats_slice,
            pair_mask=pair_mask[idx],
            source_mask=source_mask[idx],
            source_id_plus=src_plus[idx],
            target_bits=target_bits[idx],
            frac_raw=frac_raw[idx],
            old_logp=old_logp[idx],
            adv=adv[idx],
            returns=ret[idx],
            noop_logit_bias=0.0,
        ))

    return ep_objs, minibatches


# --------------------------------------------------------------------------- #
# Train-time wrapper that runs L0 then PPOActorCritic                          #
# --------------------------------------------------------------------------- #
class _PPOWithL0(torch.nn.Module):
    """Wrap PPOActorCritic so its `forward(**feats)` accepts raw features.

    The PPO loss expects to call ``policy(**mb.feats)`` and get the
    `(value, pair_logits, frac_loc, ...)` dict back. PPOActorCritic.forward
    takes POST-L0 tokens (planet_tokens, fleet_tokens). This wrapper runs
    the frozen L0 specialists first.
    """

    def __init__(self, policy: PPOActorCritic, planet_enc, fleet_enc, comet_enc):
        super().__init__()
        self.policy = policy
        self.planet_enc = planet_enc
        self.fleet_enc = fleet_enc
        self.comet_enc = comet_enc
        self.value_head = policy.value_head    # for the learner's optimizer
        self.entity_model = policy.entity_model
        self.sigma = policy.sigma

    def freeze_for_phase(self, phase: int):
        return self.policy.freeze_for_phase(phase)

    def forward(self, planet_features, fleet_features, planet_mask,
                 is_comet, pair_type_ids, routing):
        # L0 frozen forward. Handles both rollout-time (B, P, dim) and
        # BC-anchor (B, T, P, dim) shapes — comet_features must match
        # planet_features' leading shape exactly.
        with torch.no_grad():
            comet_shape = list(planet_features.shape[:-1]) + [self.comet_enc.input_dim]
            comet_features = torch.zeros(
                comet_shape,
                device=planet_features.device,
                dtype=planet_features.dtype,
            )
            comet_features[..., :18] = planet_features[..., :18]
            planet_tok = self.planet_enc(planet_features)
            comet_tok = self.comet_enc(comet_features)
            fleet_tok = self.fleet_enc(fleet_features)
            entity_self = _build_entity_self_tokens(planet_tok, comet_tok, is_comet)

        return self.policy(
            entity_self, fleet_tok, routing, planet_mask,
            is_comet=is_comet, pair_type_ids=pair_type_ids,
        )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path,
                         help="Supervised PairHead .pt to bootstrap from.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--opponent", default="random_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-heads", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.10)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--noop-logit-bias", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=1024)
    args = parser.parse_args()

    device = args.device
    print(f"[smoke] device={device} ckpt={args.ckpt}", flush=True)
    t0 = time.time()
    entity_model, fleet_enc, planet_enc, comet_enc, cfg = load_supervised(
        args.ckpt, device,
    )
    print(f"[smoke] loaded ckpt in {time.time()-t0:.1f}s "
          f"(d_model={cfg.get('d_model')}, n_steps={cfg.get('n_steps')}, "
          f"conditioner_n_layers={cfg.get('conditioner_n_layers',1)}, "
          f"head_n_layers={cfg.get('head_n_layers',1)})", flush=True)

    policy = PPOActorCritic(entity_model, sigma=args.sigma).to(device)
    breakdown = policy.freeze_for_phase(0)
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[smoke] freeze_for_phase(0) → trainable={n_trainable:,} | "
          f"by group: {breakdown}", flush=True)

    # ---- Rollout ----
    episodes: list[EpisodeBuffer] = []
    t_roll = time.time()
    for i in range(args.episodes):
        seed = args.seed_base + i
        learner_seat = i % 2
        t_ep = time.time()
        try:
            ep = run_episode(
                policy=policy, planet_enc=planet_enc, fleet_enc=fleet_enc,
                comet_enc=comet_enc, seed=seed, learner_seat=learner_seat,
                opponent_id=args.opponent, device=device,
                max_planets=args.max_planets, max_fleets=args.max_fleets,
                sigma=args.sigma, noop_logit_bias=args.noop_logit_bias,
            )
        except Exception as e:
            print(f"[smoke] EPISODE {i} (seed={seed} seat={learner_seat}) FAILED: {e}",
                  flush=True)
            import traceback
            traceback.print_exc()
            return 1
        episodes.append(ep)
        n_steps = len(ep.steps)
        n_noop = sum(1 for s in ep.steps if s.action.source_id < 0)
        n_emit = sum(s.emitted_launch for s in ep.steps)
        n_inval = sum(s.invalid_launch for s in ep.steps)
        won = (ep.winner == learner_seat)
        print(f"[smoke] ep {i:02d} seed={seed} seat={learner_seat} "
              f"steps={n_steps} noop={n_noop} emit={n_emit} inval={n_inval} "
              f"winner={ep.winner} {'WIN' if won else 'lose'} ({time.time()-t_ep:.1f}s)",
              flush=True)
    print(f"[smoke] rollout total: {time.time()-t_roll:.1f}s for "
          f"{args.episodes} eps", flush=True)

    total_steps = sum(len(e.steps) for e in episodes)
    print(f"[smoke] total learner steps: {total_steps}", flush=True)

    # ---- GAE + PPO update ----
    t_gae = time.time()
    ep_objs, mbs = episodes_to_ppo(
        episodes, minibatch_size=args.minibatch_size, device=device,
    )
    print(f"[smoke] built {len(mbs)} minibatches in {time.time()-t_gae:.1f}s", flush=True)
    if not mbs:
        print("[smoke] no steps to train on; exiting.", flush=True)
        return 0

    # Wrap with L0 for training-time forward.
    train_policy = _PPOWithL0(policy, planet_enc, fleet_enc, comet_enc).to(device)

    cfg_ppo = PPOConfig(
        clip=args.clip, target_kl=args.target_kl, epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr_heads=args.lr_heads, lr_trunk=None,
        bc_coef=0.0,  # no BC anchor for smoke
    )
    t_upd = time.time()
    metrics = ppo_update_local(
        train_policy, ep_objs, mbs,
        bc_minibatch_source=lambda _n: None,
        cfg=cfg_ppo,
    )
    print(f"[smoke] PPO update done in {time.time()-t_upd:.1f}s", flush=True)

    # Pretty-print epoch metrics.
    for em in metrics["epoch_metrics"]:
        early = " EARLY-STOP" if em.get("early_stopped") else ""
        print(f"  epoch {em['epoch']}: avg_kl={em.get('avg_kl', 0):.4f} "
              f"policy_loss={em.get('policy_loss', 0):.4f} "
              f"value_loss={em.get('value_loss', 0):.4f} "
              f"entropy={em.get('entropy', 0):.3f} "
              f"clip_frac={em.get('clip_frac', 0):.3f}{early}",
              flush=True)

    print(f"[smoke] DONE total wall={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
