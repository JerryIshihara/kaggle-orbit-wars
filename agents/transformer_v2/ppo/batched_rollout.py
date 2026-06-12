"""Vectorized B-env PPO rollout — the GPU-efficient rollout.

Instead of ``env.run`` driving one game at a time (B=1 forward per step), this
drives **B orbit_wars games in lockstep** via ``env.reset``/``env.step`` and
**batches the per-step policy forward across all B** — one ``(B, …)`` GPU call
per step instead of B. orbit_wars is simultaneous-move, so every active env
needs every agent's action each step (clean lockstep; the active set only
shrinks as games finish).

When learner and opponents share the same frozen policy object, every active
seat is merged into one batched forward per step. If a separate opponent policy
is supplied, rollout falls back to two batched forwards per step: learner seats
and opponent seats.
Per-env state — ``FleetTracker`` + T-window history — is isolated per (env, seat)
and stacked into the batch; the per-env legality/sample/project/record reuses
``smoke._finalize_step`` so each shard is IDENTICAL to the single-env path.

``run_batched_episodes`` returns ``list[EpisodeBuffer]`` aligned with ``specs``.
The env (a CPU Python game engine) becomes the bottleneck before the GPU does —
raise B until env-step dominates.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from agents.transformer_v2.featurizer.fleet_featurizer import FleetTracker
from agents.transformer_v2.featurizer.inference import featurize_observation
from agents.transformer_v2.pretrain.entity_encoder import (
    _PLANET_OWNER_NEUTRAL_IDX,
    _PLANET_OWNER_START_IDX,
    _build_entity_self_tokens,
    build_pair_type_ids,
)

from .smoke import (
    _ROUTING_KEYS,
    _TEMPORAL_KEYS,
    EpisodeBuffer,
    _RolloutHistory,
    _finalize_step,
    _frame_from_batch,
)


def _oget(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def _batched_forward(policy, planet_enc, fleet_enc, comet_enc, stores, device):
    """Stack B per-seat ``store`` dicts -> (B, [T,] P/F, …) -> frozen L0 + policy.
    Returns the policy ``out`` dict with a leading batch dim of len(stores)."""
    mdl = {k: torch.stack([s[k] for s in stores], dim=0).to(device) for k in _TEMPORAL_KEYS}
    comet_features = torch.zeros(
        list(mdl["planet_features"].shape[:-1]) + [comet_enc.input_dim],
        device=device, dtype=mdl["planet_features"].dtype,
    )
    comet_features[..., :18] = mdl["planet_features"][..., :18]
    planet_tok = planet_enc(mdl["planet_features"])
    comet_tok = comet_enc(comet_features)
    fleet_tok = fleet_enc(mdl["fleet_features"])
    entity_self = _build_entity_self_tokens(planet_tok, comet_tok, mdl["is_comet"])
    routing = {k: mdl[k] for k in _ROUTING_KEYS}
    owner_slice = mdl["planet_features"][
        ..., _PLANET_OWNER_START_IDX:_PLANET_OWNER_NEUTRAL_IDX + 1
    ]
    return policy(
        entity_self, fleet_tok, routing, mdl["planet_mask"],
        is_comet=mdl["is_comet"], pair_type_ids=mdl["pair_type_ids"],
        planet_owner_oh=owner_slice,
    )


class _Seat:
    """Per (env, seat) agent state. ``records=True`` for the learner seat."""
    __slots__ = ("env_idx", "seat", "num_players", "tracker", "history",
                 "buffer", "records", "moves")

    def __init__(self, env_idx, seat, num_players, window, records):
        self.env_idx = env_idx
        self.seat = seat
        self.num_players = num_players
        self.tracker = FleetTracker()
        self.history = _RolloutHistory(window)
        self.buffer = None          # set for learner seats
        self.records = records
        self.moves: list = []        # last action's env moves


def _batched_act(seats, envs, active, policy, planet_enc, fleet_enc, comet_enc,
                 *, max_planets, max_fleets, noop_logit_bias, device,
                 select_logit_bias=0.0, contract="v2", k_max=3):
    """Featurize every ACTIVE seat, run ONE batched forward, then per-seat
    sample/project/record. Writes ``seat.moves`` for each acting seat."""
    items = []   # (seat, obs, pid_to_idx, store)
    for seat in seats:
        if not active[seat.env_idx]:
            continue
        st = envs[seat.env_idx].state[seat.seat]
        if _oget(st, "status", "ACTIVE") != "ACTIVE":
            seat.moves = []
            continue
        obs = st["observation"] if isinstance(st, dict) else st.observation
        batch, pid_to_idx = featurize_observation(
            obs, learner_slot=seat.seat, tracker=seat.tracker,
            num_players=seat.num_players, max_planets=max_planets,
            max_fleets=max_fleets, device=device,
        )
        is_comet = batch["planet_features"][..., 0] > 0.5
        pair_type = build_pair_type_ids(batch["planet_features"], batch["planet_mask"])
        step = int(_oget(obs, "step", 0) or 0)
        seat.history.push(step, _frame_from_batch(batch, is_comet, pair_type))
        store = ({k: seat.history.frames[step][k] for k in _TEMPORAL_KEYS}
                 if seat.history.window <= 1 else seat.history.stack(step))
        items.append((seat, obs, pid_to_idx, store))
    if not items:
        return
    with torch.inference_mode():
        out = _batched_forward(policy, planet_enc, fleet_enc, comet_enc,
                               [it[3] for it in items], device)
        pls = out["pair_logits"].detach().cpu()      # (n, P, P)
        fls = out["frac_loc"].detach().cpu()          # (n, P, P)
        vals = out["value"].detach().cpu()            # (n,)
        sigma_val = float(out["sigma"].item())
    for j, (seat, obs, pid_to_idx, store) in enumerate(items):
        moves, record = _finalize_step(
            obs, pid_to_idx, pair_logits=pls[j], frac_loc=fls[j],
            value=float(vals[j].item()), sigma_val=sigma_val, store=store,
            learner_slot=seat.seat, num_players=seat.num_players,
            noop_logit_bias=noop_logit_bias,
            select_logit_bias=select_logit_bias,
            contract=contract, k_max=k_max,
        )
        seat.moves = moves
        if seat.records and seat.buffer is not None:
            seat.buffer.steps.append(record)


def _finalize_episode(env, buffer, learner_seat, *, target_cap_k_max, target_cap_lambda):
    """Winner + potential-based reward shaping + terminal ±1 (mirrors
    smoke.run_episode's post-env-run block) for one finished env."""
    final = env.steps[-1]
    rewards = [_oget(s, "reward", None) for s in final]
    ranked = [r if r is not None else float("-inf") for r in rewards]
    buffer.winner = int(max(range(len(ranked)), key=lambda i: ranked[i]))
    prev_potential = 0.0
    for stp in buffer.steps:
        potential = float(stp.score_my) - float(stp.score_enemy_max)
        dense = max(-0.02, min(0.02, (potential - prev_potential) / 200.0))
        excess = max(0, stp.n_selected_targets - int(target_cap_k_max))
        cap_penalty = float(target_cap_lambda) * (excess * excess)
        stp.reward = dense - 0.01 * float(stp.invalid_launch) - cap_penalty
        stp.done = 0.0
        prev_potential = potential
    if buffer.steps:
        buffer.steps[-1].done = 1.0
        terminal = 1.0 if buffer.winner == learner_seat else (-1.0 if buffer.winner != -1 else 0.0)
        buffer.steps[-1].reward += terminal


def run_batched_episodes(
    *,
    policy,
    opponent_policy,
    planet_enc, fleet_enc, comet_enc,
    specs,                       # list[(seed, num_players, learner_seat)]
    device: str,
    max_planets: int,
    max_fleets: int,
    sigma: float,
    noop_logit_bias: float = 0.0,
    select_logit_bias: float = 0.0,
    target_cap_k_max: int = 3,
    target_cap_lambda: float = 0.0,
    history_window: int = 1,
    max_steps: int = 1200,
    contract: str = "v2",        # learner's contract; opponent seats stay v2
    select_k_max: int = 3,
    on_step=None,                # optional callback(step, n_active) for progress
) -> list[EpisodeBuffer]:
    """Run ``len(specs)`` orbit_wars games in lockstep, batching the forward.

    ``opponent_policy`` (frozen self-play snapshot) fills the non-learner seats;
    only the learner seat is recorded. Returns one EpisodeBuffer per spec.
    """
    from kaggle_environments import make

    B = len(specs)
    envs: list[Any] = []
    buffers: list[EpisodeBuffer] = []
    learner_seats: list[_Seat] = []
    opp_seats: list[_Seat] = []
    for ei, (seed, n_players, lseat) in enumerate(specs):
        env = make("orbit_wars", configuration={"seed": seed} if seed is not None else {})
        env.reset(n_players)
        envs.append(env)
        buf = EpisodeBuffer(seed=seed, learner_seat=lseat)
        buffers.append(buf)
        ls = _Seat(ei, lseat, n_players, history_window, records=True)
        ls.buffer = buf
        learner_seats.append(ls)
        for s in range(n_players):
            if s != lseat:
                opp_seats.append(_Seat(ei, s, n_players, history_window, records=False))

    active = [not envs[i].done for i in range(B)]
    env_steps = [0] * B          # learner-turns taken per env (frozen when done)
    kw = dict(max_planets=max_planets, max_fleets=max_fleets,
              noop_logit_bias=noop_logit_bias, device=device,
              select_logit_bias=select_logit_bias)
    for t in range(max_steps):
        if not any(active):
            break
        if opponent_policy is policy:
            # True self-play: one policy, all seats — both sides sample the
            # learner's contract (consistent v3-vs-v3 when contract="v3").
            _batched_act(
                learner_seats + opp_seats,
                envs, active, policy, planet_enc, fleet_enc, comet_enc, **kw,
                contract=contract, k_max=select_k_max,
            )
        else:
            # Fixed opponent: learner samples its contract; the opponent stays
            # on v2 (the frozen baseline's native contract).
            _batched_act(learner_seats, envs, active, policy, planet_enc,
                         fleet_enc, comet_enc, **kw,
                         contract=contract, k_max=select_k_max)
            _batched_act(opp_seats, envs, active, opponent_policy, planet_enc,
                         fleet_enc, comet_enc, **kw,
                         contract="v2", k_max=select_k_max)
        for ei, env in enumerate(envs):
            if not active[ei]:
                continue
            n_players = specs[ei][1]
            moves = [None] * n_players
            moves[learner_seats[ei].seat] = learner_seats[ei].moves
            for os in opp_seats:
                if os.env_idx == ei:
                    moves[os.seat] = os.moves
            env.step(moves)
            env_steps[ei] = t + 1
            if env.done:
                active[ei] = False
        if on_step is not None:
            on_step(t, list(active), list(env_steps))   # (step, per-env active flags, per-env step counts)

    for ei, env in enumerate(envs):
        _finalize_episode(env, buffers[ei], specs[ei][2],
                          target_cap_k_max=target_cap_k_max, target_cap_lambda=target_cap_lambda)
    return buffers
