"""PPO fine-tuning for transformer_v1 action decoder.

Self-play + baseline-bot PPO. Learns from scratch on top of a
BC-trained action checkpoint. Reward: terminal ±1.
"""
from __future__ import annotations

import copy
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal
from kaggle_environments import make

from .notify import DiscordNotifier, format_iter_status, make_notifier
from .pretrain.expert_action import (
    ActionTrainStack,
    CROSS_ENTITY_VALUE_HORIZONS,
    FRAC_Z_MIN,
    compute_action_log_prob,
    load_for_inference,
    set_trainable,
    build_param_groups,
    _freeze_all,
    _save_checkpoint,
)
from .featurizer import featurize_observation, FleetTracker
from .runner import TransformerAgent

PLANETS_NORM = 24.0
SHIPS_NORMALIZER = 5000.0


def compute_potential(obs, learner_slot: int, num_players: int = 4) -> float:
    """Φ(s) — planet-count differential normalised by max-possible."""
    planets = (
        obs.get("planets") if isinstance(obs, dict)
        else getattr(obs, "planets", None)
    )
    if not planets:
        return 0.0
    my_p = en_p = my_s = en_s = 0
    for p in planets:
        pid, owner, _, _, ships, _, _ = p[:7]
        if owner == learner_slot:
            my_p += 1
            my_s += ships
        elif owner >= 0:
            en_p += 1
            en_s += ships
    denom = PLANETS_NORM * max(1, num_players - 1)
    ship_term = 0.25 * (my_s - en_s) / SHIPS_NORMALIZER
    return (my_p - en_p) / denom + 0.5 * ship_term


class RunningReturnNorm:
    """EMA normalizer for scalar return targets."""

    def __init__(self, beta: float = 0.99):
        self.beta = float(beta)
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    @property
    def std(self) -> float:
        return self.var ** 0.5

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().flatten()
        if x.numel() == 0:
            return
        batch_mean = float(x.mean().item())
        batch_var = float(x.var(unbiased=False).item()) + 1e-8
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.beta * self.mean + (1.0 - self.beta) * batch_mean
            self.var = self.beta * self.var + (1.0 - self.beta) * batch_var
        self.count += 1

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + 1e-6) + self.mean

    def state_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, sd: dict[str, float | int]) -> None:
        self.mean = float(sd["mean"])
        self.var = float(sd["var"])
        self.count = int(sd["count"])


class TransformerRolloutCollector:
    """Stochastic wrapper around TransformerAgent that records trajectories."""

    def __init__(
        self,
        stack: ActionTrainStack,
        device: str = "cpu",
        max_planets: int = 64,
        max_fleets: int = 256,
        num_players: int = 4,
        record: bool = True,
        return_norm: RunningReturnNorm | None = None,
    ):
        self.agent = TransformerAgent(
            stack, device=device, deterministic=False,
            max_planets=max_planets, max_fleets=max_fleets, num_players=num_players,
        )
        self.device = device
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.num_players = num_players
        self.record = record
        self.return_norm = return_norm
        self.trajectory: list[dict] = []
        self.acted_turn_count = 0
        self.acted_empty_moves_count = 0
        self._warned_exc = False

    def reset_trajectory(self) -> None:
        self.trajectory = []
        self.acted_turn_count = 0
        self.acted_empty_moves_count = 0

    def __call__(self, obs):
        return self._act(obs)

    def _act(self, obs):
        try:
            return self._act_impl(obs)
        except Exception as e:
            if not self._warned_exc:
                self._warned_exc = True
                print(f"[TransformerRolloutCollector._act] {type(e).__name__}: {e}", flush=True)
            return []

    def _act_impl(self, obs):
        learner_slot = int(obs.get("player", 0) if isinstance(obs, dict) else obs.player)
        rollout = self.agent._predict(obs, learner_slot, return_rollout=True)

        # Build action triples for the env
        moves = []
        if rollout["acted"] and rollout["src_idx"] >= 0 and rollout["tgt_idx"] >= 0:
            pid_to_idx = rollout["pid_to_idx"]
            idx_to_pid = {i: p for p, i in pid_to_idx.items()}
            src_pid = idx_to_pid.get(rollout["src_idx"])
            tgt_pid = idx_to_pid.get(rollout["tgt_idx"])
            if src_pid is not None and tgt_pid is not None:
                moves = TransformerAgent._target_to_moves(
                    src_pid, tgt_pid, obs, learner_slot,
                    frac_override=rollout.get("frac"),
                )

        if self.record and rollout["acted"]:
            self.acted_turn_count += 1
            if not moves:
                self.acted_empty_moves_count += 1

        if self.record:
            # Keep rollout storage isolated from per-tick inference state.
            obs_batch = {k: v.clone() for k, v in rollout["batch"].items()}
            value = float(rollout["value"])
            if self.return_norm is not None:
                value = float(
                    self.return_norm.denormalize(torch.tensor(value)).item()
                )
            self.trajectory.append({
                "obs": obs_batch,
                "obs_raw": copy.deepcopy(obs),
                "acted": int(rollout["acted"]),
                "src_idx": int(rollout["src_idx"]),
                "tgt_idx": int(rollout["tgt_idx"]),
                "frac": float(rollout["frac"]),
                "frac_z": float(rollout["frac_z"]),
                "value": value,
                "old_log_prob": float(rollout["log_prob"]),
                "src_legal_mask": rollout["src_legal_mask"].clone(),
                "tgt_legal_mask": rollout["tgt_legal_mask"].clone(),
                "pid_to_idx": copy.deepcopy(rollout["pid_to_idx"]),
                "acted_logit": float(rollout.get("acted_logit", 0.0)),
            })

        return moves


# ── Episode runner ─────────────────────────────────────────────────
_WARNED_SHORT_GAME = [False]


def _as_kaggle_fn(callable_obj):
    """Wrap a callable instance in a plain ``def`` function.

    kaggle_environments inspects ``agent.__code__.co_argcount`` to decide how
    many args to pass. Bare class instances have no ``__code__`` attribute, so
    kaggle defaults to ``(observation, configuration)``; the resulting
    TypeError is silently swallowed and the agent is treated as a no-op. Wrap
    here so dispatch always sees a function with a known argcount of 1.
    """
    def _fn(obs):
        return callable_obj(obs)
    return _fn


def _episode_telemetry(env, learner_slot: int, trajectory: list[dict] | None = None) -> dict:
    """Per-episode behaviour counters for the learner.

    Walks ``env.steps``:

    * fleets_launched — count of new own-fleet IDs first appearing in flight.
    * fleets_disappeared — count of own-fleet IDs that left the in-flight
      set between consecutive steps. Includes useful arrivals
      (capture / reinforce) AND env-side destruction (sun collision,
      out-of-bounds, planet sweep, repelled in combat). The env's
      ``orbit_wars.py`` removes fleets via ``fleets_to_remove`` for
      these causes; the obs only shows the post-removal set, so the
      cause cannot be disambiguated from observation alone.
    * fleets_in_flight_at_end — own fleets still un-resolved at the
      terminal step (definitionally useless: never arrived).
    * fleet_ships_in_flight_at_end — total ships locked in those.
    * captures — planet-ownership transitions to ``learner_slot``.
    * lost — planet-ownership transitions away from ``learner_slot``.
    * final_planets_owned — owned planets at the terminal step.
    * final_ships — planet+fleet ships for learner at terminal.
    * episode_length — len(env.steps).

    If ``trajectory`` is provided, also computes:
    * fleets_due_to_exploration — count of acted=1 trajectory steps
      where ``sigmoid(acted_logit) < 0.5`` (the Bernoulli sample beat
      the argmax — i.e., the launch happened because of stochastic
      exploration, not because the policy's own preference said launch).
    """
    fleet_ids_prev: set = set()
    owners_prev: list[int] | None = None
    fleets_launched = 0
    fleets_disappeared = 0
    captures = 0
    lost = 0

    for step in env.steps:
        obs = step[0].observation
        planets = list(getattr(obs, "planets", []) or [])
        fleets = list(getattr(obs, "fleets", []) or [])

        my_fleet_ids = {f[0] for f in fleets if len(f) >= 2 and f[1] == learner_slot}
        fleets_launched += len(my_fleet_ids - fleet_ids_prev)
        fleets_disappeared += len(fleet_ids_prev - my_fleet_ids)
        fleet_ids_prev = my_fleet_ids

        cur_owners = [p[1] if len(p) >= 2 else -1 for p in planets]
        if owners_prev is not None and len(owners_prev) == len(cur_owners):
            for prev, cur in zip(owners_prev, cur_owners):
                if prev == cur:
                    continue
                if cur == learner_slot:
                    captures += 1
                elif prev == learner_slot:
                    lost += 1
        owners_prev = cur_owners

    final = env.steps[-1][0].observation
    fp = list(getattr(final, "planets", []) or [])
    ff = list(getattr(final, "fleets", []) or [])
    planets_owned = sum(1 for p in fp if len(p) >= 2 and p[1] == learner_slot)
    ships_planet = sum(p[5] for p in fp if len(p) >= 6 and p[1] == learner_slot)
    own_in_flight = [f for f in ff if len(f) >= 7 and f[1] == learner_slot]
    fleets_in_flight_at_end = len(own_in_flight)
    ships_fleet = sum(f[6] for f in own_in_flight)

    # Exploration-driven launches (need trajectory to compute).
    fleets_due_to_exploration = 0
    if trajectory:
        import math
        for t in trajectory:
            if int(t.get("acted", 0)) != 1:
                continue
            logit = float(t.get("acted_logit", 0.0))
            # sigmoid(logit) < 0.5  <=>  logit < 0
            if logit < 0.0:
                fleets_due_to_exploration += 1

    return {
        "fleets_launched": int(fleets_launched),
        "fleets_disappeared": int(fleets_disappeared),
        "fleets_in_flight_at_end": int(fleets_in_flight_at_end),
        "fleet_ships_in_flight_at_end": int(ships_fleet),
        "fleets_due_to_exploration": int(fleets_due_to_exploration),
        "captures": int(captures),
        "lost": int(lost),
        "final_planets_owned": int(planets_owned),
        "final_ships": int(ships_planet + ships_fleet),
        "episode_length": int(len(env.steps)),
    }


def _play_episode(
    learner: TransformerRolloutCollector,
    opponent_fn,
    learner_slot: int,
    seed: int | None,
    env_config: dict | None = None,
):
    config = {}
    if seed is not None:
        config["seed"] = seed
    if env_config:
        config.update(env_config)
    env = make("orbit_wars", configuration=config, debug=False)
    learner.reset_trajectory()

    def learner_fn(obs):
        return learner(obs)

    opp_kaggle_fn = _as_kaggle_fn(opponent_fn)
    players = [learner_fn, opp_kaggle_fn] if learner_slot == 0 else [opp_kaggle_fn, learner_fn]
    env.run(players)
    final_state = env.steps[-1]
    final_reward = final_state[learner_slot].reward or 0
    all_rewards = [s.reward if s.reward is not None else 0 for s in final_state]

    if len(env.steps) < 50 and not _WARNED_SHORT_GAME[0]:
        _WARNED_SHORT_GAME[0] = True
        statuses = [getattr(s, "status", "?") for s in final_state]
        infos = [getattr(s, "info", None) for s in final_state]
        print(
            f"[short game] steps={len(env.steps)} learner_slot={learner_slot} "
            f"statuses={statuses} infos={infos}",
            flush=True,
        )
    telemetry = _episode_telemetry(env, learner_slot, trajectory=learner.trajectory)
    return learner.trajectory, final_reward, env, all_rewards, telemetry


# ── GAE ────────────────────────────────────────────────────────────
def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Generalized Advantage Estimation."""
    T = len(values)
    adv = [0.0] * T
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        adv[t] = last = delta + gamma * lam * last
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


def _n_step_return(
    rewards: list[float],
    values: list[float],
    t: int,
    n: int,
    gamma: float,
) -> float:
    """Discounted n-step return for one timestep."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(values) != len(rewards):
        raise ValueError("values and rewards must have the same length")

    T = len(rewards)
    total = 0.0
    discount = 1.0
    end = min(T, t + n)
    for i in range(t, end):
        total += discount * float(rewards[i])
        discount *= gamma
    if t + n < T:
        total += discount * float(values[t + n])
    return total


def _n_step_returns(
    rewards: list[float],
    values: list[float],
    n: int,
    gamma: float,
) -> list[float]:
    """Discounted n-step return for each timestep in a trajectory."""
    return [
        _n_step_return(rewards, values, t, n, gamma)
        for t in range(len(rewards))
    ]


def _explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """EV = 1 - Var(target - pred) / Var(target).

    Returns ``float('nan')`` when ``Var(target) < 1e-8`` (e.g. all-tied
    episodes) rather than a misleading 0.0 or divide-by-zero.
    """
    with torch.no_grad():
        target_var = target.var(unbiased=False)
        if float(target_var.item()) < 1e-8:
            return float("nan")
        err_var = (target - pred).var(unbiased=False)
        return float((1.0 - err_var / (target_var + 1e-8)).item())


# ── Trajectory packing ─────────────────────────────────────────────
def _pack_trajectory(
    traj: list[dict],
    final_reward: float,
    gamma: float,
    lam: float,
    learner_slot: int,
    it: int,
    shaping_coef: float,
    shaping_decay_iters: int,
    num_players: int = 4,
    return_norm: RunningReturnNorm | None = None,
    return_norms_h: dict[int, RunningReturnNorm] | None = None,
) -> dict | None:
    if not traj:
        return None
    T = len(traj)
    values = [s["value"] for s in traj]
    if shaping_decay_iters > 0:
        shape = shaping_coef * max(0.0, 1.0 - it / shaping_decay_iters)
    else:
        shape = shaping_coef
    phis = [
        compute_potential(s["obs_raw"], learner_slot, num_players)
        for s in traj
    ]
    shaping_rewards = []
    for t, phi in enumerate(phis):
        phi_next = phis[t + 1] if t + 1 < T else 0.0
        shaping_rewards.append(shape * (gamma * phi_next - phi))
    rewards = list(shaping_rewards)
    rewards[T - 1] += float(final_reward)
    adv, ret = compute_gae(rewards, values, gamma, lam)
    ret_t = torch.tensor(ret, dtype=torch.float32)
    if return_norm is not None:
        return_norm.update(ret_t)
    ret_h = {
        k: torch.tensor(_n_step_returns(rewards, values, k, gamma), dtype=torch.float32)
        for k in CROSS_ENTITY_VALUE_HORIZONS
    }
    if return_norms_h is not None:
        for k, ret_h_t in ret_h.items():
            if k in return_norms_h:
                return_norms_h[k].update(ret_h_t)
    return {
        "old_log_prob": torch.tensor([s["old_log_prob"] for s in traj], dtype=torch.float32),
        "adv": torch.tensor(adv, dtype=torch.float32),
        "ret": ret_t,
        "reward": torch.tensor(rewards, dtype=torch.float32),
        "acted": torch.tensor([s["acted"] for s in traj], dtype=torch.float32),
        "src_idx": torch.tensor([s["src_idx"] for s in traj], dtype=torch.long),
        "tgt_idx": torch.tensor([s["tgt_idx"] for s in traj], dtype=torch.long),
        "frac": torch.tensor([s["frac"] for s in traj], dtype=torch.float32),
        "frac_z": torch.tensor([s["frac_z"] for s in traj], dtype=torch.float32),
        "src_legal_mask": torch.stack([s["src_legal_mask"].bool() for s in traj], dim=0),
        "tgt_legal_mask": torch.stack([s["tgt_legal_mask"].bool() for s in traj], dim=0),
        "obs_batches": [s["obs"] for s in traj],
        "obs_raw": [s["obs_raw"] for s in traj],
        "shaping_reward_mean": sum(shaping_rewards) / max(1, T),
        "phi_start": phis[0] if phis else 0.0,
        "phi_end": phis[-1] if phis else 0.0,
        **{f"ret_h{k}": v for k, v in ret_h.items()},
    }


# ── PPO update ─────────────────────────────────────────────────────
def _collate_obs_batches(
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack a list of single-step batch dicts (each with B=1) into one
    minibatch dict (B=N). All tensors are moved to ``device``.
    """
    keys = batches[0].keys()
    return {
        k: torch.cat([b[k] for b in batches], dim=0).to(device)
        for k in keys
    }


# Per-factor entropy weights for the joint factored distribution.
# Bernoulli max entropy is 0.69; categoricals over ~10 planets reach ~2.3,
# so a scalar entropy_coef under-pushes the acted/frac factors. These
# weights re-balance — boost Bernoulli (push p_acted away from 0/1) and
# the frac Normal (push σ up so launches aren't all near 0.5).
ENTROPY_W_ACTED = 5.0
ENTROPY_W_SRC = 1.0
ENTROPY_W_TGT = 1.0
ENTROPY_W_FRAC = 3.0

# Effective floor on the Normal-frac σ. Was 0.05 — too tight; frac
# samples cluster at the loc, so frac is essentially deterministic
# at sigmoid(loc). 0.30 keeps frac exploration alive without breaking
# log-prob / kl computations.
FRAC_STD_MIN = 0.30
FRAC_STD_MAX = 1.0


def _action_log_prob_entropy(
    out: dict[str, torch.Tensor],
    acted: torch.Tensor,        # (N,) float 0/1
    src_idx: torch.Tensor,      # (N,) long
    tgt_idx: torch.Tensor,      # (N,) long
    frac_z: torch.Tensor,       # (N,) float — pre-sigmoid Normal sample from rollout
    src_legal_mask: torch.Tensor,   # (N, P) bool — True where legal at rollout time
    tgt_legal_mask: torch.Tensor,   # (N, P) bool
    frac_log_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint log-prob and entropy of the factored action distribution
    (Bernoulli * Categorical * Categorical * Normal-on-frac-logit),
    vectorised over the leading batch dim. Mirrors
    :func:`compute_action_log_prob` but works on a whole minibatch and
    re-applies the rollout-time legal masks so the new distribution
    matches the one the action was sampled from.

    `frac_z` is the pre-sigmoid Normal sample stored in the rollout
    (see :class:`TransformerAgent._predict`). Used directly as the
    Normal observation; no `logit()` reconstruction needed.
    """
    acted_logit = out["expert_acted_logit"]          # (N,)
    src_logits_raw = out["source_planet_logits"]     # (N, P)
    tgt_logits_raw = out["target_planet_logits"]     # (N, P)
    frac_logits = out["frac_logits"]                 # (N, P)

    neg_inf = torch.finfo(src_logits_raw.dtype).min
    src_masked = src_logits_raw.masked_fill(~src_legal_mask, neg_inf)
    tgt_masked = tgt_logits_raw.masked_fill(~tgt_legal_mask, neg_inf)
    # For non-acted rows (or rows whose legal mask happens to be empty),
    # fall back to the model's own existence-masked logits so Categorical
    # doesn't see all-(-inf) and produce NaN. Their log-prob and entropy
    # contributions are masked out below anyway.
    mask = acted.bool()
    mask_2d = mask.unsqueeze(-1)
    src_any = src_legal_mask.any(dim=-1, keepdim=True)
    tgt_any = tgt_legal_mask.any(dim=-1, keepdim=True)
    src_logits = torch.where(mask_2d & src_any, src_masked, src_logits_raw)
    tgt_logits = torch.where(mask_2d & tgt_any, tgt_masked, tgt_logits_raw)

    acted_dist = Bernoulli(logits=acted_logit)
    src_dist = Categorical(logits=src_logits)
    tgt_dist = Categorical(logits=tgt_logits)

    log_p_acted = acted_dist.log_prob(acted)                          # (N,)

    # Use safe (non-negative) indices so .log_prob and .gather don't blow
    # up on -1 sentinels in non-acted rows.
    safe_src = torch.where(mask, src_idx, torch.zeros_like(src_idx))
    safe_tgt = torch.where(mask, tgt_idx, torch.zeros_like(tgt_idx))

    log_p_src = src_dist.log_prob(safe_src)
    log_p_tgt = tgt_dist.log_prob(safe_tgt)

    frac_std = torch.clamp(frac_log_std.exp(), min=FRAC_STD_MIN, max=FRAC_STD_MAX)
    frac_loc = frac_logits.gather(1, safe_src.unsqueeze(1)).squeeze(1)
    frac_dist = Normal(loc=frac_loc, scale=frac_std)
    # Lower-truncated Normal correction: subtract log(1 - cdf(z_min))
    # to match the truncated distribution sampled in runner.py _predict.
    z_min_t = torch.full_like(frac_loc, FRAC_Z_MIN)
    p_lo = frac_dist.cdf(z_min_t)
    p_lo = torch.clamp(p_lo, max=1.0 - 1e-6)
    log_norm = torch.log1p(-p_lo)  # log(1 - p_lo) computed safely
    log_p_frac = frac_dist.log_prob(frac_z) - log_norm

    zero = torch.zeros_like(log_p_src)
    cond_log_p = log_p_src + log_p_tgt + log_p_frac
    log_prob = log_p_acted + torch.where(mask, cond_log_p, zero)

    # Per-factor weighted entropy. Bernoulli (acted) and frac Normal need
    # extra pressure since their entropies are small and their gradients
    # were getting drowned by the categoricals.
    weighted_cond_ent = (
        ENTROPY_W_SRC * src_dist.entropy()
        + ENTROPY_W_TGT * tgt_dist.entropy()
        + ENTROPY_W_FRAC * frac_dist.entropy()
    )
    entropy = (
        ENTROPY_W_ACTED * acted_dist.entropy()
        + torch.where(mask, weighted_cond_ent, zero)
    )
    return log_prob, entropy


def factored_kl(
    out_ref: dict[str, torch.Tensor],
    out_new: dict[str, torch.Tensor],
    frac_log_std_ref: torch.Tensor,
    frac_log_std_new: torch.Tensor,
    src_legal: torch.Tensor,
    tgt_legal: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL for Bernoulli * masked Cats * source-weighted Normal."""

    def _masked_cat_kl(
        ref_logits: torch.Tensor,
        new_logits: torch.Tensor,
        legal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        any_legal = legal.any(dim=-1)
        fallback_legal = torch.zeros_like(legal)
        fallback_legal[:, 0] = True
        safe_legal = torch.where(any_legal.unsqueeze(-1), legal, fallback_legal)

        neg_inf = torch.finfo(ref_logits.dtype).min
        ref_masked = ref_logits.masked_fill(~safe_legal, neg_inf)
        new_masked = new_logits.masked_fill(~safe_legal, neg_inf)
        ref_prob = torch.softmax(ref_masked, dim=-1)
        log_ref = torch.log_softmax(ref_masked, dim=-1)
        log_new = torch.log_softmax(new_masked, dim=-1)
        kl = (ref_prob * (log_ref - log_new)).sum(dim=-1)
        kl = torch.where(any_legal, kl, torch.zeros_like(kl))
        ref_prob = torch.where(
            any_legal.unsqueeze(-1),
            ref_prob,
            torch.zeros_like(ref_prob),
        )
        return kl, ref_prob, any_legal

    p_ref = torch.sigmoid(out_ref["expert_acted_logit"])
    p_new = torch.sigmoid(out_new["expert_acted_logit"])
    q_ref = 1.0 - p_ref
    q_new = 1.0 - p_new
    kl_acted = (
        p_ref * (p_ref.clamp_min(1e-8).log() - p_new.clamp_min(1e-8).log())
        + q_ref * (q_ref.clamp_min(1e-8).log() - q_new.clamp_min(1e-8).log())
    )

    kl_src, src_prob_ref, src_any = _masked_cat_kl(
        out_ref["source_planet_logits"],
        out_new["source_planet_logits"],
        src_legal,
    )
    kl_tgt, _, tgt_any = _masked_cat_kl(
        out_ref["target_planet_logits"],
        out_new["target_planet_logits"],
        tgt_legal,
    )

    sigma_ref = torch.clamp(frac_log_std_ref.exp(), min=FRAC_STD_MIN, max=FRAC_STD_MAX).to(
        device=out_ref["frac_logits"].device,
        dtype=out_ref["frac_logits"].dtype,
    )
    sigma_new = torch.clamp(frac_log_std_new.exp(), min=FRAC_STD_MIN, max=FRAC_STD_MAX).to(
        device=out_new["frac_logits"].device,
        dtype=out_new["frac_logits"].dtype,
    )
    loc_ref = out_ref["frac_logits"]
    loc_new = out_new["frac_logits"]
    kl_frac_per_planet = (
        torch.log(sigma_new / sigma_ref)
        + (sigma_ref.pow(2) + (loc_ref - loc_new).pow(2)) / (2.0 * sigma_new.pow(2))
        - 0.5
    )
    kl_frac = (src_prob_ref * kl_frac_per_planet).sum(dim=-1)

    can_act = src_any & tgt_any
    p_ref_eff = torch.where(can_act, p_ref, torch.zeros_like(p_ref))
    return kl_acted + p_ref_eff * (kl_src + kl_tgt + kl_frac)


# ── Phase configuration table ─────────────────────────────────────────────────
# Each entry drives: which modules are trainable, per-group LRs, loss coef
# defaults, and the default iteration budget for that phase.
#
# "global_decoder" is intentionally absent from every trainable_paths list —
# it stays frozen in all PPO phases.
PPO_PHASE_CONFIGS: dict[str, dict] = {
    "warmup": {
        "trainable_paths": [
            "action_decoder.head_value",
            "action_decoder.head_value_h",
        ],
        "lr_table": {
            "action_decoder.head_value":   3e-4,
            "action_decoder.head_value_h": 3e-4,
        },
        "bc_kl_coef": 0.0,
        "shaping_coef": 0.0,
        "shaping_decay_iters": 1,   # decays immediately, effectively off
        "default_iters": 10,
        "value_only": True,
        "opponent": "physical_v4",
    },
    "policy": {
        "trainable_paths": ["action_decoder", "cross"],
        "lr_table": {
            "action_decoder": 1.5e-4,
            "cross":          5e-5,
        },
        # bc_kl_coef → 0.0: anchoring to the BC policy was actively
        # harmful — BC was a near-skip-only policy; pulling toward it
        # blocked PPO from learning to launch.
        "bc_kl_coef": 0.0,
        "shaping_coef": 1.0,
        # Extended from 15 → 50 so dense per-step Φ-shaped reward
        # remains alive across the whole policy phase. With shaping=0
        # mid-phase + terminal -1 only, advantages collapsed to ~0.
        "shaping_decay_iters": 50,
        "default_iters": 50,
        "value_only": False,
        "opponent": "physical_v4",
    },
    "encoder-half": {
        "trainable_paths": [
            "action_decoder", "cross",
            "fleet_encoder.fc2", "fleet_encoder.norm",
            "planet_encoder.scalar.fc2", "planet_encoder.traj.proj",
            "planet_encoder.gate", "planet_encoder.norm",
        ],
        "lr_table": {
            "action_decoder": 3e-4,
            "cross": 1e-4,
            "fleet_encoder.fc2": 5e-5,
            "fleet_encoder.norm": 5e-5,
            "planet_encoder.scalar.fc2": 5e-5,
            "planet_encoder.traj.proj": 5e-5,
            "planet_encoder.gate": 5e-5,
            "planet_encoder.norm": 5e-5,
        },
        "bc_kl_coef": 0.05,
        "shaping_coef": 0.5,
        "shaping_decay_iters": 10,
        "default_iters": 50,
        "value_only": False,
        "opponent": "self",
    },
    "full": {
        "trainable_paths": [
            "action_decoder", "cross",
            "fleet_encoder.fc2", "fleet_encoder.norm",
            "planet_encoder.scalar.fc2", "planet_encoder.traj.proj",
            "planet_encoder.gate", "planet_encoder.norm",
            "fleet_encoder.fc1",
            "planet_encoder.scalar.fc1",
            "planet_encoder.traj.conv1", "planet_encoder.traj.conv2",
        ],
        "lr_table": {
            "action_decoder": 3e-4,
            "cross": 1e-4,
            "fleet_encoder.fc2": 5e-5,
            "fleet_encoder.norm": 5e-5,
            "planet_encoder.scalar.fc2": 5e-5,
            "planet_encoder.traj.proj": 5e-5,
            "planet_encoder.gate": 5e-5,
            "planet_encoder.norm": 5e-5,
            "fleet_encoder.fc1": 1e-5,
            "planet_encoder.scalar.fc1": 1e-5,
            "planet_encoder.traj.conv1": 1e-5,
            "planet_encoder.traj.conv2": 1e-5,
        },
        "bc_kl_coef": 0.02,
        "shaping_coef": 0.2,
        "shaping_decay_iters": 50,
        "default_iters": 50,
        "value_only": False,
        "opponent": "self",
    },
}


def _build_optimizer(
    stack: ActionTrainStack,
    lr_table: dict[str, float],
) -> torch.optim.Optimizer:
    """Build AdamW over the currently-trainable params using per-path LR table."""
    groups = build_param_groups(stack, lr_table)
    if not groups:
        raise RuntimeError("No trainable parameters found in stack")
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def _ppo_update(
    stack: ActionTrainStack,
    batch: dict,
    opt: torch.optim.Optimizer,
    *,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    bc_kl_coef: float = 0.0,
    ref_stack: ActionTrainStack | None = None,
    epochs: int = 2,
    minibatch: int = 32,
    kl_stop: float = 0.04,
    max_grad_norm: float = 0.5,
    iteration: int | None = None,
    return_norm: RunningReturnNorm | None = None,
    return_norms_h: dict[int, RunningReturnNorm] | None = None,
    aux_v_coef: float = 0.10,
    value_only: bool = False,
):
    device = next(stack.parameters()).device

    N = batch["old_log_prob"].size(0)
    old_log_prob = batch["old_log_prob"].to(device)
    adv = batch["adv"].to(device)
    ret = batch["ret"].to(device)
    acted_t = batch["acted"].to(device)
    src_idx_t = batch["src_idx"].to(device)
    tgt_idx_t = batch["tgt_idx"].to(device)
    frac_z_t = batch["frac_z"].to(device)
    ret_h_t = {
        k: batch[f"ret_h{k}"].to(device)
        for k in CROSS_ENTITY_VALUE_HORIZONS
        if f"ret_h{k}" in batch
    }
    src_legal_t = batch["src_legal_mask"].to(device).bool()
    tgt_legal_t = batch["tgt_legal_mask"].to(device).bool()
    obs_batches: list[dict[str, torch.Tensor]] = batch["obs_batches"]

    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    else:
        adv = adv - adv.mean()

    frac_log_std = stack.action_decoder.frac_log_std
    use_bc_kl = bc_kl_coef > 0.0 and ref_stack is not None and not value_only
    if use_bc_kl:
        ref_frac_log_std = ref_stack.action_decoder.frac_log_std

    clip_frac_sum = 0.0
    approx_kl_sum = 0.0
    log_ratio_abs_sum = 0.0
    bc_kl_sum = 0.0
    n_mb = skipped = 0
    stop_early = False
    last_policy = last_value = last_entropy = last_bc_kl = 0.0
    last_approx_kl = 0.0
    horizon_loss_sum = {k: 0.0 for k in CROSS_ENTITY_VALUE_HORIZONS}
    horizon_ev_sum = {k: 0.0 for k in CROSS_ENTITY_VALUE_HORIZONS}

    for epoch_idx in range(epochs):
        if stop_early:
            break
        idx = torch.randperm(N)
        for start in range(0, N, minibatch):
            mb_idx = start // minibatch
            sel = idx[start : start + minibatch]
            sel_list = sel.tolist()

            old_lp_sub = old_log_prob[sel]
            adv_sub = adv[sel]
            ret_sub = ret[sel]
            acted_sub = acted_t[sel]
            src_sub = src_idx_t[sel]
            tgt_sub = tgt_idx_t[sel]
            frac_z_sub = frac_z_t[sel]
            ret_h_sub = {k: v[sel] for k, v in ret_h_t.items()}
            src_legal_sub = src_legal_t[sel]
            tgt_legal_sub = tgt_legal_t[sel]

            mb_obs = _collate_obs_batches(
                [obs_batches[i] for i in sel_list], device,
            )
            out = stack(mb_obs)

            if value_only:
                new_lp = old_lp_sub.detach()
                entropy = torch.zeros_like(ret_sub)
            else:
                new_lp, entropy = _action_log_prob_entropy(
                    out, acted_sub, src_sub, tgt_sub, frac_z_sub,
                    src_legal_sub, tgt_legal_sub, frac_log_std,
                )
            value_pred = out["value"]                       # (mb,)
            value_target = (
                return_norm.normalize(ret_sub)
                if return_norm is not None
                else ret_sub
            )

            if value_only:
                log_ratio = torch.zeros_like(old_lp_sub)
                ratio = torch.ones_like(old_lp_sub)
                policy_loss = torch.zeros((), device=device, dtype=value_pred.dtype)
            else:
                log_ratio = new_lp - old_lp_sub
                ratio = log_ratio.exp()
                unclipped = ratio * adv_sub
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_sub
                policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(value_pred, value_target)
            entropy_term = entropy.mean()
            log_ratio_abs_mean_t = log_ratio.abs().mean()
            if iteration == 1 and not value_only:
                log_ratio_abs_mean_f = float(log_ratio_abs_mean_t.detach().item())
                suffix = " WARNING" if log_ratio_abs_mean_f > 1e-3 else ""
                print(
                    f"[ratio-sanity] iter={iteration} epoch={epoch_idx + 1} "
                    f"minibatch={mb_idx} "
                    f"log_ratio_abs_mean={log_ratio_abs_mean_f:.8f}{suffix}",
                    flush=True,
                )
            horizon_losses: dict[int, torch.Tensor] = {}
            horizon_evs: dict[int, float] = {}
            for k, ret_h in ret_h_sub.items():
                if return_norms_h is None or k not in return_norms_h:
                    continue
                key = f"value_h{k}"
                if key not in out:
                    continue
                target_h = return_norms_h[k].normalize(ret_h)
                pred_h = out[key]
                horizon_losses[k] = F.mse_loss(pred_h, target_h)
                horizon_evs[k] = _explained_variance(pred_h.detach(), target_h.detach())

            loss = value_coef * value_loss
            if not value_only:
                loss = loss + policy_loss - entropy_coef * entropy_term
            if horizon_losses:
                aux_v = sum(horizon_losses.values())
                loss = loss + (aux_v_coef / len(horizon_losses)) * aux_v

            bc_kl_val = 0.0
            if use_bc_kl:
                with torch.no_grad():
                    ref_out = ref_stack(mb_obs)
                bc_kl = factored_kl(
                    ref_out, out, ref_frac_log_std, frac_log_std,
                    src_legal_sub, tgt_legal_sub,
                ).mean()
                loss = loss + bc_kl_coef * bc_kl
                bc_kl_val = float(bc_kl.detach().item())

            with torch.no_grad():
                if value_only:
                    approx_kl_t = torch.zeros((), device=device)
                    clip_frac_t = torch.zeros((), device=device)
                else:
                    approx_kl_t = (old_lp_sub - new_lp).mean()
                    clip_frac_t = ((ratio - 1.0).abs() > clip).float().mean()

            if not torch.isfinite(loss):
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            bad_grad = False
            for p in stack.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                skipped += 1
                continue
            torch.nn.utils.clip_grad_norm_(stack.parameters(), max_norm=max_grad_norm)
            opt.step()

            n_mb += 1
            approx_kl_f = float(approx_kl_t.item())
            clip_frac_sum += float(clip_frac_t.item())
            approx_kl_sum += approx_kl_f
            log_ratio_abs_sum += float(log_ratio_abs_mean_t.detach().item())
            bc_kl_sum += bc_kl_val
            last_policy = float(policy_loss.item())
            last_value = float(value_loss.item())
            last_entropy = float(entropy_term.item())
            last_bc_kl = bc_kl_val
            last_approx_kl = approx_kl_f
            for k, loss_h in horizon_losses.items():
                horizon_loss_sum[k] += float(loss_h.detach().item())
                horizon_ev_sum[k] += horizon_evs[k]

            if approx_kl_f > kl_stop:
                iter_label = iteration if iteration is not None else "?"
                print(
                    f"[kl-early-stop] iter={iter_label} epoch={epoch_idx + 1} "
                    f"minibatch={mb_idx} approx_kl={approx_kl_f:.6f} "
                    f"kl_stop={kl_stop:.6f}",
                    flush=True,
                )
                stop_early = True
                break

    stats = {
        "policy_loss": last_policy,
        "value_loss": last_value,
        "entropy": last_entropy,
        "bc_kl": last_bc_kl,
        "clip_frac": clip_frac_sum / max(1, n_mb),
        "approx_kl": approx_kl_sum / max(1, n_mb),
        "log_ratio_abs_mean": log_ratio_abs_sum / max(1, n_mb),
        "bc_kl_mean": bc_kl_sum / max(1, n_mb),
        "skipped_mb": skipped,
        "early_stopped": stop_early,
    }
    for k in CROSS_ENTITY_VALUE_HORIZONS:
        stats[f"value_h{k}_loss"] = horizon_loss_sum[k] / max(1, n_mb)
        stats[f"value_h{k}_ev"] = horizon_ev_sum[k] / max(1, n_mb)

    # --- Main-head explained variance in raw-reward space (T-013) ---
    # Re-run a single forward pass over the full batch using the *final*
    # model state so EV reflects post-update quality rather than an
    # epoch-averaged mix.  Minibatch-iterate to avoid OOM on large buffers.
    _pred_chunks: list[torch.Tensor] = []
    _ret_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for _start in range(0, N, minibatch):
            _mb_obs = _collate_obs_batches(
                [obs_batches[i] for i in range(_start, min(_start + minibatch, N))],
                device,
            )
            _out = stack(_mb_obs)
            _pred_norm = _out["value"].detach()          # normalised space
            if return_norm is not None:
                _pred_raw = return_norm.denormalize(_pred_norm)
            else:
                _pred_raw = _pred_norm
            _pred_chunks.append(_pred_raw)
            _ret_chunks.append(ret[_start : _start + minibatch].to(device))
    if _pred_chunks:
        _pred_all = torch.cat(_pred_chunks)
        _ret_all = torch.cat(_ret_chunks)
        stats["explained_variance"] = _explained_variance(_pred_all, _ret_all)
    else:
        stats["explained_variance"] = float("nan")

    return stats


def _freeze(snap: ActionTrainStack) -> ActionTrainStack:
    snap.eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


def _eval_winrate(
    stack: ActionTrainStack,
    opponent_id: str,
    games: int,
    seed_start: int,
    device: str,
    max_planets: int = 64,
    max_fleets: int = 256,
) -> float:
    agent = TransformerAgent(
        stack, device=device, deterministic=True,
        max_planets=max_planets, max_fleets=max_fleets,
    )
    from ..cnn_v1.eval import evaluate_agent
    from ..registry import register, _REGISTRY

    eval_id = "_transformer_v1_ppo_eval"
    if eval_id in _REGISTRY:
        del _REGISTRY[eval_id]
    # Register a plain function with co_argcount=1 — kaggle's agent dispatch
    # introspects __code__.co_argcount, and a bound method's __code__ counts
    # ``self``, which makes kaggle pass 2 args and crash silently.
    def _eval_act(obs):
        return agent.act(obs)
    register(eval_id, "transformer v1 PPO eval")(_eval_act)
    result = evaluate_agent(
        eval_id, [opponent_id], games_per=games, seed_start=seed_start, verbose=False,
    )
    if eval_id in _REGISTRY:
        del _REGISTRY[eval_id]
    return result.get(opponent_id, {}).get("win_rate", 0.0)


# ── Smoke test ────────────────────────────────────────────────────
NEXT_PPO_PHASE = {
    "warmup": "policy",
    "policy": "encoder-half",
    "encoder-half": "full",
    "full": None,  # production training; no next phase
}


def _smoke_test_episode(
    ckpt_path: Path,
    opponent_id: str,
    *,
    seed: int = 0,
    device: str | None = None,
    max_planets: int = 64,
    max_fleets: int = 256,
    min_steps: int = 30,
    frac_tol: float = 0.05,
    acted_max: float = 0.95,
) -> dict:
    """Run one deterministic episode vs ``opponent_id`` and return diagnostics.

    Returns a dict with keys:
      passed: bool
      fail_reasons: list[str]   # empty when passed
      steps: int
      mean_acted: float
      mean_frac: float          # NaN if no step had acted=1
      acted_steps: int

    Isolation: uses a fresh ``make()`` call via ``run_match``; no env-state
    leaks back into the caller (training loop).  Runs in-process — no
    subprocess needed because ``make()`` constructs an independent env
    instance each call.
    """
    import math as _math
    from utils.runner import run_match
    from ..registry import register as _register, _REGISTRY as _REG

    smoke_id = "_transformer_v1_smoke"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    acted_record: list[int] = []
    frac_record: list[float] = []

    try:
        stack = load_for_inference(Path(ckpt_path), device=torch.device(device))
        agent = TransformerAgent(
            stack, device=device, deterministic=True,
            max_planets=max_planets, max_fleets=max_fleets,
        )

        def _smoke_act(obs):
            try:
                learner_slot = int(
                    obs.get("player", 0) if isinstance(obs, dict) else obs.player
                )
                rollout = agent._predict(obs, learner_slot, return_rollout=True)
                acted_record.append(int(rollout["acted"]))
                frac_record.append(float(rollout["frac"]))
                moves = []
                if rollout["acted"] and rollout["src_idx"] >= 0 and rollout["tgt_idx"] >= 0:
                    pid_to_idx = rollout["pid_to_idx"]
                    idx_to_pid = {i: p for p, i in pid_to_idx.items()}
                    src_pid = idx_to_pid.get(rollout["src_idx"])
                    tgt_pid = idx_to_pid.get(rollout["tgt_idx"])
                    if src_pid is not None and tgt_pid is not None:
                        moves = TransformerAgent._target_to_moves(
                            src_pid, tgt_pid, obs, learner_slot,
                            frac_override=rollout.get("frac"),
                        )
                return moves
            except Exception:
                acted_record.append(0)
                frac_record.append(0.0)
                return []

        if smoke_id in _REG:
            del _REG[smoke_id]
        _register(smoke_id, "transformer v1 smoke test")(_smoke_act)

        match = run_match([smoke_id, opponent_id], seed=seed)

        if smoke_id in _REG:
            del _REG[smoke_id]

        steps = len(match.env.steps)
        acted_steps = sum(acted_record)
        mean_acted = acted_steps / max(1, len(acted_record))

        acted_fracs = [frac_record[i] for i, a in enumerate(acted_record) if a == 1]
        if acted_fracs:
            mean_frac = sum(acted_fracs) / len(acted_fracs)
        else:
            mean_frac = _math.nan

        fail_reasons: list[str] = []
        if steps < min_steps:
            fail_reasons.append(f"too_short steps={steps}")
        if acted_steps < 1:
            fail_reasons.append("no_acted_step")
        if mean_acted >= acted_max:
            fail_reasons.append(f"acted_saturated mean_acted={mean_acted:.4f}")
        if not _math.isnan(mean_frac) and abs(mean_frac - 0.5) <= frac_tol:
            fail_reasons.append(f"frac_degenerate mean_frac={mean_frac:.4f}")
        if _math.isnan(mean_frac):
            if "no_acted_step" not in fail_reasons:
                fail_reasons.append("no_acted_step")

        return {
            "passed": len(fail_reasons) == 0,
            "fail_reasons": fail_reasons,
            "steps": steps,
            "mean_acted": mean_acted,
            "mean_frac": mean_frac,
            "acted_steps": acted_steps,
        }

    except Exception as exc:
        # Clean up registry entry if it was registered before the exception
        if smoke_id in _REG:
            del _REG[smoke_id]
        return {
            "passed": False,
            "fail_reasons": [f"exception: {type(exc).__name__}: {exc}"],
            "steps": 0,
            "mean_acted": 0.0,
            "mean_frac": float("nan"),
            "acted_steps": 0,
        }


# ── Main PPO training loop ─────────────────────────────────────────
def train_ppo(
    resume_action_pt: str | Path | None = None,
    out_dir: str | Path | None = None,
    iterations: int | None = None,
    episodes_per_iter: int = 16,
    phase: str = "policy",
    warmup_iters: int = 3,
    self_play_ratio: float | None = None,
    baseline_bot_ratio: float | None = None,
    baseline_bot_id: str = "physical_v4",
    eval_every: int = 10,
    eval_games: int = 20,
    eval_opponent: str = "physical_v4",
    gamma: float = 0.99,
    lam: float = 0.95,
    shaping_coef: float | None = None,
    shaping_decay_iters: int | None = None,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    bc_kl_coef: float | None = None,
    aux_v_coef: float = 0.10,
    clip: float = 0.2,
    kl_stop: float = 0.04,
    max_grad_norm: float = 0.5,
    epochs: int = 2,
    minibatch: int = 32,
    max_planets: int = 64,
    max_fleets: int = 256,
    num_players: int = 4,
    seed_start: int = 0,
    device: str | None = None,
    verbose: bool = True,
    smoke_test: bool = True,
    smoke_force: bool = False,
    smoke_opponent: str = "physical_v4",
    smoke_seed: int = 0,
    smoke_min_steps: int = 30,
    no_discord: bool = False,
    notifier: DiscordNotifier | None = None,
) -> Path:
    out_dir = Path(out_dir or f"data/runs/ppo_transformer_{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if phase not in PPO_PHASE_CONFIGS:
        raise ValueError(
            f"unsupported PPO phase {phase!r}; "
            f"choose from {sorted(PPO_PHASE_CONFIGS.keys())}"
        )
    cfg = PPO_PHASE_CONFIGS[phase]
    value_only: bool = cfg["value_only"]
    # Per-phase defaults; CLI overrides take priority when explicitly provided.
    if bc_kl_coef is None:
        bc_kl_coef = cfg["bc_kl_coef"]
    if shaping_coef is None:
        shaping_coef = cfg["shaping_coef"]
    if shaping_decay_iters is None:
        shaping_decay_iters = cfg["shaping_decay_iters"]
    if iterations is None:
        iterations = cfg["default_iters"]
    run_iterations = iterations
    # Resolve opponent mode from per-phase curriculum; CLI override takes priority.
    _phase_opponent = cfg["opponent"]  # "physical_v4" | "self"
    _curriculum_spr = 1.0 if _phase_opponent == "self" else 0.0
    if self_play_ratio is None:
        self_play_ratio = _curriculum_spr
    # baseline_bot_ratio is not actively used post-T-022; default to 0.0 when not set.
    if baseline_bot_ratio is None:
        baseline_bot_ratio = 0.0
    # Reflect runtime overrides: if --baseline-bot-id was set to something
    # other than the phase default (e.g. random_v1 for curriculum), log
    # against the actual opponent rather than the phase's nominal one.
    if _phase_opponent == "self":
        opponent_mode = "self"
    elif baseline_bot_id != "physical_v4":
        opponent_mode = baseline_bot_id
    else:
        opponent_mode = _phase_opponent

    if resume_action_pt is None:
        from .paths import ACTION_RUNS_DIR
        candidates: list[Path] = []
        for run_dir in sorted(ACTION_RUNS_DIR.iterdir()):
            if not run_dir.is_dir():
                continue
            for name in ("action_best.pt", "action_last.pt"):
                p = run_dir / name
                if p.exists():
                    candidates.append(p)
                    break
        if not candidates:
            raise FileNotFoundError("No action checkpoint found for PPO resume")
        resume_action_pt = candidates[-1]
    resume_action_pt = Path(resume_action_pt)

    stack = load_for_inference(resume_action_pt, device=device)
    return_norm = RunningReturnNorm()
    return_norms_h = {
        k: RunningReturnNorm() for k in CROSS_ENTITY_VALUE_HORIZONS
    }
    try:
        ckpt = torch.load(resume_action_pt, map_location="cpu")
        if isinstance(ckpt, dict) and "return_norm" in ckpt:
            return_norm.load_state_dict(ckpt["return_norm"])
            if verbose:
                print(
                    f"[ppo] restored return_norm "
                    f"mean={return_norm.mean:.6f} var={return_norm.var:.6f} "
                    f"count={return_norm.count}",
                    flush=True,
                )
        else:
            print(
                "[ppo] checkpoint missing return_norm; initializing fresh.",
                flush=True,
            )
        return_norms_payload = ckpt.get("return_norms") if isinstance(ckpt, dict) else None
        if isinstance(return_norms_payload, dict):
            if "main" in return_norms_payload and not (
                isinstance(ckpt, dict) and "return_norm" in ckpt
            ):
                return_norm.load_state_dict(return_norms_payload["main"])
            missing_h = []
            for k, norm_h in return_norms_h.items():
                key = f"h{k}"
                if key in return_norms_payload:
                    norm_h.load_state_dict(return_norms_payload[key])
                else:
                    missing_h.append(key)
            if missing_h:
                print(
                    f"[ppo] checkpoint missing horizon return norms {missing_h}; "
                    "initializing those fresh.",
                    flush=True,
                )
        else:
            print(
                "[ppo] checkpoint missing horizon return_norms; initializing fresh.",
                flush=True,
            )
    except Exception as e:
        print(
            f"[ppo] could not load return_norm ({type(e).__name__}: {e}); "
            "initializing fresh.",
            flush=True,
        )
    stack.train()
    _freeze_all(stack)
    set_trainable(stack, freeze=[], unfreeze=cfg["trainable_paths"])

    total_params = sum(p.numel() for p in stack.parameters())
    trainable_params = sum(p.numel() for p in stack.parameters() if p.requires_grad)
    if verbose:
        print(
            f"[ppo] device: {device}  total: {total_params/1e6:.2f}M  "
            f"trainable: {trainable_params/1e6:.2f}M  "
            f"phase: {phase}  ckpt: {resume_action_pt.name}",
            flush=True,
        )
    # Phase entry banner (always printed, not just when verbose)
    print(
        f"[phase] entering {phase}  trainable={trainable_params}  "
        f"bc_kl={bc_kl_coef}  shaping={shaping_coef}",
        flush=True,
    )

    # Discord notifier setup
    # If a notifier is passed in (chain mode), the chain owns start()/finish().
    # If None (single-phase mode), we create our own and own the full lifecycle.
    _owns_notifier = notifier is None
    if _owns_notifier:
        notifier = make_notifier(out_dir, disabled=no_discord)
        _start_header = (
            f"PPO Training Start\n"
            f"phase={phase}  opponent={opponent_mode}  iters={run_iterations}\n"
            f"ckpt={resume_action_pt.name}  out_dir={out_dir}"
        )
        notifier.start(_start_header)

    # Warmup
    try:
        with torch.no_grad():
            warm_obs = {"player": 0, "step": 0, "planets": [], "fleets": [],
                        "initial_planets": [], "comet_planet_ids": [], "comets": [],
                        "angular_velocity": 0.05}
            batch, _ = featurize_observation(
                warm_obs, learner_slot=0, tracker=FleetTracker(),
                num_players=num_players, max_planets=max_planets,
                max_fleets=max_fleets, device=device,
            )
            stack(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    except Exception as e:
        print(f"[ppo] warmup failed: {e}", flush=True)

    learner = TransformerRolloutCollector(
        stack, device=device, max_planets=max_planets,
        max_fleets=max_fleets, num_players=num_players, record=True,
        return_norm=return_norm,
    )
    opt = _build_optimizer(stack, cfg["lr_table"])
    ref_stack: ActionTrainStack | None = None
    if bc_kl_coef > 0.0 and not value_only:
        ref_stack = _freeze(copy.deepcopy(stack))
        if verbose:
            print(
                f"[ppo] BC KL regularizer enabled (coef={bc_kl_coef}); "
                "frozen reference policy taken from initial checkpoint.",
                flush=True,
            )
    best_metric = float("-inf")
    best_state = copy.deepcopy(stack.state_dict())
    # Per-phase checkpoint names; unsuffixed copies maintained for back-compat.
    best_ckpt = out_dir / f"action_ppo_{phase}_best.pt"
    last_ckpt = out_dir / f"action_ppo_{phase}_last.pt"
    log_path = out_dir / "ppo_log.jsonl"
    log_f = log_path.open("a")
    train_start = time.time()
    # Most recent eval winrate (persists across iters within a phase)
    _last_eval_wr: float | None = None
    config = {
        "iterations": run_iterations, "episodes_per_iter": episodes_per_iter,
        "requested_iterations": iterations,
        "phase": phase,
        "warmup_iters": warmup_iters,
        "gamma": gamma, "lam": lam,
        "shaping_coef": shaping_coef,
        "shaping_decay_iters": shaping_decay_iters,
        "aux_v_coef": aux_v_coef,
        "self_play_ratio": self_play_ratio,
        "baseline_bot_ratio": baseline_bot_ratio,
        "baseline_bot_id": baseline_bot_id,
        "contextual_action_decoder": (
            stack.action_decoder.__class__.__name__ == "ContextualActionDecoder"
        ),
    }

    def norm_checkpoint_extra() -> dict[str, Any]:
        return {
            "return_norm": return_norm.state_dict(),
            "return_norms": {
                "main": return_norm.state_dict(),
                **{
                    f"h{k}": norm_h.state_dict()
                    for k, norm_h in return_norms_h.items()
                },
            },
        }

    try:
        for it in range(run_iterations):
            t0 = time.time()
            buffers: list[dict] = []
            rewards_hist: list[float] = []
            opp_rewards_hist: list[float] = []   # opponent's reward — needed to detect shared-top draws
            shaping_hist: list[float] = []
            phi_start_hist: list[float] = []
            phi_end_hist: list[float] = []
            ep_lens_hist: list[int] = []      # episode lengths (steps), for episodes_short_pct
            telem_hist: list[dict] = []       # per-episode behaviour counters from _episode_telemetry
            ep_types = {"self": 0, "baseline": 0}

            for ep in range(episodes_per_iter):
                r = random.random()
                if r < self_play_ratio:
                    self_opp = TransformerRolloutCollector(
                        stack, device=device, max_planets=max_planets,
                        max_fleets=max_fleets, num_players=num_players,
                        record=False,
                    )
                    opp_fn = self_opp
                    ep_types["self"] += 1
                    opp_kind = "self"
                else:
                    from ..registry import Agent

                    opp_fn = Agent(baseline_bot_id)
                    ep_types["baseline"] += 1
                    opp_kind = "baseline"

                slot = ep % 2
                traj, reward, env, all_rewards, telem = _play_episode(
                    learner, opp_fn, learner_slot=slot,
                    seed=seed_start + it * 1000 + ep,
                )
                rewards_hist.append(reward)
                telem_hist.append(telem)
                # orbit_wars is always 2-player; learner_slot ∈ {0,1}, opp is the other.
                opp_slot = 1 - slot
                opp_reward = all_rewards[opp_slot] if opp_slot < len(all_rewards) else 0
                opp_rewards_hist.append(float(opp_reward))
                ep_lens_hist.append(len(traj))
                packed = _pack_trajectory(
                    traj, reward, gamma, lam,
                    learner_slot=slot,
                    it=it,
                    shaping_coef=shaping_coef,
                    shaping_decay_iters=shaping_decay_iters,
                    num_players=num_players,
                    return_norm=return_norm,
                    return_norms_h=return_norms_h,
                )
                if packed is not None:
                    buffers.append(packed)
                    shaping_hist.append(float(packed["shaping_reward_mean"]))
                    phi_start_hist.append(float(packed["phi_start"]))
                    phi_end_hist.append(float(packed["phi_end"]))
                if verbose:
                    ws = sum(1 for lr, ore in zip(rewards_hist, opp_rewards_hist) if lr > 0.5 and ore < 0.5)
                    ls = sum(1 for lr in rewards_hist if lr < -0.5)
                    ds = sum(1 for lr, ore in zip(rewards_hist, opp_rewards_hist) if lr > 0.5 and ore > 0.5)
                    print(
                        f"  iter {it+1:3d} ep {ep+1:2d}/{episodes_per_iter}  "
                        f"opp={opp_kind:8s}  slot={slot}  r={reward:+.0f}/{opp_reward:+.0f}  "
                        f"W/L/D={ws}/{ls}/{ds}  "
                        f"acted_empty={learner.acted_empty_moves_count}/"
                        f"{learner.acted_turn_count}",
                        flush=True,
                    )

            if not buffers:
                print(f"[ppo] iter {it+1}: EMPTY buffers; skipping", flush=True)
                continue

            tensor_keys = [
                "old_log_prob", "adv", "ret",
                "reward", "acted", "src_idx", "tgt_idx", "frac", "frac_z",
                "src_legal_mask", "tgt_legal_mask",
                *[f"ret_h{k}" for k in CROSS_ENTITY_VALUE_HORIZONS],
            ]
            batch = {
                k: torch.cat([b[k] for b in buffers], dim=0) for k in tensor_keys
            }
            batch["obs_batches"] = [
                step for buf in buffers for step in buf["obs_batches"]
            ]
            batch["obs_raw"] = [
                step for buf in buffers for step in buf["obs_raw"]
            ]
            stats = _ppo_update(
                stack, batch, opt,
                clip=clip,
                value_coef=value_coef, entropy_coef=entropy_coef,
                bc_kl_coef=bc_kl_coef, ref_stack=ref_stack,
                epochs=epochs, minibatch=minibatch,
                kl_stop=kl_stop,
                max_grad_norm=max_grad_norm,
                iteration=it + 1,
                return_norm=return_norm,
                return_norms_h=return_norms_h,
                aux_v_coef=aux_v_coef,
                value_only=value_only,
            )

            has_nan = any(not torch.isfinite(p).all() for p in stack.parameters())
            if has_nan:
                print(f"  iter {it+1}: NaN params; reverting to best", flush=True)
                stack.load_state_dict(best_state)

            eval_wr = None
            if (it + 1) % eval_every == 0:
                eval_wr = _eval_winrate(
                    stack, eval_opponent, eval_games,
                    seed_start=100_000 + it, device=device,
                    max_planets=max_planets, max_fleets=max_fleets,
                )
                _last_eval_wr = eval_wr
                if eval_wr > best_metric:
                    best_metric = eval_wr
                    best_state = copy.deepcopy(stack.state_dict())
                    _save_checkpoint(
                        best_ckpt,
                        stack=stack, epoch=it + 1,
                        stage_index=None, stage_name="ppo",
                        config=config,
                        extra=norm_checkpoint_extra(),
                    )
                    # Back-compat unsuffixed copy
                    shutil.copy2(best_ckpt, out_dir / "action_ppo_best.pt")

            mean_r = sum(rewards_hist) / max(1, len(rewards_hist))
            # True draw = both players got reward +1 (shared top score in
            # orbit_wars). The env never assigns reward=0, so the legacy
            # `len - wins - losses` formula always yielded 0.
            wins = sum(1 for lr, ore in zip(rewards_hist, opp_rewards_hist) if lr > 0.5 and ore < 0.5)
            losses = sum(1 for lr in rewards_hist if lr < -0.5)
            draws = sum(1 for lr, ore in zip(rewards_hist, opp_rewards_hist) if lr > 0.5 and ore > 0.5)
            iter_time = time.time() - t0
            total_elapsed = time.time() - train_start
            shaped_reward_mean = sum(shaping_hist) / max(1, len(shaping_hist))
            phi_start_mean = sum(phi_start_hist) / max(1, len(phi_start_hist))
            phi_end_mean = sum(phi_end_hist) / max(1, len(phi_end_hist))

            # --- Per-iter metrics dict (fed to stdout summary + Discord notifier) ---
            # frac_log_std: raw trainable log-std parameter (T-020)
            frac_log_std_val = float(stack.action_decoder.frac_log_std.item())
            # frac_log_std_exp: exp() of the above (legacy alias kept in JSONL)
            frac_log_std_exp = float(stack.action_decoder.frac_log_std.exp().item())

            # mean_sigmoid_z / frac_sample_std: aggregate sigmoid(frac_z) over acted=1 steps
            # Collected from the packed trajectory buffers.
            _sig_z_vals: list[float] = []
            for _buf in buffers:
                _acted_mask = _buf["acted"].bool()  # (T,)
                if _acted_mask.any():
                    _frac_z = _buf["frac_z"][_acted_mask]  # acted steps only
                    _sig_z_vals.extend(torch.sigmoid(_frac_z).tolist())
            mean_sigmoid_z: float | None = (
                sum(_sig_z_vals) / len(_sig_z_vals) if _sig_z_vals else None
            )
            # frac_sample_mean / frac_sample_std (T-020)
            frac_sample_mean: float | None = mean_sigmoid_z
            if len(_sig_z_vals) >= 2:
                _mu = sum(_sig_z_vals) / len(_sig_z_vals)
                frac_sample_std: float | None = (
                    sum((v - _mu) ** 2 for v in _sig_z_vals) / len(_sig_z_vals)
                ) ** 0.5
            else:
                frac_sample_std = None

            # shaping_coef_eff: effective shaping coefficient after decay at this iter
            if shaping_decay_iters > 0:
                shaping_coef_eff = shaping_coef * max(0.0, 1.0 - it / shaping_decay_iters)
            else:
                shaping_coef_eff = float(shaping_coef)

            # eta_seconds_remaining: coarse linear estimate
            eta_seconds_remaining = iter_time * (run_iterations - it - 1)

            # explained_variance: EV of main head_value in raw-reward space (T-013).
            # NaN from _explained_variance (Var(ret) ≈ 0) is stored as None so
            # downstream JSON serialisation and format strings are consistent.
            def _safe_ev(v: Any) -> float | None:
                if v is None:
                    return None
                try:
                    f = float(v)
                    return None if f != f else f  # NaN → None
                except (TypeError, ValueError):
                    return None

            explained_variance: float | None = _safe_ev(stats.get("explained_variance"))

            # explained_variance_h{k}: per-horizon EV (T-020, T-016)
            explained_variance_h: dict[int, float | None] = {
                k: _safe_ev(stats.get(f"value_h{k}_ev"))
                for k in CROSS_ENTITY_VALUE_HORIZONS
            }

            # log_ratio_abs_mean: logged every iter (T-020, T-005)
            log_ratio_abs_mean: float = stats["log_ratio_abs_mean"]

            # episodes_short_pct: fraction of episodes shorter than smoke_min_steps (T-020)
            if ep_lens_hist:
                _n_short = sum(1 for ln in ep_lens_hist if ln < smoke_min_steps)
                episodes_short_pct: float | None = round(
                    100.0 * _n_short / len(ep_lens_hist), 2
                )
            else:
                episodes_short_pct = None

            # Per-episode behaviour aggregates (means across all eps in iter)
            if telem_hist:
                def _mean(key: str) -> float:
                    return sum(t[key] for t in telem_hist) / len(telem_hist)
                game_metrics = {
                    "mean_fleets_launched": round(_mean("fleets_launched"), 2),
                    "mean_fleets_disappeared": round(_mean("fleets_disappeared"), 2),
                    "mean_fleets_in_flight_at_end": round(_mean("fleets_in_flight_at_end"), 2),
                    "mean_fleet_ships_in_flight_at_end": round(_mean("fleet_ships_in_flight_at_end"), 1),
                    "mean_fleets_due_to_exploration": round(_mean("fleets_due_to_exploration"), 2),
                    "mean_captures": round(_mean("captures"), 2),
                    "mean_lost": round(_mean("lost"), 2),
                    "mean_final_planets_owned": round(_mean("final_planets_owned"), 2),
                    "mean_final_ships": round(_mean("final_ships"), 1),
                    "mean_episode_length": round(_mean("episode_length"), 1),
                }
            else:
                game_metrics = {
                    "mean_fleets_launched": None, "mean_fleets_disappeared": None,
                    "mean_fleets_in_flight_at_end": None,
                    "mean_fleet_ships_in_flight_at_end": None,
                    "mean_fleets_due_to_exploration": None,
                    "mean_captures": None, "mean_lost": None,
                    "mean_final_planets_owned": None, "mean_final_ships": None,
                    "mean_episode_length": None,
                }

            metrics = {
                "iter": it + 1,
                "phase": phase,
                "opponent_mode": opponent_mode,
                "winrate_vs_physical_v4": _last_eval_wr,
                "mean_reward": round(mean_r, 4),
                "wins": wins, "losses": losses, "draws": draws,
                "shaped_reward_mean": round(shaped_reward_mean, 6),
                "explained_variance": explained_variance,
                "main_explained_variance": explained_variance,  # T-026 alias
                "approx_kl": round(stats["approx_kl"], 4),
                "entropy": round(stats["entropy"], 4),
                "frac_log_std": round(frac_log_std_val, 6),
                "frac_log_std_exp": round(frac_log_std_exp, 6),
                "frac_sample_mean": (
                    round(frac_sample_mean, 6) if frac_sample_mean is not None else None
                ),
                "frac_sample_std": (
                    round(frac_sample_std, 6) if frac_sample_std is not None else None
                ),
                "mean_sigmoid_z": (
                    round(mean_sigmoid_z, 6) if mean_sigmoid_z is not None else None
                ),
                "log_ratio_abs_mean": round(log_ratio_abs_mean, 8),
                "episodes_short_pct": episodes_short_pct,
                "shaping_coef_eff": round(shaping_coef_eff, 6),
                "iter_seconds": round(iter_time, 2),
                "eta_seconds_remaining": round(eta_seconds_remaining, 1),
                **{
                    f"explained_variance_h{k}": (
                        round(explained_variance_h[k], 6)
                        if explained_variance_h[k] is not None else None
                    )
                    for k in CROSS_ENTITY_VALUE_HORIZONS
                },
                **game_metrics,
            }

            # --- Per-iter ASCII stdout summary line ---
            def _fv(v: Any, fmt: str = ".3f") -> str:
                if v is None:
                    return "nan"
                try:
                    return format(float(v), fmt)
                except (TypeError, ValueError):
                    return "nan"

            def _eta_str(secs: float | None) -> str:
                if secs is None:
                    return "?"
                s = int(max(0, secs))
                if s >= 3600:
                    return f"{s // 3600}h{(s % 3600) // 60:02d}m"
                elif s >= 60:
                    return f"{s // 60}m{s % 60:02d}s"
                return f"{s}s"

            _opp_short = opponent_mode[:4] if len(opponent_mode) >= 4 else opponent_mode
            _evh_str = "/".join(
                _fv(explained_variance_h.get(k), ".2f")
                for k in CROSS_ENTITY_VALUE_HORIZONS
            )
            # --- Per-iter categorized stdout block ---
            # Single legacy line kept first for grep-friendliness; detail blocks follow.
            print(
                f"[ppo it={it+1:4d}/{run_iterations:<4d} "
                f"phase={phase:<12s} "
                f"opp={_opp_short:<4s} "
                f"wld={wins}/{losses}/{draws} "
                f"rew={_fv(mean_r, '+.3f')} "
                f"wr={_fv(_last_eval_wr, '.2f')} "
                f"ev={_fv(explained_variance, '.2f')} "
                f"evh={_evh_str} "
                f"kl={_fv(stats['approx_kl'], '.3f')} "
                f"lr={_fv(log_ratio_abs_mean, '.2e')} "
                f"ent={_fv(stats['entropy'], '.2f')} "
                f"fls={_fv(frac_log_std_val, '.3f')} "
                f"fsm={_fv(frac_sample_mean, '.2f')} "
                f"fss={_fv(frac_sample_std, '.2f')} "
                f"shp={_fv(shaping_coef_eff, '.2f')} "
                f"eps_short={_fv(episodes_short_pct, '.1f')}% "
                f"dt={iter_time:5.0f}s "
                f"ETA={_eta_str(eta_seconds_remaining)}]",
                flush=True,
            )
            print(
                "  [game]   "
                f"wld={wins}/{losses}/{draws} "
                f"wr={_fv(_last_eval_wr, '.2f')} "
                f"fleets_launched={_fv(game_metrics.get('mean_fleets_launched'), '.1f')} "
                f"in_flight_end={_fv(game_metrics.get('mean_fleets_in_flight_at_end'), '.1f')} "
                f"explo_launches={_fv(game_metrics.get('mean_fleets_due_to_exploration'), '.1f')} "
                f"captured={_fv(game_metrics.get('mean_captures'), '.1f')} "
                f"lost={_fv(game_metrics.get('mean_lost'), '.1f')} "
                f"planets_end={_fv(game_metrics.get('mean_final_planets_owned'), '.2f')} "
                f"ships_end={_fv(game_metrics.get('mean_final_ships'), '.0f')} "
                f"len={_fv(game_metrics.get('mean_episode_length'), '.0f')} "
                f"eps_short={_fv(episodes_short_pct, '.1f')}%",
                flush=True,
            )
            print(
                "  [reward] "
                f"rew={_fv(mean_r, '+.3f')} "
                f"shape_r={_fv(shaped_reward_mean, '+.4f')} "
                f"shp_coef={_fv(shaping_coef_eff, '.2f')} "
                f"phi={_fv(phi_start_mean, '+.3f')}->{_fv(phi_end_mean, '+.3f')} "
                f"ret_std={_fv(return_norm.std, '.3f')}",
                flush=True,
            )
            print(
                "  [policy] "
                f"kl={_fv(stats['approx_kl'], '.4f')}{'*' if stats['early_stopped'] else ''} "
                f"|log_ratio|={_fv(log_ratio_abs_mean, '.2e')} "
                f"clip={_fv(stats['clip_frac'], '.3f')} "
                f"ent={_fv(stats['entropy'], '.2f')} "
                f"pi_loss={_fv(stats['policy_loss'], '+.3f')} "
                f"bc_kl={_fv(stats['bc_kl_mean'], '.3f')}",
                flush=True,
            )
            print(
                "  [value]  "
                f"ev={_fv(explained_variance, '.2f')} "
                f"evh={_evh_str} "
                f"v_loss={_fv(stats['value_loss'], '.3f')}",
                flush=True,
            )
            print(
                "  [frac]   "
                f"log_std={_fv(frac_log_std_val, '.3f')} "
                f"sigma_eff={_fv(min(max(frac_log_std_exp, FRAC_STD_MIN), FRAC_STD_MAX), '.3f')} "
                f"sample_mean={_fv(frac_sample_mean, '.3f')} "
                f"sample_std={_fv(frac_sample_std, '.3f')}",
                flush=True,
            )
            print(
                "  [time]   "
                f"dt={iter_time:.0f}s elapsed={total_elapsed:.0f}s ETA={_eta_str(eta_seconds_remaining)} "
                f"samples={int(batch['old_log_prob'].size(0))} "
                f"opp_mix={dict(ep_types)}",
                flush=True,
            )

            row = {
                "iter": it + 1,
                "phase": phase,
                "opponent_mode": opponent_mode,
                "mean_reward": round(mean_r, 4),
                "wins": wins, "losses": losses, "draws": draws,
                "policy_loss": round(stats["policy_loss"], 4),
                "value_loss": round(stats["value_loss"], 4),
                "entropy": round(stats["entropy"], 4),
                "bc_kl": round(stats["bc_kl_mean"], 4),
                "clip_frac": round(stats["clip_frac"], 4),
                "approx_kl": round(stats["approx_kl"], 4),
                "log_ratio_abs_mean": round(stats["log_ratio_abs_mean"], 8),
                "samples": int(batch["old_log_prob"].size(0)),
                "episode_mix": dict(ep_types),
                "shaped_reward_mean": round(shaped_reward_mean, 6),
                "phi_start_mean": round(phi_start_mean, 6),
                "phi_end_mean": round(phi_end_mean, 6),
                "return_norm_std": round(return_norm.std, 6),
                "return_norm_count": return_norm.count,
                "early_stopped": bool(stats["early_stopped"]),
                "time_s": round(iter_time, 2),
                "elapsed_s": round(total_elapsed, 2),
                # Extended metrics from the per-iter metrics dict (T-013, T-016, T-020, T-026)
                "winrate_vs_physical_v4": _last_eval_wr,
                "explained_variance": explained_variance,
                "main_explained_variance": explained_variance,   # T-026 alias; back-compat
                "frac_log_std": round(frac_log_std_val, 6),
                "frac_log_std_exp": round(frac_log_std_exp, 6),
                "frac_sample_mean": (
                    round(frac_sample_mean, 6) if frac_sample_mean is not None else None
                ),
                "frac_sample_std": (
                    round(frac_sample_std, 6) if frac_sample_std is not None else None
                ),
                "mean_sigmoid_z": (
                    round(mean_sigmoid_z, 6) if mean_sigmoid_z is not None else None
                ),
                "episodes_short_pct": episodes_short_pct,
                "shaping_coef_eff": round(shaping_coef_eff, 6),
                "eta_seconds_remaining": round(eta_seconds_remaining, 1),
                **game_metrics,
            }
            for k in CROSS_ENTITY_VALUE_HORIZONS:
                row[f"value_h{k}_loss"] = round(stats[f"value_h{k}_loss"], 6)
                row[f"value_h{k}_ev"] = round(stats[f"value_h{k}_ev"], 6)
                row[f"explained_variance_h{k}"] = (
                    round(explained_variance_h[k], 6)
                    if explained_variance_h.get(k) is not None else None
                )
            if eval_wr is not None:
                row["eval_winrate"] = round(eval_wr, 3)
                row["best_winrate"] = round(best_metric, 3)

            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

            # --- Discord per-iter update ---
            notifier.update(
                format_iter_status(
                    metrics,
                    run_dir=out_dir,
                    phase=phase,
                    opponent_mode=opponent_mode,
                    total_iters_in_run=run_iterations,
                )
            )

            if verbose:
                es = "*" if stats["early_stopped"] else " "
                ep_str = (
                    f"{{\"self\": {ep_types['self']}, "
                    f"\"baseline\": {ep_types['baseline']}}}"
                )
                eval_str = f"  eval={eval_wr:.2f}(best={best_metric:.2f})" if eval_wr is not None else ""
                print(
                    f"iter {it+1:3d}/{run_iterations}  "
                    f"W/L/D={wins}/{losses}/{draws}  "
                    f"r={mean_r:+.3f}  "
                    f"pi={stats['policy_loss']:+.2f}  v={stats['value_loss']:.3f}  "
                    f"ent={stats['entropy']:.1f}  kl={stats['approx_kl']:.3f}{es} "
                    f"|lr|={stats['log_ratio_abs_mean']:.2e}  "
                    f"clip={stats['clip_frac']:.2f}  "
                    f"opp={ep_str}  "
                    f"shape_r={shaped_reward_mean:+.4f}  "
                    f"phi={phi_start_mean:+.3f}->{phi_end_mean:+.3f}  "
                    f"ret_std={return_norm.std:.3f}  "
                    f"h10={stats.get('value_h10_loss', 0.0):.3f}  "
                    f"h50={stats.get('value_h50_loss', 0.0):.3f}  "
                    f"t={iter_time:.1f}s elapsed={total_elapsed:.0f}s"
                    f"{eval_str}",
                    flush=True,
                )
    finally:
        log_f.close()
        if _owns_notifier:
            _total_wall = time.time() - train_start
            _best_wr_str = f"{best_metric:.3f}" if best_metric > float("-inf") else "n/a"
            notifier.finish(
                f"PPO Training Finished\n"
                f"phase={phase}  iters_completed={run_iterations}\n"
                f"best_winrate={_best_wr_str}  wall_time={_total_wall:.0f}s\n"
                f"run_dir={out_dir}"
            )

    _save_checkpoint(
        last_ckpt,
        stack=stack, epoch=run_iterations,
        stage_index=None, stage_name="ppo",
        config=config,
        extra=norm_checkpoint_extra(),
    )
    # Back-compat unsuffixed copy of last checkpoint
    shutil.copy2(last_ckpt, out_dir / "action_ppo_last.pt")
    stack.load_state_dict(best_state)
    # best ckpt might not exist for very short runs (eval never triggered)
    if not best_ckpt.exists():
        _save_checkpoint(
            best_ckpt,
            stack=stack, epoch=run_iterations,
            stage_index=None, stage_name="ppo",
            config=config,
            extra=norm_checkpoint_extra(),
        )
        # Back-compat unsuffixed copy
        shutil.copy2(best_ckpt, out_dir / "action_ppo_best.pt")
    if smoke_test:
        next_phase = NEXT_PPO_PHASE.get(phase)
        next_phase_label = next_phase if next_phase is not None else "<final>"
        result = _smoke_test_episode(
            best_ckpt,
            opponent_id=smoke_opponent,
            seed=smoke_seed,
            device=str(device),
            max_planets=max_planets,
            max_fleets=max_fleets,
            min_steps=smoke_min_steps,
        )
        if result["passed"]:
            print(
                f"[smoke-pass]  next phase: {next_phase_label}  "
                f"steps={result['steps']}  mean_acted={result['mean_acted']:.3f}  "
                f"mean_frac={result['mean_frac']:.3f}",
                flush=True,
            )
        else:
            reasons = "; ".join(result["fail_reasons"])
            print(
                f"[smoke-fail]  phase={phase}  reasons=[{reasons}]  "
                f"steps={result['steps']}  mean_acted={result['mean_acted']:.3f}  "
                f"mean_frac={result['mean_frac']:.3f}",
                flush=True,
            )
            if not smoke_force:
                raise SystemExit(2)
            print("[smoke-pass-forced]  next phase: " + next_phase_label, flush=True)

    print(f"[ppo] done. best_metric={best_metric:.3f}  checkpoints: {out_dir}", flush=True)
    return best_ckpt


@torch.no_grad()
def _log_ratio_abs_mean_for_batch(
    stack: ActionTrainStack,
    batch: dict,
    device: torch.device,
) -> float:
    obs = _collate_obs_batches(batch["obs_batches"], device)
    out = stack(obs)
    new_lp, _ = _action_log_prob_entropy(
        out,
        batch["acted"].to(device),
        batch["src_idx"].to(device),
        batch["tgt_idx"].to(device),
        batch["frac_z"].to(device),
        batch["src_legal_mask"].to(device).bool(),
        batch["tgt_legal_mask"].to(device).bool(),
        stack.action_decoder.frac_log_std,
    )
    old_lp = batch["old_log_prob"].to(device)
    return float((new_lp - old_lp).abs().mean().item())


def _ratio_sanity_check(
    ckpt_path: str | Path | None,
    device: str | None,
) -> float:
    from agents.random_v1.agent import random_valid_agent

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)
    stack = load_for_inference(ckpt_path or _default_action_ckpt(), device=torch_device)
    learner = TransformerRolloutCollector(stack, device=device, record=True)
    traj, reward, _env, _all_rewards, _telem = _play_episode(
        learner, random_valid_agent, learner_slot=0, seed=17,
    )
    packed = _pack_trajectory(
        traj, reward, gamma=0.99, lam=0.95,
        learner_slot=0, it=0,
        shaping_coef=0.0, shaping_decay_iters=50,
    )
    if packed is None:
        raise RuntimeError("ratio sanity rollout produced no trajectory")
    metric = _log_ratio_abs_mean_for_batch(stack, packed, torch_device)
    print(f"[ratio-sanity-check] log_ratio_abs_mean={metric:.8f}", flush=True)
    if metric >= 1e-4:
        raise AssertionError(f"log_ratio_abs_mean {metric:.8f} >= 1e-4")
    return metric


def _default_action_ckpt() -> Path:
    from .paths import ACTION_RUNS_DIR

    candidates: list[Path] = []
    for run_dir in sorted(ACTION_RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        for name in ("action_best.pt", "action_last.pt"):
            p = run_dir / name
            if p.exists():
                candidates.append(p)
                break
    if not candidates:
        raise FileNotFoundError("No action checkpoint found for ratio sanity check")
    return candidates[-1]


_PHASE_CHAIN = ["warmup", "policy", "encoder-half", "full"]


def _run_phase_chain(args) -> None:
    """Run all four PPO phases sequentially sharing one out_dir."""
    import argparse as _ap

    out_dir = Path(args.out_dir or f"data/runs/ppo_transformer_{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect kwargs common to every phase, excluding phase-specific ones.
    no_discord: bool = getattr(args, "no_discord", False)
    common_kwargs = {
        "out_dir": out_dir,
        "iterations": args.iterations,
        "episodes_per_iter": args.episodes_per_iter,
        "warmup_iters": args.warmup_iters,
        "self_play_ratio": args.self_play_ratio,
        "baseline_bot_ratio": args.baseline_bot_ratio,
        "baseline_bot_id": args.baseline_bot_id,
        "eval_every": args.eval_every,
        "eval_games": args.eval_games,
        "eval_opponent": args.eval_opponent,
        "gamma": args.gamma,
        "lam": args.lam,
        "shaping_coef": args.shaping_coef,
        "shaping_decay_iters": args.shaping_decay_iters,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "bc_kl_coef": args.bc_kl_coef,
        "aux_v_coef": args.aux_v_coef,
        "clip": args.clip,
        "kl_stop": args.kl_stop,
        "epochs": args.epochs,
        "minibatch": args.minibatch,
        "max_planets": args.max_planets,
        "max_fleets": args.max_fleets,
        "num_players": args.num_players,
        "seed_start": args.seed_start,
        "device": args.device,
        "verbose": args.verbose,
        "smoke_test": args.smoke_test,
        "smoke_force": args.smoke_force,
        "smoke_opponent": args.smoke_opponent,
        "smoke_seed": args.smoke_seed,
        "smoke_min_steps": args.smoke_min_steps,
        "no_discord": no_discord,
    }

    # Create one shared notifier for the entire phase chain.
    # The notifier persists its message_id to out_dir/discord_message.json so
    # each phase's update() calls land on the same pinned message.
    chain_notifier = make_notifier(out_dir, disabled=no_discord)
    chain_start = time.time()
    chain_notifier.start(
        f"PPO Phase-Chain Start (--phase all)\n"
        f"phases={' -> '.join(_PHASE_CHAIN)}  out_dir={out_dir}"
    )

    try:
        prev_best: Path | None = None
        for i, phase in enumerate(_PHASE_CHAIN):
            print(f"\n[chain] entering phase={phase}  ({i+1}/{len(_PHASE_CHAIN)})", flush=True)
            if i == 0:
                resume_pt = args.resume_action_pt  # user-supplied or None → auto-discover
            else:
                prev_phase = _PHASE_CHAIN[i - 1]
                resume_pt = out_dir / f"action_ppo_{prev_phase}_best.pt"
                if not resume_pt.exists():
                    raise RuntimeError(
                        f"phase chain broken: expected {resume_pt} not found "
                        f"(previous phase '{prev_phase}' may not have saved a checkpoint)"
                    )
            train_ppo(phase=phase, resume_action_pt=resume_pt, notifier=chain_notifier, **common_kwargs)
    finally:
        _chain_wall = time.time() - chain_start
        chain_notifier.finish(
            f"PPO Phase-Chain Complete\n"
            f"phases={' -> '.join(_PHASE_CHAIN)}  wall_time={_chain_wall:.0f}s\n"
            f"out_dir={out_dir}"
        )

    print(f"\n[chain] all phases complete. checkpoints in {out_dir}", flush=True)


def main() -> None:
    """CLI entrypoint: `python -m agents.transformer_v1.ppo`"""
    import argparse
    p = argparse.ArgumentParser(description="transformer_v1 PPO self-play training")
    p.add_argument("--resume-action-pt", "--resume-pt", default=None,
                   dest="resume_action_pt",
                   help="Path to BC-trained / PPO action checkpoint.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: data/runs/ppo_transformer_<ts>/)")
    p.add_argument("--ratio-sanity-check", action="store_true",
                   help="Collect one rollout and assert iter-0 log-ratio sanity.")
    # iterations default=None → resolved to per-phase default (50 per phase) inside train_ppo
    p.add_argument("--iterations", type=int, default=None,
                   help="Override per-phase default iteration budget (default 50 per phase).")
    p.add_argument("--episodes-per-iter", type=int, default=16, help="Episodes per iter (default 16)")
    p.add_argument(
        "--phase",
        choices=["warmup", "policy", "encoder-half", "full", "all"],
        default="policy",
        help=(
            "Training phase; determines trainable modules, LRs, and loss coefs. "
            "warmup+policy: physical_v4 opponent; encoder-half+full: self-play opponent. "
            "Use 'all' to run all four phases sequentially (warmup→policy→encoder-half→full). "
            "Health criterion: ev (explained_variance) > 0.3 after the warmup phase is the "
            "success criterion for warmup — if ev stays below 0.3 after 3+ warmup iters, "
            "the value head is not learning and downstream policy training will be unreliable."
        ),
    )
    p.add_argument("--warmup-iters", type=int, default=3)
    # default=None → resolved to per-phase curriculum value inside train_ppo
    p.add_argument("--self-play-ratio", type=float, default=None,
                   help="Override per-phase self-play ratio (default: per-phase curriculum).")
    p.add_argument("--baseline-bot-ratio", type=float, default=None,
                   help="Override per-phase baseline bot ratio (default: per-phase curriculum).")
    p.add_argument("--baseline-bot-id", default="physical_v4")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-opponent", default="physical_v4")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    # default=None → resolved to per-phase default inside train_ppo
    p.add_argument("--shaping-coef", type=float, default=None,
                   help="Override per-phase shaping coefficient.")
    p.add_argument("--shaping-decay-iters", type=int, default=None,
                   help="Override per-phase shaping decay iters.")
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    # default=None → resolved to per-phase default inside train_ppo
    p.add_argument("--bc-kl-coef", type=float, default=None,
                   dest="bc_kl_coef",
                   help="Override per-phase BC KL coefficient.")
    p.add_argument("--aux-v-coef", type=float, default=0.10)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--kl-stop", type=float, default=0.04)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=32)
    p.add_argument("--max-planets", type=int, default=64, dest="max_planets")
    p.add_argument("--max-fleets", type=int, default=256, dest="max_fleets")
    p.add_argument("--num-players", type=int, default=4, dest="num_players")
    p.add_argument("--device", default=None)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--verbose", action="store_true", default=True)
    # Smoke-test gate
    _BoolOpt = getattr(argparse, "BooleanOptionalAction", None)
    if _BoolOpt is not None:
        p.add_argument(
            "--smoke-test",
            action=_BoolOpt,
            default=True,
            dest="smoke_test",
            help="Run a deterministic smoke-test episode after training (default: true).",
        )
    else:
        p.add_argument("--smoke-test", action="store_true", default=True, dest="smoke_test")
        p.add_argument("--no-smoke-test", action="store_false", dest="smoke_test")
    p.add_argument(
        "--smoke-force",
        action="store_true",
        default=False,
        dest="smoke_force",
        help="Continue even if the smoke test fails (logs [smoke-pass-forced]).",
    )
    p.add_argument(
        "--smoke-opponent",
        default="physical_v4",
        dest="smoke_opponent",
        help="Opponent agent id for the smoke test (default: physical_v4).",
    )
    p.add_argument(
        "--smoke-seed",
        type=int,
        default=0,
        dest="smoke_seed",
        help="RNG seed for the smoke-test episode (default: 0).",
    )
    p.add_argument(
        "--smoke-min-steps",
        type=int,
        default=30,
        dest="smoke_min_steps",
        help="Minimum episode length for the smoke test to pass (default: 30).",
    )
    p.add_argument(
        "--no-discord",
        action="store_true",
        default=False,
        dest="no_discord",
        help=(
            "Disable Discord notifier even when DISCORD_TOKEN_CLAUDE etc. are set. "
            "Useful for local tests and dry runs."
        ),
    )
    args = p.parse_args()
    if args.ratio_sanity_check:
        _ratio_sanity_check(args.resume_action_pt, args.device)
        return
    delattr(args, "ratio_sanity_check")
    if args.phase == "all":
        _run_phase_chain(args)
        return
    train_ppo(**vars(args))


if __name__ == "__main__":
    main()
