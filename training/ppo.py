"""PPO self-play trainer for CNN v1.

Scaffold that implements the full loop: rollout → GAE → clipped update.
The rollout driver plays the learner policy against a mix of fixed
opponents (past snapshots + baselines) to avoid the self-play collapse
that happens with a single opponent.

PPO on a 15×50×50 conv net is not fast on CPU — expect ~1 min per
iteration with the defaults below. Shrink `episodes_per_iter` for smoke
tests. The code is correct; the compute budget is the bottleneck.
"""

from __future__ import annotations

import copy
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal

from agents import Agent
from agents.agent_cnn_v1 import BOARD_SIZE, CELL, GRID, LAUNCH_THRESHOLD, CNNv1, featurize
from kaggle_environments import make

from .bc import _planet_features
from .common import fresh_model, load_model, save_model


class StochasticPolicyAgent:
    """Wraps a CNNv1 and returns a callable compatible with env.run.

    During rollout we also record (obs, action, log_prob, value) for
    later PPO update. Reset with `.reset_trajectory()` between episodes.
    """

    def __init__(self, model: CNNv1, training: bool = True, frac_std: float = 0.2):
        self.model = model
        self.training = training
        self.frac_std = frac_std
        self.trajectory: list[dict] = []

    def reset_trajectory(self):
        self.trajectory = []

    def __call__(self, obs):
        return self._act(obs)

    def _act(self, obs):
        channels, scalars, my_planets = featurize(obs)
        moves = []
        if not my_planets:
            return moves

        with torch.no_grad():
            feat_map, scalar_feat = self.model(channels.unsqueeze(0), scalars.unsqueeze(0))
            coords = torch.tensor([[x, y] for (_, x, y, _) in my_planets], dtype=torch.float32).unsqueeze(0)
            launch, target, ship = _planet_features(self.model, channels.unsqueeze(0), scalars.unsqueeze(0), coords)
            value = self.model.value(feat_map, scalar_feat)

        launch, target, ship = launch.squeeze(0), target.squeeze(0), ship.squeeze(0)

        if self.training:
            launch_dist = Bernoulli(logits=launch)
            launch_actions = launch_dist.sample()
            target_dist = Categorical(logits=target)
            target_actions = target_dist.sample()
            frac_mean = torch.sigmoid(ship)
            frac_dist = Normal(frac_mean, self.frac_std)
            frac_actions = frac_dist.sample().clamp(0.01, 1.0)
        else:
            launch_actions = (torch.sigmoid(launch) > LAUNCH_THRESHOLD).float()
            target_actions = target.argmax(dim=-1)
            frac_actions = torch.sigmoid(ship)

        log_prob = torch.zeros(())
        if self.training:
            lp = Bernoulli(logits=launch).log_prob(launch_actions).sum()
            tp = Categorical(logits=target).log_prob(target_actions) * launch_actions
            fp = Normal(torch.sigmoid(ship), self.frac_std).log_prob(frac_actions) * launch_actions
            log_prob = lp + tp.sum() + fp.sum()

        self.trajectory.append(
            {
                "channels": channels.detach(),
                "scalars": scalars.detach(),
                "coords": coords.detach().squeeze(0),
                "launch_act": launch_actions.detach(),
                "target_act": target_actions.detach(),
                "frac_act": frac_actions.detach(),
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
            }
        )

        for i, (pid, x, y, ships) in enumerate(my_planets):
            if launch_actions[i].item() < 0.5:
                continue
            tc = int(target_actions[i].item())
            tcy, tcx = divmod(tc, GRID)
            tx = (tcx + 0.5) * CELL
            ty = (tcy + 0.5) * CELL
            n = max(1, min(int(ships), int(round(frac_actions[i].item() * ships))))
            if n < 1:
                continue
            angle = math.atan2(ty - y, tx - x)
            moves.append([pid, angle, n])
        return moves


def _play_episode(learner: StochasticPolicyAgent, opponent_fn: Callable, learner_slot: int, seed: int | None):
    config = {"seed": seed} if seed is not None else {}
    env = make("orbit_wars", configuration=config, debug=False)
    learner.reset_trajectory()

    def learner_fn(obs):  # env.run expects a plain callable, not an instance
        return learner(obs)

    players = [learner_fn, opponent_fn] if learner_slot == 0 else [opponent_fn, learner_fn]
    env.run(players)
    final_reward = env.steps[-1][learner_slot].reward or 0
    return learner.trajectory, final_reward


def _compute_gae(rewards: list[float], values: list[float], gamma: float, lam: float):
    T = len(values)
    adv = [0.0] * T
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        adv[t] = last = delta + gamma * lam * last
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


def _ppo_update(
    model: CNNv1,
    batch: dict,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    frac_std: float = 0.2,
    epochs: int = 4,
    minibatch: int = 32,
    lr: float = 3e-4,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = batch["channels"].size(0)
    old_log_prob = batch["log_prob"]
    adv = batch["adv"]
    ret = batch["ret"]

    for _ in range(epochs):
        idx = torch.randperm(N)
        for start in range(0, N, minibatch):
            sel = idx[start : start + minibatch]
            ch = batch["channels"][sel]
            sc = batch["scalars"][sel]
            co = batch["coords"][sel]
            la = batch["launch_act"][sel]
            ta = batch["target_act"][sel]
            fa = batch["frac_act"][sel]
            mk = batch["mask"][sel]

            feat_map = model.conv(ch)
            scalar_feat = model.scalar_mlp(sc)
            value = model.value(feat_map, scalar_feat)

            launch, target, ship = _planet_features(model, ch, sc, co)

            launch_lp = Bernoulli(logits=launch).log_prob(la) * mk
            target_lp = Categorical(logits=target).log_prob(ta) * la * mk
            ship_lp = Normal(torch.sigmoid(ship), frac_std).log_prob(fa) * la * mk
            new_log_prob = (launch_lp + target_lp + ship_lp).sum(dim=-1)

            ratio = torch.exp(new_log_prob - old_log_prob[sel])
            unclipped = ratio * adv[sel]
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[sel]
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = F.mse_loss(value, ret[sel])

            launch_ent = Bernoulli(logits=launch).entropy() * mk
            target_ent = Categorical(logits=target).entropy() * mk
            entropy = (launch_ent + target_ent).sum(dim=-1).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
    }


MAX_PLANETS = 16


def _pack_trajectory(traj: list[dict], final_reward: float, gamma: float, lam: float):
    if not traj:
        return None
    values = [s["value"] for s in traj]
    rewards = [0.0] * (len(traj) - 1) + [final_reward]
    adv, ret = _compute_gae(rewards, values, gamma, lam)

    N = len(traj)
    channels = torch.stack([s["channels"] for s in traj])
    scalars = torch.stack([s["scalars"] for s in traj])
    log_prob = torch.tensor([s["log_prob"] for s in traj])
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(ret, dtype=torch.float32)

    M = MAX_PLANETS
    coords = torch.zeros(N, M, 2)
    launch = torch.zeros(N, M)
    target = torch.zeros(N, M, dtype=torch.long)
    frac = torch.zeros(N, M)
    mask = torch.zeros(N, M)
    for i, s in enumerate(traj):
        n = min(M, s["coords"].size(0))
        coords[i, :n] = s["coords"][:n]
        launch[i, :n] = s["launch_act"][:n]
        target[i, :n] = s["target_act"][:n]
        frac[i, :n] = s["frac_act"][:n]
        mask[i, :n] = 1.0

    return {
        "channels": channels,
        "scalars": scalars,
        "coords": coords,
        "launch_act": launch,
        "target_act": target,
        "frac_act": frac,
        "mask": mask,
        "log_prob": log_prob,
        "adv": adv_t,
        "ret": ret_t,
    }


def train_ppo(
    iterations: int = 3,
    episodes_per_iter: int = 8,
    opponents: list[str] | None = None,
    resume_from: Path | str | None = None,
    save_to: Path | str | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    seed_start: int = 0,
    verbose: bool = True,
) -> CNNv1:
    opponents = opponents or ["random", "sniper"]
    model = load_model(resume_from) if resume_from else fresh_model()
    learner = StochasticPolicyAgent(model, training=True)

    for it in range(iterations):
        t0 = time.time()
        buffers = []
        rewards_hist = []
        for ep in range(episodes_per_iter):
            opp_id = random.choice(opponents)
            opp_fn = Agent(id=opp_id).fn
            slot = ep % 2
            traj, reward = _play_episode(
                learner, opp_fn, learner_slot=slot, seed=seed_start + it * 100 + ep
            )
            rewards_hist.append(reward)
            packed = _pack_trajectory(traj, reward, gamma, lam)
            if packed is not None:
                buffers.append(packed)

        if not buffers:
            continue

        batch = {k: torch.cat([b[k] for b in buffers], dim=0) for k in buffers[0]}
        stats = _ppo_update(model, batch)

        if verbose:
            mean_r = sum(rewards_hist) / len(rewards_hist)
            print(
                f"iter {it + 1}/{iterations}  "
                f"mean_reward={mean_r:+.2f}  "
                f"policy={stats['policy_loss']:.3f}  "
                f"value={stats['value_loss']:.3f}  "
                f"entropy={stats['entropy']:.3f}  "
                f"samples={batch['channels'].size(0)}  "
                f"({time.time() - t0:.1f}s)",
                file=sys.stderr,
            )

    model.eval()
    path = save_model(model, save_to)
    if verbose:
        print(f"saved weights: {path}", file=sys.stderr)
    return model
