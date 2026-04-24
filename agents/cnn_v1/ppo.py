"""Pure self-play PPO trainer for CNN v1.

No teacher, no behavior cloning. The learner plays against:
  - its current self (probability `self_play_ratio`), and
  - frozen snapshots of itself collected every `snapshot_every` iters.

Reward signal: potential-based shaping on planet-count + production
differential (`F = γΦ(s') − Φ(s)`), plus terminal ±1. Potential-based
shaping preserves optimal policy (Ng et al. 1999) so it cannot reward-hack.

Benchmark eval against `physical_v2` runs every `eval_every` iters and
gates the checkpoint — weights are only overwritten on the disk when
win rate improves over best-so-far.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal

from agents import Agent
from kaggle_environments import make

from . import agent as _cnn_mod
from .agent import (
    BOARD_SIZE,
    CELL,
    GRID,
    LAUNCH_THRESHOLD,
    NUM_CHANNELS,
    SCALAR_DIM,
    WEIGHTS_PATH,
    CNNv1,
    featurize,
)
from .bc import _planet_features
from .common import TRAIN_ROOT, fresh_model, load_model, save_model
from .eval import evaluate_agent


MAX_PLANETS = 16
MAX_PROD = 50.0  # normalizer for production-sum differential (avg ~12 planets × ~3 prod)


def compute_potential(obs, learner_slot: int) -> float:
    """Φ(s) = (my_planets − enemy_planets)/16 + 0.5·(my_prod − enemy_prod)/MAX_PROD."""
    planets = obs.get("planets") if isinstance(obs, dict) else getattr(obs, "planets", None)
    if not planets:
        return 0.0
    my_p = enemy_p = 0
    my_prod = enemy_prod = 0
    for p in planets:
        _, owner, _, _, _, _, prod = p
        if owner == learner_slot:
            my_p += 1
            my_prod += prod
        elif owner >= 0:
            enemy_p += 1
            enemy_prod += prod
    return (my_p - enemy_p) / 16.0 + 0.5 * (my_prod - enemy_prod) / MAX_PROD


class StochasticPolicyAgent:
    """Wraps a CNNv1. Records a trajectory when `record=True`."""

    def __init__(self, model: CNNv1, training: bool = True, record: bool = True):
        self.model = model
        self.device = next(model.parameters()).device
        self.training = training
        self.record = record
        self.trajectory: list[dict] = []

    def reset_trajectory(self):
        self.trajectory = []

    def __call__(self, obs):
        return self._act(obs)

    def _act(self, obs):
        try:
            return self._act_impl(obs)
        except Exception as e:
            if not getattr(self, "_warned_exc", False):
                self._warned_exc = True
                print(f"[_act] EXCEPTION: {type(e).__name__}: {e}", flush=True)
            return []

    def _act_impl(self, obs):
        channels, scalars, my_planets = featurize(obs)
        moves = []
        if not my_planets:
            return moves

        channels = channels.to(self.device)
        scalars = scalars.to(self.device)

        with torch.no_grad():
            feat_map, scalar_feat = self.model(channels.unsqueeze(0), scalars.unsqueeze(0))
            coords = torch.tensor(
                [[x, y] for (_, x, y, _) in my_planets], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            launch, target, ship = _planet_features(
                self.model, channels.unsqueeze(0), scalars.unsqueeze(0), coords
            )
            value = self.model.value(feat_map, scalar_feat)
            frac_std = float(torch.exp(self.model.frac_log_std).item())

        launch, target, ship = launch.squeeze(0), target.squeeze(0), ship.squeeze(0)

        # Defensive: if the model produced NaN/Inf (diverged policy), emit no
        # moves rather than crashing the env with invalid actions. Training
        # will recover via the post-update NaN check; rollout just wastes the
        # episode.
        launch_ok = torch.isfinite(launch).all().item()
        target_ok = torch.isfinite(target).all().item()
        ship_ok = torch.isfinite(ship).all().item()
        value_ok = torch.isfinite(value).all().item()
        frac_ok = math.isfinite(frac_std) and frac_std > 0
        if not (launch_ok and target_ok and ship_ok and value_ok and frac_ok):
            # One-shot diagnostic per agent instance so we know which head is
            # producing garbage when this fires.
            if not getattr(self, "_warned_nan", False):
                self._warned_nan = True
                print(
                    f"[_act] non-finite output: launch_ok={launch_ok} "
                    f"target_ok={target_ok} ship_ok={ship_ok} "
                    f"value_ok={value_ok} frac_std={frac_std}",
                    flush=True,
                )
            return moves

        if self.training:
            launch_actions = Bernoulli(logits=launch).sample()
            target_actions = Categorical(logits=target).sample()
            frac_mean = torch.sigmoid(ship)
            frac_actions = Normal(frac_mean, frac_std).sample().clamp(0.01, 1.0)
        else:
            launch_actions = (torch.sigmoid(launch) > LAUNCH_THRESHOLD).float()
            target_actions = target.argmax(dim=-1)
            frac_actions = torch.sigmoid(ship)

        if self.record:
            lp = Bernoulli(logits=launch).log_prob(launch_actions).sum()
            tp = Categorical(logits=target).log_prob(target_actions) * launch_actions
            fp = Normal(torch.sigmoid(ship), frac_std).log_prob(frac_actions) * launch_actions
            log_prob = lp + tp.sum() + fp.sum()

            player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
            phi = compute_potential(obs, player)

            # Store on CPU so the buffer doesn't pin GPU memory.
            self.trajectory.append(
                {
                    "channels": channels.detach().cpu(),
                    "scalars": scalars.detach().cpu(),
                    "coords": coords.detach().squeeze(0).cpu(),
                    "launch_act": launch_actions.detach().cpu(),
                    "target_act": target_actions.detach().cpu(),
                    "frac_act": frac_actions.detach().cpu(),
                    "log_prob": float(log_prob.item()),
                    "value": float(value.item()),
                    "phi": float(phi),
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


def _wrap_as_fn(sp_agent: StochasticPolicyAgent) -> Callable:
    """Convert a StochasticPolicyAgent into a plain callable for env.run."""
    def fn(obs):
        return sp_agent(obs)
    return fn


_WARNED_SHORT_GAME = [False]


def _play_episode(
    learner: StochasticPolicyAgent,
    opponent_fn: Callable,
    learner_slot: int,
    seed: int | None,
):
    config = {"seed": seed} if seed is not None else {}
    env = make("orbit_wars", configuration=config, debug=False)
    learner.reset_trajectory()

    learner_fn = _wrap_as_fn(learner)
    players = [learner_fn, opponent_fn] if learner_slot == 0 else [opponent_fn, learner_fn]
    env.run(players)
    final_reward = env.steps[-1][learner_slot].reward or 0

    # Diagnostic: if the game ended much earlier than expected, surface the
    # per-player status + info so we can see WHY (TIMEOUT / INVALID / ERROR).
    if len(env.steps) < 50 and not _WARNED_SHORT_GAME[0]:
        _WARNED_SHORT_GAME[0] = True
        final = env.steps[-1]
        statuses = [getattr(s, "status", "?") for s in final]
        infos = [getattr(s, "info", None) for s in final]
        print(
            f"[short game] steps={len(env.steps)} learner_slot={learner_slot} "
            f"statuses={statuses} infos={infos}",
            flush=True,
        )
    return learner.trajectory, final_reward, env


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


def _pack_trajectory(
    traj: list[dict],
    final_reward: float,
    gamma: float,
    lam: float,
    use_shaping: bool,
):
    if not traj:
        return None
    T = len(traj)
    values = [s["value"] for s in traj]
    phis = [s["phi"] for s in traj]

    rewards = [0.0] * T
    if use_shaping:
        for t in range(T - 1):
            rewards[t] = gamma * phis[t + 1] - phis[t]
        rewards[T - 1] = final_reward - phis[T - 1]
    else:
        rewards[T - 1] = final_reward

    adv, ret = _compute_gae(rewards, values, gamma, lam)

    channels = torch.stack([s["channels"] for s in traj])
    scalars = torch.stack([s["scalars"] for s in traj])
    log_prob = torch.tensor([s["log_prob"] for s in traj])
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(ret, dtype=torch.float32)

    M = MAX_PLANETS
    coords = torch.zeros(T, M, 2)
    launch = torch.zeros(T, M)
    target = torch.zeros(T, M, dtype=torch.long)
    frac = torch.zeros(T, M)
    mask = torch.zeros(T, M)
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


FRAC_LOG_STD_MIN = math.log(0.05)
FRAC_LOG_STD_MAX = math.log(1.0)


def _ppo_update(
    model: CNNv1,
    batch: dict,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    epochs: int = 2,
    minibatch: int = 32,
    lr: float = 1e-4,
    kl_stop: float = 0.4,
    ratio_max: float = 3.0,
    max_grad_norm: float = 0.1,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    N = batch["channels"].size(0)
    old_log_prob = batch["log_prob"].to(device)
    adv = batch["adv"].to(device)
    ret = batch["ret"].to(device)

    # Advantage normalization — standard PPO stabilizer. Skip if all-zero (degenerate).
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    else:
        adv = adv - adv.mean()

    clip_frac_sum = 0.0
    approx_kl_sum = 0.0
    n_mb = skipped = 0
    stop_early = False
    last_policy = last_value = last_entropy = 0.0
    last_frac_std = float(torch.exp(model.frac_log_std).item())
    epoch_times: list[float] = []

    for _ in range(epochs):
        if stop_early:
            break
        epoch_t0 = time.time()
        idx = torch.randperm(N)
        for start in range(0, N, minibatch):
            sel = idx[start : start + minibatch]
            ch = batch["channels"][sel].to(device)
            sc = batch["scalars"][sel].to(device)
            co = batch["coords"][sel].to(device)
            la = batch["launch_act"][sel].to(device)
            ta = batch["target_act"][sel].to(device)
            fa = batch["frac_act"][sel].to(device)
            mk = batch["mask"][sel].to(device)

            feat_map = model.conv(ch)
            scalar_feat = model.scalar_mlp(sc)
            value = model.value(feat_map, scalar_feat)

            launch, target, ship = _planet_features(model, ch, sc, co)

            frac_std = torch.exp(model.frac_log_std)
            launch_lp = Bernoulli(logits=launch).log_prob(la) * mk
            target_lp = Categorical(logits=target).log_prob(ta) * la * mk
            ship_lp = Normal(torch.sigmoid(ship), frac_std).log_prob(fa) * la * mk
            new_log_prob = (launch_lp + target_lp + ship_lp).sum(dim=-1)

            # Per-step log_prob is a sum over ~16 planets; small weight
            # drift can blow up the exp. Clamp the ratio to keep the loss
            # scale sane while preserving the PPO clip surface.
            log_ratio = (new_log_prob - old_log_prob[sel]).clamp(
                min=math.log(1.0 / ratio_max), max=math.log(ratio_max)
            )
            ratio = torch.exp(log_ratio)
            unclipped = ratio * adv[sel]
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[sel]
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = F.mse_loss(value, ret[sel])

            launch_ent = Bernoulli(logits=launch).entropy() * mk
            target_ent = Categorical(logits=target).entropy() * mk
            entropy = (launch_ent + target_ent).sum(dim=-1).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            if not torch.isfinite(loss):
                skipped += 1
                continue

            opt.zero_grad()
            loss.backward()
            # NaN/Inf grad guard — zero them out instead of stepping into NaN space.
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                skipped += 1
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()

            # Keep learned exploration std in a sane range.
            with torch.no_grad():
                model.frac_log_std.clamp_(FRAC_LOG_STD_MIN, FRAC_LOG_STD_MAX)

            clip_frac_sum += float(((ratio - 1.0).abs() > clip).float().mean().item())
            # Schulman's approx KL = E[(ratio - 1) - log(ratio)]; always ≥ 0.
            approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
            approx_kl_sum += approx_kl
            n_mb += 1
            last_policy = float(policy_loss.item())
            last_value = float(value_loss.item())
            last_entropy = float(entropy.item())
            last_frac_std = float(frac_std.item())

            if approx_kl > kl_stop:
                stop_early = True
                break

        epoch_times.append(time.time() - epoch_t0)

    return {
        "policy_loss": last_policy,
        "value_loss": last_value,
        "entropy": last_entropy,
        "clip_frac": clip_frac_sum / max(1, n_mb),
        "approx_kl": approx_kl_sum / max(1, n_mb),
        "frac_std": last_frac_std,
        "skipped_mb": skipped,
        "early_stopped": stop_early,
        "epoch_times": epoch_times,
    }


def _freeze(snap: CNNv1) -> CNNv1:
    snap.eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


def _eval_winrate(
    model: CNNv1, opponent_id: str, games: int, seed_start: int
) -> float:
    """Write current weights so cnn_v1_agent sees them, then run eval."""
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    _cnn_mod.reload_weights()
    result = evaluate_agent(
        "cnn_v1", [opponent_id], games_per=games, seed_start=seed_start, verbose=False
    )
    _cnn_mod.reload_weights()
    return result[opponent_id]["win_rate"]


def _save_state_and_reload(state: dict) -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, WEIGHTS_PATH)
    _cnn_mod.reload_weights()


LATEST_PATH = WEIGHTS_PATH.parent / "latest.pt"


def _save_latest(model: CNNv1, iter_num: int) -> None:
    """Checkpoint current (not-best) weights to disk so training can resume.

    Plain state_dict (same format as cnn_v1.pt) so load_model() accepts it.
    """
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LATEST_PATH)


def _has_nan_params(model: CNNv1) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return True
    return False


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def train_ppo(
    iterations: int = 200,
    episodes_per_iter: int = 32,
    opponents: list[str] | None = None,
    snapshot_every: int = 10,
    snapshot_pool_size: int = 8,
    self_play_ratio: float = 0.3,
    eval_every: int = 10,
    eval_games: int = 20,
    eval_opponent: str = "physical_v2",
    resume_from: Path | str | None = None,
    save_to: Path | str | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    use_shaping: bool = True,
    seed_start: int = 0,
    log_path: Path | str | None = None,
    device: str | None = None,
    episode_progress: bool = True,
    replay_every: int = 10,
    replay_dir: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    kl_stop_start: float = 0.4,
    kl_stop_end: float | None = 0.05,
    verbose: bool = True,
) -> CNNv1:
    """Pure self-play PPO with a snapshot ring buffer.

    - `opponents`: external opponents (e.g. physics agents). Empty/None by
      default → pure self-play. Kept as a hook for ablations; sample with
      a small probability alongside snapshots if non-empty.
    - `snapshot_every`: iters between snapshots.
    - `self_play_ratio`: P(self) when snapshot pool non-empty; otherwise
      self is used 100%.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # External checkpoint dir (e.g. mounted Google Drive, GCS) — survives
    # Colab runtime shutdown. Checked first for resume, written to every iter.
    ext_ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if ext_ckpt_dir is not None:
        ext_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ext_latest = ext_ckpt_dir / "latest.pt" if ext_ckpt_dir else None
    ext_best = ext_ckpt_dir / "cnn_v1.pt" if ext_ckpt_dir else None

    # Auto-resume: prefer external checkpoint (if provided and exists),
    # then local latest.pt, then local cnn_v1.pt, else fresh.
    resume_msg = "fresh weights"
    if resume_from is None:
        if ext_latest is not None and ext_latest.exists():
            resume_from = ext_latest
            resume_msg = f"resumed ext latest: {ext_latest}"
        elif ext_best is not None and ext_best.exists():
            resume_from = ext_best
            resume_msg = f"resumed ext best: {ext_best}"
        elif LATEST_PATH.exists():
            resume_from = LATEST_PATH
            resume_msg = f"resumed local latest: {LATEST_PATH}"
        elif WEIGHTS_PATH.exists():
            resume_from = WEIGHTS_PATH
            resume_msg = f"resumed local best: {WEIGHTS_PATH}"
    else:
        resume_msg = f"resumed: {resume_from}"

    model = load_model(resume_from) if resume_from else fresh_model()

    # GPU compat probe: torch 2.10+cu128 drops SM_60 (P100). If the assigned
    # GPU can't run torch kernels, fall back to CPU so training still runs.
    if device.type == "cuda":
        try:
            _probe = torch.zeros(4, device=device) + 1.0
            _probe.cpu()
        except Exception as e:
            print(
                f"WARN: GPU unusable ({type(e).__name__}: {e}); falling back to CPU. "
                f"On Kaggle: Settings → Accelerator → GPU T4 x2 to avoid this.",
                flush=True,
            )
            device = torch.device("cpu")

    model = model.to(device)

    # Emit device line BEFORE warmup so the log is never empty on a GPU error.
    if verbose:
        gpu_info = ""
        if device.type == "cuda":
            try:
                gpu_info = f" ({torch.cuda.get_device_name(device)})"
            except Exception:
                pass
        param_count = sum(p.numel() for p in model.parameters())
        kl_sched = (
            f"kl_stop: {kl_stop_start}→{kl_stop_end}"
            if kl_stop_end is not None
            else f"kl_stop: {kl_stop_start}"
        )
        print(
            f"device: {device}{gpu_info}  params: {param_count / 1e6:.2f}M  "
            f"iters: {iterations}  ep/iter: {episodes_per_iter}  {kl_sched}  "
            f"{resume_msg}",
            flush=True,
        )

    # Warmup: first CUDA forward pass triggers context init which can take
    # several seconds — well over orbit_wars' 1s actTimeout.
    try:
        with torch.no_grad():
            _warm_ch = torch.zeros(1, NUM_CHANNELS, GRID, GRID, device=device)
            _warm_sc = torch.zeros(1, SCALAR_DIM, device=device)
            _warm_coords = torch.zeros(1, 1, 2, device=device)
            _fm, _sf = model(_warm_ch, _warm_sc)
            _planet_features(model, _warm_ch, _warm_sc, _warm_coords)
            model.value(_fm, _sf)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    except Exception as e:
        print(f"WARN: warmup failed ({type(e).__name__}: {e}); continuing anyway", flush=True)

    learner = StochasticPolicyAgent(model, training=True, record=True)
    snapshots: deque[CNNv1] = deque(maxlen=snapshot_pool_size)

    ext_opponents = list(opponents or [])

    best_winrate = -1.0
    best_state = copy.deepcopy(model.state_dict())

    run_id = time.strftime("%Y%m%dT%H%M%S")
    if log_path is None:
        log_path = TRAIN_ROOT / f"ppo_{run_id}.jsonl"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w")

    if replay_dir is None:
        replay_dir = TRAIN_ROOT.parent / "replays" / "training" / run_id
    replay_dir = Path(replay_dir)
    if replay_every > 0:
        replay_dir.mkdir(parents=True, exist_ok=True)

    train_start = time.time()
    try:
        for it in range(iterations):
            t0 = time.time()
            buffers: list[dict] = []
            rewards_hist: list[float] = []
            ep_types = {"self": 0, "snapshot": 0, "external": 0}
            ep_steps: list[int] = []

            last_env = None
            last_opp_kind = ""
            for ep in range(episodes_per_iter):
                r = random.random()
                if ext_opponents and r < 0.1:
                    opp_id = random.choice(ext_opponents)
                    opp_fn = Agent(id=opp_id).fn
                    ep_types["external"] += 1
                    opp_kind = f"ext:{opp_id}"
                elif not snapshots or r < 0.1 + self_play_ratio:
                    # Self-play against current model (stochastic, non-recording).
                    self_opp = StochasticPolicyAgent(model, training=True, record=False)
                    opp_fn = _wrap_as_fn(self_opp)
                    ep_types["self"] += 1
                    opp_kind = "self"
                else:
                    snap = random.choice(list(snapshots))
                    snap_agent = StochasticPolicyAgent(snap, training=True, record=False)
                    opp_fn = _wrap_as_fn(snap_agent)
                    ep_types["snapshot"] += 1
                    opp_kind = "snapshot"

                slot = ep % 2
                ep_t0 = time.time()
                traj, reward, env = _play_episode(
                    learner, opp_fn, learner_slot=slot, seed=seed_start + it * 1000 + ep
                )
                rewards_hist.append(reward)
                packed = _pack_trajectory(traj, reward, gamma, lam, use_shaping)
                if packed is not None:
                    buffers.append(packed)

                last_env = env
                last_opp_kind = opp_kind
                steps = len(env.steps)
                ep_steps.append(steps)

                if episode_progress:
                    wins_so_far = sum(1 for r in rewards_hist if r > 0.5)
                    losses_so_far = sum(1 for r in rewards_hist if r < -0.5)
                    draws_so_far = len(rewards_hist) - wins_so_far - losses_so_far
                    iter_elapsed = time.time() - t0
                    print(
                        f"  iter {it + 1:3d} ep {ep + 1:2d}/{episodes_per_iter}  "
                        f"opp={opp_kind:12s}  slot={slot}  r={reward:+.0f}  "
                        f"steps={steps:3d}  "
                        f"W/L/D={wins_so_far}/{losses_so_far}/{draws_so_far}  "
                        f"ep={time.time() - ep_t0:.1f}s  "
                        f"iter={_fmt_duration(iter_elapsed)}",
                        flush=True,
                    )

            if not buffers:
                # All rollouts returned empty trajectories — usually policy
                # collapse. Print a loud notice instead of silently skipping,
                # so the user sees iter-level signal even in this failure mode.
                wins = sum(1 for r in rewards_hist if r > 0.5)
                losses = sum(1 for r in rewards_hist if r < -0.5)
                draws = len(rewards_hist) - wins - losses
                total_elapsed = time.time() - train_start
                print(
                    f"iter {it + 1:3d}/{iterations}  EMPTY_BUFFERS  "
                    f"W/L/D={wins}/{losses}/{draws}  "
                    f"opp={ep_types['self']}s/{ep_types['snapshot']}p/{ep_types['external']}x  "
                    f"elapsed={_fmt_duration(total_elapsed)}  "
                    f"(policy likely diverged; reverting to best_state)",
                    flush=True,
                )
                # Revert to the last-known-good state and continue.
                model.load_state_dict(best_state)
                _save_latest(model, it + 1)
                continue

            batch = {k: torch.cat([b[k] for b in buffers], dim=0) for k in buffers[0]}

            # Dynamic kl_stop: linear decay from kl_stop_start to kl_stop_end
            # as training progresses (tighter tolerance late, more exploration early).
            if kl_stop_end is not None and iterations > 1:
                progress = it / (iterations - 1)
                kl_stop_now = kl_stop_start + (kl_stop_end - kl_stop_start) * progress
            else:
                kl_stop_now = kl_stop_start
            stats = _ppo_update(model, batch, kl_stop=kl_stop_now)

            # NaN recovery: if the update produced non-finite params, revert
            # to best_state and keep training from there instead of emitting
            # garbage for the rest of the run.
            if _has_nan_params(model):
                if verbose:
                    print(
                        f"  iter {it + 1}: NaN in params detected; reverting to best_state",
                        flush=True,
                    )
                model.load_state_dict(best_state)

            if (it + 1) % snapshot_every == 0:
                snap = _freeze(copy.deepcopy(model))
                snapshots.append(snap)

            # Checkpoint the current model every iter so training can resume
            # from the most recent state (separate from best-by-eval cnn_v1.pt).
            _save_latest(model, it + 1)
            if ext_latest is not None:
                try:
                    torch.save(model.state_dict(), ext_latest)
                except Exception as e:
                    if verbose:
                        print(f"  ext latest save failed: {e}", flush=True)

            # Flush a replay of the last episode periodically.
            if replay_every > 0 and (it + 1) % replay_every == 0 and last_env is not None:
                rpath = replay_dir / f"iter_{it + 1:03d}_{last_opp_kind.replace(':', '_')}.html"
                try:
                    rpath.write_text(last_env.render(mode="html"))
                    if verbose:
                        print(f"  replay: {rpath}", flush=True)
                except Exception as e:
                    if verbose:
                        print(f"  replay save failed: {e}", flush=True)

            eval_wr = None
            if (it + 1) % eval_every == 0:
                eval_wr = _eval_winrate(
                    model, eval_opponent, eval_games, seed_start=100_000 + it
                )
                if eval_wr > best_winrate:
                    best_winrate = eval_wr
                    best_state = copy.deepcopy(model.state_dict())
                    # current weights already saved on disk by _eval_winrate;
                    # also mirror best to external dir.
                    if ext_best is not None:
                        try:
                            torch.save(best_state, ext_best)
                        except Exception as e:
                            if verbose:
                                print(f"  ext best save failed: {e}", flush=True)
                else:
                    _save_state_and_reload(best_state)

            mean_r = sum(rewards_hist) / max(1, len(rewards_hist))
            wins = sum(1 for r in rewards_hist if r > 0.5)
            losses = sum(1 for r in rewards_hist if r < -0.5)
            draws = len(rewards_hist) - wins - losses
            avg_steps = sum(ep_steps) / max(1, len(ep_steps))
            iter_time = time.time() - t0
            total_elapsed = time.time() - train_start
            iters_done = it + 1
            iters_remaining = iterations - iters_done
            eta_s = (total_elapsed / iters_done) * iters_remaining if iters_done else 0
            gpu_mem_gb = None
            if device.type == "cuda":
                try:
                    gpu_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                except Exception:
                    pass

            row = {
                "iter": it + 1,
                "mean_reward": round(mean_r, 4),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "avg_episode_steps": round(avg_steps, 1),
                "policy_loss": round(stats["policy_loss"], 4),
                "value_loss": round(stats["value_loss"], 4),
                "entropy": round(stats["entropy"], 4),
                "clip_frac": round(stats["clip_frac"], 4),
                "approx_kl": round(stats["approx_kl"], 4),
                "frac_std": round(stats["frac_std"], 4),
                "samples": int(batch["channels"].size(0)),
                "snapshot_count": len(snapshots),
                "ep_types": ep_types,
                "early_stopped": bool(stats["early_stopped"]),
                "skipped_mb": int(stats["skipped_mb"]),
                "epoch_times_s": [round(t, 3) for t in stats.get("epoch_times", [])],
                "kl_stop_now": round(kl_stop_now, 4),
                "time_s": round(iter_time, 2),
                "elapsed_s": round(total_elapsed, 2),
                "eta_s": round(eta_s, 2),
            }
            if gpu_mem_gb is not None:
                row["gpu_mem_gb"] = round(gpu_mem_gb, 3)
            if eval_wr is not None:
                row["eval_winrate"] = round(eval_wr, 3)
                row["best_winrate"] = round(best_winrate, 3)

            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

            if verbose:
                eval_str = (
                    f"  eval={eval_wr:.2f}(best={best_winrate:.2f})"
                    if eval_wr is not None
                    else ""
                )
                es = "*" if stats["early_stopped"] else " "
                mem_str = f"  mem={gpu_mem_gb:.1f}G" if gpu_mem_gb is not None else ""
                ep_types_str = (
                    f"{ep_types['self']}s/{ep_types['snapshot']}p/{ep_types['external']}x"
                )
                epoch_t = stats.get("epoch_times", [])
                epoch_str = "/".join(f"{t:.1f}" for t in epoch_t) if epoch_t else "-"
                print(
                    f"iter {it + 1:3d}/{iterations}  "
                    f"W/L/D={wins}/{losses}/{draws}  "
                    f"r={mean_r:+.3f}  steps={avg_steps:.0f}  "
                    f"pi={stats['policy_loss']:+.2f}  v={stats['value_loss']:.3f}  "
                    f"ent={stats['entropy']:.1f}  kl={stats['approx_kl']:.3f}{es} "
                    f"clip={stats['clip_frac']:.2f}  std={stats['frac_std']:.2f}  "
                    f"opp={ep_types_str}  snap={len(snapshots)}  "
                    f"n={batch['channels'].size(0)}  "
                    f"ep_t={epoch_str}s  "
                    f"t={_fmt_duration(iter_time)} "
                    f"elapsed={_fmt_duration(total_elapsed)} "
                    f"ETA={_fmt_duration(eta_s)}"
                    f"{mem_str}{eval_str}",
                    flush=True,
                )
    finally:
        log_f.close()

    model.load_state_dict(best_state)
    model.eval()
    path = save_model(model, save_to)
    _cnn_mod.reload_weights()
    if verbose:
        print(
            f"saved best weights: {path} (best_winrate={best_winrate:.3f}, "
            f"log: {log_path})",
            flush=True,
        )
    return model
