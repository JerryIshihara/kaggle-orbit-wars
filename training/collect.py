"""Behavior-cloning dataset collector.

Runs teacher-vs-opponent matches, featurizes every observation from the
teacher's POV, extracts per-planet labels from the teacher's actions, and
saves the result as a single .pt file of dict-of-tensors for fast reload.

Variable planet counts per step are handled with a boolean mask — each
sample is padded to MAX_PLANETS (zero-filled beyond the live planets).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch

from agents import Agent
from agents.agent_cnn_v1 import GRID, featurize
from kaggle_environments import make

from .common import TRAIN_ROOT, decode_teacher_action

MAX_PLANETS = 16


def _one_game(teacher: Agent, opponent: Agent, teacher_slot: int, seed: int | None) -> dict:
    config = {"seed": seed} if seed is not None else {}
    env = make("orbit_wars", configuration=config, debug=False)
    players = [teacher.fn, opponent.fn] if teacher_slot == 0 else [opponent.fn, teacher.fn]
    env.run(players)

    channels_list = []
    scalars_list = []
    coords_list = []
    launched_list = []
    targets_list = []
    fracs_list = []
    mask_list = []

    for step in env.steps[:-1]:
        state = step[teacher_slot]
        obs = state.observation
        action = state.action or []
        # action is list of [from_id, angle, ships] from the teacher
        ch, sc, my_planets = featurize(obs)
        if not my_planets:
            continue

        launched, targets, fracs = decode_teacher_action(action, my_planets)
        n = min(MAX_PLANETS, len(my_planets))
        coords = torch.zeros(MAX_PLANETS, 2)
        l = torch.zeros(MAX_PLANETS)
        t = torch.zeros(MAX_PLANETS, dtype=torch.long)
        f = torch.zeros(MAX_PLANETS)
        mask = torch.zeros(MAX_PLANETS)

        for i in range(n):
            _, x, y, _ = my_planets[i]
            coords[i, 0] = x
            coords[i, 1] = y
            l[i] = launched[i]
            t[i] = targets[i]
            f[i] = fracs[i]
            mask[i] = 1.0

        channels_list.append(ch)
        scalars_list.append(sc)
        coords_list.append(coords)
        launched_list.append(l)
        targets_list.append(t)
        fracs_list.append(f)
        mask_list.append(mask)

    if not channels_list:
        return {}
    return {
        "channels": torch.stack(channels_list),
        "scalars": torch.stack(scalars_list),
        "coords": torch.stack(coords_list),
        "launched": torch.stack(launched_list),
        "targets": torch.stack(targets_list),
        "fracs": torch.stack(fracs_list),
        "mask": torch.stack(mask_list),
    }


def collect_bc_dataset(
    num_games: int,
    teacher_id: str = "sniper",
    opponent_id: str = "sniper",
    save_to: Path | str | None = None,
    seed_start: int = 0,
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    teacher = Agent(id=teacher_id)
    opponent = Agent(id=opponent_id)
    accum: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("channels", "scalars", "coords", "launched", "targets", "fracs", "mask")
    }
    total_samples = 0
    for g in range(num_games):
        teacher_slot = g % 2  # alternate seats for balance
        shard = _one_game(teacher, opponent, teacher_slot, seed=seed_start + g)
        if not shard:
            continue
        for k, v in shard.items():
            accum[k].append(v)
        total_samples += shard["channels"].size(0)
        if verbose and (g + 1) % 10 == 0:
            print(f"  collected {g + 1}/{num_games} games, {total_samples} samples", file=sys.stderr)

    out = {k: torch.cat(v, dim=0) for k, v in accum.items() if v}
    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, save_to)
        if verbose:
            print(f"saved BC dataset: {save_to} ({total_samples} samples)", file=sys.stderr)
    return out


def default_dataset_path(teacher_id: str, num_games: int) -> Path:
    return TRAIN_ROOT / "datasets" / f"bc_{teacher_id}_{num_games}.pt"
