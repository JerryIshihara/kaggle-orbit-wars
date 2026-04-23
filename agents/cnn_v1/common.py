"""Shared helpers for the CNN v1 training pipeline.

Bridges the env's action format (`[[from_planet_id, angle, num_ships], ...]`)
and the policy's per-planet output tuples (launch, target_cell, ship_frac).
"""

from __future__ import annotations

import math
from pathlib import Path

from .agent import BOARD_SIZE, CELL, GRID, WEIGHTS_PATH, CNNv1

RAY_DIST = 30.0  # units along the angle that we treat as the "target" of a launch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_ROOT = _REPO_ROOT / "logs" / "training"
WEIGHTS_DIR = Path(WEIGHTS_PATH).parent


def angle_to_cell(px: float, py: float, angle: float, dist: float = RAY_DIST) -> int:
    """Project a ray from (px, py) at `angle` distance `dist`, snap to a grid cell."""
    tx = max(0.0, min(BOARD_SIZE - 0.001, px + dist * math.cos(angle)))
    ty = max(0.0, min(BOARD_SIZE - 0.001, py + dist * math.sin(angle)))
    cx = min(GRID - 1, int(tx / CELL))
    cy = min(GRID - 1, int(ty / CELL))
    return cy * GRID + cx


def decode_teacher_action(moves, my_planets):
    """Turn `[[from_id, angle, ships], ...]` into per-planet labels.

    Returns three lists aligned with `my_planets`:
      launched[i]     : 1.0 if planet i had any launches, else 0.0
      target_cell[i]  : int in [0, GRID*GRID) — first target cell if launched, else 0
      ship_frac[i]    : sum_of_ships / garrison, clipped to (0, 1], else 0

    For planets with multiple launches in a turn we use the first launch's
    target and sum the ships — good enough for a behavior-cloning warm-start.
    """
    id_to_idx = {p[0]: i for i, p in enumerate(my_planets)}
    n = len(my_planets)
    launched = [0.0] * n
    target_cell = [0] * n
    ship_frac = [0.0] * n
    ship_sum = [0] * n

    for m in moves:
        if not isinstance(m, (list, tuple)) or len(m) < 3:
            continue
        pid, angle, ships = m[0], float(m[1]), int(m[2])
        if pid not in id_to_idx:
            continue
        i = id_to_idx[pid]
        _, x, y, garrison = my_planets[i]
        if launched[i] == 0.0:
            target_cell[i] = angle_to_cell(x, y, angle)
        launched[i] = 1.0
        ship_sum[i] += ships
        ship_frac[i] = min(1.0, ship_sum[i] / max(1, garrison))
    return launched, target_cell, ship_frac


def fresh_model() -> CNNv1:
    return CNNv1()


def save_model(model: CNNv1, path: Path | str | None = None) -> Path:
    import torch

    target = Path(path) if path else WEIGHTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)
    return target


def load_model(path: Path | str | None = None) -> CNNv1:
    import torch

    src = Path(path) if path else WEIGHTS_PATH
    m = fresh_model()
    if src.exists():
        m.load_state_dict(torch.load(src, map_location="cpu"), strict=False)
    return m
