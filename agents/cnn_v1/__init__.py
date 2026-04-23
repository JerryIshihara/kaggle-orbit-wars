"""CNN v1 agent — model, training pipelines, eval all live here."""

from .agent import (
    BOARD_SIZE,
    CELL,
    CNNv1,
    D_MODEL,
    GRID,
    LAUNCH_THRESHOLD,
    MAX_STEPS,
    NUM_CHANNELS,
    SCALAR_DIM,
    SHIPS_LOG_MAX,
    WEIGHTS_PATH,
    cnn_v1_agent,
    featurize,
    reload_weights,
)
from .bc import train_bc
from .collect import collect_bc_dataset, default_dataset_path
from .common import (
    angle_to_cell,
    decode_teacher_action,
    fresh_model,
    load_model,
    save_model,
)
from .eval import evaluate_agent
from .ppo import train_ppo

__all__ = [
    "BOARD_SIZE",
    "CELL",
    "CNNv1",
    "D_MODEL",
    "GRID",
    "LAUNCH_THRESHOLD",
    "MAX_STEPS",
    "NUM_CHANNELS",
    "SCALAR_DIM",
    "SHIPS_LOG_MAX",
    "WEIGHTS_PATH",
    "angle_to_cell",
    "cnn_v1_agent",
    "collect_bc_dataset",
    "decode_teacher_action",
    "default_dataset_path",
    "evaluate_agent",
    "featurize",
    "fresh_model",
    "load_model",
    "reload_weights",
    "save_model",
    "train_bc",
    "train_ppo",
]
