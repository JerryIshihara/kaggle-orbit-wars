"""Centralized data-layout constants for transformer_v2.

All featurizers, pretrain scripts, and analysis tooling import from
here so the layout under ``data/`` is changed in one place. Layout:

    data/
    ├── replays/                  # raw episode .json.gz dumps
    ├── datasets/                 # featurized pretraining CSVs
    │   ├── fleet/                # save_episode_fleet_csv outputs
    │   ├── planet/               # save_episode_planet_csv outputs
    │   ├── entity/               # save_episode_entity_csv outputs
    │   ├── cross_entity/         # save_episode_cross_entity_csv outputs
    │   └── action/               # save_episode_action_csv outputs (expert imitation)
    └── runs/                     # training-run outputs (ckpts, logs)
        ├── fleet/                # FleetEncoder pretrain runs
        ├── planet/               # PlanetEncoder pretrain runs
        ├── entity/               # PlanetEntityEncoder pretrain runs
        ├── cross_entity/         # CrossEntityAttention pretrain runs
        └── action/               # ActionDecoder fine-tune runs
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = REPO_ROOT / "data"

# Source replays (read-only, don't write here from training code).
REPLAYS_DIR: Path = DATA_ROOT / "replays"

# Featurized datasets — one subdir per encoder kind. CSVs land here.
DATASETS_ROOT: Path = DATA_ROOT / "datasets"
FLEET_DATASET_DIR: Path = DATASETS_ROOT / "fleet"
PLANET_DATASET_DIR: Path = DATASETS_ROOT / "planet"
ENTITY_DATASET_DIR: Path = DATASETS_ROOT / "entity"
CROSS_ENTITY_DATASET_DIR: Path = DATASETS_ROOT / "cross_entity"
ACTION_DATASET_DIR: Path = DATASETS_ROOT / "action"

# Training-run outputs — one subdir per encoder kind. Each pretrain
# script writes ``<RUNS_DIR>/<timestamp>/{ckpt, log.json, ...}``.
RUNS_ROOT: Path = DATA_ROOT / "runs"
FLEET_RUNS_DIR: Path = RUNS_ROOT / "fleet"
PLANET_RUNS_DIR: Path = RUNS_ROOT / "planet"
ENTITY_RUNS_DIR: Path = RUNS_ROOT / "entity"
CROSS_ENTITY_RUNS_DIR: Path = RUNS_ROOT / "cross_entity"
ACTION_RUNS_DIR: Path = RUNS_ROOT / "action"
