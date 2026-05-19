"""Per-entity featurizers for transformer_v2.

A featurizer turns one turn of `obs` into the float vectors the encoder
projects to ``d_model``. Splitting featurization out of the encoder keeps
the nn.Module pure — the encoder only depends on a fixed raw-dim and
doesn't care how those dims are computed.

Currently exposes the fleet featurizer; planet / scalar featurizers will
land here as the v2-path stream comes online.
"""

from .fleet_featurizer import (
    DEFAULT_ENCODER_DATA_DIR,
    ENCODER_LABEL_HEADS,
    ENCODER_PRETRAIN_LABELS,
    FLEET_LABEL_FIELDS,
    FLEET_RAW_DIM,
    FleetFeaturizer,
    FleetTracker,
    TRANSFORMER_AUX_LABELS,
    featurize_fleets,
    save_episode_fleet_csv,
)
from .planet_featurizer import (
    PLANET_LABEL_FIELDS,
    PLANET_RAW_DIM,
    SCALAR_DIM as PLANET_SCALAR_DIM,
    N_FUTURE_ANCHORS as PLANET_N_FUTURE_ANCHORS,
    PlanetFeaturizer,
    featurize_planets,
    save_episode_planet_csv,
)
from .entity_featurizer import (
    ARRIVAL_HORIZONS as ENTITY_ARRIVAL_HORIZONS,
    CROSS_ENTITY_LABEL_COLS,
    CROSS_ENTITY_TACTICAL_HORIZONS,
    CROSS_ENTITY_VALUE_HORIZONS,
    ENTITY_FEATURE_COLS,
    ENTITY_LABEL_COLS,
    EntityFeaturizer,
    EPISODE_STEPS as ENTITY_EPISODE_STEPS,
    LABEL_HORIZONS as ENTITY_LABEL_HORIZONS,
    NUM_OWNER_SLOTS as ENTITY_NUM_OWNER_SLOTS,
    N_FRONTIER_CLASSES as ENTITY_N_FRONTIER_CLASSES,
    N_OWNER_CLASSES as ENTITY_N_OWNER_CLASSES,
    SECTOR_RADIUS as ENTITY_SECTOR_RADIUS,
    STAT_NAMES as ENTITY_STAT_NAMES,
    featurize_entity_inbound,
    save_episode_cross_entity_csv,
    save_episode_entity_csv,
)
from .action_featurizer import (
    ACTION_IGNORE_INDEX,
    ACTION_LABEL_COLS,
    save_episode_action_csv,
)
from .inference import featurize_observation

__all__ = [
    # fleet
    "DEFAULT_ENCODER_DATA_DIR",
    "ENCODER_LABEL_HEADS",
    "ENCODER_PRETRAIN_LABELS",
    "FLEET_LABEL_FIELDS",
    "FLEET_RAW_DIM",
    "FleetFeaturizer",
    "FleetTracker",
    "TRANSFORMER_AUX_LABELS",
    "featurize_fleets",
    "save_episode_fleet_csv",
    # planet+comet (unified)
    "PLANET_LABEL_FIELDS",
    "PLANET_RAW_DIM",
    "PLANET_SCALAR_DIM",
    "PLANET_N_FUTURE_ANCHORS",
    "PlanetFeaturizer",
    "featurize_planets",
    "save_episode_planet_csv",
    # entity (per-planet inbound-fleet summary + supervised labels)
    "ENTITY_ARRIVAL_HORIZONS",
    "ENTITY_EPISODE_STEPS",
    "ENTITY_FEATURE_COLS",
    "ENTITY_LABEL_COLS",
    "ENTITY_LABEL_HORIZONS",
    "ENTITY_NUM_OWNER_SLOTS",
    "ENTITY_N_FRONTIER_CLASSES",
    "ENTITY_N_OWNER_CLASSES",
    "ENTITY_SECTOR_RADIUS",
    "ENTITY_STAT_NAMES",
    "CROSS_ENTITY_LABEL_COLS",
    "CROSS_ENTITY_TACTICAL_HORIZONS",
    "CROSS_ENTITY_VALUE_HORIZONS",
    "EntityFeaturizer",
    "featurize_entity_inbound",
    "save_episode_cross_entity_csv",
    "save_episode_entity_csv",
    # action (snapshot-level expert imitation)
    "ACTION_IGNORE_INDEX",
    "ACTION_LABEL_COLS",
    "save_episode_action_csv",
    # inference (live obs → batch dict)
    "featurize_observation",
]
