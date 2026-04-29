"""Per-entity featurizers for transformer_v1.

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
]
