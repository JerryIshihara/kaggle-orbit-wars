"""Per-planet/comet featurizer for transformer_v1.

Mirrors ``fleet_featurizer.py``:

1. ``PlanetFeaturizer`` — one struct per planet OR comet (unified format,
   ``is_comet`` distinguishes them). Self-state + sun-relative geometry
   + future-trajectory anchors. Fleet-derived summary fields are
   intentionally absent — those will be concatenated downstream after
   the fleet aggregator.

2. ``featurize_planets(obs, *, learner_slot, num_players, max_entities)``
   — pure-Python featurization. Walks ``obs['planets']`` (every planet,
   including those marked as comets via ``obs['comet_planet_ids']``),
   computes velocity / orbital geometry / future anchors, and returns
   ``(features, mask, records)``.

3. ``save_episode_planet_csv`` — writes one CSV per replay with both
   single-step labels (encoder pretraining) and lookahead labels
   (transformer-level aux: ``owner_t+5``, ``net_ships_t+5_signed_log``).

Why split planets and comets in the same featurizer: from the encoder's
point of view they're "objects with a position, owner, ships, and a
future trajectory" — the only real difference is *how* future positions
are computed (rule-based orbits vs comet path lookup). Putting them
through the same projection lets the downstream attention layer mix
them naturally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...physics_utils import (
    P_ID,
    P_OWNER,
    P_PRODUCTION,
    P_RADIUS,
    P_SHIPS,
    P_X,
    P_Y,
    ROTATION_RADIUS_LIMIT,
    SUN_CX,
    SUN_CY,
)
from ..paths import PLANET_DATASET_DIR
from .fleet_featurizer import (
    BOARD,
    MAX_SPEED,
    SHIPS_LOG_MAX,
    _infer_learner_slot_from_episode_path,
    signed_log1p,
)

# Spec constants
NUM_OWNER_SLOTS = 4
N_OWNER_PLUS_NEUTRAL = NUM_OWNER_SLOTS + 1     # 4 players + neutral
# Dense long-horizon anchors fed to the ENCODER as input features.
# Every 2 turns out to t+50 → 25 anchors × (dx, dy) = 50 dims, dwarfing
# the rest of the per-entity vector. Rationale: with sparse anchors
# the encoder had to internally extrapolate the trajectory before
# producing a token; with dense anchors out to t+50 the encoder gets a
# direct view of long-range structure and can use its capacity for
# semantics (owner / production / comet-vs-planet) instead. The
# trajectory decoder still predicts the next 10 turns (some of which
# overlap input anchors — easy by routing — and some are off-grid odd
# horizons that require interpolation).
FUTURE_HORIZONS = tuple(range(2, 51, 2))       # 2, 4, …, 50
N_FUTURE_ANCHORS = len(FUTURE_HORIZONS)
LOOKAHEAD = 5                                  # for owner_t+5 / net_ships_t+5 labels

# Trajectory extrapolation labels (encoder-pretrain).
# 10 contiguous horizons starting from t+1; per-step (dx, dy) and a
# per-step validity mask. The mask is 0 when the replay doesn't have a
# step at t+h or the entity isn't there at that step (comet vanished).
EXTRAP_HORIZONS = tuple(range(1, 11))
N_EXTRAP_HORIZONS = len(EXTRAP_HORIZONS)

# Normalization denominators
PRODUCTION_NORM = 5.0                          # production is small int, observed max ≈ 3
DIST_SUN_NORM = 71.0                           # max sun-distance ≈ √2·50 (board corner)
ANGVEL_NORM = 0.1                              # angular_velocity is small radians/turn
ANCHOR_DXY_NORM = 50.0                         # anchor displacements bounded by board half-width

# Buckets — calibrated below from the fleet quintile work where it
# applies; planet/comet positions span a slightly different envelope so
# we keep the same edges (board geometry is shared) and reuse them.
SUN_BUCKET_EDGES = (25.0, 33.0, 40.0, 46.0)    # 5 buckets
SPEED_BUCKET_EDGES = (1e-6, 1.0, 3.0)          # 4 buckets: stationary / slow / mid / fast

N_SUN_BUCKETS = len(SUN_BUCKET_EDGES) + 1
N_SPEED_BUCKETS = len(SPEED_BUCKET_EDGES) + 1
N_OWNER_DELTA_OUTCOMES = 3                     # stayed / lost / gained (signed change)


# Number of channels per trajectory anchor: dx, dy, valid_bit. The
# validity channel matters most for comets — without it, a "path ran
# out" anchor (value 0,0) is indistinguishable from "actually no
# motion" (value 0,0). The bit lets the encoder route around invalid
# horizons explicitly, which is the difference between a static planet
# and a dying comet.
TRAJ_CHANNELS = 3

# Scalar block dim — everything in ``to_vector`` *before* the trajectory
# anchors. The encoder splits the raw vector at this index: dims
# ``[0, SCALAR_DIM)`` are static/kinematic state; dims
# ``[SCALAR_DIM, PLANET_RAW_DIM)`` are reshaped to
# ``(N_FUTURE_ANCHORS, TRAJ_CHANNELS)`` for the 1D-conv trajectory branch.
SCALAR_DIM = (
    1                                  # is_comet
    + N_OWNER_PLUS_NEUTRAL             # owner one-hot relative (5)
    + 1                                # ships log-norm
    + 1                                # production normed
    + 2                                # x, y normed
    + 2                                # vx, vy normed
    + 1                                # distance_to_sun normed
    + 1                                # radial_velocity normed
    + 1                                # tangential_velocity normed
    + 2                                # sin/cos orbital angle
    + 1                                # angular_velocity normed
)

# Total raw feature dim — must equal what ``to_vector`` packs.
PLANET_RAW_DIM = SCALAR_DIM + TRAJ_CHANNELS * N_FUTURE_ANCHORS


# ---------- Records ----------
@dataclass
class PlanetFeaturizer:
    """All fields the encoder spec asks for, in one struct.

    Self-state + sun-relative kinematics + future-trajectory anchors.
    Floats fed to the encoder via ``to_vector``; ints are tracker
    metadata for debug / loss masks.
    """

    # ---- env-tracked ----
    planet_id: int
    is_comet: bool
    owner_id: int
    ships: int
    production: int
    x: float
    y: float
    radius: float
    current_tick: int
    valid: bool

    # ---- rule-derived ----
    vx: float
    vy: float
    distance_to_sun: float
    radial_velocity: float                     # +ve = receding from sun
    tangential_velocity: float                 # signed along orbital direction
    orbital_angle: float                       # atan2(y-50, x-50), radians
    angular_velocity: float                    # env-wide scalar (0 for comets)
    # length N_FUTURE_ANCHORS, each (dx, dy, valid). ``valid`` is 0 when
    # the underlying entity has no defined position at horizon h
    # (comet path runs out, episode ends, etc.); 1 otherwise. dx/dy are
    # 0 in invalid entries — the bit is what tells the encoder to
    # ignore them.
    future_dxy: list[tuple[float, float, float]]

    def to_vector(self, learner_slot: int, num_players: int) -> list[float]:
        """Pack the model-input subset into a flat list of length
        ``PLANET_RAW_DIM``. Owner is encoded *relative to* ``learner_slot``
        so the same weights serve any seat."""
        out: list[float] = []

        # entity type
        out.append(1.0 if self.is_comet else 0.0)

        # owner relative one-hot (4 player slots + neutral)
        owner_oh = [0.0] * N_OWNER_PLUS_NEUTRAL
        if self.owner_id == -1:
            owner_oh[NUM_OWNER_SLOTS] = 1.0       # neutral slot
        elif 0 <= self.owner_id < num_players:
            rel = (self.owner_id - learner_slot) % num_players
            owner_oh[rel] = 1.0
        # else: leave all zeros (defensive — shouldn't happen)
        out.extend(owner_oh)

        # ships log-norm
        out.append(math.log1p(max(0, self.ships)) / SHIPS_LOG_MAX)

        # production
        out.append(self.production / PRODUCTION_NORM)

        # position / velocity
        out.append(self.x / BOARD)
        out.append(self.y / BOARD)
        out.append(self.vx / MAX_SPEED)
        out.append(self.vy / MAX_SPEED)

        # sun geometry
        out.append(self.distance_to_sun / DIST_SUN_NORM)
        out.append(self.radial_velocity / MAX_SPEED)
        out.append(self.tangential_velocity / MAX_SPEED)
        out.append(math.sin(self.orbital_angle))
        out.append(math.cos(self.orbital_angle))
        out.append(self.angular_velocity / ANGVEL_NORM)

        # future anchors: (dx, dy, valid) at every horizon, normalized.
        # ``valid`` is left as 0/1 — the encoder's Conv1d treats it as
        # a third channel.
        for dx, dy, valid in self.future_dxy:
            out.append(dx / ANCHOR_DXY_NORM)
            out.append(dy / ANCHOR_DXY_NORM)
            out.append(valid)

        assert len(out) == PLANET_RAW_DIM, (
            f"PLANET_RAW_DIM mismatch: got {len(out)}, expected {PLANET_RAW_DIM}"
        )
        return out

    def to_label_dict(self) -> dict[str, float]:
        """Self-state-only label values (encoder pretraining set).

        Lookahead labels (``owner_t_plus_5``, ``net_ships_t_plus_5_*``)
        are filled in by ``save_episode_planet_csv`` because they
        require multi-turn replay context — they're appended to the CSV
        row alongside what's returned here.
        """
        speed = math.hypot(self.vx, self.vy)
        return {
            # categorical (single-step)
            "distance_to_sun_bucket": _distance_to_sun_bucket(self.distance_to_sun),
            "speed_bucket": _speed_bucket(speed),
            # regression (single-step reconstruction targets)
            "recon_x_norm": self.x / BOARD,
            "recon_y_norm": self.y / BOARD,
            "recon_vx_norm": self.vx / MAX_SPEED,
            "recon_vy_norm": self.vy / MAX_SPEED,
        }


# ---------- Bucketers ----------
def _distance_to_sun_bucket(d: float) -> int:
    for i, edge in enumerate(SUN_BUCKET_EDGES):
        if d < edge:
            return i
    return len(SUN_BUCKET_EDGES)


def _speed_bucket(speed: float) -> int:
    """4 buckets: stationary (≈0) / slow / mid / fast."""
    for i, edge in enumerate(SPEED_BUCKET_EDGES):
        if speed < edge:
            return i
    return len(SPEED_BUCKET_EDGES)


# ---------- Geometry helpers ----------
def _is_orbiting(x: float, y: float, radius: float, angular_velocity: float) -> bool:
    """Mirror of ``physics_utils._is_orbiting_xy``: a planet orbits iff
    its body is fully inside ``ROTATION_RADIUS_LIMIT`` from the sun and
    ``angular_velocity`` is non-zero."""
    if angular_velocity == 0:
        return False
    return math.hypot(x - SUN_CX, y - SUN_CY) + radius < ROTATION_RADIUS_LIMIT


def _rotate_around_sun(x: float, y: float, theta: float) -> tuple[float, float]:
    dx, dy = x - SUN_CX, y - SUN_CY
    c, s = math.cos(theta), math.sin(theta)
    return SUN_CX + dx * c - dy * s, SUN_CY + dx * s + dy * c


def _planet_future_xy(
    x: float, y: float, radius: float, angular_velocity: float, dt: int,
) -> tuple[float, float]:
    """Predict (x, y) at turn t+dt for a regular planet."""
    if not _is_orbiting(x, y, radius, angular_velocity):
        return x, y
    return _rotate_around_sun(x, y, angular_velocity * dt)


def _comet_future_xy(
    comet_meta: dict, dt: int,
) -> tuple[float, float] | None:
    """Predict (x, y) at turn t+dt for a comet, or None if its path
    runs out before then."""
    paths = comet_meta["paths"]                 # list[ list[(x, y)] ]
    path_index = comet_meta["path_index"]
    seat_index = comet_meta.get("_seat_index")  # filled in by caller
    if seat_index is None:
        return None
    target_idx = path_index + dt
    path = paths[seat_index]
    if 0 <= target_idx < len(path):
        return path[target_idx][0], path[target_idx][1]
    return None


# ---------- Featurization ----------
def featurize_planets(
    obs: Any,
    *,
    learner_slot: int,
    num_players: int = 4,
    max_entities: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, list[PlanetFeaturizer]]:
    """Build (features, mask, records) for the current turn.

    ``features``: (max_entities, PLANET_RAW_DIM), zero-padded.
    ``mask``:     (max_entities,) float, 1 where entity is real.
    ``records``:  list of ``PlanetFeaturizer``, length ≤ max_entities.

    ``max_entities`` is generous by default because planet count is
    bounded by env config (typically ≤ 36) and we do not want to drop
    any.
    """
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    planets = get("planets") or []
    comet_ids = set(get("comet_planet_ids") or [])
    angular_velocity = float(get("angular_velocity") or 0.0)
    current_tick = int(get("step", 0) or 0)

    # Index comets by planet_id so we can look up their path metadata.
    # Each comet entry contains ``planet_ids`` (list of pid in the same
    # comet group) and ``paths`` (one path per pid). We attach the
    # seat index so ``_comet_future_xy`` can recover the right path.
    raw_comets = get("comets") or []
    comet_by_pid: dict[int, dict] = {}
    for comet_meta in raw_comets:
        pids = comet_meta.get("planet_ids") or []
        for i, pid in enumerate(pids):
            entry = dict(comet_meta)
            entry["_seat_index"] = i
            comet_by_pid[int(pid)] = entry

    records: list[PlanetFeaturizer] = []
    for p in planets:
        pid = int(p[P_ID])
        owner = int(p[P_OWNER])
        x = float(p[P_X])
        y = float(p[P_Y])
        radius = float(p[P_RADIUS])
        ships = int(p[P_SHIPS])
        production = int(p[P_PRODUCTION])
        is_comet = pid in comet_ids

        # Velocity: orbiting planet → tangent to radius, magnitude = ω·r;
        # comet → finite-difference along its path; static planet → 0.
        if is_comet and pid in comet_by_pid:
            comet_meta = comet_by_pid[pid]
            here = _comet_future_xy(comet_meta, 0)
            nxt = _comet_future_xy(comet_meta, 1)
            if here is not None and nxt is not None:
                vx = nxt[0] - here[0]
                vy = nxt[1] - here[1]
            else:
                vx = vy = 0.0
            entity_av = 0.0
        elif _is_orbiting(x, y, radius, angular_velocity):
            # Tangent: rotate radial vector by 90°. v = ω × r in 2D
            # gives (-ω·dy, +ω·dx) for counter-clockwise positive ω.
            entity_av = angular_velocity
            vx = -entity_av * (y - SUN_CY)
            vy = entity_av * (x - SUN_CX)
        else:
            entity_av = 0.0
            vx = vy = 0.0

        # Sun-relative kinematics
        dx_sun = x - SUN_CX
        dy_sun = y - SUN_CY
        distance_to_sun = math.hypot(dx_sun, dy_sun)
        if distance_to_sun > 1e-6:
            r_hat_x = dx_sun / distance_to_sun
            r_hat_y = dy_sun / distance_to_sun
            radial_velocity = vx * r_hat_x + vy * r_hat_y
            tangential_velocity = vx * (-r_hat_y) + vy * r_hat_x
            orbital_angle = math.atan2(dy_sun, dx_sun)
        else:
            radial_velocity = 0.0
            tangential_velocity = 0.0
            orbital_angle = 0.0

        # Future trajectory anchors: (dx, dy, valid) per horizon. The
        # validity bit is what lets the encoder distinguish "this comet
        # will be gone by then" from "this static planet doesn't move."
        anchors: list[tuple[float, float, float]] = []
        for h in FUTURE_HORIZONS:
            if is_comet and pid in comet_by_pid:
                fut = _comet_future_xy(comet_by_pid[pid], h)
            else:
                fut = _planet_future_xy(x, y, radius, angular_velocity, h)
            if fut is None:
                anchors.append((0.0, 0.0, 0.0))
            else:
                anchors.append((fut[0] - x, fut[1] - y, 1.0))

        records.append(PlanetFeaturizer(
            planet_id=pid,
            is_comet=is_comet,
            owner_id=owner,
            ships=ships,
            production=production,
            x=x, y=y, radius=radius,
            current_tick=current_tick,
            valid=True,
            vx=vx, vy=vy,
            distance_to_sun=distance_to_sun,
            radial_velocity=radial_velocity,
            tangential_velocity=tangential_velocity,
            orbital_angle=orbital_angle,
            angular_velocity=entity_av,
            future_dxy=anchors,
        ))

    # Pack to tensors
    features = torch.zeros(max_entities, PLANET_RAW_DIM, dtype=torch.float32)
    mask = torch.zeros(max_entities, dtype=torch.float32)
    for i, rec in enumerate(records[:max_entities]):
        features[i] = torch.tensor(
            rec.to_vector(learner_slot=learner_slot, num_players=num_players),
            dtype=torch.float32,
        )
        mask[i] = 1.0
    return features, mask, records[:max_entities]


# ---------- Label registry ----------
# Single-entity-derivable — used for encoder pretraining.
ENCODER_PRETRAIN_LABELS: tuple[str, ...] = (
    "distance_to_sun_bucket",
    "speed_bucket",
    "recon_x_norm",
    "recon_y_norm",
    "recon_vx_norm",
    "recon_vy_norm",
    # Trajectory decoder head — output is (B, 2*N_EXTRAP_HORIZONS),
    # paired with ``extrap_mask`` for masked MSE.
    "extrap_trajectory",
)

# Need replay lookahead — used post-attention later, not for the
# single-entity encoder. Shipped to CSV so the same dataset feeds both.
TRANSFORMER_AUX_LABELS: tuple[str, ...] = (
    "owner_t_plus_5",
    "owner_changes_in_5",
    "net_ships_t_plus_5_signed_log",
    "valid_t_plus_5",
)

PLANET_LABEL_FIELDS: tuple[str, ...] = (
    ENCODER_PRETRAIN_LABELS + TRANSFORMER_AUX_LABELS
)

# Trajectory-extrapolation columns are written separately (one per
# horizon) — they don't appear in `PLANET_LABEL_FIELDS` because they're
# loaded as a single composite tensor by the dataset.

ENCODER_LABEL_HEADS: dict[str, dict[str, Any]] = {
    "distance_to_sun_bucket": {"type": "categorical", "n_classes": N_SUN_BUCKETS},
    "speed_bucket":           {"type": "categorical", "n_classes": N_SPEED_BUCKETS},
    "recon_x_norm":           {"type": "regression"},
    "recon_y_norm":           {"type": "regression"},
    "recon_vx_norm":          {"type": "regression"},
    "recon_vy_norm":          {"type": "regression"},
    # Trajectory decoder: predict (dx, dy) at each of the next 10 turns
    # relative to current pos. Loss is masked MSE — the per-step mask is
    # 0 when the entity isn't in the replay at t+h.
    "extrap_trajectory":      {
        "type": "masked_regression",
        "n_horizons": N_EXTRAP_HORIZONS,
        "out_dim": 2 * N_EXTRAP_HORIZONS,    # flat (dx,dy) ×10
        "mask_field": "extrap_mask",
    },
}


# Column-name helpers for the CSV writer/reader. Trajectory is stored
# flat (no nesting) so the CSV stays a plain table; the dataset/model
# will reshape to ``(n_horizons, 2)`` and ``(n_horizons,)`` at load.
def _extrap_target_col_names() -> list[str]:
    cols: list[str] = []
    for h in EXTRAP_HORIZONS:
        cols.append(f"extrap_dx_h{h:02d}")
        cols.append(f"extrap_dy_h{h:02d}")
    return cols


def _extrap_mask_col_names() -> list[str]:
    return [f"extrap_mask_h{h:02d}" for h in EXTRAP_HORIZONS]


EXTRAP_TARGET_COLS = tuple(_extrap_target_col_names())
EXTRAP_MASK_COLS = tuple(_extrap_mask_col_names())


# ---------- Lookahead label resolution ----------
def _planet_state_at(steps: list, t: int, target_pid: int) -> tuple[int, int] | None:
    """Return (owner, ships) for ``target_pid`` at step ``t``, or None
    if the entity isn't present in that step's planets list (comets can
    vanish)."""
    if t >= len(steps) or not steps[t]:
        return None
    obs = steps[t][0].get("observation") if isinstance(steps[t][0], dict) else None
    if not obs:
        return None
    for p in obs.get("planets") or []:
        if int(p[P_ID]) == target_pid:
            return int(p[P_OWNER]), int(p[P_SHIPS])
    return None


def _planet_pos_at(steps: list, t: int, target_pid: int) -> tuple[float, float] | None:
    """Return (x, y) for ``target_pid`` at step ``t``, or None if absent.

    Static and orbiting planets always exist; comets can vanish (when
    they leave the comet set the planet entry can disappear from the
    obs). The mask column flips to 0 in that case so the trajectory
    loss skips the row's missing horizons.
    """
    if t >= len(steps) or not steps[t]:
        return None
    obs = steps[t][0].get("observation") if isinstance(steps[t][0], dict) else None
    if not obs:
        return None
    for p in obs.get("planets") or []:
        if int(p[P_ID]) == target_pid:
            return float(p[P_X]), float(p[P_Y])
    return None


# ---------- CSV export ----------
def save_episode_planet_csv(
    episode_path: str | Path,
    out_path: str | Path | None = None,
    *,
    learner_slot: int | None = None,
    num_players: int | None = None,
    max_entities: int = 64,
    lookahead: int = LOOKAHEAD,
) -> Path:
    """Featurize every planet/comet in every turn of one episode and
    write a CSV including both encoder-pretrain labels and transformer-
    aux labels (the latter via replay lookahead by ``lookahead`` turns).

    Each emitted row = one (turn, entity) pair, columns:
        episode_id, turn, then ``f000..f{PLANET_RAW_DIM-1}``,
        then label columns from ``PLANET_LABEL_FIELDS``,
        then debug ints (planet_id, is_comet, owner_id, ships).

    Returns the resolved output path.
    """
    import csv
    import gzip
    import json

    episode_path = Path(episode_path)
    if out_path is None:
        out_path = PLANET_DATASET_DIR / f"planet_{episode_path.stem.split('.')[0]}.csv"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(episode_path, "rb") as fh:
        replay = json.load(fh)
    steps = replay.get("steps") or []
    if learner_slot is None:
        learner_slot = _infer_learner_slot_from_episode_path(episode_path)
    if num_players is None:
        num_players = len(steps[0]) if steps else 4
    episode_id = replay.get("id") or episode_path.stem

    feature_cols = [f"f{i:03d}" for i in range(PLANET_RAW_DIM)]
    # ``masked_regression`` labels (currently just ``extrap_trajectory``)
    # are stored across many CSV columns (``extrap_dx/dy_h*`` + masks),
    # not a single column — exclude them from the header here so the
    # column count matches what the writer actually emits below.
    label_cols = [
        n for n in PLANET_LABEL_FIELDS
        if ENCODER_LABEL_HEADS.get(n, {}).get("type") != "masked_regression"
    ]
    extrap_cols = list(EXTRAP_TARGET_COLS) + list(EXTRAP_MASK_COLS)
    meta_cols = ["planet_id", "is_comet", "owner_id", "ships"]
    header = [
        "episode_id", "turn",
        *feature_cols, *label_cols, *extrap_cols, *meta_cols,
    ]

    rows_written = 0
    with out_path.open("w", newline="") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(header)
        for t, step in enumerate(steps):
            if not step:
                continue
            seat = step[learner_slot] if len(step) > learner_slot else step[0]
            obs = seat.get("observation") if isinstance(seat, dict) else None
            if not obs:
                continue

            _, _, records = featurize_planets(
                obs,
                learner_slot=learner_slot,
                num_players=num_players,
                max_entities=max_entities,
            )
            for rec in records:
                vec = rec.to_vector(learner_slot=learner_slot, num_players=num_players)
                labels = rec.to_label_dict()

                # Lookahead labels: snapshot the same planet_id at t+lookahead.
                future = _planet_state_at(steps, t + lookahead, rec.planet_id)
                if future is None:
                    valid_t5 = 0
                    owner_t5 = NUM_OWNER_SLOTS         # collapse to neutral slot
                    owner_changes = 0
                    net_ships_signed_log = 0.0
                else:
                    valid_t5 = 1
                    fut_owner, fut_ships = future
                    if fut_owner == -1:
                        owner_t5 = NUM_OWNER_SLOTS     # neutral slot
                    elif 0 <= fut_owner < num_players:
                        owner_t5 = (fut_owner - learner_slot) % num_players
                    else:
                        owner_t5 = NUM_OWNER_SLOTS     # defensive
                    owner_changes = int(fut_owner != rec.owner_id)
                    net_ships_signed_log = signed_log1p(fut_ships - rec.ships)

                # Trajectory extrapolation: per-step (dx, dy) + mask.
                # dx/dy are normalized by ``ANCHOR_DXY_NORM`` so the
                # decoder regression target stays in roughly [-1, 1].
                extrap_dx_dy: list[float] = []
                extrap_masks: list[int] = []
                for h in EXTRAP_HORIZONS:
                    fut_xy = _planet_pos_at(steps, t + h, rec.planet_id)
                    if fut_xy is None:
                        extrap_dx_dy.extend([0.0, 0.0])
                        extrap_masks.append(0)
                    else:
                        extrap_dx_dy.append((fut_xy[0] - rec.x) / ANCHOR_DXY_NORM)
                        extrap_dx_dy.append((fut_xy[1] - rec.y) / ANCHOR_DXY_NORM)
                        extrap_masks.append(1)

                writer.writerow([
                    episode_id,
                    t,
                    *vec,
                    # encoder-pretrain
                    labels["distance_to_sun_bucket"],
                    labels["speed_bucket"],
                    labels["recon_x_norm"],
                    labels["recon_y_norm"],
                    labels["recon_vx_norm"],
                    labels["recon_vy_norm"],
                    # transformer-aux
                    owner_t5,
                    owner_changes,
                    net_ships_signed_log,
                    valid_t5,
                    # trajectory extrapolation (10 × dx,dy interleaved + 10 × mask)
                    *extrap_dx_dy,
                    *extrap_masks,
                    # debug
                    rec.planet_id,
                    int(rec.is_comet),
                    rec.owner_id,
                    rec.ships,
                ])
                rows_written += 1

    print(f"[planet_featurizer] wrote {rows_written} rows → {out_path}")
    return out_path
