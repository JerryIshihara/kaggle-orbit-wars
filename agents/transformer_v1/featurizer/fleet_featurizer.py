"""Per-fleet featurizer for transformer_v1.

Three pieces:

1. ``FleetFeaturizer`` — a structured per-fleet record carrying every
   field listed in the encoder spec (env-tracked + rule-derived). Used
   for inspection, debugging, and as the source of truth for the packed
   tensor. Some fields (``fleet_id``, ``source_planet_id``,
   ``target_planet_id``, ``launch_tick``, ``current_tick``, ``valid``)
   are tracker metadata, NOT fed to the network — only the float subset
   reaches the projection. ``to_vector`` packs that subset into the
   length-``FLEET_RAW_DIM`` list the encoder expects.

2. ``FleetTracker`` — cross-turn state. The env doesn't expose a fleet's
   launch tick; we infer it by recording the first frame each
   ``fleet_id`` appears in. The tracker also culls vanished fleets so
   the dict stays bounded across a 500-turn match.

3. ``featurize_fleets(obs, *, learner_slot, tracker, max_fleets)`` —
   pure-Python featurization that walks ``obs["fleets"]``, computes
   derived fields against ``obs["planets"]``, consults the tracker for
   ``launch_tick``, and returns ``(features, mask, records)`` ready for
   ``FleetEncoder``.

Why this layout: featurization is pure Python and runs on the host once
per turn (`O(F · P)`). The encoder is a stateless 1-layer projection that
runs on whatever device the rest of the model runs on. The tracker holds
inter-turn state and is owned by the agent, not the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ...physics_utils import (
    F_ANGLE,
    F_FROM,
    F_ID,
    F_OWNER,
    F_SHIPS,
    F_X,
    F_Y,
    P_ID,
    P_OWNER,
    P_PRODUCTION,
    P_SHIPS,
    _fleet_eta_to_planet,
    fleet_speed,
)

# Spec constants — kept in sync with the design doc.
NUM_OWNER_SLOTS = 4          # 1-, 2-, 3-, 4-player; relative slots
ETA_BUCKETS = (2.0, 5.0, 10.0, 20.0, 50.0)  # right edges; "none" is its own one-hot
N_ETA_BUCKETS = len(ETA_BUCKETS) + 2         # +1 for >50, +1 for "no target"
N_MISSION_TYPES = 5                          # see _MISSION_*

# Mission-type indices.
_MISSION_ATTACK = 0          # target is hostile (enemy-owned planet/comet)
_MISSION_REINFORCE = 1       # target is self-owned planet
_MISSION_CAPTURE_NEUTRAL = 2 # target is neutral planet
_MISSION_COMET_CHASE = 3     # target is a comet (regardless of owner)
_MISSION_TRANSIT_LOST = 4    # trajectory misses every planet — usually waste

# Normalization denominators.
BOARD = 100.0
MAX_SPEED = 6.0
MAX_TICKS = 500.0
SHIPS_LOG_MAX = math.log(5000.0)
ETA_NORM = 50.0
DIST_NORM = 200.0  # diagonal of the 100×100 board ≈ 141; round up.


# Total raw feature dim — recomputed below at module load to stay
# consistent with the dataclass-derived projection.
#
# Mid-level semantics (`eta_bucket`, `mission_type_coarse`) are NOT
# included as inputs — they are encoder-training labels. Including them
# as inputs would let the encoder cheat on those losses.
FLEET_RAW_DIM = (
    # kinematics: x, y, vx, vy, speed, sin(h), cos(h)
    7
    # ship count
    + 1
    # age, distance_to_target, eta_to_target
    + 3
    # owner relative (one-hot)
    + NUM_OWNER_SLOTS
    # target owner relative (one-hot) + neutral + none
    + NUM_OWNER_SLOTS + 2
    # target ships, is_target_comet, has_target
    + 3
)


# ---------- Records ----------
@dataclass
class FleetFeaturizer:
    """All fields the encoder spec asks for, in one struct.

    Float fields are flowed into the encoder via ``to_vector``. Integer
    metadata (``fleet_id``, ``source_planet_id``, ``target_planet_id``,
    ``launch_tick``, ``current_tick``, ``valid``) is kept for trackers,
    debug, and downstream loss masks. ``mission_type_coarse`` is held as
    an int index here and turned into a one-hot at vector time.
    """

    # ---- env-tracked ----
    fleet_id: int
    owner_id: int
    ships: int
    source_planet_id: int
    target_planet_id: int | None
    x: float
    y: float
    vx: float
    vy: float
    launch_tick: int
    current_tick: int
    valid: bool

    # ---- rule-derived ----
    speed: float
    heading: float
    age_since_launch: int
    distance_to_target_now: float | None  # None when no target
    eta_to_target: float | None
    eta_bucket: int                       # index 0..N_ETA_BUCKETS-1
    target_owner_now: int | None          # planet owner (-1 neutral, None no target)
    target_ships_now: int | None
    is_target_comet: bool
    mission_type_coarse: int              # one of _MISSION_*

    # ---- supervised labels (NOT inputs) ----
    # Mid-level semantics used to supervise the fleet encoder. Filled in
    # by `featurize_fleets`; consumers pull them via ``to_label_dict`` /
    # the CSV export. Kept on the same record so they line up row-for-row
    # with the feature vector.
    will_change_owner_on_arrival: bool    # arrival flips planet ownership
    capture_margin_on_arrival: int        # signed: ships - predicted_garrison
                                          #   reinforce: == ships (always >=0)
                                          #   attack:    >0 if captures, <=0 otherwise

    def to_vector(self, learner_slot: int, num_players: int) -> list[float]:
        """Pack the model-input subset into a flat list of length FLEET_RAW_DIM.

        Owners are encoded *relative to the learner slot* so the same
        weights serve any seat — `rel = (owner - learner) % num_players`.
        Neutral (``-1``) is its own slot in the target-owner one-hot.
        """
        out: list[float] = []

        # --- kinematics ---
        out.extend([
            self.x / BOARD,
            self.y / BOARD,
            self.vx / MAX_SPEED,
            self.vy / MAX_SPEED,
            self.speed / MAX_SPEED,
            math.sin(self.heading),
            math.cos(self.heading),
        ])

        # --- ships (log-norm) ---
        out.append(math.log1p(max(0, self.ships)) / SHIPS_LOG_MAX)

        # --- age / distance / eta ---
        out.append(min(1.0, self.age_since_launch / MAX_TICKS))
        out.append(
            self.distance_to_target_now / DIST_NORM
            if self.distance_to_target_now is not None else 0.0
        )
        out.append(
            min(1.0, self.eta_to_target / ETA_NORM)
            if self.eta_to_target is not None else 0.0
        )

        # --- owner relative one-hot ---
        if 0 <= self.owner_id < num_players:
            rel = (self.owner_id - learner_slot) % num_players
        else:
            rel = 0
        owner_oh = [0.0] * NUM_OWNER_SLOTS
        owner_oh[rel] = 1.0
        out.extend(owner_oh)

        # --- target owner one-hot (NUM_OWNER_SLOTS player slots + neutral + none) ---
        tgt_oh = [0.0] * (NUM_OWNER_SLOTS + 2)
        if self.target_owner_now is None:
            tgt_oh[NUM_OWNER_SLOTS + 1] = 1.0  # "no target"
        elif self.target_owner_now == -1:
            tgt_oh[NUM_OWNER_SLOTS] = 1.0      # "neutral"
        else:
            t_rel = (self.target_owner_now - learner_slot) % num_players
            tgt_oh[t_rel] = 1.0
        out.extend(tgt_oh)

        # --- target scalars ---
        out.append(
            math.log1p(max(0, self.target_ships_now)) / SHIPS_LOG_MAX
            if self.target_ships_now is not None else 0.0
        )
        out.append(1.0 if self.is_target_comet else 0.0)
        out.append(1.0 if self.target_planet_id is not None else 0.0)

        assert len(out) == FLEET_RAW_DIM, (
            f"FLEET_RAW_DIM mismatch: got {len(out)}, expected {FLEET_RAW_DIM}"
        )
        return out

    def to_label_dict(self) -> dict[str, float]:
        """Return all supervised labels for this fleet.

        Two groups, both written to the CSV in stable order
        (``FLEET_LABEL_FIELDS = ENCODER_PRETRAIN_LABELS +
        TRANSFORMER_AUX_LABELS``):

        * ``ENCODER_PRETRAIN_LABELS`` — derivable from the per-fleet
          input vector alone, used to pretrain the single-fleet encoder
          before the deep-set / transformer aggregator. Three families:
          (A) per-fleet typing, (B) kinematic/spatial preservation,
          (C) reconstruction (SSL). See ``ENCODER_LABEL_HEADS`` for the
          per-label loss type and class count.
        * ``TRANSFORMER_AUX_LABELS`` — depend on planet state and other
          in-flight fleets; only learnable post-attention. Kept in the
          CSV so the same dataset feeds the full-model aux heads later.
        """
        margin = int(self.capture_margin_on_arrival)
        return {
            # ---- A. per-fleet typing (categorical) ----
            "mission_type_coarse": int(self.mission_type_coarse),
            "target_relationship": _target_relationship(self),
            "ships_log_bucket": _ships_log_bucket(self.ships),
            "is_long_haul": int(
                self.eta_to_target is not None and self.eta_to_target > LONG_HAUL_ETA
            ),
            # ---- B. kinematic / spatial preservation (categorical) ----
            "eta_bucket": int(self.eta_bucket),
            "heading_sector": _heading_sector(self.heading),
            "quadrant": _quadrant(self.x, self.y),
            "distance_to_sun_bucket": _distance_to_sun_bucket(self.x, self.y),
            # ---- C. reconstruction (SSL, regression, normalized) ----
            "recon_x_norm": self.x / BOARD,
            "recon_y_norm": self.y / BOARD,
            "recon_vx_norm": self.vx / MAX_SPEED,
            "recon_vy_norm": self.vy / MAX_SPEED,
            "recon_ships_log": math.log1p(max(0, self.ships)) / SHIPS_LOG_MAX,
            # ---- transformer-level aux (NOT for single-fleet encoder) ----
            "will_change_owner_on_arrival": int(self.will_change_owner_on_arrival),
            "capture_margin_on_arrival": margin,
            "capture_margin_signed_log": signed_log1p(margin),
        }


def signed_log1p(x: float) -> float:
    """``sign(x) · log1p(|x|)`` — symmetric squash for signed regression
    targets with long tails. ``f(0) = 0``, monotone, smooth at 0."""
    return math.copysign(math.log1p(abs(x)), x)


# ---------- Label bucketers (single-fleet derivable) ----------
LONG_HAUL_ETA = 20.0  # eta threshold for the "is_long_haul" binary label
SHIPS_BUCKET_EDGES = (5, 20, 50, 200)  # 5 buckets total (incl. ≥200)
# Quintile edges (5 buckets) calibrated from ~223k observed fleets in
# data/encoders/. Earlier 3-bucket version put 98% of mass in the
# middle band; this split is roughly even (~20% each).
SUN_BUCKET_EDGES = (25.0, 33.0, 40.0, 46.0)
N_HEADING_SECTORS = 8
N_QUADRANTS = 4
N_TARGET_RELATIONSHIPS = 4              # none / neutral / friendly / hostile
N_SHIPS_LOG_BUCKETS = len(SHIPS_BUCKET_EDGES) + 1
N_SUN_BUCKETS = len(SUN_BUCKET_EDGES) + 1


def _target_relationship(rec: "FleetFeaturizer") -> int:
    """0=none, 1=neutral, 2=friendly, 3=hostile. Tighter than mission —
    drops the comet/lost split, isolates pure target-vs-self relation."""
    if rec.target_owner_now is None:
        return 0
    if rec.target_owner_now == -1:
        return 1
    if rec.target_owner_now == rec.owner_id:
        return 2
    return 3


def _ships_log_bucket(ships: int) -> int:
    """Coarse fleet-size class. Edges roughly log-spaced over the
    observed range; downstream deep-set wants to count small-vs-large
    fleets separately."""
    for i, edge in enumerate(SHIPS_BUCKET_EDGES):
        if ships < edge:
            return i
    return len(SHIPS_BUCKET_EDGES)


def _heading_sector(heading: float) -> int:
    """8-way compass sector. Accepts heading in any range — normalized
    to [0, 2π). Forces (sin, cos) info to survive the projection."""
    h = heading % (2 * math.pi)
    return int(h / (2 * math.pi / N_HEADING_SECTORS)) % N_HEADING_SECTORS


def _quadrant(x: float, y: float) -> int:
    """Board quadrant relative to sun center (50, 50).
    0=NE, 1=NW, 2=SW, 3=SE."""
    east = x >= 50.0
    north = y >= 50.0
    if north and east:
        return 0
    if north and not east:
        return 1
    if not north and not east:
        return 2
    return 3


def _distance_to_sun_bucket(x: float, y: float) -> int:
    """5-way radial bucket from the sun (50, 50). Edges chosen to
    approximate quintiles of the observed fleet-radius distribution
    (~min 11, max 71); orbits matter in this game so we want the
    encoder to keep radial position cleanly readable."""
    d = math.hypot(x - 50.0, y - 50.0)
    for i, edge in enumerate(SUN_BUCKET_EDGES):
        if d < edge:
            return i
    return len(SUN_BUCKET_EDGES)


# ---------- Label registry ----------
# Single-fleet-encoder pretraining labels — every entry must be a pure
# function of the 24-dim input vector.
ENCODER_PRETRAIN_LABELS: tuple[str, ...] = (
    # A. per-fleet typing
    "mission_type_coarse",
    "target_relationship",
    "ships_log_bucket",
    "is_long_haul",
    # B. kinematic / spatial preservation
    "eta_bucket",
    "heading_sector",
    "quadrant",
    "distance_to_sun_bucket",
    # C. reconstruction (SSL)
    "recon_x_norm",
    "recon_y_norm",
    "recon_vx_norm",
    "recon_vy_norm",
    "recon_ships_log",
)

# Labels that depend on planet state and/or sibling fleets — only
# learnable post-attention; not used for single-fleet pretraining.
TRANSFORMER_AUX_LABELS: tuple[str, ...] = (
    "will_change_owner_on_arrival",
    "capture_margin_on_arrival",
    "capture_margin_signed_log",
)

FLEET_LABEL_FIELDS: tuple[str, ...] = (
    ENCODER_PRETRAIN_LABELS + TRANSFORMER_AUX_LABELS
)

# Per-label head spec. Trainers read this to wire the right loss
# (CrossEntropy / BCE / MSE) and head dim.
ENCODER_LABEL_HEADS: dict[str, dict[str, Any]] = {
    # A
    "mission_type_coarse":     {"type": "categorical", "n_classes": N_MISSION_TYPES},
    "target_relationship":     {"type": "categorical", "n_classes": N_TARGET_RELATIONSHIPS},
    "ships_log_bucket":        {"type": "categorical", "n_classes": N_SHIPS_LOG_BUCKETS},
    "is_long_haul":            {"type": "binary"},
    # B
    "eta_bucket":              {"type": "categorical", "n_classes": N_ETA_BUCKETS},
    "heading_sector":          {"type": "categorical", "n_classes": N_HEADING_SECTORS},
    "quadrant":                {"type": "categorical", "n_classes": N_QUADRANTS},
    "distance_to_sun_bucket":  {"type": "categorical", "n_classes": N_SUN_BUCKETS},
    # C
    "recon_x_norm":            {"type": "regression"},
    "recon_y_norm":            {"type": "regression"},
    "recon_vx_norm":           {"type": "regression"},
    "recon_vy_norm":           {"type": "regression"},
    "recon_ships_log":         {"type": "regression"},
}


# ---------- Tracker (cross-turn state) ----------
@dataclass
class FleetTracker:
    """Maintains ``fleet_id → launch_tick`` across turns.

    The env doesn't expose a fleet's launch tick directly; we infer it
    by recording the first frame each ``fleet_id`` appears in.

    Usage:

        tracker = FleetTracker()
        for turn_obs in episode:
            tracker.observe(turn_obs)              # update bookkeeping
            feats, mask, recs = featurize_fleets(  # consume
                turn_obs, learner_slot=0,
                tracker=tracker, max_fleets=128,
            )
    """

    launch_tick: dict[int, int] = field(default_factory=dict)
    last_seen_tick: dict[int, int] = field(default_factory=dict)

    def observe(self, obs: Any) -> None:
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        step = int(get("step", 0) or 0)
        fleets = get("fleets") or []
        live: set[int] = set()
        for f in fleets:
            fid = int(f[F_ID])
            live.add(fid)
            if fid not in self.launch_tick:
                # First time we see this fleet → it launched this turn.
                self.launch_tick[fid] = step
            self.last_seen_tick[fid] = step
        # Cull entries for fleets that vanished — keeps the dict bounded
        # over a 500-turn match where O(thousands) of fleets come and go.
        # We keep them for one extra tick in case a caller needs to
        # post-mortem the launch tick of a just-resolved fleet.
        stale = [fid for fid, t in self.last_seen_tick.items()
                 if fid not in live and step - t > 1]
        for fid in stale:
            self.launch_tick.pop(fid, None)
            self.last_seen_tick.pop(fid, None)

    def reset(self) -> None:
        self.launch_tick.clear()
        self.last_seen_tick.clear()


# ---------- Featurization ----------
def _resolve_target(fleet, planets) -> tuple[int | None, float | None, Any]:
    """Find the planet this fleet's straight trajectory hits first.

    Returns ``(planet_id, eta, planet_tuple)`` or ``(None, None, None)``
    if the fleet's heading misses every planet.
    """
    best_eta = float("inf")
    best_pid: int | None = None
    best_planet = None
    for p in planets:
        eta = _fleet_eta_to_planet(fleet, p)
        if eta is None:
            continue
        if eta < best_eta:
            best_eta = eta
            best_pid = p[P_ID]
            best_planet = p
    if best_planet is None:
        return None, None, None
    return best_pid, best_eta, best_planet


def _eta_bucket(eta: float | None) -> int:
    """Bucket ETA into discrete classes; ``None`` → last bucket ("no target")."""
    if eta is None:
        return N_ETA_BUCKETS - 1
    for i, edge in enumerate(ETA_BUCKETS):
        if eta < edge:
            return i
    return N_ETA_BUCKETS - 2  # the ">50" bucket


def _target_state_before_arrival(
    target: Any,
    eta: float,
    fleets: list,
) -> tuple[int, int]:
    """Simulate the arrival timeline against ``target`` for every fleet
    in ``fleets`` whose ETA is strictly less than ``eta``, and return
    ``(owner, garrison)`` at that moment.

    Mirrors ``physics_utils.predict_garrison_at_arrival`` but exposes
    both owner and garrison and skips its "zero out if planet ends up
    ours" remap — the labeller needs the *true* future ownership state,
    not a launch-time-relative collapse, otherwise:

      * a target that's currently ours but stolen before arrival
      * a target that's currently hostile but captured by an ally first

    both produce wrong labels.
    """
    cur_owner = target[P_OWNER]
    pprod = target[P_PRODUCTION]
    garrison = float(target[P_SHIPS])

    arrivals: list[tuple[float, int, int]] = []
    for f in fleets or []:
        t = _fleet_eta_to_planet(f, target)
        if t is None or t >= eta:
            continue
        arrivals.append((t, f[F_OWNER], f[F_SHIPS]))
    arrivals.sort(key=lambda x: x[0])

    last_t = 0.0
    i = 0
    while i < len(arrivals):
        t = arrivals[i][0]
        if cur_owner >= 0:
            garrison += pprod * max(0.0, t - last_t)
        # Group simultaneous arrivals (within 0.5 turns).
        ships_per_owner: dict[int, int] = {}
        while i < len(arrivals) and arrivals[i][0] - t < 0.5:
            _, ow, sh = arrivals[i]
            ships_per_owner[ow] = ships_per_owner.get(ow, 0) + sh
            i += 1
        # Reinforcement from current owner first.
        if cur_owner in ships_per_owner:
            garrison += ships_per_owner.pop(cur_owner)
        # Resolve attackers among themselves: top vs second.
        if ships_per_owner:
            sorted_atk = sorted(ships_per_owner.items(), key=lambda kv: -kv[1])
            top_o, top_s = sorted_atk[0]
            second_s = sorted_atk[1][1] if len(sorted_atk) > 1 else 0
            if top_s == second_s:
                pass  # attackers tie, garrison untouched
            else:
                survivor = top_s - second_s
                if survivor > garrison:
                    cur_owner = top_o
                    garrison = float(survivor - garrison)
                else:
                    garrison = float(garrison - survivor)
        last_t = t

    # Production accrues up to the moment just before this fleet lands.
    if cur_owner >= 0 and cur_owner != -1:
        garrison += pprod * max(0.0, eta - last_t)

    return cur_owner, int(math.ceil(garrison))


def _classify_mission(
    target_planet: Any,
    target_pid: int | None,
    is_target_comet: bool,
    fleet_owner: int,
) -> int:
    """Coarse mission label — never authoritative, just a feature."""
    if target_pid is None:
        return _MISSION_TRANSIT_LOST
    if is_target_comet:
        return _MISSION_COMET_CHASE
    t_owner = target_planet[P_OWNER]
    if t_owner == fleet_owner:
        return _MISSION_REINFORCE
    if t_owner == -1:
        return _MISSION_CAPTURE_NEUTRAL
    return _MISSION_ATTACK


def featurize_fleets(
    obs: Any,
    *,
    learner_slot: int,
    tracker: FleetTracker,
    max_fleets: int = 128,
    num_players: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, list[FleetFeaturizer]]:
    """Build (features, mask, records) for the current turn.

    ``features``: Tensor[max_fleets, FLEET_RAW_DIM], zero-padded.
    ``mask``:     Tensor[max_fleets] (float), 1 where fleet is real.
    ``records``:  list of ``FleetFeaturizer`` (length ≤ max_fleets); the
                   first len(records) entries align with the live rows in
                   ``features``.

    Truncation: when ``len(fleets) > max_fleets``, fleets are kept by
    smallest ETA-to-our-nearest-planet (proxy for "matters soonest"); the
    rest are dropped silently. The drop policy lives here so callers can
    swap it without touching the model.
    """
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    raw_fleets = get("fleets") or []
    raw_planets = get("planets") or []
    comet_ids = set(get("comet_planet_ids") or [])
    current_tick = int(get("step", 0) or 0)

    # Resolve target + build records for every fleet.
    raw_records: list[FleetFeaturizer] = []
    for f in raw_fleets:
        fid = int(f[F_ID])
        owner = int(f[F_OWNER])
        x = float(f[F_X])
        y = float(f[F_Y])
        angle = float(f[F_ANGLE])
        from_pid = int(f[F_FROM])
        ships = int(f[F_SHIPS])
        speed = fleet_speed(ships)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        target_pid, eta, target_planet = _resolve_target(f, raw_planets)
        is_target_comet = target_pid in comet_ids
        target_owner = target_planet[P_OWNER] if target_planet is not None else None
        target_ships = target_planet[P_SHIPS] if target_planet is not None else None
        if target_planet is not None:
            dist = math.hypot(target_planet[2] - x, target_planet[3] - y)
        else:
            dist = None

        launch_tick = tracker.launch_tick.get(fid, current_tick)
        age = max(0, current_tick - launch_tick)

        mission = _classify_mission(target_planet, target_pid, is_target_comet, owner)

        # ---- labels: arrival outcome (predicted from current timeline) ----
        # Resolve the (owner, garrison) state of the target at the
        # moment THIS fleet is about to land — *after* every earlier
        # arrival has been simulated. This is what determines whether
        # we're reinforcing or attacking on touchdown, and is required
        # to label correctly when:
        #   - the target is currently ours but flipped before arrival
        #   - the target is currently hostile but captured by an ally
        #     before arrival
        # Comparing to the launch-time owner gets both cases wrong.
        will_flip = False
        capture_margin = 0
        if target_planet is not None and eta is not None:
            other_fleets = [g for g in raw_fleets if int(g[F_ID]) != fid]
            owner_before, garrison_before = _target_state_before_arrival(
                target_planet, eta, other_fleets,
            )
            if owner_before == owner:
                # Reinforcement at touchdown — ownership cannot change,
                # net contribution is +ships.
                will_flip = False
                capture_margin = ships
            else:
                # Attack / capture at touchdown. Signed margin;
                # ownership flips iff ships strictly beat the resolved
                # garrison.
                capture_margin = ships - garrison_before
                will_flip = ships > garrison_before

        raw_records.append(FleetFeaturizer(
            fleet_id=fid,
            owner_id=owner,
            ships=ships,
            source_planet_id=from_pid,
            target_planet_id=target_pid,
            x=x, y=y, vx=vx, vy=vy,
            launch_tick=launch_tick,
            current_tick=current_tick,
            valid=True,
            speed=speed,
            heading=angle,
            age_since_launch=age,
            distance_to_target_now=dist,
            eta_to_target=eta,
            eta_bucket=_eta_bucket(eta),
            target_owner_now=target_owner,
            target_ships_now=target_ships,
            is_target_comet=is_target_comet,
            mission_type_coarse=mission,
            will_change_owner_on_arrival=will_flip,
            capture_margin_on_arrival=capture_margin,
        ))

    # Truncation by relevance: keep fleets with smallest ETA toward any
    # of our planets. Falls back to fleet ETA (target ETA) if we own
    # nothing.
    if len(raw_records) > max_fleets:
        my_planets = [p for p in raw_planets if p[P_OWNER] == learner_slot]
        def _relevance(rec_idx: int) -> float:
            f = raw_fleets[rec_idx]
            if my_planets:
                e = min(
                    (et for et in (_fleet_eta_to_planet(f, p) for p in my_planets) if et is not None),
                    default=float("inf"),
                )
                return e
            return raw_records[rec_idx].eta_to_target or float("inf")
        order = sorted(range(len(raw_records)), key=_relevance)[:max_fleets]
        raw_records = [raw_records[i] for i in order]

    # Pack into (max_fleets, FLEET_RAW_DIM) tensor.
    features = torch.zeros(max_fleets, FLEET_RAW_DIM, dtype=torch.float32)
    mask = torch.zeros(max_fleets, dtype=torch.float32)
    for i, rec in enumerate(raw_records):
        features[i] = torch.tensor(
            rec.to_vector(learner_slot=learner_slot, num_players=num_players),
            dtype=torch.float32,
        )
        mask[i] = 1.0
    return features, mask, raw_records


# Default destination for encoder training data, resolved from the repo
# root (``…/agents/transformer_v1/featurizer/fleet_featurizer.py`` →
# ``…/data/encoders``). Kept beside ``data/replays`` so both training
# inputs and derived encoder data are co-located.
DEFAULT_ENCODER_DATA_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "encoders"
)


# ---------- Encoder-training CSV export ----------
def save_episode_fleet_csv(
    episode_path: str | Path,
    out_path: str | Path | None = None,
    *,
    learner_slot: int = 0,
    num_players: int | None = None,
    max_fleets: int = 1024,
) -> Path:
    """Featurize every fleet in every turn of one episode and write a CSV.

    The replay is the standard Kaggle ``.json.gz`` dump (``steps[t][i]``
    holds player ``i``'s view at turn ``t``). We use player 0's
    observation each turn — the env-tracked fields it exposes are
    identical across seats; only the *relative* owner one-hot in
    ``to_vector`` shifts, and we control that via ``learner_slot``.

    Each emitted row = one (turn, fleet) pair, columns:
        episode_id, turn, then ``f000..f{FLEET_RAW_DIM-1}`` (model input),
        then the label columns from ``FLEET_LABEL_FIELDS``,
        then debug ints (fleet_id, owner_id, source_planet_id,
        target_planet_id, ships).

    When ``out_path`` is ``None``, the file is written to
    ``DEFAULT_ENCODER_DATA_DIR / "fleet_<episode_stem>.csv"``. Returns
    the resolved output path. The output directory is created if
    missing. Pass ``max_fleets`` large enough that no truncation happens
    during dataset generation — we want every fleet labelled.
    """
    import csv
    import gzip
    import json

    episode_path = Path(episode_path)
    if out_path is None:
        out_path = DEFAULT_ENCODER_DATA_DIR / f"fleet_{episode_path.stem.split('.')[0]}.csv"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(episode_path, "rb") as fh:
        replay = json.load(fh)

    steps = replay.get("steps") or []
    # Player count: prefer the configuration block; fall back to the
    # first step's seat count.
    if num_players is None:
        # Each turn's outer list is one entry per seat — that's the
        # ground-truth player count for the replay.
        num_players = len(steps[0]) if steps else 4
    episode_id = replay.get("id") or episode_path.stem

    tracker = FleetTracker()
    feature_cols = [f"f{i:03d}" for i in range(FLEET_RAW_DIM)]
    label_cols = list(FLEET_LABEL_FIELDS)
    meta_cols = ["fleet_id", "owner_id", "source_planet_id",
                 "target_planet_id", "ships"]
    header = ["episode_id", "turn", *feature_cols, *label_cols, *meta_cols]

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
            tracker.observe(obs)
            _, _, records = featurize_fleets(
                obs,
                learner_slot=learner_slot,
                tracker=tracker,
                max_fleets=max_fleets,
                num_players=num_players,
            )
            for rec in records:
                vec = rec.to_vector(learner_slot=learner_slot, num_players=num_players)
                labels = rec.to_label_dict()
                writer.writerow([
                    episode_id,
                    t,
                    *vec,
                    *(labels[k] for k in label_cols),
                    rec.fleet_id,
                    rec.owner_id,
                    rec.source_planet_id,
                    rec.target_planet_id if rec.target_planet_id is not None else -1,
                    rec.ships,
                ])
                rows_written += 1
    print(f"[fleet_featurizer] wrote {rows_written} rows → {out_path}")
    return out_path
