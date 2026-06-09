"""PPO smoke runner — N episodes vs random_v1, one update, no BC anchor.

Bypasses the PairHead-depth precondition (this is a development smoke test,
not a production run). Existing supervised ckpts use `head_n_layers = 1` and
`conditioner_n_layers = 1`; the smoke run validates the rollout / GAE / PPO
update math end-to-end even though the actor heads are shallower than the
spec requires for a real PPO run.

Usage::

    python -m agents.transformer_v2.ppo.smoke \\
        --episodes 10 \\
        --ckpt data/runs/entity/.../entity_encoder_best.pt \\
        --opponent random_v1 \\
        --device mps

What this validates:

  * The full debug forward through PPOActorCritic (incl. legacy value_head on glob).
  * `sample_single_target` produces well-typed Actions.
  * the per-source projection emits valid env action triples.
  * Episodes terminate; per-step rewards are computed from score deltas.
  * `compute_advantages` runs on real rollouts (not synthetic).
  * `ppo_update_local` applies a gradient step on debug `value_head` + the
    two PairHead output heads.

Out of scope for the smoke: BC anchor (no pair cache load), distributed
training, eval gate, archive.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from agents.heuristic.physical_v4.agent import (
    PHASE_TABLE,
    compute_surplus,
    phase_of,
)
from agents.transformer_v2.featurizer.fleet_featurizer import FleetTracker
from agents.transformer_v2.featurizer.inference import featurize_observation
from agents.transformer_v2.pretrain.entity_encoder import (
    EntityPretrainModel,
    _PLANET_OWNER_NEUTRAL_IDX,
    _PLANET_OWNER_START_IDX,
    _build_entity_self_tokens,
    _load_encoders,
    build_pair_type_ids,
)
from agents.transformer_v2.runner import (
    DEFAULT_COMET_RUN_DIR,
    DEFAULT_FLEET_RUN_DIR,
    DEFAULT_PLANET_RUN_DIR,
)
from agents.transformer_v2.history import HISTORY_OFFSETS

from .actor_critic import PPOActorCritic
from .gae import Episode, compute_advantages
from .learner import PPOConfig, ppo_update_local
from .loss import PPOMinibatch
from .sampler import (
    MultiTargetAction,
    legality_masks,
    project_multi_target_to_env,
    sample_multi_target,
)


# --------------------------------------------------------------------------- #
# Per-step buffer                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class StepRecord:
    """One learner-turn record."""

    planet_features: torch.Tensor
    fleet_features: torch.Tensor
    fleet_target_idx: torch.Tensor
    fleet_source_idx: torch.Tensor
    fleet_owner_slot: torch.Tensor
    fleet_ships_log: torch.Tensor
    fleet_eta_norm: torch.Tensor
    fleet_mask: torch.Tensor
    planet_mask: torch.Tensor
    is_comet: torch.Tensor
    pair_type_ids: torch.Tensor
    # Rollout-time computed:
    pair_mask: torch.Tensor                # (P, P) bool
    source_mask: torch.Tensor               # (P,) bool
    action: MultiTargetAction
    value: float
    # Rewards / dones are filled later from env score history.
    reward: float = 0.0
    done: float = 0.0
    invalid_launch: int = 0
    emitted_launch: int = 0
    n_selected_targets: int = 0            # count of launching sources (tgt != s); for soft cap
    invalid_reasons: list = None           # list[str] — per-source failure reasons
    score_my: int = 0           # total ships of learner at end of THIS turn
    score_enemy_max: int = 0
    phi: float = 0.0            # calculated potential Φ(s)=Σwᵢsᵢ (design-A critic −Φ term)

    def __post_init__(self):
        if self.invalid_reasons is None:
            self.invalid_reasons = []


@dataclass
class EpisodeBuffer:
    seed: int
    learner_seat: int
    steps: list[StepRecord] = field(default_factory=list)
    winner: int | None = None


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #
def _resolve_l0_run_dir(
    *,
    label: str,
    ckpt_name: str,
    requested: Path | None,
    default_dir: Path,
) -> Path:
    """Resolve an L0 specialist run dir across local repo and Colab layouts."""
    if requested is not None:
        p = Path(requested)
        if (p / ckpt_name).exists():
            return p
        raise FileNotFoundError(
            f"{label} encoder checkpoint not found at {p / ckpt_name}. "
            f"Pass the directory containing {ckpt_name}, not the file path."
        )

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "ckpts" / label,
        Path("/content/orbit-wars") / "ckpts" / label,
        default_dir,
    ]
    seen: set[Path] = set()
    checked: list[Path] = []
    for p in candidates:
        p = p.resolve() if p.exists() else p
        if p in seen:
            continue
        seen.add(p)
        checked.append(p / ckpt_name)
        if (p / ckpt_name).exists():
            return p

    raise FileNotFoundError(
        f"{label} encoder checkpoint {ckpt_name} was not found. "
        "Checked:\n  - " + "\n  - ".join(str(p) for p in checked)
    )


def load_supervised(
    ckpt_path: Path,
    device: str,
    planet_run_dir: Path | None = None,
    fleet_run_dir: Path | None = None,
    comet_run_dir: Path | None = None,
) -> tuple[EntityPretrainModel, Any, Any, Any, dict]:
    """Mirror TransformerAgent.load — but return raw pieces for PPO wrapping."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    d_model = int(cfg.get("d_model", 256))
    n_steps = int(cfg.get("n_steps", 6))
    d_pair = int(cfg["d_pair"]) if "d_pair" in cfg else 128
    entity_n_heads = int(cfg.get("entity_n_heads", 4))
    cross_n_heads = int(cfg.get("cross_n_heads", 8))
    cross_n_layers = int(cfg.get("cross_n_layers", 2))
    dual_n_heads = int(cfg.get("dual_n_heads", 4))
    conditioner_n_layers = int(cfg.get("conditioner_n_layers", 1))
    head_n_layers = int(cfg.get("head_n_layers", 1))

    fleet_run_dir = _resolve_l0_run_dir(
        label="fleet",
        ckpt_name="fleet_encoder_best.pt",
        requested=fleet_run_dir,
        default_dir=DEFAULT_FLEET_RUN_DIR,
    )
    planet_run_dir = _resolve_l0_run_dir(
        label="planet",
        ckpt_name="planet_encoder_best.pt",
        requested=planet_run_dir,
        default_dir=DEFAULT_PLANET_RUN_DIR,
    )
    comet_run_dir = _resolve_l0_run_dir(
        label="comet",
        ckpt_name="comet_past_best.pt",
        requested=comet_run_dir,
        default_dir=DEFAULT_COMET_RUN_DIR,
    )

    fleet_enc, planet_enc, comet_enc = _load_encoders(
        fleet_run_dir,
        planet_run_dir,
        comet_run_dir,
        device=device,
    )

    cfg = dict(cfg)
    cfg["_l0_run_dirs"] = {
        "fleet": str(fleet_run_dir),
        "planet": str(planet_run_dir),
        "comet": str(comet_run_dir),
    }

    # Architecture flags — RESPECT the ckpt so a no-consolidator / skip-L3L4
    # actor does not get fresh RANDOM modules under strict=False. Mirror
    # TransformerAgent.load (runner.py:275-286): prefer saved config, else
    # detect from state_dict keys. Without this, a no-consolidator actor builds
    # a random PlayerConsolidator, forward_with_context returns garbage
    # player_state, and the PPO critic silently trains PairCompareHead on it
    # instead of falling back to the glob value_head for debug runs.
    model_state = ckpt["model"]
    _keys = list(model_state.keys())
    if "skip_l34" in cfg:
        skip_l34 = bool(cfg["skip_l34"])
    else:
        skip_l34 = not any(
            k.startswith("dual_role.") or k.startswith("joint_role.") for k in _keys
        )
    if "with_consolidator" in cfg:
        with_consolidator = bool(cfg["with_consolidator"])
    else:
        with_consolidator = any(k.startswith("consolidator.") for k in _keys)
    # Build the pretrained value heads too when the ckpt carries them, so the
    # reward-decomposition PPO critic (design A) can read the calibrated win
    # head and warm the shared value trunk. Detected from keys (mirrors the
    # consolidator handling above).
    with_value_heads = any(k.startswith("value_heads.") for k in _keys)
    # The pretrained value heads were built with this dropout (it changes the MLP
    # module layout: dropout!=0 inserts a Dropout layer, shifting the 2nd Linear's
    # index), so it MUST match for the 56 value tensors to load (else ~half miss).
    value_dropout = float(cfg.get("value_dropout", 0.0))

    model = EntityPretrainModel(
        d_model=d_model, n_steps=n_steps, d_pair=d_pair,
        entity_n_heads=entity_n_heads,
        cross_n_heads=cross_n_heads,
        cross_n_layers=cross_n_layers,
        dual_n_heads=dual_n_heads,
        conditioner_n_layers=conditioner_n_layers,
        head_n_layers=head_n_layers,
        skip_l34=skip_l34,
        with_consolidator=with_consolidator,
        with_value_heads=with_value_heads,
        value_dropout=value_dropout,
    )
    res = model.load_state_dict(model_state, strict=False)
    if with_value_heads:
        _vh_missing = [k for k in res.missing_keys if k.startswith("value_heads.")]
        if _vh_missing:
            print(f"[load_supervised] WARNING {len(_vh_missing)} value_heads.* keys "
                  f"missing — reward-decomp critic would read fresh heads", flush=True)
    return model.to(device), fleet_enc, planet_enc, comet_enc, cfg


# --------------------------------------------------------------------------- #
# Rollout episode via env.run                                                 #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# T=10 rollout history (per-env ring of featurized frames)                    #
# --------------------------------------------------------------------------- #
# Model-input fields stacked across the HISTORY_OFFSETS window (current frame
# last). pair_mask / source_mask / the sampled action stay single-frame.
_TEMPORAL_KEYS = (
    "planet_features", "fleet_features",
    "fleet_target_idx", "fleet_source_idx", "fleet_owner_slot",
    "fleet_ships_log", "fleet_eta_norm", "fleet_mask",
    "planet_mask", "is_comet", "pair_type_ids",
)
_ROUTING_KEYS = (
    "fleet_target_idx", "fleet_source_idx", "fleet_owner_slot",
    "fleet_ships_log", "fleet_eta_norm", "fleet_mask",
)


class _RolloutHistory:
    """Per-env ring of single-frame featurized frames (no batch dim).

    ``stack(step)`` returns each ``_TEMPORAL_KEYS`` field stacked along a new
    leading T axis over the ``HISTORY_OFFSETS`` window (oldest..current); missing
    early frames are zero-filled. The current frame is LAST, matching the
    supervised cache + ``CrossEntityAttention.step_embed[-T:]`` (current = last
    slot). ``window <= 1`` disables stacking (single-frame rollout).
    """

    def __init__(self, window: int):
        self.window = int(window)
        self.frames: dict[int, dict[str, torch.Tensor]] = {}

    def push(self, step: int, frame: dict[str, torch.Tensor]) -> None:
        self.frames[step] = frame
        cutoff = step - (HISTORY_OFFSETS[0] + 5)
        for s in [s for s in self.frames if s < cutoff]:
            del self.frames[s]

    def stack(self, step: int) -> dict[str, torch.Tensor]:
        cur = self.frames[step]
        out: dict[str, torch.Tensor] = {}
        for k in _TEMPORAL_KEYS:
            ref = cur[k]
            seq = [self.frames.get(step - off) for off in HISTORY_OFFSETS]
            out[k] = torch.stack(
                [(f[k] if f is not None else torch.zeros_like(ref)) for f in seq],
                dim=0,
            )  # (T, ...)
        return out


def _frame_from_batch(batch, is_comet_cur, pair_type_cur) -> dict[str, torch.Tensor]:
    """Single-frame (no batch dim) record of one turn's model-input fields."""
    f = {k: batch[k][0].detach() for k in (
        "planet_features", "fleet_features", "fleet_target_idx", "fleet_source_idx",
        "fleet_owner_slot", "fleet_ships_log", "fleet_eta_norm", "fleet_mask",
        "planet_mask",
    )}
    f["is_comet"] = is_comet_cur[0].detach()
    f["pair_type_ids"] = pair_type_cur[0].detach()
    return f


def _forward_with_history(policy, planet_enc, fleet_enc, comet_enc, history, step, device):
    """L0 + policy forward at the configured window. Returns ``(out, store)`` where
    ``store`` holds the per-field tensors for a StepRecord — single-frame ``(P,...)``
    when ``window <= 1``, the temporal stack ``(T,...)`` the learner replays when
    ``window > 1``. Passes the learner-relative owner one-hot so the critic's
    ``player_state`` matches the learner-side recompute (``_PPOWithL0``)."""
    cur = history.frames[step]
    store = {k: cur[k] for k in _TEMPORAL_KEYS} if history.window <= 1 else history.stack(step)
    mdl = {k: store[k].unsqueeze(0).to(device) for k in _TEMPORAL_KEYS}  # add batch dim
    comet_features = torch.zeros(
        list(mdl["planet_features"].shape[:-1]) + [comet_enc.input_dim],
        device=device, dtype=mdl["planet_features"].dtype,
    )
    comet_features[..., :18] = mdl["planet_features"][..., :18]
    planet_tok = planet_enc(mdl["planet_features"])
    comet_tok = comet_enc(comet_features)
    fleet_tok = fleet_enc(mdl["fleet_features"])
    entity_self = _build_entity_self_tokens(planet_tok, comet_tok, mdl["is_comet"])
    routing = {k: mdl[k] for k in _ROUTING_KEYS}
    owner_slice = mdl["planet_features"][
        ..., _PLANET_OWNER_START_IDX:_PLANET_OWNER_NEUTRAL_IDX + 1
    ]
    out = policy(
        entity_self, fleet_tok, routing, mdl["planet_mask"],
        is_comet=mdl["is_comet"], pair_type_ids=mdl["pair_type_ids"],
        planet_owner_oh=owner_slice,
    )
    return out, store


def _project_multi_target(
    action, *, source_mask, slot_to_pid, planets, min_launch,
):
    """Per-cell projection of a bernoulli_select_multinomial_alloc_v1 action into
    env moves. Each FIRED cell ``(s, t)`` (``action.select_mask[s, t]``) with
    ``alloc_counts[s, t] >= min_launch`` emits one launch of ``ships =
    alloc_counts[s, t]`` (the multinomial already routed N = source ships, so the
    counts ARE the launch sizes — held ships stay home via ``self_counts``).
    Cells below ``min_launch`` are dropped (invalid). Returns
    ``(env_moves, n_invalid, n_emitted, invalid_reasons)``."""
    env_moves: list[list[float]] = []
    n_invalid = 0
    n_emitted = 0
    invalid_reasons: list[str] = []
    if not planets:
        return env_moves, n_invalid, n_emitted, invalid_reasons
    by_id = {int(p.id): p for p in planets}
    src_rows = source_mask.nonzero(as_tuple=False).flatten().tolist()
    for s in src_rows:
        fired_cols = action.select_mask[s].nonzero(as_tuple=False).flatten().tolist()
        for t in fired_cols:
            ships = int(action.alloc_counts[s, t].item())
            if ships < int(min_launch):
                n_invalid += 1; invalid_reasons.append("min_launch"); continue
            src_pid = slot_to_pid[s]
            tgt_pid = slot_to_pid[t]
            if src_pid < 0 or tgt_pid < 0:
                n_invalid += 1; invalid_reasons.append("pad_slot"); continue
            src_planet = by_id.get(int(src_pid))
            tgt_planet = by_id.get(int(tgt_pid))
            if src_planet is None or tgt_planet is None:
                n_invalid += 1; invalid_reasons.append("no_planet"); continue
            angle = math.atan2(tgt_planet.y - src_planet.y, tgt_planet.x - src_planet.x)
            env_moves.append([int(src_pid), float(angle), int(ships)])
            n_emitted += 1
    return env_moves, n_invalid, n_emitted, invalid_reasons


def _finalize_step(obs, pid_to_idx, *, pair_logits, frac_loc, value, sigma_val,
                   store, learner_slot, num_players, noop_logit_bias,
                   select_logit_bias: float = 0.0):
    """Legality masks -> sample -> project to env moves -> StepRecord. This is the
    post-forward logic shared by the single-env closure (see agent_fn) and the
    batched rollout (batched_rollout.py) so both produce IDENTICAL records.
    ``store`` is the per-field tensors for the StepRecord (single-frame or the
    T-window stack). ``select_logit_bias`` shifts the selection Bernoulli logit
    down (fewer fires) and MUST match the learner's update-time bias. Returns
    ``(env_moves, StepRecord)``."""
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    raw_planets = get("planets") or []
    raw_fleets = get("fleets") or []
    step = int(get("step") or 0)
    _nb, defense_buffer, min_launch, _s, _fw, _et = PHASE_TABLE[phase_of(step)]
    P = int(pair_logits.shape[0])
    planet_owner_rel = torch.full((P,), 99, dtype=torch.long)
    planet_surplus = torch.zeros(P, dtype=torch.float32)
    source_ships = torch.zeros(P, dtype=torch.long)
    planet_exists = torch.zeros(P, dtype=torch.bool)
    slot_to_pid = [-1] * P
    planets: list = []
    fleets: list = []
    if raw_planets:
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]
        enemy_fleets = [f for f in fleets if f.owner != learner_slot and f.owner >= 0]
        for planet in planets:
            pid = int(planet.id)
            if pid in pid_to_idx:
                idx = pid_to_idx[pid]
                planet_exists[idx] = True
                slot_to_pid[idx] = pid
                planet_owner_rel[idx] = 0 if int(planet.owner) == learner_slot else 1
                planet_surplus[idx] = float(compute_surplus(planet, enemy_fleets, defense_buffer))
                source_ships[idx] = int(planet.ships)

    pair_mask, source_mask = legality_masks(
        planet_owner=planet_owner_rel, surplus=planet_surplus,
        planet_exists=planet_exists, min_launch=int(min_launch),
    )
    action = sample_multi_target(
        pair_logits, frac_loc, source_ships,
        pair_mask=pair_mask, source_mask=source_mask,
        select_logit_bias=select_logit_bias,
    )

    env_moves, n_invalid, n_emitted, invalid_reasons = _project_multi_target(
        action, source_mask=source_mask,
        slot_to_pid=slot_to_pid, planets=planets, min_launch=int(min_launch),
    )

    score_my = sum(p.ships for p in planets if int(p.owner) == learner_slot) if raw_planets else 0
    score_my += sum(f.ships for f in fleets if int(f.owner) == learner_slot) if raw_fleets else 0
    score_enemy_max = 0
    if raw_planets:
        for seat in range(num_players):
            if seat == learner_slot:
                continue
            s = sum(p.ships for p in planets if int(p.owner) == seat)
            s += sum(f.ships for f in fleets if int(f.owner) == seat)
            score_enemy_max = max(score_enemy_max, s)

    record = StepRecord(
        planet_features=store["planet_features"].detach().cpu(),
        fleet_features=store["fleet_features"].detach().cpu(),
        fleet_target_idx=store["fleet_target_idx"].detach().cpu(),
        fleet_source_idx=store["fleet_source_idx"].detach().cpu(),
        fleet_owner_slot=store["fleet_owner_slot"].detach().cpu(),
        fleet_ships_log=store["fleet_ships_log"].detach().cpu(),
        fleet_eta_norm=store["fleet_eta_norm"].detach().cpu(),
        fleet_mask=store["fleet_mask"].detach().cpu(),
        planet_mask=store["planet_mask"].detach().cpu(),
        is_comet=store["is_comet"].detach().cpu(),
        pair_type_ids=_current_pair_type_ids(store["pair_type_ids"]),
        pair_mask=pair_mask, source_mask=source_mask, action=action, value=value,
        invalid_launch=n_invalid, emitted_launch=n_emitted,
        n_selected_targets=int(action.diagnostics.get("n_fired_total", 0)),
        invalid_reasons=list(invalid_reasons),
        score_my=score_my, score_enemy_max=score_enemy_max,
    )
    return env_moves, record


def make_opponent_closure(
    *,
    opponent_policy: PPOActorCritic,
    planet_enc: Any,
    fleet_enc: Any,
    comet_enc: Any,
    opponent_slot: int,
    device: str,
    max_planets: int,
    max_fleets: int,
    sigma: float,
    noop_logit_bias: float,
    num_players: int = 2,
    history_window: int = 1,
    select_logit_bias: float = 0.0,
):
    """Build an env-compatible opponent fn that samples from a FROZEN
    PPOActorCritic using the same single_target_per_source_v1 contract as the
    learner. Opponent moves are not tracked for PPO, but it keeps its own
    ``history_window`` ring so a T=10 self-play opponent plays at the same
    window as the learner (no T-asymmetry in self-play). ``select_logit_bias``
    matches the learner's selection bias so the self-play opponent fires at the
    same confidence threshold.

    Use this for self-play: load a frozen snapshot of the learner (or any
    other PPOActorCritic), wrap it here, pass the returned callable as the
    opponent_fn to run_episode.
    """
    tracker = FleetTracker()
    comet_input_dim = comet_enc.input_dim
    opponent_policy.eval()
    history = _RolloutHistory(history_window)

    def opp_fn(obs):
        get0 = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        step = int(get0("step") or 0)
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=opponent_slot,
            tracker=tracker,
            num_players=num_players,
            max_planets=max_planets,
            max_fleets=max_fleets,
            device=device,
        )
        B, P, _ = batch["planet_features"].shape
        is_comet = batch["planet_features"][..., 0] > 0.5
        pair_type_cur = build_pair_type_ids(
            batch["planet_features"], batch["planet_mask"],
        )
        history.push(step, _frame_from_batch(batch, is_comet, pair_type_cur))

        with torch.inference_mode():
            out, _store = _forward_with_history(
                opponent_policy, planet_enc, fleet_enc, comet_enc, history, step, device,
            )
            pair_logits = out["pair_logits"][0].detach().cpu()
            frac_loc = out["frac_loc"][0].detach().cpu()
            sigma_val = float(out["sigma"].item())

        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        step = int(get("step") or 0)
        _nb, defense_buffer, min_launch, _s, _fw, _et = PHASE_TABLE[phase_of(step)]

        planet_owner_rel = torch.full((P,), 99, dtype=torch.long)
        planet_surplus = torch.zeros(P, dtype=torch.float32)
        source_ships = torch.zeros(P, dtype=torch.long)
        planet_exists = torch.zeros(P, dtype=torch.bool)
        slot_to_pid = [-1] * P
        planets: list = []
        if raw_planets:
            from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
            planets = [Planet(*p) for p in raw_planets]
            fleets = [Fleet(*f) for f in raw_fleets]
            enemy_fleets = [f for f in fleets if f.owner != opponent_slot and f.owner >= 0]
            for planet in planets:
                pid = int(planet.id)
                if pid in pid_to_idx:
                    idx = pid_to_idx[pid]
                    planet_exists[idx] = True
                    slot_to_pid[idx] = pid
                    planet_owner_rel[idx] = 0 if int(planet.owner) == opponent_slot else 1
                    surplus = compute_surplus(planet, enemy_fleets, defense_buffer)
                    planet_surplus[idx] = float(surplus)
                    source_ships[idx] = int(planet.ships)

        pair_mask, source_mask = legality_masks(
            planet_owner=planet_owner_rel,
            surplus=planet_surplus,
            planet_exists=planet_exists,
            min_launch=int(min_launch),
        )
        action = sample_multi_target(
            pair_logits, frac_loc, source_ships,
            pair_mask=pair_mask, source_mask=source_mask,
            select_logit_bias=select_logit_bias,
        )

        env_moves, _ni, _ne, _ir = _project_multi_target(
            action, source_mask=source_mask,
            slot_to_pid=slot_to_pid, planets=planets, min_launch=int(min_launch),
        )
        return env_moves

    return opp_fn


def _make_learner_closure(
    *,
    policy: PPOActorCritic,
    planet_enc: Any,
    fleet_enc: Any,
    comet_enc: Any,
    learner_slot: int,
    device: str,
    buffer: EpisodeBuffer,
    max_planets: int,
    max_fleets: int,
    sigma: float,
    noop_logit_bias: float,
    history_window: int = 1,
    num_players: int = 2,
    on_step=None,
    select_logit_bias: float = 0.0,
):
    """Build an env.run-compatible agent fn that samples policy actions and
    accumulates per-step records into ``buffer``. With ``history_window > 1`` the
    closure feeds the model the HISTORY_OFFSETS T-window and each StepRecord
    carries the stacked ``(T, ...)`` inputs the learner replays. ``num_players``
    is 2 or 4 — the orbit_wars env + featurizer support both.

    ``select_logit_bias`` shifts the selection Bernoulli logit down (fires fewer
    targets); it MUST equal the learner's update-time bias so the stored action's
    logprob is recomputed consistently (the PPO ratio is otherwise desynced).

    ``on_step`` (optional): called once per LEARNER turn AFTER the step's scores are
    computed, as ``on_step(step, num_players, score_my, score_enemy_max)``. No-op
    when None (zero overhead for existing callers); any error it raises is swallowed
    so live telemetry can never break a game."""
    tracker = FleetTracker()
    comet_input_dim = comet_enc.input_dim
    history = _RolloutHistory(history_window)

    def agent_fn(obs):
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        step = int(get("step") or 0)
        # 1. Featurize the current observation; buffer it for the T-window.
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=learner_slot,
            tracker=tracker,
            num_players=num_players,
            max_planets=max_planets,
            max_fleets=max_fleets,
            device=device,
        )
        B, P, _ = batch["planet_features"].shape
        is_comet = batch["planet_features"][..., 0] > 0.5
        pair_type_cur = build_pair_type_ids(
            batch["planet_features"], batch["planet_mask"],
        )
        history.push(step, _frame_from_batch(batch, is_comet, pair_type_cur))

        # 2-3. L0 + policy forward over the configured window (T=1 or T=10).
        #      ``store`` holds the per-field tensors the StepRecord keeps — the
        #      temporal stack the learner replays when history_window > 1.
        with torch.inference_mode():
            out, store = _forward_with_history(
                policy, planet_enc, fleet_enc, comet_enc, history, step, device,
            )
            pair_logits = out["pair_logits"][0].detach().cpu()        # (P, P)
            frac_loc = out["frac_loc"][0].detach().cpu()              # (P, P)
            value = float(out["value"][0].item())
            sigma_val = float(out["sigma"].item())

        # 4. Build legality masks from the obs.
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        step = int(get("step") or 0)
        _nb, defense_buffer, min_launch, _s, _fw, _et = PHASE_TABLE[phase_of(step)]

        planet_owner_rel = torch.full((P,), 99, dtype=torch.long)
        planet_surplus = torch.zeros(P, dtype=torch.float32)
        source_ships = torch.zeros(P, dtype=torch.long)
        planet_exists = torch.zeros(P, dtype=torch.bool)
        slot_to_pid = [-1] * P
        planets: list = []
        fleets: list = []

        if raw_planets:
            from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
            planets = [Planet(*p) for p in raw_planets]
            fleets = [Fleet(*f) for f in raw_fleets]
            enemy_fleets = [f for f in fleets if f.owner != learner_slot and f.owner >= 0]
            for planet in planets:
                pid = int(planet.id)
                if pid in pid_to_idx:
                    idx = pid_to_idx[pid]
                    planet_exists[idx] = True
                    slot_to_pid[idx] = pid
                    owner_rel = 0 if int(planet.owner) == learner_slot else 1
                    planet_owner_rel[idx] = owner_rel
                    surplus = compute_surplus(planet, enemy_fleets, defense_buffer)
                    planet_surplus[idx] = float(surplus)
                    source_ships[idx] = int(planet.ships)

        pair_mask, source_mask = legality_masks(
            planet_owner=planet_owner_rel,
            surplus=planet_surplus,
            planet_exists=planet_exists,
            min_launch=int(min_launch),
        )

        # 5. Sample action (bernoulli_select_multinomial_alloc_v1).
        action = sample_multi_target(
            pair_logits, frac_loc, source_ships,
            pair_mask=pair_mask,
            source_mask=source_mask,
            select_logit_bias=select_logit_bias,
        )

        # 6. Project to env moves (counts ARE launch sizes; held ships stay home).
        env_moves, n_invalid, n_emitted, invalid_reasons = _project_multi_target(
            action, source_mask=source_mask,
            slot_to_pid=slot_to_pid, planets=planets, min_launch=int(min_launch),
        )

        # 7. Compute per-step score (for reward shaping later).
        score_my = sum(p.ships for p in planets if int(p.owner) == learner_slot) if raw_planets else 0
        score_my += sum(f.ships for f in fleets if int(f.owner) == learner_slot) if raw_fleets else 0
        score_enemy_max = 0
        if raw_planets:
            for seat in range(num_players):
                if seat == learner_slot:
                    continue
                s = sum(p.ships for p in planets if int(p.owner) == seat)
                s += sum(f.ships for f in fleets if int(f.owner) == seat)
                score_enemy_max = max(score_enemy_max, s)

        # 8. Buffer the step. The 11 model-input fields carry the full T-window
        #    stack (T, ...) so the learner replays the exact temporal forward;
        #    pair_mask / source_mask / action stay single-frame (current turn).
        record = StepRecord(
            planet_features=store["planet_features"].detach().cpu(),
            fleet_features=store["fleet_features"].detach().cpu(),
            fleet_target_idx=store["fleet_target_idx"].detach().cpu(),
            fleet_source_idx=store["fleet_source_idx"].detach().cpu(),
            fleet_owner_slot=store["fleet_owner_slot"].detach().cpu(),
            fleet_ships_log=store["fleet_ships_log"].detach().cpu(),
            fleet_eta_norm=store["fleet_eta_norm"].detach().cpu(),
            fleet_mask=store["fleet_mask"].detach().cpu(),
            planet_mask=store["planet_mask"].detach().cpu(),
            is_comet=store["is_comet"].detach().cpu(),
            pair_type_ids=_current_pair_type_ids(store["pair_type_ids"]),
            pair_mask=pair_mask,
            source_mask=source_mask,
            action=action,
            value=value,
            invalid_launch=n_invalid,
            emitted_launch=n_emitted,
            n_selected_targets=int(action.diagnostics.get("n_fired_total", 0)),
            invalid_reasons=list(invalid_reasons),
            score_my=score_my,
            score_enemy_max=score_enemy_max,
        )
        buffer.steps.append(record)

        # 9. Live per-step telemetry (best-effort; never break the game).
        if on_step is not None:
            try:
                on_step(step, num_players, score_my, score_enemy_max)
            except Exception:  # noqa: BLE001 — telemetry-grade
                pass
        return env_moves

    return agent_fn


def run_episode(
    *,
    policy: PPOActorCritic,
    planet_enc: Any,
    fleet_enc: Any,
    comet_enc: Any,
    seed: int,
    learner_seat: int,
    opponent_id: str | None,
    opponent_fn=None,
    device: str,
    max_planets: int,
    max_fleets: int,
    sigma: float,
    noop_logit_bias: float,
    # Soft cap on target count — penalises excess(k - k_max)^2 per step.
    target_cap_k_max: int = 4,
    target_cap_lambda: float = 0.0,
    history_window: int = 1,
    num_players: int = 2,
    opponent_policy=None,
    on_step=None,
    select_logit_bias: float = 0.0,
) -> EpisodeBuffer:
    """Roll out one learner episode (2P or 4P — orbit_wars supports both).

    Opponent selection (exactly one of):
      * ``opponent_policy``: a frozen ``PPOActorCritic`` snapshot — self-play; a
        fresh per-seat closure is built for each of the num_players-1 opponents.
      * ``opponent_id``: a registered heuristic agent id (random_v1, …).
      * ``opponent_fn``: a single env-compatible callable (2P back-compat only).

    ``on_step`` (optional): forwarded to the learner closure; called once per
    LEARNER turn as ``on_step(step, num_players, score_my, score_enemy_max)`` with
    cheap stats. No-op when None — zero overhead and unchanged behaviour for every
    existing caller. Used by the pool rollout to stream per-core live progress.

    ``select_logit_bias`` (multi-target contract): shifts the selection Bernoulli
    logit down so the policy fires fewer (more confident) targets. Forwarded to
    BOTH the learner and the self-play opponent closures, and MUST equal the
    learner's update-time bias (``PPOConfig.select_logit_bias``). Default 0.0
    leaves behaviour identical to before.
    """
    from kaggle_environments import make
    import agents as _agents

    buffer = EpisodeBuffer(seed=seed, learner_seat=learner_seat)

    learner_fn = _make_learner_closure(
        policy=policy, planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
        learner_slot=learner_seat, device=device, buffer=buffer,
        max_planets=max_planets, max_fleets=max_fleets,
        sigma=sigma, noop_logit_bias=noop_logit_bias,
        history_window=history_window, num_players=num_players,
        on_step=on_step, select_logit_bias=select_logit_bias,
    )
    n_opp = sum(x is not None for x in (opponent_fn, opponent_id, opponent_policy))
    if n_opp != 1:
        raise ValueError("pass exactly one of opponent_fn / opponent_id / opponent_policy")
    if opponent_fn is not None and num_players != 2:
        raise ValueError("opponent_fn is 2P-only; use opponent_policy/opponent_id for 4P")

    # One agent fn per seat: learner at ``learner_seat``; the other
    # num_players-1 seats are opponents (per-seat self-play snapshot, heuristic,
    # or the single 2P back-compat opponent_fn).
    fns = [None] * num_players
    fns[learner_seat] = learner_fn
    for seat in range(num_players):
        if seat == learner_seat:
            continue
        if opponent_policy is not None:
            fns[seat] = make_opponent_closure(
                opponent_policy=opponent_policy, planet_enc=planet_enc,
                fleet_enc=fleet_enc, comet_enc=comet_enc, opponent_slot=seat,
                device=device, max_planets=max_planets, max_fleets=max_fleets,
                sigma=sigma, noop_logit_bias=noop_logit_bias, num_players=num_players,
                select_logit_bias=select_logit_bias,
            )
        elif opponent_id is not None:
            fns[seat] = _agents.Agent(id=opponent_id).fn
        else:
            fns[seat] = opponent_fn

    env = make("orbit_wars", configuration={"seed": seed} if seed is not None else {})
    env.run(fns)

    # Determine winner from final rewards.
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    ranked = [r if r is not None else float("-inf") for r in rewards]
    buffer.winner = int(max(range(len(ranked)), key=lambda i: ranked[i]))

    # Compute per-step rewards: potential-based shaping + terminal +/- 1.
    # Soft cap on excess targets per source ("Known problems" #3 mitigation):
    # subtract `target_cap_lambda * max(0, k - k_max)^2`. NEVER truncates the
    # sampled action — the penalty just shapes future logit distributions.
    # Pull per-step scores from the buffered records (already computed at
    # turn time, so they are the LEARNER-turn snapshots).
    prev_potential = 0.0
    for i, step in enumerate(buffer.steps):
        potential = float(step.score_my) - float(step.score_enemy_max)
        dense = (potential - prev_potential) / 200.0
        dense = max(-0.02, min(0.02, dense))
        excess = max(0, step.n_selected_targets - int(target_cap_k_max))
        cap_penalty = float(target_cap_lambda) * (excess * excess)
        r = dense - 0.01 * float(step.invalid_launch) - cap_penalty
        prev_potential = potential
        step.reward = r
        step.done = 0.0

    if buffer.steps:
        buffer.steps[-1].done = 1.0
        terminal = 1.0 if buffer.winner == learner_seat else (-1.0 if buffer.winner != -1 else 0.0)
        buffer.steps[-1].reward += terminal

    # Best-effort per-game FD release: under a long fork-pool run the kaggle env
    # (and any agent sub-process it spawned) can leak file descriptors per game,
    # which compounds toward the soft RLIMIT_NOFILE. Drop our reference to the env
    # AFTER winner/rewards are read so its objects (and any FDs they hold) become
    # collectable now rather than at the next GC pause. If a future env grows an
    # explicit close()/cleanup(), call it here too — guarded so it never raises.
    # (The real safety is pool_rollout's file_system sharing strategy + the raised
    # FD limit; this just keeps per-game pressure low.)
    for _m in ("close", "cleanup"):
        _fn = getattr(env, _m, None)
        if callable(_fn):
            try:
                _fn()
            except Exception:  # noqa: BLE001
                pass
    del env

    return buffer


# --------------------------------------------------------------------------- #
# Buffer → PPOMinibatch tensor packing                                        #
# --------------------------------------------------------------------------- #
def episodes_to_ppo(
    episodes: list[EpisodeBuffer], *, minibatch_size: int, device: str,
) -> tuple[list[Episode], list[PPOMinibatch]]:
    """Convert per-step records into Episode (for GAE) and PPOMinibatch lists.

    All tensors moved to ``device`` for the update step.
    """
    packed = [_pack_episode_for_ppo(ep) for ep in episodes if ep.steps]
    return _packed_episodes_to_ppo(
        [p for p in packed if p is not None],
        minibatch_size=minibatch_size,
        device=device,
    )


_PACK_ROUTING_KEYS = (
    "fleet_target_idx", "fleet_source_idx", "fleet_owner_slot",
    "fleet_ships_log", "fleet_eta_norm", "fleet_mask",
)


def _current_pair_type_ids(x: torch.Tensor) -> torch.Tensor:
    """Return compact current-frame pair type ids for PPO replay/archive.

    Pair type ids have only 27 classes and the entity model consumes only the
    current `(P, P)` matrix even when a temporal `(T, P, P)` tensor is supplied.
    Keeping the whole T-window here multiplies rollout packing memory by T.
    """
    if x.dim() == 3:
        x = x[-1]
    return x.detach().cpu().to(torch.uint8)


def _pack_episode_for_ppo(ep: EpisodeBuffer) -> dict | None:
    """Stack one completed EpisodeBuffer into CPU tensors.

    This is intentionally top-level and pickleable so train_local_trial can run it
    in a small worker pool. The online rollout still featurizes each step before
    acting; this stage prepares completed trajectory tensors for GAE/minibatching.
    """
    if not ep.steps:
        return None

    steps = ep.steps

    def _stack(attr: str) -> torch.Tensor:
        return torch.stack([getattr(s, attr) for s in steps]).cpu()

    feats = {
        "planet_features": _stack("planet_features"),
        "fleet_features": _stack("fleet_features"),
        "planet_mask": _stack("planet_mask"),
        "is_comet": _stack("is_comet"),
        "pair_type_ids": torch.stack([
            _current_pair_type_ids(s.pair_type_ids) for s in steps
        ]).cpu(),
        "routing": {k: _stack(k) for k in _PACK_ROUTING_KEYS},
        "phi": torch.tensor(
            [float(getattr(s, "phi", 0.0)) for s in steps],
            dtype=torch.float32,
        ),
    }

    return {
        "values": torch.tensor([s.value for s in steps], dtype=torch.float32),
        "rewards": torch.tensor([s.reward for s in steps], dtype=torch.float32),
        "dones": torch.tensor([s.done for s in steps], dtype=torch.float32),
        "feats": feats,
        "pair_mask": _stack("pair_mask"),
        "source_mask": _stack("source_mask"),
        # bernoulli_select_multinomial_alloc_v1 action fields.
        "select_mask": torch.stack([s.action.select_mask for s in steps]).cpu().bool(),
        "alloc_counts": torch.stack([s.action.alloc_counts for s in steps]).cpu().long(),
        "self_counts": torch.stack([s.action.self_counts for s in steps]).cpu().long(),
        "old_logp": torch.tensor(
            [float(s.action.logprob.item()) for s in steps],
            dtype=torch.float32,
        ),
    }


def _cat_tensors(xs: list[torch.Tensor], *, device: str) -> torch.Tensor:
    return torch.cat(xs, dim=0).to(device)


def _packed_episodes_to_ppo(
    packed: list[dict], *, minibatch_size: int, device: str,
) -> tuple[list[Episode], list[PPOMinibatch]]:
    ep_objs = [
        Episode(values=p["values"], rewards=p["rewards"], dones=p["dones"])
        for p in packed
    ]
    compute_advantages(ep_objs, normalize=True)
    if not ep_objs:
        return ep_objs, []

    adv = _cat_tensors([e.advantages for e in ep_objs if e.advantages is not None],
                       device=device)
    ret = _cat_tensors([e.returns for e in ep_objs if e.returns is not None],
                       device=device)
    N = int(adv.shape[0])
    if N == 0:
        return ep_objs, []

    feats = {
        "planet_features": _cat_tensors(
            [p["feats"]["planet_features"] for p in packed], device=device),
        "fleet_features": _cat_tensors(
            [p["feats"]["fleet_features"] for p in packed], device=device),
        "planet_mask": _cat_tensors(
            [p["feats"]["planet_mask"] for p in packed], device=device),
        "is_comet": _cat_tensors(
            [p["feats"]["is_comet"] for p in packed], device=device),
        "pair_type_ids": _cat_tensors(
            [p["feats"]["pair_type_ids"] for p in packed], device=device),
        "routing": {
            k: _cat_tensors([p["feats"]["routing"][k] for p in packed], device=device)
            for k in _PACK_ROUTING_KEYS
        },
        # Calculated potential Φ(s) per step — the design-A critic subtracts it
        # (−Φ). Carried through the minibatch so the update's critic forward gets it.
        "phi": _cat_tensors([p["feats"]["phi"] for p in packed], device=device),
    }
    pair_mask = _cat_tensors([p["pair_mask"] for p in packed], device=device)
    source_mask = _cat_tensors([p["source_mask"] for p in packed], device=device)
    select_mask = _cat_tensors([p["select_mask"] for p in packed], device=device)
    alloc_counts = _cat_tensors([p["alloc_counts"] for p in packed], device=device)
    self_counts = _cat_tensors([p["self_counts"] for p in packed], device=device)
    old_logp = _cat_tensors([p["old_logp"] for p in packed], device=device)

    minibatches: list[PPOMinibatch] = []
    for start in range(0, N, minibatch_size):
        end = min(start + minibatch_size, N)
        idx = slice(start, end)
        feats_slice = {
            k: (v[idx] if isinstance(v, torch.Tensor)
                else {kk: vv[idx] for kk, vv in v.items()})
            for k, v in feats.items()
        }
        # The minibatch carries raw features; train-time _PPOWithL0 runs frozen L0.
        minibatches.append(PPOMinibatch(
            feats=feats_slice,
            pair_mask=pair_mask[idx],
            source_mask=source_mask[idx],
            select_mask=select_mask[idx],
            alloc_counts=alloc_counts[idx],
            self_counts=self_counts[idx],
            old_logp=old_logp[idx],
            adv=adv[idx],
            returns=ret[idx],
            noop_logit_bias=0.0,
        ))
    return ep_objs, minibatches


def episodes_to_ppo_parallel(
    episodes: list[EpisodeBuffer],
    *,
    minibatch_size: int,
    device: str,
    num_workers: int,
) -> tuple[list[Episode], list[PPOMinibatch]]:
    """Parallel completed-trajectory packing, then global GAE/minibatching.

    ``num_workers`` is for the post-rollout stage only. It does not change the
    online policy featurization used to act inside each game step.
    """
    num_workers = int(num_workers)
    todo = [ep for ep in episodes if ep.steps]
    if num_workers <= 1 or len(todo) <= 1:
        return episodes_to_ppo(todo, minibatch_size=minibatch_size, device=device)
    if str(device).startswith("cuda"):
        # Forking after CUDA initialization is unsafe; keep GPU update runs serial.
        return episodes_to_ppo(todo, minibatch_size=minibatch_size, device=device)

    import concurrent.futures as _fut
    import multiprocessing as _mp

    ctx = _mp.get_context("fork")
    with _fut.ProcessPoolExecutor(
        max_workers=min(num_workers, len(todo)),
        mp_context=ctx,
    ) as ex:
        packed = list(ex.map(_pack_episode_for_ppo, todo, chunksize=1))
    return _packed_episodes_to_ppo(
        [p for p in packed if p is not None],
        minibatch_size=minibatch_size,
        device=device,
    )


# --------------------------------------------------------------------------- #
# Train-time wrapper that runs L0 then PPOActorCritic                          #
# --------------------------------------------------------------------------- #
class _PPOWithL0(torch.nn.Module):
    """Wrap PPOActorCritic so its `forward(**feats)` accepts raw features.

    The PPO loss expects to call ``policy(**mb.feats)`` and get the
    `(value, pair_logits, frac_loc, ...)` dict back. PPOActorCritic.forward
    takes POST-L0 tokens (planet_tokens, fleet_tokens). This wrapper runs
    the frozen L0 specialists first.
    """

    def __init__(self, policy: PPOActorCritic, planet_enc, fleet_enc, comet_enc):
        super().__init__()
        self.policy = policy
        self.planet_enc = planet_enc
        self.fleet_enc = fleet_enc
        self.comet_enc = comet_enc
        self.value_head = policy.value_head    # for the learner's optimizer
        self.pair_compare = getattr(policy, "pair_compare", None)
        self.entity_model = policy.entity_model
        self.sigma = policy.sigma

    def freeze_for_phase(self, phase: int):
        return self.policy.freeze_for_phase(phase)

    def forward(self, planet_features, fleet_features, planet_mask,
                 is_comet, pair_type_ids, routing, phi=None):
        # L0 frozen forward. Handles both rollout-time (B, P, dim) and
        # BC-anchor (B, T, P, dim) shapes — comet_features must match
        # planet_features' leading shape exactly.
        with torch.no_grad():
            comet_shape = list(planet_features.shape[:-1]) + [self.comet_enc.input_dim]
            comet_features = torch.zeros(
                comet_shape,
                device=planet_features.device,
                dtype=planet_features.dtype,
            )
            comet_features[..., :18] = planet_features[..., :18]
            planet_tok = self.planet_enc(planet_features)
            comet_tok = self.comet_enc(comet_features)
            fleet_tok = self.fleet_enc(fleet_features)
            entity_self = _build_entity_self_tokens(planet_tok, comet_tok, is_comet)

        # PlayerConsolidator needs the learner-relative owner one-hot. It
        # lives at planet_features[..., 1:1+ENTITY_N_OWNER_CLASSES] per the
        # PlanetFeaturizer layout. Slice (no grad needed — additive only).
        owner_slice = planet_features[
            ..., _PLANET_OWNER_START_IDX:_PLANET_OWNER_NEUTRAL_IDX + 1
        ]

        return self.policy(
            entity_self, fleet_tok, routing, planet_mask,
            is_comet=is_comet, pair_type_ids=pair_type_ids,
            planet_owner_oh=owner_slice, phi=phi,
        )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path,
                         help="Supervised PairHead .pt to bootstrap from.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--opponent", default="random_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-heads", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.10)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--noop-logit-bias", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=1024)
    args = parser.parse_args()

    device = args.device
    print(f"[smoke] device={device} ckpt={args.ckpt}", flush=True)
    t0 = time.time()
    entity_model, fleet_enc, planet_enc, comet_enc, cfg = load_supervised(
        args.ckpt, device,
    )
    print(f"[smoke] loaded ckpt in {time.time()-t0:.1f}s "
          f"(d_model={cfg.get('d_model')}, n_steps={cfg.get('n_steps')}, "
          f"conditioner_n_layers={cfg.get('conditioner_n_layers',1)}, "
          f"head_n_layers={cfg.get('head_n_layers',1)})", flush=True)

    policy = PPOActorCritic(
        entity_model,
        sigma=args.sigma,
        allow_debug_glob_critic=True,
    ).to(device)
    breakdown = policy.freeze_for_phase(0)
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[smoke] freeze_for_phase(0) → trainable={n_trainable:,} | "
          f"by group: {breakdown}", flush=True)

    # ---- Rollout ----
    episodes: list[EpisodeBuffer] = []
    t_roll = time.time()
    for i in range(args.episodes):
        seed = args.seed_base + i
        learner_seat = i % 2
        t_ep = time.time()
        try:
            ep = run_episode(
                policy=policy, planet_enc=planet_enc, fleet_enc=fleet_enc,
                comet_enc=comet_enc, seed=seed, learner_seat=learner_seat,
                opponent_id=args.opponent, device=device,
                max_planets=args.max_planets, max_fleets=args.max_fleets,
                sigma=args.sigma, noop_logit_bias=args.noop_logit_bias,
            )
        except Exception as e:
            print(f"[smoke] EPISODE {i} (seed={seed} seat={learner_seat}) FAILED: {e}",
                  flush=True)
            import traceback
            traceback.print_exc()
            return 1
        episodes.append(ep)
        n_steps = len(ep.steps)
        n_noop = sum(1 for s in ep.steps if s.n_selected_targets == 0)
        n_emit = sum(s.emitted_launch for s in ep.steps)
        n_inval = sum(s.invalid_launch for s in ep.steps)
        won = (ep.winner == learner_seat)
        print(f"[smoke] ep {i:02d} seed={seed} seat={learner_seat} "
              f"steps={n_steps} noop={n_noop} emit={n_emit} inval={n_inval} "
              f"winner={ep.winner} {'WIN' if won else 'lose'} ({time.time()-t_ep:.1f}s)",
              flush=True)
    print(f"[smoke] rollout total: {time.time()-t_roll:.1f}s for "
          f"{args.episodes} eps", flush=True)

    total_steps = sum(len(e.steps) for e in episodes)
    print(f"[smoke] total learner steps: {total_steps}", flush=True)

    # ---- GAE + PPO update ----
    t_gae = time.time()
    ep_objs, mbs = episodes_to_ppo(
        episodes, minibatch_size=args.minibatch_size, device=device,
    )
    print(f"[smoke] built {len(mbs)} minibatches in {time.time()-t_gae:.1f}s", flush=True)
    if not mbs:
        print("[smoke] no steps to train on; exiting.", flush=True)
        return 0

    # Wrap with L0 for training-time forward.
    train_policy = _PPOWithL0(policy, planet_enc, fleet_enc, comet_enc).to(device)

    cfg_ppo = PPOConfig(
        clip=args.clip, target_kl=args.target_kl, epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr_heads=args.lr_heads, lr_trunk=None,
        bc_coef=0.0,  # no BC anchor for smoke
    )
    t_upd = time.time()
    metrics = ppo_update_local(
        train_policy, ep_objs, mbs,
        bc_minibatch_source=lambda _n: None,
        cfg=cfg_ppo,
    )
    print(f"[smoke] PPO update done in {time.time()-t_upd:.1f}s", flush=True)

    # Pretty-print epoch metrics.
    for em in metrics["epoch_metrics"]:
        early = " EARLY-STOP" if em.get("early_stopped") else ""
        print(f"  epoch {em['epoch']}: avg_kl={em.get('avg_kl', 0):.4f} "
              f"policy_loss={em.get('policy_loss', 0):.4f} "
              f"value_loss={em.get('value_loss', 0):.4f} "
              f"entropy={em.get('entropy', 0):.3f} "
              f"clip_frac={em.get('clip_frac', 0):.3f}{early}",
              flush=True)

    print(f"[smoke] DONE total wall={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
