"""UI-runnable agent: pair-head + 5-output stack loaded from a v2 entity-pretrain ckpt.

Loads an ``entity_encoder_best.pt`` produced by
``agents/transformer_v2/pretrain/entity_encoder.py``, instantiates the
full L1-L4 + PairHead stack, and runs inference per tick:

  1. Featurize the current obs via :func:`featurize_observation`
     (single-frame; T-history is not maintained at inference time, the
     model accepts T=1 input via L2's step-embedding tail slice).
  2. Run L0 frozen specialists on the raw features → ``planet_tok``,
     ``comet_tok``, ``fleet_tok``.
  3. ``where(is_comet, comet_tok, planet_tok)`` → ``entity_self``.
  4. Forward through :class:`EntityPretrainModel` → 5-head dict.
  5. Mask ``pair_logits`` by per-row source-launchability and per-column
     target-validity (target ≠ own); ``argmax`` the flat ``(P, P)`` grid.
  6. Use ``sigmoid(pair_frac[src, tgt])`` to size the fleet; fall back
     to ``physical_v4``'s surplus-based sizing if the frac is implausible.
  7. Validate the chosen pair through :func:`physics_utils.plan_launch`
     before emitting the move, so wrong-planet / sun / boundary launches
     are dropped instead of sent to the env.

Default ckpt resolution (override with ``TRANSFORMER_V2_CKPT`` env var
or :meth:`TransformerAgent.load`'s ``ckpt_path`` arg): newest
``entity_encoder_best.pt`` under ``data/runs/entity/<run>/``.

The L0 frozen specialists default to the d=256 ckpts under
``data/runs/{planet,fleet,comet}/specialist_*``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from ..heuristic.physical_v4.agent import (
    PHASE_TABLE,
    candidates_for_source,
    compute_surplus,
    infer_rotation_sign,
    phase_of,
)
from ..physics_utils import (
    P_ID,
    _infer_rotation_sign_raw,
    plan_launch,
)
from ..registry import register
from .featurizer import FleetTracker, featurize_observation
from .paths import ENTITY_RUNS_DIR
from .pretrain.entity_encoder import (
    EntityPretrainModel,
    _build_entity_self_tokens,
    _load_encoders,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANET_RUN_DIR = _REPO_ROOT / "data" / "runs" / "planet" / "specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep"
DEFAULT_FLEET_RUN_DIR = _REPO_ROOT / "data" / "runs" / "fleet" / "specialist_fleet_d256_40k_lr1e4_120ep"
DEFAULT_COMET_RUN_DIR = _REPO_ROOT / "data" / "runs" / "comet" / "fullpath_scalar_multitask_d256_40k_lr1e4_120ep"


def _default_ckpt() -> Path:
    """Pick the newest ``entity_encoder_best.pt`` (or ``entity_encoder_last.pt``)
    under ``data/runs/entity/<run>/``, ranked by directory mtime. Override
    with the ``TRANSFORMER_V2_CKPT`` env var.
    """
    env_override = os.environ.get("TRANSFORMER_V2_CKPT")
    if env_override:
        return Path(env_override)
    if not ENTITY_RUNS_DIR.exists():
        raise FileNotFoundError(
            f"no entity runs dir at {ENTITY_RUNS_DIR}; train an "
            "entity-pretrain model first or set TRANSFORMER_V2_CKPT "
            "to a saved ckpt path."
        )
    run_dirs = sorted(
        (d for d in ENTITY_RUNS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        for name in ("entity_encoder_best.pt", "entity_encoder_last.pt"):
            p = run_dir / name
            if p.exists():
                return p
    raise FileNotFoundError(
        f"no entity_encoder_*.pt under {ENTITY_RUNS_DIR}/*/."
    )


def _validated_learner_launch(
    *,
    source_raw,
    target_raw,
    ships: int,
    planets: list,
    fleets: list,
    initial_planets: list,
    angular_velocity: float,
    comet_ids: set[int],
    comets: list,
    player: int,
    surplus: int,
    safety_buffer: int,
    current_step: int,
) -> tuple[float, float] | None:
    """Return ``(angle, eta)`` only if the learned launch is env-valid.

    The old v2 path used the low-level ``shoot_*`` helpers directly and
    fell back to naive ``atan2`` on errors. Those helpers intentionally
    emit an angle even when the first collision is the sun, board
    boundary, or a different planet. For learned pair selection that is
    too permissive: a bad top pair becomes a real miss/out-of-map move.

    This function uses the high-level planner instead. It pins the
    learned ship count, validates first collision against moving planets
    / comets, and returns ``None`` unless the trajectory's first planet
    hit is exactly the intended target.
    """
    if source_raw is None or target_raw is None or ships <= 0:
        return None
    try:
        target_id = int(target_raw[P_ID])
        av_sign = _infer_rotation_sign_raw(planets, initial_planets)
        launch = plan_launch(
            source_raw,
            target_raw,
            planets=planets,
            fleets=fleets,
            player=player,
            angular_velocity=angular_velocity,
            av_sign=av_sign,
            comet_planet_ids=comet_ids,
            comets=comets,
            fleet_ships=int(ships),
            surplus=int(surplus),
            safety_buffer=safety_buffer,
            current_step=current_step,
        )
        if not launch.ok or launch.actual_hit_id != target_id:
            return None
        return float(launch.angle), float(launch.eta)
    except Exception:
        return None


class TransformerAgent:
    """Inference wrapper around the v2 entity-pretrain PairHead policy."""

    def __init__(
        self,
        model: EntityPretrainModel,
        fleet_enc,
        planet_enc,
        comet_enc,
        *,
        device: str = "cpu",
        deterministic: bool = True,
        max_planets: int = 64,
        max_fleets: int = 256,
        num_players: int = 4,
    ):
        self.model = model.to(device).eval()
        self.fleet_enc = fleet_enc.to(device).eval()
        self.planet_enc = planet_enc.to(device).eval()
        self.comet_enc = comet_enc.to(device).eval()
        self.device = device
        self.deterministic = deterministic
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.num_players = num_players
        # Per-episode launch-tick state. Reset on step=0 detection.
        self._tracker = FleetTracker()
        self._last_step: int | None = None

    @classmethod
    def load(
        cls,
        ckpt_path: str | Path | None = None,
        *,
        device: str | None = None,
        deterministic: bool = True,
        planet_run_dir: Path | None = None,
        fleet_run_dir: Path | None = None,
        comet_run_dir: Path | None = None,
    ) -> "TransformerAgent":
        """Reconstruct the L0 frozen specialists + the trainable
        L1-L4 + PairHead stack from a v2 entity-pretrain ckpt.

        ``ckpt_path`` defaults to the newest ``entity_encoder_best.pt``
        under ``data/runs/entity/<run>/``. The L0 ckpt dirs default to
        the d=256 specialists shipped with the repo.
        """
        ckpt_path = Path(ckpt_path) if ckpt_path is not None else _default_ckpt()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"transformer_v2 ckpt not found at {ckpt_path}. "
                "Train an entity-pretrain model first or set "
                "TRANSFORMER_V2_CKPT."
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config") or {}
        d_model = int(cfg.get("d_model", 256))
        n_steps = int(cfg.get("n_steps", 6))
        # Older ckpts (pre-d_pair widening) did not save ``d_pair`` and
        # hardcoded PairHead's projection width to 128. New runs save the
        # explicit d_pair into the config (defaulting to d_model when
        # unspecified). Read it back here so both old and new ckpts
        # deserialize correctly.
        d_pair = int(cfg["d_pair"]) if "d_pair" in cfg else 128
        # Head counts. Pre-standardization ckpts hardcoded L1 / L3 / L4 to
        # 4 heads and L2 to 8 heads. The current line saves the configured
        # head counts so both layouts deserialize correctly. State_dict
        # shapes are head-count-independent (MultiheadAttention stores
        # ``in_proj_weight (3*d_model, d_model)`` and ``out_proj`` at the
        # full width), so the keys load fine — but the FORWARD numerics
        # change if you mis-pair weights with a different head count. The
        # fallbacks here mirror the pre-standardization defaults.
        entity_n_heads = int(cfg["entity_n_heads"]) if "entity_n_heads" in cfg else 4
        cross_n_heads = int(cfg.get("cross_n_heads", 8))
        cross_n_layers = int(cfg.get("cross_n_layers", 2))
        dual_n_heads = int(cfg["dual_n_heads"]) if "dual_n_heads" in cfg else 4

        # Load L0 frozen specialists by their run dirs. _load_encoders
        # reads each ckpt's config for d_model + use_traj_branch.
        fleet_enc, planet_enc, comet_enc = _load_encoders(
            fleet_run_dir or DEFAULT_FLEET_RUN_DIR,
            planet_run_dir or DEFAULT_PLANET_RUN_DIR,
            comet_run_dir or DEFAULT_COMET_RUN_DIR,
            device=device,
        )

        model = EntityPretrainModel(
            d_model=d_model, n_steps=n_steps, d_pair=d_pair,
            entity_n_heads=entity_n_heads,
            cross_n_heads=cross_n_heads,
            cross_n_layers=cross_n_layers,
            dual_n_heads=dual_n_heads,
        )
        model.load_state_dict(ckpt["model"])

        return cls(
            model,
            fleet_enc=fleet_enc,
            planet_enc=planet_enc,
            comet_enc=comet_enc,
            device=device,
            deterministic=deterministic,
            max_planets=cfg.get("max_planets", 64),
            max_fleets=cfg.get("max_fleets", 1024),
        )

    @torch.no_grad()
    def _predict(
        self,
        obs: dict[str, Any],
        learner_slot: int,
        *,
        logit_threshold: float = 2.0,
    ) -> list[tuple[int, int, float]]:
        """Return ``(source_pid, target_pid, frac)`` triples — potentially
        MULTIPLE launches per source per turn.

        Decision rule: for every launchable source ``s`` and every real
        off-diagonal target ``t``, emit a launch whenever
        ``pair_logits[s, t] > logit_threshold`` (default 0.0, equivalent
        to ``sigmoid > 0.5``). Each emitted launch is sized by
        ``sigmoid(pair_frac[s, t])`` of the source's ships.

        Self-cells (``s == t``) and padded slots are dropped because the
        loss never supervised them — their logits are noise. The runner
        treats this as "no diagonal action", so a source with no target
        crossing the threshold simply produces zero launches that turn.

        Multi-target rows (coalition launches) are supported: when more
        than one target's logit clears the threshold for the same source,
        emit a launch per target. ``_target_to_moves`` then sizes each
        independently against the source's surplus.
        """
        # Reset launch-tick state at episode start.
        step = int(obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0))
        if self._last_step is None or step < self._last_step:
            self._tracker = FleetTracker()
        self._last_step = step

        # Single-frame featurization. The model was trained with T=6
        # history but accepts T=1 via L2's step_embed[-T:] tail slice;
        # the loss in fidelity is bounded by the L2 attention's
        # step-position generalization.
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=learner_slot,
            tracker=self._tracker,
            num_players=self.num_players,
            max_planets=self.max_planets,
            max_fleets=self.max_fleets,
            device=self.device,
        )

        # Need comet_features too — the inference featurizer doesn't
        # emit it (training-only key on the cached snapshots). Stub it
        # with zeros for now; the comet_tok pathway will just produce
        # zero-encoded tokens for comet planets, which `where` routes
        # away from for non-comet planets anyway. is_comet is taken
        # from f000 of planet_features per the comet specialist's input
        # convention.
        B, P, _ = batch["planet_features"].shape
        comet_input_dim = self.comet_enc.input_dim
        comet_features = torch.zeros(
            (B, P, comet_input_dim), device=self.device,
            dtype=batch["planet_features"].dtype,
        )
        # The first 18 dims of the planet feature vector are the same
        # scalar block the comet specialist saw (f000 = is_comet flag,
        # f001..f017 = sun-relative geometry + owner one-hot + ships).
        comet_features[..., :18] = batch["planet_features"][..., :18]

        # is_comet flag: f000 of the planet feature vector is the
        # is_comet flag in the v2 featurizer.
        is_comet = batch["planet_features"][..., 0] > 0.5

        # L0 frozen forward.
        planet_tok = self.planet_enc(batch["planet_features"])     # (B, P, d)
        comet_tok = self.comet_enc(comet_features)                  # (B, P, d)
        fleet_tok = self.fleet_enc(batch["fleet_features"])         # (B, F, d)
        entity_self = _build_entity_self_tokens(planet_tok, comet_tok, is_comet)

        routing = {
            "fleet_target_idx": batch["fleet_target_idx"],
            "fleet_source_idx": batch["fleet_source_idx"],
            "fleet_owner_slot": batch["fleet_owner_slot"],
            "fleet_ships_log": batch["fleet_ships_log"],
            "fleet_eta_norm": batch["fleet_eta_norm"],
            "fleet_mask": batch["fleet_mask"],
        }
        preds = self.model(entity_self, fleet_tok, routing, batch["planet_mask"])

        pair_logits = preds["pair_logits"].squeeze(0)               # (P, P)
        pair_frac_raw = preds["pair_frac"].squeeze(0)               # (P, P) raw logit
        idx_to_pid = {i: pid for pid, i in pid_to_idx.items()}

        # ---- Build launchable / ownership masks from obs ----
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        _neutral_bonus, defense_buffer, min_launch, _safety, _frontier_w, _eta_tol = (
            PHASE_TABLE[phase_of(step)]
        )
        owner_by_idx: dict[int, int] = {}
        launchable_by_idx: dict[int, bool] = {}
        if raw_planets:
            from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
            planets = [Planet(*p) for p in raw_planets]
            fleets = [Fleet(*f) for f in raw_fleets]
            enemy_fleets = [f for f in fleets if f.owner != learner_slot and f.owner >= 0]
            for planet in planets:
                pid = int(planet.id)
                if pid in pid_to_idx:
                    idx = pid_to_idx[pid]
                    owner = int(planet.owner)
                    owner_by_idx[idx] = owner
                    surplus = compute_surplus(planet, enemy_fleets, defense_buffer)
                    launchable_by_idx[idx] = (
                        owner == learner_slot and surplus >= min_launch
                    )

        P = pair_logits.size(0)
        # ``real_idx[i]`` = there's a real planet (not padding) at slot i.
        # Padded slots were masked out of ``pair_valid`` during training,
        # so their logits are uncalibrated noise — drop them.
        real_idx = torch.zeros(P, dtype=torch.bool, device=pair_logits.device)
        src_legal = torch.zeros(P, dtype=torch.bool, device=pair_logits.device)
        for i in range(P):
            if i not in idx_to_pid:
                continue
            real_idx[i] = True
            if launchable_by_idx.get(i, False):
                src_legal[i] = True

        if not src_legal.any():
            return []

        # Per-cell decision: a cell (s, t) fires when its logit clears the
        # threshold AND it's legal. Legality: source launchable, target
        # real, t != s (diagonal was unsupervised — exclude).
        device = pair_logits.device
        eye = torch.eye(P, dtype=torch.bool, device=device)
        cell_legal = (
            src_legal.unsqueeze(1)               # (P, 1) — source must be launchable
            & real_idx.unsqueeze(0)              # (1, P) — target must be real
            & ~eye                                # exclude self-cells (off-diagonal only)
        )
        firing = (pair_logits > logit_threshold) & cell_legal       # (P, P) bool
        if not firing.any():
            return []

        # Convert to a list of (src, tgt, frac). For each firing cell, frac
        # is sigmoid(pair_frac[s, t]). Multi-target rows produce multiple
        # entries; _target_to_moves sizes each against the source's surplus
        # independently. To keep the per-source launches well-ordered when
        # surplus is tight, emit cells in pair_logits-descending order.
        src_indices, tgt_indices = firing.nonzero(as_tuple=True)
        cell_logits = pair_logits[src_indices, tgt_indices]
        order = torch.argsort(cell_logits, descending=True)
        actions: list[tuple[int, int, float]] = []
        for k in order.tolist():
            src_idx = int(src_indices[k].item())
            tgt_idx = int(tgt_indices[k].item())
            source_pid = idx_to_pid.get(src_idx)
            target_pid = idx_to_pid.get(tgt_idx)
            if source_pid is None or target_pid is None:
                continue
            frac = float(torch.sigmoid(pair_frac_raw[src_idx, tgt_idx]).item())
            actions.append((source_pid, target_pid, frac))
        return actions

    def act(self, obs: dict[str, Any]) -> list[list]:
        """Return ``[[planet_id, angle, ships], ...]`` action triples.

        Sources can emit multiple launches per turn (one per target whose
        ``pair_logits[s, t]`` cleared the threshold). To prevent the same
        source's ship count from being double-spent across its targets,
        we budget the source's ``compute_surplus`` once and consume from
        a per-source running remainder as we walk the actions in
        logit-descending order.
        """
        learner_slot = int(obs.get("player", 0)) if isinstance(obs, dict) else int(obs.player)
        actions = self._predict(obs, learner_slot)
        if not actions:
            return []

        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        step = int(get("step") or 0)
        _neutral_bonus, defense_buffer, min_launch, _safety, _frontier_w, _eta_tol = (
            PHASE_TABLE[phase_of(step)]
        )
        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]
        enemy_fleets = [f for f in fleets if f.owner != learner_slot and f.owner >= 0]
        by_id = {p.id: p for p in planets}

        # Per-source budget: starts at compute_surplus(source, ...), drains
        # as launches commit. We also cache source.ships (used to scale
        # frac into a ship count) so multi-target rows share the same
        # baseline rather than re-reading mutated obs.
        per_source_budget: dict[int, int] = {}
        per_source_total_ships: dict[int, int] = {}
        for source_pid, _t, _f in actions:
            if source_pid in per_source_budget:
                continue
            src = by_id.get(int(source_pid))
            if src is None:
                continue
            per_source_budget[source_pid] = int(compute_surplus(
                src, enemy_fleets, defense_buffer,
            ))
            per_source_total_ships[source_pid] = int(src.ships)

        moves: list[list] = []
        for source_pid, target_pid, frac in actions:
            budget = per_source_budget.get(source_pid, 0)
            if budget < min_launch:
                continue
            base_ships = per_source_total_ships.get(source_pid, 0)
            desired = max(min_launch, int(round(frac * base_ships)))
            ships = min(budget, desired)
            if ships < min_launch:
                continue
            emitted = self._target_to_moves(
                source_pid, target_pid, obs, learner_slot,
                ships_override=ships,
            )
            if emitted:
                # _target_to_moves may have clamped via its fallback path;
                # subtract the actually-emitted ship count from the budget.
                actually_spent = int(emitted[0][2])
                per_source_budget[source_pid] = max(0, budget - actually_spent)
                moves.extend(emitted)
        return moves

    @staticmethod
    def _target_to_moves(
        source_pid: int,
        target_pid: int,
        obs: dict[str, Any],
        learner_slot: int,
        frac_override: float | None = None,
        ships_override: int | None = None,
    ) -> list[list]:
        """Translate (source, target) into a single ``[planet_id, angle,
        ships]`` triple.

        ``ships_override`` (preferred when set by the caller) pins the
        exact ship count — callers managing a per-source surplus budget
        across multiple targets use this to avoid double-spending the
        same source's ships. ``frac_override`` is the legacy single-launch
        path: caller passes a fraction and the function multiplies it by
        ``source.ships`` before clamping to surplus.

        Learned fractions are kept only when ``plan_launch`` validates
        the exact source→target trajectory. If that fixed-size launch is
        invalid, fall back to physical_v4's validated candidate sizing
        for the same target.
        """
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        initial_planets = get("initial_planets") or []
        angular_velocity = abs(float(get("angular_velocity") or 0.0))
        step = int(get("step") or 0)
        comet_planet_ids = set(get("comet_planet_ids") or [])
        comets = get("comets") or []

        phase = phase_of(step)
        neutral_bonus, defense_buffer, min_launch, safety, frontier_w, _eta_tol = (
            PHASE_TABLE[phase]
        )
        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]
        av_signed = (
            infer_rotation_sign(planets, initial_planets) * angular_velocity
        )
        raw_by_id = {row[0]: row for row in raw_planets}

        source = next((p for p in planets if p.id == source_pid), None)
        target = next((p for p in planets if p.id == target_pid), None)
        if source is None or target is None or source.owner != learner_slot:
            return []
        my_planets = [p for p in planets if p.owner == learner_slot]
        enemy_fleets = [
            f for f in fleets if f.owner != learner_slot and f.owner >= 0
        ]
        surplus = compute_surplus(source, enemy_fleets, defense_buffer)

        # Path 1: caller pinned the ship count (multi-target ``act()`` budgeter).
        # No fallback — if the trajectory is invalid at this size, drop the
        # launch rather than reshape it.
        if ships_override is not None:
            ships = min(int(ships_override), int(surplus))
            if ships < min_launch:
                return []
            planned = _validated_learner_launch(
                source_raw=raw_by_id.get(source.id),
                target_raw=raw_by_id.get(target.id),
                ships=ships,
                planets=raw_planets,
                fleets=raw_fleets,
                initial_planets=initial_planets,
                angular_velocity=angular_velocity,
                comet_ids=comet_planet_ids,
                comets=comets,
                player=learner_slot,
                surplus=surplus,
                safety_buffer=safety,
                current_step=step,
            )
            if planned is None:
                return []
            angle, _eta = planned
            return [[source.id, angle, ships]]

        # Path 2: legacy frac-only path. Validate learned-size first; fall
        # back to physical_v4 candidate sizing only when validation fails.
        if frac_override is not None and surplus >= min_launch:
            ships = min(
                int(surplus),
                max(min_launch, int(round(frac_override * source.ships))),
            )
            if ships < min_launch:
                return []
            planned = _validated_learner_launch(
                source_raw=raw_by_id.get(source.id),
                target_raw=raw_by_id.get(target.id),
                ships=ships,
                planets=raw_planets,
                fleets=raw_fleets,
                initial_planets=initial_planets,
                angular_velocity=angular_velocity,
                comet_ids=comet_planet_ids,
                comets=comets,
                player=learner_slot,
                surplus=surplus,
                safety_buffer=safety,
                current_step=step,
            )
            if planned is not None:
                angle, _eta = planned
                return [[source.id, angle, ships]]
        if surplus < min_launch:
            return []

        # Fallback: physical_v4 surplus-based sizing when frac override
        # is unavailable or yields too few ships.
        cands = candidates_for_source(
            source,
            [target],
            my_planets,
            surplus,
            av_signed,
            angular_velocity,
            neutral_bonus,
            frontier_w,
            safety,
            raw_planets,
            raw_fleets,
            raw_by_id,
            learner_slot,
            step,
            comet_planet_ids=comet_planet_ids,
            comets=comets,
        )
        if not cands:
            return []
        _t, _eta, ships, angle, _score = cands[0]
        return [[source.id, angle, ships]]


# ---- Registry hook ----
_AGENT_SINGLETON: TransformerAgent | None = None


def transformer_v2_agent(obs):
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = TransformerAgent.load()
    return _AGENT_SINGLETON.act(obs)


# Idempotent registration: ``python -m agents.transformer_v2.runner``
# imports this module twice (once via the package import chain, once
# as ``__main__``). A bare ``@register`` would raise on the second pass.
from .. import registry as _registry  # noqa: E402

if "transformer_v2" not in _registry._REGISTRY:
    _registry.register(
        "transformer_v2",
        "v2 entity-pretrain PairHead policy: L0 specialists + L1-L4 + 5-head "
        "PairHead, pair-score argmax + pair_frac sizing, fallback to physical_v4.",
    )(transformer_v2_agent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Load the agent and verify a single act() call.")
    args = parser.parse_args()
    if args.smoke_test:
        agent = TransformerAgent.load(args.ckpt, device=args.device)
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": 1729})
        obs = env.steps[0][0].observation
        moves = agent.act(obs)
        print(f"smoke-test moves: {moves}")


if __name__ == "__main__":
    main()
