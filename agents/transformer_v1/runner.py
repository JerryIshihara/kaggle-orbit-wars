"""UI-runnable agent: pair-score + frac stack loaded from a combined ckpt.

Loads a ``pair_score_best.pt`` (FleetEncoder + PlanetEncoder +
PlanetEntityEncoder + CrossEntityAttention + PairScoreHead + optional
FracHead, all in one file) and runs inference each tick:

  1. Featurize the obs via :func:`featurize_observation`.
  2. Forward through :class:`PairScoreStack` to get per-pair logits and
     (when the FracHead is present) per-pair launch-fraction loc.
  3. Mask invalid pairs — source must be ours + launchable, target
     must not be ours — then ``argmax`` the masked ``pair_logits``.
  4. Look up ``frac_loc[src, tgt]`` (or fall back to the heuristic ship
     sizing when the ckpt has no frac head), apply the deterministic
     ``sigmoid(clamp(z, FRAC_Z_MIN))`` recipe, and convert to a ship
     count.
  5. Hand off to :func:`shoot_static` / :func:`shoot_orbit` /
     :func:`shoot_comet` for the actual launch angle.

Smoke test:

    python -m agents.transformer_v1.runner --smoke-test \\
        --ckpt data/runs/pair_score/<run>/pair_score_best.pt

Plug into UI / runner:

    from agents.transformer_v1 import runner   # registers `transformer_v1`
    Agent("transformer_v1")                     # via agents.registry

The registry decorator runs on import — ``run.py`` already imports the
agents package, which pulls this module in via the registry-loading
path, so the agent is available wherever the registry is.

Default ckpt resolution (override with the ``TRANSFORMER_V1_CKPT`` env
var or :meth:`TransformerAgent.load`'s ``ckpt_path`` arg): newest
``pair_score_best.pt`` under ``data/runs/pair_score/<run>/``.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from ..physical_v4.agent import (
    PHASE_TABLE,
    candidates_for_source,
    compute_surplus,
    infer_rotation_sign,
    phase_of,
)
from ..physics_utils import (
    P_ID, P_OWNER, P_X, P_Y, P_RADIUS,
    shoot_static, shoot_orbit, shoot_comet,
    _is_orbiting_xy, _infer_rotation_sign_raw,
)
from ..registry import register
from .aggregator import CrossEntityAttention
from .encoder import (
    FleetEncoder, PlanetEncoder, PlanetEntityEncoder,
)
from .featurizer import FleetTracker, featurize_observation
from .paths import ACTION_RUNS_DIR
from .pretrain.expert_action import FRAC_Z_MIN
from .pretrain.pair_score import (
    FRAC_LOG_STD_MAX,
    FRAC_LOG_STD_MIN,
    FracHead,
    PairScoreHead,
    PairScoreStack,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
PAIR_SCORE_RUNS_DIR: Path = _REPO_ROOT / "data" / "runs" / "pair_score"


# ---- Default ckpt resolution ----
def _default_ckpt() -> Path:
    """Pick the newest ``pair_score_best.pt`` (or ``pair_score_last.pt``)
    under ``data/runs/pair_score/<run>/``, ranked by **directory mtime**
    so a freshly-pushed stage-2 run beats older runs whose lexicographic
    sort happens to put them later (e.g. ``unfreeze_cross_*`` would
    otherwise win over ``exp3_frac_only_*``). Empty run dirs without a
    valid ckpt file are skipped. Override with the ``TRANSFORMER_V1_CKPT``
    env var or by passing ``ckpt_path`` to :meth:`TransformerAgent.load`.
    """
    env_override = os.environ.get("TRANSFORMER_V1_CKPT")
    if env_override:
        return Path(env_override)
    if not PAIR_SCORE_RUNS_DIR.exists():
        raise FileNotFoundError(
            f"no pair_score runs dir at {PAIR_SCORE_RUNS_DIR}; train a "
            "pair-score (+ optional frac) policy first or set "
            "TRANSFORMER_V1_CKPT to a saved ckpt path."
        )
    # mtime-sorted, newest first.
    run_dirs = sorted(
        (d for d in PAIR_SCORE_RUNS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        for name in ("pair_score_best.pt", "pair_score_last.pt"):
            p = run_dir / name
            if p.exists():
                return p
    raise FileNotFoundError(
        f"no pair_score_*.pt under {PAIR_SCORE_RUNS_DIR}/*/."
    )


def _learner_aim(
    *,
    source_raw,
    target_raw,
    ships: int,
    obs,
    planets: list,
    initial_planets: list,
    angular_velocity: float,
    comet_ids: set[int],
    fallback_angle: float,
) -> float:
    """Pick a launch angle for the learner's (source, target, ships) using
    the motion-class-aware shoot helpers from ``physics_utils``.

    Returns ``fallback_angle`` (naive ``atan2`` of straight-line aim) on
    any error — guarantees PPO rollouts never crash on a bad shoot call,
    just like before this integration.
    """
    if source_raw is None or target_raw is None:
        return fallback_angle
    try:
        tid = int(target_raw[P_ID])
        if tid in (comet_ids or set()):
            angle, _eta, _n = shoot_comet(source_raw, target_raw, int(ships), obs)
        elif angular_velocity > 0.0 and _is_orbiting_xy(
            float(target_raw[P_X]),
            float(target_raw[P_Y]),
            float(target_raw[P_RADIUS]),
            float(angular_velocity),
        ):
            av_sign = _infer_rotation_sign_raw(planets, initial_planets)
            angle, _eta, _n = shoot_orbit(
                source_raw, target_raw, int(ships), obs, av_sign=av_sign,
            )
        else:
            angle, _eta, _n = shoot_static(source_raw, target_raw, int(ships))
        return float(angle)
    except Exception:
        # Any failure (numerical, missing field, etc.) — fall back so
        # rollout integrity is preserved.
        return fallback_angle


class TransformerAgent:
    """Inference wrapper around a pair-score (+ optional frac) policy."""

    def __init__(
        self,
        stack: PairScoreStack,
        *,
        device: str = "cpu",
        deterministic: bool = True,
        max_planets: int = 64,
        max_fleets: int = 256,
        num_players: int = 4,
    ):
        self.stack = stack.to(device).eval()
        self.device = device
        self.deterministic = deterministic
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.num_players = num_players
        # Each agent instance owns its own fleet tracker so launch ticks
        # are inferred consistently across this episode's calls.
        self._tracker = FleetTracker()

    @classmethod
    def load(
        cls,
        ckpt_path: str | Path | None = None,
        *,
        device: str | None = None,
        deterministic: bool = True,
    ) -> "TransformerAgent":
        """Reconstruct the stack from a combined ``pair_score_best.pt``.

        The ckpt is expected to carry every encoder's state_dict + the
        pair-score head state, with the FracHead state optional. Configs
        for module dims come from the ckpt's ``config`` block; max_planets
        / max_fleets / n_history default to the saved values, falling back
        to the production defaults if missing.
        """
        ckpt_path = Path(ckpt_path) if ckpt_path is not None else _default_ckpt()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"transformer_v1 ckpt not found at {ckpt_path}. "
                "Train a pair_score model first or set TRANSFORMER_V1_CKPT."
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "pair_score_head" not in ckpt:
            raise ValueError(
                f"{ckpt_path} has no 'pair_score_head' key — expected the "
                "output of agents/transformer_v1/pretrain/pair_score.py."
            )
        for k in ("fleet_encoder", "planet_encoder", "entity_encoder", "cross"):
            if k not in ckpt:
                raise ValueError(
                    f"{ckpt_path} is missing '{k}' state — pair_score ckpts "
                    "since the self-contained refactor should carry every "
                    "encoder. Re-train with current code, or pass the older "
                    "action_best.pt path instead."
                )

        cfg = ckpt.get("config") or {}
        d_model = int(cfg.get("d_model", 64))
        hidden = int(cfg.get("hidden", 128))
        frac_hidden = int(cfg.get("frac_hidden", hidden))
        max_planets = int(cfg.get("max_planets", 64))
        max_fleets = int(cfg.get("max_fleets", 256))

        fenc = FleetEncoder(d_model=d_model)
        fenc.load_state_dict(ckpt["fleet_encoder"])
        penc = PlanetEncoder(d_model=d_model)
        penc.load_state_dict(ckpt["planet_encoder"])
        eenc = PlanetEntityEncoder(d_model=d_model)
        eenc.load_state_dict(ckpt["entity_encoder"])
        cross = CrossEntityAttention(d_model=d_model)
        cross.load_state_dict(ckpt["cross"])

        pair_head = PairScoreHead(d_model=d_model, hidden=hidden)
        pair_head.load_state_dict(ckpt["pair_score_head"])

        # FracHead is optional in the ckpt — older pair-only runs didn't
        # write one. When absent, the agent falls back to physical_v4's
        # surplus-based ship sizing in :meth:`_target_to_moves`.
        frac_head: FracHead | None = None
        if "frac_head" in ckpt:
            frac_head = FracHead(d_model=d_model, hidden=frac_hidden)
            frac_head.load_state_dict(ckpt["frac_head"])

        stack = PairScoreStack(
            fleet_encoder=fenc,
            planet_encoder=penc,
            entity_encoder=eenc,
            cross=cross,
            pair_score_head=pair_head,
            frac_head=frac_head,
        )
        return cls(
            stack,
            device=device,
            deterministic=deterministic,
            max_planets=max_planets,
            max_fleets=max_fleets,
        )

    @torch.no_grad()
    def _predict(
        self,
        obs: dict[str, Any],
        learner_slot: int,
    ) -> tuple[int | None, int | None, float | None]:
        """Return ``(source_pid, target_pid, frac_or_None)`` for the
        current turn.

        ``frac`` is the deterministic ``sigmoid(clamp(z, FRAC_Z_MIN))``
        sampled at ``frac_loc[src, tgt]`` when the loaded stack has a
        FracHead; ``None`` when it doesn't, in which case the caller
        falls back to ``physical_v4``'s surplus-based ship sizing.

        Returns ``(None, None, None)`` if the agent has no legal
        ``(source, target)`` pair this turn — empty owned planets,
        no surplus, or every candidate masked out by the
        ``planet/launchable`` / ``not-self`` filters.
        """
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=learner_slot,
            tracker=self._tracker,
            num_players=self.num_players,
            max_planets=self.max_planets,
            max_fleets=self.max_fleets,
            device=self.device,
        )
        out = self.stack(batch)
        pair_logits = out["pair_logits"].squeeze(0)               # (P, P)
        frac_loc = out.get("frac_loc")
        if frac_loc is not None:
            frac_loc = frac_loc.squeeze(0)                         # (P, P)

        P = pair_logits.size(0)
        idx_to_pid = {i: pid for pid, i in pid_to_idx.items()}

        # ---- Build launchable / ownership masks from obs ----
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        step = int(get("step") or 0)
        _neutral_bonus, defense_buffer, min_launch, _safety, _frontier_w, _eta_tol = (
            PHASE_TABLE[phase_of(step)]
        )
        owner_by_idx: dict[int, int] = {}
        launchable_by_idx: dict[int, bool] = {}
        if raw_planets:
            from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

            planets = [Planet(*p) for p in raw_planets]
            fleets = [Fleet(*f) for f in raw_fleets]
            enemy_fleets = [
                f for f in fleets if f.owner != learner_slot and f.owner >= 0
            ]
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

        neg_inf = torch.finfo(pair_logits.dtype).min
        # Build per-row / per-col legal masks. A source row is legal iff
        # the corresponding planet exists, is ours, and has spend room;
        # a target column is legal iff the planet exists and is NOT ours
        # (we don't reinforce ourselves through the pair head — that's
        # what frac=0 would be for, which we don't predict).
        src_legal = torch.zeros(P, dtype=torch.bool, device=pair_logits.device)
        tgt_legal = torch.zeros(P, dtype=torch.bool, device=pair_logits.device)
        for i in range(P):
            if i not in idx_to_pid:
                continue
            if launchable_by_idx.get(i, False):
                src_legal[i] = True
            if owner_by_idx.get(i, -1) != learner_slot:
                tgt_legal[i] = True

        pair_mask = src_legal.unsqueeze(1) & tgt_legal.unsqueeze(0)   # (P, P)
        if not pair_mask.any():
            return None, None, None
        masked = pair_logits.masked_fill(~pair_mask, neg_inf)

        if self.deterministic:
            flat_idx = int(masked.reshape(-1).argmax().item())
        else:
            probs = torch.softmax(masked.reshape(-1), dim=-1)
            flat_idx = int(torch.multinomial(probs, 1).item())
        src_idx = flat_idx // P
        tgt_idx = flat_idx % P

        # Frac extraction: ``sigmoid(clamp(z, FRAC_Z_MIN))`` per the
        # inference contract that lines up with the deleted PPO
        # decoder's truncated-Normal deterministic branch. When the
        # ckpt has no FracHead, return None so ``_target_to_moves``
        # falls through to the physical_v4 sizing heuristic.
        frac: float | None = None
        if frac_loc is not None:
            z = float(frac_loc[src_idx, tgt_idx].item())
            z_clamped = max(z, FRAC_Z_MIN)
            frac = 1.0 / (1.0 + math.exp(-z_clamped))

        return idx_to_pid.get(src_idx), idx_to_pid.get(tgt_idx), frac

    def act(self, obs: dict[str, Any]) -> list[list]:
        """Return ``[[planet_id, angle, ships], ...]`` action triples."""
        learner_slot = int(obs.get("player", 0)) if isinstance(obs, dict) else int(obs.player)
        source_pid, target_pid, frac = self._predict(obs, learner_slot)
        if source_pid is None or target_pid is None:
            return []
        return self._target_to_moves(
            source_pid, target_pid, obs, learner_slot, frac_override=frac,
        )

    @staticmethod
    def _target_to_moves(
        source_pid: int,
        target_pid: int,
        obs: dict[str, Any],
        learner_slot: int,
        frac_override: float | None = None,
    ) -> list[list]:
        """Translate (source, target) planet IDs into a single
        ``[planet_id, angle, ships]`` triple via ``physical_v4``'s
        lead-aim + surplus + frontier-distance scorer.
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
        if frac_override is not None:
            # Use learned fraction instead of candidate search.
            # Guard: don't force-launch when there's no defensive surplus.
            if surplus < min_launch:
                return []
            ships = min(
                int(surplus),
                max(min_launch, int(round(frac_override * source.ships))),
            )
            if ships < min_launch:
                return []
            # Motion-class-aware aim. Falls back to the prior naive aim
            # if the shoot_* dispatch raises — keeping rollout stable
            # under any unexpected env state.
            angle = _learner_aim(
                source_raw=raw_by_id.get(source.id),
                target_raw=raw_by_id.get(target.id),
                ships=ships,
                obs=obs,
                planets=raw_planets,
                initial_planets=initial_planets,
                angular_velocity=angular_velocity,
                comet_ids=comet_planet_ids,
                fallback_angle=math.atan2(target.y - source.y, target.x - source.x),
            )
            return [[source.id, angle, ships]]
        if surplus < min_launch:
            return []

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
            comet_planet_ids=comet_planet_ids,
            comets=comets,
        )
        if not cands:
            return []
        # candidates_for_source returns tuples whose layout is documented
        # in physical_v4.agent — index 2 is ships, 3 is angle.
        _t, _eta, ships, angle, _score = cands[0]
        return [[source.id, angle, ships]]


# ---- Registry hook ----
_AGENT_SINGLETON: TransformerAgent | None = None


def transformer_v1_agent(obs):
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = TransformerAgent.load()
    return _AGENT_SINGLETON.act(obs)


# Idempotent registration: ``python -m agents.transformer_v1.runner``
# imports this module twice (once as ``agents.transformer_v1.runner``
# via the package import chain, once as ``__main__``). A bare
# ``@register`` decorator would raise on the second pass.
from .. import registry as _registry  # noqa: E402

if "transformer_v1" not in _registry._REGISTRY:
    _registry.register(
        "transformer_v1",
        "Transformer policy (action-decoder) + physical_v4 helper.",
    )(transformer_v1_agent)


# ---- CLI / smoke test ----
def _smoke_test(ckpt: str | None, device: str | None) -> None:
    agent = TransformerAgent.load(ckpt, device=device)
    fake_obs = {
        "player": 0,
        "step": 50,
        "angular_velocity": 0.05,
        "planets": [],
        "fleets": [],
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
    }
    # Pure forward pass, no physical_v4 hand-off (which needs real planets).
    t0 = time.time()
    for _ in range(20):
        agent._predict(fake_obs, learner_slot=0)
    if str(agent.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[smoke] 20 ticks in {elapsed:.2f}s ({20 / elapsed:.1f} Hz)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="Path to action ckpt. Default: latest under data/runs/action/.")
    p.add_argument("--device", default=None,
                   help="Default: cuda if available, else cpu.")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    if args.smoke_test:
        _smoke_test(args.ckpt, args.device)
        return
    print("transformer_v1 registered. Use:")
    print("  from agents.registry import Agent")
    print("  Agent('transformer_v1')(obs)")


if __name__ == "__main__":
    main()
