"""UI-runnable agent: wraps the freeze-trained action policy + physical helper.

Loads ``action_best.pt`` from a freeze-train run, runs the policy each
tick to pick a (source, target) planet pair, then delegates to the
same physics primitives ``physical_v4`` uses (lead-aim, surplus check,
ETA) to produce the ``[planet_id, angle, ships]`` action triples that
``orbit_wars`` expects.

Smoke test:

    python -m agents.transformer_v1.runner --smoke-test \\
        --ckpt data/runs/action/<run>/action_best.pt

Plug into UI / runner:

    from agents.transformer_v1 import runner   # registers `transformer_v1`
    Agent("transformer_v1")                     # via agents.registry

Note: the registry decorator runs on import. ``run.py`` already imports
the agents package, which imports this module via the registry-loading
path, so just ensure ``import agents.transformer_v1.runner`` lands in
the import graph.
"""

from __future__ import annotations

import argparse
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
from ..physics_utils import F_OWNER, P_OWNER
from ..registry import register
from .featurizer import FleetTracker, featurize_observation
from .paths import ACTION_RUNS_DIR


# ---- Default ckpt resolution ----
def _default_ckpt() -> Path:
    """Pick the newest ``action_best.pt`` (or ``action_last.pt``) under
    ``data/runs/action/``. Override with ``TRANSFORMER_V1_CKPT`` env var.
    """
    env_override = os.environ.get("TRANSFORMER_V1_CKPT")
    if env_override:
        return Path(env_override)
    if not ACTION_RUNS_DIR.exists():
        raise FileNotFoundError(
            f"no action runs dir at {ACTION_RUNS_DIR}; train an action "
            "policy first or set TRANSFORMER_V1_CKPT."
        )
    candidates: list[Path] = []
    for run_dir in sorted(ACTION_RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        for name in ("action_best.pt", "action_last.pt"):
            p = run_dir / name
            if p.exists():
                candidates.append(p)
                break
    if not candidates:
        raise FileNotFoundError(
            f"no action_*.pt under {ACTION_RUNS_DIR}/*/."
        )
    return candidates[-1]


class TransformerAgent:
    """Inference wrapper around a freeze-trained action policy."""

    def __init__(
        self,
        stack: torch.nn.Module,
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
        from .pretrain.expert_action import load_for_inference

        ckpt_path = Path(ckpt_path) if ckpt_path else _default_ckpt()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        stack = load_for_inference(ckpt_path, device=device)
        print(f"[transformer_v1] loaded {ckpt_path} on {device}")
        return cls(stack, device=device, deterministic=deterministic)

    @torch.inference_mode()
    def _predict(
        self,
        obs: dict[str, Any],
        learner_slot: int,
    ) -> tuple[int | None, int | None, dict[int, int]]:
        """Return (source_planet_id, target_planet_id, pid_to_idx)
        from the action decoder. Either id may be ``None`` if the policy
        decides no action this turn (acted gate < 0.5) or no legal
        target exists.
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

        acted_logit = float(out["expert_acted_logit"].squeeze().item())
        if acted_logit < 0.0:
            # Sigmoid < 0.5 → policy says skip this turn.
            return None, None, pid_to_idx

        src_logits = out["source_planet_logits"].squeeze(0)   # (P,)
        tgt_logits = out["target_planet_logits"].squeeze(0)   # (P,)
        # Mask own / non-existent planet slots for the source head and
        # own planets for the target head (we don't reinforce ourselves
        # in v1).
        idx_to_pid = {i: pid for pid, i in pid_to_idx.items()}
        raw_planets = obs.get("planets") if isinstance(obs, dict) else getattr(obs, "planets", None)
        owner_by_idx: dict[int, int] = {}
        if raw_planets:
            for p in raw_planets:
                pid = int(p[0])
                owner = int(p[P_OWNER])
                if pid in pid_to_idx:
                    owner_by_idx[pid_to_idx[pid]] = owner
        neg_inf = torch.finfo(src_logits.dtype).min
        for i in range(src_logits.size(0)):
            if i not in idx_to_pid:
                src_logits[i] = neg_inf
                tgt_logits[i] = neg_inf
                continue
            owner = owner_by_idx.get(i, -1)
            if owner != learner_slot:
                src_logits[i] = neg_inf  # source must be ours
            if owner == learner_slot:
                tgt_logits[i] = neg_inf  # don't pick own as target

        if torch.isinf(src_logits).all() or torch.isinf(tgt_logits).all():
            return None, None, pid_to_idx

        if self.deterministic:
            src_idx = int(src_logits.argmax().item())
            tgt_idx = int(tgt_logits.argmax().item())
        else:
            src_idx = int(torch.multinomial(torch.softmax(src_logits, -1), 1).item())
            tgt_idx = int(torch.multinomial(torch.softmax(tgt_logits, -1), 1).item())
        return idx_to_pid.get(src_idx), idx_to_pid.get(tgt_idx), pid_to_idx

    def act(self, obs: dict[str, Any]) -> list[list]:
        """Return ``[[planet_id, angle, ships], ...]`` action triples."""
        learner_slot = int(obs.get("player", 0)) if isinstance(obs, dict) else int(obs.player)
        source_pid, target_pid, _ = self._predict(obs, learner_slot)
        if source_pid is None or target_pid is None:
            return []
        return self._target_to_moves(source_pid, target_pid, obs, learner_slot)

    @staticmethod
    def _target_to_moves(
        source_pid: int,
        target_pid: int,
        obs: dict[str, Any],
        learner_slot: int,
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
    if agent.device.startswith("cuda"):
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
