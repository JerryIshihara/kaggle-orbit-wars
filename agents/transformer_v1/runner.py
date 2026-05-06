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
from ..registry import register
from .featurizer import FleetTracker, featurize_observation
from .paths import ACTION_RUNS_DIR
from .pretrain.expert_action import compute_action_log_prob, FRAC_Z_MIN


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

    @torch.no_grad()
    def _predict(
        self,
        obs: dict[str, Any],
        learner_slot: int,
        return_rollout: bool = False,
    ) -> tuple[int | None, int | None, dict[int, int]]:
        """Return (source_planet_id, target_planet_id, pid_to_idx)
        from the action decoder. Either id may be ``None`` if the policy
        decides no action this turn (acted gate < 0.5) or no legal
        target exists.

        When ``return_rollout=True``, returns a rollout dict instead
        (for PPO trajectory collection).
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

        acted_logit = out["expert_acted_logit"].squeeze(0)
        acted_logit_val = float(acted_logit.squeeze().item())
        if not return_rollout:
            if self.deterministic:
                if acted_logit_val < 0.0:
                    # Sigmoid < 0.5 → policy says skip this turn.
                    return None, None, pid_to_idx
            else:
                acted_dist = torch.distributions.Bernoulli(logits=acted_logit)
                if int(acted_dist.sample().item()) == 0:
                    return None, None, pid_to_idx

        src_logits = out["source_planet_logits"].squeeze(0)   # (P,)
        tgt_logits = out["target_planet_logits"].squeeze(0)   # (P,)
        # Mask own / non-existent planet slots for the source head and
        # own planets for the target head (we don't reinforce ourselves
        # in v1).
        idx_to_pid = {i: pid for pid, i in pid_to_idx.items()}
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
        neg_inf = torch.finfo(src_logits.dtype).min
        for i in range(src_logits.size(0)):
            if i not in idx_to_pid:
                src_logits[i] = neg_inf
                tgt_logits[i] = neg_inf
                continue
            owner = owner_by_idx.get(i, -1)
            if not launchable_by_idx.get(i, False):
                src_logits[i] = neg_inf  # source must be ours and launchable
            if owner == learner_slot:
                tgt_logits[i] = neg_inf  # don't pick own as target

        neg_inf_val = torch.finfo(src_logits.dtype).min
        all_src_inf = bool((src_logits <= neg_inf_val / 2).all())
        all_tgt_inf = bool((tgt_logits <= neg_inf_val / 2).all())
        if (all_src_inf or all_tgt_inf) and not return_rollout:
            return None, None, pid_to_idx

        if all_src_inf or all_tgt_inf:
            src_idx = -1
            tgt_idx = -1
        elif self.deterministic:
            src_idx = int(src_logits.argmax().item())
            tgt_idx = int(tgt_logits.argmax().item())
        else:
            src_idx = int(torch.multinomial(torch.softmax(src_logits, -1), 1).item())
            tgt_idx = int(torch.multinomial(torch.softmax(tgt_logits, -1), 1).item())

        if return_rollout:
            if all_src_inf or all_tgt_inf:
                acted_int = 0
            elif self.deterministic:
                acted_int = 1 if acted_logit_val >= 0.0 else 0
            else:
                acted_dist = torch.distributions.Bernoulli(logits=acted_logit)
                acted_int = int(acted_dist.sample().item())
            value = float(out.get("value", torch.zeros(1, device=self.device)).squeeze().item()) \
                if "value" in out else 0.0
            if "frac_logits" in out:
                frac_logits = out["frac_logits"].squeeze(0)
            else:
                frac_logits = torch.zeros(self.max_planets, device=self.device)

            src_legal = src_logits > neg_inf_val / 2
            tgt_legal = tgt_logits > neg_inf_val / 2

            frac_z = 0.0
            frac_val = 0.0
            if acted_int and src_idx >= 0:
                frac_loc = frac_logits[src_idx]
                if self.deterministic:
                    # Deterministic: use loc, but enforce z_min floor so the
                    # executed action never falls below the truncation point.
                    frac_z_t = torch.maximum(
                        frac_loc,
                        torch.tensor(FRAC_Z_MIN, dtype=frac_loc.dtype, device=frac_loc.device),
                    )
                else:
                    # σ floor 0.30 (was 0.05). Tight σ collapsed frac
                    # samples to the loc; keeping a non-trivial floor
                    # preserves frac exploration during rollouts.
                    frac_std = torch.clamp(
                        self.stack.action_decoder.frac_log_std.exp(),
                        min=0.30,
                        max=1.0,
                    ).to(device=frac_logits.device, dtype=frac_logits.dtype)
                    frac_dist = torch.distributions.Normal(
                        loc=frac_loc,
                        scale=frac_std,
                    )
                    # Inverse-CDF sampling for lower-truncated Normal at FRAC_Z_MIN.
                    # Single call, no rejection loop.
                    # If loc << FRAC_Z_MIN (p_lo -> 1), clamp prevents NaN.
                    z_min_t = torch.tensor(
                        FRAC_Z_MIN, dtype=frac_loc.dtype, device=frac_loc.device,
                    )
                    p_lo = frac_dist.cdf(z_min_t)
                    p_lo = torch.clamp(p_lo, max=1.0 - 1e-6)
                    u = torch.rand(1, dtype=frac_loc.dtype, device=frac_loc.device).squeeze()
                    p = p_lo + u * (1.0 - p_lo)
                    frac_z_t = frac_dist.icdf(p)
                    # Numerical safeguard in case icdf returns slightly below z_min.
                    frac_z_t = torch.maximum(frac_z_t, z_min_t)
                frac_z = float(frac_z_t.item())
                frac_val = float(torch.sigmoid(frac_z_t).item())

            log_prob = 0.0
            if hasattr(self.stack.action_decoder, "frac_log_std"):
                log_prob = compute_action_log_prob(
                    out["expert_acted_logit"].squeeze(0),
                    src_logits, tgt_logits, frac_logits,
                    self.stack.action_decoder.frac_log_std,
                    acted_int,
                    src_idx if acted_int else 0,
                    tgt_idx if acted_int else 0,
                    frac_z,
                ).item()

            rollout = {
                "acted": acted_int,
                "src_idx": src_idx,
                "tgt_idx": tgt_idx,
                "frac": frac_val,
                "frac_z": frac_z,
                "value": value,
                "log_prob": log_prob,
                "pid_to_idx": pid_to_idx,
                "src_legal_mask": src_legal.cpu(),
                "tgt_legal_mask": tgt_legal.cpu(),
                "batch": batch,
                # Bernoulli logit + p_acted on the *raw* policy. Used to
                # count exploration-driven launches: when acted_int=1 but
                # p_acted<0.5, the sample beat the argmax (entropy bonus
                # / stochasticity, not the policy's own preference).
                "acted_logit": acted_logit_val,
            }
            return rollout

        return idx_to_pid.get(src_idx), idx_to_pid.get(tgt_idx), pid_to_idx

    def act(self, obs: dict[str, Any], frac_override: float | None = None) -> list[list]:
        """Return ``[[planet_id, angle, ships], ...]`` action triples."""
        learner_slot = int(obs.get("player", 0)) if isinstance(obs, dict) else int(obs.player)
        source_pid, target_pid, _ = self._predict(obs, learner_slot)
        if source_pid is None or target_pid is None:
            return []
        return self._target_to_moves(
            source_pid, target_pid, obs, learner_slot, frac_override=frac_override,
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
            angle = math.atan2(target.y - source.y, target.x - source.x)
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
