"""Env adapter — the integration boundary with ``kaggle_environments``.

Why a separate module: rollout/PPO code stays env-API-agnostic. Swap
``KaggleOrbitWarsAdapter`` for a vectorized adapter / a fast Python
re-implementation / a Rust backend later without touching the PPO loop.

This file is intentionally a SCAFFOLD. The PPO math + sampler + loss + GAE
are wired and self-consistent without it; the call sites with ``TODO:
ENV-WIRE`` are the only places that touch the kaggle env directly. They
need to be filled in (and tested) before rollouts can run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


# --------------------------------------------------------------------------- #
# Observation + StepResult dataclasses                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Obs:
    """One snapshot's view, normalized for both seats.

    ``learner_seat`` is the seat (0 or 1 in 2P games) the learner is
    playing this episode. Featurization and legality masks are produced
    with this seat as the "self" reference.
    """

    raw_observation: Any        # the kaggle_environments observation dict
    learner_seat: int
    step_idx: int
    n_planets: int               # number of REAL planets at this step
    # Per-planet arrays (P-length, padded to max_planets):
    planet_owner_rel: torch.Tensor   # (P,) long, learner-relative; 0 = me
    planet_surplus: torch.Tensor     # (P,) float
    planet_exists: torch.Tensor      # (P,) bool
    planet_ids: list[int]            # (P,) — env planet id per slot (for emit)


@dataclass
class StepResult:
    obs_next: Obs
    reward: float                    # learner-perspective reward for THIS step
    done: bool
    info: dict[str, Any]


# --------------------------------------------------------------------------- #
# Abstract base                                                                #
# --------------------------------------------------------------------------- #
class EnvAdapter(ABC):
    """Minimal env interface the PPO rollout needs."""

    @abstractmethod
    def reset(self, *, seed: int, learner_seat: int,
              opponent_policy: Any) -> Obs: ...

    @abstractmethod
    def step(self, learner_env_actions: list[list[int]]) -> StepResult:
        """Advance the env by ONE turn.

        ``learner_env_actions`` is the projected output of
        :func:`agents.transformer_v2.ppo.project_to_env` — a list of
        ``[from_planet_id, angle, num_ships]`` triples. Empty list means
        the learner takes no launches this turn (NOOP or empty target set).

        The adapter calls the opponent policy internally to produce the
        opponent's actions for the same turn.
        """

    @abstractmethod
    def is_done(self) -> bool: ...

    @abstractmethod
    def winner(self) -> int | None: ...

    @abstractmethod
    def featurize(self, obs: Obs) -> dict[str, torch.Tensor]:
        """Turn an Obs into the model-forward feats dict.

        Must return tensors shaped to match the model's ``forward``
        signature: planet_tokens, fleet_tokens, routing dict, planet_mask,
        is_comet, pair_type_ids.
        """

    @abstractmethod
    def plan_launch(self, src_planet_id: int, tgt_planet_id: int,
                     ships: int) -> Any:
        """Wraps ``agents.transformer_v2.physics_utils.plan_launch``.

        Returns the same object shape as upstream plan_launch
        (``.ok``, ``.angle``, ``.reason``).
        """


# --------------------------------------------------------------------------- #
# Kaggle adapter — scaffold (env-touching call sites marked TODO)             #
# --------------------------------------------------------------------------- #
class KaggleOrbitWarsAdapter(EnvAdapter):
    """Wraps ``kaggle_environments.make("orbit_wars", ...)``.

    Wire-up TODOs are below. The PPO math doesn't depend on this; rollouts
    can't run until the TODOs land. Test plan: a smoke test that runs
    one episode of random_v1 vs random_v1 and checks the returned shapes.
    """

    def __init__(self, max_planets: int = 64, max_fleets: int = 1024,
                  history_T: int = 1):
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.history_T = history_T
        self._env = None
        self._learner_seat = 0
        self._opponent_policy = None

    def reset(self, *, seed: int, learner_seat: int,
              opponent_policy: Any) -> Obs:
        # TODO: ENV-WIRE
        # from kaggle_environments import make
        # self._env = make("orbit_wars", configuration={"seed": seed})
        # self._env.reset(num_agents=2)
        raise NotImplementedError("KaggleOrbitWarsAdapter.reset not wired yet")

    def step(self, learner_env_actions: list[list[int]]) -> StepResult:
        # TODO: ENV-WIRE
        # opp_obs = self._env.steps[-1][1 - self._learner_seat].observation
        # opp_action = self._opponent_policy(opp_obs)
        # actions = [None, None]
        # actions[self._learner_seat] = learner_env_actions
        # actions[1 - self._learner_seat] = opp_action
        # self._env.step(actions)
        # ... compute reward from new state ...
        raise NotImplementedError("KaggleOrbitWarsAdapter.step not wired yet")

    def is_done(self) -> bool:
        # TODO: ENV-WIRE
        raise NotImplementedError("KaggleOrbitWarsAdapter.is_done not wired yet")

    def winner(self) -> int | None:
        # TODO: ENV-WIRE
        raise NotImplementedError("KaggleOrbitWarsAdapter.winner not wired yet")

    def featurize(self, obs: Obs) -> dict[str, torch.Tensor]:
        # TODO: ENV-WIRE — call agents.transformer_v2.featurizer.inference here.
        # Returns the dict that PPOActorCritic.forward expects.
        raise NotImplementedError("KaggleOrbitWarsAdapter.featurize not wired yet")

    def plan_launch(self, src_planet_id: int, tgt_planet_id: int,
                     ships: int) -> Any:
        # Direct passthrough — the underlying util doesn't need adapter state.
        from agents.transformer_v2.physics_utils import plan_launch
        return plan_launch(self._env, src_planet_id, tgt_planet_id, ships)
