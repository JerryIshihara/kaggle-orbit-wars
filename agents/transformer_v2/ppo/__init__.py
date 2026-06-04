"""PPO package for transformer_v2.

Phase 0 minimal-actor implementation matching the design captured in
``docs/PPO_TWO_CPU_PROTOCOL.md`` and ``PPO_PSEUDOCODE.md``:

  * actor = the existing ``pair_logits`` and ``pair_frac`` outputs of the
    supervised ``PairHead`` (no new actor heads)
  * critic = mean pairwise ``PairCompareHead`` probability from learner slot
    0 to valid opponent slots, using post-L2 ``PlayerConsolidator`` state plus
    L2 ``glob``. If the actor ckpt has no consolidator, the old L2 ``glob``
    value_head is an explicit debug fallback.
  * action contract = ``single_target_per_source_v1`` (each owned source row is
    one Categorical over its legal targets + the diagonal HOLD slot; launching
    rows add a logit-normal frac with fixed sigma)
  * BC anchor = single-target CE to match the action contract
  * L0-L3 frozen for the entire PPO run

This package only implements the **single-machine Phase 0** path. The
distributed Phase 1+ pieces (file-mediated gradient averaging across A and B,
peer driver, archive script) are deferred to a follow-up PR. The
``EnvAdapter`` interface in :mod:`env_adapter` is the integration boundary
with ``kaggle_environments``; the concrete adapter has TODOs at the
env-touching call sites.

Public surface:

    from agents.transformer_v2.ppo import (
        PPOActorCritic,
        sample_single_target,
        project_to_env,
        compute_advantages,
        ppo_update_local,
    )
"""

from __future__ import annotations

from .actor_critic import PPOActorCritic
from .gae import compute_advantages
from .learner import ppo_update_local
from .loss import action_logprob, ppo_minibatch_loss, single_target_bc
from .sampler import (
    Action,
    legality_masks,
    project_to_env,
    sample_single_target,
)

__all__ = [
    "Action",
    "PPOActorCritic",
    "action_logprob",
    "compute_advantages",
    "legality_masks",
    "ppo_minibatch_loss",
    "ppo_update_local",
    "project_to_env",
    "sample_single_target",
    "single_target_bc",
]
