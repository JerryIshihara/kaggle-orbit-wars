"""Single-process Phase 0 rollout loop.

Plays one episode through the env, sampling actor decisions per learner
turn, projecting to env launches, and accumulating reward. Returns the
per-step tensors that :func:`agents.transformer_v2.ppo.gae.compute_advantages`
and :class:`agents.transformer_v2.ppo.loss.PPOMinibatch` expect.

Out of scope for this MVP: batched envs per worker process,
``torch.compile`` on the policy, shared opponent model instances across
workers. Those are spec'd in ``docs/PPO_TWO_CPU_PROTOCOL.md`` under "CPU
performance → Rollout" and should be added incrementally with measured
throughput, not pre-optimized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .env_adapter import EnvAdapter, Obs
from .sampler import Action, legality_masks, project_to_env, sample_single_target


@dataclass
class EpisodeShards:
    """Per-step tensors collected for one learner episode."""

    # Model inputs at action time — list across steps; stacked at the end.
    feats_per_step: list[dict[str, torch.Tensor]] = field(default_factory=list)
    pair_masks: list[torch.Tensor] = field(default_factory=list)
    source_masks: list[torch.Tensor] = field(default_factory=list)
    # Sampled actions (one Action per learner step).
    actions: list[Action] = field(default_factory=list)
    # Per-step scalars.
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)
    # Diagnostics (per step).
    invalid_launch: list[int] = field(default_factory=list)
    emitted_launch: list[int] = field(default_factory=list)
    n_selected_targets: list[int] = field(default_factory=list)
    invalid_reasons: list[list[str]] = field(default_factory=list)
    # Episode-level.
    winner: int | None = None
    seed: int = 0
    learner_seat: int = 0


def run_episode(
    policy,                        # PPOActorCritic in eval/no-grad mode
    adapter: EnvAdapter,
    *,
    seed: int,
    learner_seat: int,
    opponent_policy,
    min_launch: int,
    noop_logit_bias: float = 0.0,
    reward_shaper=None,            # callable: (info, invalid) -> float
    invalid_penalty: float = 0.01,
) -> EpisodeShards:
    """Roll out one learner episode and return the per-step shards.

    Reward shaping: the env reports terminal +1 / -1 / 0; the optional
    ``reward_shaper`` adds dense potential-based shaping per step (see
    protocol "Reward design"). Invalid-launch penalty is subtracted
    per-step when ``ProjectionResult.ok`` is False.
    """
    if reward_shaper is None:
        # Trivial shaper: pass-through env reward, no dense shaping.
        def reward_shaper(info, invalid):  # noqa: E306
            return float(info.get("env_reward", 0.0)) - invalid_penalty * float(invalid)

    shards = EpisodeShards(seed=seed, learner_seat=learner_seat)

    obs = adapter.reset(seed=seed, learner_seat=learner_seat,
                         opponent_policy=opponent_policy)
    policy.eval()
    with torch.inference_mode():
        while not adapter.is_done():
            feats = adapter.featurize(obs)
            # Add a batch dim of 1; the model accepts (B, P, d).
            feats_b = {k: v.unsqueeze(0) if v.dim() < 4 else v for k, v in feats.items()}
            out = policy(**feats_b)
            pair_logits = out["pair_logits"][0]    # (P, P)
            frac_loc = out["frac_loc"][0]          # (P, P)
            value = float(out["value"][0].item())

            pair_mask, source_mask = legality_masks(
                planet_owner=obs.planet_owner_rel,
                surplus=obs.planet_surplus,
                planet_exists=obs.planet_exists,
                min_launch=min_launch,
            )
            action = sample_single_target(
                pair_logits, frac_loc, out["sigma"],
                pair_mask=pair_mask, source_mask=source_mask,
                noop_logit_bias=noop_logit_bias,
            )

            proj = project_to_env(
                action,
                source_mask=source_mask,
                surplus=obs.planet_surplus,
                source_planet_ids=obs.planet_ids,
                target_planet_ids=obs.planet_ids,
                min_launch=min_launch,
                plan_launch_fn=adapter.plan_launch,
            )
            env_actions = proj.env_actions
            n_invalid = proj.n_invalid
            invalid_reasons = proj.invalid_reasons
            n_emitted = proj.n_emitted

            step_result = adapter.step(env_actions)
            reward = reward_shaper(step_result.info, n_invalid)

            shards.feats_per_step.append(feats)
            shards.pair_masks.append(pair_mask)
            shards.source_masks.append(source_mask)
            shards.actions.append(action)
            shards.values.append(value)
            shards.rewards.append(reward)
            shards.dones.append(1.0 if step_result.done else 0.0)
            shards.invalid_launch.append(n_invalid)
            shards.emitted_launch.append(n_emitted)
            shards.n_selected_targets.append(int(action.n_launch))
            shards.invalid_reasons.append(invalid_reasons)

            obs = step_result.obs_next

    shards.winner = adapter.winner()
    return shards
