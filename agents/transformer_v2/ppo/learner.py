"""Phase 0 single-machine PPO update loop.

The distributed Phase 1+ variant (file-mediated gradient averaging across A
and B, peer driver) is documented in ``docs/PPO_TWO_CPU_PROTOCOL.md`` but
not implemented here — heads-only Phase 0 grads are too cheap to justify
the sync overhead (~12 s/iter vs ~1-2 s of compute).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .gae import Episode, compute_advantages
from .loss import BCMinibatch, PPOMinibatch, ppo_minibatch_loss


@dataclass
class PPOConfig:
    clip: float = 0.10
    target_kl: float = 0.01
    epochs: int = 3
    minibatch_size: int = 1024
    lr_heads: float = 1e-4
    lr_trunk: float | None = None         # None = trunk frozen (Phase 0)
    lr_perception: float | None = None    # L2 cross+consolidator lr; None = fold into trunk. Set << lr_trunk (e.g. 1e-7) — perception drift at lr_trunk inflates policy entropy / destabilizes.
    value_coef: float = 0.5
    ent_coef: float = 0.01
    bc_coef: float = 0.05
    bc_target_weight: float = 1.0
    max_grad_norm: float = 0.5
    early_stop_kl_factor: float = 1.5     # break the epoch if running avg KL > this * target_kl
    select_logit_bias: float = 0.0        # multi-target selection-Bernoulli logit shift; MUST match the rollout sampler's value (else the PPO ratio desyncs)


def build_optimizer(policy: nn.Module, cfg: PPOConfig) -> torch.optim.Optimizer:
    """Build a 2-group AdamW: heads at lr_heads, trunk (if unfrozen) at lr_trunk."""
    # ``value_head`` is the debug glob fallback; ``pair_compare`` is the
    # post-L2 player_state critic. Access both defensively and skip whichever
    # is absent or frozen.
    head_modules = [
        getattr(policy, "value_head", None),
        getattr(policy, "pair_compare", None),
        policy.entity_model.pair_head.pair_head,
        policy.entity_model.pair_head.pair_frac_head,
    ]
    head_param_ids = set()
    head_params: list[torch.nn.Parameter] = []
    for mod in head_modules:
        if mod is None:
            continue
        for p in mod.parameters():
            if p.requires_grad and id(p) not in head_param_ids:
                head_param_ids.add(id(p))
                head_params.append(p)

    # Deep L2 perception (cross + consolidator) gets its own (smaller) lr when
    # cfg.lr_perception is set — layer-wise decay. At the trunk lr these layers
    # drift and inflate the policy entropy (the L2 destabilization). None => fold
    # them into the trunk group (legacy 2-group behaviour).
    perception_param_ids: set[int] = set()
    perception_params: list[torch.nn.Parameter] = []
    if cfg.lr_perception is not None:
        for name in ("cross", "consolidator"):
            mod = getattr(policy.entity_model, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                if p.requires_grad and id(p) not in head_param_ids \
                        and id(p) not in perception_param_ids:
                    perception_param_ids.add(id(p))
                    perception_params.append(p)

    trunk_params: list[torch.nn.Parameter] = []
    for p in policy.parameters():
        if p.requires_grad and id(p) not in head_param_ids \
                and id(p) not in perception_param_ids:
            trunk_params.append(p)

    groups = [{"params": head_params, "lr": cfg.lr_heads}]
    if trunk_params:
        lr_trunk = cfg.lr_trunk if cfg.lr_trunk is not None else cfg.lr_heads
        groups.append({"params": trunk_params, "lr": lr_trunk})
    if perception_params:
        groups.append({"params": perception_params, "lr": cfg.lr_perception})
    return torch.optim.AdamW(groups, weight_decay=0.0)


def _policy_device(policy: nn.Module) -> torch.device:
    for p in policy.parameters():
        return p.device
    return torch.device("cpu")


def _move_tree(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move_tree(v, device) for k, v in x.items()}
    return x


def _move_ppo_minibatch(mb: PPOMinibatch, device: torch.device) -> PPOMinibatch:
    def _mv(t):
        return None if t is None else t.to(device, non_blocking=True)

    return PPOMinibatch(
        feats=_move_tree(mb.feats, device),
        pair_mask=mb.pair_mask.to(device, non_blocking=True),
        source_mask=mb.source_mask.to(device, non_blocking=True),
        old_logp=mb.old_logp.to(device, non_blocking=True),
        adv=mb.adv.to(device, non_blocking=True),
        returns=mb.returns.to(device, non_blocking=True),
        tgt_idx=_mv(mb.tgt_idx),
        frac_raw=_mv(mb.frac_raw),
        select_mask=_mv(mb.select_mask),
        alloc_counts=_mv(mb.alloc_counts),
        self_counts=_mv(mb.self_counts),
        select_counts=_mv(mb.select_counts),
        alloc_extras=_mv(mb.alloc_extras),
        self_extras=_mv(mb.self_extras),
        alloc_shares=_mv(mb.alloc_shares),
        noop_logit_bias=mb.noop_logit_bias,
    )


def _move_bc_minibatch(mb: BCMinibatch | None, device: torch.device) -> BCMinibatch | None:
    if mb is None:
        return None
    return BCMinibatch(
        feats=_move_tree(mb.feats, device),
        pair_mask=mb.pair_mask.to(device, non_blocking=True),
        source_mask=mb.source_mask.to(device, non_blocking=True),
        expert_tgt_idx=mb.expert_tgt_idx.to(device, non_blocking=True),
    )


def ppo_update_local(
    policy: nn.Module,                  # PPOActorCritic
    episodes: list[Episode],
    ppo_minibatches: list[PPOMinibatch],
    bc_minibatch_source,                # callable: int -> BCMinibatch | None
    *,
    cfg: PPOConfig = PPOConfig(),
    gamma: float = 0.995,
    lam: float = 0.95,
) -> dict[str, list[float] | float]:
    """One PPO iteration: GAE → multi-epoch minibatch sweep with early-stop on KL.

    The caller is responsible for:
      * Building ``episodes`` from rollout shards with ``Episode.values``
        already filled from the rollout-time forward.
      * Building ``ppo_minibatches`` with the matching ``feats`` / masks /
        actions / old_logp from those same rollouts.
      * Supplying a ``bc_minibatch_source(size: int) -> BCMinibatch`` that
        draws from the supervised pair_cache. Pass ``lambda _: None`` to
        disable the BC anchor.

    Returns a dict of per-epoch metrics for the train log.
    """
    compute_advantages(episodes, gamma=gamma, lam=lam, normalize=True)
    # Patch normalized advantages back into the minibatches that came from the
    # caller. This is a stable identity-mapping if the caller built the
    # minibatches AFTER GAE; if they built them BEFORE GAE, this is wrong —
    # document the contract clearly.
    # ASSUMPTION: ppo_minibatches were built AFTER compute_advantages so their
    # mb.adv already reflects the normalized values. If you need to re-apply
    # GAE here, rebuild the minibatches.

    opt = build_optimizer(policy, cfg)
    device = _policy_device(policy)

    epoch_metrics: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        running_kl = 0.0
        n_mb = 0
        in_epoch_stop = False
        epoch_logs: dict[str, list[float]] = {}
        for mb_cpu in ppo_minibatches:
            mb = _move_ppo_minibatch(mb_cpu, device)
            bc_mb = (
                _move_bc_minibatch(bc_minibatch_source(mb.size), device)
                if cfg.bc_coef > 0 else None
            )
            loss, diag = ppo_minibatch_loss(
                policy, mb, bc_mb,
                clip=cfg.clip,
                value_coef=cfg.value_coef,
                ent_coef=cfg.ent_coef,
                bc_coef=cfg.bc_coef,
                bc_target_weight=cfg.bc_target_weight,
                select_logit_bias=cfg.select_logit_bias,
                # entropy-shape diagnostic on the first minibatch of each epoch
                # only (a subsample — cheap but not free)
                collect_shape=(n_mb == 0),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            opt.step()

            for k, v in diag.items():
                epoch_logs.setdefault(k, []).append(v)
            running_kl += diag["approx_kl"]
            n_mb += 1
            # IN-EPOCH KL brake. With games=64 an epoch is 100+ minibatches;
            # the between-epoch check below would let the policy run
            # arbitrarily far past target inside epoch 0 (each minibatch takes
            # a gradient step). Brake once the running average clears the same
            # early-stop threshold (short warmup so one noisy minibatch can't
            # kill the epoch), or immediately on a detonating minibatch.
            _kl_thresh = cfg.early_stop_kl_factor * cfg.target_kl
            if ((n_mb >= 4 and running_kl / n_mb > _kl_thresh)
                    or diag["approx_kl"] > 4.0 * _kl_thresh):
                in_epoch_stop = True
                break

        avg_kl = running_kl / max(1, n_mb)
        epoch_metrics.append({
            "epoch": epoch,
            "n_minibatches": n_mb,
            "avg_kl": avg_kl,
            **{k: sum(v) / len(v) for k, v in epoch_logs.items() if v},
        })
        if in_epoch_stop:
            epoch_metrics[-1]["early_stopped"] = True
            epoch_metrics[-1]["in_epoch_stopped_at_mb"] = float(n_mb)
            break
        if avg_kl > cfg.early_stop_kl_factor * cfg.target_kl:
            epoch_metrics[-1]["early_stopped"] = True
            break

    return {
        "epoch_metrics": epoch_metrics,
        "n_episodes": len(episodes),
        "n_minibatches_per_epoch": len(ppo_minibatches),
    }
