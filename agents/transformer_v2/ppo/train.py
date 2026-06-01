"""Multi-iteration PPO training loop (single machine, Phase 0).

Wraps the smoke runner's rollout + GAE + update in an outer loop, saves a
policy checkpoint per iteration, appends a JSONL row per iteration to a
train log.

This is the **Phase 0 single-machine path**. Distributed Phase 1+ (file-
mediated grad averaging across A and B), the eval gate, the self-play pool,
and the archive script are NOT in this file yet — they are spec'd in
``docs/PPO_TWO_CPU_PROTOCOL.md`` and ship in follow-up PRs. This file is
the smallest thing that actually trains: rollout N episodes per iter, do
one PPO update, save, log, repeat.

Usage::

    python -m agents.transformer_v2.ppo.train \\
        --ckpt data/runs/entity/<run>/entity_encoder_best.pt \\
        --run-id ppo_smoke_<ts> \\
        --iters 10 --episodes-per-iter 10 \\
        --opponent random_v1 \\
        --device cpu
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from .actor_critic import PPOActorCritic
from .learner import PPOConfig, ppo_update_local
from .smoke import (
    _PPOWithL0,
    episodes_to_ppo,
    load_supervised,
    make_opponent_closure,
    run_episode,
)


_ROLLOUT_WORKER: dict = {}


def _aggregate_invalid_reasons(episodes) -> dict[str, int]:
    """Histogram of invalid-launch reasons across all steps of the iter."""
    hist: dict[str, int] = {}
    for ep in episodes:
        for step in ep.steps:
            for r in step.invalid_reasons:
                hist[r] = hist.get(r, 0) + 1
    return hist


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _new_entity_model_from_cfg(cfg: dict):
    from agents.transformer_v2.pretrain.entity_encoder import EntityPretrainModel

    return EntityPretrainModel(
        d_model=cfg.get("d_model", 256),
        n_steps=cfg.get("n_steps", 6),
        d_pair=cfg.get("d_pair", cfg.get("d_model", 256)),
        entity_n_heads=cfg.get("entity_n_heads", 4),
        cross_n_heads=cfg.get("cross_n_heads", 8),
        cross_n_layers=cfg.get("cross_n_layers", 2),
        dual_n_heads=cfg.get("dual_n_heads", 4),
        conditioner_n_layers=cfg.get("conditioner_n_layers", 1),
        head_n_layers=cfg.get("head_n_layers", 1),
    )


def _load_critic_for_rollout(critic_ckpt: str | None, cfg: dict):
    if not critic_ckpt:
        return None
    from agents.transformer_v2.pretrain.cross_entity import CrossEntityCriticModel

    cstate = torch.load(Path(critic_ckpt), map_location="cpu", weights_only=False)
    ccfg = cstate.get("config", {})
    critic = CrossEntityCriticModel(
        d_model=int(ccfg.get("d_model", cfg.get("d_model", 256))),
    )
    critic.load_state_dict(cstate.get("model", cstate), strict=False)
    return critic


def _rollout_worker_init(
    ckpt: str,
    fleet_run_dir: str | None,
    planet_run_dir: str | None,
    comet_run_dir: str | None,
    critic_ckpt: str | None,
    policy_state: dict[str, torch.Tensor],
    opponent_id: str | None,
    sigma: float,
    noop_logit_bias: float,
    max_planets: int,
    max_fleets: int,
    target_cap_k_max: int,
    target_cap_lambda: float,
) -> None:
    """Initialize one CPU rollout worker for the current PPO iteration."""
    torch.set_num_threads(1)
    entity_model, fleet_enc, planet_enc, comet_enc, cfg = load_supervised(
        Path(ckpt),
        "cpu",
        planet_run_dir=Path(planet_run_dir) if planet_run_dir else None,
        fleet_run_dir=Path(fleet_run_dir) if fleet_run_dir else None,
        comet_run_dir=Path(comet_run_dir) if comet_run_dir else None,
    )
    critic_model = _load_critic_for_rollout(critic_ckpt, cfg)
    policy = PPOActorCritic(
        entity_model,
        sigma=sigma,
        critic_model=critic_model,
        allow_debug_glob_critic=(critic_model is None),
    ).to("cpu")
    policy.load_state_dict(policy_state, strict=False)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    opp_policy = None
    if opponent_id is None:
        opp_policy = PPOActorCritic(
            entity_model=_new_entity_model_from_cfg(cfg),
            sigma=sigma,
            allow_debug_glob_critic=True,
        ).to("cpu")
        opp_policy.load_state_dict(policy_state, strict=False)
        opp_policy.eval()
        for p in opp_policy.parameters():
            p.requires_grad_(False)

    _ROLLOUT_WORKER.clear()
    _ROLLOUT_WORKER.update({
        "policy": policy,
        "opp_policy": opp_policy,
        "planet_enc": planet_enc,
        "fleet_enc": fleet_enc,
        "comet_enc": comet_enc,
        "opponent_id": opponent_id,
        "sigma": sigma,
        "noop_logit_bias": noop_logit_bias,
        "max_planets": max_planets,
        "max_fleets": max_fleets,
        "target_cap_k_max": target_cap_k_max,
        "target_cap_lambda": target_cap_lambda,
    })
    print(f"[train:worker {os.getpid()}] ready", flush=True)


def _rollout_worker_run(task):
    """Run a chunk of ``(seed, learner_seat)`` episode specs on CPU."""
    w = _ROLLOUT_WORKER
    if isinstance(task, dict):
        specs = task["specs"]
        policy_state = task.get("policy_state")
        if policy_state is not None:
            w["policy"].load_state_dict(policy_state, strict=False)
            if w["opp_policy"] is not None:
                w["opp_policy"].load_state_dict(policy_state, strict=False)
    else:
        specs = task
    episodes = []
    pid = os.getpid()
    n_specs = len(specs)
    print(f"[train:worker {pid}] chunk start episodes={n_specs}", flush=True)
    for i, (seed, learner_seat) in enumerate(specs, start=1):
        t_ep = time.time()
        opp_fn = None
        if w["opp_policy"] is not None:
            opp_fn = make_opponent_closure(
                opponent_policy=w["opp_policy"],
                planet_enc=w["planet_enc"],
                fleet_enc=w["fleet_enc"],
                comet_enc=w["comet_enc"],
                opponent_slot=1 - learner_seat,
                device="cpu",
                max_planets=w["max_planets"],
                max_fleets=w["max_fleets"],
                sigma=w["sigma"],
                noop_logit_bias=w["noop_logit_bias"],
            )
        ep = run_episode(
            policy=w["policy"],
            planet_enc=w["planet_enc"],
            fleet_enc=w["fleet_enc"],
            comet_enc=w["comet_enc"],
            seed=seed,
            learner_seat=learner_seat,
            opponent_id=w["opponent_id"] if opp_fn is None else None,
            opponent_fn=opp_fn,
            device="cpu",
            max_planets=w["max_planets"],
            max_fleets=w["max_fleets"],
            sigma=w["sigma"],
            noop_logit_bias=w["noop_logit_bias"],
            target_cap_k_max=w["target_cap_k_max"],
            target_cap_lambda=w["target_cap_lambda"],
        )
        episodes.append(ep)
        print(
            f"[train:worker {pid}] episode {i}/{n_specs} "
            f"seed={seed} seat={learner_seat} "
            f"steps={len(ep.steps)} winner={ep.winner} "
            f"sec={time.time() - t_ep:.1f}",
            flush=True,
        )
    return episodes


def _chunk_specs(specs: list[tuple[int, int]], n_chunks: int) -> list[list[tuple[int, int]]]:
    chunks = [[] for _ in range(n_chunks)]
    for i, spec in enumerate(specs):
        chunks[i % n_chunks].append(spec)
    return [c for c in chunks if c]


def _rollout_parallel(
    *,
    args,
    policy: PPOActorCritic,
    episode_specs: list[tuple[int, int]],
    n_workers: int,
    pool: concurrent.futures.ProcessPoolExecutor | None = None,
) -> list:
    n_workers = max(1, min(int(n_workers), len(episode_specs)))
    chunks = _chunk_specs(episode_specs, n_workers)
    print(
        f"[train] parallel rollout start: workers={n_workers} "
        f"episodes={len(episode_specs)} "
        f"chunk_sizes={[len(c) for c in chunks]} "
        f"pool={'persistent' if pool is not None else 'per_iter'}",
        flush=True,
    )
    policy_state = _state_dict_cpu(policy)
    episodes = []
    if pool is not None:
        futs = [
            pool.submit(_rollout_worker_run, {
                "specs": chunk,
                "policy_state": policy_state,
            })
            for chunk in chunks
        ]
        for fut in concurrent.futures.as_completed(futs):
            episodes.extend(fut.result())
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_rollout_worker_init,
            initargs=(
                str(args.ckpt),
                str(args.fleet_run_dir) if args.fleet_run_dir is not None else None,
                str(args.planet_run_dir) if args.planet_run_dir is not None else None,
                str(args.comet_run_dir) if args.comet_run_dir is not None else None,
                str(args.critic_ckpt) if args.critic_ckpt is not None else None,
                policy_state,
                args.opponent,
                float(args.sigma),
                float(args.noop_logit_bias),
                int(args.max_planets),
                int(args.max_fleets),
                int(args.target_cap_k_max),
                float(args.target_cap_lambda),
            ),
        ) as pool:
            futs = [pool.submit(_rollout_worker_run, chunk) for chunk in chunks]
            for fut in concurrent.futures.as_completed(futs):
                episodes.extend(fut.result())
    episodes.sort(key=lambda ep: ep.seed)
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path,
                         help="Supervised PairHead .pt to bootstrap from. "
                              "Always required (needed for L0 + entity_model config).")
    parser.add_argument("--fleet-run-dir", type=Path, default=None,
                         help="Directory containing fleet_encoder_best.pt. "
                              "Defaults to the repo data/runs path.")
    parser.add_argument("--planet-run-dir", type=Path, default=None,
                         help="Directory containing planet_encoder_best.pt. "
                              "Defaults to the repo data/runs path.")
    parser.add_argument("--comet-run-dir", type=Path, default=None,
                         help="Directory containing comet_past_best.pt. "
                              "Defaults to the repo data/runs path.")
    parser.add_argument("--resume-ppo-ckpt", type=Path, default=None,
                         help="Optional PPO checkpoint to resume from "
                              "(e.g. policy_v10.pt). Loads policy weights AFTER "
                              "the supervised bootstrap. Use this to continue a "
                              "prior training run.")
    parser.add_argument("--critic-ckpt", type=Path, default=None,
                         help="Optional pretrained pairwise value model "
                              "(head_set='critic' cross_entity ckpt). The "
                              "redesigned critic provides PairCompareHead + "
                              "PlayerConsolidator weights; PPO uses "
                              "V(s)=2*mean_j sigmoid(pair_compare(0,j))-1.")
    parser.add_argument("--allow-glob-critic-fallback", action="store_true",
                         help="Debug only. Allow PPO to run without "
                              "--critic-ckpt using the legacy value_head on "
                              "actor L2 glob. Production PPO should not use "
                              "this path; critic must branch from L1.")
    parser.add_argument("--run-id", required=True,
                         help="Run id; outputs go to data/runs/ppo/<run_id>/.")
    parser.add_argument("--iters", type=int, default=10,
                         help="Number of PPO iterations to run.")
    parser.add_argument("--episodes-per-iter", type=int, default=32,
                         help="Default 32 (vs old 10) — kills the 15pp std-err "
                              "noise floor on winrate metrics.")
    parser.add_argument("--rollout-workers", type=int, default=1,
                         help="Episode-level CPU rollout workers. >1 splits "
                              "episodes across worker processes, then the "
                              "parent still performs GAE/PPO update on --device.")
    parser.add_argument("--opponent", default=None,
                         help="Registered agent id (e.g. random_v1, sniper_v1). "
                              "If unset (default), opponent is a FROZEN snapshot of "
                              "the learner at the start of each iter — true self-play.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-heads", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.10)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--noop-logit-bias", type=float, default=0.0)
    # Soft cap on target count per chosen source — fix for high invalid rate.
    parser.add_argument("--target-cap-k-max", type=int, default=4,
                         help="Soft target-count cap. Penalty kicks in when "
                              "Bernoulli selects more than k_max targets per "
                              "source; the production runner typically launches "
                              "<=3 from one source.")
    parser.add_argument("--target-cap-lambda", type=float, default=0.005,
                         help="Quadratic coefficient: penalty = "
                              "lambda * max(0, k - k_max)^2 subtracted from "
                              "per-step reward. Default 0.005; set 0 to disable.")
    # BC anchor (factorized) — pulls actor heads toward supervised pair set.
    parser.add_argument("--pair-cache-path", type=Path, default=None,
                         help="Path to a _pair_cache .pt file. If set, the "
                              "factorized BC anchor (bc_source + bc_target) is "
                              "enabled at --bc-coef.")
    parser.add_argument("--bc-coef", type=float, default=0.05,
                         help="BC anchor coefficient. Ignored if "
                              "--pair-cache-path is unset.")
    parser.add_argument("--bc-target-weight", type=float, default=1.0)
    parser.add_argument("--seed-base", type=int, default=100_000)
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=1,
                         help="Save a checkpoint every N iterations.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = repo_root / "data" / "runs" / "ppo" / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    config_path = run_dir / "config.json"

    # Persist the run config once at start.
    config_path.write_text(json.dumps({
        "started_at": _iso_now(),
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }, indent=2) + "\n")

    print(f"[train] run_dir={run_dir}", flush=True)
    print(f"[train] device={args.device} ckpt={args.ckpt}", flush=True)
    t0 = time.time()

    entity_model, fleet_enc, planet_enc, comet_enc, cfg = load_supervised(
        args.ckpt,
        args.device,
        planet_run_dir=args.planet_run_dir,
        fleet_run_dir=args.fleet_run_dir,
        comet_run_dir=args.comet_run_dir,
    )
    l0_dirs = cfg.get("_l0_run_dirs", {})
    if l0_dirs:
        print(
            "[train] L0 dirs "
            f"fleet={l0_dirs.get('fleet')} "
            f"planet={l0_dirs.get('planet')} "
            f"comet={l0_dirs.get('comet')}",
            flush=True,
        )
    print(f"[train] loaded supervised ckpt in {time.time()-t0:.1f}s "
          f"(d_model={cfg.get('d_model')}, n_steps={cfg.get('n_steps')}, "
          f"conditioner_n_layers={cfg.get('conditioner_n_layers', 1)}, "
          f"head_n_layers={cfg.get('head_n_layers', 1)})", flush=True)

    critic_model = None
    if args.critic_ckpt is not None:
        from agents.transformer_v2.pretrain.cross_entity import CrossEntityCriticModel
        cstate = torch.load(args.critic_ckpt, map_location=args.device, weights_only=False)
        ccfg = cstate.get("config", {})
        critic_model = CrossEntityCriticModel(d_model=int(ccfg.get("d_model", cfg.get("d_model", 256))))
        cres = critic_model.load_state_dict(cstate.get("model", cstate), strict=False)
        print(f"[train] loaded critic ckpt {args.critic_ckpt}: "
              f"missing={len(cres.missing_keys)} unexpected={len(cres.unexpected_keys)} "
              f"-> V(s)=2*mean_j sigmoid(pair_compare(0,j))-1", flush=True)
    elif not args.allow_glob_critic_fallback:
        print(
            "[train] ERROR: --critic-ckpt is required. The PPO critic must "
            "use a pretrained PairCompareHead; the legacy "
            "L2-glob value_head is debug-only. Pass "
            "--allow-glob-critic-fallback only for plumbing smoke tests.",
            flush=True,
        )
        return 2
    else:
        print(
            "[train] WARNING: using legacy L2-glob value_head fallback. "
            "This violates the current PPO critic design and is intended "
            "only for smoke/debug runs.",
            flush=True,
        )

    policy = PPOActorCritic(
        entity_model,
        sigma=args.sigma,
        critic_model=critic_model,
        allow_debug_glob_critic=args.allow_glob_critic_fallback,
    ).to(args.device)

    # Resume from a prior PPO checkpoint (overrides the supervised bootstrap
    # for the policy weights; supervised ckpt was still needed for L0 + the
    # entity_model config).
    if args.resume_ppo_ckpt is not None:
        resume = torch.load(args.resume_ppo_ckpt, map_location=args.device,
                             weights_only=False)
        result = policy.load_state_dict(resume["policy"], strict=False)
        miss = len(result.missing_keys) if hasattr(result, "missing_keys") else 0
        unexp = len(result.unexpected_keys) if hasattr(result, "unexpected_keys") else 0
        print(f"[train] resumed from {args.resume_ppo_ckpt} "
              f"(missing={miss} unexpected={unexp} "
              f"sigma={resume.get('sigma', 'n/a')} "
              f"prev_iter={resume.get('iter', 'n/a')})", flush=True)

    breakdown = policy.freeze_for_phase(0)
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[train] freeze_for_phase(0) → trainable={n_trainable:,} "
          f"by group: {breakdown}", flush=True)

    train_policy = _PPOWithL0(policy, planet_enc, fleet_enc, comet_enc).to(args.device)

    # BC anchor sampler — built once if a pair cache path is given.
    bc_sampler = None
    bc_coef = 0.0
    if args.pair_cache_path is not None:
        from .bc_cache import BCCacheSampler
        bc_sampler = BCCacheSampler(args.pair_cache_path, device=args.device,
                                     acted_only=True)
        bc_coef = float(args.bc_coef)
        print(f"[train] BC anchor enabled (bc_coef={bc_coef}) — "
              f"{len(bc_sampler)} acted snapshots available.", flush=True)
    else:
        print(f"[train] BC anchor DISABLED (no --pair-cache-path).", flush=True)

    cfg_ppo = PPOConfig(
        clip=args.clip,
        target_kl=args.target_kl,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr_heads=args.lr_heads,
        lr_trunk=None,
        value_coef=args.value_coef,
        ent_coef=args.ent_coef,
        bc_coef=bc_coef,
        bc_target_weight=args.bc_target_weight,
    )

    # Save v0 (the bootstrap state).
    torch.save({"policy": policy.state_dict(),
                "value_hidden": policy.value_head[0].out_features,
                "sigma": float(policy.sigma.item())},
               ckpt_dir / "policy_v0.pt")

    rollout_pool = None
    if args.rollout_workers > 1:
        rollout_worker_count = max(1, min(int(args.rollout_workers), int(args.episodes_per_iter)))
        ctx = mp.get_context("spawn")
        print(
            f"[train] starting persistent rollout pool: workers={rollout_worker_count}",
            flush=True,
        )
        rollout_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=rollout_worker_count,
            mp_context=ctx,
            initializer=_rollout_worker_init,
            initargs=(
                str(args.ckpt),
                str(args.fleet_run_dir) if args.fleet_run_dir is not None else None,
                str(args.planet_run_dir) if args.planet_run_dir is not None else None,
                str(args.comet_run_dir) if args.comet_run_dir is not None else None,
                str(args.critic_ckpt) if args.critic_ckpt is not None else None,
                _state_dict_cpu(policy),
                args.opponent,
                float(args.sigma),
                float(args.noop_logit_bias),
                int(args.max_planets),
                int(args.max_fleets),
                int(args.target_cap_k_max),
                float(args.target_cap_lambda),
            ),
        )

    seed_cursor = args.seed_base
    for K in range(args.iters):
        t_iter = time.time()
        opp_label = args.opponent if args.opponent else f"self@v{K}"
        print(f"\n[train] === iter {K:03d} (vs {opp_label}) ===", flush=True)

        # ---- Self-play: snapshot the learner's current weights into a
        #      frozen opponent policy at the start of this iter ----
        opp_policy = None
        if args.opponent is None and args.rollout_workers <= 1:
            opp_policy = PPOActorCritic(
                # Build a fresh entity_model with the SAME config.
                entity_model=type(entity_model)(
                    d_model=cfg.get("d_model", 256),
                    n_steps=cfg.get("n_steps", 6),
                    d_pair=cfg.get("d_pair", cfg.get("d_model", 256)),
                    entity_n_heads=cfg.get("entity_n_heads", 4),
                    cross_n_heads=cfg.get("cross_n_heads", 8),
                    cross_n_layers=cfg.get("cross_n_layers", 2),
                    dual_n_heads=cfg.get("dual_n_heads", 4),
                    conditioner_n_layers=cfg.get("conditioner_n_layers", 1),
                    head_n_layers=cfg.get("head_n_layers", 1),
                ),
                sigma=args.sigma,
                allow_debug_glob_critic=True,
            ).to(args.device)
            # Copy weights from the learner; from this point on the
            # opponent is frozen for the entire iter's rollouts.
            # The frozen opponent only needs the actor path. When the learner
            # uses a separate per-player critic, its state dict has
            # ``critic_model.*`` keys but this opponent wrapper intentionally
            # omits that critic to keep rollout cheaper. Load non-strictly so
            # actor/value_head-compatible keys still mirror the learner.
            opp_policy.load_state_dict(policy.state_dict(), strict=False)
            opp_policy.eval()
            for p in opp_policy.parameters():
                p.requires_grad_(False)

        # ---- Rollout ----
        n_wins = 0
        total_steps = 0
        total_invalid = 0
        total_emitted = 0
        total_noop = 0
        t_roll = time.time()
        episode_specs = [
            (seed_cursor + i, i % 2)
            for i in range(args.episodes_per_iter)
        ]
        seed_cursor += args.episodes_per_iter
        try:
            if args.rollout_workers > 1:
                episodes = _rollout_parallel(
                    args=args,
                    policy=policy,
                    episode_specs=episode_specs,
                    n_workers=args.rollout_workers,
                    pool=rollout_pool,
                )
            else:
                episodes = []
                for seed, learner_seat in episode_specs:
                    opp_fn = None
                    if opp_policy is not None:
                        opp_fn = make_opponent_closure(
                            opponent_policy=opp_policy,
                            planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
                            opponent_slot=1 - learner_seat,
                            device=args.device,
                            max_planets=args.max_planets, max_fleets=args.max_fleets,
                            sigma=args.sigma, noop_logit_bias=args.noop_logit_bias,
                        )
                    episodes.append(run_episode(
                        policy=policy, planet_enc=planet_enc, fleet_enc=fleet_enc,
                        comet_enc=comet_enc, seed=seed, learner_seat=learner_seat,
                        opponent_id=args.opponent if opp_fn is None else None,
                        opponent_fn=opp_fn,
                        device=args.device,
                        max_planets=args.max_planets, max_fleets=args.max_fleets,
                        sigma=args.sigma, noop_logit_bias=args.noop_logit_bias,
                        target_cap_k_max=args.target_cap_k_max,
                        target_cap_lambda=args.target_cap_lambda,
                    ))
        except Exception as e:
            print(f"[train] iter {K} rollout FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            if rollout_pool is not None:
                rollout_pool.shutdown(wait=False, cancel_futures=True)
            return 1

        for ep in episodes:
            n_wins += int(ep.winner == ep.learner_seat)
            total_steps += len(ep.steps)
            total_invalid += sum(s.invalid_launch for s in ep.steps)
            total_emitted += sum(s.emitted_launch for s in ep.steps)
            total_noop += sum(1 for s in ep.steps if s.action.source_id < 0)
        rollout_sec = time.time() - t_roll
        print(f"[train] rollout: {args.episodes_per_iter} eps "
              f"({max(1, args.rollout_workers)} workers), "
              f"{n_wins} wins, {total_steps} learner steps, "
              f"noop={total_noop} emit={total_emitted} inval={total_invalid} "
              f"({rollout_sec:.1f}s)", flush=True)

        # ---- GAE + minibatches ----
        t_gae = time.time()
        ep_objs, mbs = episodes_to_ppo(
            episodes, minibatch_size=args.minibatch_size, device=args.device,
        )
        gae_sec = time.time() - t_gae
        if not mbs:
            print(f"[train] iter {K}: no steps to train on — skipping update",
                  flush=True)
            continue

        # ---- Update ----
        t_upd = time.time()
        bc_source = (
            (lambda n, _s=bc_sampler: _s.sample(n)) if bc_sampler is not None
            else (lambda _n: None)
        )
        metrics = ppo_update_local(
            train_policy, ep_objs, mbs,
            bc_minibatch_source=bc_source,
            cfg=cfg_ppo,
        )
        update_sec = time.time() - t_upd

        # Aggregate epoch metrics (last epoch wins for KL / clip_frac).
        last_epoch = metrics["epoch_metrics"][-1] if metrics["epoch_metrics"] else {}
        first_epoch = metrics["epoch_metrics"][0] if metrics["epoch_metrics"] else {}
        early = any(em.get("early_stopped") for em in metrics["epoch_metrics"])

        # Final reward (for diagnostics).
        mean_terminal_reward = (
            sum((ep.steps[-1].reward - sum(s.reward for s in ep.steps[:-1])
                  if len(ep.steps) > 1 else ep.steps[-1].reward)
                 for ep in episodes if ep.steps)
            / max(1, sum(1 for ep in episodes if ep.steps))
        )

        # Invalid-reason histogram across all steps in this iter's episodes.
        inv_hist = _aggregate_invalid_reasons(episodes)

        # Bernoulli target-count distribution (helps tune the soft cap).
        all_k = [s.n_selected_targets for ep in episodes for s in ep.steps]
        if all_k:
            k_mean = sum(all_k) / len(all_k)
            sorted_k = sorted(all_k)
            k_p50 = sorted_k[len(sorted_k) // 2]
            k_p95 = sorted_k[min(len(sorted_k) - 1, int(0.95 * len(sorted_k)))]
            k_over_cap = sum(1 for k in all_k if k > args.target_cap_k_max)
        else:
            k_mean = k_p50 = k_p95 = k_over_cap = 0

        log_row = {
            "iter": K,
            "policy_version": K + 1,
            "timestamp": _iso_now(),
            "n_episodes": args.episodes_per_iter,
            "opponent": opp_label,
            "n_wins": n_wins,
            "winrate": n_wins / args.episodes_per_iter,
            "invalid_reasons": inv_hist,
            "target_count_mean": k_mean,
            "target_count_p50": k_p50,
            "target_count_p95": k_p95,
            "n_steps_over_cap": k_over_cap,
            "bc_enabled": bc_sampler is not None,
            "n_learner_steps": total_steps,
            "n_noop_steps": total_noop,
            "n_emitted_launches": total_emitted,
            "n_invalid_launches": total_invalid,
            "invalid_rate_per_step": total_invalid / max(1, total_steps),
            "mean_terminal_reward": mean_terminal_reward,
            "n_minibatches": len(mbs),
            "epochs_ran": len(metrics["epoch_metrics"]),
            "early_stopped": early,
            "kl_first_epoch": first_epoch.get("avg_kl"),
            "kl_last_epoch": last_epoch.get("avg_kl"),
            "policy_loss_last": last_epoch.get("policy_loss"),
            "value_loss_last": last_epoch.get("value_loss"),
            "entropy_last": last_epoch.get("entropy"),
            "clip_frac_last": last_epoch.get("clip_frac"),
            "rollout_sec": rollout_sec,
            "gae_sec": gae_sec,
            "update_sec": update_sec,
            "iter_sec": time.time() - t_iter,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(log_row) + "\n")

        print(f"[train] iter {K:03d} done in {log_row['iter_sec']:.1f}s | "
              f"winrate={log_row['winrate']:.2f} "
              f"kl={log_row['kl_last_epoch']:.4f} "
              f"value_loss={log_row['value_loss_last']:.3f} "
              f"entropy={log_row['entropy_last']:.2f} "
              f"clip_frac={log_row['clip_frac_last']:.3f}"
              f"{' EARLY' if early else ''}", flush=True)

        # ---- Save ckpt ----
        if (K + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"policy_v{K+1}.pt"
            torch.save({
                "policy": policy.state_dict(),
                "value_hidden": policy.value_head[0].out_features,
                "sigma": float(policy.sigma.item()),
                "iter": K,
                "metrics": log_row,
            }, ckpt_path)

    if rollout_pool is not None:
        print("[train] shutting down persistent rollout pool", flush=True)
        rollout_pool.shutdown()

    total_wall = time.time() - t0
    print(f"\n[train] DONE — {args.iters} iters in {total_wall:.1f}s "
          f"({total_wall/max(1,args.iters):.1f}s/iter avg)", flush=True)
    print(f"[train] log: {log_path}", flush=True)
    print(f"[train] checkpoints: {ckpt_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
