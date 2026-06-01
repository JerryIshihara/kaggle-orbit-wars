"""PPO learner step — runs on Colab GPU, one iteration of the distributed loop.

Pull + version-check the rollout shards for policy_vK from GCS, run GAE + the PPO
update (exactly the monolithic train.py update, reused), and publish BOTH the full
``policy_v{K+1}.pt`` (archival/resume) AND the <1 MB ``policy_v{K+1}.heads.pt``
trainable-delta (the only thing the VM downloads next iter). Append one row to
``train_log.jsonl``. Then exit. See docs/PPO_TWO_CPU_PROTOCOL.md + the plan.

Example (driven by notebooks/ppo_distributed_colab.ipynb):
  python -m agents.transformer_v2.ppo.learner_step \
    --policy-ckpt gs://.../ppo/<run>/checkpoints/policy_v3.pt \
    --shards      gs://.../ppo/<run>/rollouts/v3/ \
    --out-ckpt    gs://.../ppo/<run>/checkpoints/policy_v4.pt \
    --out-heads   gs://.../ppo/<run>/heads/policy_v4.heads.pt \
    --train-log   gs://.../ppo/<run>/train_log.jsonl \
    --policy-version 3 --base-version <base_id> --device cuda \
    --ckpt <entity_best.pt> --fleet/planet/comet-run-dir <dirs> \
    --pair-cache-path <cache> --bc-coef 0.05 --minibatch-size 256 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from . import shards
from .actor_critic import PPOActorCritic
from .learner import PPOConfig, ppo_update_local
from .smoke import _PPOWithL0, episodes_to_ppo, load_supervised


def _maybe_download(url: str | None, tmp: Path) -> str | None:
    if not url or not str(url).startswith("gs://"):
        return url
    local = tmp / Path(url).name
    shards.gcs_cp(url, local)
    return str(local)


def _collect_shards(shards_url: str, tmp: Path, *, policy_version: int,
                    max_shards: int | None = None) -> tuple[list, int]:
    """Download rollouts/vK/ tree, version-check every shard, return (episodes, history_window)."""
    local = tmp / "rollouts"
    shards.gcs_download_dir(shards_url, local)
    manifests = sorted(local.rglob("manifest.json"))
    if not manifests:
        raise RuntimeError(f"no manifest.json under {shards_url} (rollout incomplete?)")
    hw_seen: set[int] = set()
    items: list = []   # (man_path, hw, entry) for every shard, version-checked
    for man_path in manifests:
        man = shards.read_manifest(man_path)
        hw = int(man["history_window"])
        hw_seen.add(hw)
        if man["policy_version"] != policy_version:
            raise shards.ShardVersionError(
                f"manifest {man_path} policy_version {man['policy_version']} != {policy_version}")
        for entry in man["shards"]:
            items.append((man_path, hw, entry))
    total = len(items)
    if max_shards is not None and 0 < max_shards < total:
        # Stride-sample a representative SUBSET (every total/N-th shard) so it spans
        # all actors' seed-slices, not just the first one or two. Bounds the in-RAM
        # EpisodeBuffers to fit the learner host memory on big (multi-VM) rollouts.
        stride = total / max_shards
        items = [items[int(i * stride)] for i in range(max_shards)]
        print(f"[learner] --max-shards={max_shards}: loading {len(items)}/{total} "
              f"shards (stride-sampled subset to fit RAM)", flush=True)
    episodes = []
    for man_path, hw, entry in items:
        sp = man_path.parent / entry["name"]
        if not sp.exists():
            raise RuntimeError(f"shard {sp} listed in manifest but missing on disk")
        ep = shards.load_shard(
            sp, expect_policy_version=policy_version,
            expect_contract=shards.POLICY_ACTION_CONTRACT, expect_history_window=hw)
        episodes.append(ep)
    if len(hw_seen) != 1:
        raise shards.ShardVersionError(f"shards disagree on history_window: {hw_seen}")
    return episodes, hw_seen.pop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-ckpt", required=True, help="Full policy_vK ckpt (local or gs://).")
    ap.add_argument("--shards", required=True, help="GCS prefix gs://.../rollouts/vK/")
    ap.add_argument("--out-ckpt", required=True, help="Where to write full policy_v{K+1} (gs:// or local).")
    ap.add_argument("--out-heads", required=True, help="Where to write the <1 MB head-delta (gs:// or local).")
    ap.add_argument("--train-log", default=None, help="gs:// train_log.jsonl to append to.")
    ap.add_argument("--policy-version", type=int, required=True, help="K (the shards' version).")
    ap.add_argument("--base-version", default="base", help="Frozen-backbone id stamped into the head-delta.")
    # frozen base
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--fleet-run-dir", type=Path, default=None)
    ap.add_argument("--planet-run-dir", type=Path, default=None)
    ap.add_argument("--comet-run-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    # PPO hyperparams
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--clip", type=float, default=0.10)
    ap.add_argument("--target-kl", type=float, default=0.01)
    ap.add_argument("--lr-heads", type=float, default=1e-4)
    ap.add_argument("--lr-trunk", type=float, default=None,
                    help="lr for the unfrozen trunk/L3/L4 (default=lr_heads). At "
                         "--freeze-phase>=1 set this SMALLER than lr_heads so PPO "
                         "doesn't wreck the supervised backbone.")
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--freeze-phase", type=int, default=0,
                    help="freeze_for_phase: 0=action heads, 1=+trunk/FiLM/L4, "
                         "2=+L3 dual_role.")
    # BC anchor
    ap.add_argument("--pair-cache-path", type=Path, default=None)
    ap.add_argument("--bc-coef", type=float, default=0.0)
    ap.add_argument("--bc-target-weight", type=float, default=1.0)
    ap.add_argument("--max-shards", type=int, default=None,
                    help="cap shards loaded into RAM (stride-sampled across actors) "
                         "to fit host memory on big rollouts; default = all.")
    args = ap.parse_args()
    K = args.policy_version
    tmp = Path(tempfile.mkdtemp(prefix="ppo_learner_"))
    t0 = time.time()
    print(f"[learner] === iter {K} → v{K+1} (device={args.device}) ===", flush=True)

    # ---- frozen base + load policy_vK ----
    entity_model, fleet_enc, planet_enc, comet_enc, cfg = load_supervised(
        args.ckpt, args.device, planet_run_dir=args.planet_run_dir,
        fleet_run_dir=args.fleet_run_dir, comet_run_dir=args.comet_run_dir)
    # PPO critic = the value head inside PPOActorCritic (trained during PPO); no
    # separate critic ckpt. Matches smoke.py / rollout_worker.
    policy = PPOActorCritic(
        entity_model, sigma=args.sigma, allow_debug_glob_critic=True,
    ).to(args.device)
    pol_local = _maybe_download(args.policy_ckpt, tmp)
    pstate = torch.load(pol_local, map_location=args.device, weights_only=False)
    res = policy.load_state_dict(pstate["policy"], strict=False)
    print(f"[learner] loaded policy_v{K} (missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)})", flush=True)
    groups = policy.freeze_for_phase(args.freeze_phase)
    print(f"[learner] freeze_for_phase({args.freeze_phase}): {groups}", flush=True)
    train_policy = _PPOWithL0(policy, planet_enc, fleet_enc, comet_enc).to(args.device)

    # ---- pull + version-check shards ----
    t_pull = time.time()
    episodes, hw = _collect_shards(args.shards, tmp, policy_version=K,
                                   max_shards=args.max_shards)
    n_steps = sum(len(e.steps) for e in episodes)
    n_wins = sum(int(e.winner == e.learner_seat) for e in episodes)
    print(f"[learner] pulled {len(episodes)} shards, {n_steps} steps, history_window={hw}, "
          f"versions OK ({time.time()-t_pull:.1f}s)", flush=True)
    if not episodes:
        print("[learner] no episodes — aborting", flush=True)
        return 1

    # ---- GAE + update ----
    ep_objs, mbs = episodes_to_ppo(episodes, minibatch_size=args.minibatch_size, device=args.device)
    print(f"[learner] episodes_to_ppo: {len(episodes)} eps, {n_steps} steps, {len(mbs)} minibatches",
          flush=True)
    bc_sampler = None
    bc_coef = 0.0
    if args.pair_cache_path is not None:
        from .bc_cache import BCCacheSampler
        bc_sampler = BCCacheSampler(args.pair_cache_path, device=args.device, acted_only=True)
        bc_coef = float(args.bc_coef)
    bc_source = ((lambda n, _s=bc_sampler: _s.sample(n)) if bc_sampler is not None
                 else (lambda _n: None))
    cfg_ppo = PPOConfig(clip=args.clip, target_kl=args.target_kl, epochs=args.epochs,
                        minibatch_size=args.minibatch_size, lr_heads=args.lr_heads, lr_trunk=args.lr_trunk,
                        value_coef=args.value_coef, ent_coef=args.ent_coef,
                        bc_coef=bc_coef, bc_target_weight=args.bc_target_weight)
    t_upd = time.time()
    metrics = ppo_update_local(train_policy, ep_objs, mbs, bc_minibatch_source=bc_source, cfg=cfg_ppo)
    for em in metrics.get("epoch_metrics", []):
        print(f"[learner]   epoch {em.get('epoch')}: kl={em.get('avg_kl',0):.4f} "
              f"policy_loss={em.get('policy_loss',0):.4f} value_loss={em.get('value_loss',0):.4f} "
              f"entropy={em.get('entropy',0):.3f} clip_frac={em.get('clip_frac',0):.3f}"
              f"{' EARLY-STOP' if em.get('early_stopped') else ''}", flush=True)
    ems = metrics.get("epoch_metrics", [])
    last = ems[-1] if ems else {}
    kl0 = ems[0].get("avg_kl", 0.0) if ems else 0.0
    early = any(e.get("early_stopped") for e in ems)
    print("[learner] ===== PPO TRAINING SIGNALS =====", flush=True)
    print(f"[learner]   kl={last.get('avg_kl',0):.4f}  (target {args.target_kl}; "
          f"epoch1→last {kl0:.4f}→{last.get('avg_kl',0):.4f}"
          f"{'; EARLY-STOP hit' if early else ''})", flush=True)
    print(f"[learner]   policy_loss={last.get('policy_loss',0):.4f}  "
          f"value_loss={last.get('value_loss',0):.4f}  entropy={last.get('entropy',0):.3f}  "
          f"clip_frac={last.get('clip_frac',0):.3f}", flush=True)
    print(f"[learner]   game status: {n_wins}/{len(episodes)} learner-seat wins "
          f"({n_wins/max(1,len(episodes))*100:.0f}%) over {n_steps} steps", flush=True)

    # ---- publish full ckpt + head-delta ----
    full_local = tmp / f"policy_v{K+1}.pt"
    torch.save({"policy": policy.state_dict(),
                "value_hidden": policy.value_head[0].out_features,
                "sigma": float(policy.sigma.item()),
                "iter": K + 1, "metrics": {"winrate": n_wins / max(1, len(episodes))}},
               full_local)
    heads_local = tmp / f"policy_v{K+1}.heads.pt"
    dinfo = shards.save_trainable_delta(policy, heads_local, base_version=args.base_version)
    shards.gcs_cp(str(full_local), args.out_ckpt)
    shards.gcs_cp(str(heads_local), args.out_heads)
    print(f"[learner] saved policy_v{K+1}: full {full_local.stat().st_size/1024/1024:.1f} MB → {args.out_ckpt}",
          flush=True)
    print(f"[learner] saved head-delta: {dinfo['n_keys']} keys / {dinfo['n_params']} params "
          f"({dinfo['bytes']/1024:.1f} KB) → {args.out_heads}", flush=True)

    # ---- append train_log row ----
    row = {"iter": K, "policy_version": K, "n_episodes": len(episodes), "n_steps": n_steps,
           "winrate": round(n_wins / max(1, len(episodes)), 3), "history_window": hw,
           "avg_kl": last.get("avg_kl"), "policy_loss": last.get("policy_loss"),
           "value_loss": last.get("value_loss"), "entropy": last.get("entropy"),
           "clip_frac": last.get("clip_frac"), "update_s": round(time.time() - t_upd, 1),
           "wall_s": round(time.time() - t0, 1), "ts": shards.utcnow_iso()}
    if args.train_log:
        existing = shards.read_progress(args.train_log)  # gcloud storage cat (None if absent)
        log_local = tmp / "train_log.jsonl"
        prefix = ""
        if isinstance(existing, dict):  # cat returned a single json object — unlikely for jsonl
            prefix = json.dumps(existing) + "\n"
        # read raw jsonl via cat
        try:
            import subprocess
            raw = subprocess.run(["gcloud", "storage", "cat", args.train_log],
                                 capture_output=True, text=True)
            prefix = raw.stdout if raw.returncode == 0 else ""
        except Exception:
            prefix = ""
        log_local.write_text(prefix + json.dumps(row) + "\n")
        shards.gcs_cp(str(log_local), args.train_log)
    print(f"[learner] DONE iter {K}: winrate={row['winrate']} | "
          f"kl={row['avg_kl']} policy_loss={row['policy_loss']} value_loss={row['value_loss']} "
          f"entropy={row['entropy']} clip_frac={row['clip_frac']} | "
          f"wall={row['wall_s']}s; next rollout = v{K+1} head-delta", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
