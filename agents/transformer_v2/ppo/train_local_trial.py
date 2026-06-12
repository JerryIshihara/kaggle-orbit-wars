"""Single-machine PPO TRIAL — 2 iterations × 64 mixed 2P/4P self-play games at T=10.

Purpose: validate the rollout → process → update loop end-to-end on one RunPod box,
with VERBOSE streaming logs. This is the staged bring-up, NOT the production trainer:
  * rollout = by default ``infserver``: CPU worker processes step/featurize envs,
    then the parent process batches ready T=10 seat observations and runs the
    single CUDA PPOActorCritic forward. Workers hold no model parameters.
  * ``pool`` is CPU-forward debug only and requires ``--allow-cpu-forward-rollout``.
  * games = a balanced, seeded 2P/4P mix with diverse seeds; SELF-PLAY (every non-learner
    seat is a frozen snapshot of the learner at iter start).
  * T=10 history window (matches the joint pretrain model).
  * process = design-A P1 PBRS shaping (Φ from the 5 value signals) + GAE via
    ``episodes_to_ppo``.
  * update = ``ppo_update_local``.

Known TRIAL simplifications (fine for an infra/throughput trial, fix before a real run —
see /tmp/ppo_single_machine_plan.md §1):
  * legacy critic fallback is still available only when ``--reward-decomp`` is
    not set;
  * PPO sampler is bernoulli_select_multinomial_alloc_v2 (HOLD = the frac
    head's diagonal), matching v2 --multinomial-alloc pretrained actors;
  * optimizer is rebuilt per iter inside ``ppo_update_local``.

Example (local tiny smoke):
  .venv/bin/python -m agents.transformer_v2.ppo.train_local_trial \\
    --ckpt data/runs/joint/<run>/joint_best.pt \\
    --planet-run-dir ckpts/planet --fleet-run-dir ckpts/fleet --comet-run-dir ckpts/comet \\
    --iters 1 --games 4 --rollout batched --device cpu --max-fleets 256

Full trial (RunPod 32 vCPU):
  ... --iters 2 --games 64 --rollout infserver --device cuda --procs 28 --post-procs 4
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import multiprocessing as mp
import queue
import shutil
import socket
import random
import threading
import time
from pathlib import Path

import torch

from . import shards
from .actor_critic import PPOActorCritic
from .batched_rollout import run_batched_episodes
from .infserver_rollout import run_infserver_rollout
from .infserver_shm_rollout import run_shm_rollout
from .learner import PPOConfig, ppo_update_local
from .pool_rollout import run_pool_rollout
from .smoke import (
    _PPOWithL0,
    _pack_episode_for_ppo,
    _packed_episodes_to_ppo,
    episodes_to_ppo,
    episodes_to_ppo_parallel,
    load_supervised,
)


_LOG_FH = None  # set in main(): tee every _log line to out_dir/train.log
_STREAM = None  # optional async ppo_state/Firestore/local/GCS stream publisher


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    line = f"[trial {_ts()}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()
    if _STREAM is not None:
        _STREAM.log(line)


class _StreamPublisher:
    """Async progress/log publisher for local/GCS/Firestore state URLs.

    URL examples:
      * ``/tmp/ppo_trial_STATE.json`` local JSON, polled by stream_trial.py
      * ``gs://bucket/ppo/run/STATE.json`` GCS JSON polling
      * ``fs://ppo_runs/run_id`` Firestore push streaming via ppo_state.stream

    Failures are non-fatal: training must continue even if telemetry auth or
    network is unavailable.
    """

    def __init__(self, url: str, *, run_id: str, config: dict):
        self.url = str(url)
        self.run_id = str(run_id)
        self.config = dict(config)
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.disabled = False
        self.thread = None
        self._init_state()
        if not self.disabled:
            self.thread = threading.Thread(target=self._loop, name="ppo-stream", daemon=True)
            self.thread.start()

    def _init_state(self) -> None:
        try:
            from . import ppo_state

            st = ppo_state.read_state(self.url)
            if st is None:
                st = {
                    "run_id": self.run_id,
                    "gcs_prefix": self.config.get("gcs_out") or None,
                    "phase": ppo_state.PHASE_TRAINING,
                    "iter": 0,
                    "iters_total": int(self.config.get("iters_total", 0)),
                    "model": None,
                    "shards": None,
                    "config": self.config,
                    ppo_state.ACTOR: {
                        "id": None, "claimed_iter": -1, "heartbeat_ts": None,
                        "progress": None, "log_tail": [], "log_n": 0,
                    },
                    ppo_state.LEARNER: {
                        "id": self.run_id, "claimed_iter": 0, "heartbeat_ts": None,
                        "progress": None, "log_tail": [], "log_n": 0,
                    },
                    "actors": {},
                    "updated_by": self.run_id,
                    "message": "single-machine PPO trial stream",
                }
            else:
                st = dict(st)
                st["phase"] = ppo_state.PHASE_TRAINING
                st["run_id"] = st.get("run_id") or self.run_id
                st["config"] = {**dict(st.get("config") or {}), **self.config}
                st.setdefault(ppo_state.LEARNER, {
                    "id": self.run_id, "claimed_iter": 0, "heartbeat_ts": None,
                    "progress": None, "log_tail": [], "log_n": 0,
                })
                st["updated_by"] = self.run_id
            ppo_state.write_state(self.url, st)
        except Exception as exc:  # noqa: BLE001
            self.disabled = True
            print(f"[stream] disabled: {exc}", flush=True)

    def log(self, line: str) -> None:
        if not self.disabled:
            self.q.put(("log", str(line)))

    def progress(self, payload: dict) -> None:
        if not self.disabled:
            self.q.put(("progress", dict(payload)))

    def close(self, *, phase: str = "done") -> None:
        if self.disabled:
            return
        self.q.put(("phase", str(phase)))
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def _loop(self) -> None:
        from . import ppo_state

        pending_lines: list[str] = []
        pending_progress: dict | None = None
        pending_phase: str | None = None
        last_flush = time.time()

        def _flush() -> None:
            nonlocal pending_lines, pending_progress, pending_phase, last_flush
            if not (pending_lines or pending_progress is not None or pending_phase):
                return
            try:
                if pending_phase:
                    st = ppo_state.read_state(self.url) or {}
                    st = dict(st)
                    st["phase"] = pending_phase
                    st["updated_by"] = self.run_id
                    ppo_state.write_state(self.url, st)
                ppo_state.set_progress(
                    self.url,
                    ppo_state.LEARNER,
                    progress=pending_progress,
                    log_lines=pending_lines,
                    who=self.run_id,
                )
            except Exception as exc:  # noqa: BLE001
                self.disabled = True
                print(f"[stream] disabled: {exc}", flush=True)
            pending_lines = []
            pending_progress = None
            pending_phase = None
            last_flush = time.time()

        while not (self.stop.is_set() and self.q.empty()):
            timeout = max(0.1, 0.5 - (time.time() - last_flush))
            try:
                kind, payload = self.q.get(timeout=timeout)
            except queue.Empty:
                _flush()
                continue
            if kind == "log":
                pending_lines.append(payload)
            elif kind == "progress":
                pending_progress = payload
            elif kind == "phase":
                pending_phase = payload
            if len(pending_lines) >= 8 or (time.time() - last_flush) >= 0.5:
                _flush()
        _flush()


def _stream_progress(payload: dict) -> None:
    if _STREAM is not None:
        _STREAM.progress(payload)


def _gcs_cp(local: Path, gcs_out: str) -> None:
    """Best-effort upload (durability for the ephemeral RunPod pod). Non-fatal."""
    if not gcs_out:
        return
    dst = gcs_out.rstrip("/") + "/" + local.name
    try:
        shards.gcs_cp(str(local), dst)
        _log(f"uploaded {local.name} → {dst}")
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING gcs upload failed for {local.name}: {e}")


def _gcs_upload_tree(local_dir: Path, gcs_out: str, rel_prefix: str) -> bool:
    """Best-effort recursive upload under ``gcs_out/rel_prefix``."""
    if not gcs_out:
        return False
    dst = gcs_out.rstrip("/") + "/" + rel_prefix.strip("/")
    try:
        shards.gcs_upload_dir(local_dir, dst)
        _log(f"uploaded {local_dir.name}/ → {dst}/")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING gcs upload failed for {local_dir}: {e}")
        return False


_CONTRACT_NAMES = {
    "v4": "bounded_k_select_dirichlet_alloc_v4",
    "v3": "bounded_k_select_multinomial_alloc_v3",
}


def _vtag(policy_version: int) -> str:
    return f"v{int(policy_version):04d}"


def _machine_id() -> str:
    return socket.gethostname().split(".")[0] or "local"


def _save_policy_checkpoint(
    policy: PPOActorCritic,
    *,
    out_dir: Path,
    policy_version: int,
    cfg: dict,
    args,
    metrics: dict | None = None,
) -> tuple[Path, Path]:
    """Save both canonical versioned and legacy checkpoint names."""
    tag = _vtag(policy_version)
    payload = {
        "policy": policy.state_dict(),
        "sigma": float(policy.sigma.item()),
        "policy_version": int(policy_version),
        "iter": int(policy_version),
        "config": cfg,
        "ppo_trial": {
            "unfreeze": args.unfreeze,
            "history_window": args.history_window,
            "max_planets": args.max_planets,
            "max_fleets": args.max_fleets,
            "reward_decomp": bool(args.reward_decomp),
            "action_contract": _CONTRACT_NAMES.get(
                args.action_contract, "bernoulli_select_multinomial_alloc_v2"),
            "select_k_max": int(args.select_k_max),
            "created_ts": shards.utcnow_iso(),
            "metrics": dict(metrics or {}),
        },
    }
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    versioned = ckpt_dir / f"policy_{tag}.pt"
    legacy = out_dir / f"policy_v{int(policy_version)}.pt"
    torch.save(payload, versioned)
    shutil.copy2(versioned, legacy)
    latest = ckpt_dir / "policy_latest.pt"
    shutil.copy2(versioned, latest)
    return versioned, legacy


def _save_rollout_artifacts(
    *,
    episodes,
    packed,
    specs,
    out_dir: Path,
    policy_version: int,
    args,
    machine_id: str,
    row: dict,
    save_raw: bool = True,
) -> tuple[Path, dict]:
    """Write versioned rollout artifacts for a single policy version."""
    tag = _vtag(policy_version)
    root = out_dir / "rollouts" / tag / machine_id
    raw_dir = root / "raw"
    packed_dir = root / "ppo_packed"
    entries = []
    if save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for i, (ep, spec) in enumerate(zip(episodes, specs)):
            if ep is None:
                continue
            ent = shards.save_shard(
                ep,
                raw_dir / f"shard_{i:06d}.pt",
                policy_version=policy_version,
                history_window=args.history_window,
                git_sha_str=shards.git_sha(),
                vm_id=machine_id,
                opponent_ckpt="self_play_same_policy",
            )
            ent["num_players"] = int(spec[1])
            entries.append(ent)

    packed_entries = []
    if packed is not None:
        packed_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(packed):
            path = packed_dir / f"packed_{i:06d}.pt"
            payload = {
                "format": "ppo_packed_v1",
                "policy_version": int(policy_version),
                "history_window": int(args.history_window),
                "policy_action_contract": _CONTRACT_NAMES.get(args.action_contract, shards.POLICY_ACTION_CONTRACT),
                "created_ts": shards.utcnow_iso(),
                "packed": p,
            }
            torch.save(payload, path)
            packed_entries.append({
                "name": path.name,
                "sha256": shards.sha256_file(path),
                "bytes": path.stat().st_size,
                "n_steps": int(p["values"].shape[0]),
            })

    manifest_path = root / "manifest.json"
    manifest = {
        "format": "ppo_rollout_manifest_v2",
        "policy_version": int(policy_version),
        "policy_tag": tag,
        "policy_action_contract": _CONTRACT_NAMES.get(args.action_contract, shards.POLICY_ACTION_CONTRACT),
        "history_window": int(args.history_window),
        "machine_id": machine_id,
        "created_ts": shards.utcnow_iso(),
        "run": {
            "rollout": args.rollout,
            "games": int(args.games),
            "num_players": args.num_players,
            "seed_base": int(args.seed_base),
            "max_planets": int(args.max_planets),
            "max_fleets": int(args.max_fleets),
            "reward_decomp": bool(args.reward_decomp),
            "artifact_mode": (
                "raw_and_packed" if save_raw else "packed_only"
            ),
        },
        "summary": dict(row),
        "raw_shards": entries,
        "ppo_packed": packed_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return root, manifest


_P1_KEYS = ("planet_features", "planet_mask", "fleet_features", "fleet_mask",
            "fleet_target_idx", "fleet_source_idx", "fleet_owner_slot",
            "fleet_ships_log", "fleet_eta_norm")


_P1_SIGNAL_NAMES = ("ship_adv", "prod_adv", "planet_adv", "safety", "fleet_spd")


def apply_phi_shaping(episodes, *, win_weight: float, signal_weights, gamma: float,
                      inval_coef: float = 0.0, log_fn=None):
    """Overwrite each step's reward with the CALCULATED PBRS shaping + terminal win
    (design A). Φ(s)=Σ wᵢ sᵢ(s) from compute_p1_signals on the step's stored frame
    (learner-relative slot 0); r_t = γΦ(s_{t+1}) − Φ(s_t); the terminal step adds
    w_win·z where z=1 for a win and 0 otherwise. Weights are deliberately
    unconstrained; keep their scale small enough that shaping does not dominate
    the terminal win objective.

    ``inval_coef`` > 0 subtracts a DIRECT (non-PBRS) per-step cost on the invalid-
    launch RATE — ``inval_coef · invalid/(invalid+emitted)`` ∈ [0, inval_coef] —
    so the policy is pushed to stop firing illegal cells (the multi-target sampler
    over-fires; ~88% of attempts can be invalid). Direct cost (not a potential
    difference) because we WANT it to change the optimal policy, not stay neutral.
    Returns (mean_phi, mean_shaping_abs) for logging."""
    from agents.transformer_v2.pretrain.value_signals import compute_p1_signals
    sw = torch.as_tensor(signal_weights, dtype=torch.float32)
    all_phi, all_sh = [], []
    sig_buckets: dict[int, list] = {}  # bucket=(step//25) -> [sum(5,), count] across all eps
    with torch.no_grad():
        for ep in episodes:
            if not ep.steps:
                continue
            phis = []
            for t_s, s in enumerate(ep.steps):
                batch = {}
                for k in _P1_KEYS:
                    v = getattr(s, k, None)
                    if v is None:
                        break
                    batch[k] = v.unsqueeze(0)            # add batch dim → (1, …)
                else:
                    sig = compute_p1_signals(batch)[0, 0]  # learner-relative slot 0 → (5,)
                    phis.append(float((sig * sw).sum()))
                    b = t_s // 25                          # 25-step window bucket
                    # [sig_sum(5,), n, inval, emit, fired, fleets, my_fleets]
                    acc = sig_buckets.setdefault(
                        b, [torch.zeros(5), 0, 0, 0, 0, 0.0, 0.0])
                    acc[0] += sig.detach().cpu(); acc[1] += 1
                    acc[2] += int(getattr(s, "invalid_launch", 0) or 0)
                    acc[3] += int(getattr(s, "emitted_launch", 0) or 0)
                    acc[4] += int(getattr(s, "n_selected_targets", 0) or 0)
                    fm = batch["fleet_mask"][0]            # (T, F) — current frame last
                    cur_mask = fm[-1] if fm.dim() == 2 else fm
                    acc[5] += float(cur_mask.sum())
                    fo = batch.get("fleet_owner_slot")
                    if fo is not None:
                        fo = fo[0][-1] if fo[0].dim() == 2 else fo[0]
                        acc[6] += float(((fo == 0) & cur_mask.bool()).sum())
                    continue
                phis.append(phis[-1] if phis else 0.0)     # missing-feature fallback
            for t, s in enumerate(ep.steps):
                phi_next = phis[t + 1] if t + 1 < len(phis) else 0.0   # terminal Φ=0
                s.reward = gamma * phi_next - phis[t]       # PBRS γΦ'−Φ
                if inval_coef:
                    n_inv = int(getattr(s, "invalid_launch", 0) or 0)
                    n_emit = int(getattr(s, "emitted_launch", 0) or 0)
                    inval_rate = n_inv / max(1, n_inv + n_emit)
                    s.reward = s.reward - inval_coef * inval_rate  # direct invalid-firing cost
                s.phi = phis[t]                              # critic reads −Φ analytically
                s.value = s.value - phis[t]                  # keep stored GAE value = V−Φ
                all_sh.append(abs(s.reward))
            z = 1.0 if ep.winner == ep.learner_seat else 0.0
            ep.steps[-1].reward += win_weight * z          # terminal win (z∈{1,0})
            all_phi.extend(phis)
    # Plain-python bucket dict (mp-picklable): {bucket: [sig5_list, n, inval,
    # emit, fired, fleets, my_fleets]} — returned so parallel post workers can
    # ship their buckets back for a merged parent-side table.
    buckets_out = {
        int(b): [acc[0].tolist(), int(acc[1]), int(acc[2]), int(acc[3]),
                 int(acc[4]), float(acc[5]), float(acc[6])]
        for b, acc in sig_buckets.items()
    }
    if log_fn is not None and buckets_out:
        _log_window_tables(buckets_out, log_fn)
    import statistics as _st
    return (_st.mean(all_phi) if all_phi else 0.0,
            _st.mean(all_sh) if all_sh else 0.0,
            buckets_out)


def _merge_window_buckets(dicts):
    """Merge per-episode/job window-bucket dicts (element-wise sums)."""
    merged: dict[int, list] = {}
    for d in dicts:
        for b, row in (d or {}).items():
            b = int(b)
            if b not in merged:
                merged[b] = [list(row[0]), *row[1:]]
                continue
            m = merged[b]
            m[0] = [a + b2 for a, b2 in zip(m[0], row[0])]
            for i in range(1, 7):
                m[i] += row[i]
    return merged


def _log_window_tables(buckets: dict, log_fn) -> None:
    """Print the per-iter 25-step-window tables: 5 P1 signals + rollout stats."""
    if not buckets:
        return
    hdr = "  ".join(f"{n:>9}" for n in _P1_SIGNAL_NAMES)
    log_fn(f"signals (mean per 25-step window) | step-window  n   {hdr}")
    for b in sorted(buckets):
        sig, cnt = buckets[b][0], buckets[b][1]
        row = "  ".join(f"{s / max(1, cnt):>9.3f}" for s in sig)
        log_fn(f"  signals  [{b*25:>3}-{b*25+24:>3}]  {cnt:>4}   {row}")
    log_fn("rollout (per 25-step window) | step-window  n    inval     emit  "
           "inval%   fired/s  fleets/s  myfleets/s")
    for b in sorted(buckets):
        _sig, cnt, inv, emit, fired, fl, myfl = buckets[b]
        rate = inv / max(1, inv + emit)
        log_fn(f"  rollout  [{b*25:>3}-{b*25+24:>3}]  {cnt:>4}  {inv:>7,d}  "
               f"{emit:>7,d}  {rate:>6.1%}  {fired/max(1,cnt):>8.2f}  "
               f"{fl/max(1,cnt):>8.1f}  {myfl/max(1,cnt):>10.1f}")


def _analyze_iter_games(episodes, specs, *, log_fn, out_path=None,
                        sample_every: int = 8):
    """Per-iter WIN/LOSS analysis from the in-memory rollout episodes.

    For every game: seed / players / seat / length / winner, mean of the 5 P1
    signals over the early/mid/late thirds (subsampled every ``sample_every``
    steps), and launch behavior (emitted, fired, noop-step fraction). Logs a
    compact WIN-vs-LOSS contrast block (the per-iter "why did we win/lose")
    and optionally writes the per-game rows as JSON for offline digging.
    """
    import json as _json
    from agents.transformer_v2.pretrain.value_signals import compute_p1_signals

    rows = []
    with torch.no_grad():
        for ep, spec in zip(episodes, specs):
            if ep is None or not ep.steps:
                continue
            seed, npl, seat = spec
            n = len(ep.steps)
            thirds: list[list] = [[], [], []]
            for t_i in range(0, n, max(1, sample_every)):
                s = ep.steps[t_i]
                batch = {}
                for k in _P1_KEYS:
                    v = getattr(s, k, None)
                    if v is None:
                        break
                    batch[k] = v.unsqueeze(0)
                else:
                    sig = compute_p1_signals(batch)[0, 0].tolist()
                    thirds[min(2, (3 * t_i) // n)].append(sig)

            def _mean(group):
                if not group:
                    return [float("nan")] * 5
                return [sum(c) / len(c) for c in zip(*group)]

            emit = sum(s.emitted_launch for s in ep.steps)
            fired = sum(s.n_selected_targets for s in ep.steps)
            noop_frac = sum(
                1 for s in ep.steps if s.n_selected_targets == 0) / n
            rows.append({
                "seed": int(seed), "players": int(npl), "seat": int(seat),
                "won": int(ep.winner == ep.learner_seat)
                       if ep.winner is not None else 0,
                "winner": (None if ep.winner is None else int(ep.winner)),
                "n_steps": int(n),
                "sig_early": _mean(thirds[0]), "sig_mid": _mean(thirds[1]),
                "sig_late": _mean(thirds[2]),
                "emit": int(emit), "fired": int(fired),
                "emit_per_step": round(emit / n, 2),
                "noop_step_frac": round(noop_frac, 3),
            })

    if not rows:
        return rows
    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]

    def _agg(group, key, idx=None):
        vals = [(r[key][idx] if idx is not None else r[key]) for r in group]
        vals = [v for v in vals if v == v]  # drop NaN
        return (sum(vals) / len(vals)) if vals else float("nan")

    names = _P1_SIGNAL_NAMES
    log_fn(f"win/loss analysis: {len(wins)}W / {len(losses)}L "
           f"(2P {sum(r['won'] for r in rows if r['players']==2)}/"
           f"{sum(1 for r in rows if r['players']==2)}, "
           f"4P {sum(r['won'] for r in rows if r['players']==4)}/"
           f"{sum(1 for r in rows if r['players']==4)})")
    for label, grp in (("WIN ", wins), ("LOSS", losses)):
        if not grp:
            continue
        mid = "  ".join(
            f"{names[i][:5]}={_agg(grp, 'sig_mid', i):.3f}" for i in range(5))
        log_fn(f"  {label} n={len(grp):>2}  len={_agg(grp, 'n_steps'):>5.0f}  "
               f"emit/s={_agg(grp, 'emit_per_step'):>5.2f}  "
               f"noop%={100*_agg(grp, 'noop_step_frac'):>4.1f}  mid: {mid}")
    # The tell: which signals separate wins from losses by MID-game (earlier =
    # more causal than late-game, which is mostly the outcome itself).
    if wins and losses:
        gaps = [(_agg(wins, 'sig_mid', i) - _agg(losses, 'sig_mid', i), i)
                for i in range(5)]
        gaps = [(g, i) for g, i in gaps if g == g]
        gaps.sort(key=lambda t: -abs(t[0]))
        log_fn("  mid-game W-L signal gaps: " + "  ".join(
            f"{names[i]}={g:+.3f}" for g, i in gaps))
        for npl in (2, 4):
            seg = [r for r in losses if r["players"] == npl]
            if seg:
                seats: dict[int, int] = {}
                for r in seg:
                    seats[r["seat"]] = seats.get(r["seat"], 0) + 1
                log_fn(f"  {npl}P losses by seat: "
                       + " ".join(f"s{k}:{v}" for k, v in sorted(seats.items()))
                       + "  seeds: "
                       + ",".join(str(r["seed"]) for r in seg[:12]))
    if out_path is not None:
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(_json.dumps(rows, indent=1))
        except Exception as exc:  # noqa: BLE001 — analysis must never kill a run
            log_fn(f"win/loss analysis: JSON write failed ({exc!r})")
    return rows


def _shape_and_pack_episode_job(
    ep,
    *,
    reward_decomp: bool,
    win_weight: float,
    signal_weights,
    gamma: float,
    inval_coef: float = 0.0,
):
    """Post-worker job for one completed game.

    If reward decomposition is active, this overwrites the episode rewards with
    calculated PBRS Φ shaping inside the post worker, then stacks the trajectory
    tensors for PPO. This is safe to run as soon as a game finishes because Φ and
    terminal win labels are per-episode. ``signal_weights`` is the FULL
    5-vector — a previous version took a scalar and shaped with
    ``[scalar]*5``, silently ignoring --signal-weights under POST_PROCS>1.
    Returns the window buckets so the parent can merge + print the per-iter
    tables (signals + rollout) that the in-process path logs directly.
    """
    mphi = 0.0
    msh = 0.0
    buckets: dict = {}
    if reward_decomp:
        mphi, msh, buckets = apply_phi_shaping(
            [ep],
            win_weight=win_weight,
            signal_weights=list(signal_weights),
            gamma=gamma,
            inval_coef=inval_coef,
        )
    packed = _pack_episode_for_ppo(ep)
    return packed, {
        "mean_phi": float(mphi),
        "mean_abs_shape": float(msh),
        "n_steps": int(len(ep.steps)),
        "buckets": buckets,
    }


def _shape_and_pack_episode_path_job(
    ep_path: str,
    *,
    reward_decomp: bool,
    win_weight: float,
    signal_weights,
    gamma: float,
    delete_after: bool = False,
    inval_coef: float = 0.0,
):
    """Path-based post job: avoids sending large tensor EpisodeBuffers via mp queues."""
    try:
        ep = torch.load(ep_path, map_location="cpu", weights_only=False)
        return _shape_and_pack_episode_job(
            ep,
            reward_decomp=reward_decomp,
            win_weight=win_weight,
            signal_weights=signal_weights,
            gamma=gamma,
            inval_coef=inval_coef,
        )
    finally:
        if delete_after:
            Path(ep_path).unlink(missing_ok=True)


def build_specs(n: int, seed_base: int, mode: str) -> list[tuple[int, int, int]]:
    """``[(seed, num_players, learner_seat)]`` — diverse seeds, balanced 2P/4P mix,
    learner seat varied across the available seats. Seeded for reproducibility."""
    rng = random.Random(seed_base)
    specs: list[tuple[int, int, int]] = []
    for _ in range(n):
        npl = rng.choice((2, 4)) if mode == "mix" else int(mode)
        seed = rng.randrange(2 ** 31)
        seat = rng.randrange(npl)
        specs.append((seed, npl, seat))
    return specs


def _verify_pretrained_entity_loaded(
    ckpt: Path,
    entity_model,
    *,
    require_value_heads: bool,
) -> dict[str, int]:
    """Fail before rollout if the staged pretrained .pt did not actually load.

    This compares every tensor from ``ckpt['model']`` against the live
    EntityPretrainModel state dict. It catches stale code.tgz shape drift,
    missing value heads, or accidental fresh-module fallbacks before we spend
    RunPod time collecting rollout.
    """
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    live = entity_model.state_dict()

    missing: list[str] = []
    shape_mismatch: list[str] = []
    value_mismatch: list[str] = []
    matched = 0
    value_head_keys = 0
    for k, saved in model_state.items():
        if k.startswith("value_heads."):
            value_head_keys += 1
        cur = live.get(k)
        if cur is None:
            missing.append(k)
            continue
        if tuple(cur.shape) != tuple(saved.shape):
            shape_mismatch.append(k)
            continue
        cur_cpu = cur.detach().cpu()
        saved_cpu = saved.detach().cpu()
        if not torch.equal(cur_cpu, saved_cpu):
            value_mismatch.append(k)
            continue
        matched += 1

    def _sample(xs: list[str]) -> str:
        return ", ".join(xs[:6]) + (" ..." if len(xs) > 6 else "")

    if require_value_heads and value_head_keys == 0:
        raise RuntimeError(
            "reward-decomp rollout requires pretrained value_heads.*, but "
            f"{ckpt} has none."
        )
    bad = missing or shape_mismatch or value_mismatch
    if bad:
        raise RuntimeError(
            "pretrained ckpt did not load exactly before rollout: "
            f"missing={len(missing)} [{_sample(missing)}] "
            f"shape_mismatch={len(shape_mismatch)} [{_sample(shape_mismatch)}] "
            f"value_mismatch={len(value_mismatch)} [{_sample(value_mismatch)}]"
        )
    return {
        "ckpt_tensors": len(model_state),
        "exact_match": matched,
        "value_heads": value_head_keys,
    }


def _stream_worker_progress(progress, n_procs: int, stop: threading.Event,
                            every_s: float, t0: float | None = None) -> None:
    """Background thread: print each fork-worker's live game row every ``every_s``s,
    so a long rollout streams locally (silence must never look like a hang)."""
    while not stop.wait(every_s):
        rows = []
        steps = []
        done_total = 0
        for slot in range(n_procs):
            row = progress.get(slot)
            if row:
                st = int(row.get("step", 0))
                steps.append(st)
                done_total += int(row.get("games_done", 0))
                status = str(row.get("status", "running"))
                rows.append(
                    f"c{slot}:g{row.get('game_idx')} {row.get('num_players')}P "
                    f"{status} step={st} score={row.get('score_my')}:"
                    f"{row.get('score_enemy_max')} done={row.get('games_done')}"
                )
        if rows:
            elapsed = f" t={time.time() - t0:.0f}s" if t0 is not None else ""
            step_summary = ""
            if steps:
                steps_s = sorted(steps)
                step_summary = (
                    f" step[min/med/max]={steps_s[0]}/"
                    f"{steps_s[len(steps_s)//2]}/{steps_s[-1]}"
                )
            _log(
                f"[roll]{elapsed} cores={n_procs} active_rows={len(rows)} "
                f"worker_done_sum={done_total}{step_summary} | "
                + "  ".join(rows)
            )
            _stream_progress({
                "stage": "rollout",
                "elapsed_s": round(time.time() - t0, 1) if t0 is not None else None,
                "cores": int(n_procs),
                "active_rows": len(rows),
                "worker_done_sum": int(done_total),
                "step_min": min(steps) if steps else 0,
                "step_median": sorted(steps)[len(steps) // 2] if steps else 0,
                "step_max": max(steps) if steps else 0,
                "workers": [dict(progress.get(slot) or {}) for slot in range(n_procs)],
            })


def _log_rl_signals(ep_objs, mbs) -> None:
    """Emit compact RL/GAE/minibatch stats before PPO update."""
    if not ep_objs:
        return

    def _flat(attr: str):
        vals = [getattr(e, attr) for e in ep_objs if getattr(e, attr) is not None]
        return torch.cat(vals) if vals else torch.empty(0)

    def _stats(x: torch.Tensor) -> str:
        if x.numel() == 0:
            return "n=0"
        x = x.detach().float().cpu()
        return (
            f"n={x.numel()} mean={x.mean().item():.4f} "
            f"std={x.std(unbiased=False).item():.4f} "
            f"min={x.min().item():.4f} max={x.max().item():.4f}"
        )

    rewards = _flat("rewards")
    values = _flat("values")
    adv = _flat("advantages")
    returns = _flat("returns")
    ep_returns = torch.tensor(
        [float(e.rewards.sum().item()) for e in ep_objs if e.rewards.numel() > 0],
        dtype=torch.float32,
    )
    mb_sizes = torch.tensor([mb.size for mb in mbs], dtype=torch.float32)
    _log(f"RL signals: reward[{_stats(rewards)}] ep_return[{_stats(ep_returns)}]")
    _log(f"GAE/value: value[{_stats(values)}] adv[{_stats(adv)}] return[{_stats(returns)}]")
    if mb_sizes.numel() > 0:
        _log(f"PPO batches: n_mb={len(mbs)} size[min/mean/max]="
             f"{int(mb_sizes.min().item())}/{mb_sizes.mean().item():.1f}/"
             f"{int(mb_sizes.max().item())}")


def _snapshot_opponent(policy: PPOActorCritic, device: str) -> PPOActorCritic:
    """Frozen self-play opponent = a deep copy of the learner at iter start."""
    opp = copy.deepcopy(policy).to(device)
    opp.eval()
    for p in opp.parameters():
        p.requires_grad_(False)
    return opp


def _apply_unfreeze(policy: PPOActorCritic, level: str) -> dict:
    """Choose the trainable set.

      head : freeze_for_phase(0) — action heads + critic only (Phase 0).
      L4   : freeze_for_phase(1) — + PairHead trunk/FiLM + L4 joint_role.
      L3   : freeze_for_phase(2) — + L3 dual_role.
      L2   : freeze_for_phase(2) + CrossEntityAttention (L2) + PlayerConsolidator
             — i.e. "train from L2 up". L0 specialists + L1 PlanetEntityEncoder stay
             frozen (mirrors the pretrain's freeze_below_l2).
    """
    phase = {"head": 0, "L4": 1, "L3": 2, "L2": 2}[level]
    breakdown = policy.freeze_for_phase(phase)
    if level == "L2":
        em = policy.entity_model
        extra = 0
        for name in ("cross", "consolidator"):
            mod = getattr(em, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)
                    extra += p.numel()
        breakdown["L2_cross+consolidator"] = extra
    if getattr(policy, "reward_decomp", False):
        # design A critic = warm win head (FROZEN, keeps its calibrated winrate) +
        # trainable residual (value_head). freeze_for_phase froze value_head (it's
        # the debug fallback) — re-enable it so the critic actually fine-tunes.
        for p in policy.value_head.parameters():
            p.requires_grad_(True)
        breakdown["value_residual"] = sum(p.numel() for p in policy.value_head.parameters())
    return breakdown


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path, help="joint/supervised actor ckpt")
    p.add_argument("--init-policy", type=Path, default=None,
                   help="resume PPO from a prior policy_vK.pt: build the model from "
                        "--ckpt (architecture) then load this checkpoint's 'policy' "
                        "state_dict on top. Use to continue/tune from an earlier run.")
    p.add_argument("--opponent-ckpt", type=Path, default=None,
                   help="train the learner against a FIXED opponent policy_vK.pt "
                        "(agent A) instead of self-play: the opponent fills all "
                        "non-learner seats with these frozen weights. FORCES --rollout "
                        "batched (infserver/pool are self-play-only). Use for agent-B "
                        "training (B beats A).")
    p.add_argument("--planet-run-dir", type=Path, default=None)
    p.add_argument("--fleet-run-dir", type=Path, default=None)
    p.add_argument("--comet-run-dir", type=Path, default=None)
    p.add_argument("--critic-ckpt", type=Path, default=None,
                   help="optional PairCompare critic; omitted uses reward-decomp residual "
                        "when --reward-decomp is set, otherwise glob-critic fallback "
                        "(trial only)")
    p.add_argument("--out-dir", type=Path, default=Path("data/runs/ppo_trial"))
    # trial shape
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--games", type=int, default=64, help="games per iteration")
    p.add_argument("--num-players", default="mix", choices=["mix", "2", "4"])
    p.add_argument("--unfreeze", default="head", choices=["head", "L4", "L3", "L2"],
                   help="trainable set: head=Phase0 ... L2=train from L2 up "
                        "(cross+consolidator+L3/L4/heads; L0/L1 frozen)")
    p.add_argument("--gcs-out", default="",
                   help="optional gs://… prefix; ckpts + logs uploaded per iter "
                        "(durability for an ephemeral RunPod pod)")
    p.add_argument("--run-id", default="",
                   help="stable run id for stream/artifact metadata; default derives "
                        "from out-dir + timestamp")
    p.add_argument("--stream-url", default="",
                   help="optional live stream state URL: local path, gs://.../STATE.json, "
                        "or fs://<collection>/<doc> for Firestore push streaming")
    p.add_argument("--no-save-rollout-shards", action="store_true",
                   help="disable versioned rollout raw/packed artifact writes")
    p.add_argument("--packed-rollout-artifacts-only", action="store_true",
                   help="save/upload versioned PPO-packed rollout inputs but skip "
                        "raw EpisodeBuffer shards to avoid duplicating huge T-window "
                        "tensors on small RunPod disks")
    p.add_argument("--delete-local-rollouts-after-upload", action="store_true",
                   help="after a successful GCS upload, delete local rollout artifact "
                        "directories and temporary infserver spool files")
    p.add_argument("--rollout", default="infserver", choices=["batched", "pool", "infserver", "shm"],
                   help="batched=GPU lockstep forward (CUDA, single process); "
                        "infserver=CPU env workers + one parent-owned GPU batch "
                        "forwarder; shm=infserver variant with shared-memory request "
                        "plane (no per-step deserialize, ~1.6x faster); "
                        "pool=CPU work-stealing fork pool (CPU only)")
    p.add_argument("--shm-forwarders", type=int, default=1,
                   help="number of parallel GPU forwarder processes for --rollout shm "
                        "(L40 finding: >1 raises GPU util but not throughput, since "
                        "env.step is the floor; keep 1 unless the GPU is saturated)")
    p.add_argument("--allow-cpu-forward-rollout", action="store_true",
                   help="permit --rollout pool CPU-forward debug runs. Without this, "
                        "pool is rejected to avoid accidentally burning GPU-pod time "
                        "on CPU model inference.")
    p.add_argument("--separate-opponent-copy", action="store_true",
                   help="use a frozen deep-copy opponent during self-play. Default is "
                        "one shared policy instance for learner+opponents, so batched "
                        "rollout performs one GPU forward batch across all active seats.")
    p.add_argument("--procs", type=int, default=28,
                   help="fork rollout workers (cores) — only for --rollout pool")
    p.add_argument("--post-procs", type=int, default=0,
                   help="workers reserved for completed-trajectory PPO packing "
                        "(0/1 = serial; useful with --rollout pool, e.g. 28+4 on "
                        "a 32-vCPU pod)")
    p.add_argument("--history-window", type=int, default=10)
    p.add_argument("--max-planets", type=int, default=64)
    p.add_argument("--max-fleets", type=int, default=512)
    p.add_argument("--seed-base", type=int, default=100_000)
    p.add_argument("--progress-step-every", type=int, default=10)
    p.add_argument("--stream-every-s", type=float, default=2.0)
    p.add_argument("--infserver-max-batch", type=int, default=256)
    p.add_argument("--infserver-batch-window-ms", type=float, default=3.0)
    p.add_argument("--infserver-log-every-s", type=float, default=2.0)
    p.add_argument("--infserver-stall-timeout-s", type=float, default=180.0,
                   help="abort infserver rollout if no request/result/progress "
                        "arrives for this many seconds")
    p.add_argument("--infserver-spool-dir", type=Path, default=None,
                   help="optional local temp root for infserver episode spool files; "
                        "use /dev/shm or local disk, not network storage")
    p.add_argument("--delete-infserver-spool-after-post", action="store_true",
                   help="post-pack workers unlink episode spool files after loading "
                        "and packing them")
    # ppo
    p.add_argument("--device", default="cpu")
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr-heads", type=float, default=1e-4)
    p.add_argument("--lr-trunk", type=float, default=None,
                   help="lr for the unfrozen trunk/backbone when --unfreeze is "
                        "L4/L3/L2. Default None falls back to lr_heads (1e-4), which "
                        "is ~100x too hot for the perception layers (kl blows up). "
                        "Use ~1e-6 for stable deep-layer fine-tuning.")
    p.add_argument("--lr-perception", type=float, default=None,
                   help="lr for the L2 perception (cross + consolidator) under "
                        "--unfreeze L2. Default None folds them into --lr-trunk. Set "
                        "<< lr_trunk (e.g. 1e-7) — at lr_trunk the perception drift "
                        "inflates entropy / destabilizes the policy.")
    p.add_argument("--clip", type=float, default=0.10)
    p.add_argument("--target-kl", type=float, default=0.01)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--select-logit-bias", type=float, default=0.0,
                   help="multi-target selection-Bernoulli logit shift: "
                        "fire[s,t] ~ Bernoulli(sigmoid(pair_logits[s,t] - bias)). "
                        "Positive => fewer (more confident) fires; b=2.0 mirrors the "
                        "runner's pair_logits>2.0 threshold. Applied IDENTICALLY in the "
                        "rollout sampler and the PPO update loss (the same value, else "
                        "the ratio desyncs). Default 0.0 = unchanged behaviour.")
    p.add_argument("--action-contract", choices=("v2", "v3", "v4"), default="v2",
                   help="LEARNER action contract: v2 = bernoulli_select_"
                        "multinomial_alloc_v2 (per-cell Bernoulli select + full-N "
                        "multinomial); v3 = bounded_k_select_multinomial_alloc_v3 "
                        "(ONE k-draw select multinomial incl. a self token, "
                        "min_launch floor pre-distributed to fired targets, "
                        "extras multinomial over the remainder — floor-feasible "
                        "by construction). The FIXED OPPONENT always plays v2. "
                        "Under v3, --select-logit-bias is the SELF-token bias. v4 = bounded_k_select_dirichlet_alloc_v4 (v3 select + Dirichlet share draw; needs a ckpt with the alloc_conc head; sizes = min_launch floor + share*remainder).")
    p.add_argument("--lr-conc", type=float, default=5e-6,
               help="v4 alpha0 conc-head lr (own AdamW group; the slow group "
                    "IS the saturation guard - sharing lr_heads detonated in "
                    "stage-B pretrain).")
    p.add_argument("--select-k-max", type=int, default=3,
                   help="v3 only: max select draws per source "
                        "(k = min(this, ships//min_launch))")
    p.add_argument("--reinit-frac-head", action="store_true",
                   help="re-initialize the pair_frac_head (alloc head) fresh "
                        "before PPO — zero final layer => uniform extras "
                        "softmax at start. Use when changing the alloc "
                        "semantics (e.g. first v3 run from a v2 pretrain).")
    # design A reward-decomposition critic + calculated Φ shaping
    p.add_argument("--reward-decomp", action="store_true",
                   help="design A: scalar-winrate critic warm-started from the pretrained "
                        "win head (V=w_win*σ(ŵin)−Φ+residual) + calculated PBRS Φ shaping. "
                        "Requires value heads in the ckpt.")
    p.add_argument("--win-weight", type=float, default=0.5,
                   help="W_win: terminal-win weight (not constrained to sum to 1 with signals)")
    p.add_argument("--signal-weight", type=float, default=0.1,
                   help="per-signal shaping weight (×5 signals in Φ=Σwᵢsᵢ)")
    p.add_argument("--signal-weights", type=str, default=None,
                   help="comma-separated 5 per-signal weights (ship,prod,planet,safety,fleet_speed) "
                        "overriding --signal-weight. Set the 4th (safety) to 0 to drop the saturated "
                        "safety signal, e.g. '0.1,0.1,0.1,0,0.1'.")
    p.add_argument("--value-gamma", type=float, default=0.997,
                   help="γ for the PBRS shaping r=γΦ'−Φ (= the PPO discount)")
    p.add_argument("--inval-coef", type=float, default=0.0,
                   help="DIRECT per-step cost on the invalid-launch RATE "
                        "invalid/(invalid+emitted)∈[0,1]: r -= inval_coef·rate. Pushes the "
                        "over-firing multi-target policy to stop firing illegal cells. "
                        "Default 0.0 = off. ~PBRS per-step |r| is ~0.006, so 0.02 ≈ 3× that.")
    args = p.parse_args()
    if args.action_contract in ("v3", "v4") and args.rollout in ("pool", "shm"):
        raise SystemExit(
            f"--action-contract {args.action_contract} is wired for the batched and infserver "
            "rollouts only; use --rollout batched or infserver."
        )
    if args.rollout == "pool" and not args.allow_cpu_forward_rollout:
        raise SystemExit(
            "--rollout pool is CPU-forward only. Use --rollout infserver --device cuda "
            "for CPU env workers + batched GPU forward, or pass "
            "--allow-cpu-forward-rollout for an explicit CPU debug run."
        )
    if args.rollout == "infserver" and not str(args.device).startswith("cuda"):
        raise SystemExit("--rollout infserver requires --device cuda")
    if args.rollout == "shm" and not str(args.device).startswith("cuda"):
        raise SystemExit("--rollout shm requires --device cuda (set OW_SHM_ALLOW_CPU=1 "
                         "only for parity tests via scripts/test_shm_rollout.py)")

    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"{args.out_dir.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    machine_id = _machine_id()
    global _LOG_FH
    _LOG_FH = open(args.out_dir / "train.log", "a", buffering=1)  # line-buffered tee
    jsonl_path = args.out_dir / "train_log.jsonl"
    global _STREAM
    if args.stream_url:
        _STREAM = _StreamPublisher(
            args.stream_url,
            run_id=run_id,
            config={
                "iters_total": int(args.iters),
                "games_per_iter": int(args.games),
                "out_dir": str(args.out_dir),
                "gcs_out": args.gcs_out,
                "machine_id": machine_id,
                "history_window": int(args.history_window),
                "rollout": args.rollout,
                "procs": int(args.procs),
                "post_procs": int(args.post_procs),
                "packed_rollout_artifacts_only": bool(args.packed_rollout_artifacts_only),
                "delete_local_rollouts_after_upload": bool(args.delete_local_rollouts_after_upload),
                "infserver_spool_dir": (
                    str(args.infserver_spool_dir) if args.infserver_spool_dir else None
                ),
                "delete_infserver_spool_after_post": bool(args.delete_infserver_spool_after_post),
            },
        )
    _log(f"=== PPO RUN === iters={args.iters} games/iter={args.games} "
         f"players={args.num_players} unfreeze={args.unfreeze} T={args.history_window} "
         f"procs={args.procs} post_procs={args.post_procs} device={args.device} "
         f"P={args.max_planets} F={args.max_fleets}")
    _log(f"run_id={run_id} machine_id={machine_id} stream={args.stream_url or '(none)'}")
    _log(f"ckpt={args.ckpt} out_dir={args.out_dir} gcs_out={args.gcs_out or '(none)'}")

    # ---- load actor backbone + L0 encoders ----
    t0 = time.time()
    entity_model, fleet_enc, planet_enc, comet_enc, cfg = load_supervised(
        args.ckpt, args.device,
        planet_run_dir=args.planet_run_dir,
        fleet_run_dir=args.fleet_run_dir,
        comet_run_dir=args.comet_run_dir,
    )
    _log(f"loaded actor in {time.time()-t0:.1f}s (d_model={cfg.get('d_model')} "
         f"n_steps={cfg.get('n_steps')} action_contract={cfg.get('action_contract')})")
    pre = _verify_pretrained_entity_loaded(
        args.ckpt, entity_model,
        require_value_heads=args.reward_decomp,
    )
    _log("pretrained load check: "
         f"ckpt_tensors={pre['ckpt_tensors']} exact_match={pre['exact_match']} "
         f"value_heads={pre['value_heads']} OK")
    if int(cfg.get("n_steps", 0)) != args.history_window:
        _log(f"WARNING: ckpt n_steps={cfg.get('n_steps')} != --history-window "
             f"{args.history_window}; T mismatch would feed the model the wrong window.")

    # v3-arch ckpts walk the UNION offsets (45..0@5 + 18..0@2), not the v2
    # HISTORY_OFFSETS — thread them to every rollout history.
    _history_offsets = None
    if str(cfg.get("arch", "")) == "dual_rate_l2_v3":
        from agents.transformer_v3.history import UNION_HISTORY_OFFSETS
        _history_offsets = [int(o) for o in UNION_HISTORY_OFFSETS]
        if args.history_window != len(_history_offsets):
            raise SystemExit(
                f"v3-arch ckpt needs --history-window {len(_history_offsets)} "
                f"(union offsets walk), got {args.history_window}")
        _log(f"v3 arch: union history offsets {_history_offsets}")
    if args.action_contract == "v4" and \
            getattr(entity_model, "alloc_conc_head", None) is None:
        raise SystemExit(
            "--action-contract v4 needs a ckpt with the alloc_conc head "
            "(with_alloc_conc=True, e.g. jointv4_best.pt)")

    critic_model = None
    if args.critic_ckpt is not None:
        from agents.transformer_v2.pretrain.cross_entity import CrossEntityCriticModel
        cstate = torch.load(args.critic_ckpt, map_location=args.device, weights_only=False)
        ccfg = cstate.get("config", {})
        critic_model = CrossEntityCriticModel(
            d_model=int(ccfg.get("d_model", cfg.get("d_model", 256))))
        critic_model.load_state_dict(cstate.get("model", cstate), strict=False)
        _log(f"loaded critic {args.critic_ckpt}")
    elif not args.reward_decomp:
        _log("no --critic-ckpt → glob-critic fallback (bounded V; TRIAL only)")
    else:
        _log("no --critic-ckpt → reward-decomp critic uses pretrained win head "
             "+ trainable residual")

    _sigw = (
        [float(x) for x in args.signal_weights.split(",")]
        if args.signal_weights else [args.signal_weight] * 5
    )
    if len(_sigw) != 5:
        raise ValueError(f"--signal-weights needs 5 values (ship,prod,planet,safety,fleet_speed), got {_sigw}")
    if args.reward_decomp:
        _wsum = args.win_weight + sum(_sigw)
        _log(f"reward weights: win={args.win_weight} + signals={_sigw} "
             f"(Σ={_wsum:.2f}; not constrained to 1)")
    policy = PPOActorCritic(
        entity_model, sigma=args.sigma, critic_model=critic_model,
        allow_debug_glob_critic=(critic_model is None and not args.reward_decomp),
        reward_decomp=args.reward_decomp, win_weight=args.win_weight,
    ).to(args.device)
    if args.init_policy is not None:
        _ist = torch.load(args.init_policy, map_location=args.device, weights_only=False)
        _isd = _ist.get("policy", _ist)
        _miss, _unexp = policy.load_state_dict(_isd, strict=False)
        _log(f"init-policy: resumed from {args.init_policy} "
             f"(was iter={_ist.get('iter')} sigma={_ist.get('sigma')}); "
             f"loaded {len(_isd)} tensors, missing={len(_miss)} unexpected={len(_unexp)}")
        if len(_miss) > 20 or len(_unexp) > 20:
            _log(f"WARNING: large state_dict mismatch on --init-policy "
                 f"(missing={len(_miss)} unexpected={len(_unexp)}) — architecture may differ")
    if args.reinit_frac_head:
        # Fresh alloc head: re-init the pair_frac_head MLP and ZERO its final
        # layer, so the v3 extras softmax starts uniform over [fired..., self]
        # (a neutral start — not v1's miscalibrated random HOLD). Applied
        # AFTER --init-policy so a resume doesn't get silently wiped... unless
        # the operator explicitly passes both, which re-freshens the head.
        import torch.nn as _nn
        _fh = policy.entity_model.pair_head.pair_frac_head
        for _mod in _fh.modules():
            if isinstance(_mod, _nn.Linear):
                _nn.init.kaiming_uniform_(_mod.weight, a=5 ** 0.5)
                if _mod.bias is not None:
                    _nn.init.zeros_(_mod.bias)
        _last = [m for m in _fh.modules() if isinstance(m, _nn.Linear)][-1]
        _nn.init.zeros_(_last.weight)
        if _last.bias is not None:
            _nn.init.zeros_(_last.bias)
        _n_fresh = sum(p.numel() for p in _fh.parameters())
        _log(f"reinit-frac-head: pair_frac_head re-initialized fresh "
             f"({_n_fresh:,} params; final layer zeroed -> uniform alloc softmax)")
    breakdown = _apply_unfreeze(policy, args.unfreeze)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    _log(f"unfreeze={args.unfreeze}: trainable={n_train:,} groups={breakdown}")
    train_policy = _PPOWithL0(policy, planet_enc, fleet_enc, comet_enc).to(args.device)

    # ---- FIXED opponent (agent A) for agent-B training (else self-play) ----
    fixed_opponent = None
    if args.opponent_ckpt is not None:
        import copy as _copy
        opp_entity = _copy.deepcopy(entity_model)        # separate frozen instance
        fixed_opponent = PPOActorCritic(
            opp_entity, sigma=args.sigma, critic_model=None,
            allow_debug_glob_critic=False, reward_decomp=args.reward_decomp,
            win_weight=args.win_weight,
        ).to(args.device)
        _ost = torch.load(args.opponent_ckpt, map_location=args.device, weights_only=False)
        if "policy" in _ost:
            _osd = _ost["policy"]
        elif "model" in _ost:
            # Supervised/runner-format ckpt (e.g. a joint_best.pt baseline):
            # wrap the EntityPretrainModel weights under the PPOActorCritic
            # prefix so they actually overlay the deepcopy — a bare
            # load_state_dict(strict=False) on the raw dict would silently
            # load ZERO tensors and leave the opponent == the --ckpt base.
            _osd = {f"entity_model.{k}": v for k, v in _ost["model"].items()}
        else:
            _osd = _ost
        _omiss, _ounexp = fixed_opponent.load_state_dict(_osd, strict=False)
        fixed_opponent.eval()
        for _p in fixed_opponent.parameters():
            _p.requires_grad_(False)
        _log(f"FIXED opponent (agent A) from {args.opponent_ckpt} "
             f"(was iter={_ost.get('iter')}); loaded {len(_osd)} tensors, "
             f"missing={len(_omiss)} unexpected={len(_ounexp)}")
        if args.rollout == "pool":
            _log("--opponent-ckpt set with --rollout pool → forcing batched "
                 "(pool is self-play-only; infserver + batched support a fixed opponent)")
            args.rollout = "batched"

    cfg_ppo = PPOConfig(
        clip=args.clip, target_kl=args.target_kl, epochs=args.epochs,
        minibatch_size=args.minibatch_size, lr_heads=args.lr_heads, lr_trunk=args.lr_trunk,
        lr_perception=args.lr_perception,
        lr_conc=args.lr_conc,
        value_coef=args.value_coef, ent_coef=args.ent_coef, bc_coef=0.0,
        bc_target_weight=1.0,
        select_logit_bias=args.select_logit_bias,
    )

    init_ckpt, init_legacy = _save_policy_checkpoint(
        policy,
        out_dir=args.out_dir,
        policy_version=0,
        cfg=cfg,
        args=args,
        metrics={"source_ckpt": str(args.ckpt)},
    )
    _log(f"saved initial policy {init_ckpt.relative_to(args.out_dir)} "
         f"+ {init_legacy.name}")
    _gcs_cp(init_ckpt, args.gcs_out.rstrip("/") + "/checkpoints" if args.gcs_out else "")
    _gcs_cp(init_legacy, args.gcs_out)

    manager = mp.Manager() if args.rollout == "pool" else None

    for K in range(args.iters):
        policy_version = K
        t_iter = time.time()
        _log(f"\n========== iter {K}/{args.iters - 1} ==========")
        _stream_progress({
            "stage": "iter_start",
            "iter": int(K),
            "policy_version": int(policy_version),
            "policy_tag": _vtag(policy_version),
        })

        # Rollout must be inference-only. By default, batched self-play uses
        # one policy instance for learner+opponents, which lets batched_rollout
        # do one GPU forward across all active seats per env step. A separate
        # deep-copy opponent is available only as an explicit debug option.
        policy.eval()
        if fixed_opponent is not None:
            opp = fixed_opponent
            _log("opponent: FIXED agent A (frozen; agent-B training, batched rollout)")
        elif args.separate_opponent_copy:
            opp = _snapshot_opponent(policy, args.device)
            _log("self-play opponent: separate frozen deep-copy")
        else:
            opp = policy
            _log("self-play opponent: same policy instance "
                 "(single model; batched rollout merges all seats)")

        specs = build_specs(args.games, args.seed_base + K * args.games, args.num_players)
        n2 = sum(1 for s in specs if s[1] == 2)
        _mode = "vs FIXED agent A" if fixed_opponent is not None else "self-play"
        _log(f"rollout[{args.rollout}]: {args.games} games (2P={n2} 4P={args.games - n2}) "
             f"{_mode}, T={args.history_window}, device={args.device}")
        _stream_progress({
            "stage": "rollout_start",
            "iter": int(K),
            "policy_version": int(policy_version),
            "policy_tag": _vtag(policy_version),
            "games": int(args.games),
            "games_2p": int(n2),
            "games_4p": int(args.games - n2),
        })
        if args.rollout in {"pool", "infserver"}:
            _log(f"task queues: game_tasks={len(specs)} game_workers={args.procs} "
                 f"post_pack_workers={args.post_procs} "
                 f"rollout_mode={args.rollout}")

        t_roll = time.time()
        live_post_futures = {}
        live_post_executor = None
        if args.rollout == "batched":
            # GPU path: B games in lockstep, batched forward on CUDA, ONE process
            # (no fork → CUDA-safe). Stream per-step active-game progress.
            _last = [t_roll]

            def _on_step(t, active, env_steps, _last=_last):
                now = time.time()
                if now - _last[0] >= args.stream_every_s:
                    na = sum(active)
                    mx = max(env_steps) if env_steps else 0
                    _log(f"[roll] step {t} active={na}/{len(active)} max_env_step={mx} "
                         f"({now - t_roll:.0f}s)")
                    _last[0] = now

            episodes = run_batched_episodes(
                policy=policy, opponent_policy=opp,
                planet_enc=planet_enc, fleet_enc=fleet_enc, comet_enc=comet_enc,
                specs=specs, device=args.device,
                max_planets=args.max_planets, max_fleets=args.max_fleets,
                sigma=args.sigma, select_logit_bias=args.select_logit_bias,
                history_window=args.history_window, on_step=_on_step,
                contract=args.action_contract, select_k_max=args.select_k_max,
                history_offsets=_history_offsets,
            )
        elif args.rollout == "infserver":
            if args.post_procs > 1:
                import concurrent.futures as _fut

                post_ctx = mp.get_context("spawn")
                live_post_executor = _fut.ProcessPoolExecutor(
                    max_workers=args.post_procs,
                    mp_context=post_ctx,
                )

                live_post_paths = {}

                def _submit_post_path(idx, ep_path):
                    if not ep_path:
                        return
                    live_post_paths[int(idx)] = str(ep_path)
                    live_post_futures[idx] = live_post_executor.submit(
                        _shape_and_pack_episode_path_job,
                        str(ep_path),
                        reward_decomp=args.reward_decomp,
                        win_weight=args.win_weight,
                        signal_weights=_sigw,
                        gamma=args.value_gamma,
                        delete_after=args.delete_infserver_spool_after_post,
                        inval_coef=args.inval_coef,
                    )

                on_episode_file_done = _submit_post_path
                _log(f"post-pack queue active: {args.post_procs} workers consume "
                     "completed game files during infserver rollout")
            else:
                on_episode_file_done = None
                live_post_paths = {}

            def _on_inf_progress(st):
                _log(
                    "[roll] infserver "
                    f"t={st.get('wall_s', 0):.1f}s "
                    f"games={st.get('games_done', 0)}/{st.get('games_total', 0)} "
                    f"workers_alive={st.get('workers_alive', 0)} "
                    f"gpu_forwards={st.get('gpu_forwards', 0)} "
                    f"batch(mean/max)={st.get('batch_mean', 0):.1f}/"
                    f"{st.get('batch_max', 0)} "
                    f"fwd_ms={st.get('fwd_ms_mean', 0):.1f}"
                )
                _stream_progress({
                    "iter": int(K),
                    "policy_version": int(policy_version),
                    **dict(st),
                })

            episodes, inf_stats = run_infserver_rollout(
                policy=policy,
                opponent_policy=fixed_opponent,  # None=self-play; agent A for agent-B training
                planet_enc=planet_enc,
                fleet_enc=fleet_enc,
                comet_enc=comet_enc,
                specs=specs,
                n_workers=args.procs,
                device=args.device,
                max_planets=args.max_planets,
                max_fleets=args.max_fleets,
                sigma=args.sigma,
                noop_logit_bias=0.0,
                select_logit_bias=args.select_logit_bias,
                contract=args.action_contract,
                select_k_max=args.select_k_max,
                history_window=args.history_window,
                history_offsets=_history_offsets,
                max_batch=args.infserver_max_batch,
                batch_window_s=args.infserver_batch_window_ms / 1000.0,
                progress_every=args.progress_step_every,
                log_every_s=args.infserver_log_every_s,
                stall_timeout_s=args.infserver_stall_timeout_s,
                spool_dir=(
                    (args.infserver_spool_dir / run_id / _vtag(policy_version))
                    if args.infserver_spool_dir is not None
                    else args.out_dir / "infserver_spool" / _vtag(policy_version)
                ),
                on_progress=_on_inf_progress,
                on_episode_done=None,
                on_episode_file_done=on_episode_file_done,
            )
            bs = inf_stats.get("batch_sizes") or []
            fm = inf_stats.get("fwd_ms") or []
            _log(
                "infserver GPU summary: "
                f"forwards={len(bs)} "
                f"batch_mean={(sum(bs) / max(1, len(bs))):.1f} "
                f"batch_max={(max(bs) if bs else 0)} "
                f"fwd_ms_mean={(sum(fm) / max(1, len(fm))):.1f}"
            )
        elif args.rollout == "shm":
            # shared-memory request plane: same two-policy routing + episode shape
            # as infserver, but workers write the T=10 stack into pre-allocated
            # shm slots (no per-step pickle) → ~1.6x rollout (env.step is the floor).
            def _on_shm_progress(st):
                _log(
                    "[roll] shm "
                    f"t={st.get('wall_s', 0):.1f}s "
                    f"games={st.get('games_done', 0)}/{st.get('games_total', 0)} "
                    f"workers_alive={st.get('workers_alive', 0)} "
                    f"gpu_forwards={st.get('gpu_forwards', 0)} "
                    f"batch(mean/max)={st.get('batch_mean', 0):.1f}/"
                    f"{st.get('batch_max', 0)} "
                    f"fwd_ms={st.get('fwd_ms_mean', 0):.1f}"
                )
                _stream_progress({
                    "iter": int(K),
                    "policy_version": int(policy_version),
                    **dict(st),
                })

            episodes, inf_stats = run_shm_rollout(
                policy=policy,
                opponent_policy=fixed_opponent,  # None=self-play; agent A for agent-B
                planet_enc=planet_enc,
                fleet_enc=fleet_enc,
                comet_enc=comet_enc,
                specs=specs,
                n_workers=args.procs,
                device=args.device,
                max_planets=args.max_planets,
                max_fleets=args.max_fleets,
                sigma=args.sigma,
                n_forwarders=args.shm_forwarders,
                noop_logit_bias=0.0,
                select_logit_bias=args.select_logit_bias,
                history_window=args.history_window,
                max_batch=args.infserver_max_batch,
                spool_dir=(
                    (args.infserver_spool_dir / run_id / _vtag(policy_version))
                    if args.infserver_spool_dir is not None
                    else args.out_dir / "infserver_spool" / _vtag(policy_version)
                ),
                on_progress=_on_shm_progress,
                log_every_s=args.infserver_log_every_s,
                stall_timeout_s=args.infserver_stall_timeout_s,
                progress_every=args.progress_step_every,
            )
            bs = inf_stats.get("batch_sizes") or []
            fm = inf_stats.get("fwd_ms") or []
            _log(
                "shm GPU summary: "
                f"forwards={len(bs)} "
                f"batch_mean={(sum(bs) / max(1, len(bs))):.1f} "
                f"batch_max={(max(bs) if bs else 0)} "
                f"fwd_ms_mean={(sum(fm) / max(1, len(fm))):.1f}"
            )
        else:
            # CPU path: work-stealing fork pool (CUDA-unsafe; --device cpu only).
            progress = manager.dict()
            stop = threading.Event()
            streamer = threading.Thread(
                target=_stream_worker_progress,
                args=(progress, args.procs, stop, args.stream_every_s, t_roll),
                daemon=True)
            streamer.start()
            on_episode_done = None
            if args.post_procs > 1:
                import concurrent.futures as _fut

                # Use spawn for the downstream post workers: the parent has a
                # logging thread and a Manager active while rollout is running.
                # Spawn avoids fork-from-multithreaded-parent hazards; these
                # workers do not need model weights.
                post_ctx = mp.get_context("spawn")
                live_post_executor = _fut.ProcessPoolExecutor(
                    max_workers=args.post_procs,
                    mp_context=post_ctx,
                )

                def _submit_post(idx, buf):
                    if buf is None:
                        return
                    live_post_futures[idx] = live_post_executor.submit(
                        _shape_and_pack_episode_job,
                        buf,
                        reward_decomp=args.reward_decomp,
                        win_weight=args.win_weight,
                        signal_weights=_sigw,
                        gamma=args.value_gamma,
                        inval_coef=args.inval_coef,
                    )

                on_episode_done = _submit_post
                _log(f"post-pack queue active: {args.post_procs} workers consume "
                     "completed games during rollout")
            try:
                episodes = run_pool_rollout(
                    policy, opp, planet_enc, fleet_enc, comet_enc, specs, args.procs,
                    sigma=args.sigma, max_planets=args.max_planets, max_fleets=args.max_fleets,
                    select_logit_bias=args.select_logit_bias,
                    history_window=args.history_window,
                    worker_progress=progress, progress_step_every=args.progress_step_every,
                    on_episode_done=on_episode_done,
                )
            finally:
                stop.set()
                streamer.join(timeout=1.0)
        roll_s = time.time() - t_roll

        ok = [e for e in episodes if e is not None]
        n_fail = len(episodes) - len(ok)
        # per-(num_players) win + step summary
        def _seg(npl):
            seg = [e for e, s in zip(episodes, specs) if e is not None and s[1] == npl]
            if not seg:
                return f"{npl}P: 0 games"
            wins = sum(int(e.winner == e.learner_seat) for e in seg)
            steps = sum(len(e.steps) for e in seg)
            return f"{npl}P: {wins}/{len(seg)} wins, {steps} learner-steps"
        total_steps = sum(len(e.steps) for e in ok)
        noop = sum(1 for e in ok for s in e.steps if s.n_selected_targets == 0)
        emit = sum(s.emitted_launch for e in ok for s in e.steps)
        inval = sum(s.invalid_launch for e in ok for s in e.steps)
        _log(f"rollout done in {roll_s:.1f}s | {len(ok)} ok ({n_fail} failed) | "
             f"{_seg(2)} | {_seg(4)} | total_steps={total_steps} "
             f"noop={noop} emit={emit} inval={inval} "
             f"({total_steps / max(roll_s,1e-9):.0f} steps/s)")
        _stream_progress({
            "stage": "rollout_done",
            "iter": int(K),
            "policy_version": int(policy_version),
            "ok_games": int(len(ok)),
            "failed_games": int(n_fail),
            "total_steps": int(total_steps),
            "rollout_s": round(roll_s, 1),
            "steps_per_s": round(total_steps / max(roll_s, 1e-9), 1),
            "noop": int(noop),
            "emit": int(emit),
            "invalid": int(inval),
        })
        n_games_ok = len(ok)
        wins_2p_ct = sum(int(e.winner == e.learner_seat)
                         for e, s in zip(episodes, specs)
                         if e is not None and s[1] == 2)
        wins_4p_ct = sum(int(e.winner == e.learner_seat)
                         for e, s in zip(episodes, specs)
                         if e is not None and s[1] == 4)
        if not ok:
            _log("no episodes — aborting")
            _stream_progress({"stage": "error", "message": "no episodes"})
            if _STREAM is not None:
                _STREAM.close(phase="error")
            return 1

        # Per-iter WIN/LOSS replay analysis (signals by game phase, launches,
        # seat/seed pockets). Wrapped: diagnostics must never kill a run.
        try:
            _analyze_iter_games(
                episodes, specs, log_fn=_log,
                out_path=args.out_dir / "analysis" / f"games_v{policy_version}.json",
            )
        except Exception as _exc:  # noqa: BLE001
            _log(f"win/loss analysis failed (non-fatal): {_exc!r}")

        # run_pool_rollout's _to_inference() froze the policy (eval +
        # requires_grad_(False) on EVERY param) so the fork pool could share
        # copy-on-write weight pages. Restore the trainable set + train mode on
        # the SAME policy object (train_policy wraps it) before the PPO update,
        # else the update forward builds no grad graph and backward() fails.
        _apply_unfreeze(policy, args.unfreeze)
        train_policy.train()

        # ---- process: shaping + GAE + minibatches ----
        t_proc = time.time()
        live_post_packed = None
        live_post_stats = []
        if live_post_executor is not None:
            _log(f"post-pack drain: waiting for {len(live_post_futures)} completed-game jobs")
            live_post_packed = []
            for idx in sorted(live_post_futures):
                packed, stats = live_post_futures[idx].result()
                if packed is not None:
                    live_post_packed.append(packed)
                    live_post_stats.append(stats)
                ep_path = live_post_paths.get(int(idx))
                if ep_path and not args.delete_infserver_spool_after_post:
                    Path(ep_path).unlink(missing_ok=True)
            live_post_executor.shutdown(wait=True, cancel_futures=False)
            live_post_executor = None

        if args.reward_decomp and live_post_packed is None:
            # design A: OVERWRITE rewards with calculated PBRS Φ shaping + terminal
            # win, set per-step Φ (critic reads −Φ), adjust stored value by −Φ.
            mphi, msh, _ = apply_phi_shaping(
                ok, win_weight=args.win_weight,
                signal_weights=_sigw, gamma=args.value_gamma,
                inval_coef=args.inval_coef, log_fn=_log)
            _log(f"Φ-shaping: mean Φ={mphi:.3f} mean|r_shape|={msh:.4f} "
                 f"(w_win={args.win_weight}, w_sig={_sigw}, γ={args.value_gamma}, "
                 f"inval_coef={args.inval_coef})")
        elif args.reward_decomp and live_post_stats:
            n_steps_phi = sum(s["n_steps"] for s in live_post_stats)
            mphi = (
                sum(s["mean_phi"] * s["n_steps"] for s in live_post_stats)
                / max(1, n_steps_phi)
            )
            msh = (
                sum(s["mean_abs_shape"] * s["n_steps"] for s in live_post_stats)
                / max(1, n_steps_phi)
            )
            _log(f"Φ-shaping[post]: mean Φ={mphi:.3f} mean|r_shape|={msh:.4f} "
                 f"(w_win={args.win_weight}, w_sig={_sigw}, γ={args.value_gamma}, "
                 f"inval_coef={args.inval_coef})")
            _log_window_tables(
                _merge_window_buckets([st.get("buckets") for st in live_post_stats]),
                _log,
            )

        if live_post_packed is not None:
            ep_objs, mbs = _packed_episodes_to_ppo(
                live_post_packed,
                minibatch_size=args.minibatch_size,
                device="cpu",
            )
            proc_mode = f"live post_pack_workers={args.post_procs} pack_device=cpu"
        elif args.post_procs > 1:
            ep_objs, mbs = episodes_to_ppo_parallel(
                ok, minibatch_size=args.minibatch_size,
                device="cpu", num_workers=args.post_procs,
            )
            proc_mode = f"parallel post_procs={args.post_procs} pack_device=cpu"
        else:
            ep_objs, mbs = episodes_to_ppo(ok, minibatch_size=args.minibatch_size,
                                           device="cpu")
            proc_mode = "serial pack_device=cpu"
        _log(f"process[{proc_mode}]: {len(ep_objs)} episodes → {len(mbs)} minibatches "
             f"({time.time()-t_proc:.1f}s)")
        _log_rl_signals(ep_objs, mbs)
        _stream_progress({
            "stage": "process_done",
            "iter": int(K),
            "policy_version": int(policy_version),
            "process_mode": proc_mode,
            "episodes": int(len(ep_objs)),
            "minibatches": int(len(mbs)),
            "process_s": round(time.time() - t_proc, 1),
        })
        if not mbs:
            _log("no minibatches — skipping update"); continue

        rollout_root = None
        rollout_manifest = None
        if not args.no_save_rollout_shards:
            packed_for_archive = live_post_packed
            if packed_for_archive is None and args.post_procs <= 1:
                packed_for_archive = [_pack_episode_for_ppo(ep) for ep in ok if ep.steps]
            rollout_summary = {
                "iter": int(K),
                "policy_version": int(policy_version),
                "policy_tag": _vtag(policy_version),
                "games": int(len(ok)),
                "failed": int(n_fail),
                "total_steps": int(total_steps),
                "noop": int(noop),
                "emit": int(emit),
                "inval": int(inval),
                "rollout_s": round(roll_s, 1),
                "process_s": round(time.time() - t_proc, 1),
                "process_mode": proc_mode,
                "packed_exact_ppo_inputs": packed_for_archive is not None,
                "parent_episode_rewards_shaped": bool(args.reward_decomp and live_post_packed is None),
            }
            rollout_root, rollout_manifest = _save_rollout_artifacts(
                episodes=episodes,
                packed=packed_for_archive,
                specs=specs,
                out_dir=args.out_dir,
                policy_version=policy_version,
                args=args,
                machine_id=machine_id,
                row=rollout_summary,
                save_raw=not args.packed_rollout_artifacts_only,
            )
            rel_rollout = rollout_root.relative_to(args.out_dir)
            _log(f"saved rollout artifacts {rel_rollout} "
                 f"raw={len(rollout_manifest['raw_shards'])} "
                 f"packed={len(rollout_manifest['ppo_packed'])}")
            _stream_progress({
                "stage": "rollout_artifacts_saved",
                "iter": int(K),
                "policy_version": int(policy_version),
                "rollout_dir": str(rollout_root),
                "raw_shards": len(rollout_manifest["raw_shards"]),
                "packed_shards": len(rollout_manifest["ppo_packed"]),
            })
            uploaded_rollouts = _gcs_upload_tree(
                rollout_root,
                args.gcs_out,
                f"rollouts/{_vtag(policy_version)}/{machine_id}",
            )
            if (
                uploaded_rollouts
                and args.delete_local_rollouts_after_upload
                and args.gcs_out
            ):
                shutil.rmtree(rollout_root, ignore_errors=True)
                spool_iter_dir = args.out_dir / "infserver_spool" / _vtag(policy_version)
                shutil.rmtree(spool_iter_dir, ignore_errors=True)
                _log(f"deleted local uploaded rollout artifacts {rel_rollout} "
                     "+ infserver_spool/{_vtag(policy_version)}")

        # ---- update ----
        # Free the rollout-sized objects BEFORE the update allocates: episodes/
        # ok hold full T-stacked StepRecords and live_post_packed holds the
        # per-episode packed copies; the update only needs mbs (cat'd slices)
        # + ep_objs (scalar trajectories). Without this, 2-3 rollout-sized
        # copies coexist during the update and the 125 GB pod cgroup OOMs
        # (observed twice: max_usage == cgroup limit, killed mid-update).
        del episodes, ok
        live_post_packed = None
        live_post_stats = []
        gc.collect()
        t_upd = time.time()
        _stream_progress({
            "stage": "update_start",
            "iter": int(K),
            "from_policy_version": int(policy_version),
            "to_policy_version": int(policy_version + 1),
            "minibatches": int(len(mbs)),
        })
        metrics = ppo_update_local(
            train_policy, ep_objs, mbs,
            bc_minibatch_source=(lambda _n: None), cfg=cfg_ppo)
        ems = metrics.get("epoch_metrics", [])
        last = ems[-1] if ems else {}
        early = any(em.get("early_stopped") for em in ems)
        for em in ems:
            _log(
                f"PPO epoch {int(em.get('epoch', -1))}: "
                f"mb={int(em.get('n_minibatches', 0))} "
                f"kl={em.get('approx_kl', em.get('avg_kl', float('nan'))):.4f} "
                f"ratio={em.get('ratio_mean', float('nan')):.4f} "
                f"policy={em.get('policy_loss', float('nan')):.4f} "
                f"value={em.get('value_loss', float('nan')):.4f} "
                f"entropy={em.get('entropy', float('nan')):.4f} "
                f"clip={em.get('clip_frac', float('nan')):.3f} "
                f"nan_guarded={em.get('nan_guarded', 0):.1f}"
                f"{' early_stop' if em.get('early_stopped') else ''}"
            )
            if "ent/sel_ent" in em:
                # v3 per-stage entropy split: SELECT (targets+self softmax on
                # drawing rows) vs ALLOC ([fired..., HOLD] softmax on acting
                # rows); spikes = modes carrying >10% mass.
                _log(
                    "  ent-split: SEL ent=%.2f ppl=%.1f top1=%.2f "
                    "spikes=%.1f p_self=%.2f | ALLOC ent=%.2f ppl=%.1f "
                    "top1=%.2f spikes=%.1f hold=%.2f"
                    % (em.get("ent/sel_ent", float("nan")),
                       em.get("ent/sel_ppl", float("nan")),
                       em.get("ent/sel_top1", float("nan")),
                       em.get("ent/sel_spikes", float("nan")),
                       em.get("ent/sel_pself", float("nan")),
                       em.get("ent/al_ent", float("nan")),
                       em.get("ent/al_ppl", float("nan")),
                       em.get("ent/al_top1", float("nan")),
                       em.get("ent/al_spikes", float("nan")),
                       em.get("ent/al_hold", float("nan"))))
            if "ent/perplexity" in em:
                # FLAT vs BUMPS: ppl≈coll≈K & norm≈1 -> flat (uniform collapse);
                # coll small (2-4) with large K -> a few bumps (benign hedging).
                prof = " ".join("%.2f" % em.get("ent/p%d" % i, float("nan"))
                                for i in range(8))
                _log(
                    "  ent-shape: ppl=%.1f coll=%.1f K=%.1f norm=%.2f "
                    "top1=%.2f top3=%.2f profile=[%s]"
                    % (em.get("ent/perplexity", float("nan")),
                       em.get("ent/collision", float("nan")),
                       em.get("ent/K", float("nan")),
                       em.get("ent/norm", float("nan")),
                       em.get("ent/top1", float("nan")),
                       em.get("ent/top3", float("nan")),
                       prof))
        _log(f"update: {len(ems)} epochs ({time.time()-t_upd:.1f}s) "
             f"kl={last.get('approx_kl', float('nan')):.4f} "
             f"policy_loss={last.get('policy_loss', float('nan')):.4f} "
             f"value_loss={last.get('value_loss', float('nan')):.4f} "
             f"entropy={last.get('entropy', float('nan')):.4f} "
             f"clip_frac={last.get('clip_frac', float('nan')):.3f}"
             f"{' [early-stop]' if early else ''}")
        _stream_progress({
            "stage": "update_done",
            "iter": int(K),
            "from_policy_version": int(policy_version),
            "to_policy_version": int(policy_version + 1),
            "update_s": round(time.time() - t_upd, 1),
            "epochs": int(len(ems)),
            "kl": last.get("approx_kl", last.get("avg_kl")),
            "policy_loss": last.get("policy_loss"),
            "value_loss": last.get("value_loss"),
            "entropy": last.get("entropy"),
            "clip_frac": last.get("clip_frac"),
            "ent_shape": {k[4:]: last.get(k) for k in last if k.startswith("ent/")},
            "early_stopped": bool(early),
        })

        # durable per-iter JSONL row (survives a killed pod once uploaded)
        row = {
            "iter": K, "unfreeze": args.unfreeze,
            "policy_version_rollout": policy_version,
            "policy_version_next": policy_version + 1,
            "games": n_games_ok, "failed": n_fail,
            "wins_2p": wins_2p_ct,
            "wins_4p": wins_4p_ct,
            "total_steps": total_steps, "noop": noop, "emit": emit, "inval": inval,
            "rollout_s": round(roll_s, 1),
            "kl": last.get("approx_kl"), "policy_loss": last.get("policy_loss"),
            "value_loss": last.get("value_loss"), "entropy": last.get("entropy"),
            "clip_frac": last.get("clip_frac"), "early_stopped": early,
            "ent_shape": {k[4:]: last.get(k) for k in last if k.startswith("ent/")},
            "iter_wall_s": round(time.time() - t_iter, 1),
            "rollout_dir": (str(rollout_root) if rollout_root is not None else None),
            "rollout_manifest": (
                str(rollout_root / "manifest.json") if rollout_root is not None else None
            ),
        }
        with open(jsonl_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        ckpt, legacy_ckpt = _save_policy_checkpoint(
            policy,
            out_dir=args.out_dir,
            policy_version=policy_version + 1,
            cfg=cfg,
            args=args,
            metrics=row,
        )
        _log(f"saved {ckpt.relative_to(args.out_dir)} + {legacy_ckpt.name} "
             f"+ train_log.jsonl | iter {K} wall {time.time()-t_iter:.1f}s")
        _stream_progress({
            "stage": "checkpoint_saved",
            "iter": int(K),
            "policy_version": int(policy_version + 1),
            "checkpoint": str(ckpt),
            "legacy_checkpoint": str(legacy_ckpt),
            "iter_wall_s": round(time.time() - t_iter, 1),
        })
        # upload ckpt + logs so a killed pod retains progress
        _gcs_cp(ckpt, args.gcs_out.rstrip("/") + "/checkpoints" if args.gcs_out else "")
        _gcs_cp(legacy_ckpt, args.gcs_out)
        _gcs_cp(jsonl_path, args.gcs_out)
        _gcs_cp(args.out_dir / "train.log", args.gcs_out)

        # Cross-iter memory fix: free this iteration's rollout/pack tensors before
        # the next iter's rollout+pack. Plain reassignment frees too late (only after
        # the next iter's ~80 GB pack is materialized), so iter N and iter N+1 coexist
        # and OOM the 125 GB pod cgroup at f512. del + gc.collect keeps each iter's
        # peak independent. (Observed: iter 0 fit at 79 GB, iter 1 OOM'd rc=137.)
        del ep_objs, mbs
        gc.collect()

    _log(f"=== RUN DONE in {time.time()-t_start:.1f}s → {args.out_dir} ===")
    _stream_progress({
        "stage": "done",
        "run_id": run_id,
        "out_dir": str(args.out_dir),
        "wall_s": round(time.time() - t_start, 1),
    })
    if _STREAM is not None:
        _STREAM.close(phase="done")
    if _LOG_FH is not None:
        _gcs_cp(args.out_dir / "train.log", args.gcs_out)
        _LOG_FH.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
