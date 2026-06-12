"""Shared-memory inference-server rollout (L40-centric protocol).

Same contract as ``infserver_rollout.run_infserver_rollout`` (CPU env workers +
one parent GPU forwarder → list[EpisodeBuffer]), but the per-step inference path
uses **pre-allocated shared-memory tensor slots** instead of ``mp.Queue`` request
passing. Workers write their featurized T-stack directly into their slot and flip
a ready flag; the parent reads ready slots by *indexing* (no pickle / no
``_store_from_ipc`` deserialize) and runs one batched forward.

Why: profiling showed the queue path bottlenecks on a single GIL-bound forwarder
doing serial per-request deserialize of fat T=10 stacks — the GPU sat at 19% on an
L40. Shared memory removes that wall so the fast GPU is actually usable.

Only the inference hot-path changes. Episode completion is still signalled via a
small ``result_q`` (carries episode-file paths, not tensors). Two-policy routing
(learner→B, opponent→A) is supported via a per-slot ``is_learner`` flag.

NOTE: behaviour-preserving — workers reuse smoke._finalize_step exactly; same
sampling, same StepRecord. Validate with the CPU parity test before any L40 run.
"""
from __future__ import annotations

import os
import queue
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.multiprocessing as mp

from agents.transformer_v2.featurizer.fleet_featurizer import FleetTracker
from agents.transformer_v2.featurizer.inference import featurize_observation
from agents.transformer_v2.pretrain.entity_encoder import build_pair_type_ids
from . import smoke
from .batched_rollout import _batched_forward, _finalize_episode

_RolloutHistory = smoke._RolloutHistory
_frame_from_batch = smoke._frame_from_batch
_TEMPORAL_KEYS = smoke._TEMPORAL_KEYS


def _oget(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


# ----------------------------------------------------------------------------
# Slot layout: one slot per (worker, seat). slot = wid * max_players + seat.
# A worker writes ALL its active seats' requests, then waits for all of them.
# Flags: req_ready[slot]=1 means "obs written, please forward"; act_ready[slot]=1
# means "action written, worker may consume". Parent clears req_ready when it
# starts a forward; sets act_ready when done. Worker clears act_ready after read.
# ----------------------------------------------------------------------------

def _probe_shapes(max_planets: int, max_fleets: int, history_window: int):
    """Featurize one dummy reset to learn the per-field (dtype, frame-shape).

    Returns {key: (dtype, frame_shape_tuple)} where the stored slot tensor is
    (n_slots, T, *frame_shape). Robust to feature-dim changes (no hardcoding)."""
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": 0})
    env.reset(2)
    obs = env.state[0]["observation"]
    batch, _ = featurize_observation(
        obs, learner_slot=0, tracker=FleetTracker(), num_players=2,
        max_planets=max_planets, max_fleets=max_fleets, device="cpu",
    )
    is_comet = batch["planet_features"][..., 0] > 0.5
    pair_type = build_pair_type_ids(batch["planet_features"], batch["planet_mask"])
    frame = _frame_from_batch(batch, is_comet, pair_type)  # single-frame (no batch dim)
    return {k: (frame[k].dtype, tuple(frame[k].shape)) for k in _TEMPORAL_KEYS}


def _alloc_shared(n_slots: int, T: int, shapes: dict, max_planets: int):
    """Allocate the shared-memory buffers (CPU). All share_memory_()."""
    obs = {}
    for k, (dt, fshape) in shapes.items():
        t = torch.zeros((n_slots, T, *fshape), dtype=dt)
        t.share_memory_()
        obs[k] = t
    # action buffers: pair_logits/frac (P,P), value scalar — sized by max_planets
    P = max_planets
    act_pair = torch.zeros((n_slots, P, P), dtype=torch.float32); act_pair.share_memory_()
    act_frac = torch.zeros((n_slots, P, P), dtype=torch.float32); act_frac.share_memory_()
    act_value = torch.zeros((n_slots,), dtype=torch.float32); act_value.share_memory_()
    act_sigma = torch.zeros((n_slots,), dtype=torch.float32); act_sigma.share_memory_()
    # v4 contract: per-source alpha0 head output (zeros for v2/v3 policies)
    act_conc = torch.zeros((n_slots, P), dtype=torch.float32); act_conc.share_memory_()
    # control flags + per-slot meta (game_idx, step, seat, is_learner); -1 empty
    req_ready = torch.zeros((n_slots,), dtype=torch.uint8); req_ready.share_memory_()
    act_ready = torch.zeros((n_slots,), dtype=torch.uint8); act_ready.share_memory_()
    meta = torch.full((n_slots, 4), -1, dtype=torch.int64); meta.share_memory_()
    return {
        "obs": obs, "act_pair": act_pair, "act_frac": act_frac,
        "act_value": act_value, "act_sigma": act_sigma, "act_conc": act_conc,
        "req_ready": req_ready, "act_ready": act_ready, "meta": meta,
    }


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------

def _shm_worker(wid, specs, shm, cfg):
    """One env worker. Owns slots [wid*MP, wid*MP+MP). Plays its games; per env
    step writes each active seat's T-stack into its slot, flips req_ready, spins
    on act_ready, decodes + steps. Episode done → torch.save to spool + result_q."""
    from kaggle_environments import make
    MP = int(cfg["max_players_slots"])
    base = wid * MP
    T = int(cfg["history_window"])
    result_q = cfg["_result_q"]
    spool = Path(cfg["spool_dir"])
    learner_seat_of = {}  # game idx -> learner_seat

    obs_buf = shm["obs"]; req_ready = shm["req_ready"]; act_ready = shm["act_ready"]
    meta = shm["meta"]; act_pair = shm["act_pair"]; act_frac = shm["act_frac"]
    act_value = shm["act_value"]; act_sigma = shm["act_sigma"]
    act_conc = shm["act_conc"]

    for (idx, seed, num_players, learner_seat) in specs:
        try:
            env = make("orbit_wars", configuration={"seed": seed} if seed is not None else {})
            env.reset(num_players)
            buffer = smoke.EpisodeBuffer(seed=seed, learner_seat=learner_seat)
            trackers = [FleetTracker() for _ in range(num_players)]
            histories = [_RolloutHistory(T, offsets=cfg.get("history_offsets"))
                         for _ in range(num_players)]
            while not env.done:
                states = env.state
                step_idx = int(_oget(states[0].observation, "step", 0) or 0)
                obss = [None] * num_players; pid_maps = [None] * num_players
                stores = [None] * num_players; active = []
                for seat in range(num_players):
                    st = states[seat]
                    if _oget(st, "status", "ACTIVE") != "ACTIVE":
                        continue
                    obs = st["observation"] if isinstance(st, dict) else st.observation
                    batch, pid_to_idx = featurize_observation(
                        obs, learner_slot=seat, tracker=trackers[seat],
                        num_players=num_players, max_planets=int(cfg["max_planets"]),
                        max_fleets=int(cfg["max_fleets"]), device="cpu")
                    is_comet = batch["planet_features"][..., 0] > 0.5
                    pair_type = build_pair_type_ids(batch["planet_features"], batch["planet_mask"])
                    histories[seat].push(step_idx, _frame_from_batch(batch, is_comet, pair_type))
                    store = (histories[seat].frames[step_idx] if histories[seat].window <= 1
                             else histories[seat].stack(step_idx))
                    slot = base + seat
                    # write T-stack into shared slot
                    for k in _TEMPORAL_KEYS:
                        obs_buf[k][slot].copy_(store[k])
                    meta[slot, 0] = idx; meta[slot, 1] = step_idx; meta[slot, 2] = seat
                    meta[slot, 3] = 1 if seat == learner_seat else 0
                    act_ready[slot] = 0
                    req_ready[slot] = 1  # publish LAST (after data written)
                    obss[seat] = obs; pid_maps[seat] = pid_to_idx; stores[seat] = store
                    active.append(seat)
                # spin-wait for all active seats' actions
                for seat in active:
                    slot = base + seat
                    while act_ready[slot] == 0:
                        time.sleep(0.0002)
                # decode + step
                moves = [[] for _ in range(num_players)]
                for seat in active:
                    slot = base + seat
                    env_moves, record = smoke._finalize_step(
                        obss[seat], pid_maps[seat],
                        pair_logits=act_pair[slot].clone(), frac_loc=act_frac[slot].clone(),
                        value=float(act_value[slot]), sigma_val=float(act_sigma[slot]),
                        store=stores[seat], learner_slot=seat, num_players=num_players,
                        noop_logit_bias=float(cfg.get("noop_logit_bias", 0.0)),
                        select_logit_bias=float(cfg.get("select_logit_bias", 0.0)),
                        contract=(str(cfg.get("contract", "v2"))
                                  if seat == learner_seat
                                  else str(cfg.get("opponent_contract", "v2"))),
                        k_max=int(cfg.get("select_k_max", 3)),
                        alloc_conc=act_conc[slot].clone())
                    act_ready[slot] = 0
                    moves[seat] = env_moves
                    if seat == learner_seat:
                        buffer.steps.append(record)
                env.step(moves)
            _finalize_episode(
                env, buffer, learner_seat,
                target_cap_k_max=int(cfg.get("target_cap_k_max", 4)),
                target_cap_lambda=float(cfg.get("target_cap_lambda", 0.0)))
            ep_path = spool / f"ep_{idx}.pt"
            torch.save(buffer, ep_path)
            result_q.put((idx, str(ep_path), None))
        except Exception as e:  # noqa: BLE001
            import traceback
            result_q.put((idx, None, f"{e}\n{traceback.format_exc()}"))
    result_q.put(("__worker_done__", int(wid), None))


# ----------------------------------------------------------------------------
# Forwarder process: owns a SHARD of slots (slot % n_fwd == f). Multiple of these
# run concurrently so the inference plane isn't one GIL-bound core. Each holds its
# own (CUDA-IPC-shared) policy/encoders and does gather→forward→scatter on its
# slots. This is the speed lever past the single-forwarder 1.6x.
# ----------------------------------------------------------------------------

def _forwarder_proc(f, n_fwd, policy, opponent_policy, planet_enc, fleet_enc, comet_enc,
                    shm, device, n_slots, done_flag, max_batch):
    policy.eval()
    if opponent_policy is not None:
        opponent_policy.eval()
    planet_enc.eval(); fleet_enc.eval(); comet_enc.eval()
    my_slots = list(range(f, n_slots, n_fwd))
    req_ready = shm["req_ready"]; act_ready = shm["act_ready"]; meta = shm["meta"]
    obs_buf = shm["obs"]; act_pair = shm["act_pair"]; act_frac = shm["act_frac"]
    act_value = shm["act_value"]; act_sigma = shm["act_sigma"]
    act_conc = shm["act_conc"]

    def _fwd(pol, slots):
        stores = [{k: obs_buf[k][s] for k in _TEMPORAL_KEYS} for s in slots]
        with torch.inference_mode():
            out = _batched_forward(pol, planet_enc, fleet_enc, comet_enc, stores, device)
            pl = torch.nan_to_num(out["pair_logits"], nan=-1e9, posinf=1e9, neginf=-1e9).detach().cpu()
            fl = torch.nan_to_num(out["frac_loc"], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu()
            vv = torch.nan_to_num(out["value"], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu()
            sv = float(out["sigma"].detach().cpu().item())
            cc = out.get("alloc_conc")
            cc = (torch.nan_to_num(cc, nan=0.0).detach().cpu()
                  if cc is not None else None)
        for k, s in enumerate(slots):
            act_pair[s].copy_(pl[k]); act_frac[s].copy_(fl[k])
            act_value[s] = float(vv[k]); act_sigma[s] = sv
            if cc is not None:
                act_conc[s].copy_(cc[k])
            else:
                act_conc[s].zero_()

    while not done_flag.value:
        ready = [s for s in my_slots if int(req_ready[s]) == 1]
        if not ready:
            time.sleep(0.0002); continue
        ready = ready[:max_batch]
        if opponent_policy is None:
            _fwd(policy, ready)
        else:
            learner = [s for s in ready if int(meta[s, 3]) == 1]
            opp = [s for s in ready if int(meta[s, 3]) == 0]
            if learner:
                _fwd(policy, learner)
            if opp:
                _fwd(opponent_policy, opp)
        for s in ready:
            req_ready[s] = 0; act_ready[s] = 1


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run_shm_rollout(
    *, policy, opponent_policy=None, planet_enc, fleet_enc, comet_enc,
    specs: list[tuple[int, int, int]], n_workers: int, device: str,
    max_planets: int, max_fleets: int, sigma: float,
    n_forwarders: int = 4,
    noop_logit_bias: float = 0.0, select_logit_bias: float = 0.0,
    contract: str = "v2", opponent_contract: str = "v2", select_k_max: int = 3,
    history_window: int = 10, history_offsets: list[int] | None = None,
    max_batch: int = 256,
    spool_dir: str | Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
    log_every_s: float = 2.0, stall_timeout_s: float = 180.0,
    progress_every: int = 25,
) -> tuple[list, dict]:
    """Shared-memory variant of run_infserver_rollout. Same return shape."""
    if not specs:
        return [], {"batch_sizes": [], "fwd_ms": [], "game_stats": {}}
    if str(device).startswith("cpu") and not os.environ.get("OW_SHM_ALLOW_CPU"):
        # CPU path is allowed for parity testing (set OW_SHM_ALLOW_CPU=1)
        raise ValueError("shm rollout targets GPU; set OW_SHM_ALLOW_CPU=1 for CPU parity test")

    # normalize specs to (idx, seed, num_players, learner_seat)
    norm = [(i, s[0], s[1], s[2]) for i, s in enumerate(specs)]
    max_players = max(s[1] for s in specs)  # s[1] = num_players (seats per game)
    n_workers = max(1, min(int(n_workers), len(specs)))
    n_slots = n_workers * max_players

    shapes = _probe_shapes(max_planets, max_fleets, history_window)
    shm = _alloc_shared(n_slots, history_window, shapes, max_planets)

    spool_root = Path(spool_dir) if spool_dir is not None else Path(
        tempfile.mkdtemp(prefix="ppo_shm_spool_"))
    spool_root.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    # split games across workers round-robin
    wspecs = [[] for _ in range(n_workers)]
    for j, item in enumerate(norm):
        wspecs[j % n_workers].append(item)
    cfg = {
        "max_planets": int(max_planets), "max_fleets": int(max_fleets),
        "history_window": int(history_window), "noop_logit_bias": float(noop_logit_bias),
        "select_logit_bias": float(select_logit_bias), "max_players_slots": int(max_players),
        "contract": str(contract), "opponent_contract": str(opponent_contract),
        "select_k_max": int(select_k_max),
        "history_offsets": (list(history_offsets) if history_offsets else None),
        "spool_dir": str(spool_root), "_result_q": result_q, "progress_every": int(progress_every),
        "target_cap_k_max": 4, "target_cap_lambda": 0.0,
    }

    policy.eval()
    if opponent_policy is not None:
        opponent_policy.eval()
    planet_enc.eval(); fleet_enc.eval(); comet_enc.eval()

    n_fwd = max(1, min(int(n_forwarders), n_slots))
    done_flag = ctx.Value("b", 0)

    # spawn env workers (write requests) + forwarder processes (serve inference)
    workers = []
    for wid in range(n_workers):
        p = ctx.Process(target=_shm_worker, args=(wid, wspecs[wid], shm, cfg))
        p.start(); workers.append(p)
    forwarders = []
    for f in range(n_fwd):
        p = ctx.Process(
            target=_forwarder_proc,
            args=(f, n_fwd, policy, opponent_policy, planet_enc, fleet_enc, comet_enc,
                  shm, device, n_slots, done_flag, max_batch))
        p.start(); forwarders.append(p)

    results: dict[int, Any] = {}
    workers_done: set[int] = set()
    t0 = time.time(); last_log = t0; last_progress = t0
    n_specs = len(norm)
    allp = workers + forwarders

    def publish(stage):
        if on_progress is None:
            return
        on_progress({
            "stage": stage, "workers": n_workers, "forwarders": n_fwd,
            "workers_alive": n_workers - len(workers_done),
            "games_done": len(results), "games_total": n_specs,
            "wall_s": round(time.time() - t0, 1),
        })

    try:
        while len(results) < n_specs or len(workers_done) < n_workers:
            # drain episode-done / errors (small messages); forwarders do the inference
            try:
                while True:
                    idx, payload, err = result_q.get_nowait()
                    if idx == "__worker_done__":
                        workers_done.add(int(payload)); last_progress = time.time(); continue
                    if err is not None:
                        raise RuntimeError(f"shm worker game {idx} failed: {err}")
                    results[int(idx)] = payload; last_progress = time.time()
            except queue.Empty:
                pass

            now = time.time()
            if now - last_log >= log_every_s:
                publish("rollout_shm"); last_log = now
            if (stall_timeout_s and len(results) < n_specs
                    and now - last_progress > stall_timeout_s):
                raise RuntimeError(
                    f"shm rollout stalled: no progress {now - last_progress:.1f}s; "
                    f"games={len(results)}/{n_specs} workers_done={len(workers_done)}")
            time.sleep(0.001)
    finally:
        done_flag.value = 1  # tell forwarders to exit
        for p in allp:
            p.join(timeout=8)
        for p in allp:
            if p.is_alive():
                p.terminate()

    batch_sizes: list = []; fwd_ms: list = []  # per-forward stats now live in forwarders

    # load episodes from spool in spec order
    episodes = []
    for i in range(n_specs):
        path = results.get(i)
        if path:
            episodes.append(torch.load(path, map_location="cpu", weights_only=False))
    stats = {"batch_sizes": batch_sizes, "fwd_ms": fwd_ms,
             "n_forwards": len(batch_sizes), "wall_s": time.time() - t0}
    return episodes, stats
