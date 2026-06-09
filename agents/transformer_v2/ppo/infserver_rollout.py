"""CPU env workers + one GPU batched-forward server in the parent process.

This is the rollout shape for a single-GPU PPO trial:

* CPU worker processes own only orbit_wars envs, FleetTracker state, and T-window
  featurization.
* When a worker reaches a step, it submits every active seat's current T-window to
  the parent through a queue.
* The parent owns the single CUDA-resident PPOActorCritic + L0 encoders, batches
  ready requests from all workers, runs one forward, and returns action logits.

Workers never load or hold model parameters. CUDA is initialized only in the
parent process, so there is exactly one GPU model instance.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch

from agents.transformer_v2.featurizer.fleet_featurizer import FleetTracker
from agents.transformer_v2.featurizer.inference import featurize_observation
from agents.transformer_v2.pretrain.entity_encoder import build_pair_type_ids

from . import smoke
from .batched_rollout import _batched_forward, _finalize_episode


try:
    import torch.multiprocessing as _tmp

    _tmp.set_sharing_strategy("file_system")
except Exception:  # noqa: BLE001
    pass


_RolloutHistory = smoke._RolloutHistory
_frame_from_batch = smoke._frame_from_batch
_TEMPORAL_KEYS = smoke._TEMPORAL_KEYS


def _store_to_ipc(store: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Convert a model-input store to plain pickled arrays for worker->parent IPC.

    Sending torch tensors through multiprocessing queues uses torch's shared-memory
    storage reducer. With T=10 and large fleet caps that can fill /dev/shm during
    long rollouts. NumPy arrays travel through the queue as normal pickle payloads;
    the CUDA parent rebuilds torch tensors immediately before batching.
    """
    return {
        k: v.detach().cpu().contiguous().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in store.items()
    }


def _store_from_ipc(store: dict[str, Any]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in store.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
        elif hasattr(v, "__array_interface__"):
            out[k] = torch.from_numpy(v)
        else:
            out[k] = torch.as_tensor(v)
    return out


def _oget(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def _worker_play_one(
    *,
    wid: int,
    idx: int,
    seed: int,
    num_players: int,
    learner_seat: int,
    req_q,
    res_q,
    cfg: dict,
    make,
) -> tuple[int, smoke.EpisodeBuffer | None, dict]:
    t0 = time.perf_counter()
    env = make("orbit_wars", configuration={"seed": seed} if seed is not None else {})
    env.reset(num_players)

    buffer = smoke.EpisodeBuffer(seed=seed, learner_seat=learner_seat)
    trackers = [FleetTracker() for _ in range(num_players)]
    histories = [_RolloutHistory(int(cfg["history_window"])) for _ in range(num_players)]

    t_feat = 0.0
    t_wait = 0.0
    t_decode = 0.0
    t_env = 0.0
    n_steps = 0

    while not env.done:
        states = env.state
        step_idx = int(_oget(states[0].observation, "step", 0) or 0)
        obss = [None] * num_players
        pid_maps = [None] * num_players
        stores = [None] * num_players

        a = time.perf_counter()
        for seat in range(num_players):
            st = states[seat]
            if _oget(st, "status", "ACTIVE") != "ACTIVE":
                continue
            obs = st["observation"] if isinstance(st, dict) else st.observation
            batch, pid_to_idx = featurize_observation(
                obs,
                learner_slot=seat,
                tracker=trackers[seat],
                num_players=num_players,
                max_planets=int(cfg["max_planets"]),
                max_fleets=int(cfg["max_fleets"]),
                device="cpu",
            )
            is_comet = batch["planet_features"][..., 0] > 0.5
            pair_type = build_pair_type_ids(batch["planet_features"], batch["planet_mask"])
            histories[seat].push(step_idx, _frame_from_batch(batch, is_comet, pair_type))
            store = (
                {k: histories[seat].frames[step_idx][k] for k in _TEMPORAL_KEYS}
                if histories[seat].window <= 1
                else histories[seat].stack(step_idx)
            )
            req_id = (idx, step_idx, seat)
            # 5th field: is_learner — routes to the learner policy (B) vs the
            # fixed opponent policy (A) in the two-policy forwarder; ignored in
            # self-play (single policy serves all seats).
            req_q.put((wid, req_id, seat, _store_to_ipc(store), seat == learner_seat))
            obss[seat] = obs
            pid_maps[seat] = pid_to_idx
            stores[seat] = store
        t_feat += time.perf_counter() - a

        b = time.perf_counter()
        results = {}
        active_seats = [s for s in range(num_players) if stores[s] is not None]
        for _ in active_seats:
            req_id, seat, pair_logits, frac_loc, value, sigma_val = res_q.get()
            results[seat] = (req_id, pair_logits, frac_loc, value, sigma_val)
        t_wait += time.perf_counter() - b

        c = time.perf_counter()
        moves = [[] for _ in range(num_players)]
        for seat in active_seats:
            _req_id, pair_logits, frac_loc, value, sigma_val = results[seat]
            env_moves, record = smoke._finalize_step(
                obss[seat],
                pid_maps[seat],
                pair_logits=pair_logits,
                frac_loc=frac_loc,
                value=float(value),
                sigma_val=float(sigma_val),
                store=stores[seat],
                learner_slot=seat,
                num_players=num_players,
                noop_logit_bias=float(cfg.get("noop_logit_bias", 0.0)),
                select_logit_bias=float(cfg.get("select_logit_bias", 0.0)),
            )
            moves[seat] = env_moves
            if seat == learner_seat:
                buffer.steps.append(record)
        t_decode += time.perf_counter() - c

        d = time.perf_counter()
        env.step(moves)
        t_env += time.perf_counter() - d
        n_steps += 1

        progress_every = int(cfg.get("progress_every", 0))
        if progress_every and n_steps % progress_every == 0:
            req_q.put((
                "__progress__",
                {
                    "wid": int(wid),
                    "game_idx": int(idx),
                    "seed": int(seed),
                    "num_players": int(num_players),
                    "learner_seat": int(learner_seat),
                    "step": int(n_steps),
                    "status": "running",
                },
            ))

    _finalize_episode(
        env,
        buffer,
        learner_seat,
        target_cap_k_max=int(cfg.get("target_cap_k_max", 4)),
        target_cap_lambda=float(cfg.get("target_cap_lambda", 0.0)),
    )
    wall = time.perf_counter() - t0
    stats = {
        "wid": int(wid),
        "game_idx": int(idx),
        "seed": int(seed),
        "num_players": int(num_players),
        "learner_seat": int(learner_seat),
        "n_steps": int(n_steps),
        "wall_s": float(wall),
        "feat_s": float(t_feat),
        "gpu_wait_s": float(t_wait),
        "decode_s": float(t_decode),
        "env_s": float(t_env),
        "winner": int(buffer.winner) if buffer.winner is not None else -1,
        "won": int(buffer.winner == learner_seat) if buffer.winner is not None else 0,
        "status": "done",
    }
    return idx, buffer, stats


def _cpu_worker(wid: int, specs, req_q, res_q, result_q, cfg):
    from kaggle_environments import make

    torch.set_num_threads(1)
    spool_dir = Path(cfg["spool_dir"])
    spool_dir.mkdir(parents=True, exist_ok=True)
    for idx, (seed, num_players, learner_seat) in specs:
        try:
            idx, buffer, stats = _worker_play_one(
                wid=wid,
                idx=idx,
                seed=seed,
                num_players=num_players,
                learner_seat=learner_seat,
                req_q=req_q,
                res_q=res_q,
                cfg=cfg,
                make=make,
            )
            ep_path = spool_dir / f"episode_{idx:06d}.pt"
            torch.save(buffer, ep_path)
            result_q.put((idx, {"episode_path": str(ep_path)}, stats))
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            result_q.put((
                idx,
                {"episode_path": None},
                {
                    "wid": int(wid),
                    "game_idx": int(idx),
                    "seed": int(seed),
                    "num_players": int(num_players),
                    "learner_seat": int(learner_seat),
                    "status": "failed",
                    "error": repr(exc),
                },
            ))
    req_q.put(("__worker_done__", int(wid)))


def _load_episode_payload(payload):
    if isinstance(payload, dict) and payload.get("episode_path"):
        return torch.load(payload["episode_path"], map_location="cpu", weights_only=False)
    return payload


def _drain_results(
    result_q,
    results: dict,
    stats: dict,
    on_episode_done,
    on_episode_file_done=None,
    delete_episode_file_after_load: bool = True,
) -> int:
    n = 0
    while True:
        try:
            idx, payload, st = result_q.get_nowait()
        except queue.Empty:
            break
        buf = None
        ep_path = payload.get("episode_path") if isinstance(payload, dict) else None
        if ep_path is not None or not isinstance(payload, dict):
            buf = _load_episode_payload(payload)
            if (
                ep_path is not None
                and on_episode_file_done is None
                and delete_episode_file_after_load
            ):
                try:
                    Path(ep_path).unlink(missing_ok=True)
                except Exception:
                    pass
        if ep_path is not None and on_episode_file_done is not None:
            on_episode_file_done(int(idx), ep_path)
        results[int(idx)] = buf
        stats[int(idx)] = dict(st or {})
        if on_episode_done is not None and buf is not None:
            on_episode_done(int(idx), buf)
        n += 1
    return n


def run_infserver_rollout(
    *,
    policy,
    opponent_policy=None,
    planet_enc,
    fleet_enc,
    comet_enc,
    specs: list[tuple[int, int, int]],
    n_workers: int,
    device: str,
    max_planets: int,
    max_fleets: int,
    sigma: float,
    noop_logit_bias: float = 0.0,
    select_logit_bias: float = 0.0,
    history_window: int = 10,
    max_batch: int = 256,
    batch_window_s: float = 0.003,
    progress_every: int = 25,
    log_every_s: float = 2.0,
    stall_timeout_s: float = 180.0,
    target_cap_k_max: int = 4,
    target_cap_lambda: float = 0.0,
    spool_dir: str | Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
    on_episode_done: Callable[[int, Any], None] | None = None,
    on_episode_file_done: Callable[[int, str], None] | None = None,
) -> tuple[list, dict]:
    """Run rollout with CPU env workers and one parent-owned GPU forwarder."""
    if not specs:
        return [], {"batch_sizes": [], "fwd_ms": [], "game_stats": {}}
    if str(device).startswith("cpu"):
        raise ValueError("infserver rollout is for GPU forward; pass --device cuda")

    n_workers = max(1, min(int(n_workers), len(specs)))
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue(maxsize=max(4 * n_workers, int(max_batch) * 2))
    res_qs = [ctx.Queue(maxsize=8) for _ in range(n_workers)]
    result_q = ctx.Queue()
    spool_root = Path(spool_dir) if spool_dir is not None else Path(
        tempfile.mkdtemp(prefix="ppo_infserver_spool_")
    )
    spool_root.mkdir(parents=True, exist_ok=True)

    indexed_specs = list(enumerate(specs))
    wspecs = [[] for _ in range(n_workers)]
    for i, item in enumerate(indexed_specs):
        wspecs[i % n_workers].append(item)

    cfg = {
        "max_planets": int(max_planets),
        "max_fleets": int(max_fleets),
        "history_window": int(history_window),
        "noop_logit_bias": float(noop_logit_bias),
        "select_logit_bias": float(select_logit_bias),
        "progress_every": int(progress_every),
        "target_cap_k_max": int(target_cap_k_max),
        "target_cap_lambda": float(target_cap_lambda),
        "spool_dir": str(spool_root),
    }

    policy.eval()
    if opponent_policy is not None:
        opponent_policy.eval()
    planet_enc.eval()
    fleet_enc.eval()
    comet_enc.eval()

    def _fwd_items(pol, items):
        """One batched forward of `pol` over a list of batch request tuples;
        returns CPU (pair_logits, frac_loc, values, sigma_scalar)."""
        stores = [_store_from_ipc(it[3]) for it in items]
        with torch.inference_mode():
            out = _batched_forward(pol, planet_enc, fleet_enc, comet_enc, stores, device)
            pl = torch.nan_to_num(out["pair_logits"], nan=-1e9, posinf=1e9, neginf=-1e9).detach().cpu()
            fl = torch.nan_to_num(out["frac_loc"], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu()
            vv = torch.nan_to_num(out["value"], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu()
            sv = float(out["sigma"].detach().cpu().item())
        return pl, fl, vv, sv

    workers = []
    for wid in range(n_workers):
        p = ctx.Process(
            target=_cpu_worker,
            args=(wid, wspecs[wid], req_q, res_qs[wid], result_q, cfg),
        )
        p.start()
        workers.append(p)

    t0 = time.time()
    last_log = t0
    last_progress = t0
    workers_alive = n_workers
    workers_done: set[int] = set()
    worker_exits: dict[int, int | None] = {}
    results: dict[int, object] = {}
    game_stats: dict[int, dict] = {}
    worker_progress: dict[int, dict] = {}
    batch_sizes: list[int] = []
    fwd_ms: list[float] = []

    def publish(stage: str) -> None:
        if on_progress is None:
            return
        recent_fwd = fwd_ms[-200:]
        recent_batch = batch_sizes[-200:]
        on_progress({
            "stage": stage,
            "workers": int(n_workers),
            "workers_alive": int(workers_alive),
            "games_done": int(len(results)),
            "games_total": int(len(specs)),
            "gpu_forwards": int(len(batch_sizes)),
            "batch_mean": (
                float(sum(recent_batch) / len(recent_batch)) if recent_batch else 0.0
            ),
            "batch_max": int(max(recent_batch)) if recent_batch else 0,
            "fwd_ms_mean": (
                float(sum(recent_fwd) / len(recent_fwd)) if recent_fwd else 0.0
            ),
            "worker_progress": [worker_progress[k] for k in sorted(worker_progress)],
            "worker_exits": dict(worker_exits),
            "wall_s": round(time.time() - t0, 1),
        })

    def mark_progress() -> None:
        nonlocal last_progress
        last_progress = time.time()

    def mark_worker_done(wid: int, exitcode: int | None = 0) -> None:
        nonlocal workers_alive
        wid = int(wid)
        if wid not in workers_done:
            workers_done.add(wid)
            workers_alive = max(0, workers_alive - 1)
        worker_exits[wid] = exitcode

    while len(results) < len(specs) or workers_alive > 0:
        if _drain_results(
            result_q,
            results,
            game_stats,
            on_episode_done,
            on_episode_file_done=on_episode_file_done,
            delete_episode_file_after_load=True,
        ):
            mark_progress()

        for wid, proc in enumerate(workers):
            if wid in workers_done:
                continue
            if proc.exitcode is not None:
                mark_worker_done(wid, proc.exitcode)

        batch = []
        try:
            first = req_q.get(timeout=0.2)
        except queue.Empty:
            first = None

        if first is not None:
            pending = [first]
            deadline = time.perf_counter() + float(batch_window_s)
            while len(batch) < int(max_batch):
                if pending:
                    item = pending.pop()
                else:
                    try:
                        item = req_q.get_nowait()
                    except queue.Empty:
                        if time.perf_counter() >= deadline:
                            break
                        time.sleep(0.0002)
                        continue
                if isinstance(item, tuple) and item and item[0] == "__worker_done__":
                    mark_worker_done(int(item[1]), 0)
                    mark_progress()
                    continue
                if isinstance(item, tuple) and item and item[0] == "__progress__":
                    progress = dict(item[1])
                    worker_progress[int(progress["wid"])] = progress
                    mark_progress()
                    continue
                batch.append(item)

        if batch:
            tf = time.perf_counter()
            # res[batch_index] = (pair_logits_k, frac_loc_k, value_k_float, sigma)
            res: dict[int, tuple] = {}
            if opponent_policy is None:
                pl, fl, vv, sv = _fwd_items(policy, batch)            # self-play: one forward
                for k in range(len(batch)):
                    res[k] = (pl[k], fl[k], float(vv[k].item()), sv)
            else:
                # two-policy: route learner seats → B (policy), others → A.
                learner_idx = [k for k, it in enumerate(batch) if it[4]]
                opp_idx = [k for k, it in enumerate(batch) if not it[4]]
                for pol, idxs in ((policy, learner_idx), (opponent_policy, opp_idx)):
                    if not idxs:
                        continue
                    pl, fl, vv, sv = _fwd_items(pol, [batch[k] for k in idxs])
                    for j, k in enumerate(idxs):
                        res[k] = (pl[j], fl[j], float(vv[j].item()), sv)
            fwd_ms.append((time.perf_counter() - tf) * 1e3)
            batch_sizes.append(len(batch))
            mark_progress()
            for k, it in enumerate(batch):
                pl_k, fl_k, v_k, sv_k = res[k]
                res_qs[int(it[0])].put((it[1], int(it[2]), pl_k, fl_k, v_k, sv_k))

        if (
            stall_timeout_s
            and workers_alive > 0
            and len(results) < len(specs)
            and time.time() - last_progress > float(stall_timeout_s)
        ):
            publish("rollout_infserver_stalled")
            for proc in workers:
                if proc.is_alive():
                    proc.terminate()
            for proc in workers:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
            raise RuntimeError(
                "infserver rollout stalled: "
                f"no request/result/progress for {time.time() - last_progress:.1f}s; "
                f"games_done={len(results)}/{len(specs)} "
                f"workers_alive={workers_alive} "
                f"gpu_forwards={len(batch_sizes)} "
                f"worker_progress={list(worker_progress.values())[:12]}"
            )

        if time.time() - last_log >= float(log_every_s):
            publish("rollout_infserver")
            last_log = time.time()

    _drain_results(
        result_q,
        results,
        game_stats,
        on_episode_done,
        on_episode_file_done=on_episode_file_done,
        delete_episode_file_after_load=True,
    )
    for p in workers:
        p.join(timeout=10)
    episodes = [results.get(i) for i in range(len(specs))]
    stats = {
        "batch_sizes": batch_sizes,
        "fwd_ms": fwd_ms,
        "game_stats": game_stats,
        "workers": n_workers,
        "device": str(device),
    }
    publish("rollout_infserver_done")
    return episodes, stats
