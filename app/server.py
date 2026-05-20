from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import asyncio
import json
import re
import subprocess
import webbrowser

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agents
from utils import (
    REPLAY_ROOT,
    make_run_id,
    record_match,
    run_match,
    save_replay,
    launch_motion_miss_stats,
    time_to_target,
    trace_fleets,
    waste_ratio,
)

ROOT = Path(__file__).resolve().parent
REPLAY_ROOT.mkdir(parents=True, exist_ok=True)

# Training replays live under data/runs/<run_id>/replays/<file>.html. Each
# run's replays are written by the trainer; this server lets the user
# rsync them in from another machine and serves them for the dashboard.
DATA_RUNS = _REPO_ROOT / "data" / "runs"
DATA_RUNS.mkdir(parents=True, exist_ok=True)

# Filename convention: <outcome>_iter<NNN>_ep<MM>_<color>.html where
# outcome ∈ {win, loss, draw} and color ∈ {blue, orange} matches the
# learner's on-screen color (slot 0 = blue, slot 1 = orange — Wong
# palette per orbit_wars.js). Trainer prefers a win-of-the-iter; if
# no win, falls back to draw > highest-final_planets_owned loss.
_REPLAY_RE = re.compile(
    r"(?P<outcome>win|loss|draw)_iter(?P<iter>\d+)"
    r"_ep(?P<ep>\d+)_(?P<color>blue|orange)\.html$"
)

STREAM_DELAY_SEC = 0.003

app = FastAPI(title="Orbit Wars Dashboard")


class PlayRequest(BaseModel):
    agents: list[str]


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "dashboard.html")


@app.get("/dashboard.html")
def dashboard_alias():
    return FileResponse(ROOT / "dashboard.html")


@app.get("/api/agents")
def list_agents_endpoint():
    return {
        "agents": [
            {"id": s.id, "description": s.description}
            for s in agents.list_agent_specs()
        ]
    }


# ---------- Target-score side-by-side viewer ----------
# Picks the newest target_rank_best.pt under data/runs/target_rank/. The
# server pre-loads the stack once at import time so per-request scoring
# is just the forward pass; ~30s per replay on CPU.
_TARGET_RANK_RUNS_DIR = _REPO_ROOT / "data" / "runs" / "target_rank"
_TARGET_RANK_STACK = None
_TARGET_RANK_CFG: dict = {}
_TARGET_RANK_CKPT_PATH: Path | None = None


def _default_target_rank_ckpt() -> Path | None:
    """Newest target_rank_best.pt by directory mtime; ``None`` if absent."""
    if not _TARGET_RANK_RUNS_DIR.is_dir():
        return None
    candidates = [
        d for d in _TARGET_RANK_RUNS_DIR.iterdir()
        if d.is_dir() and (d / "target_rank_best.pt").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0] / "target_rank_best.pt"


def _ensure_target_rank_stack():
    """Lazy-load and cache the TargetRanker stack. Returns ``(stack, cfg)``
    or raises ``HTTPException(503)`` when no ckpt is available."""
    global _TARGET_RANK_STACK, _TARGET_RANK_CFG, _TARGET_RANK_CKPT_PATH
    if _TARGET_RANK_STACK is not None:
        return _TARGET_RANK_STACK, _TARGET_RANK_CFG
    ckpt = _default_target_rank_ckpt()
    if ckpt is None:
        raise HTTPException(
            status_code=503,
            detail=f"no target_rank_best.pt under {_TARGET_RANK_RUNS_DIR}/*/. "
                   "Train a target ranker before requesting target scores.",
        )
    from agents.archive.transformer_v1.inference import load_target_ranker_stack
    stack, cfg = load_target_ranker_stack(ckpt, device="cpu")
    _TARGET_RANK_STACK = stack
    _TARGET_RANK_CFG = cfg
    _TARGET_RANK_CKPT_PATH = ckpt
    print(f"[server] target ranker loaded from {ckpt} "
          f"(d_model={cfg.get('d_model')} d_rank={cfg.get('d_rank')})",
          file=sys.stderr)
    return stack, cfg


def _score_turns_stream(
    steps: list,
    stack,
    cfg: dict,
    slot: int,
    num_players: int,
):
    """Yield NDJSON events for one scoring run.

    Shared by ``/api/target_scores`` (collected into a final list) and
    ``/api/target_scores/stream`` (streamed straight to the client so
    the dashboard can show progress).
    """
    from agents.archive.transformer_v1.inference.target_ranker_scorer import (
        _ensure_label_tensors, _stack_history,
    )
    from agents.archive.transformer_v1.featurizer import FleetTracker
    from agents.archive.transformer_v1.featurizer.inference import featurize_observation
    import torch
    from collections import deque

    n_history = int(cfg.get("n_history", 3))
    max_planets = int(cfg.get("max_planets", 64))
    max_fleets = int(cfg.get("max_fleets", 1024))
    tracker = FleetTracker()
    history: deque = deque(maxlen=n_history)
    device_t = torch.device("cpu")
    total = len(steps)
    out_steps: list[dict] = []

    # Edge-emission knobs. We keep the per-turn JSON small by emitting
    # only the most-attended source→target pairs.
    TOP_TARGETS = 5             # how many targets get edges drawn
    TOP_SOURCES_PER_TARGET = 3  # top-k sources per target
    EDGE_MIN_WEIGHT = 0.05      # drop edges weaker than this — pure noise

    for t, step in enumerate(steps):
        if not step or len(step) <= slot:
            yield {"type": "progress", "current": t + 1, "total": total}
            continue
        seat = step[slot]
        obs = seat.get("observation") if isinstance(seat, dict) else None
        if obs is None:
            yield {"type": "progress", "current": t + 1, "total": total}
            continue
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=slot,
            tracker=tracker,
            num_players=num_players,
            max_planets=max_planets,
            max_fleets=max_fleets,
            device="cpu",
        )
        _ensure_label_tensors(batch, max_planets)
        history.append(batch)
        stacked = _stack_history(history, n_history, device_t)
        with torch.no_grad():
            # Capture Stage A target→source attention weights so the
            # visualizer can draw edges. Negligible extra cost (just a
            # tensor write inside MultiheadAttention).
            target_logits, tgt_valid, src_valid, t2s_attn = stack(
                stacked, return_attn=True,
            )
        logits = target_logits[0]
        valid = tgt_valid[0].bool()
        sv = src_valid[0].bool()
        attn = t2s_attn[0]                                # (P_target, P_source)
        masked = logits.clone()
        masked[~valid] = float("-inf")
        probs = torch.softmax(masked, dim=-1) if valid.any() else torch.zeros_like(logits)

        planets_out = []
        idx_to_pid: dict[int, int] = {}
        for p in (obs.get("planets") or []):
            pid = int(p[0])
            idx = pid_to_idx.get(pid)
            if idx is None or idx >= max_planets:
                continue
            idx_to_pid[idx] = pid
            planets_out.append({
                "id": pid,
                "x": float(p[2]),
                "y": float(p[3]),
                "owner": int(p[1]),
                "ships": int(p[5]),
                "logit": float(logits[idx].item()),
                "prob": float(probs[idx].item()),
                "target_valid": bool(valid[idx].item()),
            })

        # Edge extraction: pick the top-N targets by probability, then
        # for each pull the top-K source attention weights. Skip pairs
        # below EDGE_MIN_WEIGHT — those are softmax noise that would
        # just clutter the canvas. Both ends of each edge must be real
        # planets the frontend can render.
        edges = []
        if valid.any() and sv.any():
            # Top-K target indices, ordered by prob descending.
            top_tgt_vals, top_tgt_idx = probs.topk(
                min(TOP_TARGETS, probs.shape[0]),
            )
            for tgt_rank, tgt_i in enumerate(top_tgt_idx.tolist()):
                if not valid[tgt_i].item():
                    continue
                if tgt_i not in idx_to_pid:
                    continue
                # Source attention row for this target. We already know
                # padded-source positions are zero (key_padding_mask) and
                # the diagonal is zero (attn_mask=eye), so we can take
                # topk directly without further masking.
                src_row = attn[tgt_i]
                k = min(TOP_SOURCES_PER_TARGET, src_row.shape[0])
                top_src_vals, top_src_idx = src_row.topk(k)
                for w, src_i in zip(top_src_vals.tolist(), top_src_idx.tolist()):
                    if w < EDGE_MIN_WEIGHT:
                        continue
                    if not sv[src_i].item():
                        continue
                    if src_i not in idx_to_pid:
                        continue
                    edges.append({
                        "src": idx_to_pid[src_i],
                        "tgt": idx_to_pid[tgt_i],
                        "weight": float(w),
                        "tgt_prob": float(probs[tgt_i].item()),
                    })

        out_steps.append({"turn": t, "planets": planets_out, "edges": edges})
        # Progress every turn keeps the UI responsive. The dashboard
        # reads NDJSON line-by-line and updates a determinate progress
        # bar; cheap to emit, no buffering needed at this granularity.
        yield {"type": "progress", "current": t + 1, "total": total}

    yield {
        "type": "done",
        "steps": out_steps,
        "n_steps": len(out_steps),
    }


@app.get("/api/target_scores/stream")
def target_scores_stream_endpoint(
    run_id: str,
    slot: int = 0,
    mode: str = "play",
):
    """NDJSON-streaming variant of :func:`target_scores_endpoint`.

    Emits one JSON object per line:

        {"type": "init", "config": {...}, "n_total": <int>}
        {"type": "progress", "current": <int>, "total": <int>}
        ... (one progress event per turn)
        {"type": "done", "steps": [...], "n_steps": <int>}

    The frontend reads with ``fetch + ReadableStream`` and updates a
    determinate progress bar so the user knows how far the 10-30s
    scoring run has come.
    """
    sidecar = REPLAY_ROOT / mode / run_id / "game_01.steps.json"
    if not sidecar.exists():
        raise HTTPException(
            status_code=404,
            detail=f"no steps sidecar at {sidecar} — re-run the match.",
        )
    try:
        steps = json.loads(sidecar.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"bad sidecar JSON: {e}")

    stack, cfg = _ensure_target_rank_stack()
    num_players = max(2, min(4, len(steps[0]) if steps else 4))

    def generate():
        header = {
            "type": "init",
            "run_id": run_id,
            "slot": slot,
            "ckpt": str(_TARGET_RANK_CKPT_PATH) if _TARGET_RANK_CKPT_PATH else None,
            "config": {
                "d_model": cfg.get("d_model"),
                "d_rank": cfg.get("d_rank"),
                "max_planets": cfg.get("max_planets"),
                "n_history": cfg.get("n_history"),
                "player": cfg.get("player"),
            },
            "num_players": num_players,
            "n_total": len(steps),
        }
        yield json.dumps(header) + "\n"
        for ev in _score_turns_stream(steps, stack, cfg, slot, num_players):
            yield json.dumps(ev) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/api/target_scores")
def target_scores_endpoint(
    run_id: str,
    slot: int = 0,
    mode: str = "play",
):
    """Compute per-turn TargetRanker logits for the given match.

    Looks for ``{REPLAY_ROOT}/{mode}/{run_id}/game_01.steps.json``
    (written by :func:`save_replay`). Returns a JSON document the
    target-score viewer renders:

        {
          "run_id": ..., "slot": ..., "ckpt": ...,
          "config": { d_model, d_rank, max_planets, n_history, ... },
          "steps": [ { "turn": ..., "planets": [...] }, ... ],
        }
    """
    sidecar = REPLAY_ROOT / mode / run_id / "game_01.steps.json"
    if not sidecar.exists():
        raise HTTPException(
            status_code=404,
            detail=f"no steps sidecar at {sidecar} — re-run the match with "
                   "the latest server (older save_replay didn't write it).",
        )
    try:
        steps = json.loads(sidecar.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"bad sidecar JSON: {e}")

    stack, cfg = _ensure_target_rank_stack()
    # Reuse the cached stack; scorer logic lives in the inference module
    # but the cached stack avoids reload-per-request.
    from agents.archive.transformer_v1.inference.target_ranker_scorer import (
        _ensure_label_tensors, _stack_history,
    )
    from agents.archive.transformer_v1.featurizer import FleetTracker
    from agents.archive.transformer_v1.featurizer.inference import featurize_observation
    import torch
    from collections import deque

    n_history = int(cfg.get("n_history", 3))
    max_planets = int(cfg.get("max_planets", 64))
    max_fleets = int(cfg.get("max_fleets", 1024))
    num_players = max(2, min(4, len(steps[0]) if steps else 4))

    tracker = FleetTracker()
    history: deque = deque(maxlen=n_history)
    out_steps: list[dict] = []
    device_t = torch.device("cpu")

    for t, step in enumerate(steps):
        if not step or len(step) <= slot:
            continue
        seat = step[slot]
        obs = seat.get("observation") if isinstance(seat, dict) else None
        if obs is None:
            continue
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=slot,
            tracker=tracker,
            num_players=num_players,
            max_planets=max_planets,
            max_fleets=max_fleets,
            device="cpu",
        )
        _ensure_label_tensors(batch, max_planets)
        history.append(batch)
        stacked = _stack_history(history, n_history, device_t)
        with torch.no_grad():
            target_logits, tgt_valid = stack(stacked)
        logits = target_logits[0]
        valid = tgt_valid[0].bool()
        masked = logits.clone()
        masked[~valid] = float("-inf")
        probs = torch.softmax(masked, dim=-1) if valid.any() else torch.zeros_like(logits)

        planets_out = []
        for p in (obs.get("planets") or []):
            pid = int(p[0])
            idx = pid_to_idx.get(pid)
            if idx is None or idx >= max_planets:
                continue
            planets_out.append({
                "id": pid,
                "x": float(p[2]),
                "y": float(p[3]),
                "owner": int(p[1]),
                "ships": int(p[5]),
                "logit": float(logits[idx].item()),
                "prob": float(probs[idx].item()),
                "target_valid": bool(valid[idx].item()),
            })
        out_steps.append({"turn": t, "planets": planets_out})

    return {
        "run_id": run_id,
        "slot": slot,
        "ckpt": str(_TARGET_RANK_CKPT_PATH) if _TARGET_RANK_CKPT_PATH else None,
        "config": {
            "d_model": cfg.get("d_model"),
            "d_rank": cfg.get("d_rank"),
            "max_planets": cfg.get("max_planets"),
            "n_history": cfg.get("n_history"),
            "player": cfg.get("player"),
        },
        "n_steps": len(out_steps),
        "steps": out_steps,
    }


@app.get("/target_view.html")
def target_view_html():
    return FileResponse(ROOT / "target_view.html")


@app.post("/api/play")
async def play(req: PlayRequest):
    if len(req.agents) not in (2, 4):
        raise HTTPException(status_code=400, detail="need 2 or 4 agents")
    try:
        [agents.Agent(id=a) for a in req.agents]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def generate():
        # run_match is CPU-bound; push to a thread so the loop stays responsive.
        result = await asyncio.to_thread(run_match, req.agents)

        run_id = make_run_id("play", req.agents)
        run_dir = REPLAY_ROOT / "play" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        save_replay(result.env, run_dir / "game_01.html")
        record_match(
            run_dir, req.agents, result.env, result.scores, result.rewards, result.winner
        )

        print(f"[server] match {run_id}: winner=player{result.winner}", file=sys.stderr)

        yield json.dumps({"type": "init", "agents": req.agents}) + "\n"

        for t, s in enumerate(result.scores):
            yield json.dumps({"type": "step", "step": t, "scores": s}) + "\n"
            if STREAM_DELAY_SEC > 0:
                await asyncio.sleep(STREAM_DELAY_SEC)

        # Per-player fleet diagnostics — fed to the waste + tto charts.
        ws = waste_ratio(result.env)
        ts = time_to_target(result.env)
        launch_motion_stats = launch_motion_miss_stats(result.env)
        # Compact per-fleet records (owner, travel_time, outcome) for histograms.
        records = [
            {"owner": r.owner, "tt": r.travel_time,
             "outcome": r.outcome, "ships": r.initial_ships}
            for r in trace_fleets(result.env)
        ]
        yield json.dumps(
            {
                "type": "done",
                "run_id": run_id,
                "replay_url": f"/replays/play/{run_id}/game_01.html",
                "agents": req.agents,
                "rewards": result.rewards,
                "winner": result.winner,
                "waste_stats": {str(k): v for k, v in ws.items()},
                "tto_stats": {str(k): v for k, v in ts.items()},
                "fleet_records": records,
                "launch_motion_stats": {
                    str(k): v for k, v in launch_motion_stats.items()
                },
            }
        ) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


app.mount("/replays", StaticFiles(directory=REPLAY_ROOT), name="replays")
app.mount(
    "/training_replays",
    StaticFiles(directory=DATA_RUNS),
    name="training_replays",
)


# ---------- Training replay sync + listing ----------
class TrainingSyncRequest(BaseModel):
    """Body for POST /api/training/sync.

    ``remote`` is an rsync-style URL pointing at the *parent* of the
    per-run directories on the source machine, e.g.
    ``user@host:/path/to/repo/data/runs/``. We mirror only the
    ``<run>/replays/`` subtrees down to local ``data/runs/``.
    """

    remote: str


def _scan_runs() -> list[dict]:
    """Return a list of `{run_id, replays: [{name, url, iter, ep, slot}]}`
    for every directory under ``data/runs`` that has a ``replays/``
    subfolder containing matching ``win_iter*.html`` files.
    """
    runs: list[dict] = []
    if not DATA_RUNS.exists():
        return runs
    for run_dir in sorted(DATA_RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        replays_dir = run_dir / "replays"
        if not replays_dir.is_dir():
            continue
        replays: list[dict] = []
        for path in sorted(replays_dir.glob("*.html")):
            m = _REPLAY_RE.search(path.name)
            iter_n = int(m.group("iter")) if m else -1
            ep_n = int(m.group("ep")) if m else -1
            color = m.group("color") if m else "unknown"
            outcome = m.group("outcome") if m else "unknown"
            replays.append({
                "name": path.name,
                "url": f"/training_replays/{run_dir.name}/replays/{path.name}",
                "iter": iter_n,
                "ep": ep_n,
                "color": color,
                "outcome": outcome,
            })
        if replays:
            replays.sort(key=lambda r: (r["iter"], r["ep"], r["color"]))
            runs.append({"run_id": run_dir.name, "replays": replays})
    return runs


@app.get("/api/training/replays")
def list_training_replays():
    return {"runs": _scan_runs()}


@app.post("/api/training/sync")
async def sync_training_replays(req: TrainingSyncRequest):
    remote = req.remote.strip()
    if not remote:
        raise HTTPException(status_code=400, detail="remote is empty")
    # Reject obvious shell metacharacters — rsync is invoked as an arg
    # vector (no shell=True) but a malformed value would still confuse
    # the underlying SSH layer. Allow @, :, /, ., -, _, alphanumeric.
    if re.search(r"[\s;&|`$()<>\\\"']", remote):
        raise HTTPException(status_code=400, detail="invalid remote URL")
    # Always treat the remote as a directory whose children are run dirs.
    if not remote.endswith("/"):
        remote = remote + "/"

    cmd = [
        "rsync", "-az", "--prune-empty-dirs",
        "--include=*/", "--include=replays/***", "--exclude=*",
        remote, str(DATA_RUNS) + "/",
    ]

    try:
        proc = await asyncio.to_thread(
            subprocess.run, cmd,
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=504, detail=f"rsync timed out: {e}")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="rsync not installed on this machine",
        )

    ok = proc.returncode == 0
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "cmd": cmd,
        "runs": _scan_runs() if ok else [],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Orbit Wars dashboard server (FastAPI)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}/"
    print(f"dashboard: {url}")
    print(f"agents: {agents.list_agents()}")
    if not args.no_open:
        webbrowser.open(url)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
