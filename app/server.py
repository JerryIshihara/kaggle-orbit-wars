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
