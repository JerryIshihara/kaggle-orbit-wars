from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import asyncio
import json
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
)

ROOT = Path(__file__).resolve().parent
REPLAY_ROOT.mkdir(parents=True, exist_ok=True)

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

        yield json.dumps(
            {
                "type": "done",
                "run_id": run_id,
                "replay_url": f"/replays/play/{run_id}/game_01.html",
                "agents": req.agents,
                "rewards": result.rewards,
                "winner": result.winner,
            }
        ) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


app.mount("/replays", StaticFiles(directory=REPLAY_ROOT), name="replays")


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
