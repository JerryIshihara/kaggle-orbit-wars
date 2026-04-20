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
from run import REPLAY_ROOT, make_run_id
from utils import record_match

ROOT = Path(__file__).resolve().parent
REPLAY_ROOT.mkdir(parents=True, exist_ok=True)

STREAM_DELAY_SEC = 0.003

app = FastAPI(title="Orbit Wars Dashboard")


class PlayRequest(BaseModel):
    agents: list[str]


def compute_scores(env) -> list[list[int]]:
    """Per-step total ship count (planets + fleets) per player."""
    scores: list[list[int]] = []
    for step in env.steps:
        if not step:
            continue
        obs = step[0].observation
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
        n = len(step)
        per_player = [0] * n
        for p in planets:
            owner, ships = p[1], p[5]
            if 0 <= owner < n:
                per_player[owner] += ships
        for f in fleets:
            owner, ships = f[1], f[6]
            if 0 <= owner < n:
                per_player[owner] += ships
        scores.append(per_player)
    return scores


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "dashboard.html")


@app.get("/dashboard.html")
def dashboard_alias():
    return FileResponse(ROOT / "dashboard.html")


@app.get("/api/agents")
def list_agents_endpoint():
    return {"agents": agents.list_agents()}


@app.post("/api/play")
async def play(req: PlayRequest):
    if len(req.agents) not in (2, 4):
        raise HTTPException(status_code=400, detail="need 2 or 4 agents")
    try:
        players = [agents.Agent(id=a) for a in req.agents]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def generate():
        from kaggle_environments import make

        env = make("orbit_wars", debug=False)
        # env.run is synchronous and CPU-bound; run in a thread to keep the loop responsive.
        await asyncio.to_thread(env.run, [p.fn for p in players])

        scores = compute_scores(env)
        final = env.steps[-1]
        rewards = [s.reward for s in final]
        ranked = [r if r is not None else float("-inf") for r in rewards]
        winner = max(range(len(ranked)), key=lambda i: ranked[i])

        run_id = make_run_id("play", req.agents)
        run_dir = REPLAY_ROOT / "play" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        replay_path = run_dir / "game_01.html"
        replay_path.write_text(env.render(mode="html"))
        record_match(run_dir, req.agents, env, scores, rewards, winner)

        print(f"[server] match {run_id}: winner=player{winner}", file=sys.stderr)

        yield json.dumps({"type": "init", "agents": req.agents}) + "\n"

        for t, s in enumerate(scores):
            yield json.dumps({"type": "step", "step": t, "scores": s}) + "\n"
            if STREAM_DELAY_SEC > 0:
                await asyncio.sleep(STREAM_DELAY_SEC)

        yield json.dumps(
            {
                "type": "done",
                "run_id": run_id,
                "replay_url": f"/replays/play/{run_id}/game_01.html",
                "agents": req.agents,
                "rewards": rewards,
                "winner": winner,
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
