"""Render a sniper-vs-random match as a GIF for the README.

Run from repo root:
    python notebooks/_build_demo_gif.py

Writes docs/demo.gif.
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import run_match  # noqa: E402

PLAYER_COLORS = {0: "#3b6ef0", 1: "#f07b7b", 2: "#7dd89f", 3: "#f0c93b"}
NEUTRAL_COLOR = "#6b7280"
SUN_COLOR = "#f0c93b"
BG = "#0b0d17"
FG = "#e8ecf2"


def render_frame(step_state, step_idx: int, agent_ids: list[str]) -> Image.Image:
    obs = step_state[0].observation
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []

    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.set_aspect("equal")

    ax.add_patch(Circle((50, 50), 10, color=SUN_COLOR, alpha=0.25))
    ax.add_patch(Circle((50, 50), 10, color=SUN_COLOR, fill=False, lw=1.2))

    for p in planets:
        pid, owner, x, y, radius, ships, prod = p
        color = PLAYER_COLORS.get(owner, NEUTRAL_COLOR)
        alpha = 0.45 if owner < 0 else 0.85
        ax.add_patch(Circle((x, y), radius, color=color, alpha=alpha, zorder=2))
        ax.annotate(
            str(ships), (x, y), color="white", ha="center", va="center",
            fontsize=6.5, weight="bold", zorder=3,
        )

    for f in fleets:
        fid, owner, x, y, angle, from_pid, ships = f
        color = PLAYER_COLORS.get(owner, NEUTRAL_COLOR)
        size = max(8, min(40, ships * 0.15))
        dx, dy = math.cos(angle) * 2, math.sin(angle) * 2
        ax.arrow(
            x - dx, y - dy, dx * 2, dy * 2,
            color=color, head_width=1.2, head_length=1.3,
            lw=0.6, zorder=4, length_includes_head=True, alpha=0.9,
        )

    total = sum(s.reward or 0 for s in step_state)
    title = f"step {step_idx:>3}   {agent_ids[0]} vs {agent_ids[1]}"
    ax.set_title(title, color=FG, fontsize=10, pad=6, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#252b3d")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def main() -> int:
    print("running sniper vs random (seed=42)...", flush=True)
    result = run_match(["sniper", "random"], seed=42)
    total = len(result.env.steps)
    print(f"match: {total} steps, winner=player{result.winner}", flush=True)

    target_frames = 60
    stride = max(1, total // target_frames)

    print(f"rendering ~{total // stride} frames (stride={stride})...", flush=True)
    frames = [
        render_frame(result.env.steps[i], i, result.agent_ids)
        for i in range(0, total, stride)
    ]
    frames.extend([frames[-1]] * 8)

    out = Path(__file__).resolve().parent.parent / "docs" / "demo.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({size_kb:.0f} KB, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
