from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .controllers import BilateralProxyController, RandomWalkController
from .party_social import (
    BG,
    MUTED,
    PEOPLE,
    PROXY_CLAIM_LABEL,
    PROXY_COLOR,
    RANDOM_COLOR,
    TEXT,
    _draw_fly_pov,
    _draw_room,
    _make_agent,
    _should_reveal,
)

SHOWCASE_WIDTH = 1080
SHOWCASE_HEIGHT = 1350
SHOWCASE_DPI = 100
SHOWCASE_ASPECT = SHOWCASE_WIDTH / SHOWCASE_HEIGHT


def _distance_to_source(live) -> float:
    arena = live.env.arena
    return float(np.hypot(live.env.agent.x - arena.source_x, live.env.agent.y - arena.source_y))


def _result_text(live) -> str:
    if live.env.agent.found:
        return "CASE CLOSED"
    if live.done:
        return "STILL SNIFFING"
    return "SEARCHING"


def render_showcase(
    output: str | Path,
    *,
    seed: int = 13013,
    seconds: int = 15,
    fps: int = 30,
) -> Path:
    """Render the phone-first development showcase as a 1080x1350 clip.

    This is a presentation layer over the exact same FlySniffEnv/plume/controller
    state used by the benchmark plumbing. It is intentionally development-only
    and can never label either controller as a qualified MaleCNS result.
    """
    if seconds < 1:
        raise ValueError("seconds must be >= 1")
    if fps < 1:
        raise ValueError("fps must be >= 1")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    agents = [
        _make_agent("BIOLOGY-INSPIRED PROXY", BilateralProxyController, seed, PROXY_COLOR),
        _make_agent("RANDOM CONTROL", RandomWalkController, seed, RANDOM_COLOR),
    ]
    arena = agents[0].env.arena
    sim_steps_per_frame = max(1, round(3.0 * (1.0 / fps) / arena.dt))
    frames = max(1, seconds * fps)

    fig = plt.figure(
        figsize=(SHOWCASE_WIDTH / SHOWCASE_DPI, SHOWCASE_HEIGHT / SHOWCASE_DPI),
        dpi=SHOWCASE_DPI,
        facecolor=BG,
    )
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.52, 1.92, 1.02, 0.34],
        hspace=0.22,
        wspace=0.12,
    )
    title_ax = fig.add_subplot(grid[0, :])
    room_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    pov_ax = fig.add_subplot(grid[2, :])
    footer_ax = fig.add_subplot(grid[3, :])
    fig.subplots_adjust(left=0.035, right=0.965, top=0.985, bottom=0.03)

    def draw(frame: int):
        for live in agents:
            for _ in range(sim_steps_per_frame):
                if not live.done:
                    action = live.controller.act(live.obs)
                    live.obs, live.done = live.env.step(action.turn, action.speed)

        snap = agents[0].env.plume.snapshot()
        reveal = _should_reveal(
            frame=frame,
            frames=frames,
            fps=fps,
            done=any(live.done for live in agents),
        )

        title_ax.clear()
        title_ax.set_facecolor(BG)
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.72,
            "WHO FARTED?",
            ha="center",
            va="center",
            fontsize=38,
            color=TEXT,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.34,
            "same odor plume • same start • source hidden from the fly",
            ha="center",
            va="center",
            fontsize=12.5,
            color=MUTED,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.03,
            PROXY_CLAIM_LABEL,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#FBBF24",
            fontweight="bold",
        )

        for ax, live in zip(room_axes, agents, strict=True):
            _draw_room(ax, live, snap, reveal=reveal or live.done)
            ax.text(
                0.97,
                0.96,
                f"{_distance_to_source(live):.1f} m away",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9.5,
                color=TEXT,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": BG,
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
            )

        _draw_fly_pov(pov_ax, agents[0], snap)

        footer_ax.clear()
        footer_ax.set_facecolor(BG)
        footer_ax.axis("off")
        if reveal:
            culprit = PEOPLE[0][2]
            footer_ax.text(
                0.5,
                0.72,
                f"CULPRIT: {culprit}",
                ha="center",
                va="center",
                fontsize=20,
                color="#D9F99D",
                fontweight="bold",
            )
            footer_ax.text(
                0.5,
                0.22,
                "  •  ".join(f"{live.label}: {_result_text(live)}" for live in agents),
                ha="center",
                va="center",
                fontsize=10.5,
                color=TEXT,
                fontweight="bold",
            )
        else:
            footer_ax.text(
                0.5,
                0.50,
                "The green puffs are visible to you. The controllers only get modeled antenna signals.",
                ha="center",
                va="center",
                fontsize=10.5,
                color=MUTED,
                fontweight="bold",
            )
        return []

    ani = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    try:
        if output.suffix.lower() == ".gif":
            ani.save(output, writer=animation.PillowWriter(fps=fps))
        else:
            ani.save(output, writer=animation.FFMpegWriter(fps=fps, bitrate=6500))
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the 4:5 Who Farted? development showcase")
    parser.add_argument("--output", default="artifacts/showcase/who-farted-4x5.mp4")
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(render_showcase(args.output, seed=args.seed, seconds=args.seconds, fps=args.fps))
