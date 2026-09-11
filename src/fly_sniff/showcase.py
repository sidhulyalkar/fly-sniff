from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon

from .controllers import BilateralProxyController, RandomWalkController
from .party_social import (
    BG,
    CULPRIT_INDEX,
    MUTED,
    PANEL,
    PEOPLE,
    PERSON_COLORS,
    PLUME,
    PROXY_CLAIM_LABEL,
    PROXY_COLOR,
    RANDOM_COLOR,
    TEXT,
    PartyAgent,
    _draw_fly_pov,
    _draw_person,
    _make_agent,
    _should_reveal,
)

SHOWCASE_WIDTH = 1080
SHOWCASE_HEIGHT = 1350
SHOWCASE_DPI = 100
SHOWCASE_ASPECT = SHOWCASE_WIDTH / SHOWCASE_HEIGHT


def _distance_to_source(live: PartyAgent) -> float:
    arena = live.env.arena
    return float(np.hypot(live.env.agent.x - arena.source_x, live.env.agent.y - arena.source_y))


def _result_text(live: PartyAgent) -> str:
    if live.env.agent.found:
        return "CASE CLOSED"
    if live.done:
        return "STILL SNIFFING"
    return "SEARCHING"


def _draw_fly_marker(ax, live: PartyAgent) -> None:
    history = np.asarray(live.env.agent.history)
    x, y = history[-1]
    heading = live.env.agent.heading
    forward = np.array([np.cos(heading), np.sin(heading)])
    side = np.array([-forward[1], forward[0]])
    tip = np.array([x, y]) + 0.28 * forward
    back_left = np.array([x, y]) - 0.18 * forward + 0.17 * side
    back_right = np.array([x, y]) - 0.18 * forward - 0.17 * side
    ax.add_patch(
        Polygon(
            [tip, back_left, back_right],
            closed=True,
            facecolor=live.color,
            edgecolor=TEXT,
            linewidth=1.5,
            zorder=10,
        )
    )


def _draw_shared_room(
    ax,
    agents: list[PartyAgent],
    snap: np.ndarray,
    *,
    reveal: bool,
) -> None:
    arena = agents[0].env.arena
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, arena.width)
    ax.set_ylim(0, arena.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.8)

    for index, (x, y, tag) in enumerate(PEOPLE):
        _draw_person(
            ax,
            x,
            y,
            tag,
            color=PERSON_COLORS[index % len(PERSON_COLORS)],
            culprit=reveal and index == CULPRIT_INDEX,
        )

    if len(snap):
        ax.scatter(
            snap[:, 0],
            snap[:, 1],
            s=np.clip(54 * snap[:, 2], 7, 72),
            alpha=0.40,
            c=PLUME,
            edgecolors="none",
            zorder=2,
        )

    for live in agents:
        history = np.asarray(live.env.agent.history)
        ax.plot(
            history[:, 0],
            history[:, 1],
            color=live.color,
            linewidth=5.0,
            alpha=0.96,
            zorder=6,
        )
        _draw_fly_marker(ax, live)

    ax.text(
        0.025,
        0.965,
        "BIOLOGY-INSPIRED PROXY",
        transform=ax.transAxes,
        va="top",
        fontsize=11.5,
        color=PROXY_COLOR,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": BG, "edgecolor": "none", "alpha": 0.9},
    )
    ax.text(
        0.975,
        0.965,
        "RANDOM CONTROL",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        color=RANDOM_COLOR,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": BG, "edgecolor": "none", "alpha": 0.9},
    )

    elapsed = agents[0].env.agent.steps * arena.dt
    left = agents[0]
    right = agents[1]
    ax.text(
        0.025,
        0.035,
        f"{elapsed:04.1f}s  •  {left.label}: {_distance_to_source(left):.1f} m away",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9.5,
        color=TEXT,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": BG, "edgecolor": "none", "alpha": 0.86},
    )
    ax.text(
        0.975,
        0.035,
        f"{right.label}: {_distance_to_source(right):.1f} m away",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=TEXT,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": BG, "edgecolor": "none", "alpha": 0.86},
    )


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
        _make_agent("PROXY", BilateralProxyController, seed, PROXY_COLOR),
        _make_agent("RANDOM", RandomWalkController, seed, RANDOM_COLOR),
    ]
    arena = agents[0].env.arena
    sim_steps_per_frame = max(1, round(3.0 * (1.0 / fps) / arena.dt))
    frames = max(1, seconds * fps)

    fig = plt.figure(
        figsize=(SHOWCASE_WIDTH / SHOWCASE_DPI, SHOWCASE_HEIGHT / SHOWCASE_DPI),
        dpi=SHOWCASE_DPI,
        facecolor=BG,
    )
    grid = fig.add_gridspec(4, 1, height_ratios=[0.42, 1.82, 0.80, 0.26], hspace=0.20)
    title_ax = fig.add_subplot(grid[0, 0])
    room_ax = fig.add_subplot(grid[1, 0])
    pov_ax = fig.add_subplot(grid[2, 0])
    footer_ax = fig.add_subplot(grid[3, 0])
    fig.subplots_adjust(left=0.035, right=0.965, top=0.988, bottom=0.028)

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
            0.74,
            "WHO FARTED?",
            ha="center",
            va="center",
            fontsize=40,
            color=TEXT,
            fontweight="bold",
        )
        subtitle = (
            "CULPRIT REVEALED • THE PATHS BELOW ARE THE ACTUAL RUN"
            if reveal
            else "ONE PLUME • TWO SEARCH STRATEGIES • SOURCE HIDDEN FROM BOTH"
        )
        title_ax.text(
            0.5,
            0.34,
            subtitle,
            ha="center",
            va="center",
            fontsize=12.5,
            color=MUTED,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.02,
            PROXY_CLAIM_LABEL,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#FBBF24",
            fontweight="bold",
        )

        _draw_shared_room(room_ax, agents, snap, reveal=reveal)
        _draw_fly_pov(pov_ax, agents[0], snap)

        footer_ax.clear()
        footer_ax.set_facecolor(BG)
        footer_ax.axis("off")
        if reveal:
            culprit = PEOPLE[CULPRIT_INDEX][2]
            footer_ax.text(
                0.5,
                0.70,
                f"CULPRIT: {culprit}",
                ha="center",
                va="center",
                fontsize=20,
                color="#D9F99D",
                fontweight="bold",
            )
            footer_ax.text(
                0.5,
                0.18,
                "  •  ".join(f"{live.label}: {_result_text(live)}" for live in agents),
                ha="center",
                va="center",
                fontsize=11,
                color=TEXT,
                fontweight="bold",
            )
        else:
            footer_ax.text(
                0.5,
                0.50,
                "green = modeled odor visible to you • controllers receive only antenna signals",
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
