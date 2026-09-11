from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle

from .config import ArenaConfig, PlumeConfig
from .controllers import BilateralProxyController, Controller, RandomWalkController
from .env import FlySniffEnv, Observation


@dataclass
class PartyAgent:
    label: str
    env: FlySniffEnv
    controller: Controller
    obs: Observation
    done: bool = False


PEOPLE = (
    (1.0, 3.0, "A"),
    (1.0, 0.9, "B"),
    (1.0, 5.1, "C"),
    (4.8, 0.8, "D"),
    (4.8, 5.2, "E"),
    (8.2, 5.1, "F"),
)
CULPRIT_INDEX = 0


def _make_agent(label: str, factory: type[Controller], seed: int) -> PartyAgent:
    arena = ArenaConfig()
    env = FlySniffEnv(seed=seed, arena=arena, plume=PlumeConfig())
    controller = factory()
    controller.reset(seed + 101)
    return PartyAgent(label, env, controller, env.observe())


def _draw_person(ax, x: float, y: float, tag: str, *, culprit: bool = False) -> None:
    ax.add_patch(Rectangle((x - 0.12, y - 0.28), 0.24, 0.36, alpha=0.72, zorder=6))
    ax.add_patch(Circle((x, y + 0.19), 0.14, alpha=0.88, zorder=7))
    ax.text(x, y + 0.19, tag, ha="center", va="center", fontsize=7, fontweight="bold", zorder=8)
    if culprit:
        ax.add_patch(Circle((x, y), 0.42, fill=False, linewidth=3.0, zorder=9))
        ax.text(x + 0.48, y + 0.35, "💨", fontsize=18, zorder=10)


def _draw_room(ax, live: PartyAgent, snap: np.ndarray, *, reveal: bool) -> None:
    arena = live.env.arena
    ax.clear()
    ax.set_xlim(0, arena.width)
    ax.set_ylim(0, arena.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(live.label, fontsize=13, fontweight="bold")
    ax.add_patch(Rectangle((0, 0), arena.width, arena.height, fill=False, linewidth=2.2))

    for index, (x, y, tag) in enumerate(PEOPLE):
        _draw_person(ax, x, y, tag, culprit=reveal and index == CULPRIT_INDEX)

    if len(snap):
        ax.scatter(
            snap[:, 0],
            snap[:, 1],
            s=np.clip(34 * snap[:, 2], 4, 48),
            alpha=0.20,
            c="yellowgreen",
            edgecolors="none",
            zorder=2,
        )

    hist = np.asarray(live.env.agent.history)
    ax.plot(hist[:, 0], hist[:, 1], linewidth=3.0, zorder=4)
    ax.scatter([hist[-1, 0]], [hist[-1, 1]], s=75, marker="X", zorder=5)
    elapsed = live.env.agent.steps * arena.dt
    status = "FOUND IT" if live.env.agent.found else "SNIFFING..."
    ax.text(
        0.03,
        0.96,
        f"{elapsed:04.1f}s  •  {status}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.03,
        "odor visible to YOU • source position hidden from controller",
        transform=ax.transAxes,
        fontsize=7,
    )


def _draw_fly_pov(ax, live: PartyAgent, snap: np.ndarray) -> None:
    """Render a fly-centered view with odor particles made visible to the audience."""
    ax.clear()
    ax.set_title("FLY POV • odor made visible", fontsize=10, fontweight="bold")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(0.0, 5.0)
    ax.set_xticks([])
    ax.set_yticks([])
    agent = live.env.agent
    if len(snap):
        dx = snap[:, 0] - agent.x
        dy = snap[:, 1] - agent.y
        c = np.cos(agent.heading)
        s = np.sin(agent.heading)
        forward = c * dx + s * dy
        lateral = -s * dx + c * dy
        mask = (forward > 0.0) & (forward < 5.0) & (np.abs(lateral) < 2.4)
        if np.any(mask):
            ax.scatter(
                lateral[mask],
                forward[mask],
                s=np.clip(70 * snap[mask, 2] / np.maximum(forward[mask], 0.25), 5, 95),
                alpha=0.23,
                c="yellowgreen",
                edgecolors="none",
            )
    ax.plot([-0.12, -0.34], [0.18, 0.55], linewidth=2.0)
    ax.plot([0.12, 0.34], [0.18, 0.55], linewidth=2.0)
    ax.add_patch(Circle((0.0, 0.12), 0.18, alpha=0.85))
    obs = live.obs
    ax.text(
        0.03,
        0.94,
        f"L antenna {obs.left_odor:.2f}   R antenna {obs.right_odor:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )


def render_party_proxy(
    output: str | Path,
    *,
    seed: int = 13013,
    seconds: int = 15,
    fps: int = 30,
) -> Path:
    """Render the fast social concept using development-only controllers.

    This renderer can never label its controllers MaleCNS or rewired connectome.
    It exists to tune scene readability while E001/E002 qualification continues.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    agents = [
        _make_agent("BIOLOGY-INSPIRED PROXY", BilateralProxyController, seed),
        _make_agent("RANDOM CONTROL", RandomWalkController, seed),
    ]
    arena = agents[0].env.arena
    sim_steps_per_frame = max(1, round(3.0 * (1.0 / fps) / arena.dt))

    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.48, 2.7, 1.15])
    title_ax = fig.add_subplot(grid[0, :])
    room_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    pov_ax = fig.add_subplot(grid[2, :])

    def draw(_frame: int):
        for live in agents:
            for _ in range(sim_steps_per_frame):
                if not live.done:
                    action = live.controller.act(live.obs)
                    live.obs, live.done = live.env.step(action.turn, action.speed)

        snap = agents[0].env.plume.snapshot()
        title_ax.clear()
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.68,
            "WHO FARTED?",
            ha="center",
            va="center",
            fontsize=29,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.27,
            "Can a fly-brain-inspired controller follow the smell to the guilty human?",
            ha="center",
            va="center",
            fontsize=11,
        )
        title_ax.text(
            0.5,
            0.02,
            "DEVELOPMENT PROXY • NOT A MALECNS RESULT",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

        for ax, live in zip(room_axes, agents, strict=True):
            _draw_room(ax, live, snap, reveal=live.done)
        _draw_fly_pov(pov_ax, agents[0], snap)
        return []

    frames = max(1, seconds * fps)
    ani = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    try:
        if output.suffix.lower() == ".gif":
            ani.save(output, writer=animation.PillowWriter(fps=fps))
        else:
            ani.save(output, writer=animation.FFMpegWriter(fps=fps, bitrate=5200))
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Who Farted? development social concept")
    parser.add_argument("--output", default="artifacts/who-farted-proxy.mp4")
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(render_party_proxy(args.output, seed=args.seed, seconds=args.seconds, fps=args.fps))
