from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Polygon, Rectangle

from .config import ArenaConfig, PlumeConfig
from .controllers import BilateralProxyController, Controller, RandomWalkController
from .env import FlySniffEnv, Observation


@dataclass
class PartyAgent:
    label: str
    env: FlySniffEnv
    controller: Controller
    obs: Observation
    color: str
    done: bool = False


PEOPLE = (
    (1.0, 3.0, "A"),
    (1.0, 0.9, "B"),
    (1.0, 5.1, "C"),
    (4.8, 0.8, "D"),
    (4.8, 5.2, "E"),
    (8.2, 5.1, "F"),
)
PERSON_COLORS = ("#F5B7B1", "#AED6F1", "#F9E79F", "#D7BDE2", "#A9DFBF", "#F5CBA7")
CULPRIT_INDEX = 0
PROXY_CLAIM_LABEL = "DEVELOPMENT PROXY • NOT A MALECNS RESULT"
BG = "#0B0F17"
PANEL = "#111827"
TEXT = "#F8FAFC"
MUTED = "#CBD5E1"
PLUME = "#9BE15D"
PROXY_COLOR = "#38BDF8"
RANDOM_COLOR = "#FB7185"


def _make_agent(label: str, factory: type[Controller], seed: int, color: str) -> PartyAgent:
    arena = ArenaConfig()
    env = FlySniffEnv(seed=seed, arena=arena, plume=PlumeConfig())
    controller = factory()
    controller.reset(seed + 101)
    return PartyAgent(label, env, controller, env.observe(), color)


def _should_reveal(*, frame: int, frames: int, fps: int, done: bool) -> bool:
    """Reveal ground truth after success or during the final two seconds.

    The timed reveal guarantees that a short social render has a payoff even if
    a controller has not reached the source. It does not alter controller state,
    success metrics, or the scientific benchmark.
    """
    reveal_frames = max(1, 2 * fps)
    return done or frame >= max(0, frames - reveal_frames)


def _draw_stink_cloud(ax, x: float, y: float) -> None:
    """Draw a font-independent cartoon odor cloud for the reveal."""
    blobs = (
        (0.42, 0.22, 0.20),
        (0.58, 0.34, 0.25),
        (0.73, 0.22, 0.18),
        (0.86, 0.42, 0.14),
    )
    for dx, dy, radius in blobs:
        ax.add_patch(
            Circle(
                (x + dx, y + dy),
                radius,
                facecolor=PLUME,
                edgecolor="#D9F99D",
                linewidth=1.5,
                alpha=0.84,
                zorder=11,
            )
        )
    ax.text(
        x + 0.64,
        y + 0.72,
        "GUILTY",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#D9F99D",
        fontweight="bold",
        zorder=12,
    )


def _draw_person(
    ax,
    x: float,
    y: float,
    tag: str,
    *,
    color: str,
    culprit: bool = False,
) -> None:
    # Chunky silhouettes stay legible after GIF compression and phone scaling.
    ax.add_patch(
        Rectangle(
            (x - 0.19, y - 0.28),
            0.38,
            0.42,
            facecolor=color,
            edgecolor=TEXT,
            linewidth=1.1,
            zorder=7,
        )
    )
    ax.plot([x - 0.18, x - 0.34], [y + 0.03, y - 0.12], color=color, linewidth=4, zorder=6)
    ax.plot([x + 0.18, x + 0.34], [y + 0.03, y - 0.12], color=color, linewidth=4, zorder=6)
    ax.add_patch(
        Circle(
            (x, y + 0.31),
            0.19,
            facecolor=color,
            edgecolor=TEXT,
            linewidth=1.1,
            zorder=8,
        )
    )
    ax.text(
        x,
        y + 0.31,
        tag,
        ha="center",
        va="center",
        fontsize=9,
        color=BG,
        fontweight="bold",
        zorder=9,
    )
    if culprit:
        ax.add_patch(
            Circle(
                (x, y + 0.05),
                0.54,
                fill=False,
                edgecolor="#D9F99D",
                linewidth=3.2,
                zorder=10,
            )
        )
        _draw_stink_cloud(ax, x, y)


def _draw_room(ax, live: PartyAgent, snap: np.ndarray, *, reveal: bool) -> None:
    arena = live.env.arena
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, arena.width)
    ax.set_ylim(0, arena.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.5)
    ax.set_title(live.label, fontsize=14, fontweight="bold", color=live.color, pad=8)

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
            s=np.clip(42 * snap[:, 2], 5, 58),
            alpha=0.34,
            c=PLUME,
            edgecolors="none",
            zorder=2,
        )

    hist = np.asarray(live.env.agent.history)
    ax.plot(hist[:, 0], hist[:, 1], color=live.color, linewidth=4.0, zorder=4)
    # A tiny triangular fly reads more clearly than an X at phone scale.
    x, y = hist[-1]
    heading = live.env.agent.heading
    forward = np.array([np.cos(heading), np.sin(heading)])
    side = np.array([-forward[1], forward[0]])
    tip = np.array([x, y]) + 0.24 * forward
    back_left = np.array([x, y]) - 0.16 * forward + 0.15 * side
    back_right = np.array([x, y]) - 0.16 * forward - 0.15 * side
    ax.add_patch(
        Polygon(
            [tip, back_left, back_right],
            closed=True,
            facecolor=live.color,
            edgecolor=TEXT,
            linewidth=1.2,
            zorder=6,
        )
    )
    elapsed = live.env.agent.steps * arena.dt
    status = "FOUND IT" if live.env.agent.found else "SNIFFING..."
    badge_color = "#86EFAC" if live.env.agent.found else MUTED
    ax.text(
        0.03,
        0.96,
        f"{elapsed:04.1f}s   {status}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        color=badge_color,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": BG, "edgecolor": "none", "alpha": 0.88},
    )
    ax.text(
        0.03,
        0.035,
        "odor visible to you • source hidden from controller",
        transform=ax.transAxes,
        fontsize=7.5,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": BG, "edgecolor": "none", "alpha": 0.76},
    )


def _draw_fly_pov(ax, live: PartyAgent, snap: np.ndarray) -> None:
    """Render a compact fly-centered forward view with audience-visible odor."""
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_title("FLY POV  •  ODOR MADE VISIBLE", fontsize=10, fontweight="bold", color=TEXT, pad=5)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(0.0, 4.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#334155")
    agent = live.env.agent
    if len(snap):
        dx = snap[:, 0] - agent.x
        dy = snap[:, 1] - agent.y
        c = np.cos(agent.heading)
        s = np.sin(agent.heading)
        forward = c * dx + s * dy
        lateral = -s * dx + c * dy
        mask = (forward > 0.0) & (forward < 4.2) & (np.abs(lateral) < 2.4)
        if np.any(mask):
            ax.scatter(
                lateral[mask],
                forward[mask],
                s=np.clip(80 * snap[mask, 2] / np.maximum(forward[mask], 0.25), 6, 110),
                alpha=0.38,
                c=PLUME,
                edgecolors="none",
            )
    ax.plot([-0.12, -0.38], [0.2, 0.72], color=PROXY_COLOR, linewidth=2.4)
    ax.plot([0.12, 0.38], [0.2, 0.72], color="#F59E0B", linewidth=2.4)
    ax.add_patch(Circle((0.0, 0.16), 0.22, facecolor="#64748B", edgecolor=TEXT, linewidth=1.0))
    obs = live.obs
    max_odor = max(obs.left_odor, obs.right_odor, 1e-9)
    left_frac = min(1.0, obs.left_odor / max_odor)
    right_frac = min(1.0, obs.right_odor / max_odor)
    ax.text(
        0.02,
        0.91,
        f"L {obs.left_odor:.2f}",
        transform=ax.transAxes,
        fontsize=8.5,
        color=TEXT,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.98,
        0.91,
        f"R {obs.right_odor:.2f}",
        transform=ax.transAxes,
        fontsize=8.5,
        color=TEXT,
        fontweight="bold",
        ha="right",
        va="top",
    )
    ax.add_patch(Rectangle((-2.22, 3.55), 1.62 * left_frac, 0.18, facecolor=PROXY_COLOR, edgecolor="none"))
    ax.add_patch(Rectangle((0.60, 3.55), 1.62 * right_frac, 0.18, facecolor="#F59E0B", edgecolor="none"))


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

    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=BG)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.42, 2.15, 0.82], hspace=0.26, wspace=0.16)
    title_ax = fig.add_subplot(grid[0, :])
    room_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    pov_ax = fig.add_subplot(grid[2, :])
    fig.subplots_adjust(left=0.045, right=0.955, top=0.98, bottom=0.045)

    def draw(frame: int):
        for live in agents:
            for _ in range(sim_steps_per_frame):
                if not live.done:
                    action = live.controller.act(live.obs)
                    live.obs, live.done = live.env.step(action.turn, action.speed)

        snap = agents[0].env.plume.snapshot()
        reveal_ground_truth = _should_reveal(
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
            0.70,
            "WHO FARTED?",
            ha="center",
            va="center",
            fontsize=31,
            color=TEXT,
            fontweight="bold",
        )
        subtitle = (
            "CULPRIT REVEAL  •  GROUND TRUTH ONLY"
            if reveal_ground_truth
            else "SAME ODOR PLUME. SAME START. WHICH CONTROLLER FINDS THE SOURCE?"
        )
        title_ax.text(
            0.5,
            0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=10.5,
            color=MUTED,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.01,
            PROXY_CLAIM_LABEL,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#FBBF24",
            fontweight="bold",
        )

        for ax, live in zip(room_axes, agents, strict=True):
            _draw_room(ax, live, snap, reveal=reveal_ground_truth or live.done)
        _draw_fly_pov(pov_ax, agents[0], snap)
        return []

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
