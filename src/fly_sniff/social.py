from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .config import ArenaConfig, PlumeConfig
from .controllers import (
    BilateralProxyController,
    CastSurgeController,
    Controller,
    RandomWalkController,
)
from .env import FlySniffEnv


@dataclass
class LiveAgent:
    label: str
    env: FlySniffEnv
    controller: Controller
    obs: object
    done: bool = False


def _make_agents(seed: int) -> list[LiveAgent]:
    factories = [
        ("BIOLOGY-INSPIRED PROXY", BilateralProxyController),
        ("CLASSICAL CAST + SURGE", CastSurgeController),
        ("RANDOM WALK", RandomWalkController),
    ]
    agents: list[LiveAgent] = []
    for label, factory in factories:
        env = FlySniffEnv(seed=seed, arena=ArenaConfig(), plume=PlumeConfig())
        controller = factory()
        controller.reset(seed + 101)
        agents.append(LiveAgent(label, env, controller, env.observe()))
    return agents


def render_social_video(output: str | Path, seed: int = 13013, seconds: int = 24, fps: int = 30) -> Path:
    """Render the development visualization.

    This renderer is intentionally watermarked PROXY until a qualified MaleCNS graph
    is supplied in a later tranche. It cannot produce a falsely labelled connectome result.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    agents = _make_agents(seed)
    arena = agents[0].env.arena
    sim_steps_per_frame = max(1, round((1.0 / fps) / arena.dt))

    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.25])
    ax = fig.add_subplot(gs[0, :])
    metric_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]
    fig.suptitle("FLYBRAIN PLUME HUNT • DEVELOPMENT PROXY", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.955,
        "SIMULATED PLUME + PROXY CONTROL • NOT YET A MALECNS RESULT",
        ha="center",
        fontsize=11,
    )

    colors = ["tab:blue", "tab:orange", "tab:gray"]

    def draw(_frame: int):
        for live in agents:
            for _ in range(sim_steps_per_frame):
                if not live.done:
                    action = live.controller.act(live.obs)
                    live.obs, live.done = live.env.step(action.turn, action.speed)
        ax.clear()
        ax.set_xlim(0, arena.width)
        ax.set_ylim(0, arena.height)
        ax.set_aspect("equal")
        ax.set_title("Same deterministic turbulent plume seed for every controller")
        ax.set_xlabel("upwind  ←                                        →  downwind")
        ax.set_ylabel("crosswind")
        snap = agents[0].env.plume.snapshot()
        if len(snap):
            ax.scatter(
                snap[:, 0],
                snap[:, 1],
                s=np.clip(22 * snap[:, 2], 3, 28),
                alpha=0.14,
                c="purple",
            )
        ax.scatter(
            [arena.source_x],
            [arena.source_y],
            marker="*",
            s=260,
            c="black",
            label="hidden odor source",
        )
        for live, color in zip(agents, colors, strict=True):
            hist = np.asarray(live.env.agent.history)
            ax.plot(hist[:, 0], hist[:, 1], lw=2.2, color=color, label=live.label)
            ax.scatter([hist[-1, 0]], [hist[-1, 1]], s=70, color=color)
        ax.legend(loc="upper right", fontsize=8)

        metric_axes[0].clear()
        labels = [a.label for a in agents]
        distances = [a.env.distance_to_source for a in agents]
        metric_axes[0].barh(labels, distances, color=colors)
        metric_axes[0].set_xlim(0, arena.width)
        metric_axes[0].set_title("Distance to odor source")
        metric_axes[0].set_xlabel("meters (simulator units)")

        metric_axes[1].clear()
        proxy = agents[0]
        d = proxy.controller.diagnostics()
        neural = [
            proxy.obs.left_odor,
            proxy.obs.right_odor,
            d.get("dn_left", 0),
            d.get("dn_right", 0),
        ]
        metric_axes[1].bar(["ORN-L", "ORN-R", "DN-L*", "DN-R*"], neural)
        metric_axes[1].set_ylim(0, 1)
        metric_axes[1].set_title("Development activity channels (*proxy)")
        metric_axes[1].set_ylabel("normalized activity")
        return []

    frames = max(1, seconds * fps)
    ani = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    try:
        if output.suffix.lower() == ".gif":
            ani.save(output, writer=animation.PillowWriter(fps=fps))
        else:
            ani.save(output, writer=animation.FFMpegWriter(fps=fps, bitrate=4500))
    finally:
        plt.close(fig)
    return output
