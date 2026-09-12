from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .party_social import BG, CULPRIT_INDEX, MUTED, PANEL, PEOPLE, TEXT
from .recorded_showcase import (
    _agent,
    _draw_room,
    _draw_sensor_hud,
    _outcome_text,
    _precompute_histories,
)
from .recording import load_recording
from .showcase import SHOWCASE_DPI, SHOWCASE_HEIGHT, SHOWCASE_WIDTH


def load_neural_scene(path: str | Path) -> dict[str, Any]:
    scene = json.loads(Path(path).read_text())
    if scene.get("schema_version") != 1:
        raise ValueError(f"unsupported neural-scene schema {scene.get('schema_version')!r}")
    if not isinstance(scene.get("nodes"), list) or not isinstance(scene.get("edges"), list):
        raise TypeError("invalid neural scene")
    return scene


def _scene_index(scene: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    nodes = {int(node["body_id"]): node for node in scene["nodes"]}
    if not nodes:
        return nodes, np.empty((0, 2), dtype=float)
    xy = np.asarray(
        [[float(node["screen_x"]), float(node["screen_y"])] for node in nodes.values()],
        dtype=float,
    )
    return nodes, xy


def _draw_connectome_panel(
    ax,
    scene: dict[str, Any],
    scene_nodes: dict[int, dict[str, Any]],
    scene_xy: np.ndarray,
    state: dict[str, Any],
) -> None:
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        "INSIDE THE CONNECTOME",
        transform=ax.transAxes,
        va="top",
        color=TEXT,
        fontsize=11.5,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.96,
        scene["claim_label"],
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#FBBF24",
        fontsize=7.6,
        fontweight="bold",
    )

    if len(scene_xy):
        ax.scatter(
            scene_xy[:, 0],
            scene_xy[:, 1],
            s=7,
            c="#64748B",
            alpha=0.20,
            edgecolors="none",
            zorder=1,
        )

    max_edges = 1800
    edges = sorted(
        scene["edges"],
        key=lambda edge: abs(float(edge.get("weight", 0.0))),
        reverse=True,
    )[:max_edges]
    for edge in edges:
        source = scene_nodes.get(int(edge["source"]))
        target = scene_nodes.get(int(edge["target"]))
        if source is None or target is None:
            continue
        ax.plot(
            [source["screen_x"], target["screen_x"]],
            [source["screen_y"], target["screen_y"]],
            color="#475569",
            linewidth=0.35,
            alpha=0.10,
            zorder=0,
        )

    neural = state.get("neural_activity")
    if not neural:
        ax.text(
            0.5,
            0.48,
            "NO NEURAL STATE IN THIS RECORDING",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.36,
            "proxy steering is not painted as fake neuron firing",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#94A3B8",
            fontsize=8.8,
        )
        return

    active_xy: list[tuple[float, float]] = []
    active_values: list[float] = []
    active_ids: list[int] = []
    for cell in neural.get("cells", []):
        body_id = int(cell["body_id"])
        node = scene_nodes.get(body_id)
        if node is None:
            continue
        active_xy.append((float(node["screen_x"]), float(node["screen_y"])))
        active_values.append(float(cell["activity"]))
        active_ids.append(body_id)

    if active_xy:
        xy = np.asarray(active_xy, dtype=float)
        values = np.asarray(active_values, dtype=float)
        strength = np.clip(np.abs(values), 0.0, 1.0)
        sizes = 30.0 + 260.0 * strength
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=sizes * 2.0,
            c="#F8FAFC",
            alpha=0.08 + 0.20 * strength,
            edgecolors="none",
            zorder=4,
        )
        colors = np.where(values >= 0.0, "#A3E635", "#F472B6")
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=sizes,
            c=colors,
            alpha=0.45 + 0.50 * strength,
            edgecolors="none",
            zorder=5,
        )

        strongest = np.argsort(strength)[-3:][::-1]
        labels: list[str] = []
        for index in strongest:
            body_id = active_ids[int(index)]
            node = scene_nodes[body_id]
            cell_type = str(node.get("type") or node.get("instance") or body_id)
            labels.append(f"{cell_type}: {values[int(index)]:+.2f}")
        ax.text(
            0.02,
            0.04,
            "  •  ".join(labels),
            transform=ax.transAxes,
            color=TEXT,
            fontsize=7.6,
            fontweight="bold",
        )

    ax.text(
        0.98,
        0.04,
        f"{scene['nodes_positioned']:,} positioned neurons • {len(scene['edges']):,} route edges",
        transform=ax.transAxes,
        ha="right",
        color=MUTED,
        fontsize=7.2,
    )


def render_cinematic_showcase(
    recording: str | Path,
    neural_scene: str | Path,
    output: str | Path,
    *,
    seconds: int = 15,
    fps: int = 30,
) -> Path:
    bundle = load_recording(recording)
    payload = bundle["recording"]
    scene = load_neural_scene(neural_scene)
    if seconds < 1 or fps < 1:
        raise ValueError("seconds and fps must be >= 1")
    if len(payload.get("controllers", [])) < 2:
        raise ValueError("cinematic comparison requires two controllers")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recorded_frames = payload["frames"]
    histories = _precompute_histories(payload)
    scene_nodes, scene_xy = _scene_index(scene)
    video_frames = max(1, seconds * fps)
    first_label = payload["controllers"][0]["label"]
    second_label = payload["controllers"][1]["label"]
    outcome = (
        f"{_outcome_text(payload, first_label)}   •   "
        f"{_outcome_text(payload, second_label)}"
    )

    fig = plt.figure(
        figsize=(SHOWCASE_WIDTH / SHOWCASE_DPI, SHOWCASE_HEIGHT / SHOWCASE_DPI),
        dpi=SHOWCASE_DPI,
        facecolor=BG,
    )
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.34, 1.70, 0.88, 0.22],
        width_ratios=[0.92, 1.08],
        hspace=0.16,
        wspace=0.08,
    )
    title_ax = fig.add_subplot(grid[0, :])
    room_ax = fig.add_subplot(grid[1, :])
    sensor_ax = fig.add_subplot(grid[2, 0])
    brain_ax = fig.add_subplot(grid[2, 1])
    footer_ax = fig.add_subplot(grid[3, :])
    fig.subplots_adjust(left=0.035, right=0.965, top=0.99, bottom=0.025)

    def draw(video_index: int):
        fraction = 1.0 if video_frames == 1 else video_index / (video_frames - 1)
        record_index = round(fraction * (len(recorded_frames) - 1))
        current = recorded_frames[record_index]
        first = _agent(current, first_label)
        second = _agent(current, second_label)
        reveal = bool(
            first["found"]
            or second["found"]
            or video_index >= max(0, video_frames - 2 * fps)
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
            fontsize=38,
            color=TEXT,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.17,
            "SMELL → CONNECTOME → STEERING",
            ha="center",
            va="center",
            fontsize=12.5,
            color=MUTED,
            fontweight="bold",
        )

        _draw_room(room_ax, payload, histories, record_index, reveal=reveal)
        _draw_sensor_hud(sensor_ax, first)
        _draw_connectome_panel(brain_ax, scene, scene_nodes, scene_xy, first)

        footer_ax.clear()
        footer_ax.set_facecolor(BG)
        footer_ax.axis("off")
        if reveal:
            footer_ax.text(
                0.5,
                0.67,
                f"BUSTED: {PEOPLE[CULPRIT_INDEX][2]}",
                ha="center",
                color="#D9F99D",
                fontsize=17,
                fontweight="bold",
            )
            footer_ax.text(
                0.5,
                0.13,
                outcome,
                ha="center",
                color=TEXT,
                fontsize=9.5,
                fontweight="bold",
            )
        else:
            footer_ax.text(
                0.5,
                0.45,
                "green odor is audience-only • neural glow uses recorded body IDs only",
                ha="center",
                color=MUTED,
                fontsize=8.8,
                fontweight="bold",
            )
        footer_ax.text(
            0.995,
            0.02,
            f"replay {bundle['recording_sha256'][:12]}",
            ha="right",
            color="#64748B",
            fontsize=6.5,
        )
        return []

    ani = animation.FuncAnimation(
        fig,
        draw,
        frames=video_frames,
        interval=1000 / fps,
        blit=False,
    )
    try:
        if output.suffix.lower() == ".gif":
            ani.save(output, writer=animation.PillowWriter(fps=fps))
        else:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError("ffmpeg is required for MP4 output")
            ani.save(output, writer=animation.FFMpegWriter(fps=fps, bitrate=7000))
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an anatomical MaleCNS + Who Farted? cinematic replay"
    )
    parser.add_argument("recording")
    parser.add_argument("--neural-scene", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/showcase/who-farted-connectome.mp4",
    )
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(
        render_cinematic_showcase(
            args.recording,
            args.neural_scene,
            args.output,
            seconds=args.seconds,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    main()
