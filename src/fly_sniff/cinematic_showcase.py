from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection

from .party_social import BG, CULPRIT_INDEX, MUTED, PANEL, PEOPLE, TEXT
from .recorded_showcase import (
    _agent,
    _draw_room,
    _draw_sensor_hud,
    _outcome_text,
    _precompute_histories,
    _validate_social_contract,
    reveal_text,
)
from .recording import load_recording
from .scientific_view import draw_trace, trace_arrays
from .showcase import SHOWCASE_DPI, SHOWCASE_HEIGHT, SHOWCASE_WIDTH


def load_neural_scene(path: str | Path) -> dict[str, Any]:
    scene = json.loads(Path(path).read_text())
    expected = scene.pop("scene_sha256", None)
    actual = hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if not expected or expected != actual:
        raise ValueError("neural scene SHA-256 missing or mismatched; re-export the source graph")
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
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        "SOURCE SOMA POSITIONS • X/Z",
        transform=ax.transAxes,
        va="top",
        color=TEXT,
        fontsize=11.5,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.86,
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
    segments = [
        [(scene_nodes[int(e["source"])]["screen_x"], scene_nodes[int(e["source"])]["screen_y"]),
         (scene_nodes[int(e["target"])]["screen_x"], scene_nodes[int(e["target"])]["screen_y"])]
        for e in edges if int(e["source"]) in scene_nodes and int(e["target"]) in scene_nodes
    ]
    ax.add_collection(LineCollection(segments, colors="#64748B", linewidths=0.4, alpha=0.18))

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
            "Static structure only; this controller has no neural state",
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
        0.12,
        f"Showing {len(edges):,}/{len(scene['edges']):,} edges • {scene['nodes_missing_position']} missing somas",
        transform=ax.transAxes,
        ha="right",
        color=MUTED,
        fontsize=7.2,
    )


def validate_scene_binding(payload, scene):
    """Reject unbound/mismatched modeled activity before rendering any frames."""
    node_ids = {int(n["body_id"]) for n in scene["nodes"]}
    if len(node_ids) != len(scene["nodes"]):
        raise ValueError("duplicate neural scene body IDs")
    if not all(np.isfinite([n["screen_x"], n["screen_y"]]).all() for n in scene["nodes"]):
        raise ValueError("nonfinite neural scene coordinates")
    graph_digest = scene.get("graph_sha256")
    all_ids = set(scene.get("all_body_ids", node_ids))
    for frame in payload["frames"]:
        for state in frame["agents"]:
            neural = state.get("neural_activity")
            if neural is None:
                continue
            if neural.get("signal_kind") != "modeled_rate_state":
                raise ValueError("unsupported neural signal kind")
            if not graph_digest or neural.get("graph_sha256") != graph_digest:
                raise ValueError("neural state and anatomical scene graph fingerprint mismatch")
            if payload.get("graph_sha256") != graph_digest:
                raise ValueError("recording and scene graph fingerprint mismatch")
            cells = neural.get("cells", [])
            ids = [int(c["body_id"]) for c in cells]
            if len(ids) != len(set(ids)) or not set(ids).issubset(all_ids):
                raise ValueError("unknown or duplicate recorded neural body IDs")
            if any(not np.isfinite(c["activity"]) or abs(c["activity"]) > 1
                   for c in neural.get("cells", [])):
                raise ValueError("invalid modeled activity")
    # Qualification cannot be inferred from a user-editable scene label.
    scene["claim_label"] = "STRUCTURE / MODELED STATE • NOT A QUALIFIED RESULT"


def render_cinematic_showcase(
    recording: str | Path,
    neural_scene: str | Path | None,
    output: str | Path,
    *,
    seconds: int = 15,
    fps: int = 30,
) -> Path:
    bundle = load_recording(recording)
    payload = bundle["recording"]
    _validate_social_contract(payload)
    scene = load_neural_scene(neural_scene) if neural_scene else None
    if scene:
        validate_scene_binding(payload, scene)
    if seconds < 1 or fps < 1:
        raise ValueError("seconds and fps must be >= 1")
    output = Path(output)
    if output.suffix.lower() not in {".gif", ".mp4", ".png"}:
        raise ValueError("output must be GIF, MP4, or a final-frame PNG")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = payload["frames"]
    histories = _precompute_histories(payload)
    labels = [c["label"] for c in payload["controllers"][:2]]
    trace = trace_arrays(payload, labels[0])
    scene_nodes, scene_xy = _scene_index(scene) if scene else (None, None)
    video_frames = seconds * fps
    duration = float(frames[-1]["t"])
    fig = plt.figure(figsize=(SHOWCASE_WIDTH / SHOWCASE_DPI, SHOWCASE_HEIGHT / SHOWCASE_DPI),
                     dpi=SHOWCASE_DPI, facecolor=BG)
    # Full-width sensory HUD avoids squeezing labels into half a phone screen.
    ratios = [0.40, 1.48, 0.72, 0.48] + ([0.90] if scene else []) + [0.28]
    grid = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.35)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(len(ratios))]
    title, room, hud, timeline = axes[:4]
    footer = axes[-1]
    fig.subplots_adjust(left=0.055, right=0.965, top=0.985, bottom=0.035)

    def draw(video_index):
        playback_frames = max(1, video_frames - min(2 * fps, video_frames // 3))
        fraction = 1.0 if video_frames == 1 else min(1.0, video_index / max(1, playback_frames - 1))
        index = round(fraction * (len(frames) - 1))
        current = frames[index]
        first, second = (_agent(current, label) for label in labels)
        reveal = bool(first["found"] or second["found"] or video_index >= video_frames - 2 * fps)
        title.clear()
        title.axis("off")
        title.text(0.5, 0.82, "WHO FARTED?", ha="center", va="center", fontsize=35,
                   color=TEXT, weight="bold")
        title.text(0.5, 0.43, "YOU SEE THE SMELL. THE MODEL ONLY SENSES ITS ANTENNAE + AIRFLOW.",
                   ha="center", fontsize=10, color=MUTED)
        title.text(0.5, 0.15, payload["claim_boundary"], ha="center", color="#FBBF24",
                   fontsize=9, weight="bold")
        _draw_room(room, payload, histories, index, reveal=reveal)
        _draw_sensor_hud(hud, first)
        draw_trace(timeline, trace, index)
        if scene:
            _draw_connectome_panel(axes[4], scene, scene_nodes, scene_xy, first)
            axes[4].text(0.02, 0.22, "MODELED STATE ≠ SPIKES • green + / pink − • sparse top-|state|",
                         transform=axes[4].transAxes, fontsize=7, color=MUTED)
        footer.clear()
        footer.axis("off")
        result = reveal_text(first, second, reveal=reveal)
        if reveal:
            result += f": {PEOPLE[CULPRIT_INDEX][2]}"
        footer.text(0.5, 0.88, result, ha="center", color="#D9F99D", fontsize=13, weight="bold")
        if index == len(frames) - 1:
            detail = " • ".join(_outcome_text(payload, label) for label in labels)
        else:
            detail = "Same seeded plume • one development episode • no cohort inference"
        footer.text(0.5, 0.43, detail, ha="center", color=TEXT, fontsize=8)
        footer.text(0.5, 0.02,
                    f"{duration:.1f}s simulated / {seconds}s clip incl. end hold • seed {payload['seed']} • replay {bundle['recording_sha256'][:12]}",
                    ha="center", color=MUTED, fontsize=7)
        return []

    try:
        if output.suffix.lower() == ".png":
            draw(video_frames - 1)
            fig.savefig(output, dpi=SHOWCASE_DPI, facecolor=BG)
        else:
            ani = animation.FuncAnimation(fig, draw, frames=video_frames, interval=1000/fps,
                                          blit=False, cache_frame_data=False)
            if output.suffix.lower() == ".gif":
                writer = animation.PillowWriter(fps=fps)
            else:
                if not animation.writers.is_available("ffmpeg"):
                    raise RuntimeError("ffmpeg is required for MP4 output; use GIF or PNG")
                writer = animation.FFMpegWriter(fps=fps, bitrate=6500,
                                               extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            ani.save(output, writer=writer)
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the room, scientific sensory trace, and optional bound anatomy")
    parser.add_argument("recording")
    parser.add_argument("--neural-scene", help="Optional source-coordinate scene; never fabricated")
    parser.add_argument("--output", default="artifacts/showcase/who-farted-scientific.mp4")
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(render_cinematic_showcase(args.recording, args.neural_scene, args.output,
                                    seconds=args.seconds, fps=args.fps))


if __name__ == "__main__":
    main()
