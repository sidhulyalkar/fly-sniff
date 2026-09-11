from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon, Rectangle

from .party_social import (
    BG,
    CULPRIT_INDEX,
    MUTED,
    PANEL,
    PEOPLE,
    PERSON_COLORS,
    PLUME,
    TEXT,
    _draw_person,
)
from .recording import load_recording
from .showcase import SHOWCASE_DPI, SHOWCASE_HEIGHT, SHOWCASE_WIDTH


def _agent(frame: dict[str, Any], label: str) -> dict[str, Any]:
    for candidate in frame["agents"]:
        if candidate["label"] == label:
            return candidate
    raise ValueError(f"recording frame is missing agent {label!r}")


def _validate_social_contract(payload: dict[str, Any]) -> None:
    if "NOT A MALECNS RESULT" not in payload.get("claim_boundary", ""):
        raise ValueError(
            "development recording is missing the public claim boundary"
        )
    source = np.array(
        [payload["arena"]["source_x"], payload["arena"]["source_y"]],
        dtype=float,
    )
    culprit = np.array(PEOPLE[CULPRIT_INDEX][:2], dtype=float)
    if not np.allclose(source, culprit, rtol=0.0, atol=1e-12):
        raise ValueError(
            "social culprit is not colocated with the simulated odor source"
        )
    if len(payload.get("controllers", [])) < 2:
        raise ValueError(
            "recorded social comparison requires at least two controllers"
        )


def _draw_fly_marker(ax, state: dict[str, Any], color: str) -> None:
    x = float(state["x"])
    y = float(state["y"])
    heading = float(state["heading"])
    forward = np.array([np.cos(heading), np.sin(heading)])
    side = np.array([-forward[1], forward[0]])
    tip = np.array([x, y]) + 0.28 * forward
    back_left = np.array([x, y]) - 0.18 * forward + 0.17 * side
    back_right = np.array([x, y]) - 0.18 * forward - 0.17 * side
    ax.add_patch(
        Polygon(
            [tip, back_left, back_right],
            closed=True,
            facecolor=color,
            edgecolor=TEXT,
            linewidth=1.5,
            zorder=10,
        )
    )


def _precompute_histories(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    frames = payload["frames"]
    histories: dict[str, np.ndarray] = {}
    for controller in payload["controllers"]:
        label = controller["label"]
        histories[label] = np.asarray(
            [
                [_agent(frame, label)["x"], _agent(frame, label)["y"]]
                for frame in frames
            ],
            dtype=float,
        )
    return histories


def _badge_style(alpha: float = 0.9) -> dict[str, Any]:
    return {
        "boxstyle": "round,pad=0.25",
        "facecolor": BG,
        "edgecolor": "none",
        "alpha": alpha,
    }


def _draw_room(
    ax,
    payload: dict[str, Any],
    histories: dict[str, np.ndarray],
    frame_index: int,
    *,
    reveal: bool,
) -> None:
    arena = payload["arena"]
    frames = payload["frames"]
    current = frames[frame_index]
    controller_colors = {
        entry["label"]: entry["color"]
        for entry in payload["controllers"]
    }

    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, arena["width"])
    ax.set_ylim(0, arena["height"])
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

    plume = np.asarray(current["plume"], dtype=float)
    if plume.size:
        ax.scatter(
            plume[:, 0],
            plume[:, 1],
            s=np.clip(54 * plume[:, 2], 7, 72),
            alpha=0.40,
            c=PLUME,
            edgecolors="none",
            zorder=2,
        )

    for controller in payload["controllers"][:2]:
        label = controller["label"]
        color = controller_colors[label]
        history = histories[label][: frame_index + 1]
        ax.plot(
            history[:, 0],
            history[:, 1],
            color=color,
            linewidth=5.0,
            alpha=0.96,
            zorder=6,
        )
        state = _agent(current, label)
        _draw_fly_marker(ax, state, color)

    first_controller = payload["controllers"][0]
    second_controller = payload["controllers"][1]
    left = _agent(current, first_controller["label"])
    right = _agent(current, second_controller["label"])

    ax.text(
        0.025,
        0.965,
        first_controller["label"],
        transform=ax.transAxes,
        va="top",
        fontsize=11.5,
        color=first_controller["color"],
        fontweight="bold",
        bbox=_badge_style(),
    )
    ax.text(
        0.975,
        0.965,
        second_controller["label"],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        color=second_controller["color"],
        fontweight="bold",
        bbox=_badge_style(),
    )
    ax.text(
        0.025,
        0.035,
        f"{current['t']:04.1f}s  •  {left['distance_to_source']:.1f} m away",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9.5,
        color=TEXT,
        fontweight="bold",
        bbox=_badge_style(alpha=0.86),
    )
    ax.text(
        0.975,
        0.035,
        f"{right['distance_to_source']:.1f} m away",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=TEXT,
        fontweight="bold",
        bbox=_badge_style(alpha=0.86),
    )


def _draw_bar(
    ax,
    *,
    y: float,
    label: str,
    value: float,
) -> None:
    ax.text(
        0.03,
        y + 0.045,
        label,
        color=MUTED,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.add_patch(
        Rectangle(
            (0.03, y - 0.015),
            0.38,
            0.065,
            facecolor="#1F2937",
            edgecolor="#475569",
        )
    )
    ax.add_patch(
        Rectangle(
            (0.03, y - 0.015),
            0.38 * float(np.clip(value, 0.0, 1.0)),
            0.065,
            facecolor=PLUME,
            edgecolor="none",
        )
    )
    ax.text(
        0.43,
        y + 0.015,
        f"{value:.2f}",
        color=TEXT,
        fontsize=9.5,
        va="center",
    )


def _draw_sensor_hud(ax, state: dict[str, Any]) -> None:
    obs = state["observation"]
    action = state["action"]
    diag = state.get("diagnostics", {})
    left = float(obs["left_odor"])
    right = float(obs["right_odor"])
    delta = float(obs["odor_delta"])
    turn = float(action["turn"])

    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.03,
        0.91,
        "PROXY SENSORY → STEERING TRACE",
        color=TEXT,
        fontsize=11.5,
        fontweight="bold",
    )

    _draw_bar(ax, y=0.64, label="LEFT ANTENNA", value=left)
    _draw_bar(ax, y=0.43, label="RIGHT ANTENNA", value=right)
    ax.text(
        0.03,
        0.20,
        f"Δ odor (R−L): {delta:+.3f}",
        color=TEXT,
        fontsize=10,
        fontweight="bold",
    )

    wx = float(obs["wind_x_body"])
    wy = float(obs["wind_y_body"])
    magnitude = max(float(np.hypot(wx, wy)), 1e-9)
    ux = wx / magnitude
    uy = wy / magnitude
    ax.text(
        0.55,
        0.80,
        "BODY-FRAME WIND",
        color=MUTED,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.arrow(
        0.72,
        0.63,
        0.12 * ux,
        0.12 * uy,
        width=0.006,
        head_width=0.035,
        color="#93C5FD",
        length_includes_head=True,
    )
    ax.text(
        0.55,
        0.48,
        f"wind = ({wx:+.2f}, {wy:+.2f})",
        color=TEXT,
        fontsize=9.5,
    )

    if turn > 0.05:
        turn_text = f"TURN LEFT  {turn:+.2f}"
    elif turn < -0.05:
        turn_text = f"TURN RIGHT  {turn:+.2f}"
    else:
        turn_text = f"STRAIGHT  {turn:+.2f}"
    ax.text(
        0.55,
        0.30,
        turn_text,
        color="#FDE68A",
        fontsize=12,
        fontweight="bold",
    )

    if "mode_surge" in diag:
        if float(diag["mode_surge"]) > 0.5:
            mode = "ODOR-GATED UPWIND"
        else:
            mode = "CROSSWIND CAST"
        ax.text(
            0.55,
            0.16,
            mode,
            color=TEXT,
            fontsize=9.5,
            fontweight="bold",
        )

    ax.text(
        0.03,
        0.04,
        "Only modeled antenna + airflow signals drive this proxy. "
        "Green plume is audience-only.",
        color=MUTED,
        fontsize=8.8,
    )


def _status(state: dict[str, Any]) -> str:
    if state["found"]:
        return "FOUND SOURCE"
    return "DID NOT REACH SOURCE"


def render_recorded_showcase(
    recording: str | Path,
    output: str | Path,
    *,
    seconds: int = 15,
    fps: int = 30,
) -> Path:
    bundle = load_recording(recording)
    payload = bundle["recording"]
    _validate_social_contract(payload)
    if seconds < 1 or fps < 1:
        raise ValueError("seconds and fps must be >= 1")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recorded_frames = payload["frames"]
    histories = _precompute_histories(payload)
    video_frames = max(1, seconds * fps)

    fig = plt.figure(
        figsize=(
            SHOWCASE_WIDTH / SHOWCASE_DPI,
            SHOWCASE_HEIGHT / SHOWCASE_DPI,
        ),
        dpi=SHOWCASE_DPI,
        facecolor=BG,
    )
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.40, 1.86, 0.78, 0.24],
        hspace=0.18,
    )
    title_ax = fig.add_subplot(grid[0, 0])
    room_ax = fig.add_subplot(grid[1, 0])
    sensor_ax = fig.add_subplot(grid[2, 0])
    footer_ax = fig.add_subplot(grid[3, 0])
    fig.subplots_adjust(
        left=0.035,
        right=0.965,
        top=0.988,
        bottom=0.028,
    )

    first_label = payload["controllers"][0]["label"]
    second_label = payload["controllers"][1]["label"]

    def draw(video_index: int):
        if video_frames == 1:
            record_index = len(recorded_frames) - 1
        else:
            fraction = video_index / (video_frames - 1)
            record_index = round(fraction * (len(recorded_frames) - 1))
        current = recorded_frames[record_index]
        first = _agent(current, first_label)
        second = _agent(current, second_label)
        timed_reveal = video_index >= max(0, video_frames - 2 * fps)
        reveal = bool(first["found"] or second["found"] or timed_reveal)

        title_ax.clear()
        title_ax.set_facecolor(BG)
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.72,
            "WHO FARTED?",
            ha="center",
            va="center",
            fontsize=40,
            color=TEXT,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.30,
            "SAME RECORDED PLUME • WATCH THE ANTENNA SIGNALS DRIVE THE TURN",
            ha="center",
            va="center",
            fontsize=11.5,
            color=MUTED,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.01,
            payload["claim_boundary"],
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#FBBF24",
            fontweight="bold",
        )

        _draw_room(
            room_ax,
            payload,
            histories,
            record_index,
            reveal=reveal,
        )
        _draw_sensor_hud(sensor_ax, first)

        footer_ax.clear()
        footer_ax.set_facecolor(BG)
        footer_ax.axis("off")
        if reveal:
            culprit = PEOPLE[CULPRIT_INDEX][2]
            footer_ax.text(
                0.5,
                0.68,
                f"CULPRIT: {culprit}",
                ha="center",
                va="center",
                fontsize=18,
                color="#D9F99D",
                fontweight="bold",
            )
            footer_ax.text(
                0.5,
                0.16,
                f"{first_label}: {_status(first)}   •   "
                f"{second_label}: {_status(second)}",
                ha="center",
                va="center",
                fontsize=10.5,
                color=TEXT,
                fontweight="bold",
            )
        else:
            footer_ax.text(
                0.5,
                0.56,
                "source coordinates and culprit identity are never controller inputs",
                ha="center",
                va="center",
                fontsize=9.5,
                color=MUTED,
                fontweight="bold",
            )
        footer_ax.text(
            0.99,
            0.02,
            f"replay {bundle['recording_sha256'][:12]}",
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#64748B",
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
            ani.save(
                output,
                writer=animation.PillowWriter(fps=fps),
            )
        else:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError(
                    "ffmpeg is required for MP4 output; render GIF or install ffmpeg"
                )
            ani.save(
                output,
                writer=animation.FFMpegWriter(fps=fps, bitrate=6500),
            )
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Who Farted? showcase from a hashed episode recording"
        )
    )
    parser.add_argument("recording")
    parser.add_argument(
        "--output",
        default="artifacts/showcase/who-farted-recorded.mp4",
    )
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(
        render_recorded_showcase(
            args.recording,
            args.output,
            seconds=args.seconds,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    main()
