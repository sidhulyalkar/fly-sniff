from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Ellipse, Rectangle

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
from .scientific_view import draw_density
from .showcase import SHOWCASE_DPI, SHOWCASE_HEIGHT, SHOWCASE_WIDTH


def _agent(frame: dict[str, Any], label: str) -> dict[str, Any]:
    for candidate in frame["agents"]:
        if candidate["label"] == label:
            return candidate
    raise ValueError(f"recording frame is missing agent {label!r}")


def _validate_social_contract(payload: dict[str, Any]) -> None:
    if payload.get("claim_boundary") not in {
        "DEVELOPMENT PROXY • NOT A MALECNS RESULT",
        "CANDIDATE MODELED ACTIVITY • NOT A QUALIFIED MALECNS RESULT",
    }:
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
    angle = float(np.degrees(heading))
    for sign in (-1, 1):
        center = np.array([x, y]) - .08 * forward + sign * .14 * side
        ax.add_patch(Ellipse(center, .4, .17, angle=angle + sign * 24,
                             facecolor="#E2E8F0", alpha=.7, edgecolor=color, zorder=10))
    ax.add_patch(Ellipse((x, y), .43, .16, angle=angle,
                         facecolor=color, edgecolor=TEXT, linewidth=1.2, zorder=11))
    head = np.array([x, y]) + .18 * forward
    ax.add_patch(Ellipse(head, .14, .16, angle=angle, facecolor=TEXT, zorder=12))
    for sign in (-1, 1):
        antenna = head + .16 * forward + sign * .09 * side
        ax.plot([head[0], antenna[0]], [head[1], antenna[1]], color=color, lw=1.3, zorder=12)


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


def _first_found_time(payload: dict[str, Any], label: str) -> float | None:
    for frame in payload["frames"]:
        if _agent(frame, label)["found"]:
            return float(frame["t"])
    return None


def _outcome_text(payload: dict[str, Any], label: str) -> str:
    found_time = _first_found_time(payload, label)
    if found_time is not None:
        return f"{label}: FOUND SOURCE IN {found_time:.1f}s"
    final_time = float(payload["frames"][-1]["t"])
    return f"{label}: NO SOURCE IN {final_time:.0f}s"


def _badge_style(alpha: float = 0.9) -> dict[str, Any]:
    return {
        "boxstyle": "round,pad=0.25",
        "facecolor": BG,
        "edgecolor": "none",
        "alpha": alpha,
    }


def first_controller_label(payload):
    return str(payload["controllers"][0]["label"])


def reveal_text(first, second, *, reveal):
    if first["found"] or second["found"]:
        return "SOURCE REACHED"
    return "SOURCE REVEAL • NOT FOUND" if reveal else "FOLLOW THE SMELL"


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

    density_label = draw_density(ax, payload, current)
    ax.text(0.5, 0.015, density_label, transform=ax.transAxes, ha="center",
            color=MUTED, fontsize=7, bbox=_badge_style())

    states: list[dict[str, Any]] = []
    for controller in payload["controllers"][:2]:
        label = controller["label"]
        color = controller_colors[label]
        history = histories[label][: frame_index + 1]
        ax.plot(
            history[:, 0],
            history[:, 1],
            color=color,
            linewidth=2.7,
            alpha=0.96,
            zorder=6,
        )
        state = _agent(current, label)
        states.append(state)
        _draw_fly_marker(ax, state, color)

    if len(states) >= 2:
        separation = float(
            np.hypot(
                states[0]["x"] - states[1]["x"],
                states[0]["y"] - states[1]["y"],
            )
        )
        if separation < 0.18:
            ax.text(
                states[0]["x"] - 0.05,
                states[0]["y"] + 0.62,
                "BOTH START HERE" if frame_index == 0 else "PATHS OVERLAP",
                ha="center",
                color=TEXT,
                fontsize=8.8,
                fontweight="bold",
                bbox=_badge_style(alpha=0.82),
                zorder=12,
            )

    first_controller = payload["controllers"][0]
    second_controller = payload["controllers"][1]
    left = _agent(current, first_controller["label"])
    right = _agent(current, second_controller["label"])

    ax.text(
        0.025,
        0.965,
        first_controller_label(payload),
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
        str(payload["controllers"][1]["label"]),
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
    color: str = PLUME,
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
            facecolor=color,
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
        "WHAT THE MODELED FLY ACTUALLY GETS",
        color=TEXT,
        fontsize=11.5,
        fontweight="bold",
    )

    _draw_bar(ax, y=0.64, label="LEFT ANTENNA", value=left, color="#67E8F9")
    _draw_bar(ax, y=0.43, label="RIGHT ANTENNA", value=right, color="#C4B5FD")
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
        "BODY AIRFLOW: → FORWARD / ↑ LEFT",
        color=MUTED,
        fontsize=9.5,
        fontweight="bold",
    )
    compass = ax.inset_axes([0.65, 0.56, 0.20, 0.20])
    compass.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal")
    compass.axis("off")
    compass.arrow(0, 0, ux, uy, width=.03, head_width=.22, color="#93C5FD",
                  length_includes_head=True)
    ax.text(
        0.55,
        0.48,
        f"forward {wx:+.2f} • left {wy:+.2f} m/s",
        color=TEXT,
        fontsize=9.5,
    )

    valid = state.get("decision_valid", not state.get("done", False))
    if not valid:
        turn_text = "STOPPED" if state.get("done") else "END OF RECORDING"
    elif turn > 0.05:
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

    if valid and "mode_surge" in diag:
        if float(diag["mode_surge"]) > 0.5:
            mode = "SMELL DETECTED → GO UPWIND"
        else:
            mode = "SMELL LOST → CAST CROSSWIND"
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
        "No source coordinates. No culprit ID. No green-plume image.",
        color=MUTED,
        fontsize=8.8,
    )


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
    first_label = payload["controllers"][0]["label"]
    second_label = payload["controllers"][1]["label"]
    outcome = (
        f"{_outcome_text(payload, first_label)}   •   "
        f"{_outcome_text(payload, second_label)}"
    )

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
        subtitle = (
            reveal_text(first, second, reveal=reveal)
            if reveal
            else "YOU CAN SEE THE GREEN SMELL • THE FLY CAN'T"
        )
        title_ax.text(
            0.5,
            0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=12.5,
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
                f"{reveal_text(first, second, reveal=reveal)}: {culprit}",
                ha="center",
                va="center",
                fontsize=18,
                color="#D9F99D",
                fontweight="bold",
            )
            footer_ax.text(
                0.5,
                0.16,
                outcome,
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
                "same recorded plume • both strategies start from the same state",
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
