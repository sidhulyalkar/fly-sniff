from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Ellipse, FancyArrowPatch

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
from .recorded_showcase import (
    _agent,
    _badge_style,
    _draw_sensor_hud,
    _outcome_text,
    _precompute_histories,
    _validate_social_contract,
)
from .recording import load_recording
from .showcase import SHOWCASE_DPI, SHOWCASE_HEIGHT, SHOWCASE_WIDTH

DEFAULT_EVIDENCE_CONFIG = Path("configs/showcase_evidence_v2.json")


def _load_evidence_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    _validate_evidence_config(payload)
    return payload


def _validate_evidence_config(config: dict[str, Any]) -> None:
    if config.get("protocol") != "who-farted-showcase-evidence-v2":
        raise ValueError("showcase v2 requires who-farted-showcase-evidence-v2 config")
    structure = config.get("measured_structure", {})
    if not structure.get("allowed", False):
        raise ValueError("showcase v2 requires measured_structure.allowed=true")
    if structure.get("label") != "MEASURED CONNECTIVITY":
        raise ValueError("measured structure must be labeled MEASURED CONNECTIVITY")
    modeled = config.get("modeled_activity", {})
    if modeled.get("label") != "MODELED ACTIVITY":
        raise ValueError("modeled activity must be labeled MODELED ACTIVITY")
    if config.get("comparison", {}).get("real_vs_rewire_headline_allowed", False):
        raise ValueError("development showcase cannot enable real-vs-rewire headline")
    if config.get("behavioral_state", {}).get("malecns_behavior_claim_allowed", False):
        raise ValueError("development showcase cannot enable a MaleCNS behavioral claim")


def _antenna_strengths(state: dict[str, Any]) -> tuple[float, float]:
    obs = state.get("observation", {})
    left = float(np.clip(float(obs.get("left_odor", 0.0)), 0.0, 1.0))
    right = float(np.clip(float(obs.get("right_odor", 0.0)), 0.0, 1.0))
    return left, right


def _draw_fly_marker_v2(ax, state: dict[str, Any], color: str) -> None:
    """Draw a readable fly glyph whose antennae encode recorded odor only."""
    x = float(state["x"])
    y = float(state["y"])
    heading = float(state["heading"])
    left_odor, right_odor = _antenna_strengths(state)

    forward = np.array([np.cos(heading), np.sin(heading)], dtype=float)
    side = np.array([-forward[1], forward[0]], dtype=float)
    center = np.array([x, y], dtype=float)
    angle_deg = float(np.degrees(heading))

    # Quiet translucent wings. These are a glyph, not anatomical geometry.
    wing_center_l = center - 0.03 * forward + 0.12 * side
    wing_center_r = center - 0.03 * forward - 0.12 * side
    for wing_center, wing_angle in (
        (wing_center_l, angle_deg + 32.0),
        (wing_center_r, angle_deg - 32.0),
    ):
        ax.add_patch(
            Ellipse(
                wing_center,
                width=0.34,
                height=0.13,
                angle=wing_angle,
                facecolor="#E2E8F0",
                edgecolor="#94A3B8",
                linewidth=0.8,
                alpha=0.38,
                zorder=9,
            )
        )

    # Abdomen/thorax axis follows recorded heading.
    ax.add_patch(
        Ellipse(
            center - 0.07 * forward,
            width=0.38,
            height=0.18,
            angle=angle_deg,
            facecolor=color,
            edgecolor=TEXT,
            linewidth=1.25,
            zorder=10,
        )
    )
    head = center + 0.16 * forward
    ax.add_patch(
        Ellipse(
            head,
            width=0.17,
            height=0.16,
            angle=angle_deg,
            facecolor="#0F172A",
            edgecolor=TEXT,
            linewidth=1.1,
            zorder=11,
        )
    )

    # Left/right antenna brightness and reach are driven only by recorded odor.
    for sign, odor in ((1.0, left_odor), (-1.0, right_odor)):
        root = head + sign * 0.045 * side
        tip = head + (0.15 + 0.08 * odor) * forward + sign * (0.08 + 0.035 * odor) * side
        ax.plot(
            [root[0], tip[0]],
            [root[1], tip[1]],
            color=PLUME,
            linewidth=1.1 + 3.0 * odor,
            alpha=0.30 + 0.70 * odor,
            solid_capstyle="round",
            zorder=12,
        )


def _draw_room_v2(
    ax,
    payload: dict[str, Any],
    histories: dict[str, np.ndarray],
    frame_index: int,
    *,
    reveal: bool,
) -> None:
    arena = payload["arena"]
    current = payload["frames"][frame_index]
    controller_colors = {entry["label"]: entry["color"] for entry in payload["controllers"]}

    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, arena["width"])
    ax.set_ylim(0, arena["height"])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.5)

    for index, (x, y, tag) in enumerate(PEOPLE):
        _draw_person(
            ax,
            x,
            y,
            tag,
            color=PERSON_COLORS[index % len(PERSON_COLORS)],
            culprit=reveal and index == CULPRIT_INDEX,
        )

    plume = np.asarray(current.get("plume", []), dtype=float)
    if plume.size:
        # Recorded puff/sample points only. No smooth field is reconstructed here.
        ax.scatter(
            plume[:, 0],
            plume[:, 1],
            s=np.clip(68.0 * plume[:, 2], 6.0, 84.0),
            alpha=np.clip(0.20 + 0.28 * plume[:, 2], 0.20, 0.48),
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
            linewidth=4.2,
            alpha=0.82,
            zorder=6,
        )
        _draw_fly_marker_v2(ax, _agent(current, label), color)

    first = payload["controllers"][0]
    second = payload["controllers"][1]
    first_state = _agent(current, first["label"])
    second_state = _agent(current, second["label"])
    ax.text(
        0.022,
        0.972,
        "DEVELOPMENT PROXY",
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        color=first["color"],
        fontweight="bold",
        bbox=_badge_style(),
    )
    ax.text(
        0.978,
        0.972,
        "RANDOM CONTROL",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color=second["color"],
        fontweight="bold",
        bbox=_badge_style(),
    )
    ax.text(
        0.022,
        0.03,
        f"{current['t']:04.1f}s  •  proxy {first_state['distance_to_source']:.1f} m away",
        transform=ax.transAxes,
        va="bottom",
        fontsize=8.8,
        color=TEXT,
        fontweight="bold",
        bbox=_badge_style(alpha=0.84),
    )
    ax.text(
        0.978,
        0.03,
        f"control {second_state['distance_to_source']:.1f} m away",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color=TEXT,
        fontweight="bold",
        bbox=_badge_style(alpha=0.84),
    )


def _draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, alpha: float = 0.62) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.2,
            color="#94A3B8",
            alpha=alpha,
            zorder=2,
        )
    )


def _draw_node(ax, x: float, y: float, text: str, *, secondary: bool = False) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.7 if secondary else 8.4,
        color=TEXT,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#172033" if not secondary else "#111827",
            "edgecolor": "#64748B",
            "linewidth": 0.8,
            "alpha": 0.98,
        },
        zorder=4,
    )


def _recording_has_named_neural_state(payload: dict[str, Any]) -> bool:
    for frame in payload.get("frames", []):
        for agent in frame.get("agents", []):
            if agent.get("neural_state"):
                return True
    return False


def _draw_evidence_strip(
    ax,
    config: dict[str, Any],
    *,
    has_recorded_neural_state: bool,
) -> None:
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.02,
        0.89,
        config["measured_structure"]["label"],
        color="#C4B5FD",
        fontsize=9.2,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.89,
        "MaleCNS v1.0 • body-ID-resolved structural route",
        ha="right",
        color=MUTED,
        fontsize=7.3,
    )

    # Compact population-level view of exact measured edge families. It is a readability
    # projection of the sealed graph, never a new model or inferred connection.
    positions = {
        "odor": (0.07, 0.59),
        "wind": (0.07, 0.28),
        "hDeltaC": (0.30, 0.44),
        "hDeltaG": (0.49, 0.44),
        "PFL3": (0.68, 0.58),
        "PFL2": (0.68, 0.26),
        "DNa02": (0.91, 0.58),
    }
    for start, end in (
        (positions["odor"], positions["hDeltaC"]),
        (positions["wind"], positions["hDeltaC"]),
        (positions["hDeltaC"], positions["hDeltaG"]),
        (positions["hDeltaG"], positions["PFL3"]),
        (positions["hDeltaG"], positions["PFL2"]),
        (positions["PFL3"], positions["DNa02"]),
    ):
        _draw_arrow(ax, start, end)

    _draw_node(ax, *positions["odor"], "FB5AB\nodor context")
    _draw_node(ax, *positions["wind"], "PFNa/m/p\nwind prior")
    _draw_node(ax, *positions["hDeltaC"], "hΔC")
    _draw_node(ax, *positions["hDeltaG"], "hΔG")
    _draw_node(ax, *positions["PFL3"], "PFL3")
    _draw_node(ax, *positions["PFL2"], "PFL2", secondary=True)
    _draw_node(ax, *positions["DNa02"], "DNa02")

    if has_recorded_neural_state:
        status = "MODELED ACTIVITY • recorded neural state available"
        status_color = "#FDE68A"
    else:
        status = "MODELED ACTIVITY • integration animation locked until E002b/E002c recording"
        status_color = MUTED
    ax.text(
        0.02,
        0.06,
        status,
        color=status_color,
        fontsize=7.5,
        fontweight="bold",
    )


def _assert_canvas_dimensions(fig) -> None:
    width, height = fig.canvas.get_width_height()
    if (int(width), int(height)) != (SHOWCASE_WIDTH, SHOWCASE_HEIGHT):
        raise RuntimeError(
            "showcase canvas dimensions drifted: "
            f"observed={width}x{height} expected={SHOWCASE_WIDTH}x{SHOWCASE_HEIGHT}"
        )


def render_recorded_showcase_v2(
    recording: str | Path,
    output: str | Path,
    *,
    evidence_config: str | Path = DEFAULT_EVIDENCE_CONFIG,
    seconds: int = 16,
    fps: int = 30,
) -> Path:
    bundle = load_recording(recording)
    payload = bundle["recording"]
    _validate_social_contract(payload)
    config = _load_evidence_config(evidence_config)
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
    has_recorded_neural_state = _recording_has_named_neural_state(payload)

    fig = plt.figure(
        figsize=(SHOWCASE_WIDTH / SHOWCASE_DPI, SHOWCASE_HEIGHT / SHOWCASE_DPI),
        dpi=SHOWCASE_DPI,
        facecolor=BG,
    )
    grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[0.34, 1.66, 0.63, 0.61, 0.20],
        hspace=0.14,
    )
    title_ax = fig.add_subplot(grid[0, 0])
    room_ax = fig.add_subplot(grid[1, 0])
    sensor_ax = fig.add_subplot(grid[2, 0])
    evidence_ax = fig.add_subplot(grid[3, 0])
    footer_ax = fig.add_subplot(grid[4, 0])
    fig.subplots_adjust(left=0.035, right=0.965, top=0.99, bottom=0.022)
    _assert_canvas_dimensions(fig)

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
            0.23,
            "YOU CAN SEE THE SMELL • THE FLY CAN'T",
            ha="center",
            va="center",
            fontsize=11.7,
            color=MUTED,
            fontweight="bold",
        )

        _draw_room_v2(room_ax, payload, histories, record_index, reveal=reveal)
        _draw_sensor_hud(sensor_ax, first)
        _draw_evidence_strip(
            evidence_ax,
            config,
            has_recorded_neural_state=has_recorded_neural_state,
        )

        footer_ax.clear()
        footer_ax.set_facecolor(BG)
        footer_ax.axis("off")
        if reveal:
            footer_ax.text(
                0.5,
                0.70,
                f"BUSTED: {PEOPLE[CULPRIT_INDEX][2]}",
                ha="center",
                va="center",
                fontsize=15.5,
                color="#D9F99D",
                fontweight="bold",
            )
            footer_ax.text(
                0.5,
                0.20,
                outcome,
                ha="center",
                va="center",
                fontsize=8.7,
                color=TEXT,
                fontweight="bold",
            )
        else:
            footer_ax.text(
                0.5,
                0.68,
                config["allowed_science_caption_now"],
                ha="center",
                va="center",
                fontsize=8.6,
                color="#FBBF24",
                fontweight="bold",
            )
            footer_ax.text(
                0.5,
                0.20,
                "STRUCTURE RESOLVED • BEHAVIOR STILL UNDER TEST",
                ha="center",
                va="center",
                fontsize=8.4,
                color=MUTED,
                fontweight="bold",
            )
        footer_ax.text(
            0.99,
            0.02,
            f"replay {bundle['recording_sha256'][:12]}",
            ha="right",
            va="bottom",
            fontsize=6.4,
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
            ani.save(output, writer=animation.PillowWriter(fps=fps))
        else:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError("ffmpeg is required for MP4 output")
            ani.save(
                output,
                writer=animation.FFMpegWriter(
                    fps=fps,
                    bitrate=7600,
                    extra_args=["-pix_fmt", "yuv420p"],
                ),
            )
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the evidence-gated Who Farted? recorded showcase v2"
    )
    parser.add_argument("recording")
    parser.add_argument(
        "--evidence-config",
        default=str(DEFAULT_EVIDENCE_CONFIG),
    )
    parser.add_argument(
        "--output",
        default="artifacts/showcase/who-farted-recorded-v2.mp4",
    )
    parser.add_argument("--seconds", type=int, default=16)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(
        render_recorded_showcase_v2(
            args.recording,
            args.output,
            evidence_config=args.evidence_config,
            seconds=args.seconds,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    main()
