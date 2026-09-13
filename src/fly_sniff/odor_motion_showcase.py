from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .odor_events import load_diagnostics
from .odor_motion import load_analysis
from .party_social import BG, MUTED, PANEL, TEXT
from .recorded_showcase import (
    _agent,
    _draw_room,
    _precompute_histories,
    _validate_social_contract,
)
from .recording import load_recording
from .showcase import SHOWCASE_DPI, SHOWCASE_HEIGHT, SHOWCASE_WIDTH


@dataclass(frozen=True)
class ReplayContext:
    recording_bundle: dict[str, Any]
    motion_bundle: dict[str, Any]
    diagnostics_bundle: dict[str, Any]
    label: str
    motion_samples: list[dict[str, Any]]
    events: list[dict[str, Any]]


def _agent_entry(payload: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [item for item in payload.get("agents", []) if item.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one diagnostics agent labelled {label!r}")
    return matches[0]


def load_replay_context(
    recording: str | Path,
    analysis: str | Path,
    diagnostics: str | Path,
    *,
    label: str | None = None,
) -> ReplayContext:
    recording_bundle = load_recording(recording)
    motion_bundle = load_analysis(analysis)
    diagnostics_bundle = load_diagnostics(diagnostics)
    recording_payload = recording_bundle["recording"]
    _validate_social_contract(recording_payload)
    motion_payload = motion_bundle["analysis"]
    diagnostics_payload = diagnostics_bundle["diagnostics"]
    recording_sha = recording_bundle["recording_sha256"]
    motion_sha = motion_bundle["analysis_sha256"]

    if motion_payload["source_recording_sha256"] != recording_sha:
        raise ValueError("odor-motion analysis does not belong to this recording")
    if diagnostics_payload["source_recording_sha256"] != recording_sha:
        raise ValueError("odor-event diagnostics do not belong to this recording")
    if diagnostics_payload["source_odor_motion_sha256"] != motion_sha:
        raise ValueError("odor-event diagnostics do not belong to this odor-motion analysis")

    labels = [str(item["label"]) for item in recording_payload["controllers"]]
    selected = label or labels[0]
    if selected not in labels:
        raise ValueError(f"recording has no controller labelled {selected!r}")
    motion_agent = _agent_entry(motion_payload, selected)
    diagnostics_agent = _agent_entry(diagnostics_payload, selected)
    samples = motion_agent["samples"]
    if len(samples) != len(recording_payload["frames"]):
        raise ValueError("odor-motion sample count does not match recording frame count")

    return ReplayContext(
        recording_bundle=recording_bundle,
        motion_bundle=motion_bundle,
        diagnostics_bundle=diagnostics_bundle,
        label=selected,
        motion_samples=samples,
        events=diagnostics_agent["events"],
    )


def _recent_event(
    events: list[dict[str, Any]],
    index: int,
    *,
    hold_steps: int,
) -> dict[str, Any] | None:
    eligible = [
        event
        for event in events
        if event["kind"] in {"encounter", "loss", "reacquisition"}
        and 0 <= index - int(event["index"]) <= hold_steps
    ]
    return eligible[-1] if eligible else None


def _draw_timing_hud(
    ax,
    context: ReplayContext,
    frame_index: int,
    *,
    window_s: float = 2.5,
) -> None:
    samples = context.motion_samples
    recording_payload = context.recording_bundle["recording"]
    dt = float(recording_payload["dt"])
    current = samples[frame_index]
    window_steps = max(2, round(window_s / dt))
    start = max(0, frame_index - window_steps + 1)
    view = samples[start : frame_index + 1]

    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.025,
        0.93,
        "MODELED BILATERAL ODOR TIMING",
        color=TEXT,
        fontsize=11.5,
        fontweight="bold",
    )
    ax.text(
        0.975,
        0.93,
        "engineered causal comparator • not measured neural activity",
        color=MUTED,
        fontsize=7.8,
        ha="right",
    )

    trace_ax = ax.inset_axes([0.04, 0.16, 0.64, 0.66])
    trace_ax.set_facecolor("#0B1322")
    times = np.asarray([float(item["t"]) for item in view], dtype=float)
    left = np.asarray([float(item["left_response"]) for item in view], dtype=float)
    right = np.asarray([float(item["right_response"]) for item in view], dtype=float)
    evidence = np.asarray([float(item["evidence"]) for item in view], dtype=float)
    trace_ax.plot(times, left, linewidth=1.5, label="L antenna", color="#67E8F9")
    trace_ax.plot(times, right, linewidth=1.5, label="R antenna", color="#C4B5FD")
    trace_ax.plot(
        times,
        0.5 + 0.45 * evidence,
        linewidth=1.2,
        label="timing evidence",
        color="#F8FAFC",
        alpha=0.9,
    )
    trace_ax.axhline(0.5, linewidth=0.8, color="#64748B", alpha=0.5)
    trace_ax.set_ylim(-0.03, 1.03)
    if len(times) > 1:
        trace_ax.set_xlim(float(times[0]), float(times[-1]))
    trace_ax.tick_params(colors="#94A3B8", labelsize=6)
    for spine in trace_ax.spines.values():
        spine.set_color("#334155")
    trace_ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=6.8,
        labelcolor="#CBD5E1",
        ncol=3,
    )

    direction = str(current["direction"])
    direction_text = {
        "left_to_right": "LEFT → RIGHT",
        "right_to_left": "RIGHT → LEFT",
        "ambiguous": "AMBIGUOUS",
        "no_signal": "NO SIGNAL",
        "insufficient_history": "WARMING UP",
    }.get(direction, direction.upper())
    ax.text(
        0.73,
        0.72,
        "ODOR MOTION",
        color=MUTED,
        fontsize=8.2,
        fontweight="bold",
    )
    ax.text(
        0.73,
        0.62,
        direction_text,
        color=TEXT,
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.73,
        0.48,
        f"evidence {float(current['evidence']):+.2f}",
        color="#E2E8F0",
        fontsize=9,
    )
    ax.text(
        0.73,
        0.38,
        f"confidence {float(current['confidence']):.2f}",
        color="#FDE68A",
        fontsize=9,
    )
    delay = current["dominant_delay_s"]
    delay_text = "—" if delay is None else f"{1000.0 * float(delay):.0f} ms"
    ax.text(
        0.73,
        0.28,
        f"dominant lag {delay_text}",
        color="#CBD5E1",
        fontsize=8.4,
    )

    event = _recent_event(
        context.events,
        frame_index,
        hold_steps=max(1, round(0.55 / dt)),
    )
    if event is not None:
        banner = {
            "encounter": "ODOR ENCOUNTER",
            "loss": "ODOR LOST",
            "reacquisition": "PLUME REACQUIRED",
        }[str(event["kind"])]
        ax.text(
            0.73,
            0.12,
            banner,
            color="#F8FAFC",
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": BG,
                "edgecolor": "#475569",
                "alpha": 0.92,
            },
        )


def _draw_title(ax, claim_boundary: str) -> None:
    ax.clear()
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.text(
        0.5,
        0.70,
        "WHO FARTED?",
        ha="center",
        va="center",
        fontsize=38,
        color=TEXT,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.28,
        "smell visible to you • source hidden from the fly",
        ha="center",
        va="center",
        fontsize=11.5,
        color=MUTED,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.01,
        claim_boundary,
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#FBBF24",
        fontweight="bold",
    )


def _draw_footer(ax, context: ReplayContext, frame_index: int) -> None:
    ax.clear()
    ax.set_facecolor(BG)
    ax.axis("off")
    payload = context.recording_bundle["recording"]
    state = _agent(payload["frames"][frame_index], context.label)
    ax.text(
        0.01,
        0.58,
        (
            f"{context.label} • {float(payload['frames'][frame_index]['t']):.1f}s • "
            f"{float(state['distance_to_source']):.2f} m from source"
        ),
        color=TEXT,
        fontsize=8.5,
        fontweight="bold",
    )
    receipt = (
        f"rec {context.recording_bundle['recording_sha256'][:10]} • "
        f"timing {context.motion_bundle['analysis_sha256'][:10]} • "
        f"events {context.diagnostics_bundle['diagnostics_sha256'][:10]}"
    )
    ax.text(
        0.99,
        0.12,
        receipt,
        ha="right",
        color="#64748B",
        fontsize=6.5,
    )


def _build_figure():
    fig = plt.figure(
        figsize=(SHOWCASE_WIDTH / SHOWCASE_DPI, SHOWCASE_HEIGHT / SHOWCASE_DPI),
        dpi=SHOWCASE_DPI,
        facecolor=BG,
    )
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.34, 1.82, 0.82, 0.20],
        hspace=0.16,
    )
    title_ax = fig.add_subplot(grid[0, 0])
    room_ax = fig.add_subplot(grid[1, 0])
    timing_ax = fig.add_subplot(grid[2, 0])
    footer_ax = fig.add_subplot(grid[3, 0])
    fig.subplots_adjust(left=0.035, right=0.965, top=0.99, bottom=0.025)
    return fig, title_ax, room_ax, timing_ax, footer_ax


def render_temporal_showcase_frame(
    recording: str | Path,
    analysis: str | Path,
    diagnostics: str | Path,
    output: str | Path,
    *,
    frame_index: int = -1,
    label: str | None = None,
) -> Path:
    context = load_replay_context(
        recording,
        analysis,
        diagnostics,
        label=label,
    )
    payload = context.recording_bundle["recording"]
    frames = payload["frames"]
    resolved_index = frame_index if frame_index >= 0 else len(frames) + frame_index
    if not 0 <= resolved_index < len(frames):
        raise IndexError("frame_index is outside the recording")
    histories = _precompute_histories(payload)
    fig, title_ax, room_ax, timing_ax, footer_ax = _build_figure()
    try:
        _draw_title(title_ax, payload["claim_boundary"])
        _draw_room(
            room_ax,
            payload,
            histories,
            resolved_index,
            reveal=False,
        )
        _draw_timing_hud(timing_ax, context, resolved_index)
        _draw_footer(footer_ax, context, resolved_index)
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, facecolor=fig.get_facecolor())
        return destination
    finally:
        plt.close(fig)


def render_temporal_showcase(
    recording: str | Path,
    analysis: str | Path,
    diagnostics: str | Path,
    output: str | Path,
    *,
    seconds: int = 15,
    fps: int = 30,
    label: str | None = None,
) -> Path:
    if seconds < 1 or fps < 1:
        raise ValueError("seconds and fps must be >= 1")
    context = load_replay_context(
        recording,
        analysis,
        diagnostics,
        label=label,
    )
    payload = context.recording_bundle["recording"]
    frames = payload["frames"]
    histories = _precompute_histories(payload)
    video_frames = max(1, seconds * fps)
    fig, title_ax, room_ax, timing_ax, footer_ax = _build_figure()

    def draw(video_index: int):
        if video_frames == 1:
            frame_index = len(frames) - 1
        else:
            fraction = video_index / (video_frames - 1)
            frame_index = round(fraction * (len(frames) - 1))
        _draw_title(title_ax, payload["claim_boundary"])
        state = _agent(frames[frame_index], context.label)
        timed_reveal = video_index >= max(0, video_frames - 2 * fps)
        reveal = bool(state["found"] or timed_reveal)
        _draw_room(
            room_ax,
            payload,
            histories,
            frame_index,
            reveal=reveal,
        )
        _draw_timing_hud(timing_ax, context, frame_index)
        _draw_footer(footer_ax, context, frame_index)
        return []

    ani = animation.FuncAnimation(
        fig,
        draw,
        frames=video_frames,
        interval=1000 / fps,
        blit=False,
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if destination.suffix.lower() == ".gif":
            ani.save(destination, writer=animation.PillowWriter(fps=fps))
        else:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError(
                    "ffmpeg is required for MP4 output; render GIF or install ffmpeg"
                )
            ani.save(
                destination,
                writer=animation.FFMpegWriter(fps=fps, bitrate=6500),
            )
    finally:
        plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the Who Farted? room with hash-bound bilateral odor timing "
            "and event diagnostics"
        )
    )
    parser.add_argument("recording")
    parser.add_argument("analysis")
    parser.add_argument("diagnostics")
    parser.add_argument(
        "--output",
        default="artifacts/showcase/who-farted-odor-motion.mp4",
    )
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--label")
    parser.add_argument(
        "--frame",
        type=int,
        help="Render one zero-based recording frame as a still image instead of animation",
    )
    args = parser.parse_args()
    if args.frame is not None:
        print(
            render_temporal_showcase_frame(
                args.recording,
                args.analysis,
                args.diagnostics,
                args.output,
                frame_index=args.frame,
                label=args.label,
            )
        )
    else:
        print(
            render_temporal_showcase(
                args.recording,
                args.analysis,
                args.diagnostics,
                args.output,
                seconds=args.seconds,
                fps=args.fps,
                label=args.label,
            )
        )


if __name__ == "__main__":
    main()
