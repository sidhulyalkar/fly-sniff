from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation

from .party_social import BG, CULPRIT_INDEX, MUTED, PEOPLE, TEXT
from .recorded_showcase import (
    _agent,
    _draw_sensor_hud,
    _outcome_text,
    _precompute_histories,
    _validate_social_contract,
)
from .recorded_showcase_v2 import _assert_canvas_dimensions
from .recorded_showcase_v3 import _load_json, _validate_inputs
from .recording import load_recording
from .showcase import SHOWCASE_DPI, SHOWCASE_HEIGHT, SHOWCASE_WIDTH
from .showcase_v4_draw import draw_population_panel, draw_room_v4
from .showcase_v4_layers import social_display_frame_limit


def render_v4(
    recording: str | Path,
    output: str | Path,
    *,
    e002c_report: str | Path,
    fc2_audit: str | Path,
    evidence_config: str | Path,
    seconds: int = 16,
    fps: int = 30,
) -> Path:
    bundle = load_recording(recording)
    payload = bundle["recording"]
    _validate_social_contract(payload)
    config = _load_json(evidence_config)
    e002c = _load_json(e002c_report)
    fc2 = _load_json(fc2_audit)
    _validate_inputs(config, e002c, fc2)
    if seconds < 1 or fps < 1:
        raise ValueError("seconds and fps must be >= 1")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = payload["frames"]
    display_last_index = social_display_frame_limit(payload)
    histories = _precompute_histories(payload)
    n_video = max(1, seconds * fps)
    first_label = payload["controllers"][0]["label"]
    second_label = payload["controllers"][1]["label"]
    outcome = f"{_outcome_text(payload, first_label)} • {_outcome_text(payload, second_label)}"
    probe_steps = int(e002c["run_config"]["steps"])

    fig = plt.figure(
        figsize=(SHOWCASE_WIDTH / SHOWCASE_DPI, SHOWCASE_HEIGHT / SHOWCASE_DPI),
        dpi=SHOWCASE_DPI,
        facecolor=BG,
    )
    grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[0.29, 1.50, 0.49, 0.88, 0.20],
        hspace=0.11,
    )
    title_ax, room_ax, sensor_ax, mechanism_ax, footer_ax = [
        fig.add_subplot(grid[i, 0]) for i in range(5)
    ]
    fig.subplots_adjust(left=0.035, right=0.965, top=0.99, bottom=0.022)
    _assert_canvas_dimensions(fig)

    def draw(video_index: int):
        fraction = 1.0 if n_video == 1 else video_index / (n_video - 1)
        record_index = round(fraction * display_last_index)
        probe_index = round(fraction * (probe_steps - 1))
        current = frames[record_index]
        first = _agent(current, first_label)
        second = _agent(current, second_label)
        reveal = bool(
            first["found"]
            or second["found"]
            or video_index >= max(0, n_video - 2 * fps)
        )

        title_ax.clear()
        title_ax.set_facecolor(BG)
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.70,
            config["allowed_headline_now"],
            ha="center",
            va="center",
            fontsize=38,
            color=TEXT,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.22,
            "YOU CAN SEE THE SMELL • THE FLY CAN'T",
            ha="center",
            va="center",
            fontsize=11.4,
            color=MUTED,
            fontweight="bold",
        )
        draw_room_v4(room_ax, payload, histories, record_index, reveal=reveal)
        _draw_sensor_hud(sensor_ax, first)
        draw_population_panel(mechanism_ax, config, e002c, fc2, probe_index)

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
                fontsize=15.0,
                color="#D9F99D",
                fontweight="bold",
            )
            footer_ax.text(
                0.5,
                0.18,
                outcome,
                ha="center",
                va="center",
                fontsize=8.3,
                color=TEXT,
                fontweight="bold",
            )
        else:
            footer_ax.text(
                0.5,
                0.66,
                config["allowed_science_caption_now"],
                ha="center",
                va="center",
                fontsize=8.2,
                color="#FBBF24",
                fontweight="bold",
            )
            footer_ax.text(
                0.5,
                0.16,
                "SEALED MECHANISM PROBE • CHASE STILL DEVELOPMENT-ONLY",
                ha="center",
                va="center",
                fontsize=8.0,
                color=MUTED,
                fontweight="bold",
            )
        return []

    ani = animation.FuncAnimation(
        fig,
        draw,
        frames=n_video,
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
