from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .party_social import BG, CULPRIT_INDEX, MUTED, PANEL, PEOPLE, TEXT
from .recorded_showcase import (
    _agent,
    _draw_sensor_hud,
    _outcome_text,
    _precompute_histories,
    _validate_social_contract,
)
from .recorded_showcase_v2 import _assert_canvas_dimensions, _draw_arrow, _draw_node, _draw_room_v2
from .recording import load_recording
from .showcase import SHOWCASE_DPI, SHOWCASE_HEIGHT, SHOWCASE_WIDTH

DEFAULT_EVIDENCE_CONFIG = Path("configs/showcase_evidence_v3.json")
DEFAULT_E002C = Path("results/e002/pfl3-convergence-v1.json")
DEFAULT_FC2 = Path("results/route/fc2-goal-interface-audit-v1.json")


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_inputs(
    config: dict[str, Any],
    e002c: dict[str, Any],
    fc2: dict[str, Any],
    *,
    e002c_sha256: str | None = None,
    fc2_sha256: str | None = None,
) -> None:
    if config.get("protocol") != "who-farted-showcase-evidence-v3":
        raise ValueError("showcase v3 requires who-farted-showcase-evidence-v3 config")
    if config.get("comparison", {}).get("real_vs_rewire_headline_allowed", False):
        raise ValueError("showcase v3 cannot enable real-vs-rewire headline")
    if config.get("behavioral_state", {}).get("malecns_behavior_claim_allowed", False):
        raise ValueError("showcase v3 cannot enable a MaleCNS behavioral claim")
    if config.get("modeled_activity", {}).get("behavior_sync_allowed", True):
        raise ValueError("showcase v3 mechanism activity must remain separate from chase timeline")

    mechanism = config.get("mechanism_probe", {})
    expected_e002c_sha = mechanism.get("e002c_sha256")
    if expected_e002c_sha and e002c_sha256 != expected_e002c_sha:
        raise ValueError("showcase v3 E002c artifact hash mismatch")
    expected_fc2_sha = mechanism.get("fc2_sha256")
    if expected_fc2_sha and fc2_sha256 != expected_fc2_sha:
        raise ValueError("showcase v3 FC2 artifact hash mismatch")

    if e002c.get("protocol") != mechanism.get("e002c_protocol"):
        raise ValueError("unexpected E002c protocol")
    if mechanism.get("e002c_must_pass", True) and not bool(e002c.get("passed")):
        raise ValueError("showcase v3 requires passed E002c report")
    gates = {str(row.get("name")): bool(row.get("passed")) for row in e002c.get("gates", [])}
    if mechanism.get("e002b_required", True) and not gates.get("e002b_qualified", False):
        raise ValueError("showcase v3 requires E002b-qualified E002c input")

    if fc2.get("protocol") != mechanism.get("fc2_protocol"):
        raise ValueError("unexpected FC2 goal-interface protocol")
    if (
        mechanism.get("phase_mapping_must_remain_unresolved", True)
        and fc2.get("phase_mapping_status") != "unresolved"
    ):
        raise ValueError("showcase v3 must not silently promote FC2 columns to goal angles")

    expected_thresholds = [str(int(x)) for x in mechanism.get("all_thresholds_must_be_reported", [])]
    observed_thresholds = {str(key) for key in e002c.get("threshold_reports", {})}
    missing = [key for key in expected_thresholds if key not in observed_thresholds]
    if missing:
        raise ValueError(f"E002c report missing preregistered thresholds: {missing}")

    for family in ("FC2A", "FC2B", "FC2C"):
        population = fc2.get("populations", {}).get(family, {})
        columns = population.get("instance_columns", {})
        if float(columns.get("parse_fraction", 0.0)) != 1.0:
            raise ValueError(f"{family} instance columns are not fully resolved")
        threshold_10 = (
            fc2.get("interfaces", {})
            .get(family, {})
            .get("threshold_reports", {})
            .get("10", {})
        )
        if threshold_10.get("target_coverage") != "24/24":
            raise ValueError(f"{family}->PFL3 does not retain 24/24 target coverage at threshold 10")


def _trace_payload(e002c: dict[str, Any], threshold: int) -> dict[str, np.ndarray]:
    report = e002c["threshold_reports"][str(int(threshold))]
    traces: dict[str, np.ndarray] = {}
    for name in ("goal_only", "heading_only", "joint"):
        activity = np.asarray(report[name]["activity"], dtype=float)
        if activity.ndim != 2 or activity.shape[1] != 24:
            raise ValueError(f"{name} PFL3 activity must have shape (steps, 24)")
        traces[name] = np.mean(np.abs(activity), axis=1)
    return traces


def _threshold_summary(e002c: dict[str, Any]) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for key in sorted(e002c["threshold_reports"], key=lambda value: float(value)):
        convergence = e002c["threshold_reports"][key]["joint_convergence"]
        rows.append(
            (
                int(float(key)),
                int(convergence["changed_from_both_count"]),
                int(convergence["dual_reachable_count"]),
            )
        )
    return rows


def _draw_mechanism_panel(
    ax,
    config: dict[str, Any],
    e002c: dict[str, Any],
    probe_index: int,
) -> None:
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    mechanism = config["mechanism_probe"]
    threshold = int(mechanism["display_threshold"])
    traces = _trace_payload(e002c, threshold)
    steps = len(traces["joint"])
    probe_index = int(np.clip(probe_index, 0, steps - 1))

    ax.text(
        0.02,
        0.93,
        mechanism["label"],
        color="#FDE68A",
        fontsize=9.0,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.93,
        "MODELED ACTIVITY • NOT MEASURED FIRING",
        ha="right",
        color=MUTED,
        fontsize=7.0,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.81,
        "MEASURED INPUT TOPOLOGY",
        color="#C4B5FD",
        fontsize=7.3,
        fontweight="bold",
    )
    fc2_pos = (0.08, 0.55)
    epg_pos = (0.08, 0.27)
    pfl3_pos = (0.34, 0.41)
    _draw_arrow(ax, fc2_pos, pfl3_pos, alpha=0.72)
    _draw_arrow(ax, epg_pos, pfl3_pos, alpha=0.72)
    _draw_node(ax, *fc2_pos, "FC2 A/B/C\ncolumns 1–9")
    _draw_node(ax, *epg_pos, "EPG\nPB heading")
    _draw_node(ax, *pfl3_pos, "PFL3\n24 cells")

    x0, x1 = 0.50, 0.98
    y0, y1 = 0.22, 0.70
    all_values = np.concatenate(list(traces.values()))
    scale = max(float(np.max(all_values)), 1e-12)
    xs = np.linspace(x0, x1, steps)
    styles = {
        "goal_only": ("goal", "#C4B5FD"),
        "heading_only": ("heading", "#67E8F9"),
        "joint": ("joint", "#FDE68A"),
    }
    ax.text(
        x0,
        0.81,
        f"E002c 8/8 • fixed display w≥{threshold} • step {probe_index + 1}/{steps}",
        color=TEXT,
        fontsize=7.0,
        fontweight="bold",
    )
    legend_x = x0
    for name in ("goal_only", "heading_only", "joint"):
        label, color = styles[name]
        ax.text(legend_x, 0.74, label, color=color, fontsize=6.7, fontweight="bold")
        legend_x += 0.095
        ys = y0 + (traces[name] / scale) * (y1 - y0)
        ax.plot(xs, ys, color=color, linewidth=1.7, alpha=0.92)
        ax.scatter([xs[probe_index]], [ys[probe_index]], s=18, color=color, zorder=5)

    ax.axvline(
        xs[probe_index],
        ymin=y0,
        ymax=y1,
        color="#E2E8F0",
        linewidth=0.8,
        alpha=0.38,
    )
    ax.text(
        0.02,
        0.06,
        "FC2A/B/C: 24/24 PFL3 @ w≥10 • columns 1–9 parsed 100%",
        color="#C4B5FD",
        fontsize=6.5,
        fontweight="bold",
    )
    badges = " • ".join(
        f"≥{threshold_value} {changed}/{reachable}"
        for threshold_value, changed, reachable in _threshold_summary(e002c)
    )
    ax.text(
        0.98,
        0.06,
        "convergence " + badges,
        ha="right",
        color=MUTED,
        fontsize=6.2,
        fontweight="bold",
    )


def render_recorded_showcase_v3(
    recording: str | Path,
    output: str | Path,
    *,
    e002c_report: str | Path = DEFAULT_E002C,
    fc2_audit: str | Path = DEFAULT_FC2,
    evidence_config: str | Path = DEFAULT_EVIDENCE_CONFIG,
    seconds: int = 16,
    fps: int = 30,
) -> Path:
    bundle = load_recording(recording)
    payload = bundle["recording"]
    _validate_social_contract(payload)
    config = _load_json(evidence_config)
    e002c = _load_json(e002c_report)
    fc2 = _load_json(fc2_audit)
    _validate_inputs(
        config,
        e002c,
        fc2,
        e002c_sha256=_sha256_file(e002c_report),
        fc2_sha256=_sha256_file(fc2_audit),
    )
    if seconds < 1 or fps < 1:
        raise ValueError("seconds and fps must be >= 1")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recorded_frames = payload["frames"]
    histories = _precompute_histories(payload)
    video_frames = max(1, seconds * fps)
    first_label = payload["controllers"][0]["label"]
    second_label = payload["controllers"][1]["label"]
    outcome = f"{_outcome_text(payload, first_label)}   •   {_outcome_text(payload, second_label)}"
    probe_steps = int(e002c["run_config"]["steps"])

    fig = plt.figure(
        figsize=(SHOWCASE_WIDTH / SHOWCASE_DPI, SHOWCASE_HEIGHT / SHOWCASE_DPI),
        dpi=SHOWCASE_DPI,
        facecolor=BG,
    )
    grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[0.30, 1.48, 0.52, 0.92, 0.20],
        hspace=0.12,
    )
    title_ax = fig.add_subplot(grid[0, 0])
    room_ax = fig.add_subplot(grid[1, 0])
    sensor_ax = fig.add_subplot(grid[2, 0])
    mechanism_ax = fig.add_subplot(grid[3, 0])
    footer_ax = fig.add_subplot(grid[4, 0])
    fig.subplots_adjust(left=0.035, right=0.965, top=0.99, bottom=0.022)
    _assert_canvas_dimensions(fig)

    def draw(video_index: int):
        fraction = 1.0 if video_frames == 1 else video_index / (video_frames - 1)
        record_index = round(fraction * (len(recorded_frames) - 1))
        probe_index = round(fraction * (probe_steps - 1))
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
            0.22,
            "YOU CAN SEE THE SMELL • THE FLY CAN'T",
            ha="center",
            va="center",
            fontsize=11.4,
            color=MUTED,
            fontweight="bold",
        )

        _draw_room_v2(room_ax, payload, histories, record_index, reveal=reveal)
        _draw_sensor_hud(sensor_ax, first)
        _draw_mechanism_panel(mechanism_ax, config, e002c, probe_index)

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
                "MECHANISM PROBE QUALIFIED • CHASE STILL DEVELOPMENT-ONLY",
                ha="center",
                va="center",
                fontsize=8.0,
                color=MUTED,
                fontweight="bold",
            )
        footer_ax.text(
            0.99,
            0.02,
            f"replay {bundle['recording_sha256'][:12]}",
            ha="right",
            va="bottom",
            fontsize=6.2,
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
        description="Render the evidence-gated Who Farted? rapid mechanism teaser v3"
    )
    parser.add_argument("recording")
    parser.add_argument("--e002c-report", default=str(DEFAULT_E002C))
    parser.add_argument("--fc2-audit", default=str(DEFAULT_FC2))
    parser.add_argument("--evidence-config", default=str(DEFAULT_EVIDENCE_CONFIG))
    parser.add_argument("--output", default="artifacts/showcase/who-farted-rapid-v3.mp4")
    parser.add_argument("--seconds", type=int, default=16)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(
        render_recorded_showcase_v3(
            args.recording,
            args.output,
            e002c_report=args.e002c_report,
            fc2_audit=args.fc2_audit,
            evidence_config=args.evidence_config,
            seconds=args.seconds,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    main()
