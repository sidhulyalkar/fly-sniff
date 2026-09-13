from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .odor_events import load_diagnostics
from .odor_motion import load_analysis
from .recording import load_recording


def _analysis_agent(payload: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [agent for agent in payload.get("agents", []) if agent.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"odor-motion sidecar must contain exactly one agent labelled {label!r}")
    return matches[0]


def _diagnostics_agent(payload: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [agent for agent in payload.get("agents", []) if agent.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"odor-event diagnostics must contain exactly one agent labelled {label!r}")
    return matches[0]


def _event_color(kind: str) -> str:
    return {
        "encounter": "#86EFAC",
        "loss": "#FB7185",
        "reacquisition": "#FBBF24",
    }.get(kind, "#CBD5E1")


def render_odor_motion_diagnostic(
    recording: str | Path,
    analysis: str | Path,
    output: str | Path,
    *,
    label: str | None = None,
    diagnostics: str | Path | None = None,
) -> Path:
    recording_bundle = load_recording(recording)
    motion_bundle = load_analysis(analysis)
    recording_payload = recording_bundle["recording"]
    motion_payload = motion_bundle["analysis"]
    if motion_payload["source_recording_sha256"] != recording_bundle["recording_sha256"]:
        raise ValueError("odor-motion sidecar does not belong to this recording")

    controller_labels = [str(item["label"]) for item in recording_payload["controllers"]]
    selected_label = label or controller_labels[0]
    if selected_label not in controller_labels:
        raise ValueError(f"recording has no controller labelled {selected_label!r}")
    agent = _analysis_agent(motion_payload, selected_label)
    samples = agent["samples"]
    if len(samples) != len(recording_payload["frames"]):
        raise ValueError("odor-motion sample count does not match source recording")

    event_bundle: dict[str, Any] | None = None
    event_agent: dict[str, Any] | None = None
    if diagnostics is not None:
        event_bundle = load_diagnostics(diagnostics)
        event_payload = event_bundle["diagnostics"]
        if event_payload["source_recording_sha256"] != recording_bundle["recording_sha256"]:
            raise ValueError("odor-event diagnostics do not belong to this recording")
        if event_payload["source_odor_motion_sha256"] != motion_bundle["analysis_sha256"]:
            raise ValueError("odor-event diagnostics do not belong to this odor-motion analysis")
        event_agent = _diagnostics_agent(event_payload, selected_label)

    t = np.asarray([float(item["t"]) for item in samples])
    left = np.asarray([float(item["left_response"]) for item in samples])
    right = np.asarray([float(item["right_response"]) for item in samples])
    evidence = np.asarray([float(item["evidence"]) for item in samples])
    confidence = np.asarray([float(item["confidence"]) for item in samples])

    fig = plt.figure(figsize=(10, 7), facecolor="#07111F")
    grid = fig.add_gridspec(3, 1, height_ratios=[0.22, 1.0, 1.0], hspace=0.22)
    title_ax = fig.add_subplot(grid[0, 0])
    antenna_ax = fig.add_subplot(grid[1, 0])
    motion_ax = fig.add_subplot(grid[2, 0])

    title_ax.set_facecolor("#07111F")
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.72,
        "MODELED BILATERAL ODOR TIMING",
        color="#F8FAFC",
        fontsize=19,
        fontweight="bold",
    )
    subtitle = f"{selected_label} • causal replay diagnostic • not measured neural activity"
    if event_agent is not None:
        summary = event_agent["summary"]
        fraction = summary["reacquisition_within_horizon_fraction"]
        reacquisition_text = "n/a" if fraction is None else f"{100.0 * float(fraction):.0f}%"
        subtitle += (
            f" • losses {summary['loss_count']} • "
            f"reacquired≤horizon {reacquisition_text}"
        )
    title_ax.text(
        0.0,
        0.18,
        subtitle,
        color="#94A3B8",
        fontsize=10,
    )

    for ax in (antenna_ax, motion_ax):
        ax.set_facecolor("#0F172A")
        ax.tick_params(colors="#CBD5E1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(alpha=0.15)

    antenna_ax.plot(t, left, label="LEFT ANTENNA", color="#67E8F9", linewidth=1.8)
    antenna_ax.plot(t, right, label="RIGHT ANTENNA", color="#C4B5FD", linewidth=1.8)
    antenna_ax.set_ylabel("modeled response", color="#E2E8F0")
    antenna_ax.set_ylim(
        -0.02,
        max(1.02, float(max(left.max(initial=0.0), right.max(initial=0.0))) * 1.05),
    )
    antenna_ax.legend(loc="upper right", frameon=False, labelcolor="#E2E8F0")

    motion_ax.axhline(0.0, color="#64748B", linewidth=1.0)
    motion_ax.plot(t, evidence, label="L→R  + / R→L  −", color="#F8FAFC", linewidth=1.8)
    motion_ax.plot(t, confidence, label="CONFIDENCE", color="#FBBF24", linewidth=1.3)
    motion_ax.set_ylim(-1.05, 1.05)
    motion_ax.set_xlabel("recording time (s)", color="#E2E8F0")
    motion_ax.set_ylabel("timing evidence", color="#E2E8F0")
    motion_ax.legend(loc="upper right", frameon=False, labelcolor="#E2E8F0")

    if event_agent is not None:
        event_label_seen: set[str] = set()
        for event in event_agent["events"]:
            kind = str(event["kind"])
            if kind not in {"encounter", "loss", "reacquisition"}:
                continue
            event_t = float(event["t"])
            color = _event_color(kind)
            label_text = kind.upper() if kind not in event_label_seen else None
            antenna_ax.axvline(
                event_t,
                color=color,
                linewidth=1.2,
                alpha=0.85,
                label=label_text,
            )
            motion_ax.axvline(
                event_t,
                color=color,
                linewidth=1.0,
                alpha=0.55,
            )
            event_label_seen.add(kind)
        if event_label_seen:
            antenna_ax.legend(loc="upper right", frameon=False, labelcolor="#E2E8F0")

    footer_parts = [
        f"recording {recording_bundle['recording_sha256'][:12]}",
        f"analysis {motion_bundle['analysis_sha256'][:12]}",
    ]
    if event_bundle is not None:
        footer_parts.append(f"events {event_bundle['diagnostics_sha256'][:12]}")
    fig.text(
        0.99,
        0.012,
        " • ".join(footer_parts),
        ha="right",
        color="#64748B",
        fontsize=7,
    )
    fig.text(
        0.01,
        0.012,
        "No source coordinates • no plume pixels • no future samples in causal timing estimate",
        ha="left",
        color="#64748B",
        fontsize=7,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.08)

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an auditable bilateral odor-timing diagnostic from a recording sidecar"
    )
    parser.add_argument("recording")
    parser.add_argument("analysis")
    parser.add_argument(
        "--output",
        default="artifacts/showcase/odor-motion-v2.png",
    )
    parser.add_argument("--label")
    parser.add_argument(
        "--events",
        help="Optional hash-bound odor-event diagnostics sidecar",
    )
    args = parser.parse_args()
    print(
        render_odor_motion_diagnostic(
            args.recording,
            args.analysis,
            args.output,
            label=args.label,
            diagnostics=args.events,
        )
    )


if __name__ == "__main__":
    main()
