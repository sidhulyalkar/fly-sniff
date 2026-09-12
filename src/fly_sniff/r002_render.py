from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from .graph import GraphBundle
from .looming import LoomConfig, generate_loom_frames
from .rewire import degree_preserving_rewire
from .runtime import ConnectomeRuntime


def collect_trace(
    bundle: GraphBundle,
    *,
    side: str = "left",
    trajectory: str = "direct-hit",
    rewire_seed: int = 24018,
    require_qualified: bool = True,
) -> dict:
    frames = generate_loom_frames(side, trajectory, config=LoomConfig())
    rewired = degree_preserving_rewire(bundle, seed=rewire_seed)
    target = "escape_left" if side == "left" else "escape_right"
    opposite = "escape_right" if side == "left" else "escape_left"
    for role in (target, opposite):
        if role not in bundle.roles:
            raise ValueError(f"R002 render requires role {role!r}")

    output: dict[str, object] = {
        "side": side,
        "trajectory": trajectory,
        "rewire_seed": rewire_seed,
        "frames": [asdict(frame) for frame in frames],
    }
    for name, graph in (("intact", bundle), ("rewired", rewired)):
        runtime = ConnectomeRuntime(graph, require_qualified=require_qualified)
        runtime.reset()
        target_values: list[float] = []
        opposite_values: list[float] = []
        for frame in frames:
            snapshot = runtime.step(frame.inputs, readouts=(target, opposite))
            target_values.append(float(snapshot.readouts[target]))
            opposite_values.append(float(snapshot.readouts[opposite]))
        output[name] = {
            "target": target_values,
            "opposite": opposite_values,
            "peak_target": max(target_values),
            "peak_opposite": max(opposite_values),
            "lateralization": max(target_values) - max(opposite_values),
        }
    return output


def _status(bundle: GraphBundle) -> str:
    return str((bundle.manifest or {}).get("qualification_status", "unknown")).upper()


def _render_summary(trace: dict, bundle: GraphBundle, output: Path) -> None:
    frames = trace["frames"]
    times = np.asarray([frame["t"] for frame in frames], dtype=float)
    size = np.asarray(
        [
            max(frame["inputs"]["loom_size_left"], frame["inputs"]["loom_size_right"])
            for frame in frames
        ],
        dtype=float,
    )
    velocity = np.asarray(
        [
            max(
                frame["inputs"]["loom_velocity_left"],
                frame["inputs"]["loom_velocity_right"],
            )
            for frame in frames
        ],
        dtype=float,
    )
    intact = trace["intact"]
    rewired = trace["rewired"]

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_stim = fig.add_subplot(grid[0, :])
    ax_intact = fig.add_subplot(grid[1, 0])
    ax_rewired = fig.add_subplot(grid[1, 1])

    ax_stim.plot(times, size, label="angular-size drive")
    ax_stim.plot(times, velocity, label="expansion-speed drive")
    ax_stim.set_title("ONE LOOMING STIMULUS")
    ax_stim.set_xlabel("time (s)")
    ax_stim.set_ylabel("modeled sensory drive")
    ax_stim.set_ylim(-0.02, 1.02)
    ax_stim.legend(loc="upper left")

    for axis, values, title in (
        (ax_intact, intact, "MALECNS WIRING"),
        (ax_rewired, rewired, "DEGREE-PRESERVING REWIRE"),
    ):
        axis.plot(times, values["target"], label="DNp01 target side")
        axis.plot(times, values["opposite"], label="DNp01 opposite side")
        axis.set_title(f"{title}\nlateralization = {values['lateralization']:.3f}")
        axis.set_xlabel("time (s)")
        axis.set_ylabel("modeled DNp01/GF activity")
        axis.set_ylim(-0.02, 1.02)
        axis.legend(loc="upper left")

    status = _status(bundle)
    fig.suptitle(
        "Same looming input. Same modeled dynamics. Only topology changed.\n"
        f"R002 {status} | DNp01/GF escape readout, not a steering command",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _render_gif(trace: dict, bundle: GraphBundle, output: Path, *, fps: int = 20) -> None:
    frames = trace["frames"]
    intact = trace["intact"]
    rewired = trace["rewired"]
    status = _status(bundle)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    ax_stim, ax_intact, ax_rewired = axes
    fig.suptitle(
        "Same looming input. Same modeled dynamics. Only topology changed.\n"
        f"R002 {status} | modeled DNp01/GF escape readout",
        fontsize=14,
    )

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    ax_stim.set_xlim(-1, 1)
    ax_stim.set_ylim(-1, 1)
    ax_stim.set_aspect("equal")
    ax_stim.set_title("LOOMING OBJECT")
    stimulus = plt.Circle((0.0, 0.0), 0.01, fill=False, linewidth=3)
    ax_stim.add_patch(stimulus)
    stim_text = ax_stim.text(0.0, -0.88, "", ha="center", va="center")

    def prepare_readout_axis(axis, title: str):
        axis.set_xlim(-0.5, 1.5)
        axis.set_ylim(0.0, 1.02)
        axis.set_xticks([0, 1], ["TARGET\nDNp01", "OPPOSITE\nDNp01"])
        axis.set_yticks([0.0, 0.5, 1.0])
        axis.set_title(title)
        bars = axis.bar([0, 1], [0.0, 0.0], width=0.55)
        label = axis.text(0.5, 0.94, "", ha="center", va="top", transform=axis.transAxes)
        return bars, label

    intact_bars, intact_text = prepare_readout_axis(ax_intact, "MALECNS WIRING")
    rewired_bars, rewired_text = prepare_readout_axis(
        ax_rewired, "DEGREE-PRESERVING REWIRE"
    )

    max_angle = max(float(frame["angular_size_rad"]) for frame in frames) or 1.0

    def update(index: int):
        frame = frames[index]
        radius = 0.04 + 0.80 * float(frame["angular_size_rad"]) / max_angle
        stimulus.set_radius(radius)
        size_drive = max(
            frame["inputs"]["loom_size_left"], frame["inputs"]["loom_size_right"]
        )
        velocity_drive = max(
            frame["inputs"]["loom_velocity_left"],
            frame["inputs"]["loom_velocity_right"],
        )
        stim_text.set_text(
            f"t={frame['t']:.2f}s\nsize={size_drive:.2f}  expansion={velocity_drive:.2f}"
        )

        artists = [stimulus, stim_text]
        for bars, text, values in (
            (intact_bars, intact_text, intact),
            (rewired_bars, rewired_text, rewired),
        ):
            target_value = float(values["target"][index])
            opposite_value = float(values["opposite"][index])
            bars[0].set_height(target_value)
            bars[1].set_height(opposite_value)
            text.set_text(f"target − opposite = {target_value - opposite_value:+.3f}")
            artists.extend([bars[0], bars[1], text])
        return artists

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


def render_r002(
    bundle: GraphBundle,
    *,
    output_dir: str | Path,
    side: str = "left",
    trajectory: str = "direct-hit",
    rewire_seed: int = 24018,
    require_qualified: bool = True,
) -> dict:
    trace = collect_trace(
        bundle,
        side=side,
        trajectory=trajectory,
        rewire_seed=rewire_seed,
        require_qualified=require_qualified,
    )
    out = Path(output_dir)
    summary_path = out / "r002_summary.png"
    gif_path = out / "r002_intact_vs_rewire.gif"
    trace_path = out / "r002_render_trace.json"
    _render_summary(trace, bundle, summary_path)
    _render_gif(trace, bundle, gif_path)
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")

    artifacts = {}
    for path in (summary_path, gif_path, trace_path):
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt = {
        "schema": "fly-sniff-r002-render-receipt-v1",
        "qualification_status": (bundle.manifest or {}).get("qualification_status"),
        "scientific_claim_allowed": bool(
            (bundle.manifest or {}).get("scientific_claim_allowed", False)
        ),
        "side": side,
        "trajectory": trajectory,
        "rewire_seed": rewire_seed,
        "headline": "Same looming input. Same modeled dynamics. Only topology changed.",
        "claim_boundary": (
            "The animation visualizes modeled DNp01/GF escape readout under a synthetic looming "
            "adapter. It does not show perception, a biological jump, or measured neural activity."
        ),
        "artifacts": artifacts,
    }
    receipt_path = out / "r002_render_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Render R002 intact-vs-rewired looming evidence")
    parser.add_argument("circuit")
    parser.add_argument("--output-dir", default="artifacts/r002/render")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--trajectory", choices=("direct-hit", "near-miss"), default="direct-hit"
    )
    parser.add_argument("--rewire-seed", type=int, default=24018)
    parser.add_argument("--allow-candidate", action="store_true")
    args = parser.parse_args()

    bundle = GraphBundle.load(args.circuit)
    status = (bundle.manifest or {}).get("qualification_status")
    if status != "qualified" and not args.allow_candidate:
        raise SystemExit(
            "R002 public rendering requires a qualified graph. Pass --allow-candidate only for "
            "development evidence, which remains visibly labelled CANDIDATE."
        )
    receipt = render_r002(
        bundle,
        output_dir=args.output_dir,
        side=args.side,
        trajectory=args.trajectory,
        rewire_seed=args.rewire_seed,
        require_qualified=not args.allow_candidate,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
