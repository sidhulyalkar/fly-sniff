from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .final_social import Trace, _record, _verify_inputs
from .graph import GraphBundle, MaleCNSRateController
from .plume import TurbulentPlume
from .rewire import degree_preserving_rewire


def _people_for_arena(arena: ArenaConfig) -> tuple[tuple[float, float, str], ...]:
    """Return one exact source suspect plus five decorative decoys."""
    return (
        (arena.source_x, arena.source_y, "A"),
        (arena.source_x, 0.9, "B"),
        (arena.source_x, arena.height - 0.9, "C"),
        (0.40 * arena.width, 0.8, "D"),
        (0.40 * arena.width, arena.height - 0.8, "E"),
        (0.68 * arena.width, arena.height - 0.9, "F"),
    )


def _draw_person(ax, x: float, y: float, tag: str, *, culprit: bool) -> None:
    ax.add_patch(Rectangle((x - 0.12, y - 0.28), 0.24, 0.36, alpha=0.72, zorder=6))
    ax.add_patch(Circle((x, y + 0.19), 0.14, alpha=0.88, zorder=7))
    ax.text(x, y + 0.19, tag, ha="center", va="center", fontsize=7, fontweight="bold", zorder=8)
    if culprit:
        ax.add_patch(Circle((x, y), 0.42, fill=False, linewidth=3.0, zorder=9))
        ax.text(x + 0.46, y + 0.34, "ODOR SOURCE", fontsize=7, fontweight="bold", zorder=10)


def _draw_room(
    ax,
    *,
    arena: ArenaConfig,
    trace: Trace,
    snap: np.ndarray,
    index: int,
    label: str,
    reveal: bool,
) -> None:
    ax.clear()
    ax.set_xlim(0, arena.width)
    ax.set_ylim(0, arena.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.add_patch(Rectangle((0, 0), arena.width, arena.height, fill=False, linewidth=2.0))

    for person_index, (x, y, tag) in enumerate(_people_for_arena(arena)):
        _draw_person(ax, x, y, tag, culprit=reveal and person_index == 0)

    if len(snap):
        ax.scatter(
            snap[:, 0],
            snap[:, 1],
            s=np.clip(34 * snap[:, 2], 4, 48),
            alpha=0.20,
            c="yellowgreen",
            edgecolors="none",
            zorder=2,
        )

    j = min(index, len(trace.xy) - 1)
    path = trace.xy[: j + 1]
    ax.plot(path[:, 0], path[:, 1], linewidth=3.0, zorder=4)
    ax.scatter([path[-1, 0]], [path[-1, 1]], s=72, marker="X", zorder=5)
    elapsed_s = j * arena.dt
    result = "FOUND IT" if trace.success and j >= len(trace.xy) - 1 else "SNIFFING..."
    ax.text(
        0.03,
        0.96,
        f"{elapsed_s:04.1f}s  •  {result}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.03,
        "smell visible to you • source hidden from controller",
        transform=ax.transAxes,
        fontsize=6.5,
    )


def _heading_at(trace: Trace, index: int) -> float:
    j = min(max(index, 1), len(trace.xy) - 1)
    delta = trace.xy[j] - trace.xy[j - 1]
    if np.allclose(delta, 0.0):
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def _draw_fly_pov(
    ax,
    *,
    trace: Trace,
    snap: np.ndarray,
    index: int,
) -> None:
    ax.clear()
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(0.0, 5.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("FLY POV • ODOR MADE VISIBLE", fontsize=9, fontweight="bold")
    j = min(index, len(trace.xy) - 1)
    position = trace.xy[j]
    heading = _heading_at(trace, j)
    if len(snap):
        dx = snap[:, 0] - position[0]
        dy = snap[:, 1] - position[1]
        c = np.cos(heading)
        s = np.sin(heading)
        forward = c * dx + s * dy
        lateral = -s * dx + c * dy
        mask = (forward > 0.0) & (forward < 5.0) & (np.abs(lateral) < 2.4)
        if np.any(mask):
            sizes = 70 * snap[mask, 2] / np.maximum(forward[mask], 0.25)
            ax.scatter(
                lateral[mask],
                forward[mask],
                s=np.clip(sizes, 5, 95),
                alpha=0.23,
                c="yellowgreen",
                edgecolors="none",
            )
    ax.plot([-0.12, -0.34], [0.18, 0.55], linewidth=2.0)
    ax.plot([0.12, 0.34], [0.18, 0.55], linewidth=2.0)
    ax.add_patch(Circle((0.0, 0.12), 0.18, alpha=0.85))
    k = min(max(j - 1, 0), max(len(trace.odor_l) - 1, 0))
    left = trace.odor_l[k] if len(trace.odor_l) else 0.0
    right = trace.odor_r[k] if len(trace.odor_r) else 0.0
    ax.text(
        0.03,
        0.94,
        f"left antenna {left:.2f}    right antenna {right:.2f}",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
    )


def render_final_party(
    circuit_dir: str | Path,
    manifest_path: str | Path,
    receipt_path: str | Path,
    output: str | Path,
    *,
    fps: int = 30,
    seconds: int = 18,
) -> Path:
    """Render the claim-bearing Who Farted cut from sealed scientific inputs only."""
    circuit_dir = Path(circuit_dir)
    manifest = json.loads(Path(manifest_path).read_text())
    receipt = json.loads(Path(receipt_path).read_text())
    _verify_inputs(circuit_dir, manifest, receipt)

    biological = GraphBundle.load(circuit_dir)
    biological.validate(require_sign=True, require_qualified=True)
    rw = manifest["rewire"]
    rewired = degree_preserving_rewire(
        biological,
        seed=int(rw["seed"]),
        swaps_per_edge=int(rw["swaps_per_edge"]),
    )

    cfg = manifest["config"]
    arena = ArenaConfig(**cfg["arena"])
    plume_cfg = PlumeConfig(**cfg["plume"])
    sensors = SensorConfig(**cfg["sensor"])
    seed = int(manifest["heldout_seeds"][0])
    traces = [
        _record(MaleCNSRateController(biological), seed, arena, plume_cfg, sensors),
        _record(MaleCNSRateController(rewired), seed, arena, plume_cfg, sensors),
    ]

    plume = TurbulentPlume(arena, plume_cfg, seed)
    plume.warmup()
    max_steps = max(len(trace.xy) - 1 for trace in traces)
    total_frames = max(1, int(seconds * fps))
    experiment_frames = max(1, min(total_frames, int(13 * fps)))
    sim_indices = np.linspace(0, max_steps, experiment_frames, dtype=int)
    plume_snaps: dict[int, np.ndarray] = {0: plume.snapshot()}
    last = 0
    for idx in np.unique(sim_indices):
        for _ in range(last, int(idx)):
            plume.step()
        plume_snaps[int(idx)] = plume.snapshot()
        last = int(idx)

    report = receipt["summary"]["gold_report"]
    id_rows = {row["label"]: row for row in report["id_summary"]}
    pair = report["malecns_vs_rewire_spl"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.46, 2.65, 1.12])
    title_ax = fig.add_subplot(grid[0, :])
    room_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    bottom_ax = fig.add_subplot(grid[2, :])

    def draw(frame: int):
        title_ax.clear()
        title_ax.axis("off")
        for ax in room_axes:
            ax.clear()
        bottom_ax.clear()

        if frame < experiment_frames:
            idx = int(sim_indices[frame])
            reveal = frame >= int(10.5 * fps)
            title_ax.text(
                0.5,
                0.68,
                "WHO FARTED?",
                ha="center",
                va="center",
                fontsize=29,
                fontweight="bold",
            )
            title_ax.text(
                0.5,
                0.24,
                "Same fly neurons. Same smell. Then we scrambled the wiring.",
                ha="center",
                va="center",
                fontsize=10.5,
            )
            snap = plume_snaps[idx]
            _draw_room(
                room_axes[0],
                arena=arena,
                trace=traces[0],
                snap=snap,
                index=idx,
                label="REAL BRAIN WIRING",
                reveal=reveal,
            )
            _draw_room(
                room_axes[1],
                arena=arena,
                trace=traces[1],
                snap=snap,
                index=idx,
                label="SCRAMBLED WIRING",
                reveal=reveal,
            )
            _draw_fly_pov(bottom_ax, trace=traces[0], snap=snap, index=idx)
        else:
            title_ax.text(
                0.5,
                0.58,
                "THE JOKE WAS ONE PLUME. THE TEST WAS 1,000.",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
            )
            real_success = 100 * float(id_rows["malecns"]["success_rate"])
            rewire_success = 100 * float(id_rows["rewire"]["success_rate"])
            for ax, name, value in zip(
                room_axes,
                ("REAL WIRING", "SCRAMBLED"),
                (real_success, rewire_success),
                strict=True,
            ):
                ax.bar([name], [value])
                ax.set_ylim(0, 100)
                ax.set_ylabel("source found (%)")
                ax.text(0, value + 2, f"{value:.1f}%", ha="center", fontweight="bold")
            bottom_ax.axis("off")
            verdict = "BIOLOGICAL WIRING HELPED" if report["flynav_gold"] else "FLYNAV GOLD NOT ACHIEVED"
            bottom_ax.text(0.5, 0.72, verdict, ha="center", fontsize=17, fontweight="bold")
            bottom_ax.text(
                0.5,
                0.48,
                f"MaleCNS − rewire ΔSPL {pair['mean_delta']:+.3f}  •  "
                f"95% CI [{pair['ci95_low']:+.3f}, {pair['ci95_high']:+.3f}]",
                ha="center",
                fontsize=10,
            )
            bottom_ax.text(
                0.5,
                0.25,
                "modeled dynamics over MaleCNS structure • full receipt + code: "
                "github.com/sidhulyalkar/fly-sniff",
                ha="center",
                fontsize=8.5,
            )
        return []

    ani = animation.FuncAnimation(
        fig,
        draw,
        frames=total_frames,
        interval=1000 / fps,
        blit=False,
    )
    try:
        ani.save(output, writer=animation.FFMpegWriter(fps=fps, bitrate=5200))
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the receipt-verified Who Farted social cut")
    parser.add_argument("circuit")
    parser.add_argument("manifest")
    parser.add_argument("receipt")
    parser.add_argument("--output", default="artifacts/who-farted-final.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=18)
    args = parser.parse_args()
    print(
        render_final_party(
            args.circuit,
            args.manifest,
            args.receipt,
            args.output,
            fps=args.fps,
            seconds=args.seconds,
        )
    )
