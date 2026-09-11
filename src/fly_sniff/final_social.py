from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .controllers import CastSurgeController, Controller
from .env import FlySniffEnv
from .freeze import current_git_ref
from .graph import GraphBundle, MaleCNSRateController
from .plume import TurbulentPlume
from .qualified_eval import circuit_digest, verify_sealed_manifest
from .rewire import degree_preserving_rewire


@dataclass
class Trace:
    xy: np.ndarray
    odor_l: np.ndarray
    odor_r: np.ndarray
    dn_l: np.ndarray
    dn_r: np.ndarray
    success: bool


def _record(
    controller: Controller,
    seed: int,
    arena: ArenaConfig,
    plume: PlumeConfig,
    sensors: SensorConfig,
) -> Trace:
    env = FlySniffEnv(seed=seed, arena=arena, plume=plume, sensors=sensors)
    controller.reset(seed + 101)
    obs = env.observe()
    xy = [(env.agent.x, env.agent.y)]
    ol, or_, dl, dr = [], [], [], []
    done = False
    while not done:
        action = controller.act(obs)
        diag = controller.diagnostics()
        ol.append(obs.left_odor)
        or_.append(obs.right_odor)
        dl.append(diag.get("dn_left", 0.0))
        dr.append(diag.get("dn_right", 0.0))
        obs, done = env.step(action.turn, action.speed)
        xy.append((env.agent.x, env.agent.y))
    return Trace(
        xy=np.asarray(xy),
        odor_l=np.asarray(ol),
        odor_r=np.asarray(or_),
        dn_l=np.asarray(dl),
        dn_r=np.asarray(dr),
        success=env.agent.found,
    )


def _verify_inputs(circuit: Path, manifest: dict, receipt: dict) -> None:
    verify_sealed_manifest(manifest)
    if current_git_ref() != manifest.get("code_ref"):
        raise ValueError("renderer checkout does not match the sealed code_ref")
    if receipt.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("receipt does not belong to the supplied sealed manifest")
    observed = circuit_digest(circuit)
    if observed != manifest.get("circuit_sha256"):
        raise ValueError("circuit bytes do not match the sealed manifest")
    summary = receipt.get("summary", {})
    if summary.get("circuit_sha256") != observed:
        raise ValueError("receipt does not belong to the supplied circuit")
    if summary.get("code_ref") != manifest.get("code_ref"):
        raise ValueError("receipt code_ref does not match the sealed manifest")


def render_final(
    circuit_dir: str | Path,
    manifest_path: str | Path,
    receipt_path: str | Path,
    output: str | Path,
    *,
    fps: int = 30,
    seconds: int = 24,
) -> Path:
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
    controllers = [
        ("MALECNS CONNECTOME", MaleCNSRateController(biological)),
        ("DEGREE-PRESERVING REWIRE", MaleCNSRateController(rewired)),
        ("CLASSICAL PLUME SEARCH", CastSurgeController()),
    ]
    traces = [
        _record(controller, seed, arena, plume_cfg, sensors)
        for _, controller in controllers
    ]

    plume = TurbulentPlume(arena, plume_cfg, seed)
    plume.warmup()
    max_steps = max(len(t.xy) - 1 for t in traces)
    visual_frames = max(1, int(seconds * fps))
    experiment_frames = min(visual_frames, max(1, int(16 * fps)))
    sim_indices = np.linspace(0, max_steps, experiment_frames, dtype=int)
    plume_snaps: dict[int, np.ndarray] = {0: plume.snapshot()}
    last = 0
    for idx in np.unique(sim_indices):
        for _ in range(last, int(idx)):
            plume.step()
        plume_snaps[int(idx)] = plume.snapshot()
        last = int(idx)

    report = receipt["summary"]["gold_report"]
    id_rows = {r["label"]: r for r in report["id_summary"]}
    pair = report["malecns_vs_rewire_spl"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    grid = fig.add_gridspec(3, 3, height_ratios=[0.32, 2.7, 1.1])
    title_ax = fig.add_subplot(grid[0, :])
    arena_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    bottom_axes = [fig.add_subplot(grid[2, :2]), fig.add_subplot(grid[2, 2])]

    def _arena(ax, label: str, trace: Trace, idx: int, snap: np.ndarray) -> None:
        ax.clear()
        ax.set_xlim(0, arena.width)
        ax.set_ylim(0, arena.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=10, fontweight="bold")
        if len(snap):
            ax.scatter(
                snap[:, 0],
                snap[:, 1],
                s=np.clip(18 * snap[:, 2], 2, 22),
                alpha=0.15,
            )
        ax.scatter([arena.source_x], [arena.source_y], marker="*", s=130)
        j = min(idx, len(trace.xy) - 1)
        path = trace.xy[: j + 1]
        ax.plot(path[:, 0], path[:, 1], lw=2.1)
        ax.scatter([path[-1, 0]], [path[-1, 1]], s=45)
        ax.text(0.04, 0.04, "source visible to viewer only", transform=ax.transAxes, fontsize=7)

    def draw(frame: int):
        title_ax.clear()
        title_ax.axis("off")
        for ax in arena_axes + bottom_axes:
            ax.clear()

        if frame < experiment_frames:
            idx = int(sim_indices[frame])
            title_ax.text(
                0.5,
                0.60,
                "CAN A FRUIT-FLY CONNECTOME FIND AN INVISIBLE ODOR SOURCE?",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )
            title_ax.text(
                0.5,
                0.05,
                f"same sealed turbulent plume • held-out seed index 0 • seed {seed}",
                ha="center",
                fontsize=9,
            )
            snap = plume_snaps[idx]
            for ax, (label, _), trace in zip(arena_axes, controllers, traces, strict=True):
                _arena(ax, label, trace, idx, snap)

            neural_ax, status_ax = bottom_axes
            t = traces[0]
            k = min(max(idx - 1, 0), max(len(t.odor_l) - 1, 0))
            vals = [
                t.odor_l[k] if len(t.odor_l) else 0.0,
                t.odor_r[k] if len(t.odor_r) else 0.0,
                t.dn_l[k] if len(t.dn_l) else 0.0,
                t.dn_r[k] if len(t.dn_r) else 0.0,
            ]
            neural_ax.bar(["ODOR-L", "ODOR-R", "DN-L", "DN-R"], vals)
            neural_ax.set_ylim(-1.0, 1.0)
            neural_ax.set_title("MaleCNS modeled activity • not neural recordings", fontsize=9)
            status_ax.axis("off")
            status_ax.text(0.5, 0.65, "BIOLOGICAL TOPOLOGY", ha="center", fontweight="bold")
            status_ax.text(0.5, 0.42, "vs the same graph rewired", ha="center", fontsize=9)
            status_ax.text(0.5, 0.19, "same inputs • same dynamics", ha="center", fontsize=9)
        else:
            title_ax.text(
                0.5,
                0.55,
                "1,000 HELD-OUT TURBULENT PLUMES",
                ha="center",
                fontsize=17,
                fontweight="bold",
            )
            labels = ["malecns", "rewire", "classical"]
            names = ["MaleCNS", "Rewire", "Classical"]
            success = [100 * float(id_rows[x]["success_rate"]) for x in labels]
            spls = [float(id_rows[x]["mean_spl"]) for x in labels]
            ax0, ax1, ax2 = arena_axes
            ax0.bar(names, success)
            ax0.set_ylim(0, 100)
            ax0.set_ylabel("success %")
            ax0.tick_params(axis="x", labelrotation=20)
            ax1.bar(names, spls)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("mean SPL")
            ax1.tick_params(axis="x", labelrotation=20)
            ax2.axis("off")
            ax2.text(0.5, 0.72, "MaleCNS − rewire", ha="center", fontsize=11)
            ax2.text(
                0.5,
                0.54,
                f"ΔSPL {pair['mean_delta']:+.3f}",
                ha="center",
                fontsize=19,
                fontweight="bold",
            )
            ax2.text(
                0.5,
                0.39,
                f"95% CI [{pair['ci95_low']:+.3f}, {pair['ci95_high']:+.3f}]",
                ha="center",
                fontsize=10,
            )
            ax2.text(
                0.5,
                0.22,
                f"OOD success {100 * report['ood_success_rate']:.1f}%",
                ha="center",
                fontsize=10,
            )
            for ax in bottom_axes:
                ax.axis("off")
            claim = (
                "FLYNAV GOLD: BIOLOGICAL WIRING PASSED ALL PREREGISTERED GATES"
                if report["flynav_gold"]
                else "PREREGISTERED RESULT: FLYNAV GOLD NOT ACHIEVED"
            )
            bottom_axes[0].text(0.0, 0.62, claim, fontsize=12, fontweight="bold")
            bottom_axes[0].text(
                0.0,
                0.30,
                "full receipt + code: github.com/sidhulyalkar/fly-sniff",
                fontsize=10,
            )
            bottom_axes[1].text(
                0.5,
                0.5,
                "connectome ≠ physiology\nstructure tested as an inductive bias",
                ha="center",
                fontsize=9,
            )
        return []

    ani = animation.FuncAnimation(
        fig,
        draw,
        frames=visual_frames,
        interval=1000 / fps,
        blit=False,
    )
    try:
        ani.save(output, writer=animation.FFMpegWriter(fps=fps, bitrate=5000))
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a receipt-verified final social video")
    parser.add_argument("circuit")
    parser.add_argument("manifest")
    parser.add_argument("receipt")
    parser.add_argument("--output", default="artifacts/fly-sniff-final.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=24)
    args = parser.parse_args()
    print(
        render_final(
            args.circuit,
            args.manifest,
            args.receipt,
            args.output,
            fps=args.fps,
            seconds=args.seconds,
        )
    )
