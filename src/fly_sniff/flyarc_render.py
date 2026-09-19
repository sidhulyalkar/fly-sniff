from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_receipt(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    receipt_path = root / "receipt.json"
    if not receipt_path.exists():
        raise ValueError(f"missing FlyARC receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("experiment") != "flyarc-v1":
        raise ValueError("renderer accepts only flyarc-v1 receipts")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("FlyARC receipt contains no hashed files")

    mismatches: list[str] = []
    for relative, expected in files.items():
        path = root / relative
        if not path.exists():
            mismatches.append(f"{relative}: missing")
            continue
        observed = _sha256(path)
        if observed != expected:
            mismatches.append(f"{relative}: expected {expected}, observed {observed}")
    if mismatches:
        raise ValueError("FlyARC receipt validation failed: " + "; ".join(mismatches))
    return receipt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _unpack_frame(
    packed: np.ndarray,
    shapes: np.ndarray,
    index: int,
) -> np.ndarray:
    if index < 0 or index >= len(packed):
        raise IndexError(f"frame index {index} outside [0, {len(packed)})")
    height, width = (int(x) for x in shapes[index])
    return packed[index, :height, :width]


def _state_grid(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=float).ravel()
    side = math.ceil(math.sqrt(max(1, len(values))))
    grid = np.full(side * side, np.nan, dtype=float)
    grid[: len(values)] = values
    return grid.reshape(side, side)


def load_variant(run_dir: str | Path, variant: str) -> dict[str, Any]:
    root = Path(run_dir) / variant
    steps = _load_jsonl(root / "steps.jsonl")
    with np.load(root / "frames.npz") as archive:
        frames = archive["frames"].copy()
        shapes = archive["shapes"].copy()
    with np.load(root / "states.npz") as archive:
        states = archive["states"].copy()
        body_ids = archive["body_ids"].copy()
    metrics = json.loads((root / "metrics.json").read_text())

    if len(states) != len(steps):
        raise ValueError(
            f"{variant}: state/step mismatch: states={len(states)} steps={len(steps)}"
        )
    for row in steps:
        index = int(row["observation_index"])
        if index < 0 or index >= len(frames):
            raise ValueError(f"{variant}: invalid observation_index={index}")
    return {
        "variant": variant,
        "steps": steps,
        "frames": frames,
        "shapes": shapes,
        "states": states,
        "body_ids": body_ids,
        "metrics": metrics,
    }


def _variant_title(name: str) -> str:
    labels = {
        "intact": "MaleCNS topology",
        "rewire": "Degree-preserving rewire",
        "random": "Random recurrent null",
        "stateless": "No recurrence",
    }
    return labels.get(name, name)


def _arc_cmap():
    return plt.get_cmap("tab20", 16)


def build_figure(variants: list[str]) -> tuple[Figure, dict[str, Any]]:
    if not variants:
        raise ValueError("at least one variant is required")
    columns = len(variants)
    figure = plt.figure(figsize=(4.1 * columns, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, columns, height_ratios=[1.0, 0.9])
    artists: dict[str, Any] = {}
    cmap = _arc_cmap()
    norm = BoundaryNorm(np.arange(-0.5, 16.5, 1.0), cmap.N)

    for column, variant in enumerate(variants):
        world_ax = figure.add_subplot(grid[0, column])
        state_ax = figure.add_subplot(grid[1, column])
        world = world_ax.imshow(
            np.zeros((2, 2), dtype=np.uint8),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        state = state_ax.imshow(
            np.zeros((2, 2), dtype=float),
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        world_ax.set_title(_variant_title(variant), fontsize=12, weight="bold")
        world_ax.set_xticks([])
        world_ax.set_yticks([])
        state_ax.set_xticks([])
        state_ax.set_yticks([])
        state_ax.set_xlabel("modeled reservoir state • rank layout, not anatomy", fontsize=8)
        status = world_ax.text(
            0.02,
            0.02,
            "",
            transform=world_ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 3},
            color="white",
        )
        artists[variant] = {
            "world": world,
            "state": state,
            "status": status,
        }

    figure.suptitle(
        "FlyARC v1 • DEVELOPMENT EXPERIMENT • modeled neural activity, not recordings",
        fontsize=15,
        weight="bold",
    )
    return figure, artists


def render(
    run_dir: str | Path,
    output: str | Path,
    *,
    fps: int = 8,
    max_frames: int | None = None,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")
    root = Path(run_dir)
    receipt = validate_receipt(root)
    comparison = json.loads((root / "comparison.json").read_text())
    variants = list(comparison.get("results", {}).keys())
    if not variants:
        raise ValueError("comparison.json contains no variants")
    data = {variant: load_variant(root, variant) for variant in variants}
    total = max(len(item["steps"]) for item in data.values())
    if max_frames is not None:
        total = min(total, max_frames)
    if total < 1:
        raise ValueError("FlyARC run contains no action steps to render")

    figure, artists = build_figure(variants)

    def update(frame_index: int):
        changed: list[Any] = []
        for variant in variants:
            item = data[variant]
            index = min(frame_index, len(item["steps"]) - 1)
            row = item["steps"][index]
            observation_index = int(row["observation_index"])
            world = _unpack_frame(
                item["frames"],
                item["shapes"],
                observation_index,
            )
            state = _state_grid(item["states"][index])
            artists[variant]["world"].set_data(world)
            artists[variant]["world"].set_extent((0, world.shape[1], world.shape[0], 0))
            artists[variant]["state"].set_data(state)
            label = (
                f"step {row['step']}  {row['action']}\n"
                f"levels {row['levels_completed']}  reward {row['reward']:+.3f}\n"
                f"{row['state']}"
            )
            if index < frame_index:
                label += "\n(run ended)"
            artists[variant]["status"].set_text(label)
            changed.extend(
                [
                    artists[variant]["world"],
                    artists[variant]["state"],
                    artists[variant]["status"],
                ]
            )
        return changed

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=total,
        interval=1000 / fps,
        blit=False,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "MP4 rendering requires ffmpeg; use a .gif output or install ffmpeg"
            )
        writer = animation.FFMpegWriter(fps=fps, bitrate=2800)
    else:
        raise ValueError("FlyARC renderer output must end in .gif or .mp4")

    movie.save(output_path, writer=writer, dpi=120)
    plt.close(figure)

    render_receipt = {
        "experiment": "flyarc-v1-render",
        "source_receipt_sha256": _sha256(root / "receipt.json"),
        "source_paired_version_check": receipt.get("paired_version_check"),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "fps": fps,
        "frames": total,
        "claim_status": "development-visualization",
    }
    output_path.with_suffix(output_path.suffix + ".receipt.json").write_text(
        json.dumps(render_receipt, indent=2, sort_keys=True) + "\n"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a receipt-verified FlyARC comparison without rerunning ARC or MaleCNS"
    )
    parser.add_argument("run_dir")
    parser.add_argument("--output", default="artifacts/flyarc-v1.gif")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    output = render(
        args.run_dir,
        args.output,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(output)


if __name__ == "__main__":
    main()
