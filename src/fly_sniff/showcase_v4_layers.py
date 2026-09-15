from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import to_rgb

from .party_social import PLUME


def viewer_density_field(
    current: dict[str, Any], payload: dict[str, Any], *, nx: int = 88, ny: int = 54
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """Evaluate the recorded Gaussian plume on a display grid.

    This is allowed only when the recording says the puff snapshot is complete.
    The returned field is a viewer visualization and is never a controller input.
    """
    if not bool(current.get("plume_snapshot", {}).get("complete", False)):
        return None
    snapshot = np.asarray(current.get("plume", []), dtype=float)
    width = float(payload["arena"]["width"])
    height = float(payload["arena"]["height"])
    extent = (0.0, width, 0.0, height)
    if snapshot.size == 0:
        return np.zeros((ny, nx), dtype=float), extent
    if snapshot.ndim != 2 or snapshot.shape[1] != 3:
        raise ValueError("recorded plume must have x/y/sigma columns")

    xs = np.linspace(0.0, width, nx)
    ys = np.linspace(0.0, height, ny)
    xx, yy = np.meshgrid(xs, ys)
    dx = xx[..., None] - snapshot[:, 0]
    dy = yy[..., None] - snapshot[:, 1]
    sigma2 = np.maximum(snapshot[:, 2] ** 2, 1e-9)
    d2 = dx * dx + dy * dy
    local = d2 <= (4.0 * snapshot[:, 2]) ** 2
    puff_mass = float(payload["config"]["plume"]["puff_mass"])
    density = puff_mass * np.exp(-0.5 * d2 / sigma2) / (2.0 * np.pi * sigma2)
    return np.where(local, density, 0.0).sum(axis=2), extent


def recording_density_scale(
    payload: dict[str, Any],
    last_index: int,
    *,
    percentile: float = 98.0,
    max_samples: int = 90,
    nx: int = 44,
    ny: int = 27,
) -> float:
    """Freeze one plume-display scale over the complete social replay window.

    The scale is a renderer-only quantity. Sampling frames and a coarser grid keeps
    the prepass cheap while preventing per-frame normalization from making the
    plume appear to brighten or dim merely because the normalization changed.
    """
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError("recording has no frames")
    if last_index < 0 or last_index >= len(frames):
        raise ValueError("last_index outside recording")
    sample_count = min(max_samples, last_index + 1)
    indices = np.unique(np.linspace(0, last_index, sample_count, dtype=int))
    positive_chunks: list[np.ndarray] = []
    for index in indices:
        field = viewer_density_field(frames[int(index)], payload, nx=nx, ny=ny)
        if field is None:
            continue
        density, _ = field
        positive = density[density > 0.0]
        if positive.size:
            positive_chunks.append(positive)
    if not positive_chunks:
        return 1.0
    values = np.concatenate(positive_chunks)
    return max(float(np.percentile(values, percentile)), 1e-12)


def density_rgba(density: np.ndarray, *, scale: float | None = None) -> np.ndarray:
    """Convert exact density into a translucent display layer.

    When ``scale`` is supplied, the same fixed renderer scale is used across all
    frames so opacity is temporally comparable. The nonlinear alpha mapping is
    still for phone readability and is not a quantitative concentration colorbar.
    """
    density = np.asarray(density, dtype=float)
    rgba = np.zeros((*density.shape, 4), dtype=float)
    rgba[..., :3] = np.asarray(to_rgb(PLUME))
    positive = density[density > 0.0]
    if not positive.size:
        return rgba
    if scale is None:
        scale = max(float(np.percentile(positive, 98.0)), 1e-12)
    else:
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("density display scale must be finite and > 0")
    normalized = np.log1p(3.0 * density / scale) / np.log(4.0)
    rgba[..., 3] = 0.56 * np.clip(normalized, 0.0, 1.0)
    return rgba


def social_display_frame_limit(
    payload: dict[str, Any],
    *,
    reveal_hold_s: float = 2.6,
    no_success_window_s: float = 18.0,
) -> int:
    """Return the last recorded frame used by the social edit.

    The simulation is never truncated or rerun. This only chooses a replay window
    so a successful episode pays off near the end instead of showing a stationary
    post-success animal for most of the clip.
    """
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError("recording has no frames")
    final_t = float(frames[-1]["t"])
    found_times: list[float] = []
    for frame in frames:
        if any(bool(agent.get("found")) for agent in frame.get("agents", [])):
            found_times.append(float(frame["t"]))
    if found_times:
        display_end_t = min(final_t, min(found_times) + float(reveal_hold_s))
    else:
        display_end_t = min(final_t, float(no_success_window_s))
    eligible = [index for index, frame in enumerate(frames) if float(frame["t"]) <= display_end_t]
    return eligible[-1] if eligible else 0


def pfl3_population_frame(
    e002c: dict[str, Any],
    fc2: dict[str, Any],
    *,
    threshold: int,
    probe_index: int,
) -> tuple[list[int], list[int], dict[str, np.ndarray], float]:
    """Return all 24 modeled PFL3 values ordered by anatomical C label."""
    report = e002c["threshold_reports"][str(int(threshold))]
    body_ids = [int(x) for x in report["joint"]["body_ids"]]
    column_map = {
        int(k): int(v)
        for k, v in fc2["populations"]["PFL3"]["instance_columns"]["body_columns"].items()
    }
    if len(body_ids) != 24 or set(body_ids) != set(column_map):
        raise ValueError("v4 requires the exact 24 PFL3 bodies resolved by the FC2 audit")
    order = sorted(range(24), key=lambda i: (column_map[body_ids[i]], body_ids[i]))
    columns = [column_map[body_ids[i]] for i in order]
    values: dict[str, np.ndarray] = {}
    scale = 0.0
    for name in ("goal_only", "heading_only", "joint"):
        activity = np.abs(np.asarray(report[name]["activity"], dtype=float))
        if activity.ndim != 2 or activity.shape[1] != 24:
            raise ValueError(f"{name} PFL3 activity must have shape (steps, 24)")
        scale = max(scale, float(np.max(activity)))
        index = int(np.clip(probe_index, 0, activity.shape[0] - 1))
        values[name] = activity[index, order]
    return [body_ids[i] for i in order], columns, values, max(scale, 1e-12)


def blend_activity_color(base: str, strength: float) -> tuple[float, float, float]:
    bg = np.asarray(to_rgb("#172033"))
    fg = np.asarray(to_rgb(base))
    t = float(np.clip(strength, 0.0, 1.0))
    return tuple(bg * (1.0 - t) + fg * t)
