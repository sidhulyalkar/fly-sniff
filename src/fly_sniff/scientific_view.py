"""Pure, quantitative views of recorded state. No simulator execution here."""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .party_social import MUTED, PANEL, TEXT

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "odor", [(0.0, (0.03, 0.08, 0.08, 0.0)), (0.25, (0.2, 0.65, 0.15, 0.25)),
             (1.0, (0.65, 0.95, 0.25, 0.8))]
)


def density_grid(snapshot, puff_mass: float, xs, ys) -> np.ndarray:
    """Same Gaussian mixture and square 4-sigma cutoff as the simulator.

    Values are exact at raster sample points for a complete snapshot; display
    interpolation between these points is only a visual approximation.
    """
    points = np.asarray(snapshot, dtype=float).reshape(-1, 3)
    xs, ys = np.asarray(xs), np.asarray(ys)
    if not np.isfinite(points).all() or not np.isfinite(puff_mass) or puff_mass < 0:
        raise ValueError("density inputs must be finite and mass nonnegative")
    if (points[:, 2] <= 0).any():
        raise ValueError("puff sigma must be positive")
    result = np.zeros((len(ys), len(xs)))
    for x, y, sigma in points:
        xi = np.flatnonzero(np.abs(xs - x) <= 4 * sigma)
        yi = np.flatnonzero(np.abs(ys - y) <= 4 * sigma)
        if not len(xi) or not len(yi):
            continue
        variance = max(sigma**2, 1e-9)
        squared = (ys[yi, None] - y)**2 + (xs[None, xi] - x)**2
        result[np.ix_(yi, xi)] += puff_mass * np.exp(-0.5 * squared / variance) / (2*np.pi*variance)
    return result


def draw_density(ax, payload: dict[str, Any], frame: dict[str, Any]) -> str:
    meta = frame.get("plume_snapshot", {})
    mass = payload.get("plume_recording", {}).get("puff_mass")
    if mass is None:
        return "PUFF POSITIONS • density unavailable"
    arena = payload["arena"]
    xs = np.linspace(0, arena["width"], 128)
    ys = np.linspace(0, arena["height"], 80)
    density = density_grid(frame["plume"], mass, xs, ys)
    # Fixed across the entire clip; never normalize individual frames.
    ax.imshow(density, origin="lower", extent=(0, arena["width"], 0, arena["height"]),
              cmap=DENSITY_CMAP, vmin=0, vmax=10, interpolation="bilinear", zorder=1)
    complete = meta.get("complete") is True
    prefix = "MODEL DENSITY" if complete else "SAMPLED DENSITY • INCOMPLETE"
    return f"{prefix} • fixed scale 0–10 model units (clipped)"


def trace_arrays(payload, label):
    states = [next(a for a in f["agents"] if a["label"] == label) for f in payload["frames"]]
    return {
        "t": np.array([f["t"] for f in payload["frames"]]),
        "left": np.array([s["observation"]["left_odor"] for s in states]),
        "right": np.array([s["observation"]["right_odor"] for s in states]),
        "turn": np.array([s["action"]["turn"] if s.get("decision_valid", not s["done"])
                          else np.nan for s in states]),
    }


def draw_trace(ax, trace, index):
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlim(0, max(float(trace["t"][-1]), 0.05))
    # Future values stay hidden, while all curves use the same recorded clock.
    end = index + 1
    for key, color, label in (("left", "#67E8F9", "Left antenna [0,1]"),
                               ("right", "#C4B5FD", "Right antenna [0,1]"),
                               ("turn", "#FDE68A", "Turn [−1 right, +1 left]")):
        ax.plot(trace["t"][:end], trace[key][:end], color=color, lw=1.4, label=label)
    ax.axhline(0, color=MUTED, lw=0.5, alpha=0.5)
    ax.axvline(trace["t"][index], color=TEXT, lw=1, alpha=0.7)
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("Recorded simulation time (s)", color=MUTED, fontsize=8, labelpad=2)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.25), ncol=3, frameon=False,
              labelcolor=TEXT, fontsize=8)
