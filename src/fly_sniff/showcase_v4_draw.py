from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .party_social import (
    CULPRIT_INDEX,
    MUTED,
    PANEL,
    PEOPLE,
    PERSON_COLORS,
    PLUME,
    TEXT,
    _draw_person,
)
from .recorded_showcase import _agent, _badge_style
from .recorded_showcase_v2 import _draw_fly_marker_v2
from .recorded_showcase_v3 import _threshold_summary
from .showcase_v4_layers import blend_activity_color, density_rgba, pfl3_population_frame, viewer_density_field


def draw_room_v4(
    ax,
    payload: dict[str, Any],
    histories: dict[str, np.ndarray],
    frame_index: int,
    *,
    reveal: bool,
    recent_seconds: float = 5.0,
) -> None:
    arena = payload["arena"]
    current = payload["frames"][frame_index]
    colors = {entry["label"]: entry["color"] for entry in payload["controllers"]}

    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, arena["width"])
    ax.set_ylim(0, arena["height"])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.5)

    field = viewer_density_field(current, payload)
    if field is not None:
        density, extent = field
        ax.imshow(density_rgba(density), extent=extent, origin="lower", interpolation="bilinear", zorder=1)
        label = "VIEWER ODOR FIELD • complete recorded plume"
    else:
        plume = np.asarray(current.get("plume", []), dtype=float)
        if plume.size:
            ax.scatter(plume[:, 0], plume[:, 1], s=12, c=PLUME, alpha=0.3, zorder=1)
        label = "VIEWER ODOR FIELD • sampled plume fallback"
    ax.text(0.5, 0.982, label, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color="#D9F99D", fontweight="bold", bbox=_badge_style(alpha=0.76))

    for index, (x, y, tag) in enumerate(PEOPLE):
        _draw_person(ax, x, y, tag, color=PERSON_COLORS[index % len(PERSON_COLORS)], culprit=reveal and index == CULPRIT_INDEX)

    recent_steps = max(2, int(round(recent_seconds / float(payload["dt"]))))
    for controller in payload["controllers"][:2]:
        label_name = controller["label"]
        color = colors[label_name]
        history = histories[label_name][: frame_index + 1]
        if len(history) > 1:
            ax.plot(history[:, 0], history[:, 1], color=color, linewidth=1.4, alpha=0.15, zorder=5)
            recent = history[max(0, len(history) - recent_steps) :]
            ax.plot(recent[:, 0], recent[:, 1], color=color, linewidth=4.0, alpha=0.9, zorder=6)
        _draw_fly_marker_v2(ax, _agent(current, label_name), color)

    first, second = payload["controllers"][:2]
    first_state = _agent(current, first["label"])
    second_state = _agent(current, second["label"])
    ax.text(0.022, 0.94, "DEVELOPMENT PROXY", transform=ax.transAxes, va="top", fontsize=10.0, color=first["color"], fontweight="bold", bbox=_badge_style())
    ax.text(0.978, 0.94, "RANDOM CONTROL", transform=ax.transAxes, ha="right", va="top", fontsize=10.0, color=second["color"], fontweight="bold", bbox=_badge_style())
    ax.text(0.022, 0.03, f"{current['t']:04.1f}s • proxy {first_state['distance_to_source']:.1f} m away", transform=ax.transAxes, va="bottom", fontsize=8.2, color=TEXT, fontweight="bold", bbox=_badge_style(alpha=0.82))
    ax.text(0.978, 0.03, f"control {second_state['distance_to_source']:.1f} m away", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2, color=TEXT, fontweight="bold", bbox=_badge_style(alpha=0.82))


def draw_population_panel(
    ax,
    config: dict[str, Any],
    e002c: dict[str, Any],
    fc2: dict[str, Any],
    probe_index: int,
) -> None:
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    threshold = int(config["mechanism_probe"]["display_threshold"])
    _, columns, values, scale = pfl3_population_frame(e002c, fc2, threshold=threshold, probe_index=probe_index)
    ax.text(0.02, 0.92, "SEALED MODEL PROBE • SEPARATE FROM CHASE TIMELINE", color="#FDE68A", fontsize=9.1, fontweight="bold")
    ax.text(0.98, 0.92, "MODELED PFL3 STATE • NOT MEASURED FIRING", ha="right", color=MUTED, fontsize=7.0, fontweight="bold")
    ax.text(0.02, 0.77, "FC2 goal interface + direct EPG heading interface → 24-cell PFL3 state", color=TEXT, fontsize=7.7, fontweight="bold")
    ax.text(0.98, 0.77, f"E002c 8/8 • w≥{threshold} • probe step {probe_index + 1}/32", ha="right", color=MUTED, fontsize=6.9)

    left, right = 0.18, 0.98
    cell_w = (right - left) / 24.0
    row_y = {"goal_only": 0.56, "heading_only": 0.39, "joint": 0.22}
    styles = {"goal_only": ("goal", "#C4B5FD"), "heading_only": ("heading", "#67E8F9"), "joint": ("joint", "#FDE68A")}

    for col in sorted(set(columns)):
        idx = [i for i, value in enumerate(columns) if value == col]
        start = left + min(idx) * cell_w
        center = left + (min(idx) + max(idx) + 1) * 0.5 * cell_w
        ax.axvline(start, ymin=0.15, ymax=0.68, color="#475569", linewidth=0.6, alpha=0.7)
        ax.text(center, 0.67, f"C{col}", ha="center", va="center", color=MUTED, fontsize=6.0)

    for name in ("goal_only", "heading_only", "joint"):
        label, color = styles[name]
        y = row_y[name]
        ax.text(0.02, y, label, va="center", color=color, fontsize=7.3, fontweight="bold")
        for j, value in enumerate(values[name]):
            x0 = left + j * cell_w + 0.002
            x1 = left + (j + 1) * cell_w - 0.002
            ax.add_patch(plt.Rectangle((x0, y - 0.05), x1 - x0, 0.10, facecolor=blend_activity_color(color, float(value / scale)), edgecolor="#334155", linewidth=0.35))

    badges = " • ".join(f"w≥{t}: {changed}/{reachable}" for t, changed, reachable in _threshold_summary(e002c))
    ax.text(0.02, 0.07, "joint-changed / dual-reachable  " + badges, color=MUTED, fontsize=6.4)
    ax.text(0.98, 0.07, "FC2 A/B/C: C1-C9 parsed 100% • phase→angle unresolved", ha="right", color="#C4B5FD", fontsize=6.4, fontweight="bold")
