from __future__ import annotations

from typing import Any

import numpy as np

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .odor_motion import BilateralOdorMotionEstimator, OdorMotionConfig
from .odor_motion_edge_assay import SensorReplay, summarize
from .plume import TurbulentPlume


def _antennae(x: float, y: float, heading: float, separation: float):
    half = 0.5 * separation
    return (
        (x - np.sin(heading) * half, y + np.cos(heading) * half),
        (x + np.sin(heading) * half, y - np.cos(heading) * half),
    )


def _expected(heading: float) -> str | None:
    sin_heading = float(np.sin(heading))
    if np.isclose(abs(sin_heading), 1.0, atol=1e-9):
        return "left_to_right" if sin_heading > 0.0 else "right_to_left"
    return None


def run_plume_assay(
    qualification: dict[str, Any],
    motion_config: OdorMotionConfig,
    arena: ArenaConfig | None = None,
    plume_config: PlumeConfig | None = None,
    sensors: SensorConfig | None = None,
) -> dict[str, Any]:
    arena = arena or ArenaConfig()
    plume_config = plume_config or PlumeConfig()
    sensors = sensors or SensorConfig()
    cfg = qualification["fixed_plume_probe"]
    dt = float(arena.dt)
    steps = max(1, round(float(cfg["duration_s"]) / dt))
    probes: list[dict[str, Any]] = []

    for seed in cfg["seeds"]:
        plume = TurbulentPlume(arena, plume_config, int(seed))
        plume.warmup()
        states = []
        for x in cfg["x_positions"]:
            x = float(x)
            if not 0.0 <= x <= arena.width:
                raise ValueError("fixed-plume probe x is outside arena")
            for offset in cfg["y_offsets_from_source"]:
                y = arena.source_y + float(offset)
                if not 0.0 <= y <= arena.height:
                    raise ValueError("fixed-plume probe y is outside arena")
                for heading in cfg["headings_rad"]:
                    heading = float(heading)
                    states.append({
                        "x": x,
                        "y": y,
                        "heading": heading,
                        "expected": _expected(heading),
                        "sensor": SensorReplay(dt, sensors),
                        "estimator": BilateralOdorMotionEstimator(dt=dt, config=motion_config),
                        "estimates": [],
                    })

        for step in range(steps):
            for state in states:
                left_pos, right_pos = _antennae(
                    state["x"], state["y"], state["heading"], arena.antenna_separation
                )
                left_c = plume.concentration(*left_pos)
                right_c = plume.concentration(*right_pos)
                left, right = state["sensor"].sample(left_c, right_c)
                state["estimates"].append(state["estimator"].update(
                    step=step, t=step * dt, left_response=left, right_response=right
                ))
            plume.step()

        for state in states:
            probes.append({
                "seed": int(seed),
                "x": state["x"],
                "y": state["y"],
                "heading_rad": state["heading"],
                "expected_direction_from_mean_downwind_transport": state["expected"],
                **summarize(state["estimates"], state["expected"]),
            })

    longitudinal = [p for p in probes if p["expected_direction_from_mean_downwind_transport"]]
    crosswind = [p for p in probes if not p["expected_direction_from_mean_downwind_transport"]]
    directional = sum(p["directional_estimate_count"] for p in longitudinal)
    total = sum(p["sample_count"] for p in longitudinal)
    correct = sum(
        p["direction_sign_accuracy"] * p["directional_estimate_count"]
        for p in longitudinal
        if p["direction_sign_accuracy"] is not None
    )
    accuracy = None if directional == 0 else float(correct / directional)
    if directional == 0:
        status = "blocked_no_resolvable_plume_timing"
    elif accuracy is not None and accuracy <= 0.5:
        status = "blocked_direction_inconsistent"
    else:
        status = "signal_present_requires_review"
    cross_total = sum(p["sample_count"] for p in crosswind)
    return {
        "status": status,
        "probe_count": len(probes),
        "longitudinal_directional_estimate_count": directional,
        "longitudinal_directional_sample_fraction": float(directional / total) if total else 0.0,
        "longitudinal_direction_sign_accuracy": accuracy,
        "crosswind_directional_sample_fraction": (
            float(sum(p["directional_estimate_count"] for p in crosswind) / cross_total)
            if cross_total else 0.0
        ),
        "probes": probes,
    }
