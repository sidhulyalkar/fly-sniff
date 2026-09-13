from __future__ import annotations

from typing import Any

import numpy as np

from .config import ArenaConfig, SensorConfig
from .odor_motion import BilateralOdorMotionEstimator, OdorMotionConfig

DIRECTIONAL = {"left_to_right", "right_to_left"}


class SensorReplay:
    """Stationary replay of FlySniffEnv's frozen phenomenological transduction."""

    def __init__(self, dt: float, config: SensorConfig):
        self.dt = float(dt)
        self.config = config
        self.adapt = [0.0, 0.0]

    def _side(self, concentration: float, index: int) -> float:
        raw = max(0.0, self.config.concentration_gain * float(concentration))
        sat = raw / (self.config.concentration_half_sat + raw + 1e-12)
        response = 0.72 * sat + 0.28 * max(sat - self.adapt[index], 0.0)
        tau = float(self.config.adaptation_tau)
        alpha = 1.0 if tau <= 0.0 else 1.0 - np.exp(-self.dt / tau)
        self.adapt[index] += alpha * (sat - self.adapt[index])
        return float(np.clip(response, 0.0, 1.0))

    def sample(self, left: float, right: float) -> tuple[float, float]:
        return self._side(left, 0), self._side(right, 1)


def summarize(estimates, expected: str | None) -> dict[str, Any]:
    directional = [item for item in estimates if item.direction in DIRECTIONAL]
    return {
        "sample_count": len(estimates),
        "directional_estimate_count": len(directional),
        "directional_sample_fraction": float(len(directional) / len(estimates)) if estimates else 0.0,
        "direction_sign_accuracy": (
            None
            if expected is None or not directional
            else float(sum(item.direction == expected for item in directional) / len(directional))
        ),
        "mean_abs_evidence": float(np.mean([abs(item.evidence) for item in estimates])) if estimates else 0.0,
        "mean_confidence": float(np.mean([item.confidence for item in estimates])) if estimates else 0.0,
    }


def run_edge_assay(
    qualification: dict[str, Any],
    motion_config: OdorMotionConfig,
    arena: ArenaConfig | None = None,
    sensors: SensorConfig | None = None,
) -> dict[str, Any]:
    arena = arena or ArenaConfig()
    sensors = sensors or SensorConfig()
    cfg = qualification["traveling_edge"]
    dt = float(arena.dt)
    pre = max(1, round(float(cfg["pre_stimulus_s"]) / dt))
    width = max(1, round(float(cfg["whiff_duration_s"]) / dt))
    post = max(1, round(float(cfg["post_stimulus_s"]) / dt))
    concentration = float(cfg["concentration"])
    trials: list[dict[str, Any]] = []

    for latency in cfg["latencies_s"]:
        delay = max(1, round(float(latency) / dt))
        for expected in ("left_to_right", "right_to_left"):
            n = pre + delay + width + post
            left_c, right_c = np.zeros(n), np.zeros(n)
            first, second = (left_c, right_c) if expected == "left_to_right" else (right_c, left_c)
            first[pre : pre + width] = concentration
            second[pre + delay : pre + delay + width] = concentration
            sensor = SensorReplay(dt, sensors)
            estimator = BilateralOdorMotionEstimator(dt=dt, config=motion_config)
            estimates = []
            for index, (lc, rc) in enumerate(zip(left_c, right_c, strict=True)):
                left, right = sensor.sample(float(lc), float(rc))
                estimates.append(estimator.update(
                    step=index, t=index * dt, left_response=left, right_response=right
                ))
            trials.append({
                "requested_latency_s": float(latency),
                "realized_latency_s": float(delay * dt),
                "expected_direction": expected,
                **summarize(estimates, expected),
            })

    has_direction = all(row["directional_estimate_count"] > 0 for row in trials)
    correct = all(row["direction_sign_accuracy"] == 1.0 for row in trials)
    return {
        "status": "pass" if has_direction and correct else "fail",
        "all_trials_have_directional_estimate": has_direction,
        "all_qualified_estimates_have_correct_sign": correct,
        "trials": trials,
    }
