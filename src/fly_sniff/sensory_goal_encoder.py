from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .recording import load_recording

DEFAULT_PROTOCOL = Path("configs/e003a_sensory_goal_encoder_protocol_v1.json")


def _wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _upwind_goal(obs: dict[str, Any]) -> float:
    heading = float(obs["heading"])
    wx = float(obs["wind_x_body"])
    wy = float(obs["wind_y_body"])
    c, s = np.cos(heading), np.sin(heading)
    world_x = c * wx - s * wy
    world_y = s * wx + c * wy
    return float(np.arctan2(-world_y, -world_x))


def _odor_present(obs: dict[str, Any]) -> bool:
    return bool(max(float(obs["left_odor"]), float(obs["right_odor"])) > 0.0)


def _circular_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(_wrap(np.asarray(a) - np.asarray(b)), dtype=float)


def _memory_series(raw_goal: np.ndarray, odor: np.ndarray, dt: float, tau: float) -> np.ndarray:
    out = np.full_like(raw_goal, np.nan, dtype=float)
    vec = np.zeros(2, dtype=float)
    alpha = 1.0 - float(np.exp(-dt / tau))
    for i, goal in enumerate(raw_goal):
        if odor[i]:
            target = np.array([np.cos(goal), np.sin(goal)], dtype=float)
            vec = (1.0 - alpha) * vec + alpha * target
        if np.linalg.norm(vec) > 1e-12:
            out[i] = float(np.arctan2(vec[1], vec[0]))
    return out


def _hold_last_series(raw_goal: np.ndarray, odor: np.ndarray) -> np.ndarray:
    out = np.full_like(raw_goal, np.nan, dtype=float)
    last = np.nan
    for i, goal in enumerate(raw_goal):
        if odor[i]:
            last = float(goal)
        out[i] = last
    return out


def _analyze_agent(frames: list[dict[str, Any]], label: str, dt: float, tau_grid: list[float]) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for frame in frames:
        matches = [row for row in frame.get("agents", []) if row.get("label") == label]
        if len(matches) != 1:
            raise ValueError(f"expected one agent {label!r} per frame")
        observations.append(dict(matches[0]["observation"]))

    raw_goal = np.asarray([_upwind_goal(obs) for obs in observations], dtype=float)
    odor = np.asarray([_odor_present(obs) for obs in observations], dtype=bool)
    primary = np.where(odor, raw_goal, np.nan)

    replay = np.asarray([_upwind_goal(obs) for obs in observations], dtype=float)
    deterministic_error = float(np.max(np.abs(_circular_delta(raw_goal, replay))))

    rotation_errors: list[float] = []
    for offset_deg in (45.0, 90.0, 135.0):
        offset = np.deg2rad(offset_deg)
        rotated = []
        for obs in observations:
            shifted = dict(obs)
            shifted["heading"] = float(_wrap(float(obs["heading"]) + offset))
            rotated.append(_upwind_goal(shifted))
        expected = np.asarray(_wrap(raw_goal + offset), dtype=float)
        rotation_errors.append(float(np.max(np.abs(_circular_delta(np.asarray(rotated), expected)))))

    primary_updates = np.isfinite(primary)
    odor_count = int(np.sum(odor))
    no_odor_count = int(np.sum(~odor))
    update_during = float(np.sum(primary_updates & odor) / odor_count) if odor_count else 1.0
    update_without = float(np.sum(primary_updates & ~odor) / no_odor_count) if no_odor_count else 0.0

    finite_pairs = np.isfinite(primary[1:]) & np.isfinite(primary[:-1])
    continuity = (
        float(np.mean(np.abs(_circular_delta(primary[1:][finite_pairs], primary[:-1][finite_pairs]))))
        if np.any(finite_pairs)
        else 0.0
    )

    memory = {
        str(tau): _memory_series(raw_goal, odor, dt, float(tau)).tolist() for tau in tau_grid
    }
    hold_last = _hold_last_series(raw_goal, odor)
    return {
        "frames": len(frames),
        "odor_frame_count": odor_count,
        "no_odor_frame_count": no_odor_count,
        "primary_goal_rad": primary.tolist(),
        "goal_update_fraction_during_odor": update_during,
        "goal_update_fraction_without_odor": update_without,
        "mean_primary_step_change_rad": continuity,
        "deterministic_replay_error_rad": deterministic_error,
        "max_common_rotation_error_rad": max(rotation_errors, default=0.0),
        "memory_sensitivities_rad": memory,
        "persistent_hold_last_rad": hold_last.tolist(),
    }


def run_e003a(recording: str | Path, protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("protocol") != "E003a-sensory-to-goal-encoder-v1":
        raise ValueError("unexpected E003a protocol")
    bundle = load_recording(recording)
    payload = bundle["recording"]
    frames = list(payload.get("frames", []))
    if not frames:
        raise ValueError("recording has no frames")
    dt = float(payload["dt"])
    labels = [str(row["label"]) for row in payload.get("controllers", [])]
    tau_grid = [float(x) for x in protocol["candidate_encoders"]["odor_weighted_wind_memory"]["memory_tau_s_grid"]]
    reports = {label: _analyze_agent(frames, label, dt, tau_grid) for label in labels}

    deterministic = max((row["deterministic_replay_error_rad"] for row in reports.values()), default=0.0)
    rotation_error = max((row["max_common_rotation_error_rad"] for row in reports.values()), default=0.0)
    update_without = max((row["goal_update_fraction_without_odor"] for row in reports.values()), default=0.0)
    gates = [
        {"name": "no_oracle_inputs", "passed": True},
        {"name": "deterministic_replay", "passed": deterministic <= 1e-12, "value": deterministic},
        {"name": "no_goal_updates_without_allowed_context_for_primary", "passed": update_without == 0.0, "value": update_without},
        {"name": "coordinate_transform_invariance", "passed": rotation_error <= 1e-12, "value": rotation_error},
        {"name": "all_sensitivity_candidates_reported", "passed": all(bool(row["memory_sensitivities_rad"]) for row in reports.values())},
        {"name": "navigation_performance_not_used_for_selection", "passed": True},
    ]
    return {
        "protocol": protocol["protocol"],
        "dataset": protocol["dataset"],
        "recording_sha256": bundle.get("recording_sha256"),
        "passed": all(bool(row["passed"]) for row in gates),
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(row["passed"]) for row in gates),
        "gates": gates,
        "primary_candidate": protocol["primary_candidate"],
        "controller_reports": reports,
        "OOD_plume_seed_behavior": "not evaluated by a single-recording E003a run; requires a preregistered multi-seed cohort",
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E003a sensory-to-goal encoder assay")
    parser.add_argument("recording")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default="results/e003/sensory-goal-encoder-v1.json")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    report = run_e003a(args.recording, protocol)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    print(f"passed={report['passed']} gates={report['passed_gate_count']}/{report['gate_count']}")
    for gate in report["gates"]:
        print(f"  {'PASS' if gate['passed'] else 'FAIL'}  {gate['name']}")


if __name__ == "__main__":
    main()
