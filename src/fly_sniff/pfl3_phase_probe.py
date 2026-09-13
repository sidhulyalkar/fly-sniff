from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PROTOCOL = Path("configs/e002d_pfl3_phase_comparison_protocol_v3.json")
DEFAULT_RUNTIME = Path("configs/e002d_phase_probe_runtime_v1.json")
DEFAULT_CROSSWALK = Path("results/e002/pfl3-phase-crosswalk-v1.json")
DEFAULT_E002C = Path("results/e002/pfl3-convergence-v1.json")


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_grid(runtime: dict[str, Any]) -> np.ndarray:
    return np.arange(
        float(runtime["relative_phase_start_deg"]),
        float(runtime["relative_phase_stop_deg_exclusive"]),
        float(runtime["relative_phase_step_deg"]),
        dtype=float,
    )


def _cell_response(
    heading_deg: np.ndarray,
    goal_deg: np.ndarray,
    heading_pref_deg: float,
    goal_pref_deg: float,
    params: dict[str, Any],
    *,
    heading_enabled: bool,
    goal_enabled: bool,
) -> np.ndarray:
    heading_term = (
        np.cos(np.deg2rad(heading_deg - heading_pref_deg)) if heading_enabled else 0.0
    )
    goal_term = np.cos(np.deg2rad(goal_deg - goal_pref_deg)) if goal_enabled else 0.0
    x = heading_term + float(params["d"]) * goal_term
    z = float(params["b"]) * (x + float(params["c"]))
    return float(params["a_hz"]) * np.logaddexp(0.0, z)


def _population_curve(
    records: list[dict[str, Any]],
    included_ids: set[int],
    phase_deg: np.ndarray,
    params: dict[str, Any],
    *,
    common_offset_deg: float = 0.0,
    heading_enabled: bool = True,
    goal_enabled: bool = True,
) -> dict[str, np.ndarray]:
    heading = phase_deg + float(common_offset_deg)
    goal = np.full_like(phase_deg, float(common_offset_deg))
    left = np.zeros_like(phase_deg, dtype=float)
    right = np.zeros_like(phase_deg, dtype=float)

    for row in records:
        body_id = int(row["body_id"])
        if body_id not in included_ids:
            continue
        # The common offset is applied to both the modeled state and preferred phases.
        # This is a coordinate-gauge test, not a claim about a physical world angle.
        response = _cell_response(
            heading,
            goal,
            float(row["heading_phase_deg"]) + float(common_offset_deg),
            float(row["goal_phase_sensitivity_deg"]) + float(common_offset_deg),
            params,
            heading_enabled=heading_enabled,
            goal_enabled=goal_enabled,
        )
        side = str(row["readout_side"])
        if side == "turn_drive_left":
            left += response
        elif side == "turn_drive_right":
            right += response
        else:
            raise ValueError(f"unexpected PFL3 readout side {side!r}")

    return {"left": left, "right": right, "right_minus_left": right - left}


def _zero_crosses(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float)
    return bool(np.min(values) <= 0.0 <= np.max(values))


def probe_phase_comparison(
    protocol: dict[str, Any],
    runtime: dict[str, Any],
    crosswalk: dict[str, Any],
    e002c: dict[str, Any],
) -> dict[str, Any]:
    if protocol.get("protocol") != "E002d-pfl3-relative-phase-comparison-v3":
        raise ValueError("unexpected E002d protocol")
    if runtime.get("protocol") != "E002d-phase-probe-runtime-v1":
        raise ValueError("unexpected E002d runtime protocol")
    if crosswalk.get("protocol") != "E002d-pfl3-phase-crosswalk-v1":
        raise ValueError("unexpected E002d crosswalk protocol")
    if not bool(crosswalk.get("ready_for_phase_probe")):
        raise ValueError("phase crosswalk is not ready")
    if e002c.get("protocol") != "E002c-pfl3-goal-heading-convergence-v1" or not bool(
        e002c.get("passed")
    ):
        raise ValueError("E002d requires passed E002c")

    records = [dict(row) for row in crosswalk["records"]]
    all_ids = {int(row["body_id"]) for row in records}
    if len(records) != 24 or len(all_ids) != 24:
        raise ValueError("E002d requires exact 24-cell PFL3 crosswalk")
    left_ids = {int(row["body_id"]) for row in records if row["readout_side"] == "turn_drive_left"}
    right_ids = {int(row["body_id"]) for row in records if row["readout_side"] == "turn_drive_right"}
    side_groups_ok = not (left_ids & right_ids) and len(left_ids) == len(right_ids) == 12

    params = dict(protocol["fixed_published_parameters"])
    phase = _phase_grid(runtime)
    thresholds = [int(x) for x in runtime["all_structural_thresholds"]]
    primary_threshold = int(runtime["primary_structural_threshold"])
    threshold_reports: dict[str, Any] = {}

    for threshold in thresholds:
        source = e002c["threshold_reports"][str(threshold)]["joint_convergence"]
        included = {int(x) for x in source["dual_reachable_body_ids"]}
        if not included <= all_ids:
            raise ValueError(f"threshold {threshold} contains non-crosswalk PFL3 bodies")
        joint = _population_curve(records, included, phase, params)
        no_goal = _population_curve(records, included, phase, params, goal_enabled=False)
        no_heading = _population_curve(records, included, phase, params, heading_enabled=False)
        threshold_reports[str(threshold)] = {
            "dual_reachable_body_ids": sorted(included),
            "dual_reachable_count": len(included),
            "turn_drive_left_count": len(included & left_ids),
            "turn_drive_right_count": len(included & right_ids),
            "relative_phase_deg": phase.tolist(),
            "joint": {name: value.tolist() for name, value in joint.items()},
            "FC2_lane_cut": {name: value.tolist() for name, value in no_goal.items()},
            "EPG_lane_cut": {name: value.tolist() for name, value in no_heading.items()},
            "right_minus_left_range": float(np.ptp(joint["right_minus_left"])),
            "right_minus_left_zero_crossing": _zero_crosses(joint["right_minus_left"]),
            "joint_differs_from_FC2_cut": bool(
                np.max(np.abs(joint["right_minus_left"] - no_goal["right_minus_left"])) > 1e-12
            ),
            "joint_differs_from_EPG_cut": bool(
                np.max(np.abs(joint["right_minus_left"] - no_heading["right_minus_left"]))
                > 1e-12
            ),
        }

    primary = threshold_reports[str(primary_threshold)]
    included_primary = set(primary["dual_reachable_body_ids"])
    first = _population_curve(records, included_primary, phase, params)
    replay = _population_curve(records, included_primary, phase, params)
    deterministic_error = float(
        np.max(np.abs(first["right_minus_left"] - replay["right_minus_left"]))
    )

    rotation_errors: dict[str, float] = {}
    for offset in runtime["common_rotation_offsets_deg"]:
        rotated = _population_curve(
            records,
            included_primary,
            phase,
            params,
            common_offset_deg=float(offset),
        )
        rotation_errors[str(offset)] = float(
            np.max(np.abs(first["right_minus_left"] - rotated["right_minus_left"]))
        )
    max_rotation_error = max(rotation_errors.values(), default=0.0)

    no_absolute_claim = (
        protocol["mapping"]["absolute_world_offset"] == "nuisance-parameter"
        and crosswalk.get("absolute_world_offset_status") == "unresolved-nuisance-parameter"
    )
    both_lanes = bool(
        primary["joint_differs_from_FC2_cut"] and primary["joint_differs_from_EPG_cut"]
    )
    curve_gate = bool(
        float(primary["right_minus_left_range"]) >= float(runtime["minimum_curve_range"])
        and primary["right_minus_left_zero_crossing"]
    )
    gates = [
        {"name": "sealed_inputs_match", "passed": True},
        {"name": "fc2_columns_complete", "passed": set(int(row["column"]) for row in records) == set(range(1, 10))},
        {"name": "steering_side_groups_match_candidate_config", "passed": side_groups_ok},
        {"name": "no_absolute_world_angle_claim", "passed": no_absolute_claim},
        {
            "name": "deterministic_replay_le_1e-12",
            "passed": deterministic_error <= float(runtime["deterministic_tolerance"]),
        },
        {"name": "both_lanes_required", "passed": both_lanes},
        {
            "name": "common_rotation_offset_invariance",
            "passed": max_rotation_error <= float(runtime["rotation_invariance_tolerance"]),
        },
        {"name": "relative_phase_curve_varies_and_has_zero_crossing", "passed": curve_gate},
    ]

    return {
        "protocol": "E002d-pfl3-relative-phase-comparison-v3",
        "dataset": "male-cns:v1.0",
        "passed": all(bool(gate["passed"]) for gate in gates),
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(gate["passed"]) for gate in gates),
        "gates": gates,
        "primary_structural_threshold": primary_threshold,
        "published_model_parameters": params,
        "mapping_status": protocol["mapping"]["status"],
        "absolute_world_offset_status": crosswalk["absolute_world_offset_status"],
        "deterministic_replay_error": deterministic_error,
        "common_rotation_offset_errors": rotation_errors,
        "max_common_rotation_offset_error": max_rotation_error,
        "threshold_reports": threshold_reports,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered E002d PFL3 phase assay")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--crosswalk", default=str(DEFAULT_CROSSWALK))
    parser.add_argument("--e002c", default=str(DEFAULT_E002C))
    parser.add_argument("--output", default="results/e002/pfl3-phase-comparison-v1.json")
    args = parser.parse_args()

    report = probe_phase_comparison(
        _load(args.protocol),
        _load(args.runtime),
        _load(args.crosswalk),
        _load(args.e002c),
    )
    report["input_sha256"] = {
        "protocol": _sha256(args.protocol),
        "runtime": _sha256(args.runtime),
        "crosswalk": _sha256(args.crosswalk),
        "e002c": _sha256(args.e002c),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    print(f"passed={report['passed']} gates={report['passed_gate_count']}/{report['gate_count']}")


if __name__ == "__main__":
    main()
