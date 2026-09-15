from __future__ import annotations

import numpy as np

from fly_sniff.pfl3_phase_probe import _population_curve, probe_phase_comparison


def _inputs():
    goal_map = [0.0, -45.0, -90.0, -135.0, -180.0, 135.0, 90.0, 45.0, 0.0]
    records = []
    for index in range(12):
        column = (index % 9) + 1
        goal = goal_map[column - 1]
        records.append(
            {
                "body_id": 100 + index,
                "column": column,
                "readout_side": "turn_drive_left",
                "heading_phase_deg": goal + 67.5,
                "goal_phase_sensitivity_deg": goal,
            }
        )
    for index in range(12):
        column = (index % 9) + 1
        goal = goal_map[column - 1]
        records.append(
            {
                "body_id": 112 + index,
                "column": column,
                "readout_side": "turn_drive_right",
                "heading_phase_deg": goal - 67.5,
                "goal_phase_sensitivity_deg": goal,
            }
        )
    body_ids = [row["body_id"] for row in records]
    protocol = {
        "protocol": "E002d-pfl3-relative-phase-comparison-v3",
        "fixed_published_parameters": {"d": 0.63, "a_hz": 29.23, "b": 2.17, "c": -0.7},
        "mapping": {"absolute_world_offset": "nuisance-parameter", "status": "candidate-literature-crosswalk"},
        "claim_boundary": "test boundary",
    }
    runtime = {
        "protocol": "E002d-phase-probe-runtime-v1",
        "primary_structural_threshold": 5,
        "all_structural_thresholds": [1, 3, 5, 10],
        "relative_phase_start_deg": -180,
        "relative_phase_stop_deg_exclusive": 180,
        "relative_phase_step_deg": 5,
        "common_rotation_offsets_deg": [0, 45, 90, 135],
        "deterministic_tolerance": 1e-12,
        "rotation_invariance_tolerance": 1e-10,
        "minimum_curve_range": 1e-9,
    }
    crosswalk = {
        "protocol": "E002d-pfl3-phase-crosswalk-v1",
        "ready_for_phase_probe": True,
        "absolute_world_offset_status": "unresolved-nuisance-parameter",
        "records": records,
    }
    e002c = {
        "protocol": "E002c-pfl3-goal-heading-convergence-v1",
        "passed": True,
        "threshold_reports": {
            str(threshold): {"joint_convergence": {"dual_reachable_body_ids": body_ids}}
            for threshold in (1, 3, 5, 10)
        },
    }
    return protocol, runtime, crosswalk, e002c


def test_phase_probe_passes_symmetric_reference_fixture() -> None:
    report = probe_phase_comparison(*_inputs())
    assert report["passed"]
    assert report["passed_gate_count"] == report["gate_count"] == 8
    primary = report["threshold_reports"]["5"]
    assert primary["right_minus_left_zero_crossing"]
    assert primary["right_minus_left_range"] > 1.0
    assert report["max_common_rotation_offset_error"] <= 1e-10


def test_common_coordinate_offset_does_not_change_curve() -> None:
    protocol, _, crosswalk, _ = _inputs()
    records = crosswalk["records"]
    phases = np.arange(-180.0, 180.0, 5.0)
    ids = {int(row["body_id"]) for row in records}
    params = protocol["fixed_published_parameters"]
    base = _population_curve(records, ids, phases, params)
    shifted = _population_curve(records, ids, phases, params, common_offset_deg=91.0)
    np.testing.assert_allclose(base["right_minus_left"], shifted["right_minus_left"], atol=1e-12, rtol=0.0)


def test_lane_cuts_are_not_silently_identical_to_joint_model() -> None:
    report = probe_phase_comparison(*_inputs())
    primary = report["threshold_reports"]["5"]
    assert primary["joint_differs_from_FC2_cut"]
    assert primary["joint_differs_from_EPG_cut"]
