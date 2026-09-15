from __future__ import annotations

import copy

from fly_sniff.e002b_qualification import qualify_e002b_result


def _fixture() -> tuple[dict, dict]:
    result = {
        "protocol": "E002b-goal-channel-propagation-v1",
        "dataset": "male-cns:v1.0",
        "passed": True,
        "passed_gate_count": 7,
        "gate_count": 7,
        "run_config": {
            "seed": 22002,
            "steps": 32,
            "pulse_steps": 8,
            "drive_amplitude": 1.0,
            "thresholds": [1.0, 3.0, 5.0, 10.0],
        },
        "threshold_reports": {
            str(t): {
                "hDeltaM_relay_comparator": {
                    "status": "modeled",
                    "selection_status": "frozen_comparator_not_eligible_for_promotion",
                },
                "PFNp_b_sign_zero_sensitivity": {},
            }
            for t in (1, 3, 5, 10)
        },
    }
    runtime = {
        "seed": 22002,
        "steps": 32,
        "pulse_steps": 8,
        "drive_amplitude": 1.0,
        "structural_thresholds": [1, 3, 5, 10],
        "frozen_before_first_real_probe": True,
    }
    return result, runtime


def test_qualification_requires_science_runtime_comparator_and_clean_git() -> None:
    result, runtime = _fixture()
    report = qualify_e002b_result(
        result,
        runtime,
        source_result_sha256="abc",
        expected_git_sha="deadbeef",
        git_branch="feat/science-rigor-v1",
        git_sha="deadbeef",
        git_dirty_paths=[],
    )
    assert report["qualification_ready"] is True
    assert all(row["passed"] for row in report["checks"])


def test_qualification_rejects_dirty_runtime_without_changing_science_gates() -> None:
    result, runtime = _fixture()
    report = qualify_e002b_result(
        result,
        runtime,
        source_result_sha256="abc",
        expected_git_sha="deadbeef",
        git_branch="feat/science-rigor-v1",
        git_sha="deadbeef",
        git_dirty_paths=[" M src/fly_sniff/goal_channel_probe.py"],
    )
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["preregistered_science_gates_pass"]["passed"] is True
    assert checks["clean_exact_git_runtime"]["passed"] is False
    assert report["qualification_ready"] is False


def test_qualification_rejects_posthoc_comparator_promotion() -> None:
    result, runtime = _fixture()
    promoted = copy.deepcopy(result)
    promoted["threshold_reports"]["5"]["hDeltaM_relay_comparator"][
        "selection_status"
    ] = "promoted_primary"
    report = qualify_e002b_result(
        promoted,
        runtime,
        source_result_sha256="abc",
        expected_git_sha="deadbeef",
        git_branch="feat/science-rigor-v1",
        git_sha="deadbeef",
        git_dirty_paths=[],
    )
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["hDeltaM_selection_remains_frozen"]["passed"] is False
    assert report["qualification_ready"] is False
