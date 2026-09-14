from __future__ import annotations

from fly_sniff.e002e_qualification import qualify_e002e_v2


def _fixture() -> tuple[dict, dict, dict[str, str]]:
    pairs = [[-30, 30], [-60, 60]]
    reports = {
        "-60": {"turn": 0.4},
        "-30": {"turn": 0.2},
        "0": {"turn": -0.01},
        "30": {"turn": -0.2},
        "60": {"turn": -0.4},
    }
    result = {
        "protocol": "E002e-pfl3-descending-steering-v2",
        "dataset": "male-cns:v1.0",
        "passed": True,
        "passed_gate_count": 10,
        "gate_count": 10,
        "primary_structural_threshold": 5,
        "run_config": {"seed": 13014, "steps": 32, "confirmation_seed_frozen": True},
        "gates": [
            {"name": name, "passed": True}
            for name in [
                "sealed_inputs_match",
                "no_behavior_oracle_inputs",
                "deterministic_replay_le_1e-12",
                "mirrored_phase_pairs_produce_opposite_turn_signs",
                "FC2_lane_cut_destroys_phase_dependent_turning",
                "EPG_lane_cut_destroys_phase_dependent_turning",
                "PFL3_output_cut_zeroes_DNa02_turn",
                "left_right_group_swap_reverses_nonzero_mirrored_phases",
                "common_rotation_offset_invariance",
                "all_structural_thresholds_reported",
            ]
        ],
        "threshold_reports": {
            "1": {"dual_reachable_count": 24, "phase_reports": reports},
            "3": {"dual_reachable_count": 24, "phase_reports": reports},
            "5": {"dual_reachable_count": 24, "phase_reports": reports},
            "10": {"dual_reachable_count": 21, "phase_reports": reports},
        },
        "input_sha256": {"protocol": "a", "e002d": "b", "crosswalk": "c", "bundle_manifest": "d", "bundle_edges": "e"},
    }
    result["gates"][7].update(
        {
            "required_phase_count": 10,
            "reversed_phase_count": 10,
            "zero_phase": {"gate_role": "descriptive_baseline_only"},
        }
    )
    protocol = {
        "protocol": "E002e-pfl3-descending-steering-v2",
        "confirmation_seed": 13014,
        "primary_structural_threshold": 5,
        "all_structural_thresholds": [1, 3, 5, 10],
        "mirrored_phase_pairs_deg": pairs,
        "qualification_gates": [row["name"] for row in result["gates"]],
    }
    return result, protocol, dict(result["input_sha256"])


def test_e002e_qualification_seals_primary_result_and_reports_sensitivity() -> None:
    result, protocol, hashes = _fixture()
    report = qualify_e002e_v2(
        result,
        protocol,
        actual_input_sha256=hashes,
        result_sha256="result",
        expected_git_sha="abc",
        git_branch="feat/science-rigor-v1",
        git_sha="abc",
        git_dirty_paths=[],
    )
    assert report["qualification_ready"] is True
    assert all(row["passed"] for row in report["checks"])
    assert report["robustness"]["primary_threshold"] == 5


def test_e002e_qualification_rejects_hash_mismatch() -> None:
    result, protocol, hashes = _fixture()
    hashes["e002d"] = "different"
    report = qualify_e002e_v2(
        result,
        protocol,
        actual_input_sha256=hashes,
        result_sha256="result",
        expected_git_sha="abc",
        git_branch="feat/science-rigor-v1",
        git_sha="abc",
        git_dirty_paths=[],
    )
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["input_hashes_match_local_authorities"]["passed"] is False
    assert report["qualification_ready"] is False
