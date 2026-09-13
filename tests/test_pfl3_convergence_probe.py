from __future__ import annotations

import copy

import pytest

from fly_sniff.pfl3_convergence_probe import (
    build_convergence_bundle,
    probe_pfl3_convergence,
)


def _population(body_id: int, cell_type: str, instance: str) -> dict:
    return {
        "count": 1,
        "rows": [{"bodyId": body_id, "type": cell_type, "instance": instance}],
    }


def _direct(source: str, target: str, source_id: int, target_id: int) -> dict:
    return {
        "source": source,
        "target": target,
        "observed_edge_pairs": 1,
        "observed_weight_sum": 10.0,
        "edges": [{"source": source_id, "target": target_id, "weight": 10.0}],
    }


def _fixtures() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    route = {
        "dataset": "male-cns:v1.0",
        "populations": {
            "FB5AB": _population(1, "FB5AB", "FB5AB_R"),
            "PFNa_family": _population(2, "PFNa", "PFNa_R1_C1"),
            "PFNm_family": _population(3, "PFNm_a", "PFNm_a_R1_C1"),
            "PFNp_family": _population(4, "PFNp_b", "PFNp_b"),
            "hDeltaC": _population(5, "hDeltaC", "hDeltaC_01_C1"),
            "hDeltaG": _population(6, "hDeltaG", "hDeltaG_01_C1"),
            "PFL3": _population(7, "PFL3", "PFL3(PB12c)_R1_C1"),
            "PFL2": _population(8, "PFL2", "PFL2_C1"),
        },
        "direct_predictions": [
            _direct("FB5AB", "hDeltaC", 1, 5),
            _direct("PFNa_family", "hDeltaC", 2, 5),
            _direct("PFNm_family", "hDeltaC", 3, 5),
            _direct("PFNp_family", "hDeltaC", 4, 5),
            _direct("hDeltaC", "hDeltaG", 5, 6),
            _direct("hDeltaG", "PFL3", 6, 7),
            _direct("hDeltaG", "PFL2", 6, 8),
        ],
    }
    sign_report = {
        "population_signs": {
            name: {
                "body_ids": [body_id],
                "type_evidence": {cell_type: {"modeled_sign": 1}},
            }
            for name, body_id, cell_type in (
                ("FB5AB", 1, "FB5AB"),
                ("PFNa_family", 2, "PFNa"),
                ("PFNm_family", 3, "PFNm_a"),
                ("PFNp_family", 4, "PFNp_b"),
                ("hDeltaC", 5, "hDeltaC"),
                ("hDeltaG", 6, "hDeltaG"),
            )
        },
        "mandatory_sensitivity_cases": [{"affected_body_ids": [4]}],
    }
    integration_topography = {
        "populations": {
            "PFNa_family": {"body_columns": {"2": 1}},
            "PFNm_family": {"body_columns": {"3": 1}},
            "hDeltaG": {"body_columns": {"6": 1}},
        }
    }
    heading = {
        "dataset": "male-cns:v1.0",
        "populations": {
            "EPG": _population(9, "EPG", "EPG(PB08)_R1"),
        },
        "direct_predictions": [
            {
                "source": "EPG",
                "target": "PFL3",
                "edges": [
                    {
                        "source": 9,
                        "target": 7,
                        "weight": 10.0,
                        "source_instance": "EPG(PB08)_R1",
                        "target_instance": "PFL3(PB12c)_R1_C1",
                    }
                ],
            }
        ],
        "two_hop_predictions": [
            {
                "source": "EPG",
                "via": "Delta7",
                "target": "PFL3",
                "threshold_sweep": [
                    {"min_weight_each_edge": 1.0, "path_count": 3},
                    {"min_weight_each_edge": 10.0, "path_count": 1},
                ],
            }
        ],
    }
    heading_sign = {
        "predictions": {
            "EPG": {"modeled_sign": 1},
            "Delta7": {"modeled_sign": 0},
        }
    }
    qualification = {"qualification_ready": True}
    protocol = {
        "protocol": "E002c-pfl3-goal-heading-convergence-v1",
        "fixed_structural_thresholds": [1, 10],
        "model_normalization_caveat": "toy caveat",
        "claim_boundary": "toy claim boundary",
    }
    return (
        route,
        sign_report,
        integration_topography,
        heading,
        heading_sign,
        qualification,
        protocol,
    )


def test_convergence_bundle_excludes_delta7_and_builds_anatomical_probe_roles() -> None:
    route, signs, topo, heading, heading_sign, _, _ = _fixtures()
    bundle = build_convergence_bundle(route, signs, topo, heading, heading_sign, 10.0)
    assert "EPG->PFL3" in set(bundle.edges.edge_family)
    assert "probe_EPG_PB_R1" in bundle.roles
    assert bundle.roles["probe_hDeltaG_C1"] == [6]
    assert not any("Delta7" in family for family in bundle.edges.edge_family)


def test_e002c_probe_passes_causal_convergence_gates_on_toy_graph() -> None:
    route, signs, topo, heading, heading_sign, qualification, protocol = _fixtures()
    report = probe_pfl3_convergence(
        route,
        signs,
        topo,
        heading,
        heading_sign,
        qualification,
        protocol,
        seed=22003,
        steps=12,
        pulse_steps=4,
        drive_amplitude=1.0,
        artifact_hash_gate=(True, {"fixture": True}),
    )
    assert report["passed"] is True
    assert report["passed_gate_count"] == report["gate_count"] == 8
    gates = {row["name"]: row for row in report["gates"]}
    assert gates["deterministic_replay"]["value"] == 0.0
    assert gates["heading_lane_cut"]["value"] == 0.0
    assert gates["goal_lane_cut"]["value"] == 0.0
    threshold_10 = report["threshold_reports"]["10"]
    assert threshold_10["goal_only"]["deterministic_replay_error"] == 0.0
    assert threshold_10["heading_only"]["deterministic_replay_error"] == 0.0
    assert threshold_10["joint"]["deterministic_replay_error"] == 0.0
    assert threshold_10["heading_lane_cut"]["deterministic_replay_error"] == 0.0
    assert threshold_10["goal_lane_cut"]["deterministic_replay_error"] == 0.0
    assert threshold_10["joint_convergence"]["dual_reachable_body_ids"] == [7]
    assert threshold_10["joint_convergence"]["changed_from_both_count"] == 1
    assert threshold_10["Delta7_structural_only"]["modeled_sign"] == 0
    assert threshold_10["normalization"]["mean_ratio"] > 1.0


def test_e002c_rejects_delta7_sign_promotion() -> None:
    route, signs, topo, heading, heading_sign, qualification, protocol = _fixtures()
    promoted = copy.deepcopy(heading_sign)
    promoted["predictions"]["Delta7"]["modeled_sign"] = -1
    with pytest.raises(ValueError, match="Delta7 sign policy drift"):
        probe_pfl3_convergence(
            route,
            signs,
            topo,
            heading,
            promoted,
            qualification,
            protocol,
            seed=22003,
            steps=12,
            pulse_steps=4,
            drive_amplitude=1.0,
            artifact_hash_gate=(True, {"fixture": True}),
        )
