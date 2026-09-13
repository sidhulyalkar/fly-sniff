from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from fly_sniff.goal_channel_probe import build_primary_bundle, probe_goal_channel


def _population(body_id: int, cell_type: str, instance: str) -> dict:
    return {
        "count": 1,
        "rows": [{"bodyId": body_id, "type": cell_type, "instance": instance}],
    }


def _direct(source: str, target: str, source_id: int, target_id: int) -> dict:
    return {
        "source": source,
        "target": target,
        "edges": [
            {
                "source": source_id,
                "target": target_id,
                "weight": 10.0,
            }
        ],
    }


def _toy_inputs() -> tuple[dict, dict, dict, dict, dict]:
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
        "ready_for_modeled_sign_probe": True,
        "ready_with_sensitivity_requirements": True,
        "population_signs": {
            "FB5AB": {
                "body_ids": [1],
                "type_evidence": {"FB5AB": {"modeled_sign": 1}},
            },
            "PFNa_family": {
                "body_ids": [2],
                "type_evidence": {"PFNa": {"modeled_sign": 1}},
            },
            "PFNm_family": {
                "body_ids": [3],
                "type_evidence": {"PFNm_a": {"modeled_sign": 1}},
            },
            "PFNp_family": {
                "body_ids": [4],
                "type_evidence": {"PFNp_b": {"modeled_sign": 1}},
            },
            "hDeltaC": {
                "body_ids": [5],
                "type_evidence": {"hDeltaC": {"modeled_sign": 1}},
            },
            "hDeltaG": {
                "body_ids": [6],
                "type_evidence": {"hDeltaG": {"modeled_sign": 1}},
            },
        },
        "mandatory_sensitivity_cases": [
            {
                "name": "PFNp_b_unresolved_sign",
                "affected_body_ids": [4],
                "rule": "set modeled sign 0",
            }
        ],
    }
    topography = {
        "directional_role_status": "unresolved",
        "populations": {
            "PFNa_family": {"body_columns": {"2": 1}},
            "PFNm_family": {"body_columns": {"3": 1}},
            "PFNp_family": {"body_columns": {}, "column_parse_fraction": 0.0},
        },
    }
    protocol = {
        "protocol": "E002b-goal-channel-propagation-v1",
        "fixed_structural_thresholds": [1, 3, 5, 10],
        "claim_boundary": "toy E002b claim boundary",
    }
    seal = {"artifacts": {}}
    return route, sign_report, topography, protocol, seal


def test_primary_bundle_preserves_exact_roles_signs_and_column_probe_roles() -> None:
    route, sign_report, topography, _, _ = _toy_inputs()
    bundle = build_primary_bundle(route, sign_report, topography, 5.0)

    assert len(bundle.edges) == 7
    assert set(bundle.edges.sign.astype(int)) == {1}
    assert bundle.roles["probe_PFNa_family_C1"] == [2]
    assert bundle.roles["probe_PFNm_family_C1"] == [3]
    assert not any(role.startswith("probe_PFNp_family_C") for role in bundle.roles)
    assert bundle.manifest["qualification_status"] == "candidate"


def test_goal_channel_probe_passes_fixed_causal_cuts_without_turn_semantics() -> None:
    route, sign_report, topography, protocol, seal = _toy_inputs()
    raw_weights = pd.DataFrame(
        {
            "source": [5, 9, 9],
            "target": [9, 7, 8],
            "weight": [10.0, 10.0, 10.0],
        }
    )
    goal_relay = {
        "population_body_ids": {
            "hDeltaC": [5],
            "hDeltaM": [9],
            "PFL3": [7],
            "PFL2": [8],
        }
    }
    hdelta_m_sign = {"prediction": {"modeled_sign": 1}}

    report = probe_goal_channel(
        route,
        sign_report,
        topography,
        protocol,
        seal,
        seed=22002,
        steps=32,
        pulse_steps=8,
        drive_amplitude=1.0,
        artifact_hash_gate=(True, {"fixture": True}),
        raw_weights=raw_weights,
        goal_relay_authority=goal_relay,
        hdelta_m_sign_authority=hdelta_m_sign,
    )

    assert report["passed"] is True
    assert report["passed_gate_count"] == report["gate_count"] == 7
    gates = {gate["name"]: gate for gate in report["gates"]}
    assert gates["deterministic_replay"]["value"] == 0.0
    assert gates["odor_gate_negative_control"]["value"] == 0.0
    assert gates["primary_route_hDeltaC_cut"]["value"] == 0.0
    assert gates["primary_route_hDeltaG_cut"]["value"] == 0.0

    threshold_five = report["threshold_reports"]["5"]
    assert threshold_five["intact_gate_model"]["PFNa_family"]["summary"]["pfl3_peak_abs"] > 0
    assert threshold_five["hDeltaM_relay_comparator"]["status"] == "modeled"
    assert (
        threshold_five["hDeltaM_relay_comparator"]["selection_status"]
        == "frozen_comparator_not_eligible_for_promotion"
    )
    assert threshold_five["PFNp_b_sign_zero_sensitivity"]["run"]["summary"]["pfl3_peak_abs"] == 0
    assert threshold_five["column_impulses"]["probe_PFNa_family_C1"]["directional_role_status"] == "unresolved"
    assert all("turn" not in text.lower() for text in report["explicit_nonclaims"])


def test_goal_channel_probe_rejects_directional_promotion() -> None:
    route, sign_report, topography, protocol, seal = _toy_inputs()
    promoted = copy.deepcopy(topography)
    promoted["directional_role_status"] = "resolved_from_behavior"

    with pytest.raises(ValueError, match="directional roles unresolved"):
        probe_goal_channel(
            route,
            sign_report,
            promoted,
            protocol,
            seal,
            seed=22002,
            steps=32,
            pulse_steps=8,
            drive_amplitude=1.0,
        )


def test_goal_channel_probe_rejects_nonready_sign_report() -> None:
    route, sign_report, topography, protocol, seal = _toy_inputs()
    blocked = copy.deepcopy(sign_report)
    blocked["ready_for_modeled_sign_probe"] = False

    with pytest.raises(ValueError, match="not ready"):
        probe_goal_channel(
            route,
            blocked,
            topography,
            protocol,
            seal,
            seed=22002,
            steps=32,
            pulse_steps=8,
            drive_amplitude=1.0,
        )


def test_repo_protocol_and_runtime_remain_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/e002b_goal_channel_protocol_v1.json").read_text())
    runtime = json.loads((root / "configs/e002b_probe_runtime_v1.json").read_text())

    assert protocol["fixed_structural_thresholds"] == [1, 3, 5, 10]
    assert runtime["frozen_before_first_real_probe"] is True
    assert runtime["seed"] == 22002
    assert runtime["steps"] == 32
    assert runtime["pulse_steps"] == 8
    assert runtime["drive_amplitude"] == 1.0
    assert runtime["structural_thresholds"] == protocol["fixed_structural_thresholds"]
