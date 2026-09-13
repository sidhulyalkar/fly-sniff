from __future__ import annotations

import pandas as pd

from fly_sniff.integration_sign_audit import build_integration_sign_audit


def _integration_audit() -> dict:
    populations = {
        "FB5AB": [1],
        "PFNa_family": [2],
        "PFNm_family": [3],
        "PFNp_family": [4],
        "hDeltaC": [5],
        "hDeltaG": [6],
        "PFL3": [7],
        "PFL2": [8],
    }
    return {
        "populations": {
            name: {"count": len(body_ids), "rows": [{"bodyId": body_id} for body_id in body_ids]}
            for name, body_ids in populations.items()
        }
    }


def _config() -> dict:
    return {
        "protocol": "test-integration-sign-audit",
        "dataset": "male-cns:v1.0",
        "required_presynaptic_populations": [
            "FB5AB",
            "PFNa_family",
            "PFNm_family",
            "PFNp_family",
            "hDeltaC",
            "hDeltaG",
        ],
        "edge_families": [
            {"source": "FB5AB", "target": "hDeltaC"},
            {"source": "PFNa_family", "target": "hDeltaC"},
            {"source": "PFNm_family", "target": "hDeltaC"},
            {"source": "PFNp_family", "target": "hDeltaC"},
            {"source": "hDeltaC", "target": "hDeltaG"},
            {"source": "hDeltaG", "target": "PFL3"},
        ],
        "thresholds": [1, 3, 5, 10],
        "model_sign_policy": {
            "ach": 1,
            "acetylcholine": 1,
            "gaba": -1,
            "glu": 0,
            "glutamate": 0,
        },
    }


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": [1, 2, 3, 4, 5, 6],
            "target": [5, 5, 5, 5, 6, 7],
            "weight": [20.0, 10.0, 8.0, 6.0, 5.0, 12.0],
        }
    )


def test_integration_sign_audit_ready_when_all_required_presynaptic_signs_resolve() -> None:
    annotations = pd.DataFrame(
        {
            "bodyId": list(range(1, 9)),
            "predictedNt": ["ACh", "ACh", "ACh", "ACh", "ACh", "ACh", "ACh", "ACh"],
        }
    )
    report = build_integration_sign_audit(
        annotations,
        _weights(),
        _integration_audit(),
        _config(),
    )
    assert report["ready_for_modeled_sign_probe"]
    assert report["neurotransmitter_column"] == "predictedNt"
    assert report["population_signs"]["hDeltaC"]["modeled_sign_counts"] == {"1": 1}
    assert report["edge_families"]["hDeltaC->hDeltaG"]["threshold_sweep"][2][
        "signed_fraction"
    ] == 1.0


def test_integration_sign_audit_blocks_unresolved_glutamatergic_population() -> None:
    annotations = pd.DataFrame(
        {
            "bodyId": list(range(1, 9)),
            "predictedNt": ["ACh", "ACh", "ACh", "ACh", "Glu", "ACh", "ACh", "ACh"],
        }
    )
    report = build_integration_sign_audit(
        annotations,
        _weights(),
        _integration_audit(),
        _config(),
    )
    assert not report["ready_for_modeled_sign_probe"]
    assert report["population_signs"]["hDeltaC"]["unresolved_count"] == 1
    assert report["population_signs"]["hDeltaC"]["unresolved"][0]["bodyId"] == 5


def test_integration_sign_audit_reports_missing_transmitter_column_without_guessing() -> None:
    annotations = pd.DataFrame({"bodyId": list(range(1, 9)), "type": ["x"] * 8})
    report = build_integration_sign_audit(
        annotations,
        _weights(),
        _integration_audit(),
        _config(),
    )
    assert not report["ready_for_modeled_sign_probe"]
    assert report["blocking_reason"] == "missing_neurotransmitter_annotation_column"
