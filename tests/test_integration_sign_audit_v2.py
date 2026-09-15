from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from fly_sniff.integration_sign_audit_v2 import (
    _require_sha256,
    build_external_integration_sign_audit,
)


def _audit() -> dict:
    return {
        "populations": {
            "SRC": {
                "count": 2,
                "rows": [
                    {"bodyId": 1, "type": "A"},
                    {"bodyId": 2, "type": "PFNp_b"},
                ],
            },
            "MID": {
                "count": 1,
                "rows": [{"bodyId": 3, "type": "C"}],
            },
        }
    }


def _authority() -> dict:
    return {
        "source": {
            "repository": "example/catalog",
            "commit": "abc123",
            "neuprint_dataset": "male-cns:v1.0",
        },
        "types": {
            "A": {
                "predicted_neurotransmitter": "ACh",
                "confidence": 0.9,
                "modeled_sign": 1,
                "source_path": "types/A.html",
            },
            "PFNp_b": {
                "predicted_neurotransmitter": "ACh",
                "confidence": 0.532,
                "modeled_sign": 1,
                "source_path": "types/PFNp_b.html",
            },
            "C": {
                "predicted_neurotransmitter": "ACh",
                "confidence": 0.8,
                "modeled_sign": 1,
                "source_path": "types/C.html",
            },
        },
        "mandatory_sensitivity_cases": [
            {
                "name": "PFNp_b_unresolved_sign",
                "rule": "zero PFNp_b modeled sign",
                "reason": "low confidence",
            }
        ],
    }


def _config() -> dict:
    return {
        "protocol": "test-v2",
        "dataset": "male-cns:v1.0",
        "required_presynaptic_populations": ["SRC"],
        "edge_families": [{"source": "SRC", "target": "MID"}],
        "thresholds": [1, 5],
        "expected_transmitter_source": {
            "repository": "example/catalog",
            "commit": "abc123",
            "neuprint_dataset": "male-cns:v1.0",
        },
        "readiness_rule": "all exact types resolved",
        "claim_boundary": "test only",
    }


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source": 1, "target": 3, "weight": 10.0},
            {"source": 2, "target": 3, "weight": 4.0},
        ]
    )


def test_external_authority_resolves_exact_frozen_types_and_keeps_confidence() -> None:
    report = build_external_integration_sign_audit(
        _weights(),
        _audit(),
        _authority(),
        _config(),
    )

    assert report["ready_for_modeled_sign_probe"]
    assert report["ready_with_sensitivity_requirements"]
    assert report["population_signs"]["SRC"]["minimum_prediction_confidence"] == 0.532
    assert report["prediction_confidence_order"][0]["type"] == "PFNp_b"
    assert report["edge_families"]["SRC->MID"]["threshold_sweep"][0][
        "resolved_signed_edges"
    ] == 2
    assert report["edge_families"]["SRC->MID"]["threshold_sweep"][1][
        "resolved_signed_edges"
    ] == 1
    assert report["mandatory_sensitivity_cases"][0]["affected_body_ids"] == [2]


def test_external_authority_blocks_missing_exact_subtype() -> None:
    authority = _authority()
    del authority["types"]["PFNp_b"]

    report = build_external_integration_sign_audit(
        _weights(),
        _audit(),
        authority,
        _config(),
    )

    assert not report["ready_for_modeled_sign_probe"]
    assert report["population_signs"]["SRC"]["unresolved_types"] == ["PFNp_b"]
    assert (
        report["population_signs"]["SRC"]["type_evidence"]["PFNp_b"]["status"]
        == "missing_from_authority"
    )


def test_external_authority_blocks_zero_modeled_sign() -> None:
    authority = _authority()
    authority["types"]["PFNp_b"]["modeled_sign"] = 0

    report = build_external_integration_sign_audit(
        _weights(),
        _audit(),
        authority,
        _config(),
    )

    assert not report["ready_for_modeled_sign_probe"]
    assert report["population_signs"]["SRC"]["unresolved_types"] == ["PFNp_b"]


def test_external_authority_rejects_source_provenance_drift() -> None:
    authority = _authority()
    authority["source"]["commit"] = "different"

    with pytest.raises(ValueError, match="transmitter authority source mismatch"):
        build_external_integration_sign_audit(
            _weights(),
            _audit(),
            authority,
            _config(),
        )


def test_require_sha256_rejects_file_drift(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    path.write_text("{}\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _require_sha256(path, expected, "authority") == expected

    path.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="authority SHA-256 mismatch"):
        _require_sha256(path, expected, "authority")
