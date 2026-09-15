from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.steering_scaffold import build_steering_scaffold, sha256_file


def _audit() -> dict:
    populations = {
        "PFL3": {"rows": [{"bodyId": 1, "type": "PFL3"}]},
        "DNa03": {"rows": [{"bodyId": 2, "type": "DNa03"}]},
        "LAL010": {"rows": [{"bodyId": 3, "type": "LAL010"}]},
        "DNa02": {
            "rows": [
                {"bodyId": 4, "type": "DNa02", "instance": "DNa02_L"},
                {"bodyId": 5, "type": "DNa02", "instance": "DNa02_R"},
            ]
        },
    }
    direct = [
        {"source": "PFL3", "target": "DNa02", "edges": [{"source": 1, "target": 4, "weight": 12.0}]},
        {"source": "PFL3", "target": "DNa03", "edges": [{"source": 1, "target": 2, "weight": 9.0}]},
        {"source": "PFL3", "target": "LAL010", "edges": [{"source": 1, "target": 3, "weight": 7.0}]},
        {"source": "DNa03", "target": "DNa02", "edges": [{"source": 2, "target": 4, "weight": 20.0}]},
        {"source": "LAL010", "target": "DNa02", "edges": [{"source": 3, "target": 4, "weight": 18.0}]},
    ]
    return {
        "protocol": "malecns-literature-route-audit-v1",
        "dataset": "male-cns:v1.0",
        "populations": populations,
        "direct_predictions": direct,
    }


def _config() -> dict:
    return {
        "protocol": "malecns-steering-scaffold-candidate-v1",
        "dataset": "male-cns:v1.0",
        "evidence_audit_sha256": "abc",
        "evidence_authority": "authority/example.json",
        "sign_authority": "authority/signs.json",
        "sign_authority_sha256": "sign-sha",
        "populations": {
            "PFL3": [1],
            "DNa03": [2],
            "LAL010": [3],
            "DNa02": [4, 5],
        },
        "candidate_roles": {
            "steer_left": [4],
            "steer_right": [5],
            "turn_drive_left": [1],
            "turn_drive_right": [],
        },
        "excluded_from_this_scaffold": {"odor_sensory_roles": "unresolved"},
    }


def _build(**overrides):
    kwargs = {
        "type_signs": {"PFL3": 1, "DNa03": 1, "LAL010": 1},
        "source_audit_sha256": "abc",
        "sign_authority_path": "authority/signs.json",
        "sign_authority_sha256": "sign-sha",
    }
    kwargs.update(overrides)
    return build_steering_scaffold(_audit(), _config(), **kwargs)


def test_builds_candidate_scaffold_with_explicit_signs_and_provenance() -> None:
    bundle = _build()
    assert bundle.manifest["qualification_status"] == "candidate"
    assert len(bundle.nodes) == 5
    assert len(bundle.edges) == 5
    assert set(bundle.edges.sign) == {1}
    assert bundle.roles["steer_left"] == [4]
    assert "odor_left" not in bundle.roles
    assert bundle.manifest["sign_authority"] == {
        "path": "authority/signs.json",
        "sha256": "sign-sha",
    }
    bundle.validate(require_sign=True, require_qualified=False)


def test_repository_sign_authority_matches_sealed_hash() -> None:
    config_path = Path("configs/steering_scaffold_candidate_v1.json")
    config = json.loads(config_path.read_text())
    authority = Path(config["sign_authority"])
    assert authority.exists()
    assert sha256_file(authority) == config["sign_authority_sha256"]


def test_rejects_audit_hash_drift() -> None:
    with pytest.raises(ValueError, match="route-audit SHA-256"):
        _build(source_audit_sha256="different")


def test_rejects_sign_authority_hash_drift() -> None:
    with pytest.raises(ValueError, match="sign-authority SHA-256"):
        _build(sign_authority_sha256="different")


def test_rejects_sign_authority_path_drift() -> None:
    with pytest.raises(ValueError, match="sign-authority path"):
        _build(sign_authority_path="authority/other.json")


def test_rejects_unresolved_presynaptic_sign() -> None:
    with pytest.raises(ValueError, match="sign unresolved"):
        _build(type_signs={"PFL3": 0, "DNa03": 1, "LAL010": 1})


def test_rejects_population_body_id_drift() -> None:
    audit = _audit()
    audit["populations"]["PFL3"]["rows"][0]["bodyId"] = 999
    with pytest.raises(ValueError, match="population body IDs changed"):
        build_steering_scaffold(
            audit,
            _config(),
            type_signs={"PFL3": 1, "DNa03": 1, "LAL010": 1},
            source_audit_sha256="abc",
            sign_authority_path="authority/signs.json",
            sign_authority_sha256="sign-sha",
        )
