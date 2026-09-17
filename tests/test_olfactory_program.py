from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fly_sniff.olfactory_program import (
    validate_evidence_requirements,
    validate_o001,
    validate_program,
    validate_source_registry,
    validate_study,
)

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "authority" / "olfactory-computation-program-v0.json"
EVIDENCE = ROOT / "authority" / "olfactory-evidence-requirements-v0.json"
SOURCES = ROOT / "authority" / "olfactory-source-registry-v0.json"
O001 = ROOT / "authority" / "geosmin-o001-development-v0.json"


def _program() -> dict:
    return json.loads(PROGRAM.read_text())


def _evidence() -> dict:
    return json.loads(EVIDENCE.read_text())


def _o001() -> dict:
    return json.loads(O001.read_text())


def test_frozen_olfactory_program_is_valid_and_evidence_blocked():
    report = validate_study(PROGRAM, EVIDENCE, source_registry_path=SOURCES, o001_path=O001)
    assert report["status"] == "blocked"
    assert report["unresolved_count"] == 9
    assert "E009_prospective_validation_plan" in report["unresolved_authorities"]


def test_source_registry_and_o001_contract_validate():
    validate_source_registry(json.loads(SOURCES.read_text()))
    validate_o001(_o001())


def test_geosmin_calibration_cannot_be_promoted_to_headline_topology_claim():
    payload = _program()
    payload["aims"][0]["headline_topology_claim_allowed"] = True
    with pytest.raises(ValueError, match="geosmin specialist calibration"):
        validate_program(payload)


def test_o001_role_cannot_be_promoted():
    payload = _o001()
    payload["role"] = "headline_topology_claim"
    with pytest.raises(ValueError, match="cannot be promoted"):
        validate_o001(payload)


def test_o001_cannot_consume_downstream_behavioral_objectives():
    payload = _o001()
    payload["forbidden_calibration_targets"].remove("O003_heldout_behavior_score")
    with pytest.raises(ValueError, match="exclude downstream"):
        validate_o001(payload)


def test_behavioral_reward_is_forbidden_during_program_a_calibration():
    payload = _program()
    payload["program_a_rules"]["behavioral_reward_allowed_for_calibration"] = True
    with pytest.raises(ValueError, match="behavioral reward"):
        validate_program(payload)


def test_topology_specific_fit_is_forbidden_for_latent_wiring_claim():
    payload = _program()
    payload["program_a_rules"]["topology_specific_fit_allowed"] = True
    with pytest.raises(ValueError, match="topology-specific fitting"):
        validate_program(payload)


def test_confirmatory_topology_claim_cannot_shrink_null_cohort():
    payload = _program()
    payload["confirmatory_policy"]["minimum_topology_null_count"] = 8
    with pytest.raises(ValueError, match="at least 31"):
        validate_program(payload)


def test_flagship_preserves_preferred_63_null_target():
    payload = _program()
    payload["confirmatory_policy"]["preferred_topology_null_count"] = 31
    with pytest.raises(ValueError, match="63-null"):
        validate_program(payload)


def test_engineering_transfer_cannot_be_promoted_to_biological_topology_claim():
    payload = _program()
    aim = next(row for row in payload["aims"] if row["id"] == "O005")
    aim["headline_topology_claim_allowed"] = True
    with pytest.raises(ValueError, match="engineering transfer"):
        validate_program(payload)


def test_unresolved_evidence_cannot_claim_confirmatory_readiness():
    payload = _evidence()
    payload["confirmatory_execution_status"] = "ready_for_confirmatory_lock"
    with pytest.raises(ValueError, match="does not match unresolved"):
        validate_evidence_requirements(payload)


def test_qualified_evidence_requires_content_addressed_artifact():
    payload = _evidence()
    payload["authorities"][0]["status"] = "qualified"
    with pytest.raises(ValueError, match="content-addressed artifact"):
        validate_evidence_requirements(payload)


def test_all_evidence_can_unlock_only_when_content_addressed():
    payload = copy.deepcopy(_evidence())
    for index, record in enumerate(payload["authorities"]):
        record["status"] = "qualified"
        record["artifact_sha256"] = f"{index + 1:064x}"[-64:]
    payload["confirmatory_execution_status"] = "ready_for_confirmatory_lock"
    assert validate_evidence_requirements(payload) == []
