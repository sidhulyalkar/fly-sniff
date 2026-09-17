from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.olfactory_geosmin import validate_geosmin_evidence

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "authority" / "geosmin-e001-evidence-v0.json"


def _payload() -> dict:
    return json.loads(EVIDENCE.read_text())


def test_e001_published_evidence_is_valid_but_not_qualified_for_numeric_fitting():
    report = validate_geosmin_evidence(_payload())
    assert report["status"] == "development_evidence_not_qualified"
    assert report["claim_count"] == 7
    assert report["numeric_fit_claims"] == 0
    assert report["o003_conflict_reserved"] is True


def test_published_summary_cannot_be_promoted_to_numeric_parameter():
    payload = _payload()
    payload["claims"][0]["numeric_model_parameter_allowed"] = True
    with pytest.raises(ValueError, match="numeric fitted parameters"):
        validate_geosmin_evidence(payload)


def test_pipette_dilution_cannot_lose_receptor_concentration_caveat():
    payload = _payload()
    sensitivity = next(row for row in payload["claims"] if row["claim_id"].startswith("G002"))
    sensitivity["caveat"] = "absolute receptor concentration"
    with pytest.raises(ValueError, match="pipette-versus-receptor"):
        validate_geosmin_evidence(payload)


def test_geosmin_vinegar_conflict_cannot_be_used_to_fit_o001():
    payload = _payload()
    conflict = next(row for row in payload["claims"] if row["claim_id"].startswith("G007"))
    conflict["o001_use"] = "calibration_target"
    with pytest.raises(ValueError, match="may not be used to fit O001"):
        validate_geosmin_evidence(payload)


def test_chemical_identity_independent_da2_control_is_frozen():
    payload = _payload()
    ectopic = next(row for row in payload["claims"] if row["claim_id"].startswith("G006"))
    ectopic["stimulus"] = "geosmin"
    with pytest.raises(ValueError, match="chemical-identity-independent"):
        validate_geosmin_evidence(payload)


def test_e001_cannot_claim_qualification_in_same_version():
    payload = _payload()
    payload["status"] = "qualified"
    with pytest.raises(ValueError, match="must remain unqualified"):
        validate_geosmin_evidence(payload)
