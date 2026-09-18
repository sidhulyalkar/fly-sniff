from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.olfactory_geosmin import validate_geosmin_evidence

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "authority" / "geosmin-e001-qualified-v1.json"


def _payload() -> dict:
    return json.loads(AUTHORITY.read_text())


def test_e001_v1_is_qualified_for_qualitative_use_only() -> None:
    report = validate_geosmin_evidence(_payload())
    assert report["status"] == "qualified_qualitative_evidence_only"
    assert report["claim_count"] == 7
    assert report["qualification_blockers"] == 0
    assert report["qualitative_usable"] is True
    assert report["numeric_parameterization_usable"] is False
    assert report["o003_conflict_reserved"] is True


def test_e001_v1_exact_door_crosscheck_is_frozen() -> None:
    payload = _payload()
    payload["door_crosscheck"]["observed_raw_response"] = 150.0
    with pytest.raises(ValueError, match="frozen geosmin response changed"):
        validate_geosmin_evidence(payload)


def test_e001_v1_cannot_authorize_numeric_parameterization() -> None:
    payload = _payload()
    payload["numeric_adjudication"]["status"] = "qualified_numeric_fit"
    with pytest.raises(ValueError, match="numeric adjudication changed"):
        validate_geosmin_evidence(payload)


def test_e001_v1_cannot_promote_summary_statistics_to_raw_trials() -> None:
    payload = _payload()
    payload["raw_vs_summary_policy"]["published_summary_statistics_are_raw_trials"] = True
    with pytest.raises(ValueError, match="cannot promote published summaries"):
        validate_geosmin_evidence(payload)
