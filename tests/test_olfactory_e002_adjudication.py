from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.olfactory_e002_adjudication import validate_da2_adjudication

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "authority" / "flywire-da2-e002-adjudication-v1.json"


def _payload() -> dict:
    return json.loads(AUTHORITY.read_text())


def test_e002_adjudication_freezes_partial_cohort_without_promotion() -> None:
    report = validate_da2_adjudication(_payload())
    assert report["status"] == "blocked_2_osn_root_ids_and_1_side_unresolved"
    assert report["table_resolved_osn_body_records"] == 39
    assert report["publication_total_osns"] == 41
    assert report["unresolved_osn_body_records"] == 2
    assert report["unresolved_osn_side_records"] == 1
    assert report["da2_lpn_body_records"] == 11
    assert report["known_anchor_present"] is True
    assert report["confirmatory_usable"] is False


def test_e002_adjudication_cannot_guess_missing_osn_count() -> None:
    payload = _payload()
    payload["discrepancy"]["unresolved_body_count"] = 0
    with pytest.raises(ValueError, match="unresolved body count must remain 2"):
        validate_da2_adjudication(payload)


def test_e002_adjudication_cannot_reassign_na_side() -> None:
    payload = _payload()
    row = next(
        row
        for row in payload["exact_annotation_adjudication"]["body_records"]
        if row["source_side"] == "na"
    )
    row["source_side"] = "left"
    with pytest.raises(ValueError, match="source-side counts changed"):
        validate_da2_adjudication(payload)


def test_e002_adjudication_cannot_claim_confirmatory_use() -> None:
    payload = _payload()
    payload["confirmatory_usable"] = True
    with pytest.raises(ValueError, match="cannot be confirmatory-usable"):
        validate_da2_adjudication(payload)


def test_e002_adjudication_retains_known_lpn_anchor() -> None:
    payload = _payload()
    records = payload["projection_neuron_adjudication"]["body_records"]
    payload["projection_neuron_adjudication"]["body_records"] = [
        row for row in records if row["root_id"] != "720575940624106442"
    ]
    payload["projection_neuron_adjudication"]["table_resolved_count"] = 10
    with pytest.raises(ValueError, match="exactly 11 table-resolved"):
        validate_da2_adjudication(payload)
