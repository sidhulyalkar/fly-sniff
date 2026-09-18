from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fly_sniff.olfactory_structure import validate_da2_authority

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "authority" / "flywire-da2-e002-v0.json"


def _payload() -> dict:
    return json.loads(AUTHORITY.read_text())


def test_e002_is_valid_but_blocked_until_exact_cohorts_are_frozen():
    report = validate_da2_authority(_payload())
    assert report["status"] == "blocked_incomplete_body_id_adjudication"
    assert report["osn_body_records"] == 39
    assert report["da2_lpn_body_records"] == 11
    assert report["unresolved_osn_side_records"] == 1
    assert report["unresolved_discrepancies"] == 3
    assert report["known_anchor_present"] is True
    assert report["confirmatory_usable"] is False


def test_e002_cannot_switch_to_live_or_different_annotation_authority():
    payload = _payload()
    payload["dataset"]["annotation_release"] = "latest"
    with pytest.raises(ValueError, match="publication-matched"):
        validate_da2_authority(payload)


def test_or56a_population_count_and_side_constraints_are_frozen():
    payload = _payload()
    osn = next(row for row in payload["population_constraints"] if row["population_id"].startswith("or56a"))
    osn["expected_total"] = 40
    with pytest.raises(ValueError, match="total-count"):
        validate_da2_authority(payload)


def test_known_da2_anchor_cannot_be_dropped_or_retyped():
    payload = _payload()
    payload["known_anchors"][0]["cell_type"] = "other"
    with pytest.raises(ValueError, match="anchor identity changed"):
        validate_da2_authority(payload)


def test_candidate_query_is_not_a_runtime_cohort_selector():
    payload = _payload()
    pn = next(row for row in payload["population_constraints"] if row["population_id"].startswith("DA2"))
    pn["candidate_discovery_rule"] = "contains DA2"
    with pytest.raises(ValueError, match="candidate discovery rule changed"):
        validate_da2_authority(payload)


def test_qualified_state_requires_complete_body_records():
    payload = _payload()
    payload["status"] = "qualified_identity_cohort"
    with pytest.raises(ValueError, match="status does not match"):
        validate_da2_authority(payload)


def test_duplicate_body_id_is_rejected():
    payload = _payload()
    pn = next(row for row in payload["population_constraints"] if row["population_id"].startswith("DA2"))
    pn["body_records"].append(copy.deepcopy(pn["body_records"][0]))
    with pytest.raises(ValueError, match="duplicate root_id"):
        validate_da2_authority(payload)


def test_e002_retains_all_exact_frozen_annotation_records():
    payload = _payload()
    osn = next(
        row
        for row in payload["population_constraints"]
        if row["population_id"] == "or56a_osn_to_right_DA2"
    )
    pn = next(
        row
        for row in payload["population_constraints"]
        if row["population_id"] == "DA2_lPN_publication_matched"
    )
    assert osn["frozen_annotation_count"] == 39
    assert len(osn["body_records"]) == 39
    assert sum(row["side"] == "right" for row in osn["body_records"]) == 22
    assert sum(row["side"] == "left" for row in osn["body_records"]) == 16
    assert sum(row["side"] == "unresolved" for row in osn["body_records"]) == 1
    assert pn["frozen_annotation_count"] == 11
    assert len(pn["body_records"]) == 11
    assert all(row["side"] in {"left", "right"} for row in pn["body_records"])


def test_unresolved_osn_side_cannot_be_silently_guessed():
    payload = _payload()
    osn = next(
        row
        for row in payload["population_constraints"]
        if row["population_id"] == "or56a_osn_to_right_DA2"
    )
    unresolved = next(row for row in osn["body_records"] if row["side"] == "unresolved")
    unresolved["side"] = "left"
    payload["unresolved_discrepancies"] = [
        row
        for row in payload["unresolved_discrepancies"]
        if row["id"] != "orn_da2_side_unresolved_fw044213"
    ]
    with pytest.raises(ValueError, match="unresolved cohort discrepancies"):
        validate_da2_authority(payload)


def test_osn_frozen_annotation_rows_cannot_be_dropped():
    payload = _payload()
    osn = next(
        row
        for row in payload["population_constraints"]
        if row["population_id"] == "or56a_osn_to_right_DA2"
    )
    osn["body_records"].pop()
    with pytest.raises(ValueError, match="all 39 exact"):
        validate_da2_authority(payload)


def test_da2_lpn_exact_cohort_is_now_complete():
    payload = _payload()
    pn = next(
        row
        for row in payload["population_constraints"]
        if row["population_id"] == "DA2_lPN_publication_matched"
    )
    pn["body_records"].pop()
    with pytest.raises(ValueError, match="all 11 exact"):
        validate_da2_authority(payload)
