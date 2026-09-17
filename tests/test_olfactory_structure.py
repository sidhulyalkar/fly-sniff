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
    assert report["osn_body_records"] == 0
    assert report["da2_lpn_body_records"] == 1
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
