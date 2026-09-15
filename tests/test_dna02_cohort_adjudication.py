from __future__ import annotations

import copy
from pathlib import Path

import pytest

from fly_sniff.dna02_cohort_adjudication import (
    EXPECTED_COHORT,
    authority_ref,
    load_adjudication,
    validate_adjudication,
)
from fly_sniff.dna02_source import BLOCKED_FILE_MAP, load_contract

ADJUDICATION_PATH = Path("authority/program-a-dna02-cohort-adjudication-v1.json")
SOURCE_PATH = Path("authority/program-a-dna02-source-contract-v1.json")


def _adjudication() -> dict:
    return load_adjudication(ADJUDICATION_PATH)


def test_real_adjudication_resolves_cohort_only() -> None:
    payload = _adjudication()
    assert payload["status"] == "RESOLVED_COHORT_ONLY"
    assert tuple(payload["resolved_figure3c_cohort"]) == EXPECTED_COHORT
    assert payload["decision_rule"]["navigation_performance_used"] is False
    assert "does not authorize" in payload["remaining_blocker"].lower()


def test_dataverse_set_is_exactly_the_four_resolved_aliases() -> None:
    payload = _adjudication()
    rows = payload["dataverse_authority"]["bilateral_dna02_raw_folders"]
    assert {row["fly_alias"] for row in rows} == set(EXPECTED_COHORT)
    assert payload["dataverse_authority"]["exact_bilateral_dna02_raw_folder_count"] == 4
    assert all(row["restricted"] is False for row in rows)
    assert all(row["checksum_type"] == "MD5" for row in rows)


def test_code_authority_independently_names_the_same_four_aliases() -> None:
    payload = _adjudication()
    assert tuple(payload["code_authority"]["completed_bilateral_aliases"]) == EXPECTED_COHORT
    assert payload["code_authority"]["commit"] == "7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8"


def test_low_snr_note_remains_scoped_caveat_not_global_exclusion() -> None:
    conflict = _adjudication()["conflicting_evidence"]
    assert "a2_d_14" in conflict["observation"]
    assert "not panel-level" in conflict["scope_decision"].lower()


def test_tampered_dataverse_metadata_hash_fails_closed() -> None:
    payload = copy.deepcopy(_adjudication())
    payload["dataverse_authority"]["raw_metadata_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="metadata SHA-256"):
        validate_adjudication(payload)


def test_tampered_alias_set_fails_closed() -> None:
    payload = copy.deepcopy(_adjudication())
    payload["resolved_figure3c_cohort"][-1] = "a2_d_99"
    with pytest.raises(ValueError, match="four-source identity set"):
        validate_adjudication(payload)


def test_navigation_performance_cannot_enter_adjudication() -> None:
    payload = copy.deepcopy(_adjudication())
    payload["decision_rule"]["navigation_performance_used"] = True
    with pytest.raises(ValueError, match="navigation performance"):
        validate_adjudication(payload)


def test_adjudication_hash_is_bound_into_source_contract() -> None:
    payload = _adjudication()
    source = load_contract(SOURCE_PATH)
    assert source.figure3c_cohort == EXPECTED_COHORT
    assert source.figure3c_cohort_authority == authority_ref(payload)
    assert source.blockers == (BLOCKED_FILE_MAP,)
    assert source.status == "BLOCKED"


def test_md5_discovery_does_not_promote_source_file_map() -> None:
    source = load_contract(SOURCE_PATH)
    assert source.data_file_map == ()
    assert source.blockers == (BLOCKED_FILE_MAP,)
    assert any("MD5" in text for text in source.forbidden_interpretation)
