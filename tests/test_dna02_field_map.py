from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.dna02_field_map import (
    DEFAULT_AUTHORITY,
    _EXPECTED_FIELDS,
    confirm_field_map,
    load_authority,
)
from fly_sniff.dna02_source import load_contract
from fly_sniff.dna02_source_inspect import DEFAULT_CONTRACT, DEFAULT_EVIDENCE, _load_byte_evidence
from fly_sniff.freeze import canonical_sha256


def _inspection_payload() -> dict:
    contract = load_contract(DEFAULT_CONTRACT)
    evidence = _load_byte_evidence(DEFAULT_EVIDENCE)
    files = []
    for ref in contract.data_file_map:
        variables = []
        for field in _EXPECTED_FIELDS:
            shape = [1, 1] if field in {"ball_SR", "ephys_SR"} else [100, 1]
            variables.append({"name": field, "shape": shape, "matlab_class": "double"})
        files.append(
            {
                "fly_alias": ref.fly_alias,
                "file_id": ref.file_id,
                "filename": ref.filename,
                "byte_count": 1,
                "md5": "0" * 32,
                "sha256": ref.sha256,
                "schema_inventory": {
                    "mat_format": "matlab_v5",
                    "inventory_method": "scipy.io.whosmat",
                    "variables": variables,
                },
            }
        )
    payload = {
        "schema": "fly-sniff-dna02-source-inspection-v1",
        "status": "SCHEMA_INVENTORIED_PENDING_EXTRACTION_REVIEW",
        "source_contract_sha256": contract.sha256,
        "source_byte_evidence_sha256": evidence["evidence_sha256"],
        "navigation_performance_used": False,
        "physiology_statistic_computed": False,
        "raw_values_exported": False,
        "files": files,
        "next_allowed_action": "synthetic test",
        "forbidden_interpretation": [],
    }
    payload["inspection_sha256"] = canonical_sha256(payload)
    return payload


def _write_inspection(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "inspection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_real_field_map_authority_is_frozen_and_navigation_independent() -> None:
    authority = load_authority(DEFAULT_AUTHORITY)
    assert authority["status"] == "FROZEN_MAPPING_PENDING_SCHEMA_CONFIRMATION"
    assert authority["navigation_performance_used"] is False
    assert authority["frozen_channel_mapping"]["ephys_A"]["soma_side"] == "L"
    assert authority["frozen_channel_mapping"]["ephys_B"]["soma_side"] == "R"
    assert list(authority["required_raw_fields"]) == list(_EXPECTED_FIELDS)


def test_complete_authenticated_schema_confirms_field_map_without_values(tmp_path: Path) -> None:
    path = _write_inspection(tmp_path, _inspection_payload())
    result = confirm_field_map(path)
    assert result["status"] == "FIELD_MAP_CONFIRMED_PENDING_NUMERIC_EXTRACTION_REVIEW"
    assert result["raw_values_read"] is False
    assert result["physiology_statistic_computed"] is False
    assert result["navigation_performance_used"] is False
    assert len(result["files"]) == 4
    assert all(
        tuple(item["required_field_metadata"]) == _EXPECTED_FIELDS for item in result["files"]
    )


def test_missing_right_channel_blocks_before_numeric_extraction(tmp_path: Path) -> None:
    payload = _inspection_payload()
    variables = payload["files"][0]["schema_inventory"]["variables"]
    payload["files"][0]["schema_inventory"]["variables"] = [
        item for item in variables if item["name"] != "ephys_B"
    ]
    payload["inspection_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "inspection_sha256"}
    )
    path = _write_inspection(tmp_path, payload)
    with pytest.raises(ValueError, match="missing frozen raw fields"):
        confirm_field_map(path)


def test_laterality_cannot_be_rehashed_to_swap_channels(tmp_path: Path) -> None:
    authority = json.loads(DEFAULT_AUTHORITY.read_text(encoding="utf-8"))
    authority["frozen_channel_mapping"]["ephys_A"]["soma_side"] = "R"
    authority["frozen_channel_mapping"]["ephys_B"]["soma_side"] = "L"
    authority["authority_sha256"] = canonical_sha256(
        {key: value for key, value in authority.items() if key != "authority_sha256"}
    )
    path = tmp_path / "swapped-authority.json"
    path.write_text(json.dumps(authority), encoding="utf-8")
    with pytest.raises(ValueError, match="ephys_A must remain mapped to left"):
        load_authority(path)


def test_tampered_inspection_receipt_is_rejected(tmp_path: Path) -> None:
    payload = _inspection_payload()
    payload["files"][0]["filename"] = "tampered.mat"
    path = _write_inspection(tmp_path, payload)
    with pytest.raises(ValueError, match="canonical hash does not match"):
        confirm_field_map(path)
