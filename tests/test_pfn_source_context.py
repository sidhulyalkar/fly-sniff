from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fly_sniff.pfn_source import PFNSourceContract

AUTHORITY_PATH = Path("authority/program-a-pfn-source-contract-v1.json")


def _payload() -> dict:
    return json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))


def test_soma_laterality_authority_cannot_drift_silently() -> None:
    payload = copy.deepcopy(_payload())
    payload["source_preparation"]["angular_convention"]["laterality_authority"] = (
        "fan-shaped-body column"
    )
    with pytest.raises(ValueError, match="cell-body hemisphere"):
        PFNSourceContract.from_dict(payload)


def test_airflow_axis_sign_convention_cannot_flip() -> None:
    payload = copy.deepcopy(_payload())
    payload["source_preparation"]["angular_convention"]["positive_90_deg"] = "right airflow"
    with pytest.raises(ValueError, match=r"\+90 degree convention"):
        PFNSourceContract.from_dict(payload)


def test_right_soma_fold_sign_cannot_flip() -> None:
    payload = copy.deepcopy(_payload())
    payload["source_preparation"]["angular_convention"]["right_soma_ipsilateral_sign"] = 1
    with pytest.raises(ValueError, match="folding signs"):
        PFNSourceContract.from_dict(payload)


def test_source_preparation_identity_is_machine_bound() -> None:
    payload = copy.deepcopy(_payload())
    payload["source_preparation"]["cell_class"] = "generic PFN"
    with pytest.raises(ValueError, match="cell class"):
        PFNSourceContract.from_dict(payload)


def test_archive_md5_mutation_is_rejected_not_merely_rehashed() -> None:
    payload = copy.deepcopy(_payload())
    payload["data_authority"]["multipart_archive"][0]["published_md5"] = "0" * 32
    with pytest.raises(ValueError, match="inventory/MD5"):
        PFNSourceContract.from_dict(payload)


def test_full_archive_download_cannot_become_default_policy() -> None:
    payload = copy.deepcopy(_payload())
    payload["data_authority"]["whole_archive_download_is_default"] = True
    with pytest.raises(ValueError, match="may not be the default"):
        PFNSourceContract.from_dict(payload)


def test_nested_readiness_labels_cannot_disagree_with_evidence() -> None:
    payload = copy.deepcopy(_payload())
    payload["data_authority"]["readme"]["status"] = "BLOCKED_README_SHA256_UNVERIFIED"
    with pytest.raises(ValueError, match="README status"):
        PFNSourceContract.from_dict(payload)


def test_all_recordings_completed_flag_is_not_documentation_only() -> None:
    payload = copy.deepcopy(_payload())
    payload["source_preparation"]["figure4_protocol"][
        "all_recordings_completed_all_trials"
    ] = False
    with pytest.raises(ValueError, match="all 12 recordings"):
        PFNSourceContract.from_dict(payload)


def test_canonical_identity_contains_scientific_convention() -> None:
    contract = PFNSourceContract.from_dict(_payload())
    canonical = contract.to_dict()
    angular = canonical["source_preparation"]["angular_convention"]
    assert angular["laterality_authority"] == "cell-body hemisphere"
    assert angular["positive_90_deg"] == "left airflow"
    assert angular["negative_90_deg"] == "right airflow"
    assert angular["left_soma_ipsilateral_sign"] == 1
    assert angular["right_soma_ipsilateral_sign"] == -1
    assert canonical["source_contract_sha256"] == contract.sha256
