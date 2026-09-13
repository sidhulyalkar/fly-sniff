from __future__ import annotations

import copy
import hashlib

import pytest

from fly_sniff.experimental_plume import (
    expected_source_template,
    load_experimental_plume_config,
    upstream_contract_sha256,
    validate_source_receipt,
    verify_source_bytes,
)


def _concrete(document, plume):
    receipt = expected_source_template(document, plume)
    receipt["source_sha256"] = "a" * 64
    receipt["spatial_transform"] = "identity validation coordinates"
    receipt["coordinate_mapping"] = "frozen validation-only image coordinates"
    if plume == "complex":
        receipt["native_time_basis"] = {"kind": "archive_timestamps", "verified": True}
    return receipt


def test_templates_are_not_valid_until_provenance_is_concrete():
    document = load_experimental_plume_config()
    for plume in ("smooth", "complex"):
        with pytest.raises(ValueError, match="unresolved"):
            validate_source_receipt(expected_source_template(document, plume), document)


def test_concrete_frozen_source_receipts_validate_structurally():
    document = load_experimental_plume_config()
    for plume in ("smooth", "complex"):
        validate_source_receipt(_concrete(document, plume), document)


def test_wrong_shape_and_smooth_timing_profile_are_rejected():
    document = load_experimental_plume_config()
    smooth = _concrete(document, "smooth")
    smooth["source_shape"] = [1, 2, 3]
    with pytest.raises(ValueError, match="shape"):
        validate_source_receipt(smooth, document)

    smooth = _concrete(document, "smooth")
    smooth["temporal_resampling"]["profile"] = "notebook_legacy"
    with pytest.raises(ValueError, match="corrected"):
        validate_source_receipt(smooth, document)


def test_source_bytes_must_match_receipted_sha256(tmp_path):
    payload = b"experimental plume bytes\n"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    receipt = {"source_sha256": hashlib.sha256(payload).hexdigest()}
    assert verify_source_bytes(source, receipt) == receipt["source_sha256"]
    receipt["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source-file SHA-256 mismatch"):
        verify_source_bytes(source, receipt)


def test_upstream_contract_hash_changes_if_source_contract_changes():
    document = load_experimental_plume_config()
    original = upstream_contract_sha256(document)
    changed = copy.deepcopy(document)
    changed["smooth"]["source_fps"] = 99.0
    assert upstream_contract_sha256(changed) != original
