from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_video_neural.alignment_null import NULL_FRACTIONS, NULL_NAME, NULL_SELECTION_RULE
from fly_video_neural.final_protocol import (
    FINAL_LOCK_NAME,
    _verify_preload_contract,
    _write_consumption_lock,
    validate_final_authorization,
)
from fly_video_neural.mc2p_legacy import sha256_file
from fly_video_neural.provenance import implementation_fingerprint, runtime_fingerprint
from fly_video_neural.validation_gate import build_validation_unlock, load_acceptance_config


def _sha(payload: dict) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _development_bundle(root: Path, *, unlocked: bool = True) -> tuple[dict, dict]:
    config = load_acceptance_config(
        Path(__file__).resolve().parents[1] / "configs" / "validation_acceptance_v1.json"
    )
    animals = [f"fly{index}" for index in range(4)]
    qc = {
        "schema_version": 1,
        "status": "pass",
        "test_target_arrays_deserialized": False,
        "test_target_values_summarized": False,
        "split_lock_sha256": "split",
    }
    qc["report_sha256"] = _sha(qc)
    aligned_rows = []
    null_rows = []
    for index, animal in enumerate(animals):
        aligned_value = 0.5 - index * 0.02
        null_value = 0.1 + index * 0.01
        aligned_rows.append(
            {
                "animal_id": animal,
                "selected_alpha": 1.0,
                "alpha_candidates": [
                    {"alpha": 1.0, "validation": {"median_pearson_r": aligned_value}}
                ],
            }
        )
        null_rows.append(
            {
                "animal_id": animal,
                "selected_null_fraction": NULL_FRACTIONS[index % len(NULL_FRACTIONS)],
                "selected_alpha": 1.0,
                "selected_validation_metrics": {"median_pearson_r": null_value},
                "alpha_candidates": [],
            }
        )
    aligned = {
        "benchmark_id": "mc2p_future_neural_v1",
        "test_status": "locked_not_consumed",
        "split_lock_sha256": "split",
        "animals": aligned_rows,
    }
    null = {
        "benchmark_id": "mc2p_future_neural_v1",
        "test_status": "locked_not_consumed",
        "null_name": NULL_NAME,
        "null_fractions": list(NULL_FRACTIONS),
        "null_selection_rule": NULL_SELECTION_RULE,
        "split_lock_sha256": "split",
        "animals": null_rows,
    }
    unlock = build_validation_unlock(qc, aligned, null, config)
    if not unlocked:
        unlock["status"] = "blocked"
        unlock["test_consumption_allowed"] = False
        unlock["failures"] = ["synthetic block"]
        unlock.pop("report_sha256", None)
        unlock["report_sha256"] = _sha(unlock)
    receipt = {
        "schema_version": 1,
        "protocol": "mc2p-v1-development-only-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "split_lock_sha256": "split",
        "source_batches": [],
        "development_deserialized_batches": [{"path": "dev", "sha256": "dev"}],
        "held_out_test_batches_authenticated_not_deserialized": [{"path": "test", "sha256": "test"}],
        "test_target_arrays_deserialized": False,
        "validation_unlock_report_sha256": unlock["report_sha256"],
        "test_consumption_capability": False,
        "test_metrics_present": False,
    }
    receipt["receipt_sha256"] = _sha(receipt)
    for filename, document in (
        ("development-receipt.json", receipt),
        ("development-qc.json", qc),
        ("aligned-ridge-development.json", aligned),
        ("temporal-null-development.json", null),
        ("validation-unlock.json", unlock),
    ):
        (root / filename).write_text(json.dumps(document))
    return {"split_lock_sha256": "split"}, config


def test_final_authorization_reconstructs_frozen_unlock(tmp_path: Path):
    split, config = _development_bundle(tmp_path)
    bundle = validate_final_authorization(tmp_path, split, config)
    assert bundle["unlock"]["test_consumption_allowed"] is True
    assert bundle["receipt"]["test_target_arrays_deserialized"] is False


def test_final_authorization_rejects_blocked_or_tampered_unlock(tmp_path: Path):
    split, config = _development_bundle(tmp_path, unlocked=False)
    with pytest.raises(ValueError, match="does not reconstruct"):
        validate_final_authorization(tmp_path, split, config)


def test_final_authorization_rejects_development_that_deserialized_test_arrays(tmp_path: Path):
    split, config = _development_bundle(tmp_path)
    path = tmp_path / "development-receipt.json"
    receipt = json.loads(path.read_text())
    receipt["test_target_arrays_deserialized"] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha(receipt)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="no-test-deserialization"):
        validate_final_authorization(tmp_path, split, config)


def test_consumption_lock_is_one_way_and_precedes_batch_deserialization(tmp_path: Path):
    first = _write_consumption_lock(
        tmp_path,
        split_lock_sha256="split",
        development_receipt_sha256="dev",
        validation_unlock_sha256="unlock",
    )
    assert (tmp_path / FINAL_LOCK_NAME).is_file()
    assert first["reopen_after_failure_allowed"] is False
    assert first["batch_deserialization_allowed_after_this_marker_only"] is True
    with pytest.raises(ValueError, match="already consumed"):
        _write_consumption_lock(
            tmp_path,
            split_lock_sha256="split",
            development_receipt_sha256="dev",
            validation_unlock_sha256="unlock",
        )


def test_preload_contract_verifies_bytes_code_and_runtime_without_loading_npz(tmp_path: Path):
    split = tmp_path / "split.json"
    acceptance = tmp_path / "acceptance.json"
    batch = tmp_path / "session.npz"
    split.write_text("split bytes")
    acceptance.write_text("acceptance bytes")
    batch.write_bytes(b"not an npz and must not be deserialized by preload verification")
    source_batches = [{"path": str(batch.resolve()), "sha256": sha256_file(batch)}]
    bundle = {
        "receipt": {
            "source_batches": source_batches,
            "split_lock_file_sha256": sha256_file(split),
            "acceptance_config_file_sha256": sha256_file(acceptance),
            "implementation_fingerprint": implementation_fingerprint(),
            "runtime_fingerprint": runtime_fingerprint(),
        }
    }
    assert _verify_preload_contract(
        bundle,
        split_lock_path=split,
        batch_paths=[batch],
        acceptance_config_path=acceptance,
    ) == source_batches
    batch.write_bytes(b"changed")
    with pytest.raises(ValueError, match="batch bytes"):
        _verify_preload_contract(
            bundle,
            split_lock_path=split,
            batch_paths=[batch],
            acceptance_config_path=acceptance,
        )
