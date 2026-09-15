from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_video_neural.final_protocol import (
    FINAL_LOCK_NAME,
    _write_consumption_lock,
    validate_final_authorization,
)
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
        "null_name": "circular_half_session_feature_shift",
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


def test_final_authorization_rejects_blocked_or_tampered_unlock(tmp_path: Path):
    split, config = _development_bundle(tmp_path, unlocked=False)
    with pytest.raises(ValueError, match="does not reconstruct"):
        validate_final_authorization(tmp_path, split, config)


def test_consumption_lock_is_one_way(tmp_path: Path):
    first = _write_consumption_lock(
        tmp_path,
        split_lock_sha256="split",
        development_receipt_sha256="dev",
        validation_unlock_sha256="unlock",
    )
    assert (tmp_path / FINAL_LOCK_NAME).is_file()
    assert first["reopen_after_failure_allowed"] is False
    with pytest.raises(ValueError, match="already consumed"):
        _write_consumption_lock(
            tmp_path,
            split_lock_sha256="split",
            development_receipt_sha256="dev",
            validation_unlock_sha256="unlock",
        )
