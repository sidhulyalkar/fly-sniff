from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fly_video_neural.release_preflight as release_preflight
from fly_video_neural.prepare_v1 import (
    EXPECTED_PUBLIC_RELEASE_ANIMALS,
    EXPECTED_PUBLIC_RELEASE_SESSIONS,
)


def _sha(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_receipt(root: Path, *, animals: int = 8, sessions: int = 133) -> dict:
    payload = {
        "schema_version": 1,
        "protocol": "mc2p-v1-preparation-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "dataset_id": "mc2p_v1",
        "expected_public_release_animals": EXPECTED_PUBLIC_RELEASE_ANIMALS,
        "expected_public_release_sessions": EXPECTED_PUBLIC_RELEASE_SESSIONS,
        "animal_count": animals,
        "session_count": sessions,
    }
    payload["receipt_sha256"] = _sha(payload)
    (root / "preparation-receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _provenance_report(*, sessions: int = 133) -> dict:
    return {
        "report_sha256": "provenance-sha",
        "animal_count": 8,
        "session_count": sessions,
        "verified_session_batches": sessions,
        "verified_prediction_windows": sessions * 2,
        "batch_arrays_deserialized": False,
        "model_metrics_inspected": False,
        "models_fit": False,
        "test_data_consumed": False,
    }


def test_complete_release_preflight_delegates_after_authority_gate(tmp_path: Path, monkeypatch):
    root = tmp_path / "prepared"
    root.mkdir()
    receipt = _write_receipt(root)
    calls: list[Path] = []

    def fake_audit(path):
        calls.append(Path(path))
        return _provenance_report()

    monkeypatch.setattr(release_preflight, "audit_preparation_directory", fake_audit)
    report = release_preflight.audit_complete_public_release(root)

    assert calls == [root.resolve()]
    assert report["status"] == "pass"
    assert report["animal_count"] == 8
    assert report["session_count"] == 133
    assert report["verified_session_batches"] == 133
    assert report["preparation_receipt_sha256"] == receipt["receipt_sha256"]
    assert report["batch_arrays_deserialized"] is False
    assert report["model_metrics_inspected"] is False
    assert report["test_data_consumed"] is False


def test_complete_release_preflight_rejects_rehashed_132_session_receipt_before_audit(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "prepared"
    root.mkdir()
    _write_receipt(root, sessions=132)

    def forbidden_audit(path):
        raise AssertionError("partial release must be rejected before provenance audit")

    monkeypatch.setattr(release_preflight, "audit_preparation_directory", forbidden_audit)
    with pytest.raises(ValueError, match="all 133 trials"):
        release_preflight.audit_complete_public_release(root)


def test_complete_release_preflight_rejects_changed_session_authority(tmp_path: Path, monkeypatch):
    root = tmp_path / "prepared"
    root.mkdir()
    payload = _write_receipt(root)
    payload.pop("receipt_sha256")
    payload["expected_public_release_sessions"] = 132
    payload["receipt_sha256"] = _sha(payload)
    (root / "preparation-receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    monkeypatch.setattr(
        release_preflight,
        "audit_preparation_directory",
        lambda path: (_ for _ in ()).throw(AssertionError("authority drift must fail first")),
    )
    with pytest.raises(ValueError, match="session-count authority changed"):
        release_preflight.audit_complete_public_release(root)


def test_complete_release_preflight_rejects_provenance_that_verifies_only_132_batches(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "prepared"
    root.mkdir()
    _write_receipt(root)
    monkeypatch.setattr(
        release_preflight,
        "audit_preparation_directory",
        lambda path: _provenance_report(sessions=132),
    )
    with pytest.raises(RuntimeError, match="wrong public-release session count"):
        release_preflight.audit_complete_public_release(root)
