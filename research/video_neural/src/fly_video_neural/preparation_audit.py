from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .alignment import TARGET_NEURAL_BOUNDARY_POLICY
from .mc2p_legacy import sha256_file
from .prepare_v1 import EXPECTED_PUBLIC_RELEASE_ANIMALS
from .provenance import preparation_implementation_fingerprint


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_self_hashed(path: Path, key: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required preparation artifact: {path}")
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise TypeError(f"preparation artifact must be a JSON object: {path}")
    claimed = document.get(key)
    unsigned = dict(document)
    unsigned.pop(key, None)
    if not isinstance(claimed, str) or claimed != _sha(unsigned):
        raise ValueError(f"{path.name} {key} self-hash mismatch")
    return document


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} bytes do not match frozen SHA-256: {path}")


def audit_preparation_directory(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve()
    receipt_path = root / "preparation-receipt.json"
    receipt = _load_self_hashed(receipt_path, "receipt_sha256")

    if receipt.get("protocol") != "mc2p-v1-preparation-v1":
        raise ValueError("preflight requires an MC2P v1 preparation receipt")
    if receipt.get("benchmark_id") != "mc2p_future_neural_v1":
        raise ValueError("preparation receipt benchmark mismatch")
    if receipt.get("dataset_id") != "mc2p_v1":
        raise ValueError("preparation receipt dataset mismatch")
    if receipt.get("expected_public_release_animals") != EXPECTED_PUBLIC_RELEASE_ANIMALS:
        raise ValueError("preparation receipt public-release expectation changed")
    if receipt.get("animal_count") != EXPECTED_PUBLIC_RELEASE_ANIMALS:
        raise ValueError("preflight requires the complete eight-animal prepared release")
    if receipt.get("models_fit") is not False or receipt.get("test_data_consumed") is not False:
        raise ValueError("preparation receipt indicates modeling or test consumption")
    if receipt.get("target_neural_boundary_policy") != TARGET_NEURAL_BOUNDARY_POLICY:
        raise ValueError("preparation receipt strict-future neural boundary changed")
    if receipt.get("preparation_implementation_fingerprint") != preparation_implementation_fingerprint():
        raise ValueError("prepared artifacts do not match the current audited ingestion implementation")

    manifest_path = root / "mc2p-manifest.json"
    _require_file_hash(
        manifest_path,
        str(receipt.get("manifest_file_sha256")),
        "MC2P manifest",
    )
    manifest = _load_self_hashed(manifest_path, "manifest_sha256")
    if manifest["manifest_sha256"] != receipt.get("manifest_sha256"):
        raise ValueError("manifest identity does not match preparation receipt")
    if manifest.get("animal_count") != receipt.get("animal_count"):
        raise ValueError("manifest animal count does not match preparation receipt")
    if manifest.get("session_count") != receipt.get("session_count"):
        raise ValueError("manifest session count does not match preparation receipt")

    split_path = root / "session-split-lock.json"
    _require_file_hash(
        split_path,
        str(receipt.get("split_lock_file_sha256")),
        "session split lock",
    )
    split_lock = _load_self_hashed(split_path, "split_lock_sha256")
    if split_lock["split_lock_sha256"] != receipt.get("split_lock_sha256"):
        raise ValueError("split-lock identity does not match preparation receipt")
    if split_lock.get("benchmark_id") != "mc2p_future_neural_v1":
        raise ValueError("split lock benchmark mismatch")
    if split_lock.get("test_sessions_for_hyperparameter_selection") is not False:
        raise ValueError("split lock permits test-session hyperparameter selection")

    source_batches = receipt.get("source_batches")
    if not isinstance(source_batches, list) or not source_batches:
        raise ValueError("preparation receipt has no source-batch identities")
    if split_lock.get("source_batches") != source_batches:
        raise ValueError("split lock and preparation receipt bind different batch sets")
    if len(source_batches) != receipt.get("session_count"):
        raise ValueError("prepared source-batch count does not match session count")

    supplied_batch_paths: set[str] = set()
    for row in source_batches:
        if not isinstance(row, dict):
            raise TypeError("source-batch receipt entry must be an object")
        path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise TypeError("source-batch receipt entry must contain path and sha256 strings")
        if path in supplied_batch_paths:
            raise ValueError("preparation receipt contains duplicate source-batch paths")
        supplied_batch_paths.add(path)
        _require_file_hash(Path(path), digest, "prepared session batch")

    session_rows = receipt.get("sessions")
    if not isinstance(session_rows, list) or len(session_rows) != receipt.get("session_count"):
        raise ValueError("preparation receipt session rows do not match session count")
    session_ids = [str(row.get("session_id")) for row in session_rows if isinstance(row, dict)]
    if len(session_ids) != len(session_rows) or len(session_ids) != len(set(session_ids)):
        raise ValueError("prepared session identities are missing or duplicated")

    verified_windows = 0
    verified_batches = 0
    for row in session_rows:
        if not isinstance(row, dict):
            raise TypeError("preparation session receipt entry must be an object")
        session_id = row.get("session_id")
        animal_id = row.get("animal_id")
        if not isinstance(session_id, str) or not isinstance(animal_id, str):
            raise TypeError("preparation session row is missing animal/session identity")
        session_dir = root / "sessions" / session_id
        if not session_dir.is_dir():
            raise ValueError(f"missing prepared session directory: {session_dir}")

        alignment_receipt = _load_self_hashed(
            session_dir / "alignment-conversion.json",
            "receipt_sha256",
        )
        if alignment_receipt["receipt_sha256"] != row.get("alignment_conversion_sha256"):
            raise ValueError(f"alignment conversion identity mismatch for {session_id}")
        alignment_path = session_dir / "alignment.npy"
        _require_file_hash(
            alignment_path,
            str(alignment_receipt.get("output_sha256")),
            "prepared alignment array",
        )

        pose_receipt = _load_self_hashed(
            session_dir / "pose-conversion.json",
            "receipt_sha256",
        )
        if pose_receipt["receipt_sha256"] != row.get("pose_conversion_sha256"):
            raise ValueError(f"pose conversion identity mismatch for {session_id}")
        pose_path = session_dir / "pose3d.npy"
        _require_file_hash(
            pose_path,
            str(pose_receipt.get("output_sha256")),
            "prepared pose array",
        )

        windows_path = session_dir / "windows.json"
        _require_file_hash(windows_path, str(row.get("windows_sha256")), "prediction windows")
        windows = json.loads(windows_path.read_text())
        if windows.get("target_neural_boundary_policy") != TARGET_NEURAL_BOUNDARY_POLICY:
            raise ValueError(f"strict-future boundary mismatch in {session_id} windows")
        if int(windows.get("window_count", 0)) < 2:
            raise ValueError(f"prepared session {session_id} has fewer than two prediction windows")
        verified_windows += int(windows["window_count"])

        batch_receipt = _load_self_hashed(
            session_dir / "pose-neural-batch.json",
            "receipt_sha256",
        )
        if batch_receipt["receipt_sha256"] != row.get("batch_receipt_sha256"):
            raise ValueError(f"batch receipt identity mismatch for {session_id}")
        if batch_receipt.get("benchmark_id") != "mc2p_future_neural_v1":
            raise ValueError(f"batch benchmark mismatch for {session_id}")
        if batch_receipt.get("animal_id") != animal_id or batch_receipt.get("session_id") != session_id:
            raise ValueError(f"batch identity mismatch for {session_id}")
        if batch_receipt.get("windows_sha256") != row.get("windows_sha256"):
            raise ValueError(f"batch/windows provenance mismatch for {session_id}")
        if batch_receipt.get("sample_count") != windows.get("window_count"):
            raise ValueError(f"batch/window sample count mismatch for {session_id}")
        batch_path = session_dir / "pose-neural-batch.npz"
        batch_sha = sha256_file(batch_path) if batch_path.is_file() else None
        if batch_sha != row.get("batch_sha256") or batch_sha != batch_receipt.get("output_sha256"):
            raise ValueError(f"prepared batch bytes mismatch for {session_id}")
        if str(batch_path.resolve()) not in supplied_batch_paths:
            raise ValueError(f"prepared batch {session_id} is absent from frozen source-batch identities")
        if batch_receipt.get("dff_sha256") != row.get("dff_source_sha256"):
            raise ValueError(f"measured dF/F provenance mismatch for {session_id}")
        verified_batches += 1

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "mc2p-v1-preparation-preflight-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "status": "pass",
        "preparation_receipt_sha256": receipt["receipt_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "split_lock_sha256": split_lock["split_lock_sha256"],
        "animal_count": receipt["animal_count"],
        "session_count": receipt["session_count"],
        "verified_session_batches": verified_batches,
        "verified_prediction_windows": verified_windows,
        "batch_arrays_deserialized": False,
        "model_metrics_inspected": False,
        "models_fit": False,
        "test_data_consumed": False,
        "claim_boundary": (
            "This preflight verifies preparation provenance, JSON self-hashes, split identity, and file "
            "SHA-256 values only. It does not deserialize pose-neural NPZ batches, inspect model metrics, "
            "fit a decoder, or consume a held-out test result."
        ),
    }
    report["report_sha256"] = _sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a prepared MC2P v1 run before any development model fitting"
    )
    parser.add_argument("prepared_directory")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_preparation_directory(args.prepared_directory)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
