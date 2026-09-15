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
from .session_benchmark import SPLIT_SEED


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


def _require_file_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str):
        raise TypeError(f"{label} receipt is missing a SHA-256 string")
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} bytes do not match frozen SHA-256: {path}")


def _manifest_identities(manifest: dict[str, Any]) -> dict[str, str]:
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != manifest.get("session_count"):
        raise ValueError("MC2P manifest session rows do not match session count")
    identities: dict[str, str] = {}
    for row in sessions:
        if not isinstance(row, dict):
            raise TypeError("MC2P manifest session entry must be an object")
        session_id = row.get("session_id")
        animal_id = row.get("animal_id")
        if not isinstance(session_id, str) or not isinstance(animal_id, str):
            raise TypeError("MC2P manifest session entry is missing animal/session identity")
        if session_id in identities:
            raise ValueError(f"MC2P manifest repeats session {session_id!r}")
        identities[session_id] = animal_id
    animals = sorted(set(identities.values()))
    if animals != manifest.get("animals"):
        raise ValueError("MC2P manifest animal list does not match session ownership")
    if len(animals) != manifest.get("animal_count"):
        raise ValueError("MC2P manifest animal count does not match session ownership")
    return identities


def _validate_conversion_receipt(
    receipt: dict[str, Any],
    *,
    kind: str,
    output_path: Path,
    session_id: str,
) -> None:
    if receipt.get("dataset_id") != "mc2p_v1":
        raise ValueError(f"{kind} conversion dataset mismatch for {session_id}")
    if receipt.get("kind") != kind:
        raise ValueError(f"{kind} conversion kind mismatch for {session_id}")
    if receipt.get("protocol") not in {"mc2p-safe-array-copy-v1", "mc2p-legacy-conversion-v1"}:
        raise ValueError(f"unsupported {kind} conversion protocol for {session_id}")
    if receipt.get("output_path") != str(output_path.resolve()):
        raise ValueError(f"{kind} conversion output path mismatch for {session_id}")
    _require_file_hash(output_path, receipt.get("output_sha256"), f"prepared {kind} array")


def _validate_split_structure(
    split_lock: dict[str, Any],
    *,
    prepared_sessions: dict[str, str],
    sample_rows: list[dict[str, str]],
) -> None:
    if split_lock.get("benchmark_id") != "mc2p_future_neural_v1":
        raise ValueError("split lock benchmark mismatch")
    if split_lock.get("split_unit") != "session_id_within_animal":
        raise ValueError("split lock unit changed")
    if split_lock.get("split_seed") != SPLIT_SEED:
        raise ValueError("split lock seed changed")
    if split_lock.get("validation_sessions_per_animal") != 1:
        raise ValueError("split lock validation-session count changed")
    if split_lock.get("test_sessions_per_animal") != 1:
        raise ValueError("split lock test-session count changed")
    if split_lock.get("test_sessions_for_hyperparameter_selection") is not False:
        raise ValueError("split lock permits test-session hyperparameter selection")

    animal_sessions = split_lock.get("animal_sessions")
    if not isinstance(animal_sessions, dict):
        raise TypeError("split lock animal_sessions must be an object")
    prepared_animals = set(prepared_sessions.values())
    if set(animal_sessions) != prepared_animals:
        raise ValueError("split-lock animal set does not match prepared animals")

    session_to_split: dict[str, str] = {}
    for animal_id, assignments in animal_sessions.items():
        if not isinstance(assignments, dict) or set(assignments) != {"train", "validation", "test"}:
            raise ValueError(f"split assignments are malformed for animal {animal_id}")
        partitions: dict[str, list[str]] = {}
        for split_name in ("train", "validation", "test"):
            values = assignments.get(split_name)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise TypeError(f"split {split_name} sessions must be a string list for {animal_id}")
            if len(values) != len(set(values)):
                raise ValueError(f"split {split_name} repeats a session for {animal_id}")
            partitions[split_name] = values
        if len(partitions["train"]) < 1:
            raise ValueError(f"animal {animal_id} has no frozen training session")
        if len(partitions["validation"]) != 1 or len(partitions["test"]) != 1:
            raise ValueError(f"animal {animal_id} does not have exactly one validation and test session")
        combined = partitions["train"] + partitions["validation"] + partitions["test"]
        if len(combined) != len(set(combined)):
            raise ValueError(f"animal {animal_id} has a session assigned to multiple splits")
        for split_name, session_ids in partitions.items():
            for session_id in session_ids:
                if prepared_sessions.get(session_id) != animal_id:
                    raise ValueError(
                        f"split session {session_id!r} is absent or has changed animal ownership"
                    )
                if session_id in session_to_split:
                    raise ValueError(f"session {session_id!r} appears more than once in split lock")
                session_to_split[session_id] = split_name

    if set(session_to_split) != set(prepared_sessions):
        missing = sorted(set(prepared_sessions) - set(session_to_split))
        extra = sorted(set(session_to_split) - set(prepared_sessions))
        raise ValueError(
            "split lock does not partition the exact prepared session set; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    expected_sample_to_split = {
        row["sample_id"]: session_to_split[row["session_id"]] for row in sample_rows
    }
    expected_sample_to_animal = {row["sample_id"]: row["animal_id"] for row in sample_rows}
    if split_lock.get("sample_to_split") != dict(sorted(expected_sample_to_split.items())):
        raise ValueError("split-lock sample assignments do not match prepared window manifests")
    if split_lock.get("sample_to_animal") != dict(sorted(expected_sample_to_animal.items())):
        raise ValueError("split-lock sample animal identities do not match prepared window manifests")
    if split_lock.get("source_samples_sha256") != _sha(sorted(sample_rows, key=lambda row: row["sample_id"])):
        raise ValueError("split-lock source-sample identity does not match prepared window manifests")


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
    _require_file_hash(manifest_path, receipt.get("manifest_file_sha256"), "MC2P manifest")
    manifest = _load_self_hashed(manifest_path, "manifest_sha256")
    if manifest["manifest_sha256"] != receipt.get("manifest_sha256"):
        raise ValueError("manifest identity does not match preparation receipt")
    if manifest.get("animal_count") != receipt.get("animal_count"):
        raise ValueError("manifest animal count does not match preparation receipt")
    if manifest.get("session_count") != receipt.get("session_count"):
        raise ValueError("manifest session count does not match preparation receipt")
    prepared_sessions = _manifest_identities(manifest)

    split_path = root / "session-split-lock.json"
    _require_file_hash(split_path, receipt.get("split_lock_file_sha256"), "session split lock")
    split_lock = _load_self_hashed(split_path, "split_lock_sha256")
    if split_lock["split_lock_sha256"] != receipt.get("split_lock_sha256"):
        raise ValueError("split-lock identity does not match preparation receipt")

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
    receipt_sessions: dict[str, str] = {}
    for row in session_rows:
        if not isinstance(row, dict):
            raise TypeError("preparation session receipt entry must be an object")
        session_id = row.get("session_id")
        animal_id = row.get("animal_id")
        if not isinstance(session_id, str) or not isinstance(animal_id, str):
            raise TypeError("preparation session row is missing animal/session identity")
        if session_id in receipt_sessions:
            raise ValueError(f"preparation receipt repeats session {session_id!r}")
        receipt_sessions[session_id] = animal_id
    if receipt_sessions != prepared_sessions:
        raise ValueError("preparation receipt and MC2P manifest disagree on session identities")

    verified_windows = 0
    verified_batches = 0
    all_sample_rows: list[dict[str, str]] = []
    seen_sample_ids: set[str] = set()
    for row in session_rows:
        session_id = row["session_id"]
        animal_id = row["animal_id"]
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
        _validate_conversion_receipt(
            alignment_receipt,
            kind="alignment",
            output_path=alignment_path,
            session_id=session_id,
        )

        pose_receipt = _load_self_hashed(
            session_dir / "pose-conversion.json",
            "receipt_sha256",
        )
        if pose_receipt["receipt_sha256"] != row.get("pose_conversion_sha256"):
            raise ValueError(f"pose conversion identity mismatch for {session_id}")
        pose_path = session_dir / "pose3d.npy"
        _validate_conversion_receipt(
            pose_receipt,
            kind="pose3d",
            output_path=pose_path,
            session_id=session_id,
        )

        windows_path = session_dir / "windows.json"
        _require_file_hash(windows_path, row.get("windows_sha256"), "prediction windows")
        windows = json.loads(windows_path.read_text())
        if not isinstance(windows, dict) or windows.get("dataset_id") != "mc2p_v1":
            raise ValueError(f"window manifest dataset mismatch for {session_id}")
        if windows.get("target_neural_boundary_policy") != TARGET_NEURAL_BOUNDARY_POLICY:
            raise ValueError(f"strict-future boundary mismatch in {session_id} windows")
        window_rows = windows.get("windows")
        if not isinstance(window_rows, list) or len(window_rows) != windows.get("window_count"):
            raise ValueError(f"window manifest rows/count mismatch for {session_id}")
        if len(window_rows) < 2:
            raise ValueError(f"prepared session {session_id} has fewer than two prediction windows")
        for window in window_rows:
            if not isinstance(window, dict) or not isinstance(window.get("sample"), dict):
                raise TypeError(f"window sample metadata is malformed for {session_id}")
            sample = window["sample"]
            sample_id = sample.get("sample_id")
            if not isinstance(sample_id, str):
                raise TypeError(f"window sample id is malformed for {session_id}")
            if sample.get("session_id") != session_id or sample.get("animal_id") != animal_id:
                raise ValueError(f"window sample identity changed for {session_id}")
            if sample_id in seen_sample_ids:
                raise ValueError(f"prepared window sample id is duplicated: {sample_id}")
            seen_sample_ids.add(sample_id)
            all_sample_rows.append(
                {"sample_id": sample_id, "animal_id": animal_id, "session_id": session_id}
            )
        verified_windows += len(window_rows)

        batch_receipt = _load_self_hashed(
            session_dir / "pose-neural-batch.json",
            "receipt_sha256",
        )
        if batch_receipt["receipt_sha256"] != row.get("batch_receipt_sha256"):
            raise ValueError(f"batch receipt identity mismatch for {session_id}")
        if batch_receipt.get("protocol") != "mc2p-pose-neural-session-batch-v1":
            raise ValueError(f"batch protocol mismatch for {session_id}")
        if batch_receipt.get("benchmark_id") != "mc2p_future_neural_v1":
            raise ValueError(f"batch benchmark mismatch for {session_id}")
        if batch_receipt.get("dataset_id") != "mc2p_v1":
            raise ValueError(f"batch dataset mismatch for {session_id}")
        if batch_receipt.get("animal_id") != animal_id or batch_receipt.get("session_id") != session_id:
            raise ValueError(f"batch identity mismatch for {session_id}")
        if batch_receipt.get("windows_sha256") != row.get("windows_sha256"):
            raise ValueError(f"batch/windows provenance mismatch for {session_id}")
        if batch_receipt.get("pose3d_sha256") != pose_receipt.get("output_sha256"):
            raise ValueError(f"batch/pose provenance mismatch for {session_id}")
        if batch_receipt.get("sample_count") != len(window_rows):
            raise ValueError(f"batch/window sample count mismatch for {session_id}")
        feature_shape = batch_receipt.get("feature_shape")
        target_shape = batch_receipt.get("target_shape")
        if (
            not isinstance(feature_shape, list)
            or not isinstance(target_shape, list)
            or not feature_shape
            or not target_shape
            or feature_shape[0] != len(window_rows)
            or target_shape[0] != len(window_rows)
        ):
            raise ValueError(f"batch array shapes do not match sample count for {session_id}")
        pose_contract = batch_receipt.get("pose_feature_contract")
        if not isinstance(pose_contract, dict) or pose_contract.get("uses_input_behavior_frames_only") is not True:
            raise ValueError(f"pose feature contract is not input-only for {session_id}")
        target_contract = batch_receipt.get("target_contract")
        if (
            not isinstance(target_contract, dict)
            or target_contract.get("operation") != "mean_measured_dff_over_unique_future_neural_indices"
            or target_contract.get("evidence_class") != "measured_neural_activity"
        ):
            raise ValueError(f"measured neural target contract changed for {session_id}")
        batch_path = session_dir / "pose-neural-batch.npz"
        batch_sha = sha256_file(batch_path) if batch_path.is_file() else None
        if batch_sha != row.get("batch_sha256") or batch_sha != batch_receipt.get("output_sha256"):
            raise ValueError(f"prepared batch bytes mismatch for {session_id}")
        if str(batch_path.resolve()) not in supplied_batch_paths:
            raise ValueError(f"prepared batch {session_id} is absent from frozen source-batch identities")
        if batch_receipt.get("dff_sha256") != row.get("dff_source_sha256"):
            raise ValueError(f"measured dF/F provenance mismatch for {session_id}")
        verified_batches += 1

    _validate_split_structure(
        split_lock,
        prepared_sessions=prepared_sessions,
        sample_rows=all_sample_rows,
    )

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
            "This preflight verifies preparation provenance, JSON self-hashes, exact session/sample split "
            "identity, and file SHA-256 values only. It does not deserialize pose-neural NPZ batches, "
            "inspect model metrics, fit a decoder, or consume a held-out test result."
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
