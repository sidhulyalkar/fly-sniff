from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fly_video_neural.alignment import TARGET_NEURAL_BOUNDARY_POLICY
from fly_video_neural.mc2p_legacy import sha256_file
from fly_video_neural.preparation_audit import _sha, audit_preparation_directory
from fly_video_neural.provenance import preparation_implementation_fingerprint
from fly_video_neural.session_benchmark import SPLIT_SEED


def _write_self_hashed(path: Path, payload: dict, key: str) -> dict:
    document = dict(payload)
    document[key] = _sha(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def _rewrite_preparation_receipt(root: Path, mutate) -> dict:
    path = root / "preparation-receipt.json"
    receipt = json.loads(path.read_text())
    receipt.pop("receipt_sha256")
    mutate(receipt)
    return _write_self_hashed(path, receipt, "receipt_sha256")


def _rewrite_windows_and_bind(root: Path, session_dir: Path, mutate) -> None:
    windows_path = session_dir / "windows.json"
    windows = json.loads(windows_path.read_text())
    mutate(windows)
    windows_path.write_text(json.dumps(windows, indent=2, sort_keys=True) + "\n")

    batch_receipt_path = session_dir / "pose-neural-batch.json"
    batch_receipt = json.loads(batch_receipt_path.read_text())
    batch_receipt.pop("receipt_sha256")
    batch_receipt["windows_sha256"] = sha256_file(windows_path)
    batch_receipt = _write_self_hashed(batch_receipt_path, batch_receipt, "receipt_sha256")

    def update_outer(receipt: dict) -> None:
        for row in receipt["sessions"]:
            if row["session_id"] == session_dir.name:
                row["windows_sha256"] = sha256_file(windows_path)
                row["batch_receipt_sha256"] = batch_receipt["receipt_sha256"]
                return
        raise AssertionError("fixture session missing from preparation receipt")

    _rewrite_preparation_receipt(root, update_outer)


def _build_prepared_fixture(root: Path) -> dict:
    root.mkdir()
    sessions_root = root / "sessions"
    sessions_root.mkdir()
    animals = [f"animal_{index}" for index in range(1, 9)]
    manifest_sessions: list[dict[str, str]] = []
    preparation_sessions: list[dict] = []
    source_batches: list[dict[str, str]] = []
    source_sample_rows: list[dict[str, str]] = []
    animal_sessions: dict[str, dict[str, list[str]]] = {}
    sample_to_split: dict[str, str] = {}
    sample_to_animal: dict[str, str] = {}
    behavior_frame_count = 400

    for animal in animals:
        session_ids = [f"{animal}_{trial:03d}" for trial in (1, 2, 3)]
        animal_sessions[animal] = {
            "train": [session_ids[0]],
            "validation": [session_ids[1]],
            "test": [session_ids[2]],
        }
        split_by_session = {
            session_ids[0]: "train",
            session_ids[1]: "validation",
            session_ids[2]: "test",
        }
        for session_id in session_ids:
            session_dir = sessions_root / session_id
            session_dir.mkdir()
            manifest_sessions.append({"session_id": session_id, "animal_id": animal})

            alignment_path = session_dir / "alignment.npy"
            alignment_path.write_bytes(f"alignment:{session_id}".encode())
            alignment_receipt = _write_self_hashed(
                session_dir / "alignment-conversion.json",
                {
                    "schema_version": 1,
                    "protocol": "mc2p-safe-array-copy-v1",
                    "dataset_id": "mc2p_v1",
                    "kind": "alignment",
                    "source_path": f"/raw/{session_id}/indices.npy",
                    "source_sha256": _sha({"source": session_id, "kind": "alignment"}),
                    "output_path": str(alignment_path.resolve()),
                    "output_sha256": sha256_file(alignment_path),
                    "output_shape": [behavior_frame_count],
                    "output_dtype": "int64",
                },
                "receipt_sha256",
            )

            pose_path = session_dir / "pose3d.npy"
            pose_path.write_bytes(f"pose:{session_id}".encode())
            pose_receipt = _write_self_hashed(
                session_dir / "pose-conversion.json",
                {
                    "schema_version": 1,
                    "protocol": "mc2p-safe-array-copy-v1",
                    "dataset_id": "mc2p_v1",
                    "kind": "pose3d",
                    "source_path": f"/raw/{session_id}/pose3d.npy",
                    "source_sha256": _sha({"source": session_id, "kind": "pose3d"}),
                    "output_path": str(pose_path.resolve()),
                    "output_sha256": sha256_file(pose_path),
                    "output_shape": [behavior_frame_count, 38, 3],
                    "output_dtype": "float32",
                },
                "receipt_sha256",
            )

            window_rows = []
            for window_index in range(2):
                input_start = window_index * 50
                input_end = input_start + 300
                target_end = input_end + 50
                last_input_neural_index = 10 + 2 * window_index
                sample_id = f"{session_id}:window:{window_index}"
                window_rows.append(
                    {
                        "sample": {
                            "dataset_id": "mc2p_v1",
                            "sample_id": sample_id,
                            "animal_id": animal,
                            "session_id": session_id,
                        },
                        "input_behavior_frames": [input_start, input_end],
                        "target_behavior_frames": [input_end, target_end],
                        "last_input_neural_index": last_input_neural_index,
                        "target_neural_indices": [
                            last_input_neural_index + 1,
                            last_input_neural_index + 2,
                        ],
                    }
                )
                source_sample_rows.append(
                    {"sample_id": sample_id, "animal_id": animal, "session_id": session_id}
                )
                sample_to_split[sample_id] = split_by_session[session_id]
                sample_to_animal[sample_id] = animal
            windows_path = session_dir / "windows.json"
            windows_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_id": "mc2p_v1",
                        "session": {"session_id": session_id, "animal_id": animal},
                        "alignment_path": str(alignment_path.resolve()),
                        "alignment_length_behavior_frames": behavior_frame_count,
                        "target_neural_boundary_policy": TARGET_NEURAL_BOUNDARY_POLICY,
                        "history_s": 3.0,
                        "horizon_s": 0.5,
                        "stride_s": 0.5,
                        "window_count": len(window_rows),
                        "windows": window_rows,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            batch_path = session_dir / "pose-neural-batch.npz"
            batch_path.write_bytes(f"opaque-npz:{session_id}".encode())
            dff_sha = _sha({"measured_dff": session_id})
            batch_receipt = _write_self_hashed(
                session_dir / "pose-neural-batch.json",
                {
                    "schema_version": 1,
                    "protocol": "mc2p-pose-neural-session-batch-v1",
                    "benchmark_id": "mc2p_future_neural_v1",
                    "dataset_id": "mc2p_v1",
                    "animal_id": animal,
                    "session_id": session_id,
                    "sample_count": 2,
                    "windows_sha256": sha256_file(windows_path),
                    "pose3d_sha256": sha256_file(pose_path),
                    "dff_sha256": dff_sha,
                    "output_sha256": sha256_file(batch_path),
                    "feature_shape": [2, 570],
                    "target_shape": [2, 4096],
                    "pose_feature_contract": {"uses_input_behavior_frames_only": True},
                    "target_contract": {
                        "operation": "mean_measured_dff_over_unique_future_neural_indices",
                        "dff_side": 64,
                        "evidence_class": "measured_neural_activity",
                    },
                },
                "receipt_sha256",
            )
            source_batches.append(
                {"path": str(batch_path.resolve()), "sha256": sha256_file(batch_path)}
            )
            preparation_sessions.append(
                {
                    "animal_id": animal,
                    "session_id": session_id,
                    "behavior_frame_count": behavior_frame_count,
                    "alignment_conversion_sha256": alignment_receipt["receipt_sha256"],
                    "pose_conversion_sha256": pose_receipt["receipt_sha256"],
                    "windows_sha256": sha256_file(windows_path),
                    "batch_receipt_sha256": batch_receipt["receipt_sha256"],
                    "batch_sha256": sha256_file(batch_path),
                    "dff_source_sha256": dff_sha,
                    "dff_target_kind": "resized_measured_dff_64x64",
                }
            )

    manifest_path = root / "mc2p-manifest.json"
    manifest = _write_self_hashed(
        manifest_path,
        {
            "schema_version": 1,
            "dataset_id": "mc2p_v1",
            "root": "/raw/mc2p",
            "animal_count": len(animals),
            "session_count": len(manifest_sessions),
            "animals": animals,
            "neural_representations": ["two_photon_dff_imaging"],
            "sessions": manifest_sessions,
            "pickle_policy": "fixture",
        },
        "manifest_sha256",
    )

    source_batches = sorted(source_batches, key=lambda row: row["path"])
    split_path = root / "session-split-lock.json"
    split_lock = _write_self_hashed(
        split_path,
        {
            "schema_version": 1,
            "benchmark_id": "mc2p_future_neural_v1",
            "split_unit": "session_id_within_animal",
            "split_seed": SPLIT_SEED,
            "source_samples_sha256": _sha(
                sorted(source_sample_rows, key=lambda row: row["sample_id"])
            ),
            "source_batches": source_batches,
            "animal_sessions": animal_sessions,
            "sample_to_split": dict(sorted(sample_to_split.items())),
            "sample_to_animal": dict(sorted(sample_to_animal.items())),
            "validation_sessions_per_animal": 1,
            "test_sessions_per_animal": 1,
            "test_sessions_for_hyperparameter_selection": False,
        },
        "split_lock_sha256",
    )

    return _write_self_hashed(
        root / "preparation-receipt.json",
        {
            "schema_version": 1,
            "protocol": "mc2p-v1-preparation-v1",
            "benchmark_id": "mc2p_future_neural_v1",
            "dataset_id": "mc2p_v1",
            "expected_public_release_animals": 8,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": sha256_file(manifest_path),
            "animal_count": 8,
            "session_count": len(preparation_sessions),
            "target_neural_boundary_policy": TARGET_NEURAL_BOUNDARY_POLICY,
            "preparation_implementation_fingerprint": preparation_implementation_fingerprint(),
            "source_batches": source_batches,
            "split_lock_sha256": split_lock["split_lock_sha256"],
            "split_lock_file_sha256": sha256_file(split_path),
            "sessions": sorted(preparation_sessions, key=lambda row: row["session_id"]),
            "models_fit": False,
            "test_data_consumed": False,
            "claim_boundary": "fixture",
        },
        "receipt_sha256",
    )


def test_preflight_verifies_preparation_without_deserializing_batches(tmp_path: Path, monkeypatch):
    root = tmp_path / "prepared"
    _build_prepared_fixture(root)

    def fail_if_np_load_is_called(*args, **kwargs):
        raise AssertionError("preflight must not deserialize NumPy batch arrays")

    monkeypatch.setattr(np, "load", fail_if_np_load_is_called)
    report = audit_preparation_directory(root)
    assert report["status"] == "pass"
    assert report["animal_count"] == 8
    assert report["session_count"] == 24
    assert report["verified_session_batches"] == 24
    assert report["verified_prediction_windows"] == 48
    assert report["preflight_implementation_sha256"]
    assert report["batch_arrays_deserialized"] is False
    assert report["model_metrics_inspected"] is False
    assert report["test_data_consumed"] is False


def test_preflight_rejects_tampered_batch_bytes(tmp_path: Path):
    root = tmp_path / "prepared"
    _build_prepared_fixture(root)
    first_batch = next((root / "sessions").glob("*/pose-neural-batch.npz"))
    first_batch.write_bytes(first_batch.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="prepared session batch bytes"):
        audit_preparation_directory(root)


def test_preflight_rejects_rehashed_split_with_duplicate_partition(tmp_path: Path):
    root = tmp_path / "prepared"
    _build_prepared_fixture(root)
    split_path = root / "session-split-lock.json"
    split_lock = json.loads(split_path.read_text())
    split_lock.pop("split_lock_sha256")
    first_animal = min(split_lock["animal_sessions"])
    split_lock["animal_sessions"][first_animal]["test"] = list(
        split_lock["animal_sessions"][first_animal]["validation"]
    )
    split_lock = _write_self_hashed(split_path, split_lock, "split_lock_sha256")

    def update_outer(receipt: dict) -> None:
        receipt["split_lock_sha256"] = split_lock["split_lock_sha256"]
        receipt["split_lock_file_sha256"] = sha256_file(split_path)

    _rewrite_preparation_receipt(root, update_outer)
    with pytest.raises(ValueError, match="multiple splits"):
        audit_preparation_directory(root)


def test_preflight_rejects_rehashed_window_identity_drift(tmp_path: Path):
    root = tmp_path / "prepared"
    _build_prepared_fixture(root)
    session_dir = min((root / "sessions").iterdir())

    def mutate(windows: dict) -> None:
        windows["windows"][0]["sample"]["animal_id"] = "different_animal"

    _rewrite_windows_and_bind(root, session_dir, mutate)
    with pytest.raises(ValueError, match="window sample identity changed"):
        audit_preparation_directory(root)


def test_preflight_rejects_rehashed_strict_future_violation(tmp_path: Path):
    root = tmp_path / "prepared"
    _build_prepared_fixture(root)
    session_dir = min((root / "sessions").iterdir())

    def mutate(windows: dict) -> None:
        row = windows["windows"][0]
        row["target_neural_indices"] = [row["last_input_neural_index"]]

    _rewrite_windows_and_bind(root, session_dir, mutate)
    with pytest.raises(ValueError, match="strict-future neural indices violated"):
        audit_preparation_directory(root)


def test_preflight_rejects_rehashed_claim_of_model_fitting(tmp_path: Path):
    root = tmp_path / "prepared"
    _build_prepared_fixture(root)
    _rewrite_preparation_receipt(root, lambda receipt: receipt.__setitem__("models_fit", True))
    with pytest.raises(ValueError, match="modeling or test consumption"):
        audit_preparation_directory(root)


def test_preflight_rejects_rehashed_conversion_output_repointing(tmp_path: Path):
    root = tmp_path / "prepared"
    _build_prepared_fixture(root)
    session_dir = min((root / "sessions").iterdir())
    conversion_path = session_dir / "alignment-conversion.json"
    conversion = json.loads(conversion_path.read_text())
    conversion.pop("receipt_sha256")
    conversion["output_path"] = str((root / "elsewhere.npy").resolve())
    conversion = _write_self_hashed(conversion_path, conversion, "receipt_sha256")

    def update_outer(receipt: dict) -> None:
        for row in receipt["sessions"]:
            if row["session_id"] == session_dir.name:
                row["alignment_conversion_sha256"] = conversion["receipt_sha256"]
                break

    _rewrite_preparation_receipt(root, update_outer)
    with pytest.raises(ValueError, match="conversion output path mismatch"):
        audit_preparation_directory(root)
