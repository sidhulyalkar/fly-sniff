from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_video_neural import session_benchmark
from fly_video_neural.alignment import TARGET_NEURAL_BOUNDARY_POLICY
from fly_video_neural.development_protocol import run_development_protocol
from fly_video_neural.provenance import preparation_implementation_fingerprint
from fly_video_neural.session_benchmark import (
    SessionBenchmarkBatch,
    build_session_split_lock,
    load_session_batches,
)


def _sha(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_batches(root: Path) -> tuple[list[str], dict]:
    paths: list[str] = []
    chunks: list[SessionBenchmarkBatch] = []
    for animal_index in range(8):
        animal = f"fly{animal_index}"
        for session_index in range(3):
            session = f"{animal}_{session_index + 1:03d}"
            sample_ids = []
            features = []
            targets = []
            for index in range(8):
                sample_ids.append(f"{session}:{index * 50}-{index * 50 + 350}")
                features.append([index, index * index, animal_index + session_index * 0.01])
                targets.append([2 * index + animal_index, -index + session_index * 0.02])
            path = root / f"{session}.npz"
            np.savez_compressed(
                path,
                sample_ids=np.asarray(sample_ids),
                animal_ids=np.asarray([animal] * 8),
                session_ids=np.asarray([session] * 8),
                features=np.asarray(features, dtype=float),
                targets=np.asarray(targets, dtype=float),
            )
            paths.append(str(path))
            chunks.append(
                SessionBenchmarkBatch(
                    sample_ids=np.asarray(sample_ids),
                    animal_ids=np.asarray([animal] * 8),
                    session_ids=np.asarray([session] * 8),
                    features=np.asarray(features, dtype=float),
                    targets=np.asarray(targets, dtype=float),
                )
            )
    combined = SessionBenchmarkBatch(
        sample_ids=np.concatenate([chunk.sample_ids for chunk in chunks]),
        animal_ids=np.concatenate([chunk.animal_ids for chunk in chunks]),
        session_ids=np.concatenate([chunk.session_ids for chunk in chunks]),
        features=np.concatenate([chunk.features for chunk in chunks]),
        targets=np.concatenate([chunk.targets for chunk in chunks]),
    )
    _, sources = load_session_batches(paths)
    lock = build_session_split_lock(combined, source_batches=sources)
    return paths, lock


def _write_preparation_receipt(
    path: Path,
    split_path: Path,
    batches: list[str],
    split_lock: dict,
) -> dict:
    _, source_batches = load_session_batches(batches)
    sessions = []
    for batch in batches:
        session_id = Path(batch).stem
        animal_id = session_id.split("_")[0]
        sessions.append(
            {
                "animal_id": animal_id,
                "session_id": session_id,
                "batch_sha256": _file_sha(Path(batch)),
            }
        )
    receipt = {
        "schema_version": 1,
        "protocol": "mc2p-v1-preparation-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "dataset_id": "mc2p_v1",
        "expected_public_release_animals": 8,
        "animal_count": 8,
        "session_count": 24,
        "target_neural_boundary_policy": TARGET_NEURAL_BOUNDARY_POLICY,
        "preparation_implementation_fingerprint": preparation_implementation_fingerprint(),
        "source_batches": source_batches,
        "split_lock_sha256": split_lock["split_lock_sha256"],
        "split_lock_file_sha256": _file_sha(split_path),
        "sessions": sessions,
        "models_fit": False,
        "test_data_consumed": False,
    }
    receipt["receipt_sha256"] = _sha(receipt)
    path.write_text(json.dumps(receipt))
    return receipt


def _package() -> Path:
    return Path(__file__).resolve().parents[1]


def test_development_protocol_requires_preparation_chain_and_never_emits_test_metrics(
    tmp_path: Path,
):
    batches, lock = _write_batches(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(lock))
    preparation = tmp_path / "preparation-receipt.json"
    preparation_receipt = _write_preparation_receipt(preparation, split, batches, lock)
    report = run_development_protocol(
        split,
        batches,
        tmp_path / "development",
        preparation_receipt_path=preparation,
        qc_config_path=_package() / "configs" / "data_qc_v1.json",
        acceptance_config_path=_package() / "configs" / "validation_acceptance_v1.json",
    )
    assert report["preparation_receipt_sha256"] == preparation_receipt["receipt_sha256"]
    assert report["target_neural_boundary_policy"] == TARGET_NEURAL_BOUNDARY_POLICY
    assert report["test_consumption_capability"] is False
    assert report["test_metrics_present"] is False
    assert report["test_target_arrays_deserialized"] is False
    assert len(report["held_out_test_batches_authenticated_not_deserialized"]) == 8
    assert len(report["development_deserialized_batches"]) == 16
    assert report["implementation_fingerprint"]["sha256"]
    assert report["runtime_fingerprint"]["sha256"]
    assert report["primary_metric_scope"] == "all_measured_dff_pixels_with_finite_correlation"
    aligned = json.loads((tmp_path / "development" / "aligned-ridge-development.json").read_text())
    null = json.loads((tmp_path / "development" / "temporal-null-development.json").read_text())
    qc = json.loads((tmp_path / "development" / "development-qc.json").read_text())
    assert qc["test_target_arrays_deserialized"] is False
    assert aligned["test_status"] == "locked_not_consumed"
    assert null["test_status"] == "locked_not_consumed"
    assert all(row["test_metrics"] is None for row in aligned["animals"])
    assert all(row["test_metrics"] is None for row in null["animals"])
    assert all(
        candidate["test_metrics"] is None
        for row in null["animals"]
        for candidate in row["null_candidates"]
    )


def test_development_never_calls_numpy_load_on_test_batches(tmp_path: Path, monkeypatch):
    batches, lock = _write_batches(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(lock))
    preparation = tmp_path / "preparation-receipt.json"
    _write_preparation_receipt(preparation, split, batches, lock)
    test_sessions = {
        session
        for assignments in lock["animal_sessions"].values()
        for session in assignments["test"]
    }
    forbidden = {
        str(Path(batch).resolve()) for batch in batches if Path(batch).stem in test_sessions
    }

    original_load = session_benchmark.np.load

    def guarded_load(path, *args, **kwargs):
        assert str(Path(path).resolve()) not in forbidden
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(session_benchmark.np, "load", guarded_load)
    report = run_development_protocol(
        split,
        batches,
        tmp_path / "development",
        preparation_receipt_path=preparation,
        qc_config_path=_package() / "configs" / "data_qc_v1.json",
        acceptance_config_path=_package() / "configs" / "validation_acceptance_v1.json",
    )
    assert report["test_target_arrays_deserialized"] is False


def test_development_rejects_batch_changed_after_preparation_receipt(tmp_path: Path):
    batches, lock = _write_batches(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(lock))
    preparation = tmp_path / "preparation-receipt.json"
    _write_preparation_receipt(preparation, split, batches, lock)
    Path(batches[0]).write_bytes(b"changed after prepare")
    with pytest.raises(ValueError, match="session-batch bytes"):
        run_development_protocol(
            split,
            batches,
            tmp_path / "development",
            preparation_receipt_path=preparation,
            qc_config_path=_package() / "configs" / "data_qc_v1.json",
            acceptance_config_path=_package() / "configs" / "validation_acceptance_v1.json",
        )
