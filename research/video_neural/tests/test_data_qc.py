from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fly_video_neural.data_qc import audit_development_data, load_qc_config
from fly_video_neural.session_benchmark import SessionBenchmarkBatch, build_session_split_lock


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "data_qc_v1.json"
    return load_qc_config(path)


def _batch(samples_per_session: int = 5) -> SessionBenchmarkBatch:
    sample_ids: list[str] = []
    animal_ids: list[str] = []
    session_ids: list[str] = []
    features: list[list[float]] = []
    targets: list[list[float]] = []
    for session_index in range(3):
        session = f"flyA_{session_index + 1:03d}"
        for index in range(samples_per_session):
            start = index * 50
            sample_ids.append(f"{session}:{start}-{start + 350}")
            animal_ids.append("flyA")
            session_ids.append(session)
            features.append([index, session_index, index + session_index, index * 0.5])
            targets.append([index + session_index, index * 2 + 1, session_index + index * 0.25])
    return SessionBenchmarkBatch(
        sample_ids=np.asarray(sample_ids),
        animal_ids=np.asarray(animal_ids),
        session_ids=np.asarray(session_ids),
        features=np.asarray(features, dtype=float),
        targets=np.asarray(targets, dtype=float),
    )


def test_qc_never_summarizes_test_target_values():
    batch = _batch()
    lock = build_session_split_lock(batch)
    first = audit_development_data(batch, lock, _config())
    test_sessions = set(lock["animal_sessions"]["flyA"]["test"])
    changed = batch.targets.copy()
    mask = np.asarray([session in test_sessions for session in batch.session_ids])
    changed[mask] += 1_000_000.0
    mutated = SessionBenchmarkBatch(
        sample_ids=batch.sample_ids,
        animal_ids=batch.animal_ids,
        session_ids=batch.session_ids,
        features=batch.features,
        targets=changed,
    )
    second = audit_development_data(mutated, lock, _config())
    assert first == second
    assert first["test_target_values_summarized"] is False
    test_rows = [row for row in first["sessions"] if row["split"] == "test"]
    assert test_rows and all("target_distribution" not in row for row in test_rows)


def test_qc_blocks_development_sessions_with_too_few_windows():
    batch = _batch(samples_per_session=1)
    lock = build_session_split_lock(batch)
    report = audit_development_data(batch, lock, _config())
    assert report["status"] == "blocked"
    assert any("only 1 prediction windows" in message for message in report["structural_failures"])


def test_qc_reports_fixed_autocorrelation_grid_and_hash():
    batch = _batch()
    lock = build_session_split_lock(batch)
    report = audit_development_data(batch, lock, _config())
    assert report["status"] == "pass"
    assert len(report["report_sha256"]) == 64
    lags = sorted({row["lag_windows"] for row in report["development_diagnostics"]["autocorrelation"]})
    assert lags == [1, 2, 4, 10, 20]
    assert report["development_diagnostics"]["target_distribution"]["dimensions"] == 3


def test_shipped_qc_config_is_json_serializable_and_frozen():
    config = _config()
    assert config["scope"] == "train_and_validation_targets_only"
    assert config["test_target_values_may_be_summarized"] is False
    json.dumps(config, allow_nan=False)
