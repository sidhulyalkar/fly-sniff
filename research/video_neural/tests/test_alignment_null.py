from __future__ import annotations

import numpy as np

from fly_video_neural.alignment_null import circular_half_session_shift, run_alignment_null
from fly_video_neural.session_benchmark import SessionBenchmarkBatch, build_session_split_lock


def _batch() -> SessionBenchmarkBatch:
    sample_ids: list[str] = []
    animals: list[str] = []
    sessions: list[str] = []
    features: list[list[float]] = []
    targets: list[list[float]] = []
    for session_index in range(3):
        session = f"flyA_{session_index + 1:03d}"
        for index in range(8):
            start = index * 50
            sample_ids.append(f"{session}:{start}-{start + 350}")
            animals.append("flyA")
            sessions.append(session)
            features.append([index, index * index + session_index, (-1) ** index])
            targets.append([2 * index + session_index, index * index - session_index])
    return SessionBenchmarkBatch(
        sample_ids=np.asarray(sample_ids),
        animal_ids=np.asarray(animals),
        session_ids=np.asarray(sessions),
        features=np.asarray(features, dtype=float),
        targets=np.asarray(targets, dtype=float),
    )


def test_half_session_shift_preserves_feature_multiset_but_changes_pairing():
    batch = _batch()
    mask = batch.session_ids == "flyA_001"
    shifted = circular_half_session_shift(batch, mask)
    original = batch.features[mask]
    assert sorted(map(tuple, shifted)) == sorted(map(tuple, original))
    assert not np.array_equal(shifted, original)


def test_alignment_null_keeps_test_locked_by_default():
    batch = _batch()
    lock = build_session_split_lock(batch)
    report = run_alignment_null(batch, lock)
    assert report["test_status"] == "locked_not_consumed"
    assert report["null_name"] == "circular_half_session_feature_shift"
    assert all(row["test_metrics"] is None for row in report["animals"])
    assert all("selected_validation_metrics" in row for row in report["animals"])
