from __future__ import annotations

import numpy as np

from fly_video_neural.alignment_null import (
    NULL_FRACTIONS,
    NULL_NAME,
    NULL_SELECTION_RULE,
    circular_session_fraction_shift,
    run_alignment_null,
)
from fly_video_neural.session_benchmark import (
    SessionBenchmarkBatch,
    build_session_split_lock,
    subset_development_batch,
)


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


def test_quartile_session_shifts_preserve_feature_multiset_but_change_pairing():
    batch = _batch()
    mask = batch.session_ids == "flyA_001"
    original = batch.features[mask]
    for fraction in NULL_FRACTIONS:
        shifted = circular_session_fraction_shift(batch, mask, fraction)
        assert sorted(map(tuple, shifted)) == sorted(map(tuple, original))
        assert not np.array_equal(shifted, original)


def test_alignment_null_ensemble_keeps_every_test_score_locked_by_default():
    full_batch = _batch()
    lock = build_session_split_lock(full_batch)
    development_batch = subset_development_batch(full_batch, lock)
    report = run_alignment_null(development_batch, lock)
    assert report["test_status"] == "locked_not_consumed"
    assert report["null_name"] == NULL_NAME
    assert report["null_fractions"] == list(NULL_FRACTIONS)
    assert report["null_selection_rule"] == NULL_SELECTION_RULE
    for row in report["animals"]:
        assert row["test_metrics"] is None
        assert len(row["null_candidates"]) == len(NULL_FRACTIONS)
        assert all(candidate["test_metrics"] is None for candidate in row["null_candidates"])
        strongest = max(
            row["null_candidates"],
            key=lambda candidate: candidate["selected_validation_metrics"]["median_pearson_r"],
        )
        assert row["selected_null_fraction"] == strongest["fraction"]
        assert row["selected_validation_metrics"] == strongest["selected_validation_metrics"]


def test_final_primary_null_remains_the_fraction_selected_on_validation():
    batch = _batch()
    lock = build_session_split_lock(batch)
    report = run_alignment_null(batch, lock, consume_test=True)
    for row in report["animals"]:
        strongest_validation = max(
            row["null_candidates"],
            key=lambda candidate: candidate["selected_validation_metrics"]["median_pearson_r"],
        )
        assert row["selected_null_fraction"] == strongest_validation["fraction"]
        assert row["test_metrics"] == strongest_validation["test_metrics"]
