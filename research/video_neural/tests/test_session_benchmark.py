from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fly_video_neural.session_benchmark import (
    SessionBenchmarkBatch,
    build_session_split_lock,
    run_within_animal_ridge,
    verify_session_split_lock,
)


def _batch() -> SessionBenchmarkBatch:
    rng = np.random.default_rng(23)
    sample_ids: list[str] = []
    animal_ids: list[str] = []
    session_ids: list[str] = []
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights = [rng.normal(size=(5, 3)) for _ in range(2)]
    for animal in range(2):
        for session in range(4):
            for sample in range(12):
                x = rng.normal(size=5)
                y = x @ weights[animal] + rng.normal(scale=0.01, size=3)
                sample_ids.append(f"a{animal}s{session}-{sample}")
                animal_ids.append(f"a{animal}")
                session_ids.append(f"a{animal}_s{session}")
                features.append(x)
                targets.append(y)
    return SessionBenchmarkBatch(
        np.asarray(sample_ids),
        np.asarray(animal_ids),
        np.asarray(session_ids),
        np.asarray(features),
        np.asarray(targets),
    )


def test_session_split_is_one_validation_one_test_per_animal():
    lock = build_session_split_lock(_batch())
    for assignment in lock["animal_sessions"].values():
        assert len(assignment["validation"]) == 1
        assert len(assignment["test"]) == 1
        assert len(assignment["train"]) == 2


def test_session_split_is_deterministic_and_hash_bound():
    batch = _batch()
    first = build_session_split_lock(batch)
    second = build_session_split_lock(batch)
    assert first["split_lock_sha256"] == second["split_lock_sha256"]
    verify_session_split_lock(first, batch)


def test_semantic_split_tamper_fails_even_after_rehash():
    batch = _batch()
    lock = build_session_split_lock(batch)
    tampered = deepcopy(lock)
    sample = next(iter(tampered["sample_to_split"]))
    tampered["sample_to_split"][sample] = "test"
    from fly_video_neural.session_benchmark import _sha

    payload = dict(tampered)
    payload.pop("split_lock_sha256")
    tampered["split_lock_sha256"] = _sha(payload)
    with pytest.raises(ValueError, match="does not match"):
        verify_session_split_lock(tampered, batch)


def test_within_animal_ridge_keeps_test_locked_by_default():
    batch = _batch()
    lock = build_session_split_lock(batch)
    report = run_within_animal_ridge(batch, lock)
    assert report["test_status"] == "locked_not_consumed"
    assert report["aggregate_test_metrics"] is None
    assert all(row["test_metrics"] is None for row in report["animals"])


def test_explicit_test_consumption_scores_each_animal_without_cross_animal_fit():
    batch = _batch()
    lock = build_session_split_lock(batch)
    report = run_within_animal_ridge(batch, lock, consume_test=True)
    assert report["test_status"] == "consumed_explicitly"
    assert report["aggregate_test_metrics"]["animal_count"] == 2
    assert report["aggregate_test_metrics"]["median_of_animal_median_pearson_r"] > 0.95
    assert "No raw neural pixel correspondence" in report["claim_boundary"]
