from __future__ import annotations

import numpy as np
import pytest

from fly_video_neural.batch import FeatureTargetBatch
from fly_video_neural.ridge import RidgeDecoder, run_ridge_baseline
from fly_video_neural.schema import SampleWindow
from fly_video_neural.split_lock import build_split_lock


def _fixture():
    rng = np.random.default_rng(11)
    windows = []
    ids = []
    features = []
    targets = []
    weights = np.array([[2.0, -1.0], [0.5, 1.5], [-0.7, 0.2]])
    for animal in range(8):
        for sample in range(12):
            sample_id = f"fly{animal}-{sample}"
            x = rng.normal(size=3)
            y = x @ weights + rng.normal(scale=0.03, size=2)
            ids.append(sample_id)
            features.append(x)
            targets.append(y)
            windows.append(
                SampleWindow(
                    dataset_id="mc2p_v1",
                    sample_id=sample_id,
                    animal_id=f"fly{animal}",
                    session_id=f"fly{animal}_001",
                    input_start_s=sample * 4.0,
                    input_end_s=sample * 4.0 + 3.0,
                    target_start_s=sample * 4.0 + 3.0,
                    target_end_s=sample * 4.0 + 3.5,
                    video_fps=100.0,
                )
            )
    batch = FeatureTargetBatch(np.array(ids), np.array(features), np.array(targets))
    return batch, build_split_lock(windows)


def test_ridge_learns_linear_mapping():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 4))
    y = x @ rng.normal(size=(4, 3))
    model = RidgeDecoder(0.01).fit(x, y)
    assert np.mean((model.predict(x) - y) ** 2) < 1e-6


def test_test_split_is_locked_by_default():
    batch, lock = _fixture()
    report = run_ridge_baseline(batch, lock)
    assert report["test_status"] == "locked_not_consumed"
    assert report["test_metrics"] is None
    assert report["selected_alpha"] in {0.01, 0.1, 1.0, 10.0, 100.0}


def test_explicit_test_consumption_scores_ridge_and_mean():
    batch, lock = _fixture()
    report = run_ridge_baseline(batch, lock, consume_test=True)
    assert report["test_status"] == "consumed_explicitly"
    assert report["test_metrics"]["median_pearson_r"] > 0.95
    assert report["test_metrics"]["mean_r2"] > report["mean_baseline_test_metrics"]["mean_r2"]


def test_batch_must_match_split_lock_exactly():
    batch, lock = _fixture()
    bad = FeatureTargetBatch(batch.sample_ids[:-1], batch.features[:-1], batch.targets[:-1])
    with pytest.raises(ValueError, match="sample mismatch"):
        run_ridge_baseline(bad, lock)
