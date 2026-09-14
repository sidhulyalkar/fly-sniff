from __future__ import annotations

import numpy as np
import pytest

from fly_video_neural.claims import validate_public_label
from fly_video_neural.metrics import MeanTargetBaseline, summarize_metrics
from fly_video_neural.schema import EvidenceClass


def test_perfect_prediction_scores_one():
    truth = np.array([[0.0, 2.0], [1.0, 4.0], [2.0, 6.0]])
    report = summarize_metrics(truth, truth.copy())
    assert report["mean_pearson_r"] == pytest.approx(1.0)
    assert report["mean_r2"] == pytest.approx(1.0)


def test_mean_target_baseline_uses_training_targets_only():
    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    baseline = MeanTargetBaseline().fit(train)
    assert np.allclose(baseline.predict(2), [[1.0, 3.0], [1.0, 3.0]])


def test_modeled_state_cannot_be_labeled_measured_activity():
    with pytest.raises(ValueError, match="modeled latent state"):
        validate_public_label(
            "Measured neural activity across the full connectome",
            [EvidenceClass.MODELED_LATENT_NEURAL_STATE],
        )


def test_measured_neural_wording_requires_measured_evidence():
    validate_public_label(
        "Measured neural activity target",
        [EvidenceClass.MEASURED_NEURAL_ACTIVITY],
    )
