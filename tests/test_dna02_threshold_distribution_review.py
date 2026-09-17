from __future__ import annotations

import numpy as np
import pytest

from fly_sniff.dna02_threshold_distribution_review import (
    _band_masks,
    _template_correlations,
)


def test_band_masks_are_nested_and_non_overlapping() -> None:
    prominences = np.array([1.0, 2.0, 2.999, 3.0, 3.999, 4.0, 8.0])
    low, mid, high = _band_masks(prominences, (2.0, 3.0, 4.0))
    assert np.flatnonzero(low).tolist() == [1, 2]
    assert np.flatnonzero(mid).tolist() == [3, 4]
    assert np.flatnonzero(high).tolist() == [5, 6]
    assert not np.any(low & mid)
    assert not np.any(low & high)
    assert not np.any(mid & high)


def test_band_masks_reject_unordered_thresholds() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _band_masks(np.array([1.0, 2.0]), (2.0, 2.0, 3.0))


def test_template_correlation_is_scale_invariant_for_same_shape() -> None:
    template = np.array([-1.0, 0.0, 2.0, 1.0, -0.5])
    waves = np.stack([template, template * 0.2, template * 3.5])
    observed = _template_correlations(waves, template)
    np.testing.assert_allclose(observed, np.ones(3), atol=1e-12)


def test_template_correlation_separates_inverted_shape() -> None:
    template = np.array([-1.0, 0.0, 2.0, 1.0, -0.5])
    waves = np.stack([template, -template])
    observed = _template_correlations(waves, template)
    assert observed[0] > 0.999
    assert observed[1] < -0.999
