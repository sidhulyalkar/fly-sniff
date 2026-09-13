from __future__ import annotations

import numpy as np
import pytest

from fly_sniff.experimental_plume_cues import (
    crosswind_gradient,
    published_motion_cue,
)


def test_crosswind_gradient_recovers_linear_y_field():
    y = np.arange(5, dtype=np.float32)[:, None]
    frame = np.repeat(3.0 * y + 2.0, 4, axis=1)
    assert np.allclose(crosswind_gradient(frame), 3.0)


def test_motion_cue_uses_published_negative_gradient_times_temporal_sign():
    y = np.arange(5, dtype=np.float32)[:, None]
    current = np.repeat(2.0 * y, 3, axis=1)
    previous = current - 4.0
    following = current + 4.0
    expected = -8.0 * np.ones_like(current)
    assert np.allclose(published_motion_cue(previous, current, following), expected)


def test_motion_cue_rejects_mismatched_frames():
    with pytest.raises(ValueError, match="share one shape"):
        published_motion_cue(np.zeros((3, 3)), np.zeros((3, 4)), np.zeros((3, 3)))
