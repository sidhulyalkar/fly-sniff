from __future__ import annotations

import numpy as np

from fly_sniff.sensory_goal_encoder import _memory_series, _upwind_goal


def test_upwind_goal_rotates_with_heading() -> None:
    base = {
        "left_odor": 1.0,
        "right_odor": 1.0,
        "heading": 0.0,
        "wind_x_body": 1.0,
        "wind_y_body": 0.0,
    }
    rotated = dict(base)
    rotated["heading"] = np.pi / 2.0
    assert np.isclose(_upwind_goal(base), -np.pi)
    assert np.isclose(_upwind_goal(rotated), -np.pi / 2.0)


def test_memory_series_does_not_create_state_before_first_odor() -> None:
    raw = np.array([0.0, 0.0, np.pi / 2.0, np.pi / 2.0], dtype=float)
    odor = np.array([False, False, True, False], dtype=bool)
    out = _memory_series(raw, odor, dt=0.1, tau=1.0)
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert np.isfinite(out[2])
    assert np.isfinite(out[3])
