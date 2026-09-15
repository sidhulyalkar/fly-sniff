import numpy as np
import pytest

from fly_sniff.metrics import paired_bootstrap_delta, shortest_path_to_goal_region, spl


def test_shortest_path_targets_success_region_not_source_center():
    assert shortest_path_to_goal_region(10.0, 0.32) == pytest.approx(9.68)
    assert shortest_path_to_goal_region(0.20, 0.32) == 0.0


def test_spl_failure_is_zero():
    assert spl(False, 10.0, 10.0) == 0.0


def test_spl_penalizes_detours():
    assert spl(True, 10.0, 10.0) == 1.0
    assert spl(True, 10.0, 20.0) == 0.5


def test_goal_region_geometry_prevents_false_perfect_spl():
    shortest = shortest_path_to_goal_region(10.0, 0.32)
    assert spl(True, shortest, 9.9) == pytest.approx(shortest / 9.9)
    assert spl(True, shortest, 9.9) < 1.0


def test_spl_rejects_invalid_distances():
    with pytest.raises(ValueError):
        spl(True, -1.0, 2.0)
    with pytest.raises(ValueError):
        shortest_path_to_goal_region(float("nan"), 0.2)


def test_paired_bootstrap_positive_delta():
    a = np.array([0.8, 0.7, 0.9, 0.6])
    b = np.array([0.3, 0.2, 0.4, 0.1])
    mean, lo, hi = paired_bootstrap_delta(a, b, seed=1, n_boot=1000)
    assert mean > 0
    assert lo > 0
    assert hi > 0


def test_paired_bootstrap_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        paired_bootstrap_delta(np.array([1.0, np.nan]), np.array([0.0, 0.0]))
