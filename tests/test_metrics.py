import numpy as np

from fly_sniff.metrics import paired_bootstrap_delta, spl


def test_spl_failure_is_zero():
    assert spl(False, 10.0, 10.0) == 0.0


def test_spl_penalizes_detours():
    assert spl(True, 10.0, 10.0) == 1.0
    assert spl(True, 10.0, 20.0) == 0.5


def test_paired_bootstrap_positive_delta():
    a = np.array([0.8, 0.7, 0.9, 0.6])
    b = np.array([0.3, 0.2, 0.4, 0.1])
    mean, lo, hi = paired_bootstrap_delta(a, b, seed=1, n_boot=1000)
    assert mean > 0
    assert lo > 0
    assert hi > 0
