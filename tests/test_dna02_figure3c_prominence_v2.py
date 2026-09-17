from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from fly_sniff.dna02_figure3c_method_v2 import load_method_contract


def test_scipy_matches_frozen_strict_peak_prominence_fixtures() -> None:
    payload = load_method_contract()
    fixtures = payload["prominence_conformance"]["synthetic_fixtures"]
    for fixture in fixtures:
        signal = np.asarray(fixture["signal"], dtype=float)
        peaks, _ = find_peaks(signal)
        prominences = peak_prominences(signal, peaks)[0]
        assert peaks.tolist() == fixture["expected_peak_indices"]
        np.testing.assert_allclose(
            prominences,
            np.asarray(fixture["expected_prominences"], dtype=float),
            rtol=0,
            atol=1e-12,
        )


def test_endpoint_maxima_are_not_promoted_to_spikes() -> None:
    signal = np.array([3.0, 0.0, 2.0, 0.0, 4.0])
    peaks, _ = find_peaks(signal, prominence=(0.1, None))
    assert peaks.tolist() == [2]


def test_prominence_conformance_does_not_reopen_thresholds() -> None:
    payload = load_method_contract()
    conformance = payload["prominence_conformance"]
    assert "remain unchanged" in conformance["threshold_policy"]
    assert payload["gates"]["thresholds_frozen"] is True
    assert payload["gates"]["behavior_fields_opened"] is False
