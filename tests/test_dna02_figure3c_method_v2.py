from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fly_sniff.dna02_figure3c_method_v2 import (
    alignment_slices,
    bilateral_right_minus_left,
    bin_spike_indices_10ms,
    firing_rate_hz_from_counts,
    gp_random_walk_map_v2,
    load_method_contract,
    method_status,
    nonoverlap_mean_50ms,
    smoothts_exponential_v2,
)

CONTRACT = Path("authority/program-a-dna02-figure3c-method-contract-v2.json")
HASH = Path("authority/program-a-dna02-figure3c-method-contract-v2.sha256")


def test_contract_is_bound_to_frozen_thresholds_and_keeps_behavior_sealed() -> None:
    payload = load_method_contract(CONTRACT, HASH)
    assert payload["status"] == "METHOD_CONTRACT_FROZEN_BEHAVIOR_STILL_SEALED"
    assert payload["threshold_freeze"]["receipt_sha256"] == (
        "a308eb18cef1d5e524243457e5cae3d3caf0803855b43ae722e643548f11aad8"
    )
    assert payload["threshold_freeze"]["manifest_sha256"] == (
        "1367113491adc3625d52a9fdc7214820677d19495df8c56066aacd991711d082"
    )
    assert payload["gates"]["thresholds_frozen"] is True
    assert payload["gates"]["behavior_fields_opened"] is False
    assert payload["gates"]["yaw_numeric_values_inspected"] is False
    assert payload["gates"]["figure3c_statistic_computed"] is False
    assert payload["gates"]["navigation_performance_used"] is False
    assert payload["gates"]["behavior_opening_allowed"] is False


def test_smoothts_reference_fixture_and_impulse_are_deterministic() -> None:
    fixture = np.array([1, 2, 3, 4, 5], dtype=float)
    np.testing.assert_allclose(
        smoothts_exponential_v2(fixture),
        np.array([1, 1.5, 2.25, 3.125, 4.0625]),
        rtol=0,
        atol=1e-12,
    )
    impulse = np.array([0, 1, 0, 0, 0], dtype=float)
    np.testing.assert_allclose(
        smoothts_exponential_v2(impulse),
        np.array([0, 0.5, 0.25, 0.125, 0.0625]),
        rtol=0,
        atol=1e-12,
    )


def test_ten_ms_spike_binning_uses_exact_integer_boundaries() -> None:
    counts = bin_spike_indices_10ms(
        np.array([0, 99, 100, 199, 999]),
        n_ephys_samples=1000,
    )
    assert counts.tolist() == [2, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    np.testing.assert_array_equal(
        firing_rate_hz_from_counts(counts),
        np.array([200, 200, 0, 0, 0, 0, 0, 0, 0, 100], dtype=float),
    )


def test_spike_binning_fails_closed_on_partial_or_out_of_range_data() -> None:
    with pytest.raises(ValueError, match="complete 10 ms bins"):
        bin_spike_indices_10ms(np.array([], dtype=int), n_ephys_samples=1001)
    with pytest.raises(ValueError, match="outside authenticated"):
        bin_spike_indices_10ms(np.array([1000]), n_ephys_samples=1000)


def test_bilateral_predictor_is_right_minus_left() -> None:
    right = np.array([4.0, 1.0, 7.0])
    left = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        bilateral_right_minus_left(right, left),
        np.array([3.0, -1.0, 4.0]),
    )


def test_alignment_pairs_neural_with_behavior_150_ms_later() -> None:
    neural_slice, behavior_slice = alignment_slices(100)
    assert neural_slice == slice(0, 85)
    assert behavior_slice == slice(15, 100)

    neural = np.arange(100)
    behavior = np.arange(100) + 1000
    aligned_neural = neural[neural_slice]
    aligned_behavior = behavior[behavior_slice]
    assert aligned_neural[0] == 0
    assert aligned_behavior[0] == 1015
    assert aligned_neural[-1] == 84
    assert aligned_behavior[-1] == 1099


def test_fifty_ms_windows_drop_only_incomplete_trailing_window() -> None:
    values = np.arange(12, dtype=float)
    np.testing.assert_array_equal(
        nonoverlap_mean_50ms(values),
        np.array([2.0, 7.0]),
    )



def test_source_equivalent_gp_map_matches_frozen_synthetic_fixtures() -> None:
    payload = load_method_contract(CONTRACT, HASH)
    fixtures = payload["gp_map_conformance"]["synthetic_fixtures"]
    for fixture in fixtures:
        observed = gp_random_walk_map_v2(np.asarray(fixture["input"], dtype=float))
        np.testing.assert_allclose(
            observed,
            np.asarray(fixture["expected"], dtype=float),
            rtol=0,
            atol=5e-15,
        )


def test_gp_map_solves_frozen_tridiagonal_first_order_condition() -> None:
    y = np.array([2.0, -1.0, 4.0, 0.5, 3.0, -2.0], dtype=float)
    z = gp_random_walk_map_v2(y)
    lam = 0.25
    residual = z - y
    residual[0] += lam * (z[0] - z[1])
    residual[-1] += lam * (z[-1] - z[-2])
    residual[1:-1] += lam * (2 * z[1:-1] - z[:-2] - z[2:])
    np.testing.assert_allclose(residual, np.zeros_like(y), rtol=0, atol=1e-12)


def test_gp_map_handles_empty_singleton_and_rejects_matrix_input() -> None:
    np.testing.assert_array_equal(gp_random_walk_map_v2(np.array([])), np.array([]))
    np.testing.assert_array_equal(gp_random_walk_map_v2(np.array([4.2])), np.array([4.2]))
    with pytest.raises(ValueError, match="one-dimensional"):
        gp_random_walk_map_v2(np.zeros((2, 2)))


def test_gp_source_authority_is_pinned_to_pymc3_v36() -> None:
    payload = load_method_contract(CONTRACT, HASH)
    gp = payload["gp_map_conformance"]
    source = gp["pymc3_source"]
    assert source["commit"] == "081e7f4a55ce45ef50a03f8062611051d9c00ce8"
    assert source["gaussian_random_walk_blob"] == "9c5847271ae70ca26f2fef544d806bb58bf968ed"
    assert source["find_map_blob"] == "491c38b8502e486d287566e29cbd71c9fe7177b0"
    assert gp["matrix_structure"] == {
        "endpoint_diagonal": 1.25,
        "interior_diagonal": 1.5,
        "off_diagonal": -0.25,
    }


def test_historical_integral_correction_is_sensitivity_not_primary() -> None:
    payload = load_method_contract(CONTRACT, HASH)
    historical = payload["smoothts_conformance"]["historical_integral_correction"]
    assert historical["operation"] == "divide smoothts output by 1.5"
    assert "do not apply" in historical["primary_policy"]
    assert "may not replace" in historical["predeclared_sensitivity"]


def test_secondary_kinematic_evidence_is_pinned_but_behavior_remains_blocked() -> None:
    payload = load_method_contract(CONTRACT, HASH)
    secondary = payload["secondary_code_evidence"]
    assert secondary["commit"] == "7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8"
    assert secondary["python"] == "3.7.3"
    assert secondary["pymc3"] == "3.6"
    assert secondary["gaussian_random_walk"]["smooth_ratio_alpha"] == 0.2
    assert secondary["offset_correction"]["threshold_deg_per_s_per_sample"] == 0.025
    assert payload["gates"]["behavior_opening_allowed"] is False


def test_contract_hash_tamper_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["sign_convention"]["predictor_positive_direction"] = "left minus right"
    tampered = tmp_path / "contract.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical hash"):
        load_method_contract(tampered, HASH)


def test_status_exposes_remaining_gate_without_opening_behavior() -> None:
    status = method_status(CONTRACT, HASH)
    assert status["status"] == "METHOD_CONTRACT_FROZEN_BEHAVIOR_STILL_SEALED"
    assert status["gates"]["behavior_opening_allowed"] is False
    assert len(status["remaining_before_behavior_opening"]) == 1
