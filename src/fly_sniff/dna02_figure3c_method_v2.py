from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .freeze import canonical_sha256

DEFAULT_CONTRACT = Path("authority/program-a-dna02-figure3c-method-contract-v2.json")
DEFAULT_HASH = Path("authority/program-a-dna02-figure3c-method-contract-v2.sha256")

_EXPECTED_SCHEMA = "fly-sniff-dna02-figure3c-method-contract-v2"
_EXPECTED_STATUS = "METHOD_CONTRACT_FROZEN_BEHAVIOR_STILL_SEALED"
_EXPECTED_THRESHOLD_RECEIPT = (
    "a308eb18cef1d5e524243457e5cae3d3caf0803855b43ae722e643548f11aad8"
)
_EXPECTED_THRESHOLD_MANIFEST = (
    "1367113491adc3625d52a9fdc7214820677d19495df8c56066aacd991711d082"
)


def load_method_contract(
    path: str | Path = DEFAULT_CONTRACT,
    hash_path: str | Path = DEFAULT_HASH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Figure 3C method contract must be a JSON object")
    expected_hash = Path(hash_path).read_text(encoding="utf-8").strip()
    observed_hash = canonical_sha256(payload)
    if observed_hash != expected_hash:
        raise ValueError("Figure 3C method contract canonical hash does not match")
    if payload.get("schema") != _EXPECTED_SCHEMA:
        raise ValueError("unsupported Figure 3C method-contract schema")
    if payload.get("status") != _EXPECTED_STATUS:
        raise ValueError("Figure 3C method-contract status has drifted")

    freeze = payload.get("threshold_freeze", {})
    if freeze.get("receipt_sha256") != _EXPECTED_THRESHOLD_RECEIPT:
        raise ValueError("Figure 3C method contract is not bound to the frozen v2 threshold receipt")
    if freeze.get("manifest_sha256") != _EXPECTED_THRESHOLD_MANIFEST:
        raise ValueError("Figure 3C method contract is not bound to the frozen v2 threshold manifest")
    if freeze.get("status") != "THRESHOLDS_FROZEN_BEFORE_BEHAVIOR_V2":
        raise ValueError("threshold freeze prerequisite has drifted")

    published = payload.get("version_of_record", {})
    expected = {
        "doi": "10.7554/eLife.102230.3",
        "figure": "Figure 3B-C",
        "spike_bin_ms": 10,
        "smoothing_window_ms": 30,
        "smoothing_period_bins": 3,
        "smoothing_alpha": 0.5,
        "neural_to_behavior_alignment_ms": 150,
        "figure_average_bin_ms": 50,
        "primary_predictor": "right_firing_rate_hz - left_firing_rate_hz",
        "primary_outcome": "rotational_velocity_deg_per_s",
    }
    for key, value in expected.items():
        if published.get(key) != value:
            raise ValueError(f"Figure 3C published-method contract drifted at {key}")

    secondary = payload.get("secondary_code_evidence", {})
    if secondary.get("commit") != "7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8":
        raise ValueError("secondary analysis commit authority has drifted")
    if secondary.get("environment_blob") != "a41641b880dae14323e271f795a1278a6e2d8e44":
        raise ValueError("secondary analysis environment authority has drifted")
    if secondary.get("yaw_scale_to_deg_per_s", {}).get("operation") != (
        "raw_yaw * 0.000395 / 0.000436"
    ):
        raise ValueError("yaw scaling provenance has drifted")

    sign = payload.get("sign_convention", {})
    if sign.get("published_lab_convention") != (
        "clockwise/rightward rotational velocity is positive"
    ):
        raise ValueError("rotational-velocity sign convention has drifted")
    if sign.get("predictor_positive_direction") != "right DNa02 minus left DNa02":
        raise ValueError("bilateral predictor sign has drifted")

    gates = payload.get("gates", {})
    required_false = (
        "behavior_fields_opened",
        "behavior_opening_allowed",
        "figure3c_statistic_computed",
        "navigation_performance_used",
        "yaw_numeric_values_inspected",
    )
    for field in required_false:
        if gates.get(field) is not False:
            raise ValueError(f"pre-behavior method gate violated: {field}")
    if gates.get("thresholds_frozen") is not True:
        raise ValueError("thresholds must be frozen before Figure 3C method qualification")
    return payload


def smoothts_exponential_v2(values: np.ndarray, *, period_bins: int = 3) -> np.ndarray:
    """Portable recurrence matching the frozen legacy smoothts exponential fixture."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("smoothts input must be one-dimensional")
    if x.size == 0:
        return x.copy()
    if period_bins <= 1:
        raise ValueError("v2 requires an integer period length greater than one")
    alpha = 2.0 / (period_bins + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for index in range(1, x.size):
        out[index] = alpha * x[index] + (1.0 - alpha) * out[index - 1]
    return out


def bin_spike_indices_10ms(
    peak_indices: np.ndarray,
    *,
    n_ephys_samples: int,
    ephys_fs_hz: int = 10_000,
) -> np.ndarray:
    """Count zero-based spike indices in non-overlapping 10 ms bins."""
    peaks = np.asarray(peak_indices, dtype=np.int64)
    if peaks.ndim != 1:
        raise ValueError("peak indices must be one-dimensional")
    if n_ephys_samples < 0:
        raise ValueError("n_ephys_samples must be non-negative")
    if ephys_fs_hz != 10_000:
        raise ValueError("Figure 3C v2 is frozen to 10 kHz ephys")
    samples_per_bin = ephys_fs_hz // 100
    if n_ephys_samples % samples_per_bin:
        raise ValueError("v2 requires complete 10 ms bins; trailing partial bins are not allowed")
    if np.any(peaks < 0) or np.any(peaks >= n_ephys_samples):
        raise ValueError("peak index outside authenticated ephys sample range")
    counts = np.zeros(n_ephys_samples // samples_per_bin, dtype=np.int64)
    if peaks.size:
        np.add.at(counts, peaks // samples_per_bin, 1)
    return counts


def firing_rate_hz_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("spike counts must be one-dimensional")
    return counts / 0.010


def bilateral_right_minus_left(right_hz: np.ndarray, left_hz: np.ndarray) -> np.ndarray:
    right = np.asarray(right_hz, dtype=float)
    left = np.asarray(left_hz, dtype=float)
    if right.shape != left.shape or right.ndim != 1:
        raise ValueError("right and left firing-rate vectors must be same-length 1-D arrays")
    return right - left


def alignment_slices(
    n_samples: int,
    *,
    sample_rate_hz: int = 100,
    lag_ms: int = 150,
) -> tuple[slice, slice]:
    """Return neural/behavior slices pairing neural[t] with behavior[t + 150 ms]."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    lag_float = sample_rate_hz * lag_ms / 1000.0
    lag = round(lag_float)
    if abs(lag - lag_float) > 1e-12:
        raise ValueError("alignment must map to an integer sample count")
    if lag >= n_samples and n_samples:
        raise ValueError("alignment lag is not smaller than the time series")
    return slice(0, max(0, n_samples - lag)), slice(lag, n_samples)


def nonoverlap_mean_50ms(values: np.ndarray, *, sample_rate_hz: int = 100) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("window input must be one-dimensional")
    samples_per_window_float = sample_rate_hz * 0.050
    samples_per_window = round(samples_per_window_float)
    if abs(samples_per_window - samples_per_window_float) > 1e-12:
        raise ValueError("50 ms must map to an integer sample count")
    complete = x.size // samples_per_window
    if complete == 0:
        return np.array([], dtype=float)
    return x[: complete * samples_per_window].reshape(complete, samples_per_window).mean(axis=1)


def method_status(
    path: str | Path = DEFAULT_CONTRACT,
    hash_path: str | Path = DEFAULT_HASH,
) -> dict[str, Any]:
    payload = load_method_contract(path, hash_path)
    return {
        "schema": payload["schema"],
        "contract_id": payload["contract_id"],
        "contract_sha256": canonical_sha256(payload),
        "status": payload["status"],
        "threshold_freeze": payload["threshold_freeze"],
        "gates": payload["gates"],
        "remaining_before_behavior_opening": payload["remaining_before_behavior_opening"],
        "allowed_next_action": payload["allowed_next_action"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen DNa02 Figure 3C v2 method contract without opening yaw"
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--hash", dest="hash_path", default=str(DEFAULT_HASH))
    args = parser.parse_args(argv)
    print(json.dumps(method_status(args.contract, args.hash_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
