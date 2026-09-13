from __future__ import annotations

import copy

import pytest

from fly_sniff.odor_motion import (
    BilateralOdorMotionEstimator,
    OdorMotionConfig,
    analyze_recording_bundle,
    load_odor_motion_config,
)
from fly_sniff.recording import build_recording


def _run_sequence(sequence: list[tuple[float, float]]):
    estimator = BilateralOdorMotionEstimator(dt=0.05)
    return [
        estimator.update(
            step=step,
            t=step * 0.05,
            left_response=left,
            right_response=right,
        )
        for step, (left, right) in enumerate(sequence)
    ]


def test_left_leading_right_produces_positive_motion_evidence():
    estimates = _run_sequence([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    final = estimates[-1]

    assert final.evidence > 0.0
    assert final.confidence > 0.08
    assert final.direction == "left_to_right"


def test_mirroring_antennas_reverses_motion_evidence_sign():
    sequence = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    forward = _run_sequence(sequence)
    mirrored = _run_sequence([(right, left) for left, right in sequence])

    for first, second in zip(forward, mirrored, strict=True):
        assert second.evidence == pytest.approx(-first.evidence, abs=1e-12)
        assert second.confidence == pytest.approx(first.confidence, abs=1e-12)

    assert forward[2].direction == "left_to_right"
    assert mirrored[2].direction == "right_to_left"


def test_simultaneous_bilateral_pulse_has_no_direction_evidence():
    estimates = _run_sequence([(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)])

    assert all(abs(item.evidence) < 1e-12 for item in estimates)
    assert estimates[-1].direction in {"ambiguous", "no_signal"}


def test_estimator_is_prefix_causal():
    prefix = [(0.0, 0.0), (0.7, 0.0), (0.1, 0.8), (0.0, 0.2)]
    future = [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]

    prefix_only = _run_sequence(prefix)
    with_future = _run_sequence(prefix + future)

    assert with_future[: len(prefix)] == prefix_only


def test_duplicate_realized_lags_are_rejected():
    config = OdorMotionConfig(delays_s=(0.01, 0.02)).validated()
    with pytest.raises(ValueError, match="duplicate sample lags"):
        BilateralOdorMotionEstimator(dt=0.05, config=config)


def test_recording_analysis_is_hash_bound_and_v1_read_only():
    source = build_recording(seed=31, sim_seconds=0.25, plume_points=16)
    config_document, config = load_odor_motion_config()
    result = analyze_recording_bundle(
        source,
        config_document=config_document,
        config=config,
    )

    analysis = result["analysis"]
    assert analysis["source_recording_sha256"] == source["recording_sha256"]
    assert analysis["controller_access_in_v1"] is False
    assert analysis["protocol"] == "bilateral-odor-motion-analysis-v2"
    assert "source coordinates" in analysis["input_contract"]["forbidden_privileged_inputs"]
    assert len(analysis["agents"]) == len(source["recording"]["controllers"])
    assert all(
        len(agent["samples"]) == len(source["recording"]["frames"])
        for agent in analysis["agents"]
    )


def test_recording_analysis_rejects_source_hash_tampering():
    source = build_recording(seed=32, sim_seconds=0.10, plume_points=8)
    tampered = copy.deepcopy(source)
    tampered["recording"]["frames"][0]["agents"][0]["sensor_trace"]["left"][
        "response"
    ] += 0.1
    config_document, config = load_odor_motion_config()

    with pytest.raises(ValueError, match="source recording SHA-256 mismatch"):
        analyze_recording_bundle(
            tampered,
            config_document=config_document,
            config=config,
        )
