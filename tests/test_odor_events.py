from __future__ import annotations

import json

import pytest

from fly_sniff.odor_events import (
    OdorEventConfig,
    build_counterfactuals,
    build_diagnostics_bundle,
    detect_motion_events,
    detect_odor_state_events,
    load_diagnostics,
    write_diagnostics,
)
from fly_sniff.odor_motion import (
    BilateralOdorMotionEstimator,
    analyze_recording_bundle,
    load_odor_motion_config,
)
from fly_sniff.recording import build_recording


def _sample(
    step: int,
    left: float,
    right: float,
    *,
    evidence: float = 0.0,
    confidence: float = 0.0,
    direction: str = "ambiguous",
    dominant_delay_s: float | None = None,
) -> dict:
    return {
        "step": step,
        "t": step * 0.05,
        "left_response": left,
        "right_response": right,
        "evidence": evidence,
        "confidence": confidence,
        "direction": direction,
        "dominant_delay_s": dominant_delay_s,
    }


def _motion_samples(sequence: list[tuple[float, float]]) -> list[dict]:
    estimator = BilateralOdorMotionEstimator(dt=0.05)
    output = []
    for step, (left, right) in enumerate(sequence):
        estimate = estimator.update(
            step=step,
            t=step * 0.05,
            left_response=left,
            right_response=right,
        )
        output.append(
            _sample(
                step,
                left,
                right,
                evidence=estimate.evidence,
                confidence=estimate.confidence,
                direction=estimate.direction,
                dominant_delay_s=estimate.dominant_delay_s,
            )
        )
    return output


def _event_config() -> OdorEventConfig:
    return OdorEventConfig(
        odor_on_threshold=0.16,
        odor_off_threshold=0.08,
        minimum_state_duration_s=0.10,
        reacquisition_horizon_s=0.50,
        turn_response_horizon_s=0.20,
        high_confidence_threshold=0.35,
        high_confidence_refractory_s=0.10,
        temporal_shift_s=0.15,
    ).validated()


def test_hysteresis_detects_encounter_loss_and_reacquisition_without_chatter():
    samples = [
        _sample(0, 0.01, 0.01),
        _sample(1, 0.01, 0.01),
        _sample(2, 0.30, 0.20),
        _sample(3, 0.28, 0.21),
        _sample(4, 0.12, 0.12),
        _sample(5, 0.04, 0.03),
        _sample(6, 0.04, 0.03),
        _sample(7, 0.11, 0.10),
        _sample(8, 0.22, 0.18),
        _sample(9, 0.24, 0.19),
    ]

    events = detect_odor_state_events(samples, dt=0.05, config=_event_config())

    assert [event["kind"] for event in events] == [
        "encounter",
        "loss",
        "reacquisition",
    ]
    assert [event["step"] for event in events] == [2, 5, 8]
    assert events[-1]["latency_s"] == pytest.approx(0.15)


def test_motion_events_require_confidence_and_use_refractory_onsets():
    samples = [
        _sample(0, 0.0, 0.0),
        _sample(
            1,
            0.4,
            0.2,
            evidence=0.8,
            confidence=0.20,
            direction="left_to_right",
            dominant_delay_s=0.05,
        ),
        _sample(
            2,
            0.4,
            0.2,
            evidence=0.8,
            confidence=0.60,
            direction="left_to_right",
            dominant_delay_s=0.05,
        ),
        _sample(
            3,
            0.4,
            0.2,
            evidence=0.7,
            confidence=0.55,
            direction="left_to_right",
            dominant_delay_s=0.05,
        ),
        _sample(4, 0.1, 0.1),
        _sample(
            5,
            0.2,
            0.4,
            evidence=-0.7,
            confidence=0.70,
            direction="right_to_left",
            dominant_delay_s=0.10,
        ),
    ]

    events = detect_motion_events(samples, dt=0.05, config=_event_config())

    assert [event["step"] for event in events] == [2, 5]
    assert [event["direction"] for event in events] == [
        "left_to_right",
        "right_to_left",
    ]


def test_counterfactuals_preserve_symmetry_and_shifted_channel_marginal():
    sequence = [
        (0.0, 0.0),
        (0.8, 0.0),
        (0.2, 0.9),
        (0.0, 0.3),
        (0.7, 0.1),
        (0.1, 0.8),
    ]
    samples = _motion_samples(sequence)
    _, motion_config = load_odor_motion_config()

    controls = build_counterfactuals(
        samples,
        dt=0.05,
        motion_config=motion_config,
        event_config=_event_config(),
    )

    assert controls["mirror_left_right"]["max_sign_reversal_error"] < 1e-12
    assert controls["zero_temporal_evidence"]["mean_abs_evidence"] == 0.0
    shifted = controls["right_channel_circular_shift"]
    assert shifted["preserves_right_channel_marginal_exactly"] is True
    assert shifted["analysis_only_noncausal_transform"] is True
    assert shifted["realized_shift_s"] == pytest.approx(0.15)


def test_diagnostics_are_hash_bound_to_recording_and_motion_sidecar(tmp_path):
    source = build_recording(seed=47, sim_seconds=0.50, plume_points=16)
    config_document, motion_config = load_odor_motion_config()
    motion = analyze_recording_bundle(
        source,
        config_document=config_document,
        config=motion_config,
    )
    event_config = OdorEventConfig.from_document(config_document)
    diagnostics = build_diagnostics_bundle(
        source,
        motion,
        motion_config=motion_config,
        event_config=event_config,
    )

    payload = diagnostics["diagnostics"]
    assert payload["source_recording_sha256"] == source["recording_sha256"]
    assert payload["source_odor_motion_sha256"] == motion["analysis_sha256"]
    assert payload["controller_access_in_v1"] is False
    assert len(payload["agents"]) == len(source["recording"]["controllers"])
    assert payload["counterfactual_contract"]["right_channel_circular_shift"].startswith(
        "analysis-only"
    )

    path = write_diagnostics(tmp_path / "events.json", diagnostics)
    loaded = load_diagnostics(path)
    assert loaded["diagnostics_sha256"] == diagnostics["diagnostics_sha256"]

    raw = json.loads(path.read_text())
    raw["diagnostics"]["agents"][0]["summary"]["loss_count"] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="diagnostics SHA-256 mismatch"):
        load_diagnostics(path)
