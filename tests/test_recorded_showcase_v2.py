from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from fly_sniff.recorded_showcase_v2 import (
    _antenna_strengths,
    _assert_canvas_dimensions,
    _recording_has_named_neural_state,
    _validate_evidence_config,
)
from fly_sniff.showcase import SHOWCASE_DPI, SHOWCASE_HEIGHT, SHOWCASE_WIDTH


def _config() -> dict:
    return {
        "protocol": "who-farted-showcase-evidence-v2",
        "measured_structure": {
            "allowed": True,
            "label": "MEASURED CONNECTIVITY",
        },
        "modeled_activity": {"label": "MODELED ACTIVITY"},
        "comparison": {"real_vs_rewire_headline_allowed": False},
        "behavioral_state": {"malecns_behavior_claim_allowed": False},
    }


def test_evidence_config_accepts_development_contract() -> None:
    _validate_evidence_config(_config())


def test_evidence_config_rejects_premature_real_vs_rewire_claim() -> None:
    config = _config()
    config["comparison"]["real_vs_rewire_headline_allowed"] = True
    with pytest.raises(ValueError, match="real-vs-rewire"):
        _validate_evidence_config(config)


def test_evidence_config_rejects_premature_malecns_behavior_claim() -> None:
    config = _config()
    config["behavioral_state"]["malecns_behavior_claim_allowed"] = True
    with pytest.raises(ValueError, match="MaleCNS behavioral claim"):
        _validate_evidence_config(config)


def test_evidence_config_requires_distinct_evidence_labels() -> None:
    config = _config()
    config["measured_structure"]["label"] = "MODELED ACTIVITY"
    with pytest.raises(ValueError, match="MEASURED CONNECTIVITY"):
        _validate_evidence_config(config)


def test_antenna_strengths_use_only_recorded_odor_and_clip() -> None:
    state = {
        "observation": {
            "left_odor": 1.3,
            "right_odor": -0.4,
            "source_x": 99.0,
        }
    }
    assert _antenna_strengths(state) == (1.0, 0.0)


def test_recorded_neural_state_gate_defaults_closed() -> None:
    payload = {
        "frames": [
            {
                "agents": [
                    {"label": "proxy", "diagnostics": {"mode_surge": 1.0}},
                ]
            }
        ]
    }
    assert not _recording_has_named_neural_state(payload)

    payload["frames"][0]["agents"][0]["neural_state"] = {"PFL3": [0.1, 0.2]}
    assert _recording_has_named_neural_state(payload)


def test_showcase_canvas_is_exactly_1080_by_1350() -> None:
    fig = plt.figure(
        figsize=(SHOWCASE_WIDTH / SHOWCASE_DPI, SHOWCASE_HEIGHT / SHOWCASE_DPI),
        dpi=SHOWCASE_DPI,
    )
    try:
        _assert_canvas_dimensions(fig)
    finally:
        plt.close(fig)


def test_showcase_canvas_guard_rejects_dimension_drift() -> None:
    fig = plt.figure(figsize=(4.0, 4.0), dpi=100)
    try:
        with pytest.raises(RuntimeError, match="canvas dimensions drifted"):
            _assert_canvas_dimensions(fig)
    finally:
        plt.close(fig)
