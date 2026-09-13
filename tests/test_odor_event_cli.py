from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from fly_sniff.odor_event_cli import verify_motion_config_binding
from fly_sniff.odor_motion import load_odor_motion_config


def test_event_cli_accepts_json_normalized_embedded_estimator_config():
    _, config = load_odor_motion_config()
    embedded = asdict(config)
    embedded["delays_s"] = list(embedded["delays_s"])

    verify_motion_config_binding({"analysis": {"estimator": embedded}}, config)


def test_event_cli_rejects_counterfactual_estimator_config_drift():
    _, config = load_odor_motion_config()
    changed = replace(config, signal_scale=config.signal_scale * 1.5).validated()

    with pytest.raises(ValueError, match="does not match the estimator settings"):
        verify_motion_config_binding(
            {"analysis": {"estimator": asdict(config)}},
            changed,
        )
