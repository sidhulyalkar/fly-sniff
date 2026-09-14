from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.physiology_calibration import validate_protocol

CONFIG = Path("configs/physiology_calibration_v1.json")


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def test_calibration_protocol_is_valid_but_blocked_on_unresolved_authorities() -> None:
    report = validate_protocol(_config())
    assert report["valid_for_preregistration"] is True
    assert report["ready_to_fit"] is False
    assert len(report["unresolved_probe_authorities"]) == 6


def test_navigation_objective_must_remain_zero() -> None:
    config = _config()
    config["fit_contract"]["navigation_objective_weight"] = 0.01
    report = validate_protocol(config)
    assert report["valid_for_preregistration"] is False


def test_individual_edge_weight_fitting_must_remain_forbidden() -> None:
    config = _config()
    config["forbidden_parameters"].remove("individual_edge_weights")
    report = validate_protocol(config)
    assert report["valid_for_preregistration"] is False


def test_sealed_probe_authorities_unlock_ready_to_fit() -> None:
    config = _config()
    for probe in config["probes"]:
        probe["authority_status"] = "sealed"
    report = validate_protocol(config)
    assert report["valid_for_preregistration"] is True
    assert report["ready_to_fit"] is True
