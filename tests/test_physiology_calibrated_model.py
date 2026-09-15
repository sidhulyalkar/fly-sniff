from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.physiology_calibrated_model import build_calibrated_model_artifact


CONFIG = Path("configs/physiology_calibration_v1.json")


def _fit_report(config: dict) -> dict:
    return {
        "protocol": "physiology-fit-report-v1",
        "calibration_protocol": config["protocol"],
        "fit_status": "accepted_under_preregistered_objectives",
        "navigation_objective_weight": 0.0,
        "topology_variant_visible_during_fit": False,
        "final_evaluation_visible_during_fit": False,
        "parameters": {name: 1.0 for name in config["permitted_parameters"]},
        "probe_results": {
            row["id"]: {"objective_defined_before_fit": True, "passed": True}
            for row in config["probes"]
        },
    }


def test_unresolved_binding_prevents_calibrated_model_seal() -> None:
    config = json.loads(CONFIG.read_text())
    bindings = {
        "protocol": "physiology-probe-binding-v1",
        "ready_to_define_fit_objective": False,
        "unresolved_probes": ["temporal_response_scale"],
    }
    with pytest.raises(ValueError, match="unresolved probe authorities"):
        build_calibrated_model_artifact(
            config,
            bindings,
            _fit_report(config),
            calibration_config_sha256="11" * 32,
            binding_report_sha256="22" * 32,
            fit_report_sha256="33" * 32,
        )


def test_complete_authorities_and_fit_report_can_be_sealed() -> None:
    config = json.loads(CONFIG.read_text())
    bindings = {
        "protocol": "physiology-probe-binding-v1",
        "ready_to_define_fit_objective": True,
        "unresolved_probes": [],
    }
    artifact = build_calibrated_model_artifact(
        config,
        bindings,
        _fit_report(config),
        calibration_config_sha256="11" * 32,
        binding_report_sha256="22" * 32,
        fit_report_sha256="33" * 32,
    )
    assert artifact["protocol"] == "physiology-calibrated-model-v1"
    assert set(artifact["parameters"]) == set(config["permitted_parameters"])
    assert set(artifact["probe_results"]) == {row["id"] for row in config["probes"]}


def test_navigation_signal_in_fit_report_is_rejected() -> None:
    config = json.loads(CONFIG.read_text())
    bindings = {
        "protocol": "physiology-probe-binding-v1",
        "ready_to_define_fit_objective": True,
        "unresolved_probes": [],
    }
    fit = _fit_report(config)
    fit["navigation_objective_weight"] = 0.1
    with pytest.raises(ValueError, match="exactly zero"):
        build_calibrated_model_artifact(
            config,
            bindings,
            fit,
            calibration_config_sha256="11" * 32,
            binding_report_sha256="22" * 32,
            fit_report_sha256="33" * 32,
        )
