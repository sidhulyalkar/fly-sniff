from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.connectome_necessity import validate_protocol


CONFIG = Path("configs/connectome_necessity_v1.json")


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def test_connectome_necessity_v1_preregistration_is_valid() -> None:
    report = validate_protocol(_config())
    assert report["valid_for_preregistration"] is True
    assert report["passed_gate_count"] == report["gate_count"]


def test_navigation_reward_cannot_be_enabled() -> None:
    config = _config()
    config["training_contract"]["navigation_reward_allowed"] = True
    report = validate_protocol(config)
    assert report["valid_for_preregistration"] is False
    gate = next(row for row in report["gates"] if row["name"] == "navigation_training_forbidden")
    assert gate["passed"] is False


def test_confirmatory_null_floor_cannot_drop_below_63() -> None:
    config = _config()
    config["null_hierarchy"][0]["minimum_topologies"] = 8
    report = validate_protocol(config)
    assert report["valid_for_preregistration"] is False
    gate = next(
        row for row in report["gates"] if row["name"] == "confirmatory_null_count_at_least_63"
    )
    assert gate["passed"] is False


def test_topology_level_inference_is_mandatory() -> None:
    config = _config()
    config["statistics"]["topology_is_unit_of_randomization"] = False
    report = validate_protocol(config)
    assert report["valid_for_preregistration"] is False
    gate = next(row for row in report["gates"] if row["name"] == "topology_level_inference_required")
    assert gate["passed"] is False
