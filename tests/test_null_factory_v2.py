from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.null_factory_v2 import validate_protocol

CONFIG = Path("configs/null_factory_v2.json")


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def test_null_factory_v2_preregistration_is_valid() -> None:
    report = validate_protocol(_config())
    assert report["valid_for_preregistration"] is True
    assert report["passed_gate_count"] == report["gate_count"]
    assert "degree_preserving" in report["ready_families"]


def test_null_floor_cannot_drop_below_63() -> None:
    config = _config()
    config["minimum_confirmatory_topologies_per_family"] = 8
    report = validate_protocol(config)
    assert report["valid_for_preregistration"] is False


def test_performance_based_seed_selection_is_forbidden() -> None:
    config = _config()
    config["seed_contract"]["performance_based_seed_selection_allowed"] = True
    report = validate_protocol(config)
    assert report["valid_for_preregistration"] is False


def test_type_constrained_family_requires_cell_type_metadata() -> None:
    config = _config()
    family = next(row for row in config["families"] if row["name"] == "within_cell_type_rewire")
    family["required_node_metadata"] = []
    report = validate_protocol(config)
    assert report["valid_for_preregistration"] is False
