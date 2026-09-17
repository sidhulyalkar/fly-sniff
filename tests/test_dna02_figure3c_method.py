from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.dna02_figure3c_method import load_method_contract, method_status


CONTRACT = Path("authority/program-a-dna02-figure3c-method-contract-v1.json")
HASH = Path("authority/program-a-dna02-figure3c-method-contract-v1.sha256")


def test_method_contract_is_frozen_and_blocks_numeric_extraction() -> None:
    payload = load_method_contract(CONTRACT, HASH)
    assert payload["status"] == "PARAMETERS_RESOLVED_EDGE_SEMANTICS_PENDING"
    assert payload["resolved_smoothing_parameters"]["period_bins"] == 3
    assert payload["resolved_smoothing_parameters"]["alpha"] == 0.5
    assert payload["gates"]["numeric_smoothing_implementation_allowed"] is False
    assert payload["gates"]["figure3c_behavior_allowed"] is False
    assert payload["gates"]["thresholds_must_be_frozen_first"] is True
    assert payload["gates"]["navigation_performance_used"] is False


def test_method_status_preserves_unresolved_conformance_gate() -> None:
    status = method_status(CONTRACT, HASH)
    assert status["status"] == "PARAMETERS_RESOLVED_EDGE_SEMANTICS_PENDING"
    assert len(status["unresolved_before_numeric_extraction"]) >= 4
    assert status["gates"]["numeric_smoothing_implementation_allowed"] is False


def test_contract_hash_tamper_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["resolved_smoothing_parameters"]["alpha"] = 0.6
    tampered = tmp_path / "contract.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical hash"):
        load_method_contract(tampered, HASH)
