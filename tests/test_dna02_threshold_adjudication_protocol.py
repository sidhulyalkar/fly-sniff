from __future__ import annotations

import json
from pathlib import Path


PROTOCOL = Path("authority/program-a-dna02-threshold-adjudication-protocol-v1.json")


def test_threshold_adjudication_protocol_is_behavior_blind() -> None:
    payload = json.loads(PROTOCOL.read_text())
    assert payload["schema"] == "fly-sniff-dna02-threshold-adjudication-protocol-v1"
    assert payload["automatic_threshold_selection"] is False
    assert payload["navigation_performance_used"] is False
    assert payload["behavior_review_allowed_before_freeze"] is False
    assert payload["figure3c_statistic_review_allowed_before_freeze"] is False
    assert payload["required_decisions"] == 8
    assert payload["threshold_must_match_audited_candidate"] is True
    assert payload["input_fields_allowlist"] == ["ephys_SR", "ephys_A", "ephys_B"]
    assert "event_rate" not in payload["selection_basis_allowlist"]
    assert {"yaw", "fwd", "lat"}.issubset(set(payload["forbidden_inputs"]))
