from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.freeze import canonical_sha256


RECEIPT = Path("authority/program-a-dna02-threshold-first-light-v1.json")


def test_threshold_first_light_receipt_is_content_addressed_and_blocked() -> None:
    payload = json.loads(RECEIPT.read_text())
    observed = payload.pop("receipt_sha256")
    assert observed == canonical_sha256(payload)
    assert payload["schema"] == "fly-sniff-dna02-threshold-first-light-v1"
    assert payload["status"] == "BLOCKED_PENDING_HIGH_THRESHOLD_VISUAL_REVIEW"
    assert payload["prominence_audit_sha256"] == (
        "4c421c803a696c5193b645412f674875c0746509043b835b11b8501d49fc2c9a"
    )
    assert payload["threshold_qc_sha256"] == (
        "da7b8d837c5c8116847a9f58b281516500bf7f7bbb775d5a8f0aea783c864958"
    )
    assert payload["thresholds_frozen"] is False
    assert payload["automatic_threshold_selection"] is False
    assert payload["behavior_fields_reviewed"] is False
    assert payload["yaw_reviewed"] is False
    assert payload["figure3c_statistic_reviewed"] is False
    assert payload["navigation_performance_used"] is False
