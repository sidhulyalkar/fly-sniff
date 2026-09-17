from __future__ import annotations

import json

from fly_sniff.freeze import canonical_sha256


RECEIPT = "authority/program-a-dna02-threshold-high-review-v1.json"


def test_high_threshold_review_receipt_is_content_addressed_and_blocked() -> None:
    with open(RECEIPT, encoding="utf-8") as handle:
        payload = json.load(handle)
    observed = payload.pop("receipt_sha256")
    assert observed == canonical_sha256(payload)
    assert payload["schema"] == "fly-sniff-dna02-threshold-high-review-v1"
    assert payload["status"] == "BLOCKED_PENDING_WAVEFORM_DISTRIBUTION_REVIEW"
    assert payload["thresholds_frozen"] is False
    assert payload["automatic_threshold_selection"] is False
    assert payload["behavior_fields_reviewed"] is False
    assert payload["yaw_reviewed"] is False
    assert payload["figure3c_statistic_reviewed"] is False
    assert payload["navigation_performance_used"] is False
