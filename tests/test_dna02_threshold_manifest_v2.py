from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.dna02_threshold_manifest_v2 import freeze_manifest_v2
from fly_sniff.freeze import canonical_sha256

FREEZE_EVIDENCE = Path("authority/program-a-dna02-threshold-freeze-evidence-v2.json")
DISTRIBUTION_EVIDENCE = Path(
    "authority/program-a-dna02-threshold-distribution-freeze-evidence-v2.json"
)
DECISIONS = Path("authority/program-a-dna02-threshold-decisions-v2.json")
RECEIPT = Path("authority/program-a-dna02-threshold-freeze-receipt-v2.json")


def _rehash(payload: dict, field: str) -> dict:
    result = dict(payload)
    result.pop(field, None)
    result[field] = canonical_sha256(result)
    return result


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_frozen_receipt_reproduces_full_manifest(tmp_path: Path) -> None:
    receipt = json.loads(RECEIPT.read_text())
    observed_receipt_hash = receipt.pop("receipt_sha256")
    assert observed_receipt_hash == canonical_sha256(receipt)

    generated = freeze_manifest_v2(
        freeze_evidence_path=FREEZE_EVIDENCE,
        distribution_evidence_path=DISTRIBUTION_EVIDENCE,
        decisions_path=DECISIONS,
        output_path=tmp_path / "manifest.json",
        code_ref=receipt["code_ref"],
    )

    assert generated["manifest_sha256"] == receipt["manifest_sha256"]
    assert generated["status"] == "THRESHOLDS_FROZEN_BEFORE_BEHAVIOR_V2"
    assert generated["threshold_count"] == 8
    assert generated["behavior_fields_reviewed"] is False
    assert generated["yaw_reviewed"] is False
    assert generated["figure3c_statistic_reviewed"] is False
    assert generated["navigation_performance_used"] is False
    assert generated["event_rate_used_for_selection"] is False
    assert generated["pre_behavior_sensitivity_profiles"] == receipt[
        "pre_behavior_sensitivity_profiles"
    ]

    compact_thresholds = [
        {
            "fly_alias": item["fly_alias"],
            "soma_side": item["soma_side"],
            "prominence_quantile": item["selected_prominence_quantile"],
            "threshold": item["selected_threshold"],
        }
        for item in generated["thresholds"]
    ]
    assert compact_thresholds == receipt["thresholds"]


def test_freezer_rejects_event_rate_selection(tmp_path: Path) -> None:
    decisions = json.loads(DECISIONS.read_text())
    decisions["event_rate_used_for_selection"] = True
    tampered = _write(
        tmp_path / "event-rate-decisions.json",
        _rehash(decisions, "decisions_sha256"),
    )
    receipt = json.loads(RECEIPT.read_text())

    with pytest.raises(ValueError, match="event_rate_used_for_selection"):
        freeze_manifest_v2(
            freeze_evidence_path=FREEZE_EVIDENCE,
            distribution_evidence_path=DISTRIBUTION_EVIDENCE,
            decisions_path=tampered,
            output_path=tmp_path / "out.json",
            code_ref=receipt["code_ref"],
        )


def test_freezer_rejects_behavior_contamination(tmp_path: Path) -> None:
    decisions = json.loads(DECISIONS.read_text())
    decisions["yaw_reviewed"] = True
    tampered = _write(
        tmp_path / "yaw-decisions.json",
        _rehash(decisions, "decisions_sha256"),
    )
    receipt = json.loads(RECEIPT.read_text())

    with pytest.raises(ValueError, match="yaw_reviewed"):
        freeze_manifest_v2(
            freeze_evidence_path=FREEZE_EVIDENCE,
            distribution_evidence_path=DISTRIBUTION_EVIDENCE,
            decisions_path=tampered,
            output_path=tmp_path / "out.json",
            code_ref=receipt["code_ref"],
        )


def test_freezer_rejects_unreviewed_primary_quantile(tmp_path: Path) -> None:
    decisions = json.loads(DECISIONS.read_text())
    decisions["decisions"][0]["selected_prominence_quantile"] = 0.99
    decisions["decisions"][0]["selected_threshold"] = 1.4040669693332608
    tampered = _write(
        tmp_path / "unreviewed-q-decisions.json",
        _rehash(decisions, "decisions_sha256"),
    )
    receipt = json.loads(RECEIPT.read_text())

    with pytest.raises(ValueError, match="distribution-reviewed"):
        freeze_manifest_v2(
            freeze_evidence_path=FREEZE_EVIDENCE,
            distribution_evidence_path=DISTRIBUTION_EVIDENCE,
            decisions_path=tampered,
            output_path=tmp_path / "out.json",
            code_ref=receipt["code_ref"],
        )


def test_freezer_rejects_threshold_not_on_audited_candidate(tmp_path: Path) -> None:
    decisions = json.loads(DECISIONS.read_text())
    decisions["decisions"][0]["selected_threshold"] += 0.01
    tampered = _write(
        tmp_path / "off-grid-decisions.json",
        _rehash(decisions, "decisions_sha256"),
    )
    receipt = json.loads(RECEIPT.read_text())

    with pytest.raises(ValueError, match="exact audited candidate"):
        freeze_manifest_v2(
            freeze_evidence_path=FREEZE_EVIDENCE,
            distribution_evidence_path=DISTRIBUTION_EVIDENCE,
            decisions_path=tampered,
            output_path=tmp_path / "out.json",
            code_ref=receipt["code_ref"],
        )


def test_freezer_rejects_distribution_review_rebinding(tmp_path: Path) -> None:
    evidence = json.loads(DISTRIBUTION_EVIDENCE.read_text())
    evidence["parent_distribution_review_sha256"] = "0" * 64
    tampered = _write(
        tmp_path / "distribution-evidence.json",
        _rehash(evidence, "evidence_sha256"),
    )
    receipt = json.loads(RECEIPT.read_text())

    with pytest.raises(ValueError, match="different distribution review"):
        freeze_manifest_v2(
            freeze_evidence_path=FREEZE_EVIDENCE,
            distribution_evidence_path=tampered,
            decisions_path=DECISIONS,
            output_path=tmp_path / "out.json",
            code_ref=receipt["code_ref"],
        )
