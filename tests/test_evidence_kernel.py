from __future__ import annotations

import copy

import pytest

from fly_sniff.claims import ClaimKind, support_claim
from fly_sniff.evidence import EvidenceClass, EvidenceLedger, EvidenceRecord


def _record(
    record_id: str,
    subject: str,
    evidence_class: EvidenceClass,
    *,
    predicate: str = "has_property",
    provenance_ref: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        subject=subject,
        predicate=predicate,
        value=True,
        evidence_class=evidence_class,
        authority="test-authority",
        dataset="synthetic:test",
        provenance_ref=provenance_ref,
    )


def test_ledger_hash_is_independent_of_record_order() -> None:
    first = _record("a", "body:1", EvidenceClass.MEASURED_STRUCTURE)
    second = _record("b", "body:2", EvidenceClass.PREDICTED_ANNOTATION)
    left = EvidenceLedger.build([first, second])
    right = EvidenceLedger.build([second, first])
    assert left.sha256 == right.sha256
    assert left.to_dict() == right.to_dict()


def test_ledger_rejects_duplicate_record_ids() -> None:
    record = _record("same", "body:1", EvidenceClass.MEASURED_STRUCTURE)
    with pytest.raises(ValueError, match="duplicate evidence record_id"):
        EvidenceLedger.build([record, record])


def test_modeled_or_fitted_evidence_requires_run_provenance() -> None:
    with pytest.raises(ValueError, match="requires provenance_ref"):
        EvidenceLedger.build([_record("m", "body:1", EvidenceClass.MODELED_STATE)])

    ledger = EvidenceLedger.build(
        [
            _record(
                "m",
                "body:1",
                EvidenceClass.MODELED_STATE,
                provenance_ref="run:abc",
            )
        ]
    )
    assert ledger.records[0].provenance_ref == "run:abc"


def test_ledger_detects_tampering_after_seal() -> None:
    ledger = EvidenceLedger.build(
        [_record("a", "body:1", EvidenceClass.MEASURED_STRUCTURE)]
    )
    payload = copy.deepcopy(ledger.to_dict())
    payload["records"][0]["value"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        EvidenceLedger.from_dict(payload)


def test_structural_evidence_cannot_support_measured_physiology_claim() -> None:
    ledger = EvidenceLedger.build(
        [_record("edge", "edge:1->2", EvidenceClass.MEASURED_STRUCTURE)]
    )
    structural = support_claim(
        ledger,
        kind=ClaimKind.STRUCTURAL,
        subjects=("edge:1->2",),
    )
    physiology = support_claim(
        ledger,
        kind=ClaimKind.MEASURED_PHYSIOLOGY,
        subjects=("edge:1->2",),
    )
    assert structural.supported is True
    assert physiology.supported is False
    assert "measured_physiology" in physiology.reason


def test_claim_support_requires_direct_evidence_for_every_subject() -> None:
    ledger = EvidenceLedger.build(
        [
            _record("a", "body:1", EvidenceClass.MEASURED_STRUCTURE),
            _record("b", "body:2", EvidenceClass.PREDICTED_ANNOTATION),
        ]
    )
    result = support_claim(
        ledger,
        kind=ClaimKind.STRUCTURAL,
        subjects=("body:1", "body:2"),
    )
    assert result.supported is False
    assert "body:2" in result.reason
