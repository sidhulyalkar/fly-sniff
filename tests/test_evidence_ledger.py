from __future__ import annotations

from fly_sniff.evidence import (
    ClaimStatus,
    EvidenceClass,
    EvidenceFact,
    EvidenceLedger,
    EvidenceRecord,
    EntityRef,
    compile_claim,
)


def _ledger() -> EvidenceLedger:
    return EvidenceLedger(
        records=(
            EvidenceRecord(
                entity=EntityRef(dataset="male-cns:v1.0", body_id=14975, type="hDeltaC"),
                facts=(
                    EvidenceFact(
                        statement="body-ID-resolved structural candidate",
                        evidence_class=EvidenceClass.MEASURED_STRUCTURE,
                        authority="MaleCNS v1.0",
                        confidence="exact",
                    ),
                    EvidenceFact(
                        statement="rate state is produced by an explicit model",
                        evidence_class=EvidenceClass.MODEL_ASSUMPTION,
                        authority="fly-sniff rate model",
                        confidence="explicit",
                    ),
                    EvidenceFact(
                        statement="modeled state exists for the run",
                        evidence_class=EvidenceClass.MODELED_STATE,
                        authority="run:test",
                        confidence="deterministic",
                    ),
                ),
                forbidden_claims=("measured odor-gated neural activity",),
            ),
        )
    )


def test_evidence_ledger_roundtrip_hash() -> None:
    ledger = _ledger()
    payload = ledger.to_dict()
    loaded = EvidenceLedger.from_dict(payload)
    assert loaded.to_dict()["ledger_sha256"] == payload["ledger_sha256"]
    assert loaded.records[0].entity.body_id == 14975


def test_tampered_evidence_ledger_hash_is_rejected() -> None:
    payload = _ledger().to_dict()
    payload["records"][0]["facts"][0]["statement"] = "tampered"
    try:
        EvidenceLedger.from_dict(payload)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered evidence ledger was accepted")


def test_structural_claim_requires_measured_structure() -> None:
    decision = compile_claim(
        ClaimStatus.STRUCTURAL_CANDIDATE,
        subject="hDeltaC",
        evidence_classes={EvidenceClass.MEASURED_STRUCTURE},
    )
    assert decision.supported is True
    assert "structural candidate" in decision.allowed_wording


def test_topology_claim_requires_matched_null_confirmation() -> None:
    classes = {
        EvidenceClass.MEASURED_STRUCTURE,
        EvidenceClass.MODEL_ASSUMPTION,
        EvidenceClass.MODELED_STATE,
        EvidenceClass.BEHAVIORAL_OUTPUT,
    }
    blocked = compile_claim(
        ClaimStatus.TOPOLOGY_DEPENDENCE_SUPPORTED,
        subject="odor navigation",
        evidence_classes=classes,
    )
    assert blocked.supported is False
    assert any("topology-null" in reason for reason in blocked.rationale)

    allowed = compile_claim(
        ClaimStatus.TOPOLOGY_DEPENDENCE_SUPPORTED,
        subject="odor navigation",
        evidence_classes=classes,
        topology_null_confirmed=True,
    )
    assert allowed.supported is True
