from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.evidence import EvidenceClass, EvidenceLedger

AUTHORITY_PATH = Path("authority/program-a-physiology-evidence-v1.json")


def _ledger() -> EvidenceLedger:
    payload = json.loads(AUTHORITY_PATH.read_text())
    return EvidenceLedger.from_dict(payload)


def test_program_a_physiology_ledger_is_hash_valid() -> None:
    payload = json.loads(AUTHORITY_PATH.read_text())
    ledger = EvidenceLedger.from_dict(payload)
    assert ledger.sha256 == payload["ledger_sha256"]
    assert len(ledger.records) == 9


def test_direct_physiology_records_remain_external_preparations() -> None:
    ledger = _ledger()
    measured = ledger.select(evidence_class=EvidenceClass.MEASURED_PHYSIOLOGY)
    assert measured
    assert all(record.subject.startswith("external:") for record in measured)
    assert not any(record.subject.startswith("MaleCNS:") for record in measured)


def test_malecns_pfn_transfer_is_explicit_cross_dataset_prior() -> None:
    ledger = _ledger()
    records = ledger.select(subject="MaleCNS:PFNa_PFNm_PFNp_candidates")
    assert len(records) == 1
    record = records[0]
    assert record.evidence_class is EvidenceClass.CROSS_DATASET_PRIOR
    assert "not exact MaleCNS body-ID physiology" in str(record.value)


def test_pfn_laterality_authority_is_cell_body_hemisphere() -> None:
    ledger = _ledger()
    records = ledger.select(predicate="airflow_tuning_geometry")
    assert len(records) == 1
    assert records[0].value["laterality_authority"] == "cell-body hemisphere"


def test_pfl3_laterality_does_not_promote_soma_side() -> None:
    ledger = _ledger()
    records = ledger.select(predicate="lateralized_steering_relationship")
    assert len(records) == 1
    assert "Do not infer PFL3 steering side" in (records[0].notes or "")


def test_dna02_delay_record_preserves_treadmill_caveat() -> None:
    ledger = _ledger()
    records = ledger.select(predicate="neural_to_turn_lag")
    assert len(records) == 1
    value = records[0].value
    assert value["reported_average_lag_ms"] == 150
    assert "treadmill inertia" in value["interpretation_caveat"]
    assert "Do not freeze 150 ms" in (records[0].notes or "")


def test_hdelta_confounded_record_cannot_masquerade_as_measured_physiology() -> None:
    ledger = _ledger()
    records = ledger.select(predicate="functional_attribution_caveat")
    assert len(records) == 1
    assert records[0].evidence_class is EvidenceClass.CROSS_DATASET_PRIOR
    assert "Blocks hDeltaC-specific quantitative physiology calibration" in (
        records[0].notes or ""
    )
