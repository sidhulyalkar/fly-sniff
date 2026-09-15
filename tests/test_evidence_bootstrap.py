from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.evidence import EvidenceClass
from fly_sniff.evidence.bootstrap import build_ledger

CONFIG = Path("configs/evidence_bootstrap_v1.json")


def test_real_bootstrap_config_builds_hash_bound_ledger() -> None:
    ledger = build_ledger(json.loads(CONFIG.read_text()))
    payload = ledger.to_dict()
    assert len(ledger.records) == 6
    assert len(payload["ledger_sha256"]) == 64
    classes = ledger.evidence_classes()
    assert EvidenceClass.MEASURED_STRUCTURE in classes
    assert EvidenceClass.PREDICTED_ANNOTATION in classes
    assert EvidenceClass.MODELED_STATE in classes
    assert EvidenceClass.MODEL_ASSUMPTION in classes
    for record in ledger.records:
        for fact in record.facts:
            assert fact.authority_sha256 is not None
            assert len(fact.authority_sha256) == 64


def test_bootstrap_rejects_authority_dataset_mismatch(tmp_path: Path) -> None:
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps({"dataset": "wrong:v1"}) + "\n")
    config = {
        "protocol": "evidence-bootstrap-v1",
        "dataset": "male-cns:v1.0",
        "records": [
            {
                "entity": {"dataset": "male-cns:v1.0", "name": "test"},
                "authority_path": authority.name,
                "facts": [
                    {
                        "statement": "test fact",
                        "evidence_class": "MEASURED_STRUCTURE",
                        "confidence": "exact",
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="authority dataset mismatch"):
        build_ledger(config, root=tmp_path)
