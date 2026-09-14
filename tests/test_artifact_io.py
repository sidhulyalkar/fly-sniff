from __future__ import annotations

import json

import pytest

from fly_sniff.artifact_io import file_sha256, load_typed_artifact, write_typed_artifact
from fly_sniff.evidence import EvidenceClass, EvidenceLedger, EvidenceRecord


def _ledger() -> EvidenceLedger:
    return EvidenceLedger.build(
        [
            EvidenceRecord(
                record_id="edge-1",
                subject="edge:1->2",
                predicate="exists",
                value=True,
                evidence_class=EvidenceClass.MEASURED_STRUCTURE,
                authority="synthetic-test",
                dataset="synthetic:v1",
            )
        ]
    )


def test_typed_artifact_roundtrip_and_exact_file_hash(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    ledger = _ledger()
    write_typed_artifact(path, ledger)

    restored = load_typed_artifact(path)
    assert isinstance(restored, EvidenceLedger)
    assert restored.sha256 == ledger.sha256
    assert len(file_sha256(path)) == 64


def test_typed_artifact_refuses_overwrite_by_default(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    write_typed_artifact(path, _ledger())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_typed_artifact(path, _ledger())


def test_typed_artifact_detects_scientific_tampering(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    write_typed_artifact(path, _ledger())
    payload = json.loads(path.read_text())
    payload["records"][0]["value"] = False
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hash mismatch"):
        load_typed_artifact(path)


def test_typed_artifact_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text('{"schema":"not-a-fly-sniff-artifact"}')
    with pytest.raises(ValueError, match="unsupported kernel artifact schema"):
        load_typed_artifact(path)
