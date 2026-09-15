from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import EvidenceClass, EvidenceRecord

LEDGER_PROTOCOL = "fly-sniff-evidence-ledger-v1"


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _entity_key(record: EvidenceRecord) -> tuple[str, int | None, str | None, str | None]:
    entity = record.entity
    return (entity.dataset, entity.body_id, entity.type, entity.name)


@dataclass(frozen=True)
class EvidenceLedger:
    records: tuple[EvidenceRecord, ...]
    protocol: str = LEDGER_PROTOCOL

    def validate(self) -> None:
        if self.protocol != LEDGER_PROTOCOL:
            raise ValueError(f"unexpected evidence ledger protocol: {self.protocol!r}")
        if not self.records:
            raise ValueError("evidence ledger must contain at least one record")
        seen: set[tuple[str, int | None, str | None, str | None]] = set()
        for record in self.records:
            record.validate()
            key = _entity_key(record)
            if key in seen:
                raise ValueError(f"duplicate evidence entity: {key}")
            seen.add(key)

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol": self.protocol,
            "records": [record.to_dict() for record in self.records],
        }
        if include_sha256:
            payload["ledger_sha256"] = canonical_sha256(payload)
        return payload

    def evidence_classes(self) -> set[EvidenceClass]:
        return {
            fact.evidence_class
            for record in self.records
            for fact in record.facts
        }

    def records_for_body_id(self, body_id: int) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if record.entity.body_id == int(body_id))

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, verify_sha256: bool = True) -> EvidenceLedger:
        expected = payload.get("ledger_sha256")
        bare = {key: value for key, value in payload.items() if key != "ledger_sha256"}
        if verify_sha256 and expected is not None:
            actual = canonical_sha256(bare)
            if str(expected) != actual:
                raise ValueError(f"evidence ledger SHA-256 mismatch: expected {expected}, got {actual}")
        ledger = cls(
            records=tuple(EvidenceRecord.from_dict(dict(row)) for row in bare["records"]),
            protocol=str(bare.get("protocol", "")),
        )
        ledger.validate()
        return ledger

    @classmethod
    def load(cls, path: str | Path, *, verify_sha256: bool = True) -> EvidenceLedger:
        return cls.from_dict(json.loads(Path(path).read_text()), verify_sha256=verify_sha256)

    def save(self, path: str | Path) -> Path:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fly-sniff EvidenceLedger v1 artifact")
    parser.add_argument("ledger")
    args = parser.parse_args()
    ledger = EvidenceLedger.load(args.ledger)
    classes = sorted(str(value) for value in ledger.evidence_classes())
    print(f"valid=True records={len(ledger.records)} classes={','.join(classes)}")
    print(f"sha256={ledger.to_dict()['ledger_sha256']}")


if __name__ == "__main__":
    main()
