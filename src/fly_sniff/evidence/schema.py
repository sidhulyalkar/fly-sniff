from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EvidenceClass(StrEnum):
    """Epistemic classes that must never be silently conflated."""

    MEASURED_STRUCTURE = "MEASURED_STRUCTURE"
    MEASURED_PHYSIOLOGY = "MEASURED_PHYSIOLOGY"
    PREDICTED_ANNOTATION = "PREDICTED_ANNOTATION"
    CROSS_DATASET_PRIOR = "CROSS_DATASET_PRIOR"
    MODEL_ASSUMPTION = "MODEL_ASSUMPTION"
    FITTED_PARAMETER = "FITTED_PARAMETER"
    MODELED_STATE = "MODELED_STATE"
    BEHAVIORAL_OUTPUT = "BEHAVIORAL_OUTPUT"


@dataclass(frozen=True)
class EntityRef:
    dataset: str
    body_id: int | None = None
    type: str | None = None
    name: str | None = None

    def validate(self) -> None:
        if not self.dataset.strip():
            raise ValueError("entity dataset must be non-empty")
        if self.body_id is None and not (self.type or self.name):
            raise ValueError("entity must identify a body_id, type, or name")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EntityRef:
        entity = cls(
            dataset=str(payload["dataset"]),
            body_id=None if payload.get("body_id") is None else int(payload["body_id"]),
            type=None if payload.get("type") is None else str(payload["type"]),
            name=None if payload.get("name") is None else str(payload["name"]),
        )
        entity.validate()
        return entity


@dataclass(frozen=True)
class EvidenceFact:
    statement: str
    evidence_class: EvidenceClass
    authority: str
    confidence: str
    caveat: str | None = None
    authority_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.statement.strip():
            raise ValueError("evidence statement must be non-empty")
        if not self.authority.strip():
            raise ValueError("evidence authority must be non-empty")
        if not self.confidence.strip():
            raise ValueError("evidence confidence must be non-empty")
        if self.authority_sha256 is not None:
            value = self.authority_sha256.lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError("authority_sha256 must be a 64-character hex digest")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidenceFact:
        fact = cls(
            statement=str(payload["statement"]),
            evidence_class=EvidenceClass(str(payload["evidence_class"])),
            authority=str(payload["authority"]),
            confidence=str(payload["confidence"]),
            caveat=None if payload.get("caveat") is None else str(payload["caveat"]),
            authority_sha256=(
                None
                if payload.get("authority_sha256") is None
                else str(payload["authority_sha256"])
            ),
            metadata=dict(payload.get("metadata", {})),
        )
        fact.validate()
        return fact


@dataclass(frozen=True)
class EvidenceRecord:
    entity: EntityRef
    facts: tuple[EvidenceFact, ...]
    forbidden_claims: tuple[str, ...] = ()
    allowed_claims: tuple[str, ...] = ()

    def validate(self) -> None:
        self.entity.validate()
        if not self.facts:
            raise ValueError("evidence record must contain at least one fact")
        for fact in self.facts:
            fact.validate()
        if any(not text.strip() for text in self.forbidden_claims):
            raise ValueError("forbidden claims must be non-empty strings")
        if any(not text.strip() for text in self.allowed_claims):
            raise ValueError("allowed claims must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for fact in payload["facts"]:
            fact["evidence_class"] = str(fact["evidence_class"])
        payload["facts"] = list(payload["facts"])
        payload["forbidden_claims"] = list(payload["forbidden_claims"])
        payload["allowed_claims"] = list(payload["allowed_claims"])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidenceRecord:
        record = cls(
            entity=EntityRef.from_dict(dict(payload["entity"])),
            facts=tuple(EvidenceFact.from_dict(dict(row)) for row in payload["facts"]),
            forbidden_claims=tuple(str(x) for x in payload.get("forbidden_claims", [])),
            allowed_claims=tuple(str(x) for x in payload.get("allowed_claims", [])),
        )
        record.validate()
        return record
