from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .evidence import EvidenceClass, EvidenceLedger


class ClaimKind(str, Enum):
    """Narrow claim types that can be supported directly by typed evidence."""

    STRUCTURAL = "structural"
    MEASURED_PHYSIOLOGY = "measured_physiology"
    PREDICTED_ANNOTATION = "predicted_annotation"
    MODELED_STATE = "modeled_state"
    BEHAVIORAL_OUTPUT = "behavioral_output"


_REQUIRED_EVIDENCE: dict[ClaimKind, EvidenceClass] = {
    ClaimKind.STRUCTURAL: EvidenceClass.MEASURED_STRUCTURE,
    ClaimKind.MEASURED_PHYSIOLOGY: EvidenceClass.MEASURED_PHYSIOLOGY,
    ClaimKind.PREDICTED_ANNOTATION: EvidenceClass.PREDICTED_ANNOTATION,
    ClaimKind.MODELED_STATE: EvidenceClass.MODELED_STATE,
    ClaimKind.BEHAVIORAL_OUTPUT: EvidenceClass.BEHAVIORAL_OUTPUT,
}


@dataclass(frozen=True)
class ClaimSupport:
    kind: ClaimKind
    subjects: tuple[str, ...]
    supported: bool
    supporting_record_ids: tuple[str, ...]
    reason: str


def support_claim(
    ledger: EvidenceLedger,
    *,
    kind: ClaimKind,
    subjects: tuple[str, ...],
    predicate: str | None = None,
) -> ClaimSupport:
    """Check whether every requested subject has direct evidence of the required class.

    This intentionally does not infer a stronger functional claim from a structural
    path, or measured physiology from modeled activity. Higher-order claims such as
    topology dependence require an experiment receipt and belong in a later layer.
    """

    ledger.validate()
    if not subjects:
        raise ValueError("claim subjects must be non-empty")

    required_class = _REQUIRED_EVIDENCE[kind]
    supporting_ids: list[str] = []
    missing: list[str] = []
    for subject in subjects:
        matches = ledger.select(
            subject=subject,
            predicate=predicate,
            evidence_class=required_class,
        )
        if matches:
            supporting_ids.extend(record.record_id for record in matches)
        else:
            missing.append(subject)

    if missing:
        return ClaimSupport(
            kind=kind,
            subjects=subjects,
            supported=False,
            supporting_record_ids=tuple(sorted(set(supporting_ids))),
            reason=(
                f"missing {required_class.value} evidence for subjects: "
                + ", ".join(sorted(missing))
            ),
        )

    return ClaimSupport(
        kind=kind,
        subjects=subjects,
        supported=True,
        supporting_record_ids=tuple(sorted(set(supporting_ids))),
        reason=f"all subjects have direct {required_class.value} evidence",
    )
