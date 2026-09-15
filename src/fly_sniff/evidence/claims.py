from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .schema import EvidenceClass


class ClaimStatus(StrEnum):
    STRUCTURAL_CANDIDATE = "STRUCTURAL_CANDIDATE"
    FUNCTIONAL_MODEL_SUPPORTED = "FUNCTIONAL_MODEL_SUPPORTED"
    TOPOLOGY_DEPENDENCE_SUPPORTED = "TOPOLOGY_DEPENDENCE_SUPPORTED"
    CAUSAL_PATHWAY_DEPENDENCE_SUPPORTED = "CAUSAL_PATHWAY_DEPENDENCE_SUPPORTED"
    BIOLOGICAL_GENERALIZATION_SUPPORTED = "BIOLOGICAL_GENERALIZATION_SUPPORTED"


@dataclass(frozen=True)
class ClaimDecision:
    status: ClaimStatus
    supported: bool
    allowed_wording: str | None
    forbidden_wording: tuple[str, ...]
    rationale: tuple[str, ...]


_REQUIREMENTS: dict[ClaimStatus, frozenset[EvidenceClass]] = {
    ClaimStatus.STRUCTURAL_CANDIDATE: frozenset({EvidenceClass.MEASURED_STRUCTURE}),
    ClaimStatus.FUNCTIONAL_MODEL_SUPPORTED: frozenset(
        {
            EvidenceClass.MEASURED_STRUCTURE,
            EvidenceClass.MODEL_ASSUMPTION,
            EvidenceClass.MODELED_STATE,
        }
    ),
    ClaimStatus.TOPOLOGY_DEPENDENCE_SUPPORTED: frozenset(
        {
            EvidenceClass.MEASURED_STRUCTURE,
            EvidenceClass.MODEL_ASSUMPTION,
            EvidenceClass.MODELED_STATE,
            EvidenceClass.BEHAVIORAL_OUTPUT,
        }
    ),
    ClaimStatus.CAUSAL_PATHWAY_DEPENDENCE_SUPPORTED: frozenset(
        {
            EvidenceClass.MEASURED_STRUCTURE,
            EvidenceClass.MODELED_STATE,
            EvidenceClass.BEHAVIORAL_OUTPUT,
        }
    ),
    ClaimStatus.BIOLOGICAL_GENERALIZATION_SUPPORTED: frozenset(
        {
            EvidenceClass.MEASURED_STRUCTURE,
            EvidenceClass.MEASURED_PHYSIOLOGY,
            EvidenceClass.BEHAVIORAL_OUTPUT,
        }
    ),
}


def compile_claim(
    status: ClaimStatus,
    *,
    subject: str,
    evidence_classes: Iterable[EvidenceClass],
    topology_null_confirmed: bool = False,
    acute_intervention_confirmed: bool = False,
    biological_replication_confirmed: bool = False,
) -> ClaimDecision:
    """Compile conservative claim wording from explicit evidence and run-level sentries.

    This intentionally refuses to infer topology, causal, or biological-generalization
    support from evidence classes alone. Those statuses require dedicated run-level
    confirmations supplied by the caller.
    """
    observed = set(evidence_classes)
    missing = sorted(str(value) for value in _REQUIREMENTS[status] - observed)
    sentry_ok = True
    sentry_reason: str | None = None
    if status is ClaimStatus.TOPOLOGY_DEPENDENCE_SUPPORTED and not topology_null_confirmed:
        sentry_ok = False
        sentry_reason = "matched topology-null inference has not been confirmed"
    elif status is ClaimStatus.CAUSAL_PATHWAY_DEPENDENCE_SUPPORTED and not acute_intervention_confirmed:
        sentry_ok = False
        sentry_reason = "acute intervention dependence has not been confirmed"
    elif status is ClaimStatus.BIOLOGICAL_GENERALIZATION_SUPPORTED and not biological_replication_confirmed:
        sentry_ok = False
        sentry_reason = "independent biological replication/generalization has not been confirmed"

    supported = not missing and sentry_ok
    rationale: list[str] = []
    if missing:
        rationale.append(f"missing evidence classes: {', '.join(missing)}")
    if sentry_reason:
        rationale.append(sentry_reason)
    if supported:
        rationale.append("all required evidence classes and run-level sentries are satisfied")

    wording: dict[ClaimStatus, str] = {
        ClaimStatus.STRUCTURAL_CANDIDATE: (
            f"{subject} is supported as a body-ID-resolved structural candidate in the measured connectome."
        ),
        ClaimStatus.FUNCTIONAL_MODEL_SUPPORTED: (
            f"Under the stated model assumptions, {subject} supports the modeled computation."
        ),
        ClaimStatus.TOPOLOGY_DEPENDENCE_SUPPORTED: (
            f"Under frozen dynamics and matched controls, {subject} depends on measured topology beyond the tested null ensemble."
        ),
        ClaimStatus.CAUSAL_PATHWAY_DEPENDENCE_SUPPORTED: (
            f"Under the frozen model, acute intervention supports causal dependence on {subject}."
        ),
        ClaimStatus.BIOLOGICAL_GENERALIZATION_SUPPORTED: (
            f"The {subject} result is supported by measured structure, physiology, behavioral output, and independent biological replication."
        ),
    }
    forbidden = (
        "measured neural activity",
        "the connectome learned to smell",
        "the fly brain found the source",
    )
    return ClaimDecision(
        status=status,
        supported=supported,
        allowed_wording=wording[status] if supported else None,
        forbidden_wording=forbidden,
        rationale=tuple(rationale),
    )
