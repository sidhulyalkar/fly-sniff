from __future__ import annotations

from collections.abc import Iterable

from .schema import EvidenceClass

FORBIDDEN_MODELED_LABEL_FRAGMENTS = (
    "measured neural activity",
    "recorded neural activity",
    "reconstructed the brain",
    "reconstructed brain activity",
    "actual connectome activity",
)


def validate_public_label(label: str, evidence: Iterable[EvidenceClass]) -> None:
    normalized = label.casefold()
    classes = set(evidence)
    if EvidenceClass.MODELED_LATENT_NEURAL_STATE in classes:
        for fragment in FORBIDDEN_MODELED_LABEL_FRAGMENTS:
            if fragment in normalized:
                raise ValueError(
                    f"modeled latent state cannot be presented as {fragment!r}; use inferred/modeled wording"
                )
    if (
        "measured" in normalized
        and EvidenceClass.MEASURED_NEURAL_ACTIVITY not in classes
        and ("neural" in normalized or "brain" in normalized)
    ):
        raise ValueError("measured neural wording requires measured_neural_activity evidence")


def milestone_zero_claims() -> dict[str, list[str]]:
    return {
        "allowed": [
            "predict measured neural population activity from held-out fly behavior video",
            "compare video and pose representations on animal-held-out neural decoding",
            "infer a modeled latent neural state with explicit uncertainty",
        ],
        "forbidden": [
            "reconstruct the complete fly brain from video",
            "recover exact activity of the full connectome from behavior alone",
            "call modeled connectome activity measured or recorded neural activity",
        ],
    }
