from __future__ import annotations

import math
from typing import Any

import numpy as np

SUPPORT_STATUS = "supports_prespecified_predictive_generalization"
NONSUPPORT_STATUS = "does_not_meet_prespecified_support_rule"
INSUFFICIENT_STATUS = "insufficient_scorable_animals"


def exact_one_sided_sign_pvalue(positive: int, total: int) -> float:
    if total < 1:
        raise ValueError("sign test requires at least one scorable animal")
    if positive < 0 or positive > total:
        raise ValueError("positive count must be within [0, total]")
    numerator = sum(math.comb(total, count) for count in range(positive, total + 1))
    return numerator / (2**total)


def evaluate_final_inference(
    animal_effects: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    scorable: list[dict[str, Any]] = []
    for row in animal_effects:
        value = row.get("paired_alignment_effect")
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            continue
        scorable.append({"animal_id": row["animal_id"], "paired_alignment_effect": numeric})

    total = len(scorable)
    positive = sum(row["paired_alignment_effect"] > 0.0 for row in scorable)
    zero = sum(row["paired_alignment_effect"] == 0.0 for row in scorable)
    negative = total - positive - zero
    median_effect = (
        float(np.median([row["paired_alignment_effect"] for row in scorable]))
        if scorable
        else None
    )
    p_value = exact_one_sided_sign_pvalue(positive, total) if total else None

    minimum = int(config["final_minimum_scorable_animals"])
    alpha = float(config["final_alpha"])
    median_threshold = float(config["final_require_median_effect_gt"])
    if total < minimum:
        status = INSUFFICIENT_STATUS
    elif (
        p_value is not None
        and p_value <= alpha
        and median_effect is not None
        and median_effect > median_threshold
    ):
        status = SUPPORT_STATUS
    else:
        status = NONSUPPORT_STATUS

    return {
        "schema_version": 1,
        "protocol": "mc2p-final-animal-inference-v1",
        "unit": config["final_inference_unit"],
        "test": config["final_test"],
        "alternative": config["final_alternative"],
        "null_positive_probability": config["final_null_positive_probability"],
        "zero_effect_policy": config["final_zero_effect_policy"],
        "minimum_scorable_animals": minimum,
        "alpha": alpha,
        "require_median_effect_gt": median_threshold,
        "scorable_animals": total,
        "positive_effect_animals": positive,
        "zero_effect_animals": zero,
        "negative_effect_animals": negative,
        "median_paired_effect": median_effect,
        "exact_one_sided_sign_pvalue": p_value,
        "status": status,
        "supportive": status == SUPPORT_STATUS,
        "claim_boundary": (
            "The inferential unit is the animal. Pixels and overlapping windows contribute to each "
            "animal's held-out prediction score but are not counted as independent replicates. Support "
            "requires the prespecified aligned-minus-validation-selected-null direction to generalize "
            "across held-out sessions under this exact rule. It does not establish causality, spike-level "
            "prediction, a cross-animal raw-pixel decoder, or a connectome mechanism."
        ),
    }
