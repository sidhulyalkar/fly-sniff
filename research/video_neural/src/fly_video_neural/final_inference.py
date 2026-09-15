from __future__ import annotations

import math
from typing import Any

import numpy as np

SUPPORT_STATUS = "supports_prespecified_predictive_generalization"
NONSUPPORT_STATUS = "does_not_meet_prespecified_support_rule"
INSUFFICIENT_STATUS = "insufficient_prespecified_test_population"
INCOMPLETE_STATUS = "incomplete_prespecified_test_population"


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
    *,
    confirmatory_animal_ids: list[str],
) -> dict[str, Any]:
    if config["final_inference_population"] != "validation_eligible_animals_only":
        raise ValueError("final inference population must remain validation-eligible animals only")
    if not confirmatory_animal_ids:
        raise ValueError("confirmatory animal population must be frozen before final inference")
    if len(confirmatory_animal_ids) != len(set(confirmatory_animal_ids)):
        raise ValueError("confirmatory animal ids must be unique")

    by_animal: dict[str, dict[str, Any]] = {}
    for row in animal_effects:
        animal = str(row["animal_id"])
        if animal in by_animal:
            raise ValueError(f"duplicate final effect row for animal {animal!r}")
        by_animal[animal] = row

    confirmatory_ids = sorted(confirmatory_animal_ids)
    confirmatory_set = set(confirmatory_ids)
    descriptive_only_ids = sorted(set(by_animal) - confirmatory_set)
    missing_ids = sorted(confirmatory_set - set(by_animal))
    scorable: list[dict[str, Any]] = []
    unscorable_ids: list[str] = []
    for animal in confirmatory_ids:
        row = by_animal.get(animal)
        if row is None:
            continue
        value = row.get("paired_alignment_effect")
        if value is None:
            unscorable_ids.append(animal)
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            unscorable_ids.append(animal)
            continue
        scorable.append({"animal_id": animal, "paired_alignment_effect": numeric})

    population_size = len(confirmatory_ids)
    total = len(scorable)
    positive = sum(row["paired_alignment_effect"] > 0.0 for row in scorable)
    zero = sum(row["paired_alignment_effect"] == 0.0 for row in scorable)
    negative = total - positive - zero
    median_effect = (
        float(np.median([row["paired_alignment_effect"] for row in scorable]))
        if scorable
        else None
    )

    minimum = int(config["final_minimum_scorable_animals"])
    alpha = float(config["final_alpha"])
    median_threshold = float(config["final_require_median_effect_gt"])
    require_complete = bool(config["final_require_all_validation_eligible_test_effects_computable"])
    complete_population = not missing_ids and not unscorable_ids and total == population_size
    p_value = (
        exact_one_sided_sign_pvalue(positive, total)
        if complete_population and total >= minimum
        else None
    )

    if population_size < minimum:
        status = INSUFFICIENT_STATUS
    elif require_complete and not complete_population:
        status = INCOMPLETE_STATUS
    elif total < minimum:
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
        "population": config["final_inference_population"],
        "test": config["final_test"],
        "alternative": config["final_alternative"],
        "null_positive_probability": config["final_null_positive_probability"],
        "zero_effect_policy": config["final_zero_effect_policy"],
        "minimum_scorable_animals": minimum,
        "require_all_validation_eligible_test_effects_computable": require_complete,
        "alpha": alpha,
        "require_median_effect_gt": median_threshold,
        "confirmatory_animal_ids": confirmatory_ids,
        "confirmatory_population_size": population_size,
        "descriptive_only_validation_ineligible_animal_ids": descriptive_only_ids,
        "missing_confirmatory_animal_ids": missing_ids,
        "unscorable_confirmatory_animal_ids": sorted(unscorable_ids),
        "complete_confirmatory_population": complete_population,
        "scorable_animals": total,
        "positive_effect_animals": positive,
        "zero_effect_animals": zero,
        "negative_effect_animals": negative,
        "median_paired_effect": median_effect,
        "exact_one_sided_sign_pvalue": p_value,
        "status": status,
        "supportive": status == SUPPORT_STATUS,
        "claim_boundary": (
            "The inferential unit is the animal and the confirmatory population is frozen before test as "
            "the validation-eligible animal IDs. Validation-ineligible animals may be reported descriptively "
            "but cannot enter confirmatory inference. A missing or non-computable held-out effect for any "
            "confirmatory animal cannot shrink the denominator and instead makes the prespecified final "
            "population incomplete. Pixels and overlapping windows are not counted as independent replicates. "
            "Support under this rule does not establish causality, spike-level prediction, a cross-animal "
            "raw-pixel decoder, or a connectome mechanism."
        ),
    }
