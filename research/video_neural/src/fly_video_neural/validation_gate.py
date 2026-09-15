from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .alignment_null import NULL_FRACTIONS, NULL_NAME, NULL_SELECTION_RULE


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_acceptance_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_acceptance_config(document)
    return document


def validate_acceptance_config(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("validation acceptance requires schema_version=1")
    if document.get("protocol") != "mc2p-validation-unlock-v1":
        raise ValueError("unexpected validation acceptance protocol")
    if document.get("benchmark_id") != "mc2p_future_neural_v1":
        raise ValueError("validation acceptance is frozen to mc2p_future_neural_v1")
    if document.get("alignment_null") != NULL_NAME:
        raise ValueError("validation acceptance null changed")
    if document.get("alignment_null_fractions") != list(NULL_FRACTIONS):
        raise ValueError("validation acceptance null fractions changed")
    if document.get("alignment_null_selection") != NULL_SELECTION_RULE:
        raise ValueError("validation acceptance null-selection rule changed")
    if document.get("minimum_eligible_animals") != 4:
        raise ValueError("minimum eligible animal count changed")
    if document.get("require_strict_majority_positive_effect") is not True:
        raise ValueError("strict-majority rule changed")
    if document.get("require_median_effect_gt") != 0.0:
        raise ValueError("median alignment-effect threshold changed")
    if document.get("may_change_after_validation_scores") is not False:
        raise ValueError("validation unlock criteria cannot change after scores")


def _selected_validation(row: dict[str, Any]) -> dict[str, Any]:
    if "selected_validation_metrics" in row:
        return row["selected_validation_metrics"]
    selected_alpha = row["selected_alpha"]
    matches = [
        candidate
        for candidate in row["alpha_candidates"]
        if candidate["alpha"] == selected_alpha
    ]
    if len(matches) != 1:
        raise ValueError("could not reconstruct selected validation metrics")
    return matches[0]["validation"]


def build_validation_unlock(
    qc_report: dict[str, Any],
    aligned_report: dict[str, Any],
    null_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_acceptance_config(config)
    failures: list[str] = []
    if qc_report.get("status") != "pass":
        failures.append("development data QC did not pass")
    if qc_report.get("test_target_values_summarized") is not False:
        failures.append("development QC summarized test target values")
    for name, report in (("aligned", aligned_report), ("null", null_report)):
        if report.get("benchmark_id") != "mc2p_future_neural_v1":
            failures.append(f"{name} report has wrong benchmark id")
        if report.get("test_status") != "locked_not_consumed":
            failures.append(f"{name} report consumed test data")
    if null_report.get("null_name") != config["alignment_null"]:
        failures.append("alignment-null identity mismatch")
    if null_report.get("null_fractions") != config["alignment_null_fractions"]:
        failures.append("alignment-null fractions mismatch")
    if null_report.get("null_selection_rule") != config["alignment_null_selection"]:
        failures.append("alignment-null selection-rule mismatch")
    split_hashes = {
        qc_report.get("split_lock_sha256"),
        aligned_report.get("split_lock_sha256"),
        null_report.get("split_lock_sha256"),
    }
    if len(split_hashes) != 1 or None in split_hashes:
        failures.append("QC/aligned/null reports do not bind the same split lock")

    aligned = {row["animal_id"]: row for row in aligned_report.get("animals", [])}
    null = {row["animal_id"]: row for row in null_report.get("animals", [])}
    if set(aligned) != set(null):
        failures.append("aligned and null reports cover different animals")
    effects: list[dict[str, Any]] = []
    for animal in sorted(set(aligned) & set(null)):
        aligned_metrics = _selected_validation(aligned[animal])
        null_metrics = _selected_validation(null[animal])
        aligned_r = float(aligned_metrics["median_pearson_r"])
        null_r = float(null_metrics["median_pearson_r"])
        effects.append(
            {
                "animal_id": animal,
                "aligned_validation_median_pearson_r": aligned_r,
                "strongest_null_validation_median_pearson_r": null_r,
                "selected_null_fraction": null[animal].get("selected_null_fraction"),
                "paired_effect": aligned_r - null_r,
            }
        )

    eligible = len(effects)
    positive = sum(row["paired_effect"] > 0 for row in effects)
    median_effect = (
        float(np.median([row["paired_effect"] for row in effects])) if effects else None
    )
    if eligible < config["minimum_eligible_animals"]:
        failures.append(
            f"only {eligible} eligible animals; need at least {config['minimum_eligible_animals']}"
        )
    if (
        config["require_strict_majority_positive_effect"]
        and eligible
        and positive <= eligible / 2
    ):
        failures.append("aligned decoder does not beat strongest null in a strict majority of animals")
    if median_effect is None or median_effect <= config["require_median_effect_gt"]:
        failures.append("median paired alignment effect against strongest null is not positive")

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "benchmark_id": config["benchmark_id"],
        "status": "unlocked_for_single_test_consumption" if not failures else "blocked",
        "split_lock_sha256": aligned_report.get("split_lock_sha256"),
        "qc_report_sha256": qc_report.get("report_sha256"),
        "acceptance_config_sha256": _sha(config),
        "alignment_null": config["alignment_null"],
        "alignment_null_fractions": config["alignment_null_fractions"],
        "alignment_null_selection": config["alignment_null_selection"],
        "eligible_animals": eligible,
        "positive_effect_animals": positive,
        "positive_effect_fraction": positive / eligible if eligible else 0.0,
        "median_paired_effect": median_effect,
        "animal_effects": effects,
        "failures": failures,
        "test_consumption_allowed": not failures,
        "claim_boundary": (
            "This development gate only authorizes one explicit test evaluation. It compares aligned "
            "decoding with the strongest prespecified null selected on validation, not test. It is not "
            "a test result and cannot establish a neural-decoding claim."
        ),
    }
    report["report_sha256"] = _sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate one-way MC2P v1 test consumption")
    parser.add_argument("config")
    parser.add_argument("qc_report")
    parser.add_argument("aligned_report")
    parser.add_argument("null_report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_acceptance_config(args.config)
    qc = json.loads(Path(args.qc_report).read_text())
    aligned = json.loads(Path(args.aligned_report).read_text())
    null = json.loads(Path(args.null_report).read_text())
    report = build_validation_unlock(qc, aligned, null, config)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": report["status"], "sha256": report["report_sha256"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
