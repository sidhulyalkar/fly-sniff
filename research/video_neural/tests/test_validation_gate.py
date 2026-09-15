from __future__ import annotations

from pathlib import Path

from fly_video_neural.alignment_null import NULL_FRACTIONS, NULL_NAME, NULL_SELECTION_RULE
from fly_video_neural.validation_gate import build_validation_unlock, load_acceptance_config


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "validation_acceptance_v1.json"
    return load_acceptance_config(path)


def _row(animal: str, value: float | None, *, fraction: float | None = None) -> dict:
    row = {
        "animal_id": animal,
        "selected_alpha": 1.0,
        "selected_validation_metrics": {"median_pearson_r": value},
        "alpha_candidates": [],
    }
    if fraction is not None:
        row["selected_null_fraction"] = fraction
    return row


def _reports(aligned_values: list[float | None], null_values: list[float | None]):
    animals = [f"fly{index}" for index in range(len(aligned_values))]
    qc = {
        "status": "pass",
        "test_target_values_summarized": False,
        "split_lock_sha256": "split",
        "report_sha256": "qc",
    }
    aligned = {
        "benchmark_id": "mc2p_future_neural_v1",
        "test_status": "locked_not_consumed",
        "split_lock_sha256": "split",
        "animals": [_row(animal, value) for animal, value in zip(animals, aligned_values, strict=True)],
    }
    null = {
        "benchmark_id": "mc2p_future_neural_v1",
        "test_status": "locked_not_consumed",
        "null_name": NULL_NAME,
        "null_fractions": list(NULL_FRACTIONS),
        "null_selection_rule": NULL_SELECTION_RULE,
        "split_lock_sha256": "split",
        "animals": [
            _row(animal, value, fraction=NULL_FRACTIONS[index % len(NULL_FRACTIONS)])
            for index, (animal, value) in enumerate(zip(animals, null_values, strict=True))
        ],
    }
    return qc, aligned, null


def test_validation_gate_unlocks_only_after_positive_majority_and_median_effect():
    qc, aligned, null = _reports([0.5, 0.4, 0.3, 0.2], [0.1, 0.2, 0.25, 0.1])
    report = build_validation_unlock(qc, aligned, null, _config())
    assert report["status"] == "unlocked_for_single_test_consumption"
    assert report["test_consumption_allowed"] is True
    assert report["positive_effect_animals"] == 4
    assert report["median_paired_effect"] > 0
    assert report["alignment_null_fractions"] == list(NULL_FRACTIONS)
    assert report["ineligible_animals"] == []
    assert all("selected_null_fraction" in row for row in report["animal_effects"])


def test_validation_gate_blocks_if_test_was_already_consumed():
    qc, aligned, null = _reports([0.5, 0.4, 0.3, 0.2], [0.1, 0.2, 0.25, 0.1])
    aligned["test_status"] = "consumed_explicitly"
    report = build_validation_unlock(qc, aligned, null, _config())
    assert report["status"] == "blocked"
    assert report["test_consumption_allowed"] is False
    assert any("consumed test" in failure for failure in report["failures"])


def test_validation_gate_blocks_nonmajority_effect_against_strongest_null():
    qc, aligned, null = _reports([0.3, 0.3, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2])
    report = build_validation_unlock(qc, aligned, null, _config())
    assert report["status"] == "blocked"
    assert report["positive_effect_animals"] == 2


def test_validation_gate_rejects_null_ensemble_contract_drift():
    qc, aligned, null = _reports([0.5, 0.4, 0.3, 0.2], [0.1, 0.2, 0.25, 0.1])
    null["null_fractions"] = [0.5]
    report = build_validation_unlock(qc, aligned, null, _config())
    assert report["status"] == "blocked"
    assert any("fractions mismatch" in failure for failure in report["failures"])


def test_noncomputable_validation_metric_makes_animal_ineligible_and_can_block():
    qc, aligned, null = _reports(
        [0.5, 0.4, 0.3, 0.2],
        [0.1, 0.2, None, 0.1],
    )
    report = build_validation_unlock(qc, aligned, null, _config())
    assert report["status"] == "blocked"
    assert report["eligible_animals"] == 3
    assert report["test_consumption_allowed"] is False
    assert report["ineligible_animals"] == [
        {
            "animal_id": "fly2",
            "reason": "validation_median_pearson_r_not_computable",
            "aligned_metric_computable": True,
            "selected_null_metric_computable": False,
            "selected_null_fraction": NULL_FRACTIONS[2],
        }
    ]
    assert any("only 3 eligible animals" in failure for failure in report["failures"])
