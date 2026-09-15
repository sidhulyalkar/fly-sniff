from __future__ import annotations

from pathlib import Path

import pytest

from fly_video_neural.final_inference import (
    INSUFFICIENT_STATUS,
    NONSUPPORT_STATUS,
    SUPPORT_STATUS,
    evaluate_final_inference,
    exact_one_sided_sign_pvalue,
)
from fly_video_neural.validation_gate import load_acceptance_config


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "validation_acceptance_v1.json"
    return load_acceptance_config(path)


def _effects(values: list[float | None]) -> list[dict]:
    return [
        {"animal_id": f"fly{index}", "paired_alignment_effect": value}
        for index, value in enumerate(values)
    ]


def test_exact_one_sided_sign_probabilities_match_small_n_contract():
    assert exact_one_sided_sign_pvalue(7, 8) == pytest.approx(9 / 256)
    assert exact_one_sided_sign_pvalue(6, 8) == pytest.approx(37 / 256)
    assert exact_one_sided_sign_pvalue(7, 7) == pytest.approx(1 / 128)
    assert exact_one_sided_sign_pvalue(6, 7) == pytest.approx(8 / 128)
    assert exact_one_sided_sign_pvalue(6, 6) == pytest.approx(1 / 64)


def test_seven_of_eight_positive_effects_support_prespecified_rule():
    report = evaluate_final_inference(
        _effects([0.2, 0.1, 0.08, 0.06, 0.04, 0.03, 0.01, -0.01]),
        _config(),
    )
    assert report["status"] == SUPPORT_STATUS
    assert report["supportive"] is True
    assert report["scorable_animals"] == 8
    assert report["positive_effect_animals"] == 7
    assert report["exact_one_sided_sign_pvalue"] == pytest.approx(9 / 256)
    assert report["median_paired_effect"] > 0


def test_six_of_eight_positive_effects_do_not_support_rule():
    report = evaluate_final_inference(
        _effects([0.2, 0.1, 0.08, 0.06, 0.04, 0.03, -0.01, -0.02]),
        _config(),
    )
    assert report["status"] == NONSUPPORT_STATUS
    assert report["supportive"] is False
    assert report["exact_one_sided_sign_pvalue"] == pytest.approx(37 / 256)


def test_zero_effect_counts_as_nonpositive():
    report = evaluate_final_inference(
        _effects([0.2, 0.1, 0.08, 0.06, 0.04, 0.03, 0.01, 0.0]),
        _config(),
    )
    assert report["positive_effect_animals"] == 7
    assert report["zero_effect_animals"] == 1
    assert report["status"] == SUPPORT_STATUS


def test_fewer_than_six_scorable_animals_is_insufficient_even_if_all_positive():
    report = evaluate_final_inference(
        _effects([0.2, 0.1, 0.08, 0.06, 0.04, None, None, None]),
        _config(),
    )
    assert report["status"] == INSUFFICIENT_STATUS
    assert report["supportive"] is False
    assert report["scorable_animals"] == 5
