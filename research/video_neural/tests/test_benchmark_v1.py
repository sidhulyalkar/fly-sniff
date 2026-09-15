from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_video_neural.benchmark import load_benchmark, validate_benchmark


def _path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "benchmark_v1.json"


def test_v1_records_pre_score_scientific_correction():
    document = load_benchmark(_path())
    assert document["supersedes"] == "mc2p_future_neural_v0"
    assert document["revision_status"] == "pre_data_scoring_correction"
    assert document["revision_was_informed_by_benchmark_scores"] is False
    assert document["expected_public_release_animals"] == 8
    assert document["measurement_rates_hz"] == {
        "behavior_video": 100.0,
        "two_photon_nominal": 16.0,
    }


def test_v1_primary_is_within_animal_session_heldout():
    document = load_benchmark(_path())
    primary = document["tasks"]["primary"]
    assert primary["split_unit"] == "session_id_within_animal"
    assert primary["session_split_seed"] == 2701
    assert primary["validation_sessions_per_animal"] == 1
    assert primary["test_sessions_per_animal"] == 1
    assert primary["target"] == "future_mean_dff_image"
    assert primary["raw_neural_pixel_target_allowed"] is True
    assert document["test_consumption_policy"] == "one_way_validation_unlock_then_final_runner"
    assert document["test_consumption_requires_validation_unlock"] is True
    assert document["development_commands_may_consume_test"] is False


def test_v1_forbids_raw_neural_pixels_for_unseen_animals():
    document = json.loads(_path().read_text())
    document["tasks"]["secondary"]["raw_neural_pixel_target_allowed"] = True
    with pytest.raises(ValueError, match="raw cross-animal"):
        validate_benchmark(document)


def test_v1_cross_animal_lane_stays_blocked_until_representation_is_frozen():
    document = json.loads(_path().read_text())
    document["tasks"]["secondary"]["status"] = "ready"
    with pytest.raises(ValueError, match="representation contract"):
        validate_benchmark(document)


def test_v1_rejects_weaker_test_consumption_policy():
    document = json.loads(_path().read_text())
    document["test_consumption_policy"] = "explicit_flag"
    with pytest.raises(ValueError, match="one-way"):
        validate_benchmark(document)


def test_v1_rejects_partial_public_release_contract():
    document = json.loads(_path().read_text())
    document["expected_public_release_animals"] = 7
    with pytest.raises(ValueError, match="eight-animal"):
        validate_benchmark(document)
