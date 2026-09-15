from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import EvidenceClass

EXPECTED_MODEL_LANES = [
    "pose_only",
    "video_only",
    "video_plus_pose",
    "video_pose_plus_measured_context",
]

V0_ID = "mc2p_future_neural_v0"
V1_ID = "mc2p_future_neural_v1"


def load_benchmark(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_benchmark(document)
    return document


def validate_benchmark(document: dict[str, Any]) -> None:
    benchmark_id = document.get("benchmark_id")
    if benchmark_id == V0_ID:
        _validate_v0(document)
        return
    if benchmark_id == V1_ID:
        _validate_v1(document)
        return
    raise ValueError(f"unsupported benchmark_id {benchmark_id!r}")


def _validate_common(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("benchmark requires schema_version=1")
    if document.get("dataset_id") != "mc2p_v1":
        raise ValueError("benchmark is frozen to mc2p_v1")
    if document.get("target_evidence_class") != EvidenceClass.MEASURED_NEURAL_ACTIVITY:
        raise ValueError("target must remain measured neural activity")
    if document.get("model_lanes") != EXPECTED_MODEL_LANES:
        raise ValueError("model-lane ordering changed")
    if document.get("connectome_prior_status") != "future_separately_versioned_lane":
        raise ValueError("connectome prior must remain outside measured-neural benchmark")
    if document.get("whole_connectome_reconstruction_claim_allowed") is not False:
        raise ValueError("whole-connectome reconstruction claim is forbidden")
    if document.get("ridge_alpha_grid") != [0.01, 0.1, 1.0, 10.0, 100.0]:
        raise ValueError("ridge alpha grid changed")
    if document.get("model_selection_metric") != "median_pearson_r":
        raise ValueError("model-selection metric must be median_pearson_r")


def _validate_v0(document: dict[str, Any]) -> None:
    _validate_common(document)
    if document.get("input_history_s") != 3.0 or document.get("prediction_horizon_s") != 0.5:
        raise ValueError("v0 temporal contract is frozen to 3.0 s history -> 0.5 s target")
    if document.get("window_stride_s") != 0.5:
        raise ValueError("v0 window stride is frozen to 0.5 s")
    if document.get("rest_behavior_included") is not True:
        raise ValueError("v0 retains rest behavior")
    if document.get("alignment_contract") != "monotonic_behavior_frame_to_neural_frame_index":
        raise ValueError("v0 requires a monotonic behavior-to-neural alignment")
    if document.get("split_seed") != 1701:
        raise ValueError("v0 split seed is frozen to 1701")
    if document.get("expected_primary_dataset_animals") != 8:
        raise ValueError("v0 expects the eight-animal public MC2P release")
    if document.get("expected_split_counts_if_eight") != {"train": 5, "validation": 1, "test": 2}:
        raise ValueError("v0 eight-animal split counts must remain 5/1/2")
    if document.get("split_unit") != "animal_id":
        raise ValueError("v0 split unit must be animal_id")
    if document.get("session_may_cross_split") is not False:
        raise ValueError("sessions may not cross data splits")
    if document.get("input_target_overlap_allowed") is not False:
        raise ValueError("input/target temporal overlap must remain forbidden")
    if document.get("test_animals_for_hyperparameter_selection") is not False:
        raise ValueError("test animals may not be used for hyperparameter selection")
    if document.get("test_consumption_requires_explicit_flag") is not True:
        raise ValueError("v0 requires explicit acknowledgement before test consumption")


def _validate_v1(document: dict[str, Any]) -> None:
    _validate_common(document)
    if document.get("supersedes") != V0_ID:
        raise ValueError("v1 must explicitly supersede v0")
    if document.get("revision_status") != "pre_data_scoring_correction":
        raise ValueError("v1 revision status must remain pre_data_scoring_correction")
    if document.get("revision_was_informed_by_benchmark_scores") is not False:
        raise ValueError("v1 correction must remain uninformed by benchmark scores")
    if document.get("expected_public_release_animals") != 8:
        raise ValueError("v1 expects the complete eight-animal public MC2P release")
    rates = document.get("measurement_rates_hz")
    if rates != {"behavior_video": 100.0, "two_photon_nominal": 16.0}:
        raise ValueError("v1 measurement-rate contract changed")
    if document.get("alignment_authority") != "closest_timestamp_behavior_frame_to_neural_frame":
        raise ValueError("v1 timestamp-alignment authority changed")
    tasks = document.get("tasks")
    if not isinstance(tasks, dict):
        raise TypeError("v1 requires primary and secondary task contracts")
    primary = tasks.get("primary", {})
    if primary.get("name") != "within_animal_session_heldout":
        raise ValueError("v1 primary task must remain within-animal session-heldout")
    if (
        primary.get("input_history_s") != 3.0
        or primary.get("prediction_horizon_s") != 0.5
        or primary.get("window_stride_s") != 0.5
    ):
        raise ValueError("v1 primary temporal contract is frozen to 3.0/0.5/0.5 s")
    if primary.get("split_unit") != "session_id_within_animal":
        raise ValueError("v1 primary split unit must be session within animal")
    if primary.get("session_split_seed") != 2701:
        raise ValueError("v1 primary session split seed is frozen to 2701")
    if primary.get("validation_sessions_per_animal") != 1:
        raise ValueError("v1 primary requires one validation session per animal")
    if primary.get("test_sessions_per_animal") != 1:
        raise ValueError("v1 primary requires one test session per animal")
    if primary.get("raw_neural_pixel_target_allowed") is not True:
        raise ValueError("v1 within-animal primary task allows measured neural image targets")
    if primary.get("target") != "future_mean_dff_image":
        raise ValueError("v1 primary target must remain future_mean_dff_image")
    if primary.get("minimum_sessions_per_animal") != 3:
        raise ValueError("v1 primary task requires at least three sessions per animal")
    if primary.get("test_sessions_for_hyperparameter_selection") is not False:
        raise ValueError("v1 primary test sessions may not tune hyperparameters")
    secondary = tasks.get("secondary", {})
    if secondary.get("name") != "across_animal_subject_invariant":
        raise ValueError("v1 secondary task must remain across-animal subject-invariant")
    if secondary.get("split_unit") != "animal_id":
        raise ValueError("v1 secondary split unit must remain animal_id")
    if secondary.get("status") != "blocked_pending_neural_representation_contract":
        raise ValueError("v1 cross-animal task must remain blocked pending representation contract")
    if secondary.get("raw_neural_pixel_target_allowed") is not False:
        raise ValueError("raw cross-animal neural-pixel targets are forbidden in v1")
    if secondary.get("test_animals_for_hyperparameter_selection") is not False:
        raise ValueError("v1 secondary test animals may not tune hyperparameters")
    if document.get("test_consumption_policy") != "one_way_validation_unlock_then_final_runner":
        raise ValueError("v1 test consumption must use the one-way validation-unlock final runner")
    if document.get("test_consumption_requires_validation_unlock") is not True:
        raise ValueError("v1 final evaluation requires the frozen validation unlock")
    if document.get("development_commands_may_consume_test") is not False:
        raise ValueError("v1 development commands may not consume held-out test sessions")
