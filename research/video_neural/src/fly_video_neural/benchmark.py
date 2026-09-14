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


def load_benchmark(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_benchmark(document)
    return document


def validate_benchmark(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("benchmark requires schema_version=1")
    if document.get("dataset_id") != "mc2p_v1":
        raise ValueError("v0 benchmark is frozen to mc2p_v1")
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
    if document.get("target_evidence_class") != EvidenceClass.MEASURED_NEURAL_ACTIVITY:
        raise ValueError("v0 target must remain measured neural activity")
    if document.get("model_lanes") != EXPECTED_MODEL_LANES:
        raise ValueError("v0 model-lane ordering changed")
    if document.get("connectome_prior_status") != "future_separately_versioned_lane":
        raise ValueError("connectome prior must remain outside milestone v0")
    if document.get("whole_connectome_reconstruction_claim_allowed") is not False:
        raise ValueError("whole-connectome reconstruction claim is forbidden in v0")
    if document.get("test_animals_for_hyperparameter_selection") is not False:
        raise ValueError("test animals may not be used for hyperparameter selection")
