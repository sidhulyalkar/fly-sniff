from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.experiment_protocol import ExperimentSpec
from fly_sniff.zero_shot_experiment import seal_zero_shot, validate_zero_shot_spec

SPEC = Path("configs/zero_shot_latent_wiring_v1.json")


def _payload() -> dict:
    return json.loads(SPEC.read_text())


def test_zero_shot_spec_passes_all_preregistration_gates() -> None:
    spec = ExperimentSpec.from_dict(_payload())
    report = validate_zero_shot_spec(spec)
    assert report["valid_for_preregistration"] is True
    assert report["passed_gate_count"] == report["gate_count"]


def test_zero_shot_spec_rejects_navigation_training() -> None:
    payload = _payload()
    payload["training"]["navigation_reward_allowed"] = True
    spec = ExperimentSpec.from_dict(payload)
    report = validate_zero_shot_spec(spec)
    gate = next(row for row in report["gates"] if row["name"] == "navigation_training_forbidden")
    assert gate["passed"] is False
    assert report["valid_for_preregistration"] is False


def test_zero_shot_spec_rejects_small_null_ensemble() -> None:
    payload = _payload()
    payload["nulls"]["minimum_confirmatory_topologies_per_family"] = 8
    spec = ExperimentSpec.from_dict(payload)
    report = validate_zero_shot_spec(spec)
    gate = next(
        row
        for row in report["gates"]
        if row["name"] == "confirmatory_null_hierarchy_and_count_frozen"
    )
    assert gate["passed"] is False


def test_zero_shot_lock_requires_exact_artifact_set(tmp_path: Path) -> None:
    spec = ExperimentSpec.from_dict(_payload())
    with pytest.raises(ValueError, match="artifact set mismatch"):
        seal_zero_shot(
            spec,
            artifact_paths={},
            code_ref="deadbeef",
            runtime={"python": "test"},
        )

    paths = {}
    for name in spec.metadata["required_artifacts"]:
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n")
        paths[name] = path
    lock = seal_zero_shot(
        spec,
        artifact_paths=paths,
        code_ref="deadbeef",
        runtime={"python": "test"},
    )
    assert len(lock["lock_sha256"]) == 64
    assert set(lock["artifact_sha256"]) == set(spec.metadata["required_artifacts"])
