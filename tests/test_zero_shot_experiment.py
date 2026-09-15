from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.evidence.bootstrap import build_ledger
from fly_sniff.experiment_protocol import ExperimentSpec
from fly_sniff.zero_shot_experiment import seal_zero_shot, validate_zero_shot_spec

SPEC = Path("configs/zero_shot_latent_wiring_v1.json")


def _payload() -> dict:
    return json.loads(SPEC.read_text())


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _valid_artifacts(tmp_path: Path) -> dict[str, Path]:
    bootstrap = json.loads(Path("configs/evidence_bootstrap_v1.json").read_text())
    ledger = build_ledger(bootstrap)
    ledger_path = ledger.save(tmp_path / "evidence-ledger.json")

    model_path = _write_json(
        tmp_path / "physiology-model.json",
        {
            "protocol": "physiology-calibrated-model-v1",
            "fit_status": "accepted_under_preregistered_objectives",
            "parameters": {"global_recurrent_gain": 1.0},
            "probe_results": {"probe": {"passed": True}},
        },
    )
    plume_path = _write_json(
        tmp_path / "plume.json",
        {
            "protocol": "zero-shot-experimental-plume-evidence-v1",
            "source_protocol": "experimental-plume-sensory-validation-v3",
            "status": "qualified-for-sensory-evaluation",
            "controller_access": False,
            "navigation_performance_used": False,
            "source_bytes_verified": True,
            "native_time_basis_verified": True,
            "published_cue_reference_comparison_passed": True,
            "physical_pixel_to_fly_mapping_frozen": True,
            "bilateral_sensor_geometry_frozen": True,
            "receipt_sha256": {
                "source": "11" * 32,
                "archive": "22" * 32,
                "cue_reference": "33" * 32,
                "physical_geometry": "44" * 32,
            },
        },
    )
    structure_path = _write_json(
        tmp_path / "structure.json",
        {
            "protocol": "zero-shot-olfactory-structure-evidence-v1",
            "source_protocol": "olfactory-motion-structural-audit-v1",
            "source_status": "candidate_audit_pass_complete_structural_motif",
            "controller_access": False,
            "navigation_performance_used": False,
            "body_ids_selected_from_navigation": False,
            "external_connectome_body_ids_imported": False,
            "functional_motion_claim_allowed": False,
            "navigation_claim_allowed": False,
            "source_audit_sha256": "55" * 32,
        },
    )
    return {
        "evidence_ledger": ledger_path,
        "physiology_calibrated_model": model_path,
        "experimental_plume_receipt": plume_path,
        "olfactory_motion_structural_audit": structure_path,
        "null_factory_protocol": Path("configs/null_factory_v2.json"),
        "connectome_necessity_protocol": Path("configs/connectome_necessity_v1.json"),
    }


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

    paths = _valid_artifacts(tmp_path)
    lock = seal_zero_shot(
        spec,
        artifact_paths=paths,
        code_ref="deadbeef",
        runtime={"python": "test"},
    )
    assert len(lock["lock_sha256"]) == 64
    assert set(lock["artifact_sha256"]) == set(spec.metadata["required_artifacts"])
    assert lock["runtime"]["artifact_validation"]["experimental_plume_receipt"]["valid"] is True


def test_zero_shot_lock_rejects_semantically_invalid_artifact(tmp_path: Path) -> None:
    spec = ExperimentSpec.from_dict(_payload())
    paths = _valid_artifacts(tmp_path)
    _write_json(paths["experimental_plume_receipt"], {})
    with pytest.raises(ValueError, match="failed semantic validation"):
        seal_zero_shot(
            spec,
            artifact_paths=paths,
            code_ref="deadbeef",
            runtime={"python": "test"},
        )
