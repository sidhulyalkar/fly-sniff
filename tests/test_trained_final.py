import copy

import pandas as pd
import pytest

from fly_sniff.controllers import CastSurgeController
from fly_sniff.evaluate import manifest_digest
from fly_sniff.freeze import make_rewire_seeds
from fly_sniff.graph import GraphBundle
from fly_sniff.trained_final import (
    TRAINED_FINAL_HELDOUT_EPISODES,
    TRAINED_FINAL_OOD_EPISODES,
    TRAINED_FINAL_SCHEMA,
    TRAINED_FINAL_SPLIT_SEED,
    build_trained_factories,
    build_trained_final_manifest,
    verify_trained_final_manifest,
)
from fly_sniff.training import (
    DynamicsParameters,
    TaskOptimizedMaleCNSController,
    canonical_sha256,
    load_training_config,
)


def _bundle(dataset="synthetic-trained-final"):
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame(
        {
            "source": pd.Series(dtype=int),
            "target": pd.Series(dtype=int),
            "weight": pd.Series(dtype=float),
            "sign": pd.Series(dtype=int),
        }
    )
    roles = {
        "odor_context_left": [1],
        "odor_context_right": [2],
        "wind_basis_left": [3],
        "wind_basis_right": [4],
        "steer_left": [5],
        "steer_right": [6],
    }
    return GraphBundle(
        nodes,
        edges,
        roles,
        {"qualification_status": "qualified", "dataset": dataset},
    )


def _params(scale=1.0):
    return DynamicsParameters(
        tau_s=0.25 * scale,
        activation_gain=1.6,
        recurrent_gain=1.0,
        odor_gain=1.0,
        wind_basis_gain=1.0,
        turn_gain=2.4,
    )


def test_trained_final_factories_use_each_topologys_own_parameters():
    intact = _bundle("intact")
    lesion = _bundle("lesion")
    rewire_a = _bundle("rewire-a")
    rewire_b = _bundle("rewire-b")
    verified = {
        "intact": {"bundle": intact, "parameters": _params(1.0), "report": {}},
        "rewires": [
            {"seed": 10, "bundle": rewire_a, "parameters": _params(1.2), "report": {}},
            {"seed": 11, "bundle": rewire_b, "parameters": _params(1.4), "report": {}},
        ],
        "lesion": {"bundle": lesion, "parameters": _params(1.6), "report": {}},
    }
    factories, labels = build_trained_factories(verified, model_dt_s=0.05)
    assert labels == ["rewire", "rewire_01"]
    assert isinstance(factories["classical"](), CastSurgeController)

    intact_controller = factories["malecns"]()
    rewire_controller = factories["rewire"]()
    rewire_two_controller = factories["rewire_01"]()
    lesion_controller = factories["lesion"]()
    assert isinstance(intact_controller, TaskOptimizedMaleCNSController)
    assert isinstance(rewire_controller, TaskOptimizedMaleCNSController)
    assert isinstance(rewire_two_controller, TaskOptimizedMaleCNSController)
    assert isinstance(lesion_controller, TaskOptimizedMaleCNSController)
    assert intact_controller.parameters.tau_s == pytest.approx(0.25)
    assert rewire_controller.parameters.tau_s == pytest.approx(0.30)
    assert rewire_two_controller.parameters.tau_s == pytest.approx(0.35)
    assert lesion_controller.parameters.tau_s == pytest.approx(0.40)


def test_trained_final_manifest_binds_matched_training_and_passing_e002(monkeypatch):
    config = load_training_config("configs/task_optimization_v1.json")
    bundle = _bundle()
    intact_parameter_hash = "intact-parameter-hash"
    intact_audit_hash = "intact-audit-hash"
    execution_audit_hash = "execution-audit-hash"
    seeds = make_rewire_seeds(n=8)
    verified = {
        "intact": {
            "bundle": bundle,
            "parameters": _params(),
            "report": {
                "trained_parameter_sha256": intact_parameter_hash,
                "audit_receipt_sha256": intact_audit_hash,
            },
        },
        "rewires": [
            {
                "seed": seed,
                "bundle": bundle,
                "parameters": _params(),
                "report": {"trained_parameter_sha256": f"rewire-parameter-{seed}"},
            }
            for seed in seeds
        ],
        "lesion": {
            "bundle": bundle,
            "parameters": _params(),
            "report": {"trained_parameter_sha256": "lesion-parameter-hash"},
        },
        "optimizer_budget_sha256": "budget-hash",
        "optimizer_execution_audit_sha256": execution_audit_hash,
    }
    monkeypatch.setattr(
        "fly_sniff.trained_final.validate_matched_training_artifact",
        lambda bundle, config, matched_report: verified,
    )
    matched_report = {"protocol": "matched-task-optimization-controls-v1", "frozen": True}
    e002 = {
        "protocol": config["trained_e002"]["protocol"],
        "passed": True,
        "graph_sha256": bundle.replay_fingerprint(),
        "training_config_sha256": canonical_sha256(config),
        "trained_parameter_sha256": intact_parameter_hash,
        "training_audit_receipt_sha256": intact_audit_hash,
    }
    manifest = build_trained_final_manifest(
        bundle=bundle,
        circuit_sha256="file-digest",
        config=config,
        matched_report=matched_report,
        trained_e002=e002,
        code_ref="deadbeef",
    )
    verify_trained_final_manifest(manifest)
    assert manifest["schema"] == TRAINED_FINAL_SCHEMA
    assert manifest["split_seed"] == TRAINED_FINAL_SPLIT_SEED
    assert len(manifest["heldout_seeds"]) == TRAINED_FINAL_HELDOUT_EPISODES
    assert len(manifest["ood_seeds"]) == TRAINED_FINAL_OOD_EPISODES
    assert manifest["matched_training_report_sha256"] == canonical_sha256(matched_report)
    assert manifest["trained_e002_sha256"] == canonical_sha256(e002)
    assert manifest["optimizer_execution_audit_sha256"] == execution_audit_hash
    assert manifest["model_contract"]["controller"] == TaskOptimizedMaleCNSController.name
    assert manifest["model_contract"]["intact_parameter_sha256"] == intact_parameter_hash
    assert manifest["model_contract"]["lesion_parameter_sha256"] == "lesion-parameter-hash"
    assert set(manifest["model_contract"]["rewire_parameter_sha256_by_seed"]) == {
        str(seed) for seed in seeds
    }


def test_trained_final_manifest_rejects_protocol_seed_shopping(monkeypatch):
    config = load_training_config("configs/task_optimization_v1.json")
    bundle = _bundle()
    seeds = make_rewire_seeds(n=8)
    verified = {
        "intact": {
            "bundle": bundle,
            "parameters": _params(),
            "report": {
                "trained_parameter_sha256": "p",
                "audit_receipt_sha256": "a",
            },
        },
        "rewires": [
            {
                "seed": seed,
                "bundle": bundle,
                "parameters": _params(),
                "report": {"trained_parameter_sha256": str(seed)},
            }
            for seed in seeds
        ],
        "lesion": {
            "bundle": bundle,
            "parameters": _params(),
            "report": {"trained_parameter_sha256": "l"},
        },
        "optimizer_budget_sha256": "b",
        "optimizer_execution_audit_sha256": "execution-audit",
    }
    monkeypatch.setattr(
        "fly_sniff.trained_final.validate_matched_training_artifact",
        lambda bundle, config, matched_report: verified,
    )
    e002 = {
        "protocol": config["trained_e002"]["protocol"],
        "passed": True,
        "graph_sha256": bundle.replay_fingerprint(),
        "training_config_sha256": canonical_sha256(config),
        "trained_parameter_sha256": "p",
        "training_audit_receipt_sha256": "a",
    }
    manifest = build_trained_final_manifest(
        bundle=bundle,
        circuit_sha256="digest",
        config=config,
        matched_report={"x": 1},
        trained_e002=e002,
        code_ref="deadbeef",
    )
    changed = copy.deepcopy(manifest)
    changed["split_seed"] += 1
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = manifest_digest(changed)
    with pytest.raises(ValueError, match="split seed"):
        verify_trained_final_manifest(changed)


def test_trained_final_manifest_requires_execution_audit_binding(monkeypatch):
    config = load_training_config("configs/task_optimization_v1.json")
    bundle = _bundle()
    seeds = make_rewire_seeds(n=8)
    verified = {
        "intact": {
            "bundle": bundle,
            "parameters": _params(),
            "report": {"trained_parameter_sha256": "p", "audit_receipt_sha256": "a"},
        },
        "rewires": [
            {
                "seed": seed,
                "bundle": bundle,
                "parameters": _params(),
                "report": {"trained_parameter_sha256": str(seed)},
            }
            for seed in seeds
        ],
        "lesion": {
            "bundle": bundle,
            "parameters": _params(),
            "report": {"trained_parameter_sha256": "l"},
        },
        "optimizer_budget_sha256": "b",
        "optimizer_execution_audit_sha256": "execution-audit",
    }
    monkeypatch.setattr(
        "fly_sniff.trained_final.validate_matched_training_artifact",
        lambda bundle, config, matched_report: verified,
    )
    e002 = {
        "protocol": config["trained_e002"]["protocol"],
        "passed": True,
        "graph_sha256": bundle.replay_fingerprint(),
        "training_config_sha256": canonical_sha256(config),
        "trained_parameter_sha256": "p",
        "training_audit_receipt_sha256": "a",
    }
    manifest = build_trained_final_manifest(
        bundle=bundle,
        circuit_sha256="digest",
        config=config,
        matched_report={"x": 1},
        trained_e002=e002,
        code_ref="deadbeef",
    )
    changed = copy.deepcopy(manifest)
    changed.pop("optimizer_execution_audit_sha256")
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = manifest_digest(changed)
    with pytest.raises(ValueError, match="execution audit"):
        verify_trained_final_manifest(changed)
