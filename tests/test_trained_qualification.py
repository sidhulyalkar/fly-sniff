import json

import pandas as pd
import pytest

from fly_sniff import trained_qualification as tq
from fly_sniff.graph import GraphBundle
from fly_sniff.training import (
    DynamicsParameters,
    canonical_sha256,
    default_parameters,
    make_training_seed_split,
    optimizer_budget_receipt,
)


def _config():
    return {
        "protocol": "task-optimized-connectome-dynamics-v1",
        "connectome_sensory_interface": {
            "odor_mode": "mean_bilateral_nondirectional",
            "odor_roles": ["odor_context_left", "odor_context_right"],
            "direction_source": "signed_pfn_basis_from_body_frame_airflow_arrival",
            "wind_roles": ["wind_basis_left", "wind_basis_right"],
        },
        "seed_namespace": {
            "base": 2_100_000_000,
            "span": 1000,
            "split_seed": 17,
            "train_episodes": 6,
            "validation_episodes": 4,
        },
        "objective": {
            "success_weight": 0.5,
            "spl_weight": 0.4,
            "terminal_progress_weight": 0.1,
        },
        "optimizer": {
            "kind": "log-space cross-entropy method",
            "population": 4,
            "generations": 2,
            "elite_fraction": 0.5,
            "episodes_per_candidate": 3,
            "update_rate": 0.6,
            "initial_sigma_fraction_of_log_range": 0.2,
            "minimum_log_sigma": 0.03,
            "optimizer_seed": 23,
            "common_random_numbers": True,
        },
        "development_gate": {
            "minimum_validation_objective_delta_vs_own_untrained_default": 0.02,
            "maximum_validation_success_rate_drop_vs_own_untrained_default": 0.01,
        },
        "trainable_parameters": {
            "tau_s": {"default": 0.25, "min": 0.05, "max": 2.0},
            "activation_gain": {"default": 1.6, "min": 0.5, "max": 4.0},
            "recurrent_gain": {"default": 1.0, "min": 0.25, "max": 2.5},
            "odor_gain": {"default": 1.0, "min": 0.25, "max": 4.0},
            "wind_basis_gain": {"default": 1.0, "min": 0.25, "max": 4.0},
            "turn_gain": {"default": 2.4, "min": 0.5, "max": 6.0},
        },
        "trained_e002": {
            "protocol": "E002-trained-odor-gated-pfn-basis-v1",
            "probe_seed": 13013,
            "probe_steps": 40,
            "odor_level": 0.8,
            "crosswind_magnitude": 0.7,
            "minimum_signed_edge_fraction": 0.60,
            "minimum_odor_on_wind_turn_separation": 0.05,
            "minimum_odor_gating_separation_delta": 0.02,
            "maximum_odor_laterality_error": 1e-12,
            "maximum_lesioned_peak_turn": 0.02,
            "maximum_deterministic_replay_error": 1e-12,
            "memory_policy": "not primary",
        },
    }


def _parameters():
    return DynamicsParameters(
        tau_s=0.25,
        activation_gain=1.6,
        recurrent_gain=1.0,
        odor_gain=1.0,
        wind_basis_gain=1.0,
        turn_gain=2.4,
    )


def _bundle():
    nodes = pd.DataFrame({"bodyId": list(range(1, 7))})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 4],
            "target": [5, 6, 5, 6],
            "weight": [5, 5, 5, 5],
            "sign": [1, 1, 1, 1],
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
        {"qualification_status": "candidate", "dataset": "synthetic"},
    )


def _training_report(bundle, config, *, development_passed=True):
    parameters = _parameters()
    train, validation = make_training_seed_split(config)
    budget = optimizer_budget_receipt(
        config,
        train_seed_count=len(train),
        validation_seed_count=len(validation),
    )
    baseline_validation = {
        "n": len(validation),
        "objective": 0.50,
        "success_rate": 0.50,
    }
    trained_validation = {
        "n": len(validation),
        "objective": 0.52 if development_passed else 0.49,
        "success_rate": 0.50,
    }
    history = []
    for generation in range(int(config["optimizer"]["generations"])):
        count = int(config["optimizer"]["episodes_per_candidate"])
        batch = train[generation : generation + count]
        receipts = [
            {
                "index": index,
                "parameter_sha256": canonical_sha256(
                    {"generation": generation, "candidate": index}
                ),
                "objective": 0.5,
            }
            for index in range(int(config["optimizer"]["population"]))
        ]
        history.append(
            {
                "generation": generation,
                "seed_batch": batch,
                "seed_batch_sha256": canonical_sha256(batch),
                "candidate_receipts": receipts,
            }
        )
    objective_delta = trained_validation["objective"] - baseline_validation["objective"]
    success_delta = trained_validation["success_rate"] - baseline_validation["success_rate"]
    report = {
        "protocol": config["protocol"],
        "training_config_sha256": canonical_sha256(config),
        "graph_sha256": bundle.replay_fingerprint(),
        "sensory_interface": config["connectome_sensory_interface"],
        "train_seeds": train,
        "validation_seeds": validation,
        "train_seed_sha256": canonical_sha256(train),
        "validation_seed_sha256": canonical_sha256(validation),
        "train_seed_count": len(train),
        "validation_seed_count": len(validation),
        "final_test_namespace_touched": False,
        "optimizer_budget": budget,
        "optimizer_budget_sha256": canonical_sha256(budget),
        "baseline_parameters": default_parameters(config).to_dict(),
        "baseline_train": {"n": len(train)},
        "baseline_validation": baseline_validation,
        "trained_train": {"n": len(train)},
        "trained_validation": trained_validation,
        "trained_parameters": parameters.to_dict(),
        "trained_parameter_sha256": canonical_sha256(parameters.to_dict()),
        "development_gate_passed": development_passed,
        "validation_objective_delta": objective_delta,
        "validation_success_rate_delta": success_delta,
        "history": history,
    }
    report["audit_receipt_sha256"] = canonical_sha256(
        {
            "graph_sha256": report["graph_sha256"],
            "training_config_sha256": report["training_config_sha256"],
            "train_seed_sha256": report["train_seed_sha256"],
            "validation_seed_sha256": report["validation_seed_sha256"],
            "trained_parameter_sha256": report["trained_parameter_sha256"],
            "optimizer_budget_sha256": report["optimizer_budget_sha256"],
        }
    )
    return report


def _passing_probe():
    return tq.TrainedProbeResult(
        odor_on_downwind_left_turn=-0.20,
        odor_on_downwind_right_turn=0.20,
        odor_off_downwind_left_turn=-0.02,
        odor_off_downwind_right_turn=0.02,
        odor_on_wind_separation=0.40,
        odor_off_wind_separation=0.04,
        odor_gating_separation_delta=0.36,
        upwind_laterality_correct=True,
        odor_laterality_error=0.0,
        lesioned_peak_turn=0.0,
        deterministic_error=0.0,
    )


def test_training_report_is_bound_to_graph_config_and_parameter_hash():
    config = _config()
    bundle = _bundle()
    report = _training_report(bundle, config)
    parameters = tq.parameters_from_training_report(report, bundle, config)
    assert parameters == _parameters()

    changed = dict(report)
    changed["graph_sha256"] = "wrong"
    with pytest.raises(ValueError, match="graph fingerprint"):
        tq.parameters_from_training_report(changed, bundle, config)

    changed = dict(report)
    changed["trained_parameter_sha256"] = "wrong"
    with pytest.raises(ValueError, match="parameter hash"):
        tq.parameters_from_training_report(changed, bundle, config)


def test_training_report_rejects_final_test_leakage():
    config = _config()
    bundle = _bundle()
    report = _training_report(bundle, config)
    report["final_test_namespace_touched"] = True
    with pytest.raises(ValueError, match="final-test namespace"):
        tq.parameters_from_training_report(report, bundle, config)


def test_training_report_rejects_incomplete_optimizer_history_even_with_valid_budget_receipt():
    config = _config()
    bundle = _bundle()
    report = _training_report(bundle, config)
    report["history"][0]["candidate_receipts"].pop()
    with pytest.raises(ValueError, match="optimizer history|execution budget"):
        tq.parameters_from_training_report(report, bundle, config)


def test_trained_e002_requires_every_frozen_gate(monkeypatch):
    config = _config()
    bundle = _bundle()
    report = _training_report(bundle, config)
    monkeypatch.setattr(tq, "probe_trained_candidate", lambda *args, **kwargs: _passing_probe())

    qualified = tq.qualify_trained_candidate(bundle, report, config)
    assert qualified["passed"]
    assert qualified["passed_gate_count"] == qualified["gate_count"]
    assert qualified["protocol"] == "E002-trained-odor-gated-pfn-basis-v1"

    failed_report = _training_report(bundle, config, development_passed=False)
    failed = tq.qualify_trained_candidate(bundle, failed_report, config)
    assert not failed["passed"]
    failed_names = {gate["name"] for gate in failed["gates"] if not gate["passed"]}
    assert failed_names == {"training_development_gate"}


def test_trained_e002_rejects_bad_odor_gating(monkeypatch):
    config = _config()
    bundle = _bundle()
    report = _training_report(bundle, config)
    passing = _passing_probe()
    bad = tq.TrainedProbeResult(
        **{**passing.__dict__, "odor_gating_separation_delta": 0.0}
    )
    monkeypatch.setattr(tq, "probe_trained_candidate", lambda *args, **kwargs: bad)
    result = tq.qualify_trained_candidate(bundle, report, config)
    assert not result["passed"]
    gate = next(
        gate for gate in result["gates"] if gate["name"] == "odor_gates_pfn_basis_response"
    )
    assert gate["passed"] is False


def test_training_report_round_trip_is_json_safe(tmp_path):
    config = _config()
    bundle = _bundle()
    report = _training_report(bundle, config)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    loaded = json.loads(path.read_text())
    assert tq.parameters_from_training_report(loaded, bundle, config) == _parameters()
