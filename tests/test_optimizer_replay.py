import copy

import pandas as pd
import pytest

from fly_sniff import training
from fly_sniff.graph import GraphBundle
from fly_sniff.optimizer_replay import replay_single_optimizer_history
from fly_sniff.reproducible_training import seal_training_runtime


def _bundle():
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame(
        {
            "source": pd.Series(dtype=int),
            "target": pd.Series(dtype=int),
            "weight": pd.Series(dtype=float),
            "sign": pd.Series(dtype=int),
        }
    )
    return GraphBundle(
        nodes,
        edges,
        {
            "odor_context_left": [1],
            "odor_context_right": [2],
            "wind_basis_left": [3],
            "wind_basis_right": [4],
            "steer_left": [5],
            "steer_right": [6],
        },
        {"qualification_status": "candidate", "dataset": "synthetic-cem-replay"},
    )


def _config():
    config = copy.deepcopy(training.load_training_config("configs/task_optimization_v1.json"))
    config["optimizer"].update(
        {
            "population": 4,
            "generations": 2,
            "episodes_per_candidate": 3,
        }
    )
    return config


def _summary(seeds, objective):
    return {
        "n": len(seeds),
        "objective": float(objective),
        "success_rate": 0.5,
        "mean_spl": 0.5,
        "mean_terminal_progress": 0.0,
        "mean_path_length": 1.0,
        "mean_final_distance": 1.0,
    }


def _trained_report(monkeypatch):
    config = _config()
    bundle = _bundle()

    def fake_evaluate(bundle, parameters, seeds, config, **kwargs):
        objective = (
            0.30
            + 0.02 * parameters.odor_gain
            + 0.01 * parameters.wind_basis_gain
            + 0.005 * parameters.turn_gain
        )
        return _summary(seeds, objective)

    monkeypatch.setattr(training, "evaluate_parameters", fake_evaluate)
    report = training.optimize_dynamics(bundle, config, require_qualified=False)
    return seal_training_runtime(report), config


def test_deterministic_cem_replay_reconstructs_seed_batches_candidates_and_updates(monkeypatch):
    report, config = _trained_report(monkeypatch)
    replay = replay_single_optimizer_history(report, config)
    assert replay["status"] == "deterministic_cem_mechanics_replayed"
    assert replay["generation_count"] == 2
    assert replay["trained_parameter_sha256"] == report["trained_parameter_sha256"]
    assert replay["optimizer_history_sha256"] == report["optimizer_history_sha256"]
    assert len(replay["generations"]) == 2


def test_deterministic_cem_replay_rejects_candidate_hash_tampering(monkeypatch):
    report, config = _trained_report(monkeypatch)
    changed = copy.deepcopy(report)
    changed["history"][0]["candidate_receipts"][1]["parameter_sha256"] = "0" * 64
    changed["optimizer_history_sha256"] = training.canonical_sha256(changed["history"])
    with pytest.raises(RuntimeError, match="candidate parameter hash mismatch"):
        replay_single_optimizer_history(changed, config)


def test_deterministic_cem_replay_rejects_seed_batch_tampering(monkeypatch):
    report, config = _trained_report(monkeypatch)
    changed = copy.deepcopy(report)
    changed["history"][0]["seed_batch"] = list(reversed(changed["history"][0]["seed_batch"]))
    changed["history"][0]["seed_batch_sha256"] = training.canonical_sha256(
        changed["history"][0]["seed_batch"]
    )
    changed["optimizer_history_sha256"] = training.canonical_sha256(changed["history"])
    with pytest.raises(RuntimeError, match="seed batch mismatch"):
        replay_single_optimizer_history(changed, config)


def test_deterministic_cem_replay_rejects_derived_distribution_tampering(monkeypatch):
    report, config = _trained_report(monkeypatch)
    changed = copy.deepcopy(report)
    changed["history"][0]["distribution_log_sigma"]["tau_s"] += 0.01
    changed["optimizer_history_sha256"] = training.canonical_sha256(changed["history"])
    with pytest.raises(RuntimeError, match="distribution_log_sigma.tau_s mismatch"):
        replay_single_optimizer_history(changed, config)
