import copy
import json

import numpy as np
import pandas as pd
import pytest

import fly_sniff.training as training
from fly_sniff.env import Observation
from fly_sniff.graph import GraphBundle
from fly_sniff.training import (
    DynamicsParameters,
    TaskOptimizedMaleCNSController,
    default_parameters,
    load_training_config,
    make_training_seed_split,
    optimize_dynamics,
    validate_parameters,
)


def _config(tmp_path):
    payload = {
        "protocol": "task-optimized-connectome-dynamics-v1",
        "claim_boundary": "test",
        "seed_namespace": {
            "base": 2_100_000_000,
            "span": 1000,
            "split_seed": 17,
            "train_episodes": 6,
            "validation_episodes": 4,
        },
        "objective": {
            "version": "navigation-objective-v1",
            "success_weight": 0.5,
            "spl_weight": 0.4,
            "terminal_progress_weight": 0.1,
            "privileged_training_signal": "training-only distance",
        },
        "trainable_parameters": {
            "tau_s": {"default": 0.25, "min": 0.05, "max": 2.0},
            "activation_gain": {"default": 1.6, "min": 0.5, "max": 4.0},
            "recurrent_gain": {"default": 1.0, "min": 0.25, "max": 2.5},
            "odor_gain": {"default": 1.0, "min": 0.25, "max": 4.0},
            "wind_forward_gain": {"default": 1.0, "min": 0.25, "max": 4.0},
            "wind_backward_gain": {"default": 1.0, "min": 0.25, "max": 4.0},
            "wind_cross_gain": {"default": 1.0, "min": 0.25, "max": 4.0},
            "turn_gain": {"default": 2.4, "min": 0.5, "max": 6.0},
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
            "minimum_validation_objective_delta_vs_own_untrained_default": 0.0,
            "maximum_validation_success_rate_drop_vs_own_untrained_default": 0.01,
        },
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(payload))
    return load_training_config(path)


def _bundle(*, qualified=False):
    nodes = pd.DataFrame({"bodyId": [1, 2]})
    edges = pd.DataFrame(
        {
            "source": pd.Series(dtype=int),
            "target": pd.Series(dtype=int),
            "weight": pd.Series(dtype=float),
            "sign": pd.Series(dtype=int),
        }
    )
    roles = {
        "odor_left": [1],
        "odor_right": [2],
        "steer_left": [1],
        "steer_right": [2],
    }
    status = "qualified" if qualified else "candidate"
    return GraphBundle(
        nodes,
        edges,
        roles,
        {"qualification_status": status, "dataset": "synthetic"},
    )


def _params(**overrides):
    values = {
        "tau_s": 0.25,
        "activation_gain": 1.6,
        "recurrent_gain": 1.0,
        "odor_gain": 1.0,
        "wind_forward_gain": 1.0,
        "wind_backward_gain": 1.0,
        "wind_cross_gain": 1.0,
        "turn_gain": 2.4,
    }
    values.update(overrides)
    return DynamicsParameters.from_mapping(values)


def _observation(left=1.0, right=0.0):
    return Observation(
        left_odor=left,
        right_odor=right,
        mean_odor=0.5 * (left + right),
        odor_delta=right - left,
        wind_x_body=0.0,
        wind_y_body=0.0,
        heading=0.0,
    )


def test_training_seed_namespace_is_disjoint_from_final_test(tmp_path):
    config = _config(tmp_path)
    train, validation = make_training_seed_split(config)
    assert not set(train) & set(validation)
    assert min(train + validation) > training.FINAL_TEST_MAX_SEED
    assert (train, validation) == make_training_seed_split(config)


def test_parameter_contract_rejects_out_of_bounds_values(tmp_path):
    config = _config(tmp_path)
    validate_parameters(default_parameters(config), config)
    with pytest.raises(ValueError, match="odor_gain"):
        validate_parameters(_params(odor_gain=9.0), config)


def test_task_optimized_controller_keeps_graph_fixed_and_applies_tied_odor_gain():
    bundle = _bundle()
    before_nodes = bundle.nodes.copy(deep=True)
    before_edges = bundle.edges.copy(deep=True)
    before_roles = copy.deepcopy(bundle.roles)

    baseline = TaskOptimizedMaleCNSController(
        bundle,
        _params(odor_gain=1.0),
        require_qualified=False,
    )
    boosted = TaskOptimizedMaleCNSController(
        bundle,
        _params(odor_gain=2.0),
        require_qualified=False,
    )
    baseline.reset(1)
    boosted.reset(1)
    baseline_turn = baseline.act(_observation()).turn
    boosted_turn = boosted.act(_observation()).turn

    assert boosted_turn > baseline_turn > 0.0
    snapshot = boosted.input_snapshot()
    assert snapshot is not None
    left = next(item for item in snapshot["roles"] if item["role"] == "odor_left")
    right = next(item for item in snapshot["roles"] if item["role"] == "odor_right")
    assert left["gain"] == right["gain"] == 2.0
    assert left["raw_value"] == 1.0
    assert left["value"] == 2.0
    pd.testing.assert_frame_equal(bundle.nodes, before_nodes)
    pd.testing.assert_frame_equal(bundle.edges, before_edges)
    assert bundle.roles == before_roles


def test_candidate_training_requires_explicit_override(tmp_path):
    config = _config(tmp_path)
    params = default_parameters(config)
    with pytest.raises(ValueError, match="not sealed"):
        TaskOptimizedMaleCNSController(_bundle(), params)
    TaskOptimizedMaleCNSController(_bundle(), params, require_qualified=False)


def test_cem_is_deterministic_and_uses_only_development_seeds(tmp_path, monkeypatch):
    config = _config(tmp_path)
    bundle = _bundle(qualified=True)
    seen_seed_batches = []

    def fake_evaluate(
        bundle,
        parameters,
        seeds,
        config,
        *,
        require_qualified=True,
        arena=None,
        plume=None,
        sensors=None,
    ):
        assert require_qualified
        assert all(seed > training.FINAL_TEST_MAX_SEED for seed in seeds)
        seen_seed_batches.append(tuple(seeds))
        objective = 1.0 - abs(parameters.odor_gain - 2.0) / 10.0
        return {
            "n": len(seeds),
            "objective": objective,
            "success_rate": 1.0,
            "mean_spl": objective,
            "mean_terminal_progress": objective,
            "mean_path_length": 1.0,
            "mean_final_distance": 1.0,
        }

    monkeypatch.setattr(training, "evaluate_parameters", fake_evaluate)
    first = optimize_dynamics(bundle, config)
    first_batches = list(seen_seed_batches)
    seen_seed_batches.clear()
    second = optimize_dynamics(bundle, config)

    assert first["trained_parameters"] == second["trained_parameters"]
    assert first["history"] == second["history"]
    assert first_batches == seen_seed_batches
    assert first["final_test_namespace_touched"] is False
    assert first["trained_parameter_sha256"] == second["trained_parameter_sha256"]


def test_config_rejects_seed_namespace_that_can_touch_final_test(tmp_path):
    config = _config(tmp_path)
    config["seed_namespace"]["base"] = training.FINAL_TEST_MAX_SEED
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="overlaps"):
        load_training_config(path)
