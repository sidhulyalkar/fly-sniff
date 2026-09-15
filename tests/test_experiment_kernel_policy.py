from __future__ import annotations

import json
from pathlib import Path


def test_experiment_kernel_policy_freezes_program_boundaries() -> None:
    policy = json.loads(Path("configs/experiment_kernel_policy_v1.json").read_text())

    latent = policy["programs"]["latent_wiring"]
    assert latent["navigation_reward_allowed"] is False
    assert latent["topology_specific_fit_allowed"] is False
    assert latent["whole_graph_backprop_allowed"] is False
    assert latent["independent_calibration_targets_required"] is True

    inductive = policy["programs"]["topology_inductive_bias"]
    assert inductive["topology_specific_fit_requires_equal_budget"] is True
    assert inductive["whole_graph_backprop_allowed"] is False

    learning = policy["programs"]["biological_learning"]
    assert learning["explicit_plasticity_scope_required"] is True
    assert learning["whole_graph_backprop_allowed"] is False


def test_confirmatory_policy_requires_graph_level_credibility_guards() -> None:
    policy = json.loads(Path("configs/experiment_kernel_policy_v1.json").read_text())
    confirmatory = policy["confirmatory_topology_claim"]

    assert confirmatory["minimum_frozen_topology_nulls"] >= 31
    assert confirmatory["preferred_flagship_topology_nulls"] >= 63
    assert confirmatory["paired_episode_conditions_required"] is True
    assert confirmatory["hidden_final_entropy_required"] is True
    assert confirmatory["one_way_final_required"] is True
    assert confirmatory["negative_results_retained"] is True


def test_structure_and_environment_lanes_remain_controller_isolated() -> None:
    policy = json.loads(Path("configs/experiment_kernel_policy_v1.json").read_text())
    integration = policy["integration"]

    assert integration["pr_22_controller_access_allowed"] is False
    assert integration["pr_23_controller_access_allowed"] is False
    assert integration["existing_task_optimization_program"] == "topology_inductive_bias"
    assert integration["primary_future_odor_program"] == "latent_wiring"
