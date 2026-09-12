import copy

import pandas as pd
import pytest

from fly_sniff import trained_qualification as tq
from fly_sniff import training
from fly_sniff.env import Observation
from fly_sniff.graph import GraphBundle
from fly_sniff.training import (
    TaskOptimizedMaleCNSController,
    canonical_sha256,
    default_parameters,
    make_training_seed_split,
    optimize_dynamics,
    train_matched_control_cohort,
)


def _config():
    return training.load_training_config("configs/task_optimization_v1.json")


def _bundle(*, qualified=True, overlap_sensory_and_steering=False):
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame(
        {
            "source": pd.Series(dtype=int),
            "target": pd.Series(dtype=int),
            "weight": pd.Series(dtype=float),
            "sign": pd.Series(dtype=int),
        }
    )
    steer_left = [3] if overlap_sensory_and_steering else [5]
    roles = {
        "odor_context_left": [1],
        "odor_context_right": [2],
        "wind_basis_left": [3],
        "wind_basis_right": [4],
        "steer_left": steer_left,
        "steer_right": [6],
    }
    return GraphBundle(
        nodes,
        edges,
        roles,
        {
            "qualification_status": "qualified" if qualified else "candidate",
            "dataset": "synthetic-redteam",
        },
    )


def _summary(seeds, objective=0.5):
    return {
        "n": len(seeds),
        "objective": float(objective),
        "success_rate": 0.5,
        "mean_spl": 0.5,
        "mean_terminal_progress": 0.0,
        "mean_path_length": 1.0,
        "mean_final_distance": 1.0,
    }


def _training_report(bundle, config):
    params = default_parameters(config)
    train, validation = make_training_seed_split(config)
    return {
        "protocol": config["protocol"],
        "training_config_sha256": canonical_sha256(config),
        "graph_sha256": bundle.replay_fingerprint(),
        "final_test_namespace_touched": False,
        "train_seed_sha256": canonical_sha256(train),
        "validation_seed_sha256": canonical_sha256(validation),
        "train_seed_count": len(train),
        "validation_seed_count": len(validation),
        "trained_parameters": params.to_dict(),
        "trained_parameter_sha256": canonical_sha256(params.to_dict()),
        "development_gate_passed": True,
        "validation_objective_delta": 0.02,
        "validation_success_rate_delta": 0.0,
    }


def test_candidate_selection_never_uses_validation_inside_cem(monkeypatch):
    """Validation may gate the frozen result, but it must never rank CEM candidates."""
    config = copy.deepcopy(_config())
    config["optimizer"].update(
        {"population": 4, "generations": 2, "episodes_per_candidate": 3}
    )
    bundle = _bundle()
    train, validation = make_training_seed_split(config)
    train_set = set(train)
    validation_set = set(validation)
    calls = []

    def fake_evaluate(bundle, parameters, seeds, config, **kwargs):
        calls.append(tuple(seeds))
        return _summary(seeds)

    monkeypatch.setattr(training, "evaluate_parameters", fake_evaluate)
    optimize_dynamics(bundle, config)

    # baseline train, baseline validation, CEM candidates, trained train, trained validation
    generation_calls = calls[2:-2]
    assert generation_calls
    assert all(set(batch) <= train_set for batch in generation_calls)
    assert all(not (set(batch) & validation_set) for batch in generation_calls)
    assert set(calls[1]) == validation_set
    assert set(calls[-1]) == validation_set


def test_development_and_final_seed_namespaces_are_disjoint_by_construction():
    config = _config()
    train, validation = make_training_seed_split(config)
    assert set(train).isdisjoint(validation)
    assert min(train + validation) > training.FINAL_TEST_MAX_SEED


def test_observation_api_exposes_no_direct_position_source_or_wall_state():
    """Guard the controller boundary against direct source/boundary-coordinate leakage."""
    forbidden = {
        "x",
        "y",
        "source_x",
        "source_y",
        "distance_to_source",
        "wall_distance",
        "left_wall_distance",
        "right_wall_distance",
    }
    assert forbidden.isdisjoint(Observation.__dataclass_fields__)


def test_controller_does_not_use_world_heading_when_body_sensory_inputs_match():
    bundle = _bundle(qualified=False)
    params = default_parameters(_config())
    a = TaskOptimizedMaleCNSController(bundle, params, require_qualified=False)
    b = TaskOptimizedMaleCNSController(bundle, params, require_qualified=False)
    a.reset(11)
    b.reset(11)
    first = Observation(0.4, 0.4, 0.4, 0.0, -0.3, 0.2, -2.2)
    second = Observation(0.4, 0.4, 0.4, 0.0, -0.3, 0.2, 1.7)
    assert a.act(first) == b.act(second)
    assert a.activity.tolist() == pytest.approx(b.activity.tolist())


def test_v1_matched_control_contract_refuses_fewer_than_eight_rewires(monkeypatch):
    """A 2-rewire run must not be labelable as the frozen v1 matched cohort."""
    config = _config()
    bundle = _bundle()

    monkeypatch.setattr(
        training,
        "optimize_dynamics",
        lambda *args, **kwargs: {
            "trained_validation": {
                "objective": 0.5,
                "success_rate": 0.5,
                "mean_spl": 0.5,
            }
        },
    )
    monkeypatch.setattr(
        training,
        "degree_preserving_rewire",
        lambda bundle, seed, swaps_per_edge: GraphBundle(
            bundle.nodes.copy(),
            bundle.edges.copy(),
            copy.deepcopy(bundle.roles),
            {
                **(bundle.manifest or {}),
                "rewire": {
                    "mixing_complete": True,
                    "accepted_swaps": 0,
                    "target_swaps": 0,
                    "swaps_per_edge": swaps_per_edge,
                },
            },
        ),
    )

    with pytest.raises(ValueError, match="eight|rewire_count|frozen"):
        train_matched_control_cohort(bundle, config, rewire_count=2)


def test_v1_matched_control_contract_refuses_changed_swap_budget(monkeypatch):
    """The null topology budget is part of the frozen comparison, not a CLI tuning knob."""
    config = _config()
    bundle = _bundle()
    monkeypatch.setattr(training, "optimize_dynamics", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="swaps_per_edge|frozen|8"):
        train_matched_control_cohort(bundle, config, rewire_count=8, swaps_per_edge=1)


def test_rewire_completion_cannot_be_asserted_by_boolean_only():
    """A spoofed mixing_complete flag must not override contradictory swap receipts."""
    bundle = _bundle()
    bundle.manifest["rewire"] = {
        "mixing_complete": True,
        "accepted_swaps": 0,
        "target_swaps": 100,
        "attempted_swaps": 1,
        "swaps_per_edge": 8,
    }
    with pytest.raises(RuntimeError, match="rewire|swap|mix"):
        training._require_complete_rewire(bundle)


def test_optimizer_rejects_role_or_topology_mutation_during_training(monkeypatch):
    """The graph identity must be sealed before optimization, not fingerprinted only afterward."""
    config = copy.deepcopy(_config())
    config["optimizer"].update(
        {"population": 4, "generations": 1, "episodes_per_candidate": 2}
    )
    bundle = _bundle()
    calls = 0

    def mutating_evaluate(bundle, parameters, seeds, config, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            bundle.roles["steer_left"].append(1)
        return _summary(seeds)

    monkeypatch.setattr(training, "evaluate_parameters", mutating_evaluate)
    with pytest.raises(RuntimeError, match="graph|topology|role|mutat"):
        optimize_dynamics(bundle, config)


def test_sensory_and_steering_roles_must_be_disjoint_to_make_lesion_interpretable():
    """Direct sensory injection into a steering-role neuron would bypass the edge lesion."""
    bundle = _bundle(qualified=False, overlap_sensory_and_steering=True)
    params = default_parameters(_config())
    with pytest.raises(ValueError, match="overlap|disjoint|steer"):
        TaskOptimizedMaleCNSController(bundle, params, require_qualified=False)


def test_objective_rejects_nonfinite_or_out_of_contract_components():
    config = _config()
    with pytest.raises(ValueError, match="SPL|objective|finite|range"):
        training._episode_objective(
            success=True,
            episode_spl=1.01,
            terminal_progress=0.0,
            config=config,
        )
    with pytest.raises(ValueError, match="progress|objective|finite|range"):
        training._episode_objective(
            success=False,
            episode_spl=0.0,
            terminal_progress=float("nan"),
            config=config,
        )


def test_cem_ties_have_a_canonical_candidate_order(monkeypatch):
    """Equal scores must not delegate scientific selection to an unstable sort tie."""
    config = copy.deepcopy(_config())
    config["optimizer"].update(
        {"population": 4, "generations": 1, "episodes_per_candidate": 2}
    )
    bundle = _bundle()

    monkeypatch.setattr(
        training,
        "evaluate_parameters",
        lambda bundle, parameters, seeds, config, **kwargs: _summary(seeds, objective=0.5),
    )
    report = optimize_dynamics(bundle, config)
    assert report["history"][0]["best_parameters"] == default_parameters(config).to_dict()


def test_training_report_seed_receipts_are_recomputed_not_trusted():
    """final_test_namespace_touched=False is not itself evidence of no peeking."""
    config = _config()
    bundle = _bundle()
    report = _training_report(bundle, config)
    report["train_seed_sha256"] = "forged"
    report["validation_seed_sha256"] = "forged"

    with pytest.raises(ValueError, match="seed|split|receipt|hash"):
        tq.parameters_from_training_report(report, bundle, config)


def test_training_report_development_gate_is_recomputed_not_trusted():
    """A boolean gate cannot be allowed to contradict the sealed validation deltas."""
    config = _config()
    bundle = _bundle()
    report = _training_report(bundle, config)
    report["development_gate_passed"] = True
    report["validation_objective_delta"] = -1.0
    report["validation_success_rate_delta"] = -1.0

    with pytest.raises(ValueError, match="development|gate|validation"):
        tq.parameters_from_training_report(report, bundle, config)


def test_optimizer_report_seals_an_auditable_compute_budget(monkeypatch):
    """Equal-budget claims need a machine-checkable episode/evaluation receipt."""
    config = copy.deepcopy(_config())
    config["optimizer"].update(
        {"population": 4, "generations": 2, "episodes_per_candidate": 3}
    )
    bundle = _bundle()
    monkeypatch.setattr(
        training,
        "evaluate_parameters",
        lambda bundle, parameters, seeds, config, **kwargs: _summary(seeds),
    )
    report = optimize_dynamics(bundle, config)
    receipt = report["optimizer_budget"]
    assert receipt["population"] == 4
    assert receipt["generations"] == 2
    assert receipt["episodes_per_candidate"] == 3
    assert receipt["candidate_evaluations"] == 8
    assert receipt["candidate_episodes"] == 24


def test_matched_controls_use_identical_optimizer_config_for_every_topology(monkeypatch):
    """Positive sentry: every topology must see the same frozen optimizer object."""
    config = _config()
    bundle = _bundle()
    seen_hashes = []

    def fake_optimize(bundle, config, **kwargs):
        seen_hashes.append(canonical_sha256(config))
        return {
            "trained_validation": {
                "objective": 0.5,
                "success_rate": 0.5,
                "mean_spl": 0.5,
            }
        }

    def fake_rewire(bundle, seed, swaps_per_edge):
        manifest = copy.deepcopy(bundle.manifest) or {}
        manifest["rewire"] = {
            "mixing_complete": True,
            "accepted_swaps": 0,
            "target_swaps": 0,
            "attempted_swaps": 0,
            "swaps_per_edge": swaps_per_edge,
        }
        return GraphBundle(
            bundle.nodes.copy(),
            bundle.edges.copy(),
            copy.deepcopy(bundle.roles),
            manifest,
        )

    monkeypatch.setattr(training, "optimize_dynamics", fake_optimize)
    monkeypatch.setattr(training, "degree_preserving_rewire", fake_rewire)
    monkeypatch.setattr(training, "_require_complete_rewire", lambda bundle: None)
    train_matched_control_cohort(bundle, config, rewire_count=8, swaps_per_edge=8)
    assert len(seen_hashes) == 10  # intact + 8 rewires + lesion
    assert len(set(seen_hashes)) == 1
