import pandas as pd
import pytest

from fly_sniff import training_redteam as redteam
from fly_sniff.env import Observation
from fly_sniff.graph import GraphBundle
from fly_sniff.training import FINAL_TEST_MAX_SEED, default_parameters, make_training_seed_split


def _config():
    from fly_sniff.training import load_training_config

    return load_training_config("configs/task_optimization_v1.json")


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
        {"qualification_status": "candidate", "dataset": "synthetic-redteam"},
    )


def test_diagnostic_seed_namespace_is_separate_from_training_validation_and_final():
    config = _config()
    diagnostic = redteam.make_diagnostic_seeds(config, count=12)
    train, validation = make_training_seed_split(config)
    assert len(set(diagnostic)) == 12
    assert min(diagnostic) > FINAL_TEST_MAX_SEED
    assert set(diagnostic).isdisjoint(train)
    assert set(diagnostic).isdisjoint(validation)


def test_sensory_masks_change_only_explicit_controller_channels():
    observation = Observation(
        left_odor=0.8,
        right_odor=0.2,
        mean_odor=0.5,
        odor_delta=-0.6,
        wind_x_body=-0.7,
        wind_y_body=0.3,
        heading=1.25,
    )
    odor_off = redteam.mask_observation(observation, odor_scale=0.0, wind_scale=1.0)
    assert odor_off.left_odor == 0.0
    assert odor_off.right_odor == 0.0
    assert odor_off.mean_odor == 0.0
    assert odor_off.odor_delta == 0.0
    assert odor_off.wind_x_body == pytest.approx(observation.wind_x_body)
    assert odor_off.wind_y_body == pytest.approx(observation.wind_y_body)
    assert odor_off.heading == pytest.approx(observation.heading)

    wind_off = redteam.mask_observation(observation, odor_scale=1.0, wind_scale=0.0)
    assert wind_off.wind_x_body == 0.0
    assert wind_off.wind_y_body == 0.0
    assert wind_off.mean_odor == pytest.approx(observation.mean_odor)
    assert wind_off.heading == pytest.approx(observation.heading)


def test_shortcut_battery_is_diagnostic_only_and_contains_geometry_attacks(monkeypatch):
    config = _config()
    bundle = _bundle()
    parameters = default_parameters(config)
    monkeypatch.setattr(
        redteam,
        "parameters_from_training_report",
        lambda report, bundle, config: parameters,
    )

    def fake_episode(
        bundle,
        parameters,
        seed,
        *,
        arena,
        plume,
        sensors,
        odor_scale,
        wind_scale,
    ):
        success = bool(odor_scale > 0.0 and wind_scale > 0.0)
        return {
            "seed": int(seed),
            "success": success,
            "spl": 0.5 if success else 0.0,
            "path_length": 1.0,
            "final_distance": 1.0,
        }

    monkeypatch.setattr(redteam, "_run_diagnostic_episode", fake_episode)
    report = redteam.run_shortcut_redteam(
        bundle,
        {
            "audit_receipt_sha256": "audit",
            "trained_parameter_sha256": "parameters",
        },
        config,
        episode_count=2,
    )

    assert report["status"] == "diagnostic_only_not_a_gate"
    assert "passed" not in report
    assert "threshold" not in report
    assert set(report["scenarios"]) == {
        "normal",
        "odor_clamped",
        "wind_clamped",
        "odor_and_wind_clamped",
        "source_crosswind_low",
        "source_crosswind_high",
        "mirrored_world",
    }
    mirrored = report["scenarios"]["mirrored_world"]
    assert mirrored["plume"]["wind_speed"] < 0.0
    assert mirrored["arena"]["source_x"] > mirrored["arena"]["start_x"]
    assert report["comparisons"]["normal_minus_odor_clamped_success_rate"] == 1.0
