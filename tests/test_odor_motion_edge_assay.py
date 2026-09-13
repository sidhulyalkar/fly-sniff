from __future__ import annotations

import copy

import pytest

from fly_sniff.config import ArenaConfig, PlumeConfig, SensorConfig
from fly_sniff.env import FlySniffEnv
from fly_sniff.odor_motion import load_odor_motion_config
from fly_sniff.odor_motion_edge_assay import SensorReplay, run_edge_assay
from fly_sniff.odor_motion_plume_assay import run_plume_assay


def test_sensor_replay_matches_environment_first_observation():
    env = FlySniffEnv(seed=13013)
    env.observe()
    trace = env.sensor_trace()
    replay = SensorReplay(env.arena.dt, env.sensor_config)
    left, right = replay.sample(trace.left.concentration, trace.right.concentration)
    assert left == pytest.approx(trace.left.response, abs=1e-12)
    assert right == pytest.approx(trace.right.response, abs=1e-12)


def test_edge_assay_recovers_every_frozen_direction_and_latency():
    document, motion_config = load_odor_motion_config()
    result = run_edge_assay(
        document["qualification"], motion_config, ArenaConfig(), SensorConfig()
    )
    assert result["status"] == "pass"
    assert result["all_trials_have_directional_estimate"] is True
    assert result["all_qualified_estimates_have_correct_sign"] is True
    assert len(result["trials"]) == 8
    assert all(row["direction_sign_accuracy"] == 1.0 for row in result["trials"])


def test_fixed_plume_assay_is_deterministic_on_small_frozen_grid():
    document, motion_config = load_odor_motion_config()
    qualification = copy.deepcopy(document["qualification"])
    probe = qualification["fixed_plume_probe"]
    probe["seeds"] = [13013]
    probe["duration_s"] = 0.25
    probe["x_positions"] = [6.0]
    probe["y_offsets_from_source"] = [0.0]
    first = run_plume_assay(
        qualification, motion_config, ArenaConfig(), PlumeConfig(), SensorConfig()
    )
    second = run_plume_assay(
        qualification, motion_config, ArenaConfig(), PlumeConfig(), SensorConfig()
    )
    assert first == second
    assert first["probe_count"] == 3
