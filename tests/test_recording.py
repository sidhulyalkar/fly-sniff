import json

import pytest

from fly_sniff.controllers import BilateralProxyController
from fly_sniff.env import Observation
from fly_sniff.recorded_showcase import _validate_social_contract
from fly_sniff.recording import build_recording, load_recording, write_recording


def _odor_with_body_wind(*, heading: float) -> Observation:
    return Observation(
        left_odor=0.4,
        right_odor=0.4,
        mean_odor=0.4,
        odor_delta=0.0,
        wind_x_body=0.0,
        wind_y_body=-0.7,
        heading=heading,
    )


def test_proxy_steers_from_body_frame_wind_not_world_heading():
    first = BilateralProxyController()
    second = BilateralProxyController()
    first.reset(7)
    second.reset(7)

    first_turn = first.act(_odor_with_body_wind(heading=0.0)).turn
    second_turn = second.act(_odor_with_body_wind(heading=2.4)).turn

    assert first_turn == pytest.approx(second_turn)
    assert first_turn > 0.0


def test_recording_is_elapsed_time_aligned_and_hash_verified(tmp_path):
    bundle = build_recording(seed=11, sim_seconds=0.15, plume_points=12)
    payload = bundle["recording"]
    frames = payload["frames"]

    assert frames[0]["t"] == 0.0
    assert frames[1]["t"] == pytest.approx(payload["dt"])
    assert frames[0]["plume_t"] > frames[0]["t"]
    for expected_step, frame in enumerate(frames):
        assert frame["step"] == expected_step
        assert frame["t"] == pytest.approx(expected_step * payload["dt"])
    assert len(payload["controllers"]) == 2
    assert "NOT A MALECNS RESULT" in payload["claim_boundary"]

    path = write_recording(tmp_path / "episode.json", bundle)
    loaded = load_recording(path)
    assert loaded["recording_sha256"] == bundle["recording_sha256"]
    _validate_social_contract(loaded["recording"])


def test_recording_seals_model_contract_and_configs():
    bundle = build_recording(seed=13, sim_seconds=0.10, plume_points=8)
    payload = bundle["recording"]

    assert payload["model_contract"]["mathematical_model"] == "docs/MATHEMATICAL_MODEL.md"
    assert payload["model_contract"]["plume_model"] == "stochastic-puff-2d-v1"
    assert payload["model_contract"]["sensor_model"] == "bilateral-phenomenological-v1"
    assert "physical left/right antenna sample coordinates" in payload["model_contract"][
        "causal_replay_chain"
    ]
    assert "explicit modeled graph-role drive when using the neural controller" in payload[
        "model_contract"
    ]["causal_replay_chain"]
    assert payload["config"]["arena"]["dt"] == pytest.approx(payload["dt"])
    assert payload["config"]["plume"]["wind_speed"] > 0.0
    assert payload["config"]["sensor"]["adaptation_tau"] > 0.0


def test_recording_sensor_trace_is_synchronized_to_observation():
    bundle = build_recording(seed=15, sim_seconds=0.10, plume_points=8)
    payload = bundle["recording"]

    for frame in payload["frames"]:
        for agent in frame["agents"]:
            trace = agent["sensor_trace"]
            obs = agent["observation"]
            assert trace["signal_kind"] == "modeled_antenna_transduction"
            assert trace["left"]["response"] == pytest.approx(obs["left_odor"])
            assert trace["right"]["response"] == pytest.approx(obs["right_odor"])
            assert trace["left"]["concentration"] >= 0.0
            assert trace["right"]["concentration"] >= 0.0


def test_recording_marks_sampled_plume_frames_as_incomplete():
    bundle = build_recording(seed=17, sim_seconds=0.10, plume_points=8)
    payload = bundle["recording"]

    assert payload["plume_recording"]["max_recorded_points"] == 8
    assert payload["plume_recording"]["puff_mass"] > 0.0
    assert any(not frame["plume_snapshot"]["complete"] for frame in payload["frames"])
    for frame in payload["frames"]:
        meta = frame["plume_snapshot"]
        assert meta["recorded_count"] == len(frame["plume"])
        assert meta["recorded_count"] <= meta["full_count"]
        assert 0.0 < meta["sample_fraction"] <= 1.0


def test_recording_can_capture_complete_plume_for_exact_density_replay():
    bundle = build_recording(seed=19, sim_seconds=0.10, plume_points=600)
    payload = bundle["recording"]

    assert payload["plume_recording"]["max_model_puffs"] == 600
    assert payload["plume_recording"]["max_recorded_points"] == 600
    for frame in payload["frames"]:
        meta = frame["plume_snapshot"]
        assert meta["complete"]
        assert meta["recorded_count"] == meta["full_count"] == len(frame["plume"])
        assert meta["sample_fraction"] == pytest.approx(1.0)


def test_recording_rejects_tampering(tmp_path):
    bundle = build_recording(seed=12, sim_seconds=0.10, plume_points=8)
    path = write_recording(tmp_path / "episode.json", bundle)
    raw = json.loads(path.read_text())
    raw["recording"]["arena"]["source_x"] += 1.0
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_recording(path)
