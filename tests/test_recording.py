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
    assert payload["config"]["arena"]["dt"] == pytest.approx(payload["dt"])
    assert payload["config"]["plume"]["wind_speed"] > 0.0
    assert payload["config"]["sensor"]["adaptation_tau"] > 0.0


def test_recording_rejects_tampering(tmp_path):
    bundle = build_recording(seed=12, sim_seconds=0.10, plume_points=8)
    path = write_recording(tmp_path / "episode.json", bundle)
    raw = json.loads(path.read_text())
    raw["recording"]["arena"]["source_x"] += 1.0
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_recording(path)
