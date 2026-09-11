import pytest

from fly_sniff.looming import LoomingStimulusConfig, generate_looming_frames


def test_approach_generates_positive_expansion_drive():
    config = LoomingStimulusConfig(frames=40, dt=0.02, initial_distance=2.0, speed=1.0)
    frames = generate_looming_frames(config)
    drives = [row["inputs"]["loom_left"] + row["inputs"]["loom_right"] for row in frames]
    assert drives[0] == 0.0
    assert max(drives[1:]) > 0.0
    assert frames[-1]["stimulus"]["distance"] < frames[0]["stimulus"]["distance"]


def test_recede_is_negative_control_for_positive_expansion_channel():
    config = LoomingStimulusConfig(
        frames=30,
        dt=0.02,
        initial_distance=2.0,
        min_distance=0.1,
        speed=0.5,
        mode="recede",
    )
    frames = generate_looming_frames(config)
    drives = [row["inputs"]["loom_left"] + row["inputs"]["loom_right"] for row in frames]
    assert max(drives) == pytest.approx(0.0)


def test_left_azimuth_lateralizes_abstract_drive_leftward():
    config = LoomingStimulusConfig(
        frames=20,
        dt=0.02,
        initial_distance=1.0,
        speed=0.5,
        azimuth_deg=-45.0,
    )
    frames = generate_looming_frames(config)
    active = next(row for row in frames if row["inputs"]["loom_left"] > 0.0)
    assert active["inputs"]["loom_left"] > active["inputs"]["loom_right"]


def test_right_azimuth_lateralizes_abstract_drive_rightward():
    config = LoomingStimulusConfig(
        frames=20,
        dt=0.02,
        initial_distance=1.0,
        speed=0.5,
        azimuth_deg=45.0,
    )
    frames = generate_looming_frames(config)
    active = next(row for row in frames if row["inputs"]["loom_right"] > 0.0)
    assert active["inputs"]["loom_right"] > active["inputs"]["loom_left"]


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="azimuth"):
        generate_looming_frames(LoomingStimulusConfig(azimuth_deg=120.0))
