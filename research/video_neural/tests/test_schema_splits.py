from __future__ import annotations

import pytest

from fly_video_neural.schema import SampleWindow
from fly_video_neural.splits import make_animal_disjoint_splits, validate_animal_disjoint_splits


def _window(animal: str, session: str, index: int) -> SampleWindow:
    return SampleWindow(
        dataset_id="mc2p_v1",
        sample_id=f"{animal}-{index}",
        animal_id=animal,
        session_id=session,
        input_start_s=float(index * 4),
        input_end_s=float(index * 4 + 3),
        target_start_s=float(index * 4 + 3),
        target_end_s=float(index * 4 + 3.5),
        video_fps=100.0,
        neural_rate_hz=10.0,
        pose_available=True,
        context_available=True,
    )


def test_future_target_cannot_overlap_input():
    with pytest.raises(ValueError, match="temporal overlap"):
        SampleWindow(
            dataset_id="mc2p_v1",
            sample_id="bad",
            animal_id="a",
            session_id="s",
            input_start_s=0,
            input_end_s=3,
            target_start_s=2.9,
            target_end_s=3.4,
            video_fps=100,
        )


def test_split_is_animal_disjoint_and_deterministic():
    windows = [_window(f"fly{i}", f"session{i}", 0) for i in range(8)]
    first = make_animal_disjoint_splits(windows, seed=17)
    second = make_animal_disjoint_splits(windows, seed=17)
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}


def test_session_crossing_animal_splits_is_rejected():
    windows = [_window("flyA", "shared", 0), _window("flyB", "shared", 1), _window("flyC", "c", 2)]
    assignment = {"flyA": "train", "flyB": "test", "flyC": "validation"}
    with pytest.raises(ValueError, match="session leakage"):
        validate_animal_disjoint_splits(windows, assignment)
