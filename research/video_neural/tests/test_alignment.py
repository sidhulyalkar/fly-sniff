from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fly_video_neural.alignment import (
    TARGET_NEURAL_BOUNDARY_POLICY,
    load_safe_alignment,
    materialize_session_windows,
    materialize_windows,
)
from fly_video_neural.mc2p import inspect_session


def _session(tmp_path: Path):
    path = tmp_path / "201008_G23xU1_Fly1_001"
    path.mkdir()
    for name in ("behData_images_camera_1.mp4", "sync_indices.pkl", "2p_dff.mm", "pose_result.pkl"):
        (path / name).write_bytes(b"")
    return inspect_session(path)


def test_safe_alignment_rejects_pickle_even_if_upstream_uses_it(tmp_path: Path):
    path = tmp_path / "sync_indices.pkl"
    path.write_bytes(b"not loaded")
    with pytest.raises(ValueError, match="non-pickled"):
        load_safe_alignment(path)


def test_safe_alignment_rejects_nonmonotonic_mapping(tmp_path: Path):
    path = tmp_path / "indices.npy"
    np.save(path, np.array([0, 1, 3, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="monotonic"):
        load_safe_alignment(path)


def test_windows_use_future_nonoverlapping_target_and_unique_neural_indices(tmp_path: Path):
    session = _session(tmp_path)
    alignment = np.repeat(np.arange(80, dtype=np.int64), 5)
    windows = materialize_windows(
        session, alignment, video_fps=100, history_s=3, horizon_s=0.5, stride_s=0.5
    )
    assert len(windows) == 2
    first = windows[0]
    assert first.input_behavior_frames == (0, 300)
    assert first.target_behavior_frames == (300, 350)
    assert first.sample.input_end_s == first.sample.target_start_s == 3.0
    assert first.last_input_neural_index == 59
    assert first.target_neural_indices == tuple(range(60, 70))
    assert all(index > first.last_input_neural_index for index in first.target_neural_indices)
    second = windows[1]
    assert second.input_behavior_frames == (50, 350)
    assert second.target_behavior_frames == (350, 400)


def test_shared_boundary_neural_index_is_excluded_from_future_target(tmp_path: Path):
    session = _session(tmp_path)
    alignment = np.asarray([0, 1, 1, 2, 3, 4], dtype=np.int64)
    windows = materialize_windows(
        session,
        alignment,
        video_fps=2,
        history_s=1,
        horizon_s=1,
        stride_s=1,
    )
    first = windows[0]
    assert first.input_behavior_frames == (0, 2)
    assert first.target_behavior_frames == (2, 4)
    assert first.last_input_neural_index == 1
    assert first.target_neural_indices == (2,)


def test_window_manifest_records_strict_future_neural_boundary_policy(tmp_path: Path):
    session = _session(tmp_path)
    alignment_path = tmp_path / "indices.npy"
    np.save(alignment_path, np.repeat(np.arange(80, dtype=np.int64), 5), allow_pickle=False)
    report = materialize_session_windows(
        session.path,
        alignment_path,
        video_fps=100,
        history_s=3,
        horizon_s=0.5,
        stride_s=0.5,
    )
    assert report["target_neural_boundary_policy"] == TARGET_NEURAL_BOUNDARY_POLICY
    assert all(
        min(row["target_neural_indices"]) > row["last_input_neural_index"]
        for row in report["windows"]
    )


def test_short_session_yields_no_windows(tmp_path: Path):
    session = _session(tmp_path)
    alignment = np.arange(100, dtype=np.int64)
    assert materialize_windows(session, alignment) == []
