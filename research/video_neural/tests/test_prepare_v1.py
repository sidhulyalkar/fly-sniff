from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fly_video_neural.prepare_v1 import (
    _validate_behavior_frame_alignment,
    prepare_mc2p_v1,
)


def _touch(root: Path, name: str) -> None:
    (root / name).write_bytes(b"")


def _minimal_pickle_session(root: Path, animal_index: int) -> Path:
    session = root / f"2010{animal_index:02d}_G23xU1_Fly{animal_index}_001"
    session.mkdir()
    _touch(session, "behData_images_camera_1.mp4")
    _touch(session, "2p_dff.mm")
    _touch(session, "sync_indices.pkl")
    _touch(session, "pose_result.pkl")
    return session


def test_prepare_refuses_nonempty_output_before_touching_data(tmp_path: Path):
    root = tmp_path / "mc2p"
    root.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.txt").write_text("immutable")
    with pytest.raises(ValueError, match="non-empty preparation directory"):
        prepare_mc2p_v1(root, output)


def test_prepare_blocks_partial_public_release(tmp_path: Path):
    root = tmp_path / "mc2p"
    root.mkdir()
    _minimal_pickle_session(root, 1)
    with pytest.raises(ValueError, match="complete eight-animal"):
        prepare_mc2p_v1(root, tmp_path / "out")


def test_prepare_requires_explicit_pickle_trust_after_release_completeness_gate(tmp_path: Path):
    root = tmp_path / "mc2p"
    root.mkdir()
    for animal_index in range(1, 9):
        _minimal_pickle_session(root, animal_index)
    with pytest.raises(ValueError, match="trust-upstream-pickle"):
        prepare_mc2p_v1(root, tmp_path / "out")


def test_pose_and_alignment_behavior_frame_counts_must_match(tmp_path: Path):
    alignment = tmp_path / "alignment.npy"
    pose = tmp_path / "pose3d.npy"
    np.save(alignment, np.arange(10, dtype=np.int64), allow_pickle=False)
    np.save(pose, np.zeros((9, 38, 3), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="behavior-frame count mismatch"):
        _validate_behavior_frame_alignment(alignment, pose)


def test_matching_pose_and_alignment_behavior_frame_counts_are_accepted(tmp_path: Path):
    alignment = tmp_path / "alignment.npy"
    pose = tmp_path / "pose3d.npy"
    np.save(alignment, np.arange(10, dtype=np.int64), allow_pickle=False)
    np.save(pose, np.zeros((10, 38, 3), dtype=np.float32), allow_pickle=False)
    assert _validate_behavior_frame_alignment(alignment, pose) == 10
