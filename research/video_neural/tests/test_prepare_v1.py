from __future__ import annotations

from pathlib import Path

import pytest

from fly_video_neural.prepare_v1 import prepare_mc2p_v1


def _touch(root: Path, name: str) -> None:
    (root / name).write_bytes(b"")


def test_prepare_refuses_nonempty_output_before_touching_data(tmp_path: Path):
    root = tmp_path / "mc2p"
    root.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.txt").write_text("immutable")
    with pytest.raises(ValueError, match="non-empty preparation directory"):
        prepare_mc2p_v1(root, output)


def test_prepare_requires_explicit_pickle_trust(tmp_path: Path):
    root = tmp_path / "mc2p"
    root.mkdir()
    session = root / "201008_G23xU1_Fly1_001"
    session.mkdir()
    _touch(session, "behData_images_camera_1.mp4")
    _touch(session, "2p_dff.mm")
    _touch(session, "sync_indices.pkl")
    _touch(session, "pose_result.pkl")
    with pytest.raises(ValueError, match="trust-upstream-pickle"):
        prepare_mc2p_v1(root, tmp_path / "out")
