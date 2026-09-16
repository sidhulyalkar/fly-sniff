from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fly_video_neural.pose_neural import (
    build_session_pose_neural_batch,
    open_dff_memmap,
    root_relative_pose,
    summarize_pose_window,
)


def _windows(path: Path) -> None:
    document = {
        "dataset_id": "mc2p_v1",
        "windows": [
            {
                "sample": {
                    "sample_id": "fly1_001:0-350",
                    "animal_id": "fly1",
                    "session_id": "fly1_001",
                },
                "input_behavior_frames": [0, 300],
                "target_behavior_frames": [300, 350],
                "target_neural_indices": list(range(60, 70)),
            },
            {
                "sample": {
                    "sample_id": "fly1_001:50-400",
                    "animal_id": "fly1",
                    "session_id": "fly1_001",
                },
                "input_behavior_frames": [50, 350],
                "target_behavior_frames": [350, 400],
                "target_neural_indices": list(range(70, 80)),
            },
        ],
    }
    path.write_text(json.dumps(document))


def test_root_relative_pose_zeroes_each_upstream_root():
    pose = np.arange(2 * 38 * 3, dtype=np.float32).reshape(2, 38, 3)
    rooted = root_relative_pose(pose)
    for root in (0, 5, 10, 19, 24, 29):
        assert np.allclose(rooted[:, root], 0.0)


def test_pose_summary_is_fixed_570_dimensions_and_uses_motion():
    pose = np.zeros((8, 38, 3), dtype=np.float32)
    pose[:, 1, 0] = np.arange(8)
    features = summarize_pose_window(pose)
    assert features.shape == (570,)
    assert np.any(features != 0)


def test_dff_memmap_requires_exact_frame_geometry(tmp_path: Path):
    bad = tmp_path / "bad.mm"
    bad.write_bytes(b"abc")
    with pytest.raises(ValueError, match="exact number"):
        open_dff_memmap(bad, side=4)


def test_session_batch_uses_input_pose_and_future_measured_dff(tmp_path: Path):
    windows = tmp_path / "windows.json"
    _windows(windows)
    pose = np.zeros((400, 38, 3), dtype=np.float32)
    pose[:, 1, 0] = np.arange(400, dtype=np.float32)
    np.save(tmp_path / "pose3d.npy", pose)
    dff = np.arange(80 * 4 * 4, dtype=np.float32).reshape(80, 4, 4)
    dff.tofile(tmp_path / "dff.mm")

    receipt = build_session_pose_neural_batch(
        windows,
        tmp_path / "pose3d.npy",
        tmp_path / "dff.mm",
        tmp_path / "batch.npz",
        tmp_path / "receipt.json",
        dff_side=4,
    )
    with np.load(tmp_path / "batch.npz", allow_pickle=False) as batch:
        assert batch["features"].shape == (2, 570)
        assert batch["targets"].shape == (2, 16)
        expected = dff[60:70].mean(axis=0).reshape(-1)
        assert np.allclose(batch["targets"][0], expected)
        assert batch["animal_ids"].tolist() == ["fly1", "fly1"]
        assert batch["session_ids"].tolist() == ["fly1_001", "fly1_001"]
    assert receipt["target_contract"]["evidence_class"] == "measured_neural_activity"
    pose_contract = receipt["pose_feature_contract"]
    assert pose_contract["uses_input_behavior_frames_only"] is True
    assert pose_contract["root_group_joint_partition_source"] == "MC2P augmentation.RootSet"
    assert "project-defined" in pose_contract["three_dimensional_application"]
    assert "project-defined" in pose_contract["statistics_origin"]


def test_session_batch_rejects_future_neural_index_beyond_dff(tmp_path: Path):
    windows = tmp_path / "windows.json"
    _windows(windows)
    document = json.loads(windows.read_text())
    document["windows"][0]["target_neural_indices"] = [999]
    windows.write_text(json.dumps(document))
    np.save(tmp_path / "pose3d.npy", np.zeros((400, 38, 3), dtype=np.float32))
    np.zeros((80, 4, 4), dtype=np.float32).tofile(tmp_path / "dff.mm")
    with pytest.raises(ValueError, match="out of bounds"):
        build_session_pose_neural_batch(
            windows,
            tmp_path / "pose3d.npy",
            tmp_path / "dff.mm",
            tmp_path / "batch.npz",
            tmp_path / "receipt.json",
            dff_side=4,
        )
