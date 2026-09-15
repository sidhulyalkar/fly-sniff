from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from fly_video_neural.mc2p_legacy import convert_legacy_pickle


def _dump(path: Path, value) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def test_pickle_conversion_requires_explicit_trust_before_deserialization(tmp_path: Path):
    source = tmp_path / "sync_indices.pkl"
    _dump(source, np.array([0, 0, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="trust-upstream-pickle"):
        convert_legacy_pickle(source, tmp_path / "indices.npy", tmp_path / "receipt.json", kind="alignment")
    assert not (tmp_path / "indices.npy").exists()


def test_alignment_conversion_hashes_and_normalizes_output(tmp_path: Path):
    source = tmp_path / "sync_indices.pkl"
    _dump(source, {"sync_indices": np.array([0.0, 0.0, 1.0, 2.0])})
    receipt = convert_legacy_pickle(
        source,
        tmp_path / "indices.npy",
        tmp_path / "receipt.json",
        kind="alignment",
        trust_upstream_pickle=True,
    )
    assert np.array_equal(np.load(tmp_path / "indices.npy", allow_pickle=False), [0, 0, 1, 2])
    assert receipt["output_dtype"] == "int64"
    assert len(receipt["source_sha256"]) == len(receipt["output_sha256"]) == 64
    assert json.loads((tmp_path / "receipt.json").read_text())["receipt_sha256"] == receipt["receipt_sha256"]


def test_nonmonotonic_alignment_is_rejected(tmp_path: Path):
    source = tmp_path / "sync_indices.pkl"
    _dump(source, np.array([0, 2, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="monotonic"):
        convert_legacy_pickle(
            source,
            tmp_path / "indices.npy",
            tmp_path / "receipt.json",
            kind="alignment",
            trust_upstream_pickle=True,
        )


def test_pose3d_conversion_requires_exact_keypoint_contract(tmp_path: Path):
    source = tmp_path / "pose_result.pkl"
    _dump(source, {"points3d": np.zeros((12, 38, 3), dtype=np.float64)})
    receipt = convert_legacy_pickle(
        source,
        tmp_path / "pose3d.npy",
        tmp_path / "receipt.json",
        kind="pose3d",
        trust_upstream_pickle=True,
    )
    pose = np.load(tmp_path / "pose3d.npy", allow_pickle=False)
    assert pose.shape == (12, 38, 3)
    assert pose.dtype == np.float32
    assert receipt["output_shape"] == [12, 38, 3]


def test_unknown_alignment_dict_fails_closed(tmp_path: Path):
    source = tmp_path / "sync_indices.pkl"
    _dump(source, {"mystery": np.arange(5)})
    with pytest.raises(ValueError, match="recognized alignment key"):
        convert_legacy_pickle(
            source,
            tmp_path / "indices.npy",
            tmp_path / "receipt.json",
            kind="alignment",
            trust_upstream_pickle=True,
        )
