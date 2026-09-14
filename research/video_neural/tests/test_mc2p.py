from __future__ import annotations

from pathlib import Path

import pytest

from fly_video_neural.mc2p import build_manifest, inspect_session, parse_session_identity


def _touch(root: Path, name: str) -> None:
    (root / name).write_bytes(b"")


def _session(root: Path, name: str, *, roi: bool = False) -> Path:
    path = root / name
    path.mkdir()
    _touch(path, "behData_images_camera_1.mp4")
    _touch(path, "sync_indices.pkl")
    _touch(path, "2p_dff.mm")
    _touch(path, "pose_result.pkl")
    _touch(path, "rest.npy")
    if roi:
        _touch(path, "roi_traces.npy")
    return path


def test_session_identity_groups_trials_under_same_animal():
    animal, trial = parse_session_identity("201008_G23xU1_Fly1_006")
    assert animal == "201008_G23xU1_Fly1"
    assert trial == "006"


def test_inspector_prefers_roi_target_only_when_present(tmp_path: Path):
    image_session = inspect_session(_session(tmp_path, "201008_G23xU1_Fly1_001"))
    roi_session = inspect_session(_session(tmp_path, "201014_G23xU1_Fly2_002", roi=True))
    assert image_session.neural_representation == "two_photon_dff_imaging"
    assert image_session.roi_traces is None
    assert roi_session.neural_representation == "segmented_roi_traces"
    assert roi_session.roi_traces is not None


def test_manifest_counts_animals_not_trials(tmp_path: Path):
    _session(tmp_path, "201008_G23xU1_Fly1_001")
    _session(tmp_path, "201008_G23xU1_Fly1_002")
    _session(tmp_path, "201014_G23xU1_Fly2_001")
    manifest = build_manifest(tmp_path)
    assert manifest["animal_count"] == 2
    assert manifest["session_count"] == 3
    assert len(manifest["manifest_sha256"]) == 64


def test_missing_sync_blocks_session(tmp_path: Path):
    path = tmp_path / "201008_G23xU1_Fly1_001"
    path.mkdir()
    _touch(path, "behData_images_camera_1.mp4")
    _touch(path, "2p_dff.mm")
    with pytest.raises(ValueError, match="synchronization"):
        inspect_session(path)


def test_inspection_never_deserializes_pickle(tmp_path: Path):
    path = _session(tmp_path, "201008_G23xU1_Fly1_001")
    (path / "sync_indices.pkl").write_text("this is deliberately not a pickle")
    manifest = build_manifest(tmp_path)
    assert manifest["session_count"] == 1
    assert "not deserialized" in manifest["pickle_policy"]
