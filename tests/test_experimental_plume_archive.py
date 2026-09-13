from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest

from fly_sniff.experimental_plume import load_experimental_plume_config
from fly_sniff.experimental_plume_archive import inspect_experimental_plume_archive
from fly_sniff.experimental_plume_cues import crosswind_gradient, published_motion_cue
from fly_sniff.experimental_plume_reproduction import compute_cue_moments, reproduce_complex_cues


def _tiny_document():
    document = copy.deepcopy(load_experimental_plume_config())
    document["smooth"]["dataset"] = "/dataset2"
    document["smooth"]["expected_shape"] = [4, 3, 2]
    document["smooth"]["source_fps"] = 15.0
    document["complex"]["dataset"] = "/scratch/frames"
    document["complex"]["expected_shape"] = [5, 3, 2]
    document["complex"]["intensity_scale"] = 1.0
    return document


def test_smooth_archive_inspection_reads_real_dataset_shape(tmp_path):
    document = _tiny_document()
    source = tmp_path / "smooth.h5"
    with h5py.File(source, "w") as archive:
        archive.create_dataset("dataset2", data=np.zeros((4, 3, 2), dtype=np.uint8))
    report = inspect_experimental_plume_archive(source, "smooth", document)
    assert report["status"] == "archive_qualified_for_publication_reproduction"
    assert report["source_dataset"]["shape"] == [4, 3, 2]
    assert report["native_time_basis"]["fps"] == 15.0
    assert report["native_time_basis"]["file_internal_timing"] is False


def test_complex_archive_uses_explicit_file_timestamps(tmp_path):
    document = _tiny_document()
    source = tmp_path / "complex.nwb"
    with h5py.File(source, "w") as archive:
        scratch = archive.create_group("scratch")
        scratch.create_dataset("frames", data=np.zeros((5, 3, 2), dtype=np.float32))
        scratch.create_dataset("timestamps", data=np.arange(5, dtype=np.float64) / 60.0)
    report = inspect_experimental_plume_archive(source, "complex", document)
    assert report["status"] == "archive_qualified"
    assert report["native_time_basis"]["kind"] == "explicit_timestamps"
    assert report["native_time_basis"]["median_fps"] == pytest.approx(60.0)
    assert report["native_sensory_timing_allowed"] is True


def test_complex_archive_without_time_basis_fails_closed(tmp_path):
    document = _tiny_document()
    source = tmp_path / "complex-no-time.nwb"
    with h5py.File(source, "w") as archive:
        scratch = archive.create_group("scratch")
        scratch.create_dataset("frames", data=np.zeros((5, 3, 2), dtype=np.float32))
    report = inspect_experimental_plume_archive(source, "complex", document)
    assert report["status"] == "blocked_missing_native_time_basis"
    assert report["native_time_basis"]["kind"] == "unresolved"
    assert report["native_sensory_timing_allowed"] is False


def test_archive_inspection_rejects_wrong_internal_shape(tmp_path):
    document = _tiny_document()
    source = tmp_path / "wrong.h5"
    with h5py.File(source, "w") as archive:
        archive.create_dataset("dataset2", data=np.zeros((4, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape mismatch"):
        inspect_experimental_plume_archive(source, "smooth", document)


def test_streaming_cue_moments_match_independent_batch_calculation(tmp_path):
    values = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
    source = tmp_path / "frames.h5"
    with h5py.File(source, "w") as archive:
        dataset = archive.create_dataset("frames", data=values)
        summary = compute_cue_moments(dataset)
    gradients = np.stack([crosswind_gradient(frame) for frame in values])
    motions = np.stack(
        [published_motion_cue(values[i - 1], values[i], values[i + 1]) for i in range(1, 4)]
    )
    np.testing.assert_allclose(summary["gradient_mean"], gradients.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(summary["gradient_std"], gradients.std(axis=0), rtol=1e-6)
    np.testing.assert_allclose(summary["motion_mean"], motions.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(summary["motion_std"], motions.std(axis=0), rtol=1e-6)
    assert summary["gradient_samples"] == 5
    assert summary["motion_samples"] == 3


def test_complex_reproduction_remains_pending_reference_comparison(tmp_path):
    document = _tiny_document()
    source = tmp_path / "complex.nwb"
    values = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
    with h5py.File(source, "w") as archive:
        scratch = archive.create_group("scratch")
        scratch.create_dataset("frames", data=values)
        scratch.create_dataset("timestamps", data=np.arange(5, dtype=np.float64) / 60.0)
    receipt = reproduce_complex_cues(
        source,
        tmp_path / "summary.npz",
        tmp_path / "receipt.json",
        document,
    )
    assert receipt["status"] == "computed_pending_reference_comparison"
    assert receipt["reference_comparison_status"] == "not_run_no_reference_summary_supplied"
    assert receipt["gradient_samples"] == 5
    assert receipt["motion_samples"] == 3
    assert receipt["controller_access"] is False
    assert receipt["navigation_performance_used"] is False
