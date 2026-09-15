from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from fly_sniff.physical_sensory import (
    AxisRole,
    BilateralSensorGeometry,
    PhysicalPlumeCalibration,
    PhysicalPlumeStatus,
    SensorGeometryStatus,
    SensorOffset,
    assemble_physical_sensory_transform,
    bilinear_sample,
    body_sensor_world_positions,
    causal_native_frame_index,
    sample_bilateral_concentration,
    world_to_archive_pixel,
)

POLICY_PATH = Path("configs/program_a_physical_sensory_policy_v1.json")


def _sha(value: int) -> str:
    return f"{value:064x}"


def _blocked_plume() -> PhysicalPlumeCalibration:
    return PhysicalPlumeCalibration(
        calibration_id="smooth-plume-physical-v1",
        source_doi="10.5061/dryad.g27mq71",
        source_file="10302017_10cms_bounded_2.h5",
        source_dataset_path="/dataset2",
        source_shape=(3600, 406, 216),
        native_fps=15.0,
        publication_authority="Alvarez-Salvado et al. 2018 doi:10.7554/eLife.37815",
        publication_field_downwind_mm_max=300.0,
        publication_field_crosswind_halfspan_mm_max=80.0,
        publication_nominal_mm_per_pixel=0.74,
    )


def _pass_plume() -> PhysicalPlumeCalibration:
    return dataclasses.replace(
        _blocked_plume(),
        source_sha256=_sha(1),
        source_bytes_verified=True,
        axis0_role=AxisRole.DOWNWIND,
        axis1_role=AxisRole.CROSSWIND,
        axis0_positive_direction=1,
        axis1_positive_direction=1,
        axis0_mm_per_pixel=0.74,
        axis1_mm_per_pixel=0.74,
        source_index_axis0=0.0,
        source_index_axis1=10.0,
        archive_spatial_transform="verified crop-only transform preserving native pixel scale",
    )


def _blocked_geometry() -> BilateralSensorGeometry:
    return BilateralSensorGeometry(
        geometry_id="dmel-bilateral-olfactory-v1",
        species="Drosophila melanogaster",
        sex_context="not yet frozen",
        preparation_context="scale-calibrated frontal anatomy image",
        anatomy_authority="Jurgens et al. 2024 doi:10.1093/genetics/iyae129",
        landmark_definition="centroid of the third antennal segment/funiculus on each side",
        projection_convention="body centered; +forward anterior, +lateral left",
        measurement_method="manual landmark measurement from scale-calibrated source image",
    )


def _pass_geometry() -> BilateralSensorGeometry:
    return dataclasses.replace(
        _blocked_geometry(),
        sex_context="adult wild-type; source preparation sex retained in measurement receipt",
        measurement_source_sha256=_sha(2),
        measurement_source_verified=True,
        left_offset=SensorOffset(forward_mm=0.12, lateral_mm=0.09),
        right_offset=SensorOffset(forward_mm=0.11, lateral_mm=-0.08),
        uncertainty_mm=0.01,
    )


def test_committed_policy_preserves_real_blockers_and_resolved_axis_roles() -> None:
    policy = json.loads(POLICY_PATH.read_text())
    assert policy["navigation_performance_used"] is False
    plume = policy["smooth_plume_evidence"]
    assert plume["status"] == "BLOCKED_SOURCE_BYTES_UNVERIFIED"
    assert plume["source_sha256"] is None
    assert plume["array_axis_mapping"]["raw_axis0"] == "downwind"
    assert plume["array_axis_mapping"]["raw_axis1"] == "crosswind"
    assert plume["array_axis_mapping"]["authority_blob_sha"] == (
        "6efc37b16155622df3986d4dc50fed423173700d"
    )
    assert plume["array_axis_positive_directions"] is None
    assert plume["source_index_in_archive"] is None
    geometry = policy["bilateral_sensor_geometry"]
    assert geometry["status"] == "BLOCKED_REQUIRES_ANATOMY_MEASUREMENT"
    assert geometry["left_offset_mm"] is None
    assert geometry["right_offset_mm"] is None
    assert "ArenaConfig.antenna_separation = 0.12" in geometry["forbidden_authority"]


def test_publication_scale_does_not_promote_unverified_source() -> None:
    plume = _blocked_plume()
    plume.validate()
    assert plume.publication_nominal_mm_per_pixel == pytest.approx(0.74)
    assert plume.status is PhysicalPlumeStatus.BLOCKED_SOURCE_BYTES_UNVERIFIED


def test_verified_bytes_without_axis_authority_remain_blocked() -> None:
    plume = dataclasses.replace(
        _blocked_plume(), source_sha256=_sha(1), source_bytes_verified=True
    )
    plume.validate()
    assert plume.status is PhysicalPlumeStatus.BLOCKED_ARRAY_ORIENTATION_UNVERIFIED


def test_array_shape_cannot_be_used_as_implicit_orientation() -> None:
    plume = dataclasses.replace(
        _blocked_plume(),
        source_sha256=_sha(1),
        source_bytes_verified=True,
        axis0_mm_per_pixel=0.74,
        axis1_mm_per_pixel=0.74,
        source_index_axis0=0.0,
        source_index_axis1=10.0,
        archive_spatial_transform="verified crop",
    )
    plume.validate()
    assert plume.status is PhysicalPlumeStatus.BLOCKED_ARRAY_ORIENTATION_UNVERIFIED
    with pytest.raises(ValueError, match="must PASS"):
        world_to_archive_pixel(downwind_mm=10.0, crosswind_mm=0.0, plume=plume)


def test_axis_orientation_changes_physical_plume_identity() -> None:
    plume = _pass_plume()
    swapped = dataclasses.replace(
        plume,
        axis0_role=AxisRole.CROSSWIND,
        axis1_role=AxisRole.DOWNWIND,
    )
    reversed_downwind = dataclasses.replace(plume, axis0_positive_direction=-1)

    assert plume.status is PhysicalPlumeStatus.PASS_PHYSICAL_PLUME
    assert swapped.status is PhysicalPlumeStatus.PASS_PHYSICAL_PLUME
    assert plume.sha256 != swapped.sha256
    assert plume.sha256 != reversed_downwind.sha256


def test_sensor_geometry_remains_blocked_without_anatomy_measurement() -> None:
    geometry = _blocked_geometry()
    geometry.validate()
    assert geometry.status is SensorGeometryStatus.BLOCKED_REQUIRES_ANATOMY_MEASUREMENT
    assert geometry.inter_sensor_distance_mm is None


def test_simulator_geometry_cannot_be_promoted_to_anatomy_authority() -> None:
    with pytest.raises(ValueError, match="simulator geometry"):
        dataclasses.replace(_blocked_geometry(), simulator_assumption_used=True).validate()


def test_left_and_right_offsets_are_independent_not_forced_symmetric() -> None:
    geometry = _pass_geometry()
    geometry.validate()
    assert geometry.status is SensorGeometryStatus.PASS_BILATERAL_SENSOR_GEOMETRY
    assert geometry.left_offset != geometry.right_offset
    assert geometry.left_offset is not None and geometry.right_offset is not None
    assert geometry.left_offset.forward_mm != geometry.right_offset.forward_mm
    assert geometry.inter_sensor_distance_mm is not None


def test_transform_refuses_any_blocked_input() -> None:
    with pytest.raises(ValueError, match="physical plume is not PASS"):
        assemble_physical_sensory_transform(
            _blocked_plume(), _pass_geometry(), transform_id="physical-sensory-v1"
        )
    with pytest.raises(ValueError, match="bilateral sensor geometry is not PASS"):
        assemble_physical_sensory_transform(
            _pass_plume(), _blocked_geometry(), transform_id="physical-sensory-v1"
        )


def test_body_pose_uses_measured_asymmetric_offsets() -> None:
    geometry = _pass_geometry()
    left, right = body_sensor_world_positions(
        body_downwind_mm=10.0,
        body_crosswind_mm=5.0,
        heading_rad=0.0,
        geometry=geometry,
    )
    assert left == pytest.approx((10.12, 5.09))
    assert right == pytest.approx((10.11, 4.92))


def test_world_to_archive_mapping_respects_frozen_axis_roles() -> None:
    plume = _pass_plume()
    axis0, axis1 = world_to_archive_pixel(
        downwind_mm=7.4,
        crosswind_mm=-3.7,
        plume=plume,
    )
    assert axis0 == pytest.approx(10.0)
    assert axis1 == pytest.approx(5.0)


def test_bilinear_sampling_is_deterministic_and_subpixel() -> None:
    axis0 = np.arange(4, dtype=np.float64)[:, None]
    axis1 = np.arange(5, dtype=np.float64)[None, :]
    frame = 10.0 * axis0 + 2.0 * axis1

    sampled = bilinear_sample(frame, 1.25, 2.5)
    assert sampled == pytest.approx(17.5)
    assert sampled != float(frame[1, 2])


def test_bilinear_sampling_fails_closed_outside_archive() -> None:
    frame = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        bilinear_sample(frame, -0.01, 1.0)
    with pytest.raises(ValueError, match="outside"):
        bilinear_sample(frame, 1.0, 4.01)


def test_causal_native_frame_hold_never_uses_a_future_measurement() -> None:
    fps = 15.0
    assert causal_native_frame_index(time_seconds=0.0, native_fps=fps, frame_count=100) == 0
    assert causal_native_frame_index(time_seconds=0.065, native_fps=fps, frame_count=100) == 0
    assert causal_native_frame_index(time_seconds=0.067, native_fps=fps, frame_count=100) == 1
    assert causal_native_frame_index(time_seconds=1.0, native_fps=fps, frame_count=100) == 15


def test_end_to_end_pair_sampling_binds_exact_calibrations() -> None:
    plume = _pass_plume()
    geometry = _pass_geometry()
    transform = assemble_physical_sensory_transform(
        plume, geometry, transform_id="physical-sensory-v1"
    )
    frame = np.add.outer(np.arange(406, dtype=np.float64), np.arange(216, dtype=np.float64))

    left, right = sample_bilateral_concentration(
        frame,
        body_downwind_mm=7.4,
        body_crosswind_mm=0.0,
        heading_rad=0.0,
        plume=plume,
        geometry=geometry,
        transform=transform,
    )
    assert left != right

    changed_geometry = dataclasses.replace(
        geometry,
        left_offset=SensorOffset(forward_mm=0.13, lateral_mm=0.09),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        sample_bilateral_concentration(
            frame,
            body_downwind_mm=7.4,
            body_crosswind_mm=0.0,
            heading_rad=0.0,
            plume=plume,
            geometry=changed_geometry,
            transform=transform,
        )


def test_source_coordinates_cannot_be_enabled_for_controller() -> None:
    plume = _pass_plume()
    geometry = _pass_geometry()
    transform = assemble_physical_sensory_transform(
        plume, geometry, transform_id="physical-sensory-v1"
    )
    with pytest.raises(ValueError, match="source coordinates"):
        dataclasses.replace(transform, source_coordinates_visible_to_controller=True).validate()


def test_physical_artifact_hashes_change_with_scientific_transform() -> None:
    plume = _pass_plume()
    geometry = _pass_geometry()
    changed_plume = dataclasses.replace(plume, source_index_axis1=11.0)
    changed_geometry = dataclasses.replace(geometry, uncertainty_mm=0.02)

    assert plume.sha256 != changed_plume.sha256
    assert geometry.sha256 != changed_geometry.sha256
