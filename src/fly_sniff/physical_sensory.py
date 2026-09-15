from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any

import numpy as np

from .freeze import canonical_sha256

_HEX = frozenset("0123456789abcdef")


class AxisRole(str, enum.Enum):
    DOWNWIND = "downwind"
    CROSSWIND = "crosswind"


class PhysicalPlumeStatus(str, enum.Enum):
    BLOCKED_SOURCE_BYTES_UNVERIFIED = "BLOCKED_SOURCE_BYTES_UNVERIFIED"
    BLOCKED_ARRAY_ORIENTATION_UNVERIFIED = "BLOCKED_ARRAY_ORIENTATION_UNVERIFIED"
    PASS_PHYSICAL_PLUME = "PASS_PHYSICAL_PLUME"


class SensorGeometryStatus(str, enum.Enum):
    BLOCKED_REQUIRES_ANATOMY_MEASUREMENT = "BLOCKED_REQUIRES_ANATOMY_MEASUREMENT"
    PASS_BILATERAL_SENSOR_GEOMETRY = "PASS_BILATERAL_SENSOR_GEOMETRY"


class OutOfBoundsRule(str, enum.Enum):
    ERROR = "error"


class TemporalSamplingRule(str, enum.Enum):
    CAUSAL_NATIVE_FRAME_HOLD = "causal_native_frame_hold"


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")


def _required_text(value: str, *, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _finite_optional(value: float | None, *, field: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{field} must be finite when supplied")


@dataclasses.dataclass(frozen=True)
class PhysicalPlumeCalibration:
    """Physical coordinate contract for an archived experimental plume.

    Publication-scale facts may be present while the artifact remains BLOCKED. PASS requires
    verified source bytes plus an explicit array-axis/origin calibration. Array shape alone is
    never accepted as orientation authority.
    """

    calibration_id: str
    source_doi: str
    source_file: str
    source_dataset_path: str
    source_shape: tuple[int, int, int]
    native_fps: float
    publication_authority: str
    publication_field_downwind_mm_max: float
    publication_field_crosswind_halfspan_mm_max: float
    publication_nominal_mm_per_pixel: float
    source_sha256: str | None = None
    source_bytes_verified: bool = False
    axis0_role: AxisRole | None = None
    axis1_role: AxisRole | None = None
    axis0_positive_direction: int | None = None
    axis1_positive_direction: int | None = None
    axis0_mm_per_pixel: float | None = None
    axis1_mm_per_pixel: float | None = None
    source_index_axis0: float | None = None
    source_index_axis1: float | None = None
    archive_spatial_transform: str | None = None
    navigation_performance_used: bool = False
    schema: str = "fly-sniff-physical-plume-calibration-v1"

    @property
    def status(self) -> PhysicalPlumeStatus:
        if not self.source_bytes_verified or self.source_sha256 is None:
            return PhysicalPlumeStatus.BLOCKED_SOURCE_BYTES_UNVERIFIED
        orientation_fields = (
            self.axis0_role,
            self.axis1_role,
            self.axis0_positive_direction,
            self.axis1_positive_direction,
            self.axis0_mm_per_pixel,
            self.axis1_mm_per_pixel,
            self.source_index_axis0,
            self.source_index_axis1,
            self.archive_spatial_transform,
        )
        if any(value is None for value in orientation_fields):
            return PhysicalPlumeStatus.BLOCKED_ARRAY_ORIENTATION_UNVERIFIED
        return PhysicalPlumeStatus.PASS_PHYSICAL_PLUME

    def validate(self) -> None:
        if self.schema != "fly-sniff-physical-plume-calibration-v1":
            raise ValueError(f"unsupported physical plume schema: {self.schema}")
        for field, value in (
            ("calibration_id", self.calibration_id),
            ("source_doi", self.source_doi),
            ("source_file", self.source_file),
            ("source_dataset_path", self.source_dataset_path),
            ("publication_authority", self.publication_authority),
        ):
            _required_text(value, field=field)
        if len(self.source_shape) != 3 or any(value <= 0 for value in self.source_shape):
            raise ValueError("source_shape must be positive (time, axis0, axis1)")
        if not math.isfinite(self.native_fps) or self.native_fps <= 0:
            raise ValueError("native_fps must be positive and finite")
        for field, value in (
            ("publication_field_downwind_mm_max", self.publication_field_downwind_mm_max),
            (
                "publication_field_crosswind_halfspan_mm_max",
                self.publication_field_crosswind_halfspan_mm_max,
            ),
            ("publication_nominal_mm_per_pixel", self.publication_nominal_mm_per_pixel),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be positive and finite")
        if self.navigation_performance_used:
            raise ValueError("physical plume calibration may not use navigation performance")
        if self.source_sha256 is not None:
            _validate_sha256(self.source_sha256, field="source_sha256")
        if self.source_bytes_verified and self.source_sha256 is None:
            raise ValueError("source_bytes_verified requires a concrete source_sha256")

        if (self.axis0_role is None) != (self.axis1_role is None):
            raise ValueError("array axis roles must be supplied together")
        if self.axis0_role is not None and self.axis0_role is self.axis1_role:
            raise ValueError("axis0_role and axis1_role must be distinct")
        for field, value in (
            ("axis0_positive_direction", self.axis0_positive_direction),
            ("axis1_positive_direction", self.axis1_positive_direction),
        ):
            if value is not None and value not in (-1, 1):
                raise ValueError(f"{field} must be -1 or +1")
        for field, value in (
            ("axis0_mm_per_pixel", self.axis0_mm_per_pixel),
            ("axis1_mm_per_pixel", self.axis1_mm_per_pixel),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{field} must be positive and finite when supplied")
        _finite_optional(self.source_index_axis0, field="source_index_axis0")
        _finite_optional(self.source_index_axis1, field="source_index_axis1")
        if self.archive_spatial_transform is not None:
            _required_text(self.archive_spatial_transform, field="archive_spatial_transform")

        if self.status is PhysicalPlumeStatus.PASS_PHYSICAL_PLUME:
            assert self.axis0_role is not None and self.axis1_role is not None
            assert self.axis0_positive_direction is not None
            assert self.axis1_positive_direction is not None
            assert self.axis0_mm_per_pixel is not None
            assert self.axis1_mm_per_pixel is not None
            assert self.source_index_axis0 is not None
            assert self.source_index_axis1 is not None
            if self.axis0_role is self.axis1_role:
                raise ValueError("PASS plume calibration requires one downwind and one crosswind axis")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "calibration_id": self.calibration_id,
            "status": self.status.value,
            "source_doi": self.source_doi,
            "source_file": self.source_file,
            "source_dataset_path": self.source_dataset_path,
            "source_shape": list(self.source_shape),
            "native_fps": self.native_fps,
            "publication_authority": self.publication_authority,
            "publication_field_downwind_mm_max": self.publication_field_downwind_mm_max,
            "publication_field_crosswind_halfspan_mm_max": (
                self.publication_field_crosswind_halfspan_mm_max
            ),
            "publication_nominal_mm_per_pixel": self.publication_nominal_mm_per_pixel,
            "source_sha256": self.source_sha256,
            "source_bytes_verified": self.source_bytes_verified,
            "axis0_role": None if self.axis0_role is None else self.axis0_role.value,
            "axis1_role": None if self.axis1_role is None else self.axis1_role.value,
            "axis0_positive_direction": self.axis0_positive_direction,
            "axis1_positive_direction": self.axis1_positive_direction,
            "axis0_mm_per_pixel": self.axis0_mm_per_pixel,
            "axis1_mm_per_pixel": self.axis1_mm_per_pixel,
            "source_index_axis0": self.source_index_axis0,
            "source_index_axis1": self.source_index_axis1,
            "archive_spatial_transform": self.archive_spatial_transform,
            "navigation_performance_used": self.navigation_performance_used,
            "claim_boundary": [
                "publication scale facts do not establish archived-array orientation or origin",
                "temporal interpolation cannot create measurements above the native acquisition rate",
                "this artifact is physical plume calibration, not a navigation result",
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["calibration_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PhysicalPlumeCalibration:
        calibration = cls(
            calibration_id=str(payload["calibration_id"]),
            source_doi=str(payload["source_doi"]),
            source_file=str(payload["source_file"]),
            source_dataset_path=str(payload["source_dataset_path"]),
            source_shape=tuple(int(value) for value in payload["source_shape"]),
            native_fps=float(payload["native_fps"]),
            publication_authority=str(payload["publication_authority"]),
            publication_field_downwind_mm_max=float(
                payload["publication_field_downwind_mm_max"]
            ),
            publication_field_crosswind_halfspan_mm_max=float(
                payload["publication_field_crosswind_halfspan_mm_max"]
            ),
            publication_nominal_mm_per_pixel=float(
                payload["publication_nominal_mm_per_pixel"]
            ),
            source_sha256=payload.get("source_sha256"),
            source_bytes_verified=bool(payload.get("source_bytes_verified", False)),
            axis0_role=(
                None if payload.get("axis0_role") is None else AxisRole(payload["axis0_role"])
            ),
            axis1_role=(
                None if payload.get("axis1_role") is None else AxisRole(payload["axis1_role"])
            ),
            axis0_positive_direction=payload.get("axis0_positive_direction"),
            axis1_positive_direction=payload.get("axis1_positive_direction"),
            axis0_mm_per_pixel=payload.get("axis0_mm_per_pixel"),
            axis1_mm_per_pixel=payload.get("axis1_mm_per_pixel"),
            source_index_axis0=payload.get("source_index_axis0"),
            source_index_axis1=payload.get("source_index_axis1"),
            archive_spatial_transform=payload.get("archive_spatial_transform"),
            navigation_performance_used=bool(payload.get("navigation_performance_used", False)),
            schema=str(payload.get("schema", "fly-sniff-physical-plume-calibration-v1")),
        )
        calibration.validate()
        claimed_hash = payload.get("calibration_sha256")
        if claimed_hash is not None and claimed_hash != calibration.sha256:
            raise ValueError("physical plume calibration hash mismatch")
        if payload.get("status") not in (None, calibration.status.value):
            raise ValueError("physical plume status does not match resolved fields")
        return calibration


@dataclasses.dataclass(frozen=True)
class SensorOffset:
    forward_mm: float
    lateral_mm: float

    def validate(self) -> None:
        if not math.isfinite(self.forward_mm) or not math.isfinite(self.lateral_mm):
            raise ValueError("sensor offsets must be finite")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return {"forward_mm": self.forward_mm, "lateral_mm": self.lateral_mm}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SensorOffset:
        offset = cls(forward_mm=float(payload["forward_mm"]), lateral_mm=float(payload["lateral_mm"]))
        offset.validate()
        return offset


@dataclasses.dataclass(frozen=True)
class BilateralSensorGeometry:
    """Evidence-backed left/right olfactory sampling landmarks in the fly body frame."""

    geometry_id: str
    species: str
    sex_context: str
    preparation_context: str
    anatomy_authority: str
    landmark_definition: str
    projection_convention: str
    measurement_method: str
    measurement_source_sha256: str | None = None
    measurement_source_verified: bool = False
    left_offset: SensorOffset | None = None
    right_offset: SensorOffset | None = None
    uncertainty_mm: float | None = None
    simulator_assumption_used: bool = False
    navigation_performance_used: bool = False
    schema: str = "fly-sniff-bilateral-sensor-geometry-v1"

    @property
    def status(self) -> SensorGeometryStatus:
        if (
            not self.measurement_source_verified
            or self.measurement_source_sha256 is None
            or self.left_offset is None
            or self.right_offset is None
            or self.uncertainty_mm is None
        ):
            return SensorGeometryStatus.BLOCKED_REQUIRES_ANATOMY_MEASUREMENT
        return SensorGeometryStatus.PASS_BILATERAL_SENSOR_GEOMETRY

    def validate(self) -> None:
        if self.schema != "fly-sniff-bilateral-sensor-geometry-v1":
            raise ValueError(f"unsupported bilateral sensor geometry schema: {self.schema}")
        for field, value in (
            ("geometry_id", self.geometry_id),
            ("species", self.species),
            ("sex_context", self.sex_context),
            ("preparation_context", self.preparation_context),
            ("anatomy_authority", self.anatomy_authority),
            ("landmark_definition", self.landmark_definition),
            ("projection_convention", self.projection_convention),
            ("measurement_method", self.measurement_method),
        ):
            _required_text(value, field=field)
        if self.simulator_assumption_used:
            raise ValueError("simulator geometry may not serve as Program A anatomy authority")
        if self.navigation_performance_used:
            raise ValueError("sensor geometry may not use navigation performance")
        if self.measurement_source_sha256 is not None:
            _validate_sha256(self.measurement_source_sha256, field="measurement_source_sha256")
        if self.measurement_source_verified and self.measurement_source_sha256 is None:
            raise ValueError("verified anatomy source requires measurement_source_sha256")
        if self.left_offset is not None:
            self.left_offset.validate()
        if self.right_offset is not None:
            self.right_offset.validate()
        if self.uncertainty_mm is not None and (
            not math.isfinite(self.uncertainty_mm) or self.uncertainty_mm < 0
        ):
            raise ValueError("uncertainty_mm must be finite and non-negative")
        if self.status is SensorGeometryStatus.PASS_BILATERAL_SENSOR_GEOMETRY:
            assert self.left_offset is not None and self.right_offset is not None
            if self.left_offset == self.right_offset:
                raise ValueError("left and right sampling landmarks may not be identical")

    @property
    def inter_sensor_distance_mm(self) -> float | None:
        if self.left_offset is None or self.right_offset is None:
            return None
        return math.hypot(
            self.left_offset.forward_mm - self.right_offset.forward_mm,
            self.left_offset.lateral_mm - self.right_offset.lateral_mm,
        )

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "geometry_id": self.geometry_id,
            "status": self.status.value,
            "species": self.species,
            "sex_context": self.sex_context,
            "preparation_context": self.preparation_context,
            "anatomy_authority": self.anatomy_authority,
            "landmark_definition": self.landmark_definition,
            "projection_convention": self.projection_convention,
            "measurement_method": self.measurement_method,
            "measurement_source_sha256": self.measurement_source_sha256,
            "measurement_source_verified": self.measurement_source_verified,
            "left_offset": None if self.left_offset is None else self.left_offset.to_dict(),
            "right_offset": None if self.right_offset is None else self.right_offset.to_dict(),
            "inter_sensor_distance_mm": self.inter_sensor_distance_mm,
            "uncertainty_mm": self.uncertainty_mm,
            "simulator_assumption_used": self.simulator_assumption_used,
            "navigation_performance_used": self.navigation_performance_used,
            "claim_boundary": [
                "sampling landmarks are anatomy measurements, not simulator tuning parameters",
                "left and right offsets are represented independently and need not be symmetric",
                "anatomical uncertainty is retained instead of optimized on navigation",
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["geometry_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BilateralSensorGeometry:
        geometry = cls(
            geometry_id=str(payload["geometry_id"]),
            species=str(payload["species"]),
            sex_context=str(payload["sex_context"]),
            preparation_context=str(payload["preparation_context"]),
            anatomy_authority=str(payload["anatomy_authority"]),
            landmark_definition=str(payload["landmark_definition"]),
            projection_convention=str(payload["projection_convention"]),
            measurement_method=str(payload["measurement_method"]),
            measurement_source_sha256=payload.get("measurement_source_sha256"),
            measurement_source_verified=bool(payload.get("measurement_source_verified", False)),
            left_offset=(
                None if payload.get("left_offset") is None else SensorOffset.from_dict(payload["left_offset"])
            ),
            right_offset=(
                None
                if payload.get("right_offset") is None
                else SensorOffset.from_dict(payload["right_offset"])
            ),
            uncertainty_mm=payload.get("uncertainty_mm"),
            simulator_assumption_used=bool(payload.get("simulator_assumption_used", False)),
            navigation_performance_used=bool(payload.get("navigation_performance_used", False)),
            schema=str(payload.get("schema", "fly-sniff-bilateral-sensor-geometry-v1")),
        )
        geometry.validate()
        claimed_hash = payload.get("geometry_sha256")
        if claimed_hash is not None and claimed_hash != geometry.sha256:
            raise ValueError("bilateral sensor geometry hash mismatch")
        if payload.get("status") not in (None, geometry.status.value):
            raise ValueError("bilateral sensor status does not match resolved fields")
        return geometry


@dataclasses.dataclass(frozen=True)
class PhysicalSensoryTransform:
    """Hash-bound causal transform from fly body pose to archived plume samples."""

    transform_id: str
    plume_calibration_sha256: str
    sensor_geometry_sha256: str
    spatial_interpolation: str = "bilinear"
    temporal_sampling_rule: TemporalSamplingRule = TemporalSamplingRule.CAUSAL_NATIVE_FRAME_HOLD
    out_of_bounds_rule: OutOfBoundsRule = OutOfBoundsRule.ERROR
    source_coordinates_visible_to_controller: bool = False
    future_frame_interpolation_allowed: bool = False
    navigation_performance_used: bool = False
    schema: str = "fly-sniff-physical-sensory-transform-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-physical-sensory-transform-v1":
            raise ValueError(f"unsupported physical sensory transform schema: {self.schema}")
        _required_text(self.transform_id, field="transform_id")
        _validate_sha256(self.plume_calibration_sha256, field="plume_calibration_sha256")
        _validate_sha256(self.sensor_geometry_sha256, field="sensor_geometry_sha256")
        if self.spatial_interpolation != "bilinear":
            raise ValueError("Program A physical sensory v1 freezes bilinear spatial interpolation")
        if self.temporal_sampling_rule is not TemporalSamplingRule.CAUSAL_NATIVE_FRAME_HOLD:
            raise ValueError("Program A v1 permits only causal native-frame hold")
        if self.out_of_bounds_rule is not OutOfBoundsRule.ERROR:
            raise ValueError("Program A v1 freezes out-of-bounds behavior to error")
        if self.source_coordinates_visible_to_controller:
            raise ValueError("source coordinates may not be exposed to the controller")
        if self.future_frame_interpolation_allowed:
            raise ValueError("future-frame interpolation is forbidden")
        if self.navigation_performance_used:
            raise ValueError("physical sensory transform may not use navigation performance")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "transform_id": self.transform_id,
            "status": "PASS_PHYSICAL_SENSORY_TRANSFORM",
            "plume_calibration_sha256": self.plume_calibration_sha256,
            "sensor_geometry_sha256": self.sensor_geometry_sha256,
            "spatial_interpolation": self.spatial_interpolation,
            "temporal_sampling_rule": self.temporal_sampling_rule.value,
            "out_of_bounds_rule": self.out_of_bounds_rule.value,
            "source_coordinates_visible_to_controller": self.source_coordinates_visible_to_controller,
            "future_frame_interpolation_allowed": self.future_frame_interpolation_allowed,
            "navigation_performance_used": self.navigation_performance_used,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["transform_sha256"] = self.sha256
        return payload


def assemble_physical_sensory_transform(
    plume: PhysicalPlumeCalibration,
    geometry: BilateralSensorGeometry,
    *,
    transform_id: str,
) -> PhysicalSensoryTransform:
    plume.validate()
    geometry.validate()
    if plume.status is not PhysicalPlumeStatus.PASS_PHYSICAL_PLUME:
        raise ValueError(f"physical plume is not PASS: {plume.status.value}")
    if geometry.status is not SensorGeometryStatus.PASS_BILATERAL_SENSOR_GEOMETRY:
        raise ValueError(f"bilateral sensor geometry is not PASS: {geometry.status.value}")
    transform = PhysicalSensoryTransform(
        transform_id=transform_id,
        plume_calibration_sha256=plume.sha256,
        sensor_geometry_sha256=geometry.sha256,
    )
    transform.validate()
    return transform


def body_sensor_world_positions(
    *,
    body_downwind_mm: float,
    body_crosswind_mm: float,
    heading_rad: float,
    geometry: BilateralSensorGeometry,
) -> tuple[tuple[float, float], tuple[float, float]]:
    geometry.validate()
    if geometry.status is not SensorGeometryStatus.PASS_BILATERAL_SENSOR_GEOMETRY:
        raise ValueError("bilateral sensor geometry must PASS before sampling")
    assert geometry.left_offset is not None and geometry.right_offset is not None
    if not all(math.isfinite(value) for value in (body_downwind_mm, body_crosswind_mm, heading_rad)):
        raise ValueError("body pose must be finite")

    cosine = math.cos(heading_rad)
    sine = math.sin(heading_rad)

    def world(offset: SensorOffset) -> tuple[float, float]:
        return (
            body_downwind_mm + cosine * offset.forward_mm - sine * offset.lateral_mm,
            body_crosswind_mm + sine * offset.forward_mm + cosine * offset.lateral_mm,
        )

    return world(geometry.left_offset), world(geometry.right_offset)


def world_to_archive_pixel(
    *,
    downwind_mm: float,
    crosswind_mm: float,
    plume: PhysicalPlumeCalibration,
) -> tuple[float, float]:
    plume.validate()
    if plume.status is not PhysicalPlumeStatus.PASS_PHYSICAL_PLUME:
        raise ValueError("physical plume calibration must PASS before coordinate conversion")
    assert plume.axis0_role is not None and plume.axis1_role is not None
    assert plume.axis0_positive_direction is not None
    assert plume.axis1_positive_direction is not None
    assert plume.axis0_mm_per_pixel is not None and plume.axis1_mm_per_pixel is not None
    assert plume.source_index_axis0 is not None and plume.source_index_axis1 is not None
    if not math.isfinite(downwind_mm) or not math.isfinite(crosswind_mm):
        raise ValueError("world coordinates must be finite")

    role_value = {
        AxisRole.DOWNWIND: downwind_mm,
        AxisRole.CROSSWIND: crosswind_mm,
    }
    axis0 = plume.source_index_axis0 + (
        plume.axis0_positive_direction * role_value[plume.axis0_role] / plume.axis0_mm_per_pixel
    )
    axis1 = plume.source_index_axis1 + (
        plume.axis1_positive_direction * role_value[plume.axis1_role] / plume.axis1_mm_per_pixel
    )
    return axis0, axis1


def bilinear_sample(frame: np.ndarray, axis0: float, axis1: float) -> float:
    value = np.asarray(frame)
    if value.ndim != 2:
        raise ValueError(f"bilinear_sample requires a 2-D frame, got {value.shape}")
    if not math.isfinite(axis0) or not math.isfinite(axis1):
        raise ValueError("sample coordinates must be finite")
    if axis0 < 0 or axis1 < 0 or axis0 > value.shape[0] - 1 or axis1 > value.shape[1] - 1:
        raise ValueError("sample coordinate is outside the archived plume frame")

    lo0 = int(math.floor(axis0))
    lo1 = int(math.floor(axis1))
    hi0 = min(lo0 + 1, value.shape[0] - 1)
    hi1 = min(lo1 + 1, value.shape[1] - 1)
    weight0 = axis0 - lo0
    weight1 = axis1 - lo1

    v00 = float(value[lo0, lo1])
    v10 = float(value[hi0, lo1])
    v01 = float(value[lo0, hi1])
    v11 = float(value[hi0, hi1])
    return (
        (1.0 - weight0) * (1.0 - weight1) * v00
        + weight0 * (1.0 - weight1) * v10
        + (1.0 - weight0) * weight1 * v01
        + weight0 * weight1 * v11
    )


def causal_native_frame_index(*, time_seconds: float, native_fps: float, frame_count: int) -> int:
    if not math.isfinite(time_seconds) or time_seconds < 0:
        raise ValueError("time_seconds must be finite and non-negative")
    if not math.isfinite(native_fps) or native_fps <= 0:
        raise ValueError("native_fps must be positive and finite")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    index = int(math.floor(time_seconds * native_fps + 1e-12))
    if index >= frame_count:
        raise ValueError("requested time is outside the archived plume duration")
    return index


def sample_bilateral_concentration(
    frame: np.ndarray,
    *,
    body_downwind_mm: float,
    body_crosswind_mm: float,
    heading_rad: float,
    plume: PhysicalPlumeCalibration,
    geometry: BilateralSensorGeometry,
    transform: PhysicalSensoryTransform,
) -> tuple[float, float]:
    transform.validate()
    if transform.plume_calibration_sha256 != plume.sha256:
        raise ValueError("transform/plume calibration hash mismatch")
    if transform.sensor_geometry_sha256 != geometry.sha256:
        raise ValueError("transform/sensor geometry hash mismatch")
    left_world, right_world = body_sensor_world_positions(
        body_downwind_mm=body_downwind_mm,
        body_crosswind_mm=body_crosswind_mm,
        heading_rad=heading_rad,
        geometry=geometry,
    )
    left_pixel = world_to_archive_pixel(
        downwind_mm=left_world[0], crosswind_mm=left_world[1], plume=plume
    )
    right_pixel = world_to_archive_pixel(
        downwind_mm=right_world[0], crosswind_mm=right_world[1], plume=plume
    )
    return (
        bilinear_sample(frame, *left_pixel),
        bilinear_sample(frame, *right_pixel),
    )
