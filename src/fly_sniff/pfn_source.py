from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from .freeze import canonical_sha256

BLOCKED_README = "BLOCKED_README_SHA256_UNVERIFIED"
BLOCKED_MEMBER_MAP = "BLOCKED_ARCHIVE_MEMBER_MAP_UNRESOLVED"
BLOCKED_RECORDINGS = "BLOCKED_FIGURE4_RECORDING_SET_UNRESOLVED"
READY = "READY_FOR_EXTRACTION"

_HEX = frozenset("0123456789abcdef")
_EXPECTED_DIRECTIONS = (-135, -90, -45, 0, 45, 90, 135, 180)
_EXPECTED_ARCHIVE_MD5 = (
    ("Currier2020.z01", "5c6a96fb9a56c9ca145990b688413d15"),
    ("Currier2020.zip.001", "72aeaa0e96ee557f81c70c8364ca5fb1"),
    ("Currier2020.zip.002", "0c73c27f13345a77cce73b83de03fd62"),
    ("Currier2020.zip.003", "93d193d66346a2242f1234477fc5b67e"),
    ("Currier2020.zip.004", "384561a3ae0acb829e60a8b4552a29d0"),
    ("Currier2020.zip.005", "c7acddb8fd23334c6ebce996e9409e8f"),
    ("Currier2020.zip.006", "1b88b94cb2cd3774cb57835418bb0a14"),
    ("Currier2020.zip.007", "9be635bab9b9571c36e4fa49861d5a59"),
    ("Currier2020.zip.008", "2542ab153fff740d62b5dc3c2e6c3e1e"),
    ("Currier2020.zip.009", "d0f8cf976694f0a604a1cdc7aa1fa185"),
    ("Currier2020.zip.010", "843b567cfaf1e839e0297bfc7a8640d4"),
    ("Currier2020.zip.011", "414c7409446b55dc70310f8562d9c672"),
    ("Currier2020.zip.012", "c35f9fe0a618640ad995c62dbf2ed9c3"),
    ("Currier2020.zip.013", "f2c15c70aec25f00d63282f87aee40d0"),
)
_EXPECTED_ARCHIVE_NAMES = tuple(name for name, _ in _EXPECTED_ARCHIVE_MD5)


def _require_text(value: str, *, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _validate_hex(value: str, *, field: str, length: int) -> None:
    if len(value) != length or any(char not in _HEX for char in value):
        raise ValueError(f"{field} must be a lowercase {length}-character hexadecimal digest")


def _wrap_deg(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if np.isclose(wrapped, -180.0) and value > 0 else wrapped


def mean_response_vector_angle_deg(
    directions_deg: list[float] | tuple[float, ...] | np.ndarray,
    responses: list[float] | tuple[float, ...] | np.ndarray,
) -> float:
    """Reproduce the Figure 4E signed mean-response vector angle.

    Baseline-subtracted inhibitory responses are intentionally retained as negative
    vector coefficients. They must not be rectified or clipped before this step.
    """

    directions = np.asarray(directions_deg, dtype=np.float64)
    values = np.asarray(responses, dtype=np.float64)
    if directions.ndim != 1 or values.ndim != 1 or directions.shape != values.shape:
        raise ValueError("directions and responses must be equal-length one-dimensional arrays")
    if directions.size != 8:
        raise ValueError("Figure 4 preferred-direction calculation requires eight directions")
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(values)):
        raise ValueError("directions and responses must be finite")
    if tuple(int(value) for value in directions) != _EXPECTED_DIRECTIONS:
        raise ValueError("Figure 4 direction order must remain the published eight-direction set")

    vector = np.mean(values * np.exp(1j * np.deg2rad(directions)))
    if abs(vector) <= np.finfo(np.float64).eps:
        raise ValueError("preferred direction is undefined for a zero resultant vector")
    return _wrap_deg(float(np.rad2deg(np.angle(vector))))


def fold_angle_to_ipsilateral_deg(angle_deg: float, *, soma_side: str) -> float:
    """Fold left/right soma angles so positive values are ipsilateral."""

    side = soma_side.upper()
    if side not in {"L", "R"}:
        raise ValueError("soma_side must be L or R")
    return _wrap_deg(float(angle_deg) if side == "L" else -float(angle_deg))


def circular_mean_deg(angles_deg: list[float] | tuple[float, ...] | np.ndarray) -> float:
    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("angles must be a non-empty finite one-dimensional array")
    vector = np.mean(np.exp(1j * np.deg2rad(angles)))
    if abs(vector) <= np.finfo(np.float64).eps:
        raise ValueError("circular mean is undefined for a zero resultant vector")
    return _wrap_deg(float(np.rad2deg(np.angle(vector))))


def bootstrap_circular_mean_ci_deg(
    angles_deg: list[float] | tuple[float, ...] | np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return center and an unwrapped percentile CI around the circular mean."""

    angles = np.asarray(angles_deg, dtype=np.float64)
    center = circular_mean_deg(angles)
    if resamples != 10_000 or seed != 0:
        raise ValueError("Program A PFN uncertainty is frozen to 10000 resamples with seed 0")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, angles.size, size=(resamples, angles.size))
    radians = np.deg2rad(angles[indices])
    vectors = np.mean(np.exp(1j * radians), axis=1)
    if np.any(np.abs(vectors) <= np.finfo(np.float64).eps):
        raise ValueError("bootstrap produced an undefined zero-resultant circular mean")
    means = np.rad2deg(np.angle(vectors))
    deltas = (means - center + 180.0) % 360.0 - 180.0
    lower_delta, upper_delta = np.percentile(deltas, [2.5, 97.5])
    return center, center + float(lower_delta), center + float(upper_delta)


@dataclasses.dataclass(frozen=True)
class ArchiveMemberRef:
    member_path: str
    container_files: tuple[str, ...]
    sha256: str

    def validate(self) -> None:
        _require_text(self.member_path, field="archive member path")
        if not self.container_files:
            raise ValueError("archive member must name at least one Dryad container file")
        if any(name not in _EXPECTED_ARCHIVE_NAMES for name in self.container_files):
            raise ValueError("archive member references an unknown Dryad multipart file")
        _validate_hex(self.sha256, field="archive member sha256", length=64)


@dataclasses.dataclass(frozen=True)
class RecordingRef:
    recording_id: str
    soma_side: str
    archive_member_path: str

    def validate(self) -> None:
        _require_text(self.recording_id, field="recording_id")
        _require_text(self.archive_member_path, field="archive_member_path")
        if self.soma_side.upper() not in {"L", "R"}:
            raise ValueError("recording soma_side must be L or R")


@dataclasses.dataclass(frozen=True)
class PFNSourceContract:
    contract_id: str
    paper_doi: str
    paper_title: str
    target_figure: str
    published_recording_count: int
    published_claim: str
    dryad_doi: str
    dryad_published_date: str
    published_total_size_gb: float
    dryad_license: str
    readme_filename: str
    readme_file_stream_id: int
    readme_published_size_bytes: int
    readme_md5: str
    readme_sha256: str | None
    whole_archive_download_is_default: bool
    archive_inventory: tuple[tuple[str, str], ...]
    archive_member_map: tuple[ArchiveMemberRef, ...]
    species: str
    cell_class: str
    airflow_only: bool
    all_recordings_completed_all_trials: bool
    zero_deg_definition: str
    positive_90_deg_definition: str
    negative_90_deg_definition: str
    laterality_authority: str
    left_soma_ipsilateral_sign: int
    right_soma_ipsilateral_sign: int
    recording_set: tuple[RecordingRef, ...]
    directions_deg: tuple[int, ...]
    airflow_speed_cm_per_s: float
    repetitions_per_direction: int
    trials_per_recording: int
    baseline_window_s: tuple[float, float]
    response_window_s: tuple[float, float]
    per_trial_response_definition: str
    repeat_aggregation_definition: str
    preferred_direction_method: str
    negative_responses_are_clipped: bool
    target_per_recording_statistic: str
    hemisphere_fold_definition: str
    aggregate_statistic: str
    uncertainty_rule: str
    bootstrap_resamples: int
    bootstrap_seed: int
    absolute_gain_transferred: bool
    temporal_shape_transferred: bool
    evidence_ledger_record_ids: tuple[str, ...]
    navigation_performance_used: bool
    allowed_interpretation: tuple[str, ...]
    forbidden_interpretation: tuple[str, ...]
    schema: str = "fly-sniff-pfn-calibration-source-v1"

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.readme_sha256 is None:
            blockers.append(BLOCKED_README)
        if not self.archive_member_map:
            blockers.append(BLOCKED_MEMBER_MAP)
        if len(self.recording_set) != self.published_recording_count:
            blockers.append(BLOCKED_RECORDINGS)
        return tuple(sorted(blockers))

    @property
    def status(self) -> str:
        return READY if not self.blockers else "BLOCKED"

    def validate(self) -> None:
        if self.schema != "fly-sniff-pfn-calibration-source-v1":
            raise ValueError(f"unsupported PFN source schema: {self.schema}")
        _require_text(self.contract_id, field="contract_id")
        if self.paper_doi != "10.7554/eLife.61510":
            raise ValueError("PFN source contract must bind the version-of-record eLife DOI")
        if self.paper_title != (
            "Encoding and control of orientation to airflow by a set of Drosophila "
            "fan-shaped body neurons"
        ):
            raise ValueError("PFN source contract paper title does not match the frozen authority")
        if self.target_figure != "Figure 4D-E":
            raise ValueError("Program A PFN v1 target is frozen to Figure 4D-E")
        if self.published_recording_count != 12:
            raise ValueError("Figure 4 source contract must retain the published 12 recordings")
        _require_text(self.published_claim, field="published_claim")

        if self.dryad_doi != "10.5061/dryad.vq83bk3rh":
            raise ValueError("PFN source contract must bind the published Dryad DOI")
        if self.dryad_published_date != "2020-12-31":
            raise ValueError("Dryad publication date does not match the frozen authority")
        if not np.isclose(self.published_total_size_gb, 117.49):
            raise ValueError("Dryad total published size does not match the frozen authority")
        if self.dryad_license != "CC0-1.0":
            raise ValueError("Dryad license does not match the frozen authority")
        if self.readme_filename != "Currier2020README.rtf" or self.readme_file_stream_id != 536042:
            raise ValueError("PFN source contract must retain the published README identity")
        if self.readme_published_size_bytes != 2550:
            raise ValueError("README published size does not match the frozen authority")
        _validate_hex(self.readme_md5, field="README md5", length=32)
        if self.readme_md5 != "91e5213503788fcde11c0f5aa3e91f43":
            raise ValueError("README MD5 does not match the public repository record")
        if self.readme_sha256 is not None:
            _validate_hex(self.readme_sha256, field="README sha256", length=64)
        if self.whole_archive_download_is_default:
            raise ValueError("full 117.49 GB archive download may not be the default PFN workflow")
        if self.navigation_performance_used:
            raise ValueError("PFN source selection may not use navigation performance")

        if self.archive_inventory != _EXPECTED_ARCHIVE_MD5:
            raise ValueError("Dryad multipart inventory/MD5 values must match the frozen authority")

        if self.species != "Drosophila melanogaster":
            raise ValueError("PFN source preparation species must remain Drosophila melanogaster")
        if self.cell_class != "P-F2N3 ventral P-FN whole-cell electrophysiology":
            raise ValueError("PFN source preparation cell class has changed")
        if not self.airflow_only:
            raise ValueError("Figure 4 source protocol must remain airflow-only")
        if not self.all_recordings_completed_all_trials:
            raise ValueError("Figure 4 contract must preserve that all 12 recordings completed 40 trials")
        if self.zero_deg_definition != "frontal airflow":
            raise ValueError("PFN 0 degree convention must remain frontal airflow")
        if self.positive_90_deg_definition != "left airflow":
            raise ValueError("PFN +90 degree convention must remain left airflow")
        if self.negative_90_deg_definition != "right airflow":
            raise ValueError("PFN -90 degree convention must remain right airflow")
        if self.laterality_authority != "cell-body hemisphere":
            raise ValueError("PFN laterality authority must remain cell-body hemisphere")
        if self.left_soma_ipsilateral_sign != 1 or self.right_soma_ipsilateral_sign != -1:
            raise ValueError("PFN soma-side folding signs must remain L=+1 and R=-1")

        if self.directions_deg != _EXPECTED_DIRECTIONS:
            raise ValueError("Figure 4 directions must remain the published eight-direction set")
        if self.airflow_speed_cm_per_s != 25:
            raise ValueError("Figure 4 airflow speed must remain 25 cm/s")
        if self.repetitions_per_direction != 5 or self.trials_per_recording != 40:
            raise ValueError("Figure 4 must retain five repeats per direction and 40 trials")
        if self.baseline_window_s != (-1.5, -0.5):
            raise ValueError("PFN baseline window must remain 1 s ending 500 ms before onset")
        if self.response_window_s != (0.5, 1.5):
            raise ValueError("PFN response window must remain 1 s starting 500 ms after onset")
        if self.per_trial_response_definition != (
            "mean firing rate during response window minus mean firing rate during baseline window"
        ):
            raise ValueError("PFN per-trial response definition has changed")
        if self.repeat_aggregation_definition != (
            "mean across the five repetitions for each airflow direction"
        ):
            raise ValueError("PFN repeat aggregation definition has changed")
        if "signed response vector" not in self.preferred_direction_method:
            raise ValueError("PFN preferred-direction method must retain signed response vectors")
        if self.negative_responses_are_clipped:
            raise ValueError("published signed baseline-subtracted responses may not be clipped")

        if self.target_per_recording_statistic != "published mean response-vector angle":
            raise ValueError("PFN per-recording target statistic has changed")
        if self.hemisphere_fold_definition != (
            "multiply angle by +1 for left cell-body hemisphere and -1 for right cell-body "
            "hemisphere so ipsilateral is positive"
        ):
            raise ValueError("PFN hemisphere-fold definition has changed")
        if self.aggregate_statistic != (
            "circular mean of hemisphere-folded preferred-direction angles"
        ):
            raise ValueError("PFN aggregate statistic has changed")
        if "95% percentile bootstrap" not in self.uncertainty_rule:
            raise ValueError("PFN uncertainty text must preserve the frozen percentile bootstrap")
        if self.bootstrap_resamples != 10_000 or self.bootstrap_seed != 0:
            raise ValueError("PFN uncertainty rule is frozen to 10000 resamples with seed 0")
        if self.absolute_gain_transferred or self.temporal_shape_transferred:
            raise ValueError("Program A PFN v1 transfers geometry, not absolute gain or dynamics")

        member_paths: set[str] = set()
        for member in self.archive_member_map:
            member.validate()
            if member.member_path in member_paths:
                raise ValueError(f"duplicate archive member path: {member.member_path}")
            member_paths.add(member.member_path)

        recording_ids: set[str] = set()
        for recording in self.recording_set:
            recording.validate()
            if recording.recording_id in recording_ids:
                raise ValueError(f"duplicate Figure 4 recording_id: {recording.recording_id}")
            if recording.archive_member_path not in member_paths:
                raise ValueError("Figure 4 recording references an unresolved archive member")
            recording_ids.add(recording.recording_id)
        if self.recording_set and len(self.recording_set) != self.published_recording_count:
            raise ValueError("resolved Figure 4 recording set must contain exactly 12 recordings")

        required_evidence = {
            "currier2021-ventral-pfn-airflow-basis",
            "pfn-malecns-transfer-caveat",
        }
        if not required_evidence.issubset(self.evidence_ledger_record_ids):
            raise ValueError("PFN source contract must retain physiology and transfer-caveat evidence IDs")
        if not self.allowed_interpretation or not self.forbidden_interpretation:
            raise ValueError("allowed and forbidden PFN interpretation boundaries are required")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "contract_id": self.contract_id,
            "status": self.status,
            "blockers": list(self.blockers),
            "paper": {
                "doi": self.paper_doi,
                "title": self.paper_title,
                "target_figure": self.target_figure,
                "published_recording_count": self.published_recording_count,
                "published_claim": self.published_claim,
            },
            "data_authority": {
                "repository": "Dryad",
                "doi": self.dryad_doi,
                "published_date": self.dryad_published_date,
                "published_total_size_gb": self.published_total_size_gb,
                "license": self.dryad_license,
                "readme": {
                    "filename": self.readme_filename,
                    "dryad_file_stream_id": self.readme_file_stream_id,
                    "published_size_bytes": self.readme_published_size_bytes,
                    "published_md5": self.readme_md5,
                    "sha256": self.readme_sha256,
                    "status": READY if self.readme_sha256 is not None else BLOCKED_README,
                },
                "multipart_archive": [
                    {"filename": name, "published_md5": md5}
                    for name, md5 in self.archive_inventory
                ],
                "archive_member_map": [dataclasses.asdict(item) for item in self.archive_member_map],
                "archive_member_map_status": (
                    READY if self.archive_member_map else BLOCKED_MEMBER_MAP
                ),
                "whole_archive_download_is_default": self.whole_archive_download_is_default,
            },
            "source_preparation": {
                "species": self.species,
                "cell_class": self.cell_class,
                "figure4_protocol": {
                    "airflow_speed_cm_per_s": self.airflow_speed_cm_per_s,
                    "directions_deg": list(self.directions_deg),
                    "airflow_only": self.airflow_only,
                    "trials_per_recording": self.trials_per_recording,
                    "repetitions_per_direction": self.repetitions_per_direction,
                    "published_recording_count": self.published_recording_count,
                    "all_recordings_completed_all_trials": self.all_recordings_completed_all_trials,
                },
                "angular_convention": {
                    "zero_deg": self.zero_deg_definition,
                    "positive_90_deg": self.positive_90_deg_definition,
                    "negative_90_deg": self.negative_90_deg_definition,
                    "laterality_authority": self.laterality_authority,
                    "left_soma_ipsilateral_sign": self.left_soma_ipsilateral_sign,
                    "right_soma_ipsilateral_sign": self.right_soma_ipsilateral_sign,
                },
            },
            "published_response_definition": {
                "baseline_window_relative_to_stimulus_s": list(self.baseline_window_s),
                "response_window_relative_to_stimulus_s": list(self.response_window_s),
                "per_trial_response": self.per_trial_response_definition,
                "repeat_aggregation": self.repeat_aggregation_definition,
                "preferred_direction_method": self.preferred_direction_method,
                "negative_responses_are_clipped": self.negative_responses_are_clipped,
            },
            "program_a_normalized_target": {
                "per_recording_statistic": self.target_per_recording_statistic,
                "hemisphere_fold": self.hemisphere_fold_definition,
                "aggregate_statistic": self.aggregate_statistic,
                "bootstrap_resamples": self.bootstrap_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "uncertainty_rule": self.uncertainty_rule,
                "absolute_firing_rate_gain_transferred": self.absolute_gain_transferred,
                "temporal_response_shape_transferred": self.temporal_shape_transferred,
                "status": "FROZEN_DEFINITION_PENDING_SOURCE_BYTES",
            },
            "figure4_recording_set": [dataclasses.asdict(item) for item in self.recording_set],
            "figure4_recording_set_status": (
                READY
                if len(self.recording_set) == self.published_recording_count
                else BLOCKED_RECORDINGS
            ),
            "evidence_ledger_record_ids": list(self.evidence_ledger_record_ids),
            "navigation_performance_used": self.navigation_performance_used,
            "allowed_interpretation": list(self.allowed_interpretation),
            "forbidden_interpretation": list(self.forbidden_interpretation),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["source_contract_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PFNSourceContract:
        paper = payload["paper"]
        data = payload["data_authority"]
        readme = data["readme"]
        preparation = payload["source_preparation"]
        protocol = preparation["figure4_protocol"]
        angular = preparation["angular_convention"]
        response = payload["published_response_definition"]
        target = payload["program_a_normalized_target"]
        contract = cls(
            contract_id=str(payload["contract_id"]),
            paper_doi=str(paper["doi"]),
            paper_title=str(paper["title"]),
            target_figure=str(paper["target_figure"]),
            published_recording_count=int(paper["published_recording_count"]),
            published_claim=str(paper["published_claim"]),
            dryad_doi=str(data["doi"]),
            dryad_published_date=str(data["published_date"]),
            published_total_size_gb=float(data["published_total_size_gb"]),
            dryad_license=str(data["license"]),
            readme_filename=str(readme["filename"]),
            readme_file_stream_id=int(readme["dryad_file_stream_id"]),
            readme_published_size_bytes=int(readme["published_size_bytes"]),
            readme_md5=str(readme["published_md5"]),
            readme_sha256=None if readme.get("sha256") is None else str(readme["sha256"]),
            whole_archive_download_is_default=bool(data["whole_archive_download_is_default"]),
            archive_inventory=tuple(
                (str(item["filename"]), str(item["published_md5"]))
                for item in data["multipart_archive"]
            ),
            archive_member_map=tuple(
                ArchiveMemberRef(
                    member_path=str(item["member_path"]),
                    container_files=tuple(str(value) for value in item["container_files"]),
                    sha256=str(item["sha256"]),
                )
                for item in data.get("archive_member_map", [])
            ),
            species=str(preparation["species"]),
            cell_class=str(preparation["cell_class"]),
            airflow_only=bool(protocol["airflow_only"]),
            all_recordings_completed_all_trials=bool(
                protocol["all_recordings_completed_all_trials"]
            ),
            zero_deg_definition=str(angular["zero_deg"]),
            positive_90_deg_definition=str(angular["positive_90_deg"]),
            negative_90_deg_definition=str(angular["negative_90_deg"]),
            laterality_authority=str(angular["laterality_authority"]),
            left_soma_ipsilateral_sign=int(angular["left_soma_ipsilateral_sign"]),
            right_soma_ipsilateral_sign=int(angular["right_soma_ipsilateral_sign"]),
            recording_set=tuple(
                RecordingRef(
                    recording_id=str(item["recording_id"]),
                    soma_side=str(item["soma_side"]),
                    archive_member_path=str(item["archive_member_path"]),
                )
                for item in payload.get("figure4_recording_set", [])
            ),
            directions_deg=tuple(int(value) for value in protocol["directions_deg"]),
            airflow_speed_cm_per_s=float(protocol["airflow_speed_cm_per_s"]),
            repetitions_per_direction=int(protocol["repetitions_per_direction"]),
            trials_per_recording=int(protocol["trials_per_recording"]),
            baseline_window_s=tuple(
                float(value) for value in response["baseline_window_relative_to_stimulus_s"]
            ),
            response_window_s=tuple(
                float(value) for value in response["response_window_relative_to_stimulus_s"]
            ),
            per_trial_response_definition=str(response["per_trial_response"]),
            repeat_aggregation_definition=str(response["repeat_aggregation"]),
            preferred_direction_method=str(response["preferred_direction_method"]),
            negative_responses_are_clipped=bool(response["negative_responses_are_clipped"]),
            target_per_recording_statistic=str(target["per_recording_statistic"]),
            hemisphere_fold_definition=str(target["hemisphere_fold"]),
            aggregate_statistic=str(target["aggregate_statistic"]),
            uncertainty_rule=str(target["uncertainty_rule"]),
            bootstrap_resamples=int(target["bootstrap_resamples"]),
            bootstrap_seed=int(target["bootstrap_seed"]),
            absolute_gain_transferred=bool(target["absolute_firing_rate_gain_transferred"]),
            temporal_shape_transferred=bool(target["temporal_response_shape_transferred"]),
            evidence_ledger_record_ids=tuple(
                str(value) for value in payload["evidence_ledger_record_ids"]
            ),
            navigation_performance_used=bool(payload["navigation_performance_used"]),
            allowed_interpretation=tuple(
                str(value) for value in payload["allowed_interpretation"]
            ),
            forbidden_interpretation=tuple(
                str(value) for value in payload["forbidden_interpretation"]
            ),
            schema=str(payload["schema"]),
        )
        contract.validate()

        claimed_readme_status = readme.get("status")
        expected_readme_status = READY if contract.readme_sha256 is not None else BLOCKED_README
        if claimed_readme_status is not None and claimed_readme_status != expected_readme_status:
            raise ValueError("PFN README status does not match source-byte verification")
        claimed_member_status = data.get("archive_member_map_status")
        expected_member_status = READY if contract.archive_member_map else BLOCKED_MEMBER_MAP
        if claimed_member_status is not None and claimed_member_status != expected_member_status:
            raise ValueError("PFN archive-member status does not match the member map")
        claimed_recording_status = payload.get("figure4_recording_set_status")
        expected_recording_status = (
            READY
            if len(contract.recording_set) == contract.published_recording_count
            else BLOCKED_RECORDINGS
        )
        if (
            claimed_recording_status is not None
            and claimed_recording_status != expected_recording_status
        ):
            raise ValueError("PFN recording-set status does not match the frozen recordings")
        claimed_target_status = target.get("status")
        if (
            claimed_target_status is not None
            and claimed_target_status != "FROZEN_DEFINITION_PENDING_SOURCE_BYTES"
        ):
            raise ValueError("PFN normalized-target status is invalid")

        claimed_status = payload.get("status")
        if claimed_status is not None and claimed_status != contract.status:
            raise ValueError("PFN source contract status does not match blockers")
        claimed_blockers = payload.get("blockers")
        if claimed_blockers is not None and tuple(claimed_blockers) != contract.blockers:
            raise ValueError("PFN source contract blockers do not match resolved inputs")
        claimed_hash = payload.get("source_contract_sha256")
        if claimed_hash is not None and claimed_hash != contract.sha256:
            raise ValueError("PFN source contract hash mismatch")
        return contract


def load_contract(path: str | Path) -> PFNSourceContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("PFN source contract must be a JSON object")
    return PFNSourceContract.from_dict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Program A PFN source ingress")
    parser.add_argument("manifest")
    args = parser.parse_args(argv)
    contract = load_contract(args.manifest)
    print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    return 0 if contract.status == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
