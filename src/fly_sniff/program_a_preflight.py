from __future__ import annotations

import argparse
import dataclasses
import enum
import json
from pathlib import Path
from typing import Any

from .artifact_io import load_json, write_typed_artifact
from .experiment import (
    ArtifactRef,
    ExperimentLock,
    ExperimentPhase,
    ExperimentProgram,
    ExperimentSpec,
)
from .freeze import canonical_sha256

_HEX = frozenset("0123456789abcdef")

REQUIRED_PROGRAM_A_ARTIFACT_KINDS = (
    "evidence_ledger",
    "malecns_graph",
    "role_sign_laterality_authority",
    "physiology_calibration_protocol",
    "calibrated_dynamics",
    "plume_contract",
    "bilateral_sensor_geometry",
    "topology_null_protocol",
    "intervention_protocol",
    "environment_ood_contract",
    "final_entropy_commitment",
)


class DependencyStatus(str, enum.Enum):
    PASS = "pass"
    BLOCKED = "blocked"


@dataclasses.dataclass(frozen=True)
class DependencyGate:
    artifact: ArtifactRef
    status: DependencyStatus
    notes: str | None = None

    def validate(self) -> None:
        self.artifact.validate()
        if self.notes is not None and not self.notes.strip():
            raise ValueError("dependency notes must be non-empty when supplied")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "artifact": self.artifact.to_dict(),
            "status": self.status.value,
        }
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DependencyGate:
        gate = cls(
            artifact=ArtifactRef.from_dict(payload["artifact"]),
            status=DependencyStatus(payload["status"]),
            notes=payload.get("notes"),
        )
        gate.validate()
        return gate


@dataclasses.dataclass(frozen=True)
class DependencyManifest:
    dependencies: tuple[DependencyGate, ...]
    schema: str = "fly-sniff-program-a-dependencies-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-program-a-dependencies-v1":
            raise ValueError(f"unsupported dependency manifest schema: {self.schema}")
        if not self.dependencies:
            raise ValueError("dependency manifest must contain at least one dependency")
        names: set[str] = set()
        for dependency in self.dependencies:
            dependency.validate()
            if dependency.artifact.name in names:
                raise ValueError(f"duplicate dependency artifact name: {dependency.artifact.name}")
            names.add(dependency.artifact.name)

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["manifest_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DependencyManifest:
        manifest = cls(
            dependencies=tuple(
                DependencyGate.from_dict(item) for item in payload.get("dependencies", [])
            ),
            schema=str(payload.get("schema", "fly-sniff-program-a-dependencies-v1")),
        )
        manifest.validate()
        claimed_hash = payload.get("manifest_sha256")
        if claimed_hash is not None and claimed_hash != manifest.sha256:
            raise ValueError("dependency manifest hash mismatch")
        return manifest


@dataclasses.dataclass(frozen=True)
class ProgramAReadinessReport:
    experiment_id: str
    spec_sha256: str
    dependency_manifest_sha256: str
    blockers: tuple[str, ...]
    schema: str = "fly-sniff-program-a-readiness-v1"

    @property
    def ready_to_lock(self) -> bool:
        return not self.blockers

    def validate(self) -> None:
        if self.schema != "fly-sniff-program-a-readiness-v1":
            raise ValueError(f"unsupported Program A readiness schema: {self.schema}")
        if not self.experiment_id.strip():
            raise ValueError("readiness experiment_id must be non-empty")
        _validate_sha256(self.spec_sha256, field="spec_sha256")
        _validate_sha256(self.dependency_manifest_sha256, field="dependency_manifest_sha256")
        if tuple(sorted(set(self.blockers))) != self.blockers:
            raise ValueError("readiness blockers must be unique and sorted")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "spec_sha256": self.spec_sha256,
            "dependency_manifest_sha256": self.dependency_manifest_sha256,
            "status": "READY_TO_LOCK" if self.ready_to_lock else "BLOCKED",
            "blockers": list(self.blockers),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["readiness_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProgramAReadinessReport:
        report = cls(
            experiment_id=str(payload["experiment_id"]),
            spec_sha256=str(payload["spec_sha256"]),
            dependency_manifest_sha256=str(payload["dependency_manifest_sha256"]),
            blockers=tuple(str(value) for value in payload.get("blockers", [])),
            schema=str(payload.get("schema", "fly-sniff-program-a-readiness-v1")),
        )
        report.validate()
        expected_status = "READY_TO_LOCK" if report.ready_to_lock else "BLOCKED"
        if payload.get("status") not in (None, expected_status):
            raise ValueError("readiness status does not match blocker set")
        claimed_hash = payload.get("readiness_sha256")
        if claimed_hash is not None and claimed_hash != report.sha256:
            raise ValueError("readiness report hash mismatch")
        return report


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")


def build_readiness_report(
    spec: ExperimentSpec,
    dependency_manifest: DependencyManifest,
) -> ProgramAReadinessReport:
    dependency_manifest.validate()
    blockers: list[str] = []

    try:
        spec.validate()
    except (TypeError, ValueError) as exc:
        blockers.append(f"experiment_spec_invalid:{exc}")

    if spec.program is not ExperimentProgram.LATENT_WIRING:
        blockers.append("program_must_be_latent_wiring")
    if spec.phase is not ExperimentPhase.CONFIRMATORY:
        blockers.append("phase_must_be_confirmatory")
    if len(spec.primary_metrics) != 1:
        blockers.append("exactly_one_headline_primary_metric_required")

    policy = spec.training_policy
    if policy.navigation_reward_allowed:
        blockers.append("navigation_reward_forbidden")
    if policy.topology_specific_fit_allowed:
        blockers.append("topology_specific_fit_forbidden")
    if policy.whole_graph_backprop_allowed:
        blockers.append("whole_graph_backprop_forbidden")
    if not policy.calibration_targets:
        blockers.append("independent_calibration_targets_required")

    final = spec.final_policy
    if not final.topology_claim:
        blockers.append("topology_claim_must_be_explicit")
    if final.topology_null_count < 31:
        blockers.append("at_least_31_topology_nulls_required")
    if not final.paired_episodes_required:
        blockers.append("paired_episode_conditions_required")
    if not final.hidden_final_entropy_required:
        blockers.append("hidden_final_entropy_required")
    if not final.one_way_final:
        blockers.append("one_way_final_required")
    if not final.negative_results_retained:
        blockers.append("negative_results_must_be_retained")
    if not spec.null_families:
        blockers.append("at_least_one_null_family_required")

    gates_by_kind: dict[str, list[DependencyGate]] = {}
    for gate in dependency_manifest.dependencies:
        gates_by_kind.setdefault(gate.artifact.kind, []).append(gate)
        if gate.artifact not in spec.artifacts:
            blockers.append(f"dependency_not_exactly_bound_in_spec:{gate.artifact.name}")
        if gate.status is DependencyStatus.BLOCKED:
            blockers.append(f"dependency_blocked:{gate.artifact.name}")

    for required_kind in REQUIRED_PROGRAM_A_ARTIFACT_KINDS:
        if required_kind not in gates_by_kind:
            blockers.append(f"missing_dependency_kind:{required_kind}")

    return ProgramAReadinessReport(
        experiment_id=spec.experiment_id,
        spec_sha256=_safe_spec_sha256(spec),
        dependency_manifest_sha256=dependency_manifest.sha256,
        blockers=tuple(sorted(set(blockers))),
    )


def assemble_program_a_lock(
    spec: ExperimentSpec,
    dependency_manifest: DependencyManifest,
    readiness: ProgramAReadinessReport,
    *,
    code_ref: str,
    runtime_sha256: str,
) -> ExperimentLock:
    current = build_readiness_report(spec, dependency_manifest)
    if readiness.sha256 != current.sha256:
        raise ValueError("readiness report does not match the current spec/dependency manifest")
    if not current.ready_to_lock:
        raise ValueError("Program A experiment is BLOCKED and cannot be locked")
    lock = ExperimentLock(spec=spec, code_ref=code_ref, runtime_sha256=runtime_sha256)
    lock.validate()
    return lock


def _safe_spec_sha256(spec: ExperimentSpec) -> str:
    try:
        return spec.sha256
    except (TypeError, ValueError):
        payload = dataclasses.asdict(spec)
        return canonical_sha256(payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_spec(path: Path) -> ExperimentSpec:
    return ExperimentSpec.from_dict(load_json(path))


def _load_manifest(path: Path) -> DependencyManifest:
    return DependencyManifest.from_dict(load_json(path))


def _preflight_main(args: argparse.Namespace) -> int:
    spec = _load_spec(Path(args.spec))
    manifest = _load_manifest(Path(args.dependencies))
    report = build_readiness_report(spec, manifest)
    _write_json(Path(args.output), report.to_dict())
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


def _lock_main(args: argparse.Namespace) -> int:
    spec = _load_spec(Path(args.spec))
    manifest = _load_manifest(Path(args.dependencies))
    readiness = ProgramAReadinessReport.from_dict(load_json(args.readiness))
    lock = assemble_program_a_lock(
        spec,
        manifest,
        readiness,
        code_ref=args.code_ref,
        runtime_sha256=args.runtime_sha256,
    )
    write_typed_artifact(args.output, lock)
    print(json.dumps({"status": "LOCKED", "lock_sha256": lock.sha256}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed Program A preregistration preflight and lock assembly."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--spec", required=True)
    preflight.add_argument("--dependencies", required=True)
    preflight.add_argument("--output", required=True)
    preflight.set_defaults(func=_preflight_main)

    lock = subparsers.add_parser("lock")
    lock.add_argument("--spec", required=True)
    lock.add_argument("--dependencies", required=True)
    lock.add_argument("--readiness", required=True)
    lock.add_argument("--code-ref", required=True)
    lock.add_argument("--runtime-sha256", required=True)
    lock.add_argument("--output", required=True)
    lock.set_defaults(func=_lock_main)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
