from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .freeze import canonical_sha256


_HEX = frozenset("0123456789abcdef")


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")


class ExperimentProgram(str, Enum):
    """Scientific question represented by an experiment."""

    LATENT_WIRING = "latent_wiring"
    TOPOLOGY_INDUCTIVE_BIAS = "topology_inductive_bias"
    BIOLOGICAL_LEARNING = "biological_learning"


class ExperimentPhase(str, Enum):
    DEVELOPMENT = "development"
    CONFIRMATORY = "confirmatory"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class ArtifactRef:
    """Content-addressed scientific input or output."""

    name: str
    kind: str
    sha256: str
    uri: str

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("artifact name must be non-empty")
        if not self.kind.strip():
            raise ValueError(f"artifact {self.name}: kind must be non-empty")
        if not self.uri.strip():
            raise ValueError(f"artifact {self.name}: uri must be non-empty")
        _validate_sha256(self.sha256, field=f"artifact {self.name} sha256")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "name": self.name,
            "kind": self.kind,
            "sha256": self.sha256,
            "uri": self.uri,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactRef:
        artifact = cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            sha256=str(payload["sha256"]),
            uri=str(payload["uri"]),
        )
        artifact.validate()
        return artifact


@dataclass(frozen=True)
class TrainingPolicy:
    """What optimization freedom exists before evaluation."""

    navigation_reward_allowed: bool
    topology_specific_fit_allowed: bool
    equal_budget_across_topologies: bool
    whole_graph_backprop_allowed: bool
    calibration_targets: tuple[str, ...] = ()
    plasticity_scope: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "navigation_reward_allowed": self.navigation_reward_allowed,
            "topology_specific_fit_allowed": self.topology_specific_fit_allowed,
            "equal_budget_across_topologies": self.equal_budget_across_topologies,
            "whole_graph_backprop_allowed": self.whole_graph_backprop_allowed,
            "calibration_targets": list(self.calibration_targets),
            "plasticity_scope": list(self.plasticity_scope),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingPolicy:
        return cls(
            navigation_reward_allowed=bool(payload["navigation_reward_allowed"]),
            topology_specific_fit_allowed=bool(payload["topology_specific_fit_allowed"]),
            equal_budget_across_topologies=bool(payload["equal_budget_across_topologies"]),
            whole_graph_backprop_allowed=bool(payload["whole_graph_backprop_allowed"]),
            calibration_targets=tuple(str(x) for x in payload.get("calibration_targets", [])),
            plasticity_scope=tuple(str(x) for x in payload.get("plasticity_scope", [])),
        )


@dataclass(frozen=True)
class FinalPolicy:
    """Predeclared rules for claim-bearing evaluation."""

    topology_claim: bool
    topology_null_count: int
    paired_episodes_required: bool
    hidden_final_entropy_required: bool
    negative_results_retained: bool
    one_way_final: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_claim": self.topology_claim,
            "topology_null_count": self.topology_null_count,
            "paired_episodes_required": self.paired_episodes_required,
            "hidden_final_entropy_required": self.hidden_final_entropy_required,
            "negative_results_retained": self.negative_results_retained,
            "one_way_final": self.one_way_final,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FinalPolicy:
        return cls(
            topology_claim=bool(payload["topology_claim"]),
            topology_null_count=int(payload["topology_null_count"]),
            paired_episodes_required=bool(payload["paired_episodes_required"]),
            hidden_final_entropy_required=bool(payload["hidden_final_entropy_required"]),
            negative_results_retained=bool(payload["negative_results_retained"]),
            one_way_final=bool(payload["one_way_final"]),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    """Machine-readable scientific contract before a run is sealed."""

    experiment_id: str
    program: ExperimentProgram
    phase: ExperimentPhase
    scientific_question: str
    evidence_ledger_sha256: str
    artifacts: tuple[ArtifactRef, ...]
    null_families: tuple[str, ...]
    interventions: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    training_policy: TrainingPolicy
    final_policy: FinalPolicy
    claim_boundary: tuple[str, ...] = ()
    schema: str = "fly-sniff-experiment-spec-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-experiment-spec-v1":
            raise ValueError(f"unsupported experiment spec schema: {self.schema}")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.scientific_question.strip():
            raise ValueError("scientific_question must be non-empty")
        _validate_sha256(self.evidence_ledger_sha256, field="evidence_ledger_sha256")
        if not self.primary_metrics:
            raise ValueError("at least one primary metric is required")

        artifact_names: set[str] = set()
        for artifact in self.artifacts:
            artifact.validate()
            if artifact.name in artifact_names:
                raise ValueError(f"duplicate artifact name: {artifact.name}")
            artifact_names.add(artifact.name)

        policy = self.training_policy
        if policy.whole_graph_backprop_allowed:
            raise ValueError(
                "whole-graph backpropagation is outside the credible v1 experiment programs"
            )

        if self.program is ExperimentProgram.LATENT_WIRING:
            if policy.navigation_reward_allowed:
                raise ValueError("latent-wiring experiments forbid navigation reward")
            if policy.topology_specific_fit_allowed:
                raise ValueError("latent-wiring experiments forbid topology-specific fitting")
            if not policy.calibration_targets:
                raise ValueError(
                    "latent-wiring experiments require independent calibration targets"
                )

        if self.program is ExperimentProgram.TOPOLOGY_INDUCTIVE_BIAS:
            if policy.topology_specific_fit_allowed and not policy.equal_budget_across_topologies:
                raise ValueError(
                    "topology-specific fitting requires equal optimization budgets across topologies"
                )

        if self.program is ExperimentProgram.BIOLOGICAL_LEARNING:
            if not policy.plasticity_scope:
                raise ValueError(
                    "biological-learning experiments require an explicit plasticity_scope"
                )

        final = self.final_policy
        if final.topology_null_count < 0:
            raise ValueError("topology_null_count must be non-negative")
        if self.phase is ExperimentPhase.CONFIRMATORY:
            if not final.negative_results_retained:
                raise ValueError("confirmatory experiments must retain negative results")
            if not final.one_way_final:
                raise ValueError("confirmatory experiments require a one-way final run")
            if not final.hidden_final_entropy_required:
                raise ValueError(
                    "confirmatory experiments require hidden final entropy before model freeze"
                )
            if final.topology_claim:
                if final.topology_null_count < 31:
                    raise ValueError(
                        "confirmatory topology claims require at least 31 frozen topology nulls"
                    )
                if not final.paired_episodes_required:
                    raise ValueError(
                        "confirmatory topology claims require paired episode conditions"
                    )
                if not self.null_families:
                    raise ValueError("topology claims require at least one null family")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "program": self.program.value,
            "phase": self.phase.value,
            "scientific_question": self.scientific_question,
            "evidence_ledger_sha256": self.evidence_ledger_sha256,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "null_families": list(self.null_families),
            "interventions": list(self.interventions),
            "primary_metrics": list(self.primary_metrics),
            "training_policy": self.training_policy.to_dict(),
            "final_policy": self.final_policy.to_dict(),
            "claim_boundary": list(self.claim_boundary),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["spec_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentSpec:
        spec = cls(
            experiment_id=str(payload["experiment_id"]),
            program=ExperimentProgram(payload["program"]),
            phase=ExperimentPhase(payload["phase"]),
            scientific_question=str(payload["scientific_question"]),
            evidence_ledger_sha256=str(payload["evidence_ledger_sha256"]),
            artifacts=tuple(ArtifactRef.from_dict(item) for item in payload.get("artifacts", [])),
            null_families=tuple(str(x) for x in payload.get("null_families", [])),
            interventions=tuple(str(x) for x in payload.get("interventions", [])),
            primary_metrics=tuple(str(x) for x in payload.get("primary_metrics", [])),
            training_policy=TrainingPolicy.from_dict(payload["training_policy"]),
            final_policy=FinalPolicy.from_dict(payload["final_policy"]),
            claim_boundary=tuple(str(x) for x in payload.get("claim_boundary", [])),
            schema=str(payload.get("schema", "fly-sniff-experiment-spec-v1")),
        )
        spec.validate()
        claimed_hash = payload.get("spec_sha256")
        if claimed_hash is not None and claimed_hash != spec.sha256:
            raise ValueError("experiment spec hash mismatch")
        return spec


@dataclass(frozen=True)
class ExperimentLock:
    """Content-addressed experiment contract bound to an exact code/runtime state."""

    spec: ExperimentSpec
    code_ref: str
    runtime_sha256: str
    schema: str = "fly-sniff-experiment-lock-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-experiment-lock-v1":
            raise ValueError(f"unsupported experiment lock schema: {self.schema}")
        self.spec.validate()
        if not self.code_ref.strip():
            raise ValueError("experiment lock code_ref must be non-empty")
        _validate_sha256(self.runtime_sha256, field="runtime_sha256")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "spec": self.spec.to_dict(),
            "code_ref": self.code_ref,
            "runtime_sha256": self.runtime_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["lock_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentLock:
        lock = cls(
            spec=ExperimentSpec.from_dict(payload["spec"]),
            code_ref=str(payload["code_ref"]),
            runtime_sha256=str(payload["runtime_sha256"]),
            schema=str(payload.get("schema", "fly-sniff-experiment-lock-v1")),
        )
        lock.validate()
        claimed_hash = payload.get("lock_sha256")
        if claimed_hash is not None and claimed_hash != lock.sha256:
            raise ValueError("experiment lock hash mismatch")
        return lock


@dataclass(frozen=True)
class RunReceipt:
    """Hash-bound result record tied to one exact experiment lock."""

    run_id: str
    lock_sha256: str
    status: RunStatus
    result_artifacts: tuple[ArtifactRef, ...] = ()
    metric_summary: tuple[tuple[str, float], ...] = ()
    notes: str | None = None
    schema: str = "fly-sniff-run-receipt-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-run-receipt-v1":
            raise ValueError(f"unsupported run receipt schema: {self.schema}")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        _validate_sha256(self.lock_sha256, field="lock_sha256")
        artifact_names: set[str] = set()
        for artifact in self.result_artifacts:
            artifact.validate()
            if artifact.name in artifact_names:
                raise ValueError(f"duplicate result artifact name: {artifact.name}")
            artifact_names.add(artifact.name)
        metric_names = [name for name, _ in self.metric_summary]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("metric_summary contains duplicate metric names")
        if self.status is RunStatus.COMPLETED and not self.result_artifacts:
            raise ValueError("completed runs require at least one content-addressed result artifact")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "schema": self.schema,
            "run_id": self.run_id,
            "lock_sha256": self.lock_sha256,
            "status": self.status.value,
            "result_artifacts": [artifact.to_dict() for artifact in self.result_artifacts],
            "metric_summary": {
                name: value for name, value in sorted(self.metric_summary, key=lambda item: item[0])
            },
        }
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["receipt_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunReceipt:
        metrics = payload.get("metric_summary", {})
        receipt = cls(
            run_id=str(payload["run_id"]),
            lock_sha256=str(payload["lock_sha256"]),
            status=RunStatus(payload["status"]),
            result_artifacts=tuple(
                ArtifactRef.from_dict(item) for item in payload.get("result_artifacts", [])
            ),
            metric_summary=tuple((str(name), float(value)) for name, value in metrics.items()),
            notes=payload.get("notes"),
            schema=str(payload.get("schema", "fly-sniff-run-receipt-v1")),
        )
        receipt.validate()
        claimed_hash = payload.get("receipt_sha256")
        if claimed_hash is not None and claimed_hash != receipt.sha256:
            raise ValueError("run receipt hash mismatch")
        return receipt
