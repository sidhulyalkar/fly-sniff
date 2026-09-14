from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SPEC_PROTOCOL = "fly-sniff-experiment-spec-v1"
LOCK_PROTOCOL = "fly-sniff-experiment-lock-v1"
RECEIPT_PROTOCOL = "fly-sniff-run-receipt-v1"


def canonical_sha256(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset: dict[str, Any]
    circuit: dict[str, Any]
    evidence: dict[str, Any]
    dynamics: dict[str, Any]
    environment: dict[str, Any]
    interventions: list[dict[str, Any]]
    nulls: dict[str, Any]
    metrics: dict[str, Any]
    training: dict[str, Any]
    final: dict[str, Any]
    protocol: str = SPEC_PROTOCOL
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.protocol != SPEC_PROTOCOL:
            raise ValueError(f"unexpected experiment spec protocol: {self.protocol!r}")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        for name in ("dataset", "circuit", "evidence", "dynamics", "environment", "nulls", "metrics", "training", "final"):
            if not getattr(self, name):
                raise ValueError(f"experiment section {name!r} must be non-empty")
        if not self.interventions:
            raise ValueError("experiment must define at least one intervention/control")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentSpec:
        spec = cls(
            experiment_id=str(payload["experiment_id"]),
            dataset=dict(payload["dataset"]),
            circuit=dict(payload["circuit"]),
            evidence=dict(payload["evidence"]),
            dynamics=dict(payload["dynamics"]),
            environment=dict(payload["environment"]),
            interventions=[dict(row) for row in payload["interventions"]],
            nulls=dict(payload["nulls"]),
            metrics=dict(payload["metrics"]),
            training=dict(payload["training"]),
            final=dict(payload["final"]),
            protocol=str(payload.get("protocol", "")),
            metadata=dict(payload.get("metadata", {})),
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ExperimentLock:
    experiment_id: str
    spec_sha256: str
    artifact_sha256: dict[str, str]
    code_ref: str
    runtime: dict[str, Any]
    protocol: str = LOCK_PROTOCOL

    def to_dict(self, *, include_lock_sha256: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if include_lock_sha256:
            payload["lock_sha256"] = canonical_sha256(payload)
        return payload

    def validate(self) -> None:
        if self.protocol != LOCK_PROTOCOL:
            raise ValueError(f"unexpected experiment lock protocol: {self.protocol!r}")
        if len(self.spec_sha256) != 64:
            raise ValueError("spec_sha256 must be a SHA-256 digest")
        if not self.artifact_sha256:
            raise ValueError("experiment lock requires at least one artifact hash")
        for name, value in self.artifact_sha256.items():
            if not name or len(value) != 64:
                raise ValueError("artifact hashes must use non-empty names and SHA-256 digests")
        if not self.code_ref.strip():
            raise ValueError("code_ref must be non-empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, verify_sha256: bool = True) -> ExperimentLock:
        expected = payload.get("lock_sha256")
        bare = {key: value for key, value in payload.items() if key != "lock_sha256"}
        if verify_sha256 and expected is not None:
            actual = canonical_sha256(bare)
            if str(expected) != actual:
                raise ValueError(f"experiment lock SHA-256 mismatch: expected {expected}, got {actual}")
        lock = cls(
            experiment_id=str(bare["experiment_id"]),
            spec_sha256=str(bare["spec_sha256"]),
            artifact_sha256={str(k): str(v) for k, v in bare["artifact_sha256"].items()},
            code_ref=str(bare["code_ref"]),
            runtime=dict(bare["runtime"]),
            protocol=str(bare.get("protocol", "")),
        )
        lock.validate()
        return lock


@dataclass(frozen=True)
class RunReceipt:
    run_id: str
    experiment_lock_sha256: str
    variant: str
    seeds: list[int]
    result_artifact_sha256: dict[str, str]
    runtime: dict[str, Any]
    protocol: str = RECEIPT_PROTOCOL
    summary: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.protocol != RECEIPT_PROTOCOL:
            raise ValueError(f"unexpected run receipt protocol: {self.protocol!r}")
        if not self.run_id.strip() or not self.variant.strip():
            raise ValueError("run_id and variant must be non-empty")
        if len(self.experiment_lock_sha256) != 64:
            raise ValueError("experiment_lock_sha256 must be a SHA-256 digest")
        if not self.seeds:
            raise ValueError("run receipt requires at least one seed")
        if not self.result_artifact_sha256:
            raise ValueError("run receipt requires at least one result artifact hash")

    def to_dict(self, *, include_receipt_sha256: bool = True) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        if include_receipt_sha256:
            payload["receipt_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, verify_sha256: bool = True) -> RunReceipt:
        expected = payload.get("receipt_sha256")
        bare = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        if verify_sha256 and expected is not None:
            actual = canonical_sha256(bare)
            if str(expected) != actual:
                raise ValueError(f"run receipt SHA-256 mismatch: expected {expected}, got {actual}")
        receipt = cls(
            run_id=str(bare["run_id"]),
            experiment_lock_sha256=str(bare["experiment_lock_sha256"]),
            variant=str(bare["variant"]),
            seeds=[int(x) for x in bare["seeds"]],
            result_artifact_sha256={str(k): str(v) for k, v in bare["result_artifact_sha256"].items()},
            runtime=dict(bare["runtime"]),
            protocol=str(bare.get("protocol", "")),
            summary=dict(bare.get("summary", {})),
        )
        receipt.validate()
        return receipt


def seal_experiment(
    spec: ExperimentSpec,
    *,
    artifact_paths: dict[str, str | Path],
    code_ref: str,
    runtime: dict[str, Any],
) -> ExperimentLock:
    spec_payload = spec.to_dict()
    lock = ExperimentLock(
        experiment_id=spec.experiment_id,
        spec_sha256=canonical_sha256(spec_payload),
        artifact_sha256={name: file_sha256(path) for name, path in sorted(artifact_paths.items())},
        code_ref=code_ref,
        runtime=dict(runtime),
    )
    lock.validate()
    return lock


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or seal fly-sniff experiment protocols")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("spec")

    verify_lock = sub.add_parser("verify-lock")
    verify_lock.add_argument("lock")

    verify_receipt = sub.add_parser("verify-receipt")
    verify_receipt.add_argument("receipt")

    args = parser.parse_args()
    if args.command == "validate":
        spec = ExperimentSpec.load(args.spec)
        print(f"valid=True experiment_id={spec.experiment_id} sha256={canonical_sha256(spec.to_dict())}")
    elif args.command == "verify-lock":
        lock = ExperimentLock.from_dict(json.loads(Path(args.lock).read_text()))
        print(f"valid=True experiment_id={lock.experiment_id} sha256={lock.to_dict()['lock_sha256']}")
    else:
        receipt = RunReceipt.from_dict(json.loads(Path(args.receipt).read_text()))
        print(f"valid=True run_id={receipt.run_id} sha256={receipt.to_dict()['receipt_sha256']}")


if __name__ == "__main__":
    main()
