from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .evidence import EvidenceLedger
from .experiment import ExperimentLock, ExperimentSpec, RunReceipt


TypedArtifact = EvidenceLedger | ExperimentSpec | ExperimentLock | RunReceipt


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of exact file bytes."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object top-level values."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{source}: expected a JSON object")
    return payload


def load_typed_artifact(path: str | Path) -> TypedArtifact:
    """Load and validate a kernel artifact from its explicit schema tag."""

    payload = load_json(path)
    schema = payload.get("schema")
    if schema == "fly-sniff-evidence-ledger-v1":
        return EvidenceLedger.from_dict(payload)
    if schema == "fly-sniff-experiment-spec-v1":
        return ExperimentSpec.from_dict(payload)
    if schema == "fly-sniff-experiment-lock-v1":
        return ExperimentLock.from_dict(payload)
    if schema == "fly-sniff-run-receipt-v1":
        return RunReceipt.from_dict(payload)
    raise ValueError(f"unsupported kernel artifact schema: {schema!r}")


def _artifact_payload(artifact: TypedArtifact) -> dict[str, Any]:
    if isinstance(artifact, EvidenceLedger | ExperimentSpec | ExperimentLock | RunReceipt):
        return artifact.to_dict()
    raise TypeError(f"unsupported artifact type: {type(artifact)!r}")


def write_typed_artifact(
    path: str | Path,
    artifact: TypedArtifact,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a validated artifact using canonical human-readable JSON.

    The scientific identity remains the artifact's canonical embedded hash. The exact
    file-byte hash can additionally be obtained with :func:`file_sha256` when a
    manifest needs to bind the serialized representation itself.
    """

    target = Path(path)
    payload = _artifact_payload(artifact)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")

    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return target
