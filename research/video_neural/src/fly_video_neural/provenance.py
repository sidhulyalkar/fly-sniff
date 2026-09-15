from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

CRITICAL_SOURCE_FILES = (
    "alignment_null.py",
    "data_qc.py",
    "development_protocol.py",
    "final_protocol.py",
    "metrics.py",
    "provenance.py",
    "ridge.py",
    "session_benchmark.py",
    "validation_gate.py",
)
PREPARATION_SOURCE_FILES = (
    "alignment.py",
    "mc2p.py",
    "mc2p_legacy.py",
    "pose_neural.py",
    "prepare_v1.py",
    "provenance.py",
)


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(files: tuple[str, ...]) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema_version": 1,
        "files": {name: _file_sha(root / name) for name in files},
    }
    payload["sha256"] = _sha(payload)
    return payload


def implementation_fingerprint() -> dict[str, Any]:
    return _fingerprint(CRITICAL_SOURCE_FILES)


def preparation_implementation_fingerprint() -> dict[str, Any]:
    return _fingerprint(PREPARATION_SOURCE_FILES)


def runtime_fingerprint() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    payload["sha256"] = _sha(payload)
    return payload
