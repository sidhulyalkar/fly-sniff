from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

RUNTIME_PROTOCOL = "fly-sniff-numerical-runtime-v1"
NUMERICAL_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "networkx",
    "matplotlib",
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in NUMERICAL_DISTRIBUTIONS:
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"required numerical dependency {distribution!r} is not installed"
            ) from exc
    return packages


def numerical_compatibility_identity() -> dict[str, Any]:
    """Return the runtime fields allowed to affect deterministic numerical replay."""
    version_info = sys.version_info
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": f"{version_info.major}.{version_info.minor}.{version_info.micro}",
        "packages": _package_versions(),
    }


def numerical_compatibility_sha256() -> str:
    return _canonical_sha256(numerical_compatibility_identity())


def runtime_environment_receipt() -> dict[str, Any]:
    """Record numerical identity plus non-gating platform diagnostics."""
    numerical = numerical_compatibility_identity()
    return {
        "protocol": RUNTIME_PROTOCOL,
        "numerical_compatibility": numerical,
        "numerical_compatibility_sha256": _canonical_sha256(numerical),
        "platform_diagnostics": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_executable_implementation": platform.python_implementation(),
        },
    }


def verify_runtime_environment_receipt(
    receipt: dict[str, Any],
    *,
    require_current_numerical_match: bool = True,
) -> str:
    """Verify receipt integrity and optionally require exact current numerical identity.

    Platform fields are intentionally diagnostic only. The compatibility gate binds the
    exact Python patch version and exact numerical-library versions because task training,
    stochastic seed generation, and deterministic replay rely on that numerical stack.
    """
    if not isinstance(receipt, dict):
        raise TypeError("runtime environment receipt must be a mapping")
    if receipt.get("protocol") != RUNTIME_PROTOCOL:
        raise ValueError("runtime environment receipt protocol mismatch")
    numerical = receipt.get("numerical_compatibility")
    if not isinstance(numerical, dict):
        raise TypeError("runtime environment receipt is missing numerical compatibility fields")
    observed_hash = _canonical_sha256(numerical)
    if receipt.get("numerical_compatibility_sha256") != observed_hash:
        raise ValueError("runtime numerical compatibility hash mismatch")

    packages = numerical.get("packages")
    if not isinstance(packages, dict):
        raise TypeError("runtime numerical compatibility receipt is missing package versions")
    if set(packages) != set(NUMERICAL_DISTRIBUTIONS):
        raise ValueError("runtime numerical dependency set differs from the frozen v1 contract")
    if not all(isinstance(value, str) and value for value in packages.values()):
        raise ValueError("runtime numerical package versions must be non-empty strings")

    if require_current_numerical_match:
        current = numerical_compatibility_identity()
        if numerical != current:
            raise RuntimeError(
                "current numerical runtime differs from the runtime that produced the training "
                "artifact; deterministic promotion is refused"
            )
    return observed_hash
