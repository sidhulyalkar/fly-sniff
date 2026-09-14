from __future__ import annotations

import hashlib
import json
import pickle
import pickletools
from pathlib import Path
from typing import Any

import numpy as np

ALIGNMENT_KEYS = (
    "indices",
    "sync_indices",
    "alignment",
    "behavior_to_neural",
    "behav_to_neural",
    "neural_indices",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pickle_opcodes(path: str | Path) -> dict[str, Any]:
    global_refs: list[str] = []
    stack_global_count = 0
    with Path(path).open("rb") as handle:
        try:
            for opcode, arg, _ in pickletools.genops(handle):
                if opcode.name == "GLOBAL" and isinstance(arg, str):
                    global_refs.append(arg)
                elif opcode.name == "STACK_GLOBAL":
                    stack_global_count += 1
        except Exception as exc:
            raise ValueError(f"cannot parse pickle opcode stream: {exc}") from exc
    return {
        "global_references": sorted(set(global_refs)),
        "stack_global_opcode_count": stack_global_count,
        "security_note": (
            "Opcode inspection is provenance metadata only and does not make pickle safe. "
            "Deserialization occurs only after explicit trust acknowledgement."
        ),
    }


def _trusted_pickle_load(path: Path, *, trust_upstream_pickle: bool) -> Any:
    if not trust_upstream_pickle:
        raise ValueError(
            "refusing to deserialize pickle without --trust-upstream-pickle; "
            "pickle can execute code during loading"
        )
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - explicit trust gate is the converter boundary


def _as_alignment(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        present = [key for key in ALIGNMENT_KEYS if key in value]
        if len(present) != 1:
            keys = sorted(map(str, value.keys()))[:20]
            raise ValueError(
                "alignment pickle dict must contain exactly one recognized alignment key; "
                f"recognized keys present={present}, available keys={keys}"
            )
        value = value[present[0]]
    array = np.asarray(value)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("alignment must be a one-dimensional array with at least two frames")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("alignment values must be numeric")
    if not np.isfinite(array).all():
        raise ValueError("alignment contains non-finite values")
    if not np.all(array == np.floor(array)):
        raise ValueError("alignment values must be integer neural frame indices")
    normalized = np.asarray(array, dtype=np.int64)
    if (normalized < 0).any():
        raise ValueError("alignment cannot contain negative neural frame indices")
    if (np.diff(normalized) < 0).any():
        raise ValueError("behavior-to-neural alignment must be monotonic non-decreasing")
    return normalized


def _as_pose3d(value: Any) -> np.ndarray:
    if not isinstance(value, dict) or "points3d" not in value:
        keys = sorted(map(str, value.keys()))[:20] if isinstance(value, dict) else []
        raise ValueError(
            "pose pickle must be a dict containing points3d; "
            f"top-level type={type(value).__name__}, available keys={keys}"
        )
    array = np.asarray(value["points3d"])
    if array.ndim != 3 or array.shape[1:] != (38, 3):
        raise ValueError(
            "points3d must have shape [behavior_frames, 38, 3]; "
            f"observed shape={tuple(array.shape)}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("points3d must contain finite numeric values")
    return np.asarray(array, dtype=np.float32)


def convert_legacy_pickle(
    source: str | Path,
    output: str | Path,
    receipt: str | Path,
    *,
    kind: str,
    trust_upstream_pickle: bool = False,
) -> dict[str, Any]:
    source_path = Path(source)
    output_path = Path(output)
    receipt_path = Path(receipt)
    if source_path.suffix.lower() not in {".pkl", ".pickle"}:
        raise ValueError("legacy converter accepts only .pkl/.pickle sources")
    if kind not in {"alignment", "pose3d"}:
        raise ValueError("kind must be 'alignment' or 'pose3d'")
    source_sha = sha256_file(source_path)
    opcode_report = inspect_pickle_opcodes(source_path)
    loaded = _trusted_pickle_load(source_path, trust_upstream_pickle=trust_upstream_pickle)
    converted = _as_alignment(loaded) if kind == "alignment" else _as_pose3d(loaded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, converted, allow_pickle=False)
    output_sha = sha256_file(output_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "mc2p-legacy-conversion-v1",
        "dataset_id": "mc2p_v1",
        "kind": kind,
        "source_format": "python_pickle",
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha,
        "source_size_bytes": source_path.stat().st_size,
        "trust_acknowledged": bool(trust_upstream_pickle),
        "pickle_opcode_inspection": opcode_report,
        "output_format": "numpy_npy_allow_pickle_false",
        "output_path": str(output_path.resolve()),
        "output_sha256": output_sha,
        "output_shape": list(converted.shape),
        "output_dtype": str(converted.dtype),
        "claim_boundary": (
            "Conversion establishes file provenance and normalized array structure only; "
            "it does not validate biological correctness or synchronization accuracy."
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
