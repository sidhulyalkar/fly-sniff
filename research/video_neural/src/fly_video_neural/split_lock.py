from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import SampleWindow
from .splits import make_animal_disjoint_splits, sample_split_map


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_window_manifests(paths: Iterable[str | Path]) -> list[SampleWindow]:
    samples: list[SampleWindow] = []
    for path in paths:
        document = json.loads(Path(path).read_text())
        if document.get("dataset_id") != "mc2p_v1":
            raise ValueError(f"window manifest {path} is not mc2p_v1")
        for window in document.get("windows", []):
            samples.append(SampleWindow.from_dict(window["sample"]))
    if not samples:
        raise ValueError("no synchronized sample windows found")
    return samples


def build_split_lock(windows: Iterable[SampleWindow], *, seed: int = 1701) -> dict[str, Any]:
    rows = sorted(windows, key=lambda row: row.sample_id)
    sample_ids = [row.sample_id for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample ids must be globally unique before split locking")
    if {row.dataset_id for row in rows} != {"mc2p_v1"}:
        raise ValueError("v0 split lock accepts only mc2p_v1 windows")
    animal_to_split = make_animal_disjoint_splits(rows, seed=seed)
    sample_to_split = sample_split_map(rows, animal_to_split)
    split_counts = Counter(sample_to_split.values())
    animal_counts = Counter(animal_to_split.values())
    canonical_samples = [row.to_dict() for row in rows]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "mc2p_future_neural_v0",
        "split_seed": seed,
        "split_unit": "animal_id",
        "source_samples_sha256": _sha(canonical_samples),
        "animal_to_split": dict(sorted(animal_to_split.items())),
        "animal_counts": dict(sorted(animal_counts.items())),
        "sample_counts": dict(sorted(split_counts.items())),
        "sample_to_split": dict(sorted(sample_to_split.items())),
        "hyperparameter_selection_may_use_test": False,
    }
    payload["split_lock_sha256"] = _sha(payload)
    return payload


def verify_split_lock(lock: dict[str, Any], windows: Iterable[SampleWindow]) -> None:
    supplied = dict(lock)
    claimed = supplied.pop("split_lock_sha256", None)
    if claimed != _sha(supplied):
        raise ValueError("split lock self-hash mismatch")
    rows = sorted(windows, key=lambda row: row.sample_id)
    if lock.get("source_samples_sha256") != _sha([row.to_dict() for row in rows]):
        raise ValueError("split lock does not match synchronized sample windows")
    if lock.get("hyperparameter_selection_may_use_test") is not False:
        raise ValueError("test split cannot be used for hyperparameter selection")
    expected = sample_split_map(rows, lock["animal_to_split"])
    if lock.get("sample_to_split") != dict(sorted(expected.items())):
        raise ValueError("split lock sample assignments do not follow animal assignments")
