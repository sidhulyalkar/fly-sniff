from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .experimental_plume import (
    PROTOCOL,
    file_sha256,
    load_experimental_plume_config,
    upstream_contract_sha256,
)

ARCHIVE_PROTOCOL = "experimental-plume-archive-inspection-v3"
_TIME_DATASET_NAMES = ("timestamps", "frame_times", "time", "times")
_RATE_KEYS = ("fps", "frame_rate", "rate", "sampling_rate")


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if np.isscalar(value):
        return value
    return str(value)


def _attrs(obj: h5py.Dataset | h5py.Group) -> dict[str, Any]:
    return {str(key): _json_scalar(value) for key, value in obj.attrs.items()}


def _timestamps_basis(parent: h5py.Group, frame_count: int) -> dict[str, Any] | None:
    for name in _TIME_DATASET_NAMES:
        if name not in parent or not isinstance(parent[name], h5py.Dataset):
            continue
        values = np.asarray(parent[name][...], dtype=np.float64).reshape(-1)
        if len(values) != frame_count or not np.all(np.isfinite(values)):
            continue
        differences = np.diff(values)
        if len(differences) == 0 or not np.all(differences > 0):
            continue
        median_dt = float(np.median(differences))
        return {
            "kind": "explicit_timestamps",
            "evidence": "file_internal",
            "dataset_path": parent[name].name,
            "frame_count": frame_count,
            "median_dt_seconds": median_dt,
            "median_fps": 1.0 / median_dt,
            "max_abs_dt_deviation_seconds": float(np.max(np.abs(differences - median_dt))),
            "start_seconds": float(values[0]),
            "stop_seconds": float(values[-1]),
        }
    return None


def _rate_from_attrs(dataset: h5py.Dataset, parent: h5py.Group) -> dict[str, Any] | None:
    for location, obj in (("dataset", dataset), ("parent_group", parent)):
        for key in _RATE_KEYS:
            if key not in obj.attrs:
                continue
            try:
                rate = float(np.asarray(obj.attrs[key]).reshape(()))
            except (TypeError, ValueError):
                continue
            if np.isfinite(rate) and rate > 0:
                return {
                    "kind": "fixed_rate",
                    "evidence": "file_internal_attribute",
                    "attribute_location": location,
                    "attribute_name": key,
                    "fps": rate,
                    "dt_seconds": 1.0 / rate,
                }
    if "starting_time" in parent and isinstance(parent["starting_time"], h5py.Dataset):
        starting_time = parent["starting_time"]
        if "rate" in starting_time.attrs:
            rate = float(np.asarray(starting_time.attrs["rate"]).reshape(()))
            if np.isfinite(rate) and rate > 0:
                return {
                    "kind": "fixed_rate",
                    "evidence": "nwb_starting_time_rate",
                    "dataset_path": starting_time.name,
                    "fps": rate,
                    "dt_seconds": 1.0 / rate,
                }
    return None


def _complex_time_basis(dataset: h5py.Dataset) -> dict[str, Any]:
    parent = dataset.parent
    return (
        _timestamps_basis(parent, int(dataset.shape[0]))
        or _rate_from_attrs(dataset, parent)
        or {
            "kind": "unresolved",
            "evidence": "none_found_in_dataset_or_parent",
            "searched_timestamp_datasets": list(_TIME_DATASET_NAMES),
            "searched_rate_attributes": list(_RATE_KEYS),
            "parent_group": parent.name,
        }
    )


def inspect_experimental_plume_archive(
    source_file: str | Path,
    plume: str,
    document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document = load_experimental_plume_config() if document is None else document
    if plume not in {"smooth", "complex"}:
        raise ValueError("plume must be 'smooth' or 'complex'")
    source = document[plume]
    expected_path = source["dataset"]
    expected_shape = tuple(int(value) for value in source["expected_shape"])
    source_path = Path(source_file)
    source_sha = file_sha256(source_path)

    with h5py.File(source_path, "r") as archive:
        if expected_path not in archive:
            raise ValueError(f"required plume dataset {expected_path!r} is absent from archive")
        dataset = archive[expected_path]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"required plume path {expected_path!r} is not an HDF5 dataset")
        actual_shape = tuple(int(value) for value in dataset.shape)
        if actual_shape != expected_shape:
            raise ValueError(f"plume dataset shape mismatch: expected {expected_shape}, got {actual_shape}")
        dataset_receipt = {
            "path": dataset.name,
            "shape": list(actual_shape),
            "dtype": str(dataset.dtype),
            "chunks": None if dataset.chunks is None else list(dataset.chunks),
            "compression": dataset.compression,
            "attributes": _attrs(dataset),
            "parent_attributes": _attrs(dataset.parent),
        }
        if plume == "smooth":
            rate = float(source["source_fps"])
            time_basis = {
                "kind": "fixed_rate",
                "evidence": "frozen_public_source_contract",
                "fps": rate,
                "dt_seconds": 1.0 / rate,
                "measurement_interval_ms": 1000.0 / rate,
                "file_internal_timing": False,
                "note": "Native 15 Hz measurements only; interpolated publication frames are not new measurements.",
            }
            status = "archive_qualified_for_publication_reproduction"
            timing_allowed = True
        else:
            time_basis = _complex_time_basis(dataset)
            timing_allowed = time_basis["kind"] != "unresolved"
            status = "archive_qualified" if timing_allowed else "blocked_missing_native_time_basis"

    report = {
        "schema_version": 1,
        "protocol": ARCHIVE_PROTOCOL,
        "source_contract_protocol": PROTOCOL,
        "plume": plume,
        "status": status,
        "source_file_name": source_path.name,
        "source_sha256": source_sha,
        "source_size_bytes": source_path.stat().st_size,
        "source_dataset": dataset_receipt,
        "native_time_basis": time_basis,
        "source_contract_sha256": upstream_contract_sha256(document),
        "navigation_performance_used": False,
        "controller_access": False,
        "native_sensory_timing_allowed": timing_allowed,
        "claim_boundary": (
            "Archive identity, dataset structure, and available time-basis evidence only; "
            "no physical pixel-to-fly mapping, biological dynamics, or navigation performance."
        ),
    }
    report["report_sha256"] = _canonical_sha(report)
    return report


def write_archive_report(
    source_file: str | Path,
    plume: str,
    output: str | Path,
    document: dict[str, Any] | None = None,
) -> Path:
    report = inspect_experimental_plume_archive(source_file, plume, document)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect experimental-plume HDF5/NWB bytes")
    parser.add_argument("--config")
    parser.add_argument("--plume", choices=("smooth", "complex"), required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    document = load_experimental_plume_config(args.config)
    path = write_archive_report(args.source_file, args.plume, args.output, document)
    report = json.loads(path.read_text())
    print(path)
    print(report["report_sha256"])
    print(report["status"])


if __name__ == "__main__":
    main()
