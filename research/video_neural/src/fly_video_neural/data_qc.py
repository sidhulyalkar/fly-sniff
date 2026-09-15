from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .session_benchmark import (
    SessionBenchmarkBatch,
    load_session_batches,
    verify_development_projection,
)

SAMPLE_START = re.compile(r".+:(?P<start>\d+)-\d+$")


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_qc_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_qc_config(document)
    return document


def validate_qc_config(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("data QC requires schema_version=1")
    if document.get("protocol") != "mc2p-development-data-qc-v1":
        raise ValueError("unexpected data QC protocol")
    if document.get("benchmark_id") != "mc2p_future_neural_v1":
        raise ValueError("data QC is frozen to mc2p_future_neural_v1")
    if document.get("scope") != "train_and_validation_targets_only":
        raise ValueError("QC may summarize only train and validation targets")
    if document.get("test_target_values_may_be_summarized") is not False:
        raise ValueError("test target values must remain hidden from development QC")
    if document.get("minimum_windows_per_development_session") != 2:
        raise ValueError("minimum development-session window count is frozen to 2")
    if document.get("autocorrelation_lags_windows") != [1, 2, 4, 10, 20]:
        raise ValueError("autocorrelation lag grid changed")
    if document.get("diagnostics_may_redefine_primary_metric") is not False:
        raise ValueError("QC diagnostics cannot redefine the benchmark metric")


def _sample_start(sample_id: str) -> int:
    match = SAMPLE_START.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"sample id does not expose behavior-frame start: {sample_id!r}")
    return int(match.group("start"))


def _target_temporal_correlation(
    targets: np.ndarray, lag: int
) -> dict[str, float | int | None]:
    if lag < 1:
        raise ValueError("lag must be positive")
    if targets.shape[0] <= lag:
        return {"valid_targets": 0, "median_pearson_r": None, "mean_pearson_r": None}
    left = targets[:-lag]
    right = targets[lag:]
    left0 = left - left.mean(axis=0)
    right0 = right - right.mean(axis=0)
    denom = np.sqrt((left0 * left0).sum(axis=0) * (right0 * right0).sum(axis=0))
    valid = denom > 0
    if not valid.any():
        return {"valid_targets": 0, "median_pearson_r": None, "mean_pearson_r": None}
    corr = (left0[:, valid] * right0[:, valid]).sum(axis=0) / denom[valid]
    return {
        "valid_targets": int(valid.sum()),
        "median_pearson_r": float(np.median(corr)),
        "mean_pearson_r": float(np.mean(corr)),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    std = values.std(axis=0)
    return {
        "dimensions": int(values.shape[1]),
        "samples": int(values.shape[0]),
        "constant_dimensions": int((std == 0).sum()),
        "constant_fraction": float((std == 0).mean()),
        "median_dimension_std": float(np.median(std)),
        "p05_dimension_std": float(np.quantile(std, 0.05)),
        "p95_dimension_std": float(np.quantile(std, 0.95)),
    }


def audit_development_data(
    batch: SessionBenchmarkBatch,
    lock: dict[str, Any],
    config: dict[str, Any],
    *,
    source_batches: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    validate_qc_config(config)
    verify_development_projection(lock, batch, source_batches=source_batches)
    split_map = lock["sample_to_split"]
    split = np.asarray([split_map[str(sample)] for sample in batch.sample_ids])
    failures: list[str] = []
    session_rows: list[dict[str, Any]] = []
    autocorrelation_rows: list[dict[str, Any]] = []

    for session in sorted(set(batch.session_ids.tolist())):
        mask = batch.session_ids == session
        session_splits = sorted(set(split[mask].tolist()))
        if len(session_splits) != 1:
            failures.append(f"session {session!r} crosses split assignments")
            continue
        split_name = session_splits[0]
        if split_name not in {"train", "validation"}:
            failures.append(f"development projection unexpectedly contains {split_name!r} session")
            continue
        sample_count = int(mask.sum())
        row: dict[str, Any] = {
            "session_id": session,
            "animal_id": str(batch.animal_ids[mask][0]),
            "split": split_name,
            "sample_count": sample_count,
            "target_values_summarized": True,
        }
        if sample_count < config["minimum_windows_per_development_session"]:
            failures.append(
                f"development session {session!r} has only {sample_count} prediction windows"
            )
        starts = np.asarray([_sample_start(str(sample)) for sample in batch.sample_ids[mask]])
        order = np.argsort(starts)
        targets = batch.targets[mask][order]
        row["target_distribution"] = _distribution_summary(targets)
        for lag in config["autocorrelation_lags_windows"]:
            autocorrelation_rows.append(
                {
                    "session_id": session,
                    "animal_id": row["animal_id"],
                    "split": split_name,
                    "lag_windows": lag,
                    "lag_seconds": lag * config["window_stride_s"],
                    **_target_temporal_correlation(targets, lag),
                }
            )
        session_rows.append(row)

    if len(batch.sample_ids) == 0:
        failures.append("no train/validation samples available for QC")
    test_sample_count = sum(
        split_name == "test" for split_name in lock["sample_to_split"].values()
    )
    test_session_count = sum(
        len(assignments["test"]) for assignments in lock["animal_sessions"].values()
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "benchmark_id": config["benchmark_id"],
        "status": "pass" if not failures else "blocked",
        "split_lock_sha256": lock["split_lock_sha256"],
        "qc_config_sha256": _sha(config),
        "source_batches": sorted(source_batches or [], key=lambda row: row["path"]),
        "development_sample_count": int(len(batch.sample_ids)),
        "test_sample_count_metadata_only": int(test_sample_count),
        "test_session_count_metadata_only": int(test_session_count),
        "test_target_arrays_deserialized": False,
        "test_target_values_summarized": False,
        "structural_failures": failures,
        "development_diagnostics": {
            "feature_distribution": _distribution_summary(batch.features),
            "target_distribution": _distribution_summary(batch.targets),
            "autocorrelation": autocorrelation_rows,
        },
        "sessions": session_rows,
        "interpretation": (
            "Variance and temporal-autocorrelation diagnostics are descriptive and use only train and "
            "validation arrays. Held-out test batch bytes are authenticated outside this function but "
            "test target arrays are not deserialized. Diagnostics cannot change the frozen primary "
            "metric, model-selection rule, split assignment, or test policy."
        ),
    }
    report["report_sha256"] = _sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit an exact MC2P v1 train+validation projection without test deserialization"
    )
    parser.add_argument("config")
    parser.add_argument("split_lock")
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    batch, source_batches = load_session_batches(args.batches)
    lock = json.loads(Path(args.split_lock).read_text())
    config = load_qc_config(args.config)
    report = audit_development_data(batch, lock, config, source_batches=source_batches)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "sha256": report["report_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
