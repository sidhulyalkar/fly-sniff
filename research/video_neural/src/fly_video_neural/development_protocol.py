from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .alignment import TARGET_NEURAL_BOUNDARY_POLICY
from .alignment_null import run_alignment_null
from .data_qc import audit_development_data, load_qc_config
from .mc2p_legacy import sha256_file
from .provenance import (
    implementation_fingerprint,
    preparation_implementation_fingerprint,
    runtime_fingerprint,
)
from .session_benchmark import load_session_batches, run_within_animal_ridge
from .validation_gate import build_validation_unlock, load_acceptance_config

EXPECTED_PUBLIC_RELEASE_ANIMALS = 8


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fresh_output_dir(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty development directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _source_batch_receipts(paths: list[str | Path]) -> list[dict[str, str]]:
    return sorted(
        [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for path in paths
        ],
        key=lambda row: row["path"],
    )


def _session_split(lock: dict[str, Any], animal_id: str, session_id: str) -> str:
    assignments = lock["animal_sessions"].get(animal_id)
    if assignments is None:
        raise ValueError(f"prepared session {session_id!r} has unknown animal {animal_id!r}")
    matches = [name for name in ("train", "validation", "test") if session_id in assignments[name]]
    if len(matches) != 1:
        raise ValueError(f"prepared session {session_id!r} does not have one frozen split assignment")
    return matches[0]


def _partition_authenticated_batch_paths(
    receipt: dict[str, Any],
    split_lock: dict[str, Any],
    batch_paths: list[str | Path],
) -> tuple[list[dict[str, str]], list[str | Path], list[str | Path]]:
    source_batches = _source_batch_receipts(batch_paths)
    if receipt.get("source_batches") != source_batches:
        raise ValueError("session-batch bytes do not match the preparation receipt")

    sessions = receipt.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != receipt.get("session_count"):
        raise ValueError("preparation receipt is missing complete per-session batch provenance")
    by_sha: dict[str, dict[str, Any]] = {}
    for row in sessions:
        batch_sha = row.get("batch_sha256")
        session_id = row.get("session_id")
        animal_id = row.get("animal_id")
        if not all(isinstance(value, str) and value for value in (batch_sha, session_id, animal_id)):
            raise ValueError("preparation session provenance is incomplete")
        if batch_sha in by_sha:
            raise ValueError("preparation receipt has ambiguous duplicate session-batch hashes")
        by_sha[batch_sha] = row
        _session_split(split_lock, animal_id, session_id)

    path_by_resolved = {str(Path(path).resolve()): path for path in batch_paths}
    development_paths: list[str | Path] = []
    test_paths: list[str | Path] = []
    matched_sessions: set[str] = set()
    for source in source_batches:
        row = by_sha.get(source["sha256"])
        if row is None:
            raise ValueError("authenticated session batch is not represented in preparation sessions")
        session_id = row["session_id"]
        if session_id in matched_sessions:
            raise ValueError(f"prepared session {session_id!r} was matched more than once")
        matched_sessions.add(session_id)
        split_name = _session_split(split_lock, row["animal_id"], session_id)
        path = path_by_resolved[source["path"]]
        if split_name == "test":
            test_paths.append(path)
        else:
            development_paths.append(path)

    if len(matched_sessions) != len(sessions):
        raise ValueError("not every prepared session batch was supplied to development")
    expected_test_sessions = sum(
        len(assignments["test"]) for assignments in split_lock["animal_sessions"].values()
    )
    if len(test_paths) != expected_test_sessions:
        raise ValueError("authenticated held-out test batch count does not match the frozen split")
    if not development_paths:
        raise ValueError("no train/validation session batches remain after frozen split partition")
    return source_batches, development_paths, test_paths


def _load_and_validate_preparation_receipt(
    preparation_receipt_path: str | Path,
    split_lock_path: str | Path,
    batch_paths: list[str | Path],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    list[str | Path],
    list[str | Path],
]:
    receipt = json.loads(Path(preparation_receipt_path).read_text())
    claimed = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or claimed != _sha(unsigned):
        raise ValueError("preparation receipt self-hash mismatch")
    if receipt.get("protocol") != "mc2p-v1-preparation-v1":
        raise ValueError("development requires the frozen MC2P v1 preparation receipt")
    if receipt.get("benchmark_id") != "mc2p_future_neural_v1":
        raise ValueError("preparation receipt benchmark mismatch")
    if receipt.get("animal_count") != EXPECTED_PUBLIC_RELEASE_ANIMALS:
        raise ValueError("development requires the complete eight-animal prepared release")
    if receipt.get("models_fit") is not False or receipt.get("test_data_consumed") is not False:
        raise ValueError("preparation receipt indicates model fitting or test consumption")
    if receipt.get("target_neural_boundary_policy") != TARGET_NEURAL_BOUNDARY_POLICY:
        raise ValueError("preparation receipt does not enforce the strict future neural boundary")
    if receipt.get("preparation_implementation_fingerprint") != preparation_implementation_fingerprint():
        raise ValueError("preparation implementation bytes do not match the current audited ingestion code")
    if receipt.get("split_lock_file_sha256") != sha256_file(split_lock_path):
        raise ValueError("split-lock bytes do not match the preparation receipt")
    split_lock = json.loads(Path(split_lock_path).read_text())
    if receipt.get("split_lock_sha256") != split_lock.get("split_lock_sha256"):
        raise ValueError("split-lock identity does not match the preparation receipt")
    source_batches, development_paths, test_paths = _partition_authenticated_batch_paths(
        receipt,
        split_lock,
        batch_paths,
    )
    return receipt, split_lock, source_batches, development_paths, test_paths


def run_development_protocol(
    split_lock_path: str | Path,
    batch_paths: list[str | Path],
    output_dir: str | Path,
    *,
    preparation_receipt_path: str | Path,
    qc_config_path: str | Path,
    acceptance_config_path: str | Path,
) -> dict[str, Any]:
    output = _fresh_output_dir(output_dir)
    (
        preparation_receipt,
        split_lock,
        all_source_batches,
        development_paths,
        test_paths,
    ) = _load_and_validate_preparation_receipt(
        preparation_receipt_path,
        split_lock_path,
        batch_paths,
    )
    batch, development_source_batches = load_session_batches(development_paths)
    expected_development_sources = _source_batch_receipts(development_paths)
    if development_source_batches != expected_development_sources:
        raise RuntimeError("development batch identity changed after preparation verification")
    authenticated_test_batches = _source_batch_receipts(test_paths)
    qc_config = load_qc_config(qc_config_path)
    acceptance_config = load_acceptance_config(acceptance_config_path)

    qc = audit_development_data(
        batch,
        split_lock,
        qc_config,
        source_batches=development_source_batches,
    )
    aligned = run_within_animal_ridge(
        batch,
        split_lock,
        source_batches=development_source_batches,
        consume_test=False,
    )
    null = run_alignment_null(
        batch,
        split_lock,
        source_batches=development_source_batches,
        consume_test=False,
    )
    unlock = build_validation_unlock(qc, aligned, null, acceptance_config)

    reports = {
        "development-qc.json": qc,
        "aligned-ridge-development.json": aligned,
        "temporal-null-development.json": null,
        "validation-unlock.json": unlock,
    }
    for filename, report in reports.items():
        (output / filename).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if aligned["test_status"] != "locked_not_consumed":
        raise RuntimeError("development protocol unexpectedly consumed aligned test data")
    if null["test_status"] != "locked_not_consumed":
        raise RuntimeError("development protocol unexpectedly consumed null test data")
    if any(row["test_metrics"] is not None for row in aligned["animals"]):
        raise RuntimeError("development aligned report contains test metrics")
    if any(row["test_metrics"] is not None for row in null["animals"]):
        raise RuntimeError("development null report contains primary test metrics")
    if any(
        candidate["test_metrics"] is not None
        for row in null["animals"]
        for candidate in row["null_candidates"]
    ):
        raise RuntimeError("development null ensemble contains test metrics")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "mc2p-v1-development-only-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "preparation_receipt_sha256": preparation_receipt["receipt_sha256"],
        "preparation_receipt_file_sha256": sha256_file(preparation_receipt_path),
        "preparation_implementation_fingerprint": preparation_receipt[
            "preparation_implementation_fingerprint"
        ],
        "target_neural_boundary_policy": preparation_receipt["target_neural_boundary_policy"],
        "split_lock_sha256": split_lock["split_lock_sha256"],
        "split_lock_file_sha256": sha256_file(split_lock_path),
        "source_batches": all_source_batches,
        "development_deserialized_batches": development_source_batches,
        "held_out_test_batches_authenticated_not_deserialized": authenticated_test_batches,
        "test_target_arrays_deserialized": False,
        "qc_config_file_sha256": sha256_file(qc_config_path),
        "acceptance_config_file_sha256": sha256_file(acceptance_config_path),
        "implementation_fingerprint": implementation_fingerprint(),
        "runtime_fingerprint": runtime_fingerprint(),
        "qc_report_sha256": qc["report_sha256"],
        "validation_unlock_report_sha256": unlock["report_sha256"],
        "validation_status": unlock["status"],
        "test_consumption_capability": False,
        "test_metrics_present": False,
        "primary_metric_scope": "all_measured_dff_pixels_with_finite_correlation",
        "primary_metric_limitation": (
            "The v1 primary median Pearson metric spans the full measured dF/F image rather than a "
            "neural-support mask. Background or low-information pixels may therefore reduce sensitivity. "
            "This limitation is frozen for v1 and may motivate a separately versioned v2 metric."
        ),
        "claim_boundary": (
            "This command is development-only and requires the audited PREPARE receipt and exact derived "
            "batch bytes. Held-out test batches are authenticated by byte hash but their arrays are not "
            "deserialized. An unlock status authorizes a separate one-way final action but is not a test result."
        ),
    }
    receipt["receipt_sha256"] = _sha(receipt)
    (output / "development-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MC2P v1 development QC and validation without test-target deserialization"
    )
    parser.add_argument("split_lock")
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--preparation-receipt", required=True)
    parser.add_argument("--qc-config", required=True)
    parser.add_argument("--acceptance-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_development_protocol(
        args.split_lock,
        args.batches,
        args.output,
        preparation_receipt_path=args.preparation_receipt,
        qc_config_path=args.qc_config,
        acceptance_config_path=args.acceptance_config,
    )
    print(
        json.dumps(
            {
                "status": report["validation_status"],
                "test_consumption_capability": False,
                "sha256": report["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
