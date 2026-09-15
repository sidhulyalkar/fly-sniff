from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .alignment_null import run_alignment_null
from .data_qc import audit_development_data, load_qc_config
from .mc2p_legacy import sha256_file
from .provenance import implementation_fingerprint, runtime_fingerprint
from .session_benchmark import load_session_batches, run_within_animal_ridge
from .validation_gate import build_validation_unlock, load_acceptance_config


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fresh_output_dir(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty development directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_development_protocol(
    split_lock_path: str | Path,
    batch_paths: list[str | Path],
    output_dir: str | Path,
    *,
    qc_config_path: str | Path,
    acceptance_config_path: str | Path,
) -> dict[str, Any]:
    output = _fresh_output_dir(output_dir)
    batch, source_batches = load_session_batches(batch_paths)
    split_lock = json.loads(Path(split_lock_path).read_text())
    qc_config = load_qc_config(qc_config_path)
    acceptance_config = load_acceptance_config(acceptance_config_path)

    qc = audit_development_data(
        batch,
        split_lock,
        qc_config,
        source_batches=source_batches,
    )
    aligned = run_within_animal_ridge(
        batch,
        split_lock,
        source_batches=source_batches,
        consume_test=False,
    )
    null = run_alignment_null(
        batch,
        split_lock,
        source_batches=source_batches,
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
        "split_lock_sha256": split_lock["split_lock_sha256"],
        "split_lock_file_sha256": sha256_file(split_lock_path),
        "source_batches": source_batches,
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
            "This command is development-only. It cannot consume held-out test sessions. "
            "An unlock status authorizes a separate one-way final action but is not a test result."
        ),
    }
    receipt["receipt_sha256"] = _sha(receipt)
    (output / "development-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MC2P v1 development QC and validation without test consumption"
    )
    parser.add_argument("split_lock")
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--qc-config", required=True)
    parser.add_argument("--acceptance-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_development_protocol(
        args.split_lock,
        args.batches,
        args.output,
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
