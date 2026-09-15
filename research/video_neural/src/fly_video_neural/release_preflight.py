from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .preparation_audit import audit_preparation_directory
from .prepare_v1 import EXPECTED_PUBLIC_RELEASE_ANIMALS, EXPECTED_PUBLIC_RELEASE_SESSIONS


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_self_hashed_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing preparation receipt: {path}")
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise TypeError("preparation receipt must be a JSON object")
    unsigned = dict(document)
    claimed = unsigned.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or claimed != _sha(unsigned):
        raise ValueError("preparation receipt self-hash mismatch")
    return document


def audit_complete_public_release(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve()
    receipt = _load_self_hashed_receipt(root / "preparation-receipt.json")

    if receipt.get("expected_public_release_animals") != EXPECTED_PUBLIC_RELEASE_ANIMALS:
        raise ValueError("preparation receipt animal-count authority changed")
    if receipt.get("expected_public_release_sessions") != EXPECTED_PUBLIC_RELEASE_SESSIONS:
        raise ValueError("preparation receipt session-count authority changed")
    if receipt.get("animal_count") != EXPECTED_PUBLIC_RELEASE_ANIMALS:
        raise ValueError("preflight requires the complete eight-animal public MC2P release")
    if receipt.get("session_count") != EXPECTED_PUBLIC_RELEASE_SESSIONS:
        raise ValueError("preflight requires all 133 trials in the public MC2P release")

    provenance = audit_preparation_directory(root)
    if provenance.get("animal_count") != EXPECTED_PUBLIC_RELEASE_ANIMALS:
        raise RuntimeError("provenance audit returned the wrong public-release animal count")
    if provenance.get("session_count") != EXPECTED_PUBLIC_RELEASE_SESSIONS:
        raise RuntimeError("provenance audit returned the wrong public-release session count")
    if provenance.get("verified_session_batches") != EXPECTED_PUBLIC_RELEASE_SESSIONS:
        raise RuntimeError("provenance audit did not verify all 133 public-release session batches")
    if provenance.get("batch_arrays_deserialized") is not False:
        raise RuntimeError("public-release preflight unexpectedly deserialized batch arrays")
    if provenance.get("model_metrics_inspected") is not False:
        raise RuntimeError("public-release preflight unexpectedly inspected model metrics")
    if provenance.get("models_fit") is not False or provenance.get("test_data_consumed") is not False:
        raise RuntimeError("public-release preflight crossed the no-model/no-test boundary")

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "mc2p-v1-complete-release-preflight-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "status": "pass",
        "expected_public_release_animals": EXPECTED_PUBLIC_RELEASE_ANIMALS,
        "expected_public_release_sessions": EXPECTED_PUBLIC_RELEASE_SESSIONS,
        "animal_count": provenance["animal_count"],
        "session_count": provenance["session_count"],
        "verified_session_batches": provenance["verified_session_batches"],
        "verified_prediction_windows": provenance["verified_prediction_windows"],
        "preparation_receipt_sha256": receipt["receipt_sha256"],
        "provenance_audit_report_sha256": provenance["report_sha256"],
        "batch_arrays_deserialized": False,
        "model_metrics_inspected": False,
        "models_fit": False,
        "test_data_consumed": False,
        "claim_boundary": (
            "This envelope first requires the authoritative eight-animal, 133-trial public MC2P release, "
            "then delegates byte, split, window, and provenance verification to the preparation auditor. "
            "It does not deserialize pose-neural NPZ batches, inspect model metrics, fit a decoder, or "
            "consume a held-out test result."
        ),
    }
    report["report_sha256"] = _sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the complete eight-animal, 133-trial MC2P v1 preparation before development"
    )
    parser.add_argument("prepared_directory")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_complete_public_release(args.prepared_directory)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
