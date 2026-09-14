from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_io import file_sha256, load_json, load_typed_artifact
from .evidence import EvidenceLedger
from .experiment import ExperimentLock, ExperimentSpec, RunReceipt
from .final_entropy import FinalEntropyCommitment, FinalSeedReceipt


def _summary(path: Path) -> dict[str, object]:
    payload = load_json(path)
    schema = payload.get("schema")
    if schema == "fly-sniff-final-entropy-commitment-v1":
        artifact = FinalEntropyCommitment.from_dict(payload)
        scientific_hash = artifact.sha256
    elif schema == "fly-sniff-final-seed-receipt-v1":
        artifact = FinalSeedReceipt.from_dict(payload)
        scientific_hash = artifact.sha256
    else:
        artifact = load_typed_artifact(path)
        scientific_hash = artifact.sha256

    if isinstance(artifact, EvidenceLedger):
        kind = "evidence_ledger"
    elif isinstance(artifact, ExperimentSpec):
        kind = "experiment_spec"
    elif isinstance(artifact, ExperimentLock):
        kind = "experiment_lock"
    elif isinstance(artifact, RunReceipt):
        kind = "run_receipt"
    elif isinstance(artifact, FinalEntropyCommitment):
        kind = "final_entropy_commitment"
    elif isinstance(artifact, FinalSeedReceipt):
        kind = "final_seed_receipt"
    else:
        raise TypeError(f"unsupported artifact type: {type(artifact)!r}")

    return {
        "status": "valid",
        "kind": kind,
        "schema": schema,
        "scientific_sha256": scientific_hash,
        "file_sha256": file_sha256(path),
        "path": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a fly-sniff evidence/experiment/provenance artifact."
    )
    parser.add_argument("artifact", help="Path to a JSON artifact")
    args = parser.parse_args(argv)
    print(json.dumps(_summary(Path(args.artifact)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
