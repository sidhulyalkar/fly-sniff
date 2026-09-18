from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_LINEAGE = {
    "v1": {
        "code_head": "8726aaef51900d003f4b41d400f05ff6508c18e7",
        "protocol": "o002-within-study-development-v1",
        "receipt_sha256": "fd7fd52338f6f5bab9b132208d56bbeb34c93e7b2bf6aee0ad494c33a06e43e1",
    },
    "v2": {
        "code_head": "0fad162ca1c3e1b9ac8a432dc53bb8675e6d9f79",
        "protocol": "o002-coding-robustness-development-v2",
        "receipt_sha256": "89f738e5a6b2426f303b5d45bd3015e0d7ac6870d1b6907e5336013410f4c2c4",
    },
    "v3": {
        "code_head": "7fe9f622d90967bd1dd9be8c37dc09d48469816c",
        "protocol": "o002-subspace-stability-development-v3",
        "receipt_sha256": "437d9880f3a8a9b17a56e31ab3a3ee8d5f9f43a57f509805fddb2dda2afa2dd7",
    },
}


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError("O002 freeze must be a JSON object")
    return payload


def validate_o002_freeze(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("O002 freeze requires schema_version=1")
    if payload.get("freeze_id") != "O002-development-freeze-v1":
        raise ValueError("O002 freeze id changed")
    if payload.get("program_id") != "olfactory-computation-v0":
        raise ValueError("O002 freeze program id changed")
    if payload.get("status") != "frozen_development_finding_not_confirmatory":
        raise ValueError("O002 freeze status changed")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise TypeError("O002 freeze requires source")
    if source.get("door_commit") != "db323a496577c4b4a72b5c2fcd1859e07521ffb5":
        raise ValueError("O002 frozen DoOR source changed")
    if source.get("study_id") != "Hallem.2006.EN":
        raise ValueError("O002 frozen study changed")
    if source.get("feature_identity") != "source_responding_unit":
        raise ValueError("O002 frozen feature identity changed")

    lineage = payload.get("lineage")
    if lineage != EXPECTED_LINEAGE:
        raise ValueError("O002 frozen lineage changed")

    findings = payload.get("frozen_findings")
    if not isinstance(findings, dict):
        raise TypeError("O002 freeze requires frozen_findings")
    matrix = findings.get("matrix")
    if not isinstance(matrix, dict):
        raise TypeError("O002 freeze requires matrix finding")
    if matrix.get("complete_odors_excluding_sfr") != 110:
        raise ValueError("O002 complete-odor count changed")
    if matrix.get("responding_units") != 24:
        raise ValueError("O002 responding-unit count changed")
    if matrix.get("imputation_used") is not False:
        raise ValueError("O002 freeze cannot introduce imputation")

    coding = findings.get("coding_decomposition")
    if not isinstance(coding, dict):
        raise TypeError("O002 freeze requires coding_decomposition")
    if coding.get("direction_only_balanced_accuracy") != 0.610917785917786:
        raise ValueError("O002 frozen direction-only result changed")
    if coding.get("channel_identity_shuffle_null_q95") != 0.28924374236874234:
        raise ValueError("O002 frozen channel-shuffle q95 changed")

    stability = findings.get("stability")
    if not isinstance(stability, dict):
        raise TypeError("O002 freeze requires stability")
    if stability.get("direction_subspace_8d_balanced_accuracy") != 0.6035917785917787:
        raise ValueError("O002 frozen 8D direction result changed")
    if stability.get("direction_subspace_24d_balanced_accuracy") != 0.610917785917786:
        raise ValueError("O002 frozen 24D direction result changed")

    continuation = payload.get("continuation_policy")
    if not isinstance(continuation, dict):
        raise TypeError("O002 freeze requires continuation_policy")
    if continuation.get("further_model_or_threshold_search_on_same_O002_table_allowed") is not False:
        raise ValueError("O002 freeze cannot reopen model/threshold search")
    if continuation.get("negative_or_heterogeneous_class_results_retained") is not True:
        raise ValueError("O002 freeze must retain heterogeneous class results")

    forbidden = set(payload.get("forbidden_promotions", []))
    required_forbidden = {
        "odor identity decoding",
        "behavioral valence decoding",
        "receptor-specific causal mechanism",
        "cross-study generalization",
        "connectome topology effect",
        "O003 or O004 authorization",
    }
    if not required_forbidden <= forbidden:
        raise ValueError("O002 freeze lost a claim boundary")

    return {
        "status": payload["status"],
        "study_id": source["study_id"],
        "responding_units": matrix["responding_units"],
        "complete_odors": matrix["complete_odors_excluding_sfr"],
        "v1_receipt_sha256": lineage["v1"]["receipt_sha256"],
        "v2_receipt_sha256": lineage["v2"]["receipt_sha256"],
        "v3_receipt_sha256": lineage["v3"]["receipt_sha256"],
        "same_table_model_search_closed": True,
        "confirmatory_usable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen O002 development finding")
    parser.add_argument("freeze")
    args = parser.parse_args()
    print(json.dumps(validate_o002_freeze(_load(args.freeze)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
