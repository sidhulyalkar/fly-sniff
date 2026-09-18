from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .olfactory_door import sha256_file
from .olfactory_e006_audit import _canonical_sha

EXPECTED_STUDY = "Stensmyr.2012.WT"
EXPECTED_UNIT = "ab4B"
EXPECTED_ODOR = "geosmin"
EXPECTED_RAW_RESPONSE = 146.4
EXPECTED_DOOR_COMMIT = "db323a496577c4b4a72b5c2fcd1859e07521ffb5"


def _load_e006_receipt(artifact: Path) -> dict[str, Any]:
    receipt_path = artifact / "door-e006-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing E006 receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    source = receipt.get("source", {})
    observed_commit = (
        source.get("commit")
        or source.get("commit_sha")
        or source.get("git_commit")
        or receipt.get("source_commit")
    )
    if observed_commit != EXPECTED_DOOR_COMMIT:
        raise ValueError(
            "E001 cross-check requires the exact frozen DoOR source commit "
            f"{EXPECTED_DOOR_COMMIT}; got {observed_commit!r}"
        )
    return receipt


def crosscheck_e001_door(
    artifact_dir: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    artifact = Path(artifact_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty E001 DoOR cross-check: {output}")
    output.mkdir(parents=True, exist_ok=True)

    receipt = _load_e006_receipt(artifact)
    long_path = artifact / "door-responses-long.csv"
    if not long_path.is_file():
        raise FileNotFoundError(f"missing E006 long-form artifact: {long_path}")

    expected_long_sha = (
        receipt.get("long_form_artifact", {}).get("sha256")
        or receipt.get("outputs", {}).get("long_form", {}).get("sha256")
        or receipt.get("long_form_sha256")
    )
    if expected_long_sha and sha256_file(long_path) != expected_long_sha:
        raise ValueError("E001 cross-check E006 long-form sha256 mismatch")

    frame = pd.read_csv(long_path)
    required = {
        "responding_unit",
        "odor_name",
        "study_id",
        "response_status",
        "response_value",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"E006 long-form lacks E001 cross-check columns: {missing}")

    study = frame[frame["study_id"].eq(EXPECTED_STUDY)].copy()
    if study.empty:
        raise ValueError(f"frozen E006 lacks study {EXPECTED_STUDY}")

    observed = study[
        study["response_status"].eq("observed")
        & study["response_value"].notna()
        & study["odor_name"].notna()
    ].copy()
    observed["response_value"] = pd.to_numeric(observed["response_value"], errors="raise")

    geosmin = observed[
        observed["odor_name"].astype(str).str.casefold().eq(EXPECTED_ODOR)
    ].copy()
    target = geosmin[geosmin["responding_unit"].eq(EXPECTED_UNIT)]
    if len(target) != 1:
        raise ValueError(
            "expected exactly one observed Stensmyr.2012.WT ab4B geosmin cell; "
            f"got {len(target)}"
        )
    response = float(target.iloc[0]["response_value"])
    if response != EXPECTED_RAW_RESPONSE:
        raise ValueError(
            "Stensmyr.2012.WT ab4B geosmin raw response changed: "
            f"{response} != {EXPECTED_RAW_RESPONSE}"
        )

    observations = [
        {
            "responding_unit": str(row["responding_unit"]),
            "odor_name": str(row["odor_name"]),
            "raw_response": float(row["response_value"]),
        }
        for _, row in geosmin.sort_values(["responding_unit", "odor_name"]).iterrows()
    ]

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "e001-door-crosscheck-v1",
        "authority_id": "E001_geosmin_receptor_physiology",
        "program_id": "olfactory-computation-v0",
        "status": "source_transcription_crosscheck_complete_not_numeric_calibration",
        "input": {
            "door_commit": EXPECTED_DOOR_COMMIT,
            "e006_receipt_sha256": sha256_file(artifact / "door-e006-receipt.json"),
            "e006_long_form_sha256": sha256_file(long_path),
        },
        "study": {
            "study_id": EXPECTED_STUDY,
            "observed_cells": int(len(observed)),
            "observed_responding_units": int(observed["responding_unit"].nunique()),
            "observed_odor_names": int(observed["odor_name"].nunique()),
        },
        "geosmin": {
            "observed_cells": int(len(geosmin)),
            "observations": observations,
            "ab4B_raw_response": response,
            "ab4B_exact_source_transcription_match": True,
        },
        "interpretation": {
            "qualitative_selectivity_crosscheck_supported": True,
            "numeric_model_parameter_allowed": False,
            "cross_study_scale_comparability_established": False,
            "raw_trial_data_established": False,
        },
        "claim_boundary": (
            "This receipt verifies that the frozen E006 transcription retains the exact pinned DoOR "
            "Stensmyr.2012.WT ab4B geosmin source cell. The raw value 146.4 is used only "
            "as a source-integrity sentinel. This does not establish a receptor-site concentration, "
            "cross-study numerical scale, raw-trial provenance, or a fitted O001 parameter."
        ),
    }
    report["receipt_sha256"] = _canonical_sha(report)
    receipt_path = output / "e001-door-crosscheck-receipt.json"
    receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    (output / "SUMMARY.txt").write_text(
        "\n".join(
            [
                "E001 DOOR CROSS-CHECK V1",
                f"status: {report['status']}",
                f"study: {EXPECTED_STUDY}",
                "source_commit: " + EXPECTED_DOOR_COMMIT,
                f"observed_study_cells: {len(observed)}",
                f"observed_geosmin_cells: {len(geosmin)}",
                f"ab4B_geosmin_raw_response: {response}",
                "numeric_model_parameter_allowed: false",
                f"receipt_sha256: {report['receipt_sha256']}",
                "",
                "CLAIM BOUNDARY",
                report["claim_boundary"],
            ]
        )
        + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-check frozen E001 Stensmyr.2012.WT source cells against E006"
    )
    parser.add_argument("artifact_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = crosscheck_e001_door(args.artifact_dir, output_dir=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
