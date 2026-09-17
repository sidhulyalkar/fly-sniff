from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "db323a496577c4b4a72b5c2fcd1859e07521ffb5"
EXPECTED_TREE = "2b673e851cd0760715c1b74d6b3a6035c1af72cd"
EXPECTED_UNITS = 78
INDEX_GIT_BLOBS = {
    "data/ORs.csv": "b88dc16a9cb417d3fb8587a9434359eee1ecf55c",
    "data/door_mappings.csv": "2f5d9c6fcac5b450b152401072e6f86a440dfb3a",
}
ODOR_METADATA = ("Class", "Name", "InChIKey", "CID", "CAS")
MISSING_TOKENS = {"", "NA"}
MAPPING_KEYS = ("receptor", "OSN", "glomerulus", "code", "code.OSN")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def validate_source_authority(authority: dict[str, Any]) -> None:
    if authority.get("schema_version") != 1:
        raise ValueError("E006 source authority requires schema_version=1")
    if authority.get("authority_id") != "E006_odor_panel_receptor_responses":
        raise ValueError("E006 source authority id changed")
    if authority.get("source_id") != "S002_door_data":
        raise ValueError("E006 source id changed")
    if authority.get("repository") != "ropensci/DoOR.data":
        raise ValueError("E006 source repository changed")
    if authority.get("commit") != EXPECTED_COMMIT or authority.get("tree") != EXPECTED_TREE:
        raise ValueError("E006 source commit/tree does not match the frozen ingester")
    if authority.get("checkout_policy") != "clean_git_checkout_at_exact_commit":
        raise ValueError("E006 clean-checkout policy changed")
    if authority.get("required_index_git_blobs") != INDEX_GIT_BLOBS:
        raise ValueError("E006 frozen index Git-blob identities changed")
    if authority.get("expected_responding_units") != EXPECTED_UNITS:
        raise ValueError("E006 responding-unit count changed")
    if authority.get("normalization_policy") != "none_during_ingestion":
        raise ValueError("E006 ingestion may not normalize responses")
    if authority.get("aggregation_policy") != "none_during_ingestion":
        raise ValueError("E006 ingestion may not aggregate study responses")
    if authority.get("status") != "source_frozen_ingestion_not_yet_executed":
        raise ValueError("E006 source authority status changed without a new version")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_checkout(root: str | Path) -> dict[str, str]:
    source = Path(root).resolve()
    commit = _git(source, "rev-parse", "HEAD^{commit}")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    status = _git(source, "status", "--porcelain", "--untracked-files=all")
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"DoOR checkout commit changed: expected {EXPECTED_COMMIT}, got {commit}")
    if tree != EXPECTED_TREE:
        raise ValueError(f"DoOR checkout tree changed: expected {EXPECTED_TREE}, got {tree}")
    if status:
        raise ValueError("DoOR checkout must be clean; source files may not be locally modified")
    for relative, expected_blob in INDEX_GIT_BLOBS.items():
        path = source / relative
        if not path.is_file():
            raise ValueError(f"missing frozen DoOR index file: {relative}")
        observed_blob = _git(source, "hash-object", relative)
        if observed_blob != expected_blob:
            raise ValueError(f"DoOR index Git blob does not match frozen source: {relative}")
    return {"commit": commit, "tree": tree}


def read_r_csv2(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";", quotechar='"')
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty DoOR CSV: {source}") from exc
        if not header or any(not name for name in header):
            raise ValueError(f"DoOR CSV has malformed header: {source}")
        full_header = [".row_id", *header]
        rows: list[dict[str, str]] = []
        for line_number, values in enumerate(reader, start=2):
            if not values:
                continue
            if len(values) != len(full_header):
                raise ValueError(
                    f"DoOR R row-name offset mismatch in {source.name}:{line_number}: "
                    f"expected {len(full_header)} fields, got {len(values)}"
                )
            rows.append(dict(zip(full_header, values, strict=True)))
    if not rows:
        raise ValueError(f"DoOR CSV contains no rows: {source}")
    return full_header, rows


def load_responding_units(root: str | Path) -> list[str]:
    _, rows = read_r_csv2(Path(root) / "data" / "ORs.csv")
    units = [row["OR"] for row in rows]
    if len(units) != EXPECTED_UNITS or len(set(units)) != EXPECTED_UNITS:
        raise ValueError(f"expected exactly {EXPECTED_UNITS} unique DoOR responding units")
    if any(not unit for unit in units):
        raise ValueError("DoOR responding-unit index contains an empty unit")
    return units


def load_mapping_rows(root: str | Path) -> list[dict[str, str]]:
    _, rows = read_r_csv2(Path(root) / "data" / "door_mappings.csv")
    required = {"receptor", "sensillum", "OSN", "glomerulus", "code", "code.OSN"}
    if not required <= set(rows[0]):
        raise ValueError("DoOR mapping table is missing required identity fields")
    return rows


def mapping_candidates(unit: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in rows:
        if any(row.get(key) == unit for key in MAPPING_KEYS):
            candidates.append(dict(row))
    return candidates


def parse_response_unit(
    path: str | Path, responding_unit: str
) -> tuple[list[dict[str, Any]], list[str]]:
    header, rows = read_r_csv2(path)
    if not set(ODOR_METADATA) <= set(header):
        raise ValueError(f"{responding_unit}: response table lacks frozen odor metadata columns")
    study_columns = [name for name in header if name not in {".row_id", *ODOR_METADATA}]
    if not study_columns:
        raise ValueError(f"{responding_unit}: response table has no study-specific columns")
    observations: list[dict[str, Any]] = []
    for row in rows:
        odor_metadata = {
            "source_row_id": row[".row_id"],
            "odor_class": None if row["Class"] in MISSING_TOKENS else row["Class"],
            "odor_name": None if row["Name"] in MISSING_TOKENS else row["Name"],
            "inchikey": None if row["InChIKey"] in MISSING_TOKENS else row["InChIKey"],
            "cid": None if row["CID"] in MISSING_TOKENS else row["CID"],
            "cas": None if row["CAS"] in MISSING_TOKENS else row["CAS"],
        }
        for study in study_columns:
            raw = row[study]
            missing = raw in MISSING_TOKENS
            response: float | None
            if missing:
                response = None
            else:
                try:
                    response = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{responding_unit}: non-numeric response {raw!r} in {study} "
                        f"row {row['.row_id']}"
                    ) from exc
            observations.append(
                {
                    "responding_unit": responding_unit,
                    **odor_metadata,
                    "study_id": study,
                    "response_status": "missing" if missing else "observed",
                    "raw_response": None if missing else raw,
                    "response_value": response,
                }
            )
    return observations, study_columns


def _write_long_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "responding_unit",
        "source_row_id",
        "odor_class",
        "odor_name",
        "inchikey",
        "cid",
        "cas",
        "study_id",
        "response_status",
        "raw_response",
        "response_value",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def ingest_door(
    root: str | Path,
    output_directory: str | Path,
    *,
    authority_path: str | Path,
) -> dict[str, Any]:
    authority_file = Path(authority_path).resolve()
    authority = _load_json(authority_file)
    validate_source_authority(authority)

    source = Path(root).resolve()
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty E006 output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    git_identity = verify_checkout(source)
    units = load_responding_units(source)
    mappings = load_mapping_rows(source)

    all_rows: list[dict[str, Any]] = []
    studies: set[str] = set()
    source_files: list[dict[str, Any]] = []
    unit_mapping_payload: dict[str, list[dict[str, str]]] = {}
    for unit in units:
        response_path = source / "data" / f"{unit}.csv"
        if not response_path.is_file():
            raise ValueError(f"missing frozen DoOR responding-unit file: {response_path.name}")
        parsed, unit_studies = parse_response_unit(response_path, unit)
        all_rows.extend(parsed)
        studies.update(unit_studies)
        source_files.append(
            {
                "responding_unit": unit,
                "path": f"data/{unit}.csv",
                "sha256": sha256_file(response_path),
                "observation_cells": len(parsed),
            }
        )
        unit_mapping_payload[unit] = mapping_candidates(unit, mappings)

    long_path = output / "door-responses-long.csv"
    _write_long_csv(all_rows, long_path)
    mapping_path = output / "door-unit-mappings.json"
    mapping_path.write_text(json.dumps(unit_mapping_payload, indent=2, sort_keys=True) + "\n")

    observed = sum(row["response_status"] == "observed" for row in all_rows)
    missing = len(all_rows) - observed
    geosmin_rows = [
        row
        for row in all_rows
        if row["odor_name"] == "geosmin" and row["response_status"] == "observed"
    ]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "door-e006-source-resolved-ingestion-v0",
        "authority_id": "E006_odor_panel_receptor_responses",
        "source_authority_path": str(authority_file),
        "source_authority_sha256": sha256_file(authority_file),
        "source_repository": "ropensci/DoOR.data",
        "source_commit": git_identity["commit"],
        "source_tree": git_identity["tree"],
        "responding_unit_count": len(units),
        "study_column_count": len(studies),
        "response_cells": len(all_rows),
        "observed_response_cells": observed,
        "missing_response_cells": missing,
        "geosmin_observed_cells": len(geosmin_rows),
        "source_files": sorted(source_files, key=lambda row: row["responding_unit"]),
        "index_files": [
            {
                "path": relative,
                "git_blob": INDEX_GIT_BLOBS[relative],
                "sha256": sha256_file(source / relative),
            }
            for relative in sorted(INDEX_GIT_BLOBS)
        ],
        "long_form_artifact": {
            "path": str(long_path),
            "sha256": sha256_file(long_path),
        },
        "mapping_artifact": {
            "path": str(mapping_path),
            "sha256": sha256_file(mapping_path),
        },
        "normalization_applied": False,
        "aggregation_applied": False,
        "missingness_preserved": True,
        "study_identity_preserved": True,
        "claim_boundary": (
            "This artifact is a source-resolved transcription of DoOR response tables. It does not make "
            "responses from different studies or assay scales directly comparable, does not establish dose "
            "response without concentration metadata, and does not qualify a biological mechanism."
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path = output / "door-e006-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the frozen DoOR source into E006 long-form evidence"
    )
    parser.add_argument("door_checkout")
    parser.add_argument("--authority", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = ingest_door(args.door_checkout, args.output, authority_path=args.authority)
    print(
        json.dumps(
            {
                "status": "ingested_not_qualified",
                "authority_id": report["authority_id"],
                "responding_units": report["responding_unit_count"],
                "observed_cells": report["observed_response_cells"],
                "sha256": report["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
