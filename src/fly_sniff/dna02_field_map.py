from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .dna02_source import load_contract
from .dna02_source_inspect import DEFAULT_CONTRACT, DEFAULT_EVIDENCE, _load_byte_evidence
from .freeze import canonical_sha256

DEFAULT_AUTHORITY = Path("authority/program-a-dna02-field-map-authority-v1.json")

_EXPECTED_REPOSITORY = "wilson-lab/rayshubskiy_elife_102230_secondary_analysis_code"
_EXPECTED_COMMIT = "7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8"
_EXPECTED_NOTEBOOK_BLOB = "0da2089b468c172f700881b714bfdda99a6fe424"
_EXPECTED_LOADER_BLOB = "160147b83d5a5d5238b0cb558ce00319a9711d76"
_EXPECTED_ALIASES = ("a2_d_08", "a2_d_12", "a2_d_13", "a2_d_14")
_EXPECTED_LABELS = ("a2_l", "a2_r")
_EXPECTED_FIELDS = (
    "ball_SR",
    "t_ball",
    "fwd",
    "yaw",
    "lat",
    "ephys_SR",
    "t_ephys",
    "stim",
    "ephys_A",
    "ephys_B",
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp = Path(handle.name)
    os.replace(temp, path)


def load_authority(path: str | Path = DEFAULT_AUTHORITY) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DNa02 field-map authority must be a JSON object")
    claimed = str(payload.get("authority_sha256", ""))
    body = {key: value for key, value in payload.items() if key != "authority_sha256"}
    if canonical_sha256(body) != claimed:
        raise ValueError("DNa02 field-map authority canonical hash does not match")
    if payload.get("schema") != "fly-sniff-dna02-field-map-authority-v1":
        raise ValueError("unsupported DNa02 field-map authority schema")
    if payload.get("status") != "FROZEN_MAPPING_PENDING_SCHEMA_CONFIRMATION":
        raise ValueError("DNa02 field-map authority status has changed")
    if payload.get("navigation_performance_used") is not False:
        raise ValueError("DNa02 field-map authority may not use navigation performance")

    code = payload.get("code_authority")
    if not isinstance(code, dict):
        raise TypeError("code_authority must be an object")
    if code.get("repository") != _EXPECTED_REPOSITORY or code.get("commit") != _EXPECTED_COMMIT:
        raise ValueError("DNa02 field-map code authority repository/commit has drifted")
    notebook = code.get("import_notebook")
    loader = code.get("loader")
    if not isinstance(notebook, dict) or not isinstance(loader, dict):
        raise TypeError("field-map notebook/loader authorities must be objects")
    if notebook.get("path") != "import_preprocess_data.ipynb" or notebook.get("git_blob_sha1") != _EXPECTED_NOTEBOOK_BLOB:
        raise ValueError("DNa02 import-notebook authority has drifted")
    if (
        loader.get("path") != "a2lib/proc_utils.py"
        or loader.get("git_blob_sha1") != _EXPECTED_LOADER_BLOB
        or loader.get("function") != "load_sr_matfile_data"
    ):
        raise ValueError("DNa02 raw-loader authority has drifted")

    cohort = payload.get("bilateral_cohort_labels")
    if not isinstance(cohort, dict) or tuple(sorted(cohort)) != _EXPECTED_ALIASES:
        raise ValueError("DNa02 bilateral field-map cohort has drifted")
    if any(tuple(cohort[alias]) != _EXPECTED_LABELS for alias in _EXPECTED_ALIASES):
        raise ValueError("DNa02 bilateral neuron-label ordering must remain [a2_l, a2_r]")

    required = payload.get("required_raw_fields")
    if not isinstance(required, dict) or tuple(required) != _EXPECTED_FIELDS:
        raise ValueError("DNa02 required raw-field set/order has drifted")
    mapping = payload.get("frozen_channel_mapping")
    if not isinstance(mapping, dict):
        raise TypeError("frozen_channel_mapping must be an object")
    if mapping.get("ephys_A", {}).get("secondary_label") != "a2_l" or mapping.get("ephys_A", {}).get("soma_side") != "L":
        raise ValueError("ephys_A must remain mapped to left a2/DNa02")
    if mapping.get("ephys_B", {}).get("secondary_label") != "a2_r" or mapping.get("ephys_B", {}).get("soma_side") != "R":
        raise ValueError("ephys_B must remain mapped to right a2/DNa02")
    if "yaw" not in mapping:
        raise ValueError("yaw behavior mapping is required")
    return payload


def _inspection_field_metadata(schema_inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(schema_inventory, dict):
        raise TypeError("schema_inventory must be an object")
    if isinstance(schema_inventory.get("variables"), list):
        return {
            str(item["name"]): {
                "shape": item.get("shape"),
                "matlab_class": item.get("matlab_class"),
            }
            for item in schema_inventory["variables"]
            if isinstance(item, dict) and "name" in item
        }
    datasets = schema_inventory.get("datasets")
    top_level = schema_inventory.get("top_level")
    if isinstance(datasets, list) and isinstance(top_level, list):
        by_name = {
            str(item["name"]): {"shape": item.get("shape"), "dtype": item.get("dtype")}
            for item in datasets
            if isinstance(item, dict) and "name" in item
        }
        return {str(name): by_name.get(str(name), {"top_level_object": True}) for name in top_level}
    raise ValueError("inspection receipt exposes neither classic MATLAB variables nor HDF5 top-level metadata")


def confirm_field_map(
    inspection_path: str | Path,
    *,
    authority_path: str | Path = DEFAULT_AUTHORITY,
    contract_path: str | Path = DEFAULT_CONTRACT,
    evidence_path: str | Path = DEFAULT_EVIDENCE,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    authority = load_authority(authority_path)
    inspection = json.loads(Path(inspection_path).read_text(encoding="utf-8"))
    if not isinstance(inspection, dict):
        raise TypeError("DNa02 source inspection receipt must be a JSON object")
    claimed = str(inspection.get("inspection_sha256", ""))
    inspection_body = {
        key: value for key, value in inspection.items() if key != "inspection_sha256"
    }
    if canonical_sha256(inspection_body) != claimed:
        raise ValueError("DNa02 source inspection receipt canonical hash does not match")
    confirmation = authority["schema_confirmation"]
    if inspection.get("schema") != confirmation["receipt_schema"]:
        raise ValueError("unexpected DNa02 source inspection schema")
    if inspection.get("status") != confirmation["required_receipt_status"]:
        raise ValueError("DNa02 source inspection receipt is not at the required pre-extraction state")
    if inspection.get("navigation_performance_used") is not False:
        raise ValueError("schema confirmation may not consume navigation performance")
    if inspection.get("physiology_statistic_computed") is not False:
        raise ValueError("schema confirmation requires a receipt created before physiology extraction")
    if inspection.get("raw_values_exported") is not False:
        raise ValueError("schema confirmation receipt must not export raw values")

    contract = load_contract(contract_path)
    evidence = _load_byte_evidence(Path(evidence_path))
    if inspection.get("source_contract_sha256") != contract.sha256:
        raise ValueError("inspection receipt does not bind the current DNa02 source contract")
    if inspection.get("source_byte_evidence_sha256") != evidence["evidence_sha256"]:
        raise ValueError("inspection receipt does not bind the current DNa02 source-byte evidence")

    files = inspection.get("files")
    if not isinstance(files, list) or len(files) != len(_EXPECTED_ALIASES):
        raise ValueError("inspection receipt must contain the four bilateral DNa02 source files")
    by_alias = {str(item.get("fly_alias")): item for item in files if isinstance(item, dict)}
    if tuple(sorted(by_alias)) != _EXPECTED_ALIASES:
        raise ValueError("inspection receipt bilateral cohort differs from frozen field-map authority")
    contract_by_alias = {ref.fly_alias: ref for ref in contract.data_file_map}

    required_fields = set(_EXPECTED_FIELDS)
    confirmed_files: list[dict[str, Any]] = []
    for alias in _EXPECTED_ALIASES:
        item = by_alias[alias]
        ref = contract_by_alias[alias]
        if item.get("filename") != ref.filename or item.get("sha256") != ref.sha256:
            raise ValueError(f"inspection source identity differs from contract for {alias}")
        metadata = _inspection_field_metadata(item.get("schema_inventory"))
        missing = sorted(required_fields - set(metadata))
        if missing:
            raise ValueError(f"authenticated source {alias} is missing frozen raw fields: {missing}")
        confirmed_files.append(
            {
                "fly_alias": alias,
                "filename": ref.filename,
                "sha256": ref.sha256,
                "mat_format": item["schema_inventory"].get("mat_format"),
                "required_field_metadata": {
                    field: metadata[field] for field in _EXPECTED_FIELDS
                },
            }
        )

    result: dict[str, Any] = {
        "schema": "fly-sniff-dna02-field-map-confirmation-v1",
        "status": "FIELD_MAP_CONFIRMED_PENDING_NUMERIC_EXTRACTION_REVIEW",
        "field_map_authority_sha256": authority["authority_sha256"],
        "source_inspection_sha256": claimed,
        "source_contract_sha256": contract.sha256,
        "source_byte_evidence_sha256": evidence["evidence_sha256"],
        "frozen_channel_mapping": authority["frozen_channel_mapping"],
        "required_raw_fields": list(_EXPECTED_FIELDS),
        "files": confirmed_files,
        "navigation_performance_used": False,
        "raw_values_read": False,
        "physiology_statistic_computed": False,
        "next_allowed_action": (
            "Implement the prespecified DNa02 numeric extraction against this confirmed field mapping; "
            "do not alter channel laterality or raw-field selection based on extracted or navigation results."
        ),
    }
    result["confirmation_sha256"] = canonical_sha256(result)
    if output_path is not None:
        _atomic_write_json(Path(output_path), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen DNa02 raw-field authority and optionally confirm an authenticated schema receipt"
    )
    parser.add_argument("--authority", default=str(DEFAULT_AUTHORITY))
    parser.add_argument("--inspection")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.inspection is None:
        result = load_authority(args.authority)
    else:
        if args.out is None:
            parser.error("--out is required when --inspection is supplied")
        result = confirm_field_map(
            args.inspection,
            authority_path=args.authority,
            output_path=args.out,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
