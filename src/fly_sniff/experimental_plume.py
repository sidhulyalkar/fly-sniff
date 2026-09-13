from __future__ import annotations

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

PROTOCOL = "experimental-plume-sensory-validation-v3"
SCHEMA_VERSION = 1
DEFAULT_CONFIG_RESOURCE = "configs/experimental_plume_v3.json"


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_config_text() -> str:
    resource = files("fly_sniff").joinpath(DEFAULT_CONFIG_RESOURCE)
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    return (Path(__file__).resolve().parents[2] / "configs" / "experimental_plume_v3.json").read_text()


def load_experimental_plume_config(path: str | Path | None = None) -> dict[str, Any]:
    document = json.loads(Path(path).read_text() if path is not None else _default_config_text())
    if document.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported experimental plume protocol {document.get('protocol')!r}")
    if document.get("controller_access") is not False:
        raise ValueError("experimental plume validation must not grant controller access")
    if document.get("navigation_performance_used") is not False:
        raise ValueError("experimental plume validation must not use navigation performance")
    if document.get("may_select_source_or_preprocessing_from_navigation") is not False:
        raise ValueError("source/preprocessing selection must remain independent of navigation")
    return document


def upstream_contract_sha256(document: dict[str, Any]) -> str:
    payload = {
        "protocol": document["protocol"],
        "upstream": document["upstream"],
        "smooth": document["smooth"],
        "complex": document["complex"],
        "published_cue_contract": document["published_cue_contract"],
    }
    return _canonical_sha(payload)


def expected_source_template(document: dict[str, Any], plume: str) -> dict[str, Any]:
    if plume not in {"smooth", "complex"}:
        raise ValueError("plume must be 'smooth' or 'complex'")
    source = document[plume]
    if plume == "smooth":
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "plume": plume,
            "source_url": f"https://doi.org/{source['doi']}",
            "source_version": source["doi"],
            "source_file": source["file"],
            "source_sha256": "<required after download>",
            "source_dataset_path": source["dataset"],
            "source_shape": source["expected_shape"],
            "native_time_basis": {"fps": source["source_fps"], "kind": "fixed_rate"},
            "selected_frame_range": [0, source["expected_shape"][0]],
            "intensity_transform": "upstream crop/pad/resize/uint8 preparation; receipt exact implementation",
            "temporal_resampling": {
                "profile": source["temporal_profiles"]["primary"],
                "target_fps": source["temporal_profiles"]["target_fps"],
            },
            "spatial_transform": "<required exact transform>",
            "coordinate_mapping": "<required physical/image mapping>",
            "upstream_contract_sha256": upstream_contract_sha256(document),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "plume": plume,
        "source_url": f"https://doi.org/10.48324/dandi.{source['dandiset']}/{source['version']}",
        "source_version": source["version"],
        "source_file": source["path"],
        "source_sha256": "<required after download>",
        "source_dataset_path": source["dataset"],
        "source_shape": source["expected_shape"],
        "native_time_basis": "<required from NWB/archive timestamps>",
        "selected_frame_range": [0, source["expected_shape"][0]],
        "intensity_transform": f"scale={source['intensity_scale']}",
        "temporal_resampling": "none until native time basis is verified",
        "spatial_transform": "none for published temporal-map validation",
        "coordinate_mapping": "<required before bilateral agent sampling>",
        "upstream_contract_sha256": upstream_contract_sha256(document),
    }


def _require_concrete(value: Any, field: str) -> None:
    if value is None:
        raise ValueError(f"source receipt field {field!r} is unresolved")
    if isinstance(value, str) and (not value.strip() or value.startswith("<required")):
        raise ValueError(f"source receipt field {field!r} is unresolved")


def validate_source_receipt(receipt: dict[str, Any], document: dict[str, Any]) -> None:
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("protocol") != PROTOCOL:
        raise ValueError("unsupported experimental plume receipt protocol/schema")
    plume = receipt.get("plume")
    if plume not in {"smooth", "complex"}:
        raise ValueError("experimental plume receipt must identify smooth or complex source")
    for field in document["required_receipt_fields"]:
        if field not in receipt:
            raise ValueError(f"source receipt is missing required field {field!r}")
        _require_concrete(receipt[field], field)
    source = document[plume]
    if receipt["source_dataset_path"] != source["dataset"]:
        raise ValueError("source dataset path does not match frozen upstream contract")
    if list(receipt["source_shape"]) != list(source["expected_shape"]):
        raise ValueError("source shape does not match frozen upstream contract")
    if receipt["upstream_contract_sha256"] != upstream_contract_sha256(document):
        raise ValueError("upstream contract SHA-256 mismatch")
    sha = str(receipt["source_sha256"])
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha.lower()):
        raise ValueError("source_sha256 must be a concrete 64-character hexadecimal SHA-256")
    if plume == "smooth":
        if receipt["source_file"] != source["file"]:
            raise ValueError("smooth source filename does not match frozen contract")
        time_basis = receipt["native_time_basis"]
        if not isinstance(time_basis, dict) or float(time_basis.get("fps", -1)) != float(source["source_fps"]):
            raise ValueError("smooth native time basis does not match frozen source rate")
        resampling = receipt["temporal_resampling"]
        if not isinstance(resampling, dict):
            raise TypeError("smooth temporal_resampling must be explicit")
        if resampling.get("profile") != source["temporal_profiles"]["primary"]:
            raise ValueError("primary smooth validation must use the frozen corrected profile")
    else:
        if receipt["source_file"] != source["path"]:
            raise ValueError("complex DANDI asset path does not match frozen contract")
        if receipt["source_version"] != source["version"]:
            raise ValueError("complex DANDI version does not match frozen contract")


def write_template(path: str | Path, plume: str, document: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(expected_source_template(document, plume), indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the experimental-plume v3 provenance airlock")
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument("--plume", choices=("smooth", "complex"))
    parser.add_argument("--write-template")
    args = parser.parse_args()
    document = load_experimental_plume_config(args.config)
    if args.write_template:
        if args.plume is None:
            parser.error("--write-template requires --plume")
        print(write_template(args.write_template, args.plume, document))
        return
    if args.manifest is None:
        parser.error("provide --manifest or --write-template")
    receipt = json.loads(Path(args.manifest).read_text())
    validate_source_receipt(receipt, document)
    print("experimental plume source receipt: valid")
    print(upstream_contract_sha256(document))


if __name__ == "__main__":
    main()
