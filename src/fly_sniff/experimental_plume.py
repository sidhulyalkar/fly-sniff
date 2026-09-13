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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


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
    return _canonical_sha({
        "protocol": document["protocol"],
        "upstream": document["upstream"],
        "smooth": document["smooth"],
        "complex": document["complex"],
        "published_cue_contract": document["published_cue_contract"],
    })


def expected_source_template(document: dict[str, Any], plume: str) -> dict[str, Any]:
    if plume not in {"smooth", "complex"}:
        raise ValueError("plume must be 'smooth' or 'complex'")
    source = document[plume]
    if plume == "smooth":
        temporal = source["temporal_profiles"]
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
            "intensity_transform": "upstream prepare_smooth_frame contract; receipt exact implementation",
            "temporal_resampling": {
                "role": "publication_reproduction_only",
                "profile": temporal["publication_reproduction_primary"],
                "target_fps": temporal["publication_target_fps"],
                "creates_new_measurements": False,
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


def _expected_source_url(plume: str, source: dict[str, Any]) -> str:
    if plume == "smooth":
        return f"https://doi.org/{source['doi']}"
    return f"https://doi.org/10.48324/dandi.{source['dandiset']}/{source['version']}"


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
    if receipt["source_url"] != _expected_source_url(plume, source):
        raise ValueError("source URL does not match frozen upstream contract")
    expected_version = source["doi"] if plume == "smooth" else source["version"]
    if receipt["source_version"] != expected_version:
        raise ValueError("source version does not match frozen upstream contract")
    expected_file = source["file"] if plume == "smooth" else source["path"]
    if receipt["source_file"] != expected_file:
        raise ValueError("source file does not match frozen upstream contract")
    if receipt["source_dataset_path"] != source["dataset"]:
        raise ValueError("source dataset path does not match frozen upstream contract")
    if list(receipt["source_shape"]) != list(source["expected_shape"]):
        raise ValueError("source shape does not match frozen upstream contract")
    if list(receipt["selected_frame_range"]) != [0, int(source["expected_shape"][0])]:
        raise ValueError("selected frame range does not match frozen source-validation range")
    if receipt["upstream_contract_sha256"] != upstream_contract_sha256(document):
        raise ValueError("upstream contract SHA-256 mismatch")
    sha = str(receipt["source_sha256"]).lower()
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("source_sha256 must be a concrete 64-character hexadecimal SHA-256")
    if plume == "smooth":
        time_basis = receipt["native_time_basis"]
        if not isinstance(time_basis, dict) or float(time_basis.get("fps", -1)) != float(source["source_fps"]):
            raise ValueError("smooth native time basis does not match frozen source rate")
        resampling = receipt["temporal_resampling"]
        temporal = source["temporal_profiles"]
        if not isinstance(resampling, dict):
            raise TypeError("smooth temporal_resampling must be explicit")
        if resampling.get("role") != "publication_reproduction_only":
            raise ValueError("smooth interpolation must remain publication-reproduction only")
        if resampling.get("profile") != temporal["publication_reproduction_primary"]:
            raise ValueError("publication reproduction must use the frozen corrected profile")
        if float(resampling.get("target_fps", -1)) != float(temporal["publication_target_fps"]):
            raise ValueError("smooth publication target frame rate does not match frozen profile")
        if resampling.get("creates_new_measurements") is not False:
            raise ValueError("smooth interpolation must never be receipted as new measurements")
    elif receipt["intensity_transform"] != f"scale={source['intensity_scale']}":
        raise ValueError("complex intensity transform does not match frozen DANDI contract")


def verify_source_bytes(path: str | Path, receipt: dict[str, Any]) -> str:
    actual = file_sha256(path)
    expected = str(receipt["source_sha256"]).lower()
    if actual != expected:
        raise ValueError(f"source-file SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def write_template(path: str | Path, plume: str, document: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(expected_source_template(document, plume), indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the experimental-plume v3 provenance airlock")
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument("--source-file")
    parser.add_argument("--structural-only", action="store_true")
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
    if not args.structural_only:
        if args.source_file is None:
            parser.error("byte verification requires --source-file; use --structural-only explicitly otherwise")
        verify_source_bytes(args.source_file, receipt)
        print("experimental plume source bytes: verified")
    else:
        print("experimental plume manifest structure: valid; source bytes NOT VERIFIED")
    print(upstream_contract_sha256(document))


if __name__ == "__main__":
    main()
