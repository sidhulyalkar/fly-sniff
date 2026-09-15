from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from scipy.io import whosmat

from .dna02_source import READY, load_contract
from .freeze import canonical_sha256

CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_CONTRACT = Path("authority/program-a-dna02-source-contract-v1.json")
DEFAULT_EVIDENCE = Path("authority/program-a-dna02-source-byte-evidence-v1.json")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _hash_file(path: Path, *, chunk_bytes: int = CHUNK_BYTES) -> tuple[int, str, str]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return size, md5.hexdigest(), sha256.hexdigest()


def _load_byte_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DNa02 source-byte evidence must be a JSON object")
    claimed = str(payload.get("evidence_sha256", ""))
    without_hash = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    if canonical_sha256(without_hash) != claimed:
        raise ValueError("DNa02 source-byte evidence canonical hash does not match")
    if payload.get("status") != "SHA256_BYTE_IDENTITIES_RESOLVED":
        raise ValueError("DNa02 source-byte evidence is not resolved")
    return payload


def detect_mat_format(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.read(128)
    if b"MATLAB 7.3 MAT-file" in header:
        return "matlab_v7_3_hdf5"
    if b"MATLAB 5.0 MAT-file" in header:
        return "matlab_v5"
    return "matlab_classic_or_unknown"


def _inspect_classic(path: Path) -> dict[str, Any]:
    try:
        variables = whosmat(path)
    except Exception as exc:  # scipy exposes several format-specific exception classes
        raise ValueError(f"scipy could not inventory MATLAB variables in {path.name}: {exc}") from exc
    return {
        "inventory_method": "scipy.io.whosmat",
        "variables": [
            {"name": name, "shape": list(shape), "matlab_class": matlab_class}
            for name, shape, matlab_class in variables
        ],
    }


def _inspect_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "MATLAB v7.3/HDF5 source detected. Install the source extra with "
            "`python -m pip install -e '.[source]'` and rerun inspection."
        ) from exc

    datasets: list[dict[str, Any]] = []
    groups: list[str] = []
    skipped_reference_objects = 0
    with h5py.File(path, "r") as handle:
        top_level = sorted(str(name) for name in handle)

        def visit(name: str, obj: Any) -> None:
            nonlocal skipped_reference_objects
            if name == "#refs#" or name.startswith("#refs#/"):
                skipped_reference_objects += 1
                return
            if isinstance(obj, h5py.Dataset):
                datasets.append(
                    {
                        "name": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )
            elif isinstance(obj, h5py.Group):
                groups.append(name)

        handle.visititems(visit)
    return {
        "inventory_method": "h5py metadata only; #refs# payload omitted",
        "top_level": top_level,
        "groups": sorted(groups),
        "datasets": sorted(datasets, key=lambda item: item["name"]),
        "skipped_reference_objects": skipped_reference_objects,
    }


def inspect_mat_schema(path: Path) -> dict[str, Any]:
    mat_format = detect_mat_format(path)
    if mat_format == "matlab_v7_3_hdf5":
        inventory = _inspect_hdf5(path)
    else:
        inventory = _inspect_classic(path)
    return {"mat_format": mat_format, **inventory}


def _match_local_paths(paths: list[Path], expected_filenames: set[str]) -> dict[str, Path]:
    by_name: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"DNa02 source path is not a file: {path}")
        name = path.name
        if name not in expected_filenames:
            raise ValueError(f"unexpected DNa02 source filename: {name}")
        if name in by_name:
            raise ValueError(f"duplicate DNa02 source filename supplied: {name}")
        by_name[name] = path
    missing = expected_filenames - set(by_name)
    if missing:
        raise ValueError(f"missing frozen DNa02 source files: {sorted(missing)}")
    return by_name


def inspect_sources(
    source_paths: list[str | Path],
    *,
    contract_path: str | Path = DEFAULT_CONTRACT,
    evidence_path: str | Path = DEFAULT_EVIDENCE,
    output_path: str | Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if contract.status != READY:
        raise ValueError("DNa02 source contract is not READY_FOR_EXTRACTION")
    evidence = _load_byte_evidence(Path(evidence_path))
    evidence_files = evidence.get("files")
    if not isinstance(evidence_files, list) or len(evidence_files) != 4:
        raise ValueError("DNa02 byte evidence must contain exactly four files")

    evidence_by_name = {str(item["filename"]): item for item in evidence_files}
    if len(evidence_by_name) != len(evidence_files):
        raise ValueError("DNa02 byte evidence filenames must be unique")
    contract_by_name = {ref.filename: ref for ref in contract.data_file_map}
    if set(contract_by_name) != set(evidence_by_name):
        raise ValueError("DNa02 source contract and byte evidence filename sets differ")

    local_by_name = _match_local_paths(
        [Path(value).expanduser().resolve() for value in source_paths],
        set(contract_by_name),
    )

    reports: list[dict[str, Any]] = []
    for filename in sorted(contract_by_name):
        ref = contract_by_name[filename]
        evidence_ref = evidence_by_name[filename]
        path = local_by_name[filename]
        size, observed_md5, observed_sha256 = _hash_file(path)
        if size != int(evidence_ref["byte_count"]):
            raise ValueError(
                f"byte-count mismatch for {filename}: expected {evidence_ref['byte_count']}, got {size}"
            )
        if observed_md5 != str(evidence_ref["observed_md5"]):
            raise ValueError(f"MD5 mismatch for {filename}")
        if observed_sha256 != ref.sha256 or observed_sha256 != str(evidence_ref["observed_sha256"]):
            raise ValueError(f"SHA-256 mismatch for {filename}")

        reports.append(
            {
                "fly_alias": ref.fly_alias,
                "file_id": ref.file_id,
                "filename": filename,
                "byte_count": size,
                "md5": observed_md5,
                "sha256": observed_sha256,
                "schema_inventory": inspect_mat_schema(path),
            }
        )

    payload: dict[str, Any] = {
        "schema": "fly-sniff-dna02-source-inspection-v1",
        "status": "SCHEMA_INVENTORIED_PENDING_EXTRACTION_REVIEW",
        "source_contract_sha256": contract.sha256,
        "source_byte_evidence_sha256": evidence["evidence_sha256"],
        "navigation_performance_used": False,
        "physiology_statistic_computed": False,
        "raw_values_exported": False,
        "files": reports,
        "next_allowed_action": (
            "Review the authenticated MATLAB schema inventory and freeze a deterministic field mapping "
            "for the already-prespecified Figure 3C statistic before extracting values."
        ),
        "forbidden_interpretation": [
            "schema inspection calibrates DNa02 dynamics",
            "schema inspection validates the 150 ms alignment as a universal biological delay",
            "schema inspection authorizes odor-navigation evaluation",
        ],
    }
    payload["inspection_sha256"] = canonical_sha256(payload)
    _atomic_write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate and inventory the four frozen DNa02 MATLAB source files without "
            "extracting physiology values"
        )
    )
    parser.add_argument("sources", nargs="+", help="The four exact Dataverse .mat files")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = inspect_sources(
        args.sources,
        contract_path=args.contract,
        evidence_path=args.evidence,
        output_path=args.out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
