from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .dna02_cohort_adjudication import load_adjudication
from .freeze import canonical_sha256

CHUNK_BYTES = 8 * 1024 * 1024
DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
_USER_AGENT = "fly-sniff-dna02-byte-hash/1.0 (scientific provenance only)"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def source_specs(adjudication_path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    adjudication = load_adjudication(adjudication_path)
    cohort = tuple(adjudication["resolved_figure3c_cohort"])
    rows = adjudication["dataverse_authority"]["bilateral_dna02_raw_folders"]
    by_alias = {str(row["fly_alias"]): row for row in rows}
    if set(by_alias) != set(cohort):
        raise ValueError("adjudicated cohort and Dataverse raw-folder set differ")

    specs: list[dict[str, Any]] = []
    for alias in cohort:
        row = by_alias[alias]
        if row["checksum_type"] != "MD5":
            raise ValueError(f"expected Dataverse MD5 authority for {alias}")
        if bool(row["restricted"]):
            raise ValueError(f"adjudicated source file is unexpectedly restricted: {alias}")
        specs.append(
            {
                "fly_alias": alias,
                "file_id": int(row["file_id"]),
                "filename": str(row["filename"]),
                "expected_size_bytes": int(row["filesize"]),
                "expected_md5": str(row["checksum_value"]),
            }
        )
    return str(adjudication["adjudication_sha256"]), specs


def stream_hash_dataverse_file(
    spec: dict[str, Any],
    *,
    timeout_s: float = 120.0,
    chunk_bytes: int = CHUNK_BYTES,
) -> dict[str, Any]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    expected_size = int(spec["expected_size_bytes"])
    if expected_size <= 0:
        raise ValueError("expected source filesize must be positive")
    file_id = int(spec["file_id"])
    url = DATAVERSE_FILE_URL.format(file_id=file_id)
    request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/octet-stream"})

    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    observed_size = 0
    with urlopen(request, timeout=timeout_s) as response:
        final_url = str(response.geturl())
        if not final_url.startswith("https://"):
            raise ValueError("Dataverse source file redirected to a non-HTTPS URL")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != expected_size:
            raise ValueError(
                f"Dataverse Content-Length changed for {spec['fly_alias']}: "
                f"expected {expected_size}, got {content_length}"
            )
        while True:
            chunk = response.read(chunk_bytes)
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > expected_size:
                raise ValueError(f"Dataverse source exceeded frozen size for {spec['fly_alias']}")
            md5.update(chunk)
            sha256.update(chunk)

    if observed_size != expected_size:
        raise ValueError(
            f"Dataverse source size mismatch for {spec['fly_alias']}: "
            f"expected {expected_size}, got {observed_size}"
        )
    observed_md5 = md5.hexdigest()
    if observed_md5 != spec["expected_md5"]:
        raise ValueError(
            f"Dataverse MD5 mismatch for {spec['fly_alias']}; refusing SHA-256 promotion"
        )

    return {
        "fly_alias": spec["fly_alias"],
        "file_id": file_id,
        "filename": spec["filename"],
        "source_url": url,
        "final_url": final_url,
        "byte_count": observed_size,
        "published_md5": spec["expected_md5"],
        "observed_md5": observed_md5,
        "observed_sha256": sha256.hexdigest(),
        "md5_matches_dataverse_authority": True,
        "raw_bytes_retained": False,
        "navigation_performance_used": False,
    }


def hash_adjudicated_sources(
    output_path: str | Path,
    *,
    adjudication_path: str | Path,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    adjudication_sha256, specs = source_specs(adjudication_path)
    receipts = [stream_hash_dataverse_file(spec, timeout_s=timeout_s) for spec in specs]
    payload = {
        "schema": "fly-sniff-dna02-source-byte-manifest-v1",
        "status": "SHA256_BYTE_IDENTITIES_RESOLVED",
        "adjudication_sha256": adjudication_sha256,
        "persistent_id": "doi:10.7910/DVN/0NCLP1",
        "navigation_performance_used": False,
        "raw_bytes_retained": False,
        "files": receipts,
        "boundary": (
            "This manifest resolves source-byte identity only. It does not compute physiology "
            "statistics, calibrate model dynamics, or authorize navigation evaluation."
        ),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    _atomic_write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream-hash the four adjudicated DNa02 source files without retaining raw bytes"
    )
    parser.add_argument(
        "--adjudication",
        default="authority/program-a-dna02-cohort-adjudication-v1.json",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    result = hash_adjudicated_sources(
        args.out,
        adjudication_path=args.adjudication,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
