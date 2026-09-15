from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .dna02_source import READY, load_contract
from .dna02_source_inspect import DEFAULT_CONTRACT, DEFAULT_EVIDENCE, _load_byte_evidence
from .freeze import canonical_sha256

CHUNK_BYTES = 8 * 1024 * 1024
DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
_USER_AGENT = "fly-sniff-dna02-local-fetch/1.0 (verified scientific source acquisition)"


def expected_sources(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT,
    evidence_path: str | Path = DEFAULT_EVIDENCE,
) -> list[dict[str, Any]]:
    contract = load_contract(contract_path)
    if contract.status != READY:
        raise ValueError("DNa02 source contract is not READY_FOR_EXTRACTION")
    evidence = _load_byte_evidence(Path(evidence_path))
    evidence_by_alias = {str(item["fly_alias"]): item for item in evidence["files"]}
    if len(evidence_by_alias) != 4:
        raise ValueError("DNa02 source-byte evidence must contain four unique fly aliases")

    specs: list[dict[str, Any]] = []
    for ref in contract.data_file_map:
        item = evidence_by_alias.get(ref.fly_alias)
        if item is None:
            raise ValueError(f"missing source-byte evidence for {ref.fly_alias}")
        if int(item["file_id"]) != ref.file_id or str(item["filename"]) != ref.filename:
            raise ValueError(f"source contract/evidence identity mismatch for {ref.fly_alias}")
        if str(item["observed_sha256"]) != ref.sha256:
            raise ValueError(f"source contract/evidence SHA-256 mismatch for {ref.fly_alias}")
        specs.append(
            {
                "fly_alias": ref.fly_alias,
                "file_id": ref.file_id,
                "filename": ref.filename,
                "byte_count": int(item["byte_count"]),
                "md5": str(item["observed_md5"]),
                "sha256": ref.sha256,
            }
        )
    return sorted(specs, key=lambda item: item["fly_alias"])


def _verify_hashes(
    *,
    filename: str,
    size: int,
    md5: str,
    sha256: str,
    expected: dict[str, Any],
) -> None:
    if size != int(expected["byte_count"]):
        raise ValueError(
            f"byte-count mismatch for {filename}: expected {expected['byte_count']}, got {size}"
        )
    if md5 != str(expected["md5"]):
        raise ValueError(f"MD5 mismatch for {filename}")
    if sha256 != str(expected["sha256"]):
        raise ValueError(f"SHA-256 mismatch for {filename}")


def _hash_existing(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return size, md5.hexdigest(), sha256.hexdigest()


def download_one(
    spec: dict[str, Any],
    *,
    output_dir: Path,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / str(spec["filename"])
    if target.exists():
        size, md5, sha256 = _hash_existing(target)
        _verify_hashes(
            filename=target.name,
            size=size,
            md5=md5,
            sha256=sha256,
            expected=spec,
        )
        return {
            **spec,
            "status": "verified_existing",
            "downloaded": False,
            "navigation_performance_used": False,
        }

    request_url = DATAVERSE_FILE_URL.format(file_id=int(spec["file_id"]))
    request = Request(
        request_url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/octet-stream"},
    )
    md5_hasher = hashlib.md5(usedforsecurity=False)
    sha_hasher = hashlib.sha256()
    size = 0
    temp_path: Path | None = None
    try:
        with urlopen(request, timeout=timeout_s) as response:
            final_url = str(response.geturl())
            if not final_url.startswith("https://"):
                raise ValueError("Dataverse source redirected to a non-HTTPS URL")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != int(spec["byte_count"]):
                raise ValueError(
                    f"Content-Length mismatch for {spec['filename']}: "
                    f"expected {spec['byte_count']}, got {content_length}"
                )
            with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as handle:
                temp_path = Path(handle.name)
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > int(spec["byte_count"]):
                        raise ValueError(f"source exceeded frozen byte count for {spec['filename']}")
                    md5_hasher.update(chunk)
                    sha_hasher.update(chunk)
                    handle.write(chunk)
        observed_md5 = md5_hasher.hexdigest()
        observed_sha256 = sha_hasher.hexdigest()
        _verify_hashes(
            filename=str(spec["filename"]),
            size=size,
            md5=observed_md5,
            sha256=observed_sha256,
            expected=spec,
        )
        if temp_path is None:
            raise RuntimeError("temporary source path was not created")
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return {
        **spec,
        "status": "downloaded_and_verified",
        "downloaded": True,
        "navigation_performance_used": False,
    }


def download_sources(
    output_dir: str | Path,
    *,
    contract_path: str | Path = DEFAULT_CONTRACT,
    evidence_path: str | Path = DEFAULT_EVIDENCE,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    specs = expected_sources(contract_path=contract_path, evidence_path=evidence_path)
    files = [download_one(spec, output_dir=output, timeout_s=timeout_s) for spec in specs]
    contract = load_contract(contract_path)
    evidence = _load_byte_evidence(Path(evidence_path))
    payload: dict[str, Any] = {
        "schema": "fly-sniff-dna02-local-source-fetch-v1",
        "status": "ALL_FOUR_SOURCES_VERIFIED_LOCAL",
        "source_contract_sha256": contract.sha256,
        "source_byte_evidence_sha256": evidence["evidence_sha256"],
        "navigation_performance_used": False,
        "files": files,
        "next_allowed_action": "Run fly-sniff-dna02-inspect on these exact four files.",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt = output / "dna02-local-source-fetch-receipt.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=output, delete=False) as handle:
        handle.write(encoded)
        temp_receipt = Path(handle.name)
    os.replace(temp_receipt, receipt)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and authenticate the four frozen DNa02 Figure 3C source files"
    )
    parser.add_argument("--out", required=True, help="Directory for the verified .mat files")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    result = download_sources(
        args.out,
        contract_path=args.contract,
        evidence_path=args.evidence,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
