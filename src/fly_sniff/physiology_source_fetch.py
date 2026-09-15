from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from .dna02_source import load_contract as load_dna02_contract

PFN_README_URL = "https://datadryad.org/downloads/file_stream/536042"
PFN_README_FILENAME = "Currier2020README.rtf"
PFN_README_EXPECTED_MD5 = "91e5213503788fcde11c0f5aa3e91f43"
PFN_README_MAX_BYTES = 65_536

DNA02_PERSISTENT_ID = "doi:10.7910/DVN/0NCLP1"
DNA02_DATAVERSE_API = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId="
    + quote(DNA02_PERSISTENT_ID, safe="")
)
DNA02_METADATA_MAX_BYTES = 16 * 1024 * 1024

_USER_AGENT = "fly-sniff-source-ingress/1.0 (scientific provenance fetch)"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _bounded_get(url: str, *, max_bytes: int, timeout_s: float = 30.0) -> tuple[bytes, dict[str, str]]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - frozen HTTPS authorities
        final_url = str(response.geturl())
        if not final_url.startswith("https://"):
            raise ValueError("source fetch redirected to a non-HTTPS URL")
        length_header = response.headers.get("Content-Length")
        if length_header is not None and int(length_header) > max_bytes:
            raise ValueError(
                f"source advertised {length_header} bytes, exceeding cap of {max_bytes}"
            )
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError(f"source exceeded byte cap of {max_bytes}")
        headers = {
            "content_type": str(response.headers.get("Content-Type", "")),
            "content_length": str(response.headers.get("Content-Length", "")),
            "etag": str(response.headers.get("ETag", "")),
            "last_modified": str(response.headers.get("Last-Modified", "")),
            "final_url": final_url,
        }
    return payload, headers


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def fetch_pfn_readme(
    output_dir: str | Path,
    *,
    url: str = PFN_README_URL,
    max_bytes: int = PFN_README_MAX_BYTES,
) -> dict[str, Any]:
    """Fetch only the tiny Dryad README and emit a content-addressed receipt."""

    output = Path(output_dir)
    raw, headers = _bounded_get(url, max_bytes=max_bytes)
    observed_md5 = md5_bytes(raw)
    observed_sha256 = sha256_bytes(raw)
    receipt = {
        "schema": "fly-sniff-pfn-readme-fetch-v1",
        "navigation_performance_used": False,
        "requested_url": url,
        "final_url": headers["final_url"],
        "filename": PFN_README_FILENAME,
        "byte_count": len(raw),
        "max_bytes": max_bytes,
        "published_md5": PFN_README_EXPECTED_MD5,
        "observed_md5": observed_md5,
        "observed_sha256": observed_sha256,
        "md5_matches_published_authority": observed_md5 == PFN_README_EXPECTED_MD5,
        "http": {key: value for key, value in headers.items() if key != "final_url"},
    }
    if observed_md5 != PFN_README_EXPECTED_MD5:
        raise ValueError(
            "PFN README bytes do not match the published MD5; refusing to publish a receipt"
        )
    _atomic_write_bytes(output / PFN_README_FILENAME, raw)
    _atomic_write_json(output / "pfn-readme-fetch-receipt.json", receipt)
    return receipt


def _require_dict(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def normalize_dataverse_inventory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize Dataverse latestVersion file metadata without downloading file contents."""

    root = _require_dict(payload, field="Dataverse response")
    if root.get("status") != "OK":
        raise ValueError(f"Dataverse response status is not OK: {root.get('status')!r}")
    data = _require_dict(root.get("data"), field="Dataverse data")
    latest = _require_dict(data.get("latestVersion"), field="Dataverse latestVersion")
    files = latest.get("files")
    if not isinstance(files, list):
        raise TypeError("Dataverse latestVersion.files must be a list")

    inventory: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for entry in files:
        item = _require_dict(entry, field="Dataverse file entry")
        data_file = _require_dict(item.get("dataFile"), field="Dataverse dataFile")
        file_id = int(data_file["id"])
        if file_id <= 0:
            raise ValueError("Dataverse file IDs must be positive")
        if file_id in seen_ids:
            raise ValueError(f"duplicate Dataverse file id: {file_id}")
        seen_ids.add(file_id)
        checksum = data_file.get("checksum") or {}
        if not isinstance(checksum, dict):
            raise TypeError("Dataverse checksum must be an object when present")
        inventory.append(
            {
                "file_id": file_id,
                "filename": str(data_file.get("filename", "")),
                "directory_label": str(data_file.get("directoryLabel", "")),
                "filesize": int(data_file.get("filesize", 0)),
                "content_type": str(data_file.get("contentType", "")),
                "restricted": bool(item.get("restricted", False)),
                "checksum_type": None if not checksum else str(checksum.get("type", "")),
                "checksum_value": None if not checksum else str(checksum.get("value", "")),
            }
        )
    inventory.sort(key=lambda item: (item["directory_label"], item["filename"], item["file_id"]))
    return inventory


def _candidate_terms_from_dna02_contract(contract_path: str | Path) -> dict[str, list[str]]:
    contract = load_dna02_contract(contract_path)
    raw_map = dict(contract.known_raw_session_candidates)
    return {
        alias: list(dict.fromkeys([alias, *raw_map.get(alias, ())]))
        for alias in contract.candidate_bilateral_aliases
    }


def build_dna02_metadata_review(
    raw_payload: dict[str, Any],
    *,
    contract_path: str | Path,
    raw_sha256: str,
) -> dict[str, Any]:
    inventory = normalize_dataverse_inventory(raw_payload)
    terms = _candidate_terms_from_dna02_contract(contract_path)
    candidate_matches: dict[str, list[dict[str, Any]]] = {}
    for alias, search_terms in terms.items():
        matches: list[dict[str, Any]] = []
        for item in inventory:
            haystack = f"{item['directory_label']}/{item['filename']}".lower()
            matched_terms = [term for term in search_terms if term.lower() in haystack]
            if matched_terms:
                matches.append({**item, "matched_terms": matched_terms})
        candidate_matches[alias] = matches

    data = _require_dict(raw_payload["data"], field="Dataverse data")
    latest = _require_dict(data["latestVersion"], field="Dataverse latestVersion")
    return {
        "schema": "fly-sniff-dna02-dataverse-metadata-review-v1",
        "navigation_performance_used": False,
        "persistent_id": DNA02_PERSISTENT_ID,
        "raw_metadata_sha256": raw_sha256,
        "dataset_id": data.get("id"),
        "version_number": latest.get("versionNumber"),
        "version_minor_number": latest.get("versionMinorNumber"),
        "version_state": latest.get("versionState"),
        "file_count": len(inventory),
        "inventory": inventory,
        "candidate_matches": candidate_matches,
        "automatic_cohort_resolution": False,
        "automatic_file_map_promotion": False,
        "review_boundary": (
            "Filename/session matching is discovery evidence only. It must not resolve the Figure 3C "
            "cohort or promote file mappings without independent panel-level authority and byte hashes."
        ),
    }


def fetch_dna02_dataverse_metadata(
    output_dir: str | Path,
    *,
    contract_path: str | Path,
    url: str = DNA02_DATAVERSE_API,
    max_bytes: int = DNA02_METADATA_MAX_BYTES,
) -> dict[str, Any]:
    """Fetch Harvard Dataverse metadata only, never neural file bytes."""

    output = Path(output_dir)
    raw, headers = _bounded_get(url, max_bytes=max_bytes)
    raw_sha256 = sha256_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Harvard Dataverse metadata response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError("Harvard Dataverse metadata response must be a JSON object")
    review = build_dna02_metadata_review(
        payload,
        contract_path=contract_path,
        raw_sha256=raw_sha256,
    )
    receipt = {
        "schema": "fly-sniff-dna02-dataverse-fetch-v1",
        "navigation_performance_used": False,
        "requested_url": url,
        "final_url": headers["final_url"],
        "byte_count": len(raw),
        "max_bytes": max_bytes,
        "raw_metadata_sha256": raw_sha256,
        "http": {key: value for key, value in headers.items() if key != "final_url"},
    }
    _atomic_write_bytes(output / "dna02-dataverse-raw.json", raw)
    _atomic_write_json(output / "dna02-dataverse-fetch-receipt.json", receipt)
    _atomic_write_json(output / "dna02-dataverse-review.json", review)
    return review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch only bounded physiology-source metadata/README evidence for Program A"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pfn = subparsers.add_parser("pfn-readme", help="Fetch and hash only Currier2020README.rtf")
    pfn.add_argument("--out", required=True)
    pfn.add_argument("--url", default=PFN_README_URL)
    pfn.add_argument("--max-bytes", type=int, default=PFN_README_MAX_BYTES)

    dna02 = subparsers.add_parser("dna02-metadata", help="Fetch Harvard Dataverse metadata only")
    dna02.add_argument("--out", required=True)
    dna02.add_argument(
        "--contract",
        default="authority/program-a-dna02-source-contract-v1.json",
    )
    dna02.add_argument("--url", default=DNA02_DATAVERSE_API)
    dna02.add_argument("--max-bytes", type=int, default=DNA02_METADATA_MAX_BYTES)

    args = parser.parse_args(argv)
    if args.command == "pfn-readme":
        result = fetch_pfn_readme(args.out, url=args.url, max_bytes=args.max_bytes)
    else:
        result = fetch_dna02_dataverse_metadata(
            args.out,
            contract_path=args.contract,
            url=args.url,
            max_bytes=args.max_bytes,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
