from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from .physiology_source_fetch import (
    _atomic_write_bytes,
    _atomic_write_json,
    _bounded_get,
    md5_bytes,
    sha256_bytes,
)

PFN_DRYAD_DOI = "10.5061/dryad.vq83bk3rh"
PFN_README_FILE_STREAM_ID = 536042
PFN_README_PRIMARY_URL = f"https://datadryad.org/downloads/file_stream/{PFN_README_FILE_STREAM_ID}"
PFN_README_MIRROR_RECORD = "https://zenodo.org/records/4407395"
PFN_README_MIRROR_URL = (
    f"{PFN_README_MIRROR_RECORD}/files/Currier2020README.rtf?download=1"
)
PFN_README_FILENAME = "Currier2020README.rtf"
PFN_README_EXPECTED_MD5 = "91e5213503788fcde11c0f5aa3e91f43"
PFN_README_MAX_BYTES = 65_536


def fetch_pfn_readme(
    output_dir: str | Path,
    *,
    primary_url: str = PFN_README_PRIMARY_URL,
    mirror_url: str | None = PFN_README_MIRROR_URL,
    max_bytes: int = PFN_README_MAX_BYTES,
) -> dict[str, Any]:
    """Fetch the PFN README while preserving Dryad as checksum authority.

    The Zenodo copy is transport-only and is used only when the canonical Dryad
    endpoint rejects anonymous download with HTTP 401/403. Any accepted bytes
    must still match the MD5 published by Dryad for Currier2020README.rtf.
    """

    attempts: list[dict[str, Any]] = []
    transport = "dryad_primary"
    requested_url = primary_url
    try:
        raw, headers = _bounded_get(primary_url, max_bytes=max_bytes)
        attempts.append({"transport": transport, "url": primary_url, "result": "success"})
    except HTTPError as exc:
        attempts.append(
            {
                "transport": transport,
                "url": primary_url,
                "result": "http_error",
                "http_status": int(exc.code),
            }
        )
        if exc.code not in {401, 403} or mirror_url is None:
            raise
        transport = "zenodo_dryad_archive_mirror"
        requested_url = mirror_url
        raw, headers = _bounded_get(mirror_url, max_bytes=max_bytes)
        attempts.append({"transport": transport, "url": mirror_url, "result": "success"})

    observed_md5 = md5_bytes(raw)
    if observed_md5 != PFN_README_EXPECTED_MD5:
        raise ValueError(
            "PFN README bytes do not match Dryad's published MD5; refusing receipt"
        )

    output = Path(output_dir)
    receipt = {
        "schema": "fly-sniff-pfn-readme-fetch-v2",
        "navigation_performance_used": False,
        "scientific_authority": {
            "repository": "Dryad",
            "doi": PFN_DRYAD_DOI,
            "file_stream_id": PFN_README_FILE_STREAM_ID,
            "filename": PFN_README_FILENAME,
            "published_md5": PFN_README_EXPECTED_MD5,
        },
        "transport_policy": (
            "Dryad is the scientific/checksum authority. The Zenodo Dryad archive copy is "
            "transport-only and is eligible solely after Dryad returns HTTP 401/403."
        ),
        "retrieval_transport": transport,
        "requested_url": requested_url,
        "primary_url": primary_url,
        "mirror_record": PFN_README_MIRROR_RECORD,
        "mirror_url": mirror_url,
        "attempts": attempts,
        "final_url": headers["final_url"],
        "filename": PFN_README_FILENAME,
        "byte_count": len(raw),
        "max_bytes": max_bytes,
        "observed_md5": observed_md5,
        "observed_sha256": sha256_bytes(raw),
        "md5_matches_dryad_authority": True,
        "http": {key: value for key, value in headers.items() if key != "final_url"},
    }
    _atomic_write_bytes(output / PFN_README_FILENAME, raw)
    _atomic_write_json(output / "pfn-readme-fetch-receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and hash Currier2020README.rtf with checksum-gated Dryad mirror fallback"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--primary-url", default=PFN_README_PRIMARY_URL)
    parser.add_argument("--mirror-url", default=PFN_README_MIRROR_URL)
    parser.add_argument("--max-bytes", type=int, default=PFN_README_MAX_BYTES)
    parser.add_argument(
        "--no-mirror-fallback",
        action="store_true",
        help="Fail on Dryad 401/403 instead of trying the checksum-matched Zenodo archive copy",
    )
    args = parser.parse_args(argv)
    result = fetch_pfn_readme(
        args.out,
        primary_url=args.primary_url,
        mirror_url=None if args.no_mirror_fallback else args.mirror_url,
        max_bytes=args.max_bytes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
