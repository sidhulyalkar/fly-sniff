from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .physiology_source_fetch import (
    _atomic_write_bytes,
    _atomic_write_json,
    _bounded_get,
    md5_bytes,
    sha256_bytes,
)

DNA02_README_FILE_ID = 11634832
DNA02_README_URL = f"https://dataverse.harvard.edu/api/access/datafile/{DNA02_README_FILE_ID}"
DNA02_README_FILENAME = "README.md"
DNA02_README_EXPECTED_MD5 = "cfb0cb853be3a454b965d74eb5c772c0"
DNA02_README_MAX_BYTES = 65_536


def fetch_dna02_readme(
    output_dir: str | Path,
    *,
    url: str = DNA02_README_URL,
    max_bytes: int = DNA02_README_MAX_BYTES,
) -> dict[str, Any]:
    raw, headers = _bounded_get(url, max_bytes=max_bytes)
    observed_md5 = md5_bytes(raw)
    if observed_md5 != DNA02_README_EXPECTED_MD5:
        raise ValueError(
            "DNa02 Dataverse README bytes do not match repository MD5; refusing receipt"
        )
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("DNa02 Dataverse README is not valid UTF-8") from exc

    output = Path(output_dir)
    receipt = {
        "schema": "fly-sniff-dna02-dataverse-readme-fetch-v1",
        "navigation_performance_used": False,
        "file_id": DNA02_README_FILE_ID,
        "filename": DNA02_README_FILENAME,
        "requested_url": url,
        "final_url": headers["final_url"],
        "byte_count": len(raw),
        "max_bytes": max_bytes,
        "published_md5": DNA02_README_EXPECTED_MD5,
        "observed_md5": observed_md5,
        "observed_sha256": sha256_bytes(raw),
        "http": {key: value for key, value in headers.items() if key != "final_url"},
    }
    _atomic_write_bytes(output / DNA02_README_FILENAME, raw)
    _atomic_write_json(output / "dna02-readme-fetch-receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and hash the tiny DNa02 Dataverse README")
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default=DNA02_README_URL)
    parser.add_argument("--max-bytes", type=int, default=DNA02_README_MAX_BYTES)
    args = parser.parse_args(argv)
    result = fetch_dna02_readme(args.out, url=args.url, max_bytes=args.max_bytes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
