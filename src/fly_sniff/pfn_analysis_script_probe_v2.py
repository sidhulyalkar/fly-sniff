from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import pfn_analysis_script_probe as v1
from .pfn_source import load_contract

_RECORD_ID = "4407395"
_ARCHIVE_PART_RE = re.compile(r"^Currier2020\.zip\.\d{3}$")


def _fallback_content_url(name: str) -> str:
    return f"https://zenodo.org/records/{_RECORD_ID}/files/{quote(name)}?download=1"


def archive_part_inventory(
    contract_path: str | Path = v1.DEFAULT_SOURCE_CONTRACT,
) -> list[dict[str, Any]]:
    contract = load_contract(contract_path)
    published = list(contract.archive_inventory)
    payload = v1._record_payload()
    files = payload["files"]
    by_name = {
        str(item["key"]): item
        for item in files
        if isinstance(item, dict)
        and (
            str(item.get("key")) == "Currier2020.z01"
            or _ARCHIVE_PART_RE.match(str(item.get("key")))
        )
    }
    expected_names = {name for name, _ in published}
    if set(by_name) != expected_names:
        missing = sorted(expected_names - set(by_name))
        extra = sorted(set(by_name) - expected_names)
        raise ValueError(
            "Zenodo archive-part inventory differs from Dryad authority: "
            f"missing={missing}, extra={extra}"
        )

    inventory: list[dict[str, Any]] = []
    logical_start = 0
    for name, published_md5 in published:
        item = by_name[name]
        checksum = str(item.get("checksum", ""))
        if checksum != f"md5:{published_md5}":
            raise ValueError(f"transport checksum for {name} does not match Dryad authority")
        size = int(item.get("size", 0))
        if size <= 0:
            raise ValueError(f"invalid transport size for {name}")
        links = item.get("links")
        content_url = ""
        if isinstance(links, dict):
            content_url = str(links.get("content", ""))
        if not content_url.startswith("https://"):
            content_url = _fallback_content_url(name)
        inventory.append(
            {
                "filename": name,
                "size_bytes": size,
                "published_md5": published_md5,
                "transport_url": content_url,
                "logical_start": logical_start,
                "logical_end": logical_start + size - 1,
            }
        )
        logical_start += size
    return inventory


def probe_analysis_scripts(output_path: str | Path) -> dict[str, Any]:
    original = v1.archive_part_inventory
    try:
        v1.archive_part_inventory = archive_part_inventory
        return v1.probe_analysis_scripts(output_path)
    finally:
        v1.archive_part_inventory = original


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Range-fetch only the two allowlisted PFN analysis scripts using checksum-gated "
            "Zenodo content-URL fallback when legacy metadata omits links.content"
        )
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = probe_analysis_scripts(args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
