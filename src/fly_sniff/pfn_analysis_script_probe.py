from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import re
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

from .freeze import canonical_sha256
from .pfn_archive_probe import (
    CENTRAL_FILE_SIG,
    MAX_TAIL_BYTES,
    ZENODO_RECORD_API,
    _bounded_get,
    _find_footer,
    fetch_suffix_range,
    fetch_transport_metadata,
    required_suffix_bytes,
)
from .pfn_source import load_contract

TARGET_SCRIPTS = (
    "Ephys/analyzeWindTuning.m",
    "Ephys/windTuningPFN.m",
)
MAX_SCRIPT_COMPRESSED_BYTES = 32 * 1024
MAX_SCRIPT_UNCOMPRESSED_BYTES = 128 * 1024
MAX_TOTAL_MEMBER_BYTES = 64 * 1024
DEFAULT_SOURCE_CONTRACT = Path("authority/program-a-pfn-source-contract-v1.json")
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_ARCHIVE_PART_RE = re.compile(r"^Currier2020\.zip\.\d{3}$")


def _record_payload() -> dict[str, Any]:
    raw, _, status = _bounded_get(
        ZENODO_RECORD_API,
        max_bytes=2 * 1024 * 1024,
        headers={"Accept": "application/json"},
    )
    if status != 200:
        raise ValueError(f"Zenodo metadata returned HTTP {status}, expected 200")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Zenodo metadata must be a JSON object")
    files = payload.get("files")
    if not isinstance(files, list):
        raise TypeError("Zenodo metadata files field must be a list")
    return payload


def archive_part_inventory(
    contract_path: str | Path = DEFAULT_SOURCE_CONTRACT,
) -> list[dict[str, Any]]:
    contract = load_contract(contract_path)
    published = list(contract.archive_inventory)
    payload = _record_payload()
    files = payload["files"]
    by_name = {
        str(item["key"]): item
        for item in files
        if isinstance(item, dict)
        and (str(item.get("key")) == "Currier2020.z01" or _ARCHIVE_PART_RE.match(str(item.get("key"))))
    }
    if set(by_name) != {name for name, _ in published}:
        missing = sorted({name for name, _ in published} - set(by_name))
        extra = sorted(set(by_name) - {name for name, _ in published})
        raise ValueError(f"Zenodo archive-part inventory differs from Dryad authority: missing={missing}, extra={extra}")

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
        if not isinstance(links, dict) or not str(links.get("content", "")).startswith("https://"):
            raise ValueError(f"missing HTTPS content transport for {name}")
        inventory.append(
            {
                "filename": name,
                "size_bytes": size,
                "published_md5": published_md5,
                "transport_url": str(links["content"]),
                "logical_start": logical_start,
                "logical_end": logical_start + size - 1,
            }
        )
        logical_start += size
    return inventory


def _zip64_values(
    *,
    uncompressed32: int,
    compressed32: int,
    local_offset32: int,
    disk_start32: int,
    extra: bytes,
) -> tuple[int, int, int, int]:
    zip64: bytes | None = None
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, field_len = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        if cursor + field_len > len(extra):
            raise ValueError("truncated ZIP extra field")
        field = extra[cursor : cursor + field_len]
        cursor += field_len
        if field_id == 0x0001:
            zip64 = field
    values = [uncompressed32, compressed32, local_offset32, disk_start32]
    sentinels = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF]
    widths = [8, 8, 8, 4]
    if any(value == sentinel for value, sentinel in zip(values, sentinels, strict=True)):
        if zip64 is None:
            raise ValueError("ZIP64 sentinel present without ZIP64 extended information")
        zcursor = 0
        for index, (value, sentinel, width) in enumerate(zip(values, sentinels, widths, strict=True)):
            if value != sentinel:
                continue
            if zcursor + width > len(zip64):
                raise ValueError("truncated ZIP64 extended information")
            fmt = "<Q" if width == 8 else "<L"
            values[index] = int(struct.unpack_from(fmt, zip64, zcursor)[0])
            zcursor += width
    return tuple(int(value) for value in values)


def target_central_entries(full_tail: bytes, footer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    anchor = int(footer["central_directory_anchor_offset_in_suffix"])
    start = anchor - int(footer["central_directory_size"])
    if start < 0:
        raise ValueError("complete central directory is not present")
    cursor = start
    targets: dict[str, dict[str, Any]] = {}
    while cursor < anchor:
        if cursor + 46 > anchor:
            raise ValueError("truncated central-directory header")
        values = struct.unpack_from("<4s6H3L5H2L", full_tail, cursor)
        if values[0] != CENTRAL_FILE_SIG:
            raise ValueError("invalid central-directory file signature")
        flags = int(values[3])
        compression = int(values[4])
        crc32 = int(values[7])
        compressed32 = int(values[8])
        uncompressed32 = int(values[9])
        filename_len = int(values[10])
        extra_len = int(values[11])
        comment_len = int(values[12])
        disk_start32 = int(values[13])
        local_offset32 = int(values[16])
        filename_start = cursor + 46
        extra_start = filename_start + filename_len
        end = extra_start + extra_len + comment_len
        if end > anchor:
            raise ValueError("central-directory entry extends past declared boundary")
        raw_name = full_tail[filename_start:extra_start]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = raw_name.decode(encoding, errors="strict")
        if name in TARGET_SCRIPTS:
            extra = full_tail[extra_start : extra_start + extra_len]
            uncompressed, compressed, local_offset, disk_start = _zip64_values(
                uncompressed32=uncompressed32,
                compressed32=compressed32,
                local_offset32=local_offset32,
                disk_start32=disk_start32,
                extra=extra,
            )
            targets[name] = {
                "path": name,
                "flags": flags,
                "compression_method": compression,
                "crc32": crc32,
                "compressed_size": compressed,
                "uncompressed_size": uncompressed,
                "local_header_offset": local_offset,
                "disk_start": disk_start,
            }
        cursor = end
    if set(targets) != set(TARGET_SCRIPTS):
        raise ValueError(f"central directory did not resolve both target scripts: {sorted(targets)}")
    return targets


def _fetch_part_range(part: dict[str, Any], *, start: int, length: int) -> tuple[bytes, dict[str, Any]]:
    if start < 0 or length <= 0 or start + length > int(part["size_bytes"]):
        raise ValueError("physical part range is out of bounds")
    end = start + length - 1
    data, headers, status = _bounded_get(
        str(part["transport_url"]),
        max_bytes=length,
        headers={"Accept": "application/octet-stream", "Range": f"bytes={start}-{end}"},
    )
    if status != 206:
        raise ValueError(f"archive part ignored bounded range request: HTTP {status}")
    match = _CONTENT_RANGE_RE.fullmatch(headers["content_range"])
    if match is None:
        raise ValueError("bounded archive-part response lacks valid Content-Range")
    observed_start, observed_end, total = (int(value) for value in match.groups())
    if (observed_start, observed_end, total) != (start, end, int(part["size_bytes"])):
        raise ValueError("archive-part Content-Range differs from requested frozen range")
    if len(data) != length:
        raise ValueError("archive-part response length differs from requested range")
    return data, {
        "filename": part["filename"],
        "range_start": start,
        "range_end": end,
        "received_bytes": len(data),
        "range_sha256": hashlib.sha256(data).hexdigest(),
    }


def fetch_logical_range(
    parts: list[dict[str, Any]],
    *,
    logical_start: int,
    length: int,
) -> tuple[bytes, list[dict[str, Any]]]:
    if logical_start < 0 or length <= 0 or length > MAX_TOTAL_MEMBER_BYTES:
        raise ValueError("logical member range exceeds frozen bounds")
    logical_end = logical_start + length - 1
    if logical_end >= sum(int(part["size_bytes"]) for part in parts):
        raise ValueError("logical range exceeds reconstructed archive size")
    chunks: list[bytes] = []
    receipts: list[dict[str, Any]] = []
    remaining_start = logical_start
    remaining = length
    for part in parts:
        part_start = int(part["logical_start"])
        part_end = int(part["logical_end"])
        if remaining_start > part_end or remaining_start + remaining - 1 < part_start:
            continue
        physical_start = max(remaining_start, part_start) - part_start
        available = int(part["size_bytes"]) - physical_start
        take = min(remaining, available)
        chunk, receipt = _fetch_part_range(part, start=physical_start, length=take)
        chunks.append(chunk)
        receipts.append(receipt)
        remaining_start += take
        remaining -= take
        if remaining == 0:
            break
    if remaining != 0:
        raise ValueError("failed to reconstruct complete bounded logical range")
    return b"".join(chunks), receipts


def fetch_script_member(
    entry: dict[str, Any],
    *,
    parts: list[dict[str, Any]],
) -> dict[str, Any]:
    if entry["disk_start"] != 0:
        raise ValueError("expected externally split single-disk ZIP metadata")
    if entry["flags"] & 0x1:
        raise ValueError("encrypted ZIP members are not allowed")
    if entry["compressed_size"] > MAX_SCRIPT_COMPRESSED_BYTES:
        raise ValueError("target script compressed size exceeds frozen cap")
    if entry["uncompressed_size"] > MAX_SCRIPT_UNCOMPRESSED_BYTES:
        raise ValueError("target script uncompressed size exceeds frozen cap")

    fixed, fixed_receipts = fetch_logical_range(
        parts,
        logical_start=int(entry["local_header_offset"]),
        length=30,
    )
    values = struct.unpack("<4s5H3L2H", fixed)
    if values[0] != b"PK\x03\x04":
        raise ValueError("target script local-header signature is invalid")
    local_flags = int(values[2])
    local_compression = int(values[3])
    filename_len = int(values[9])
    extra_len = int(values[10])
    if local_flags != entry["flags"] or local_compression != entry["compression_method"]:
        raise ValueError("local and central ZIP headers disagree")
    data_start = int(entry["local_header_offset"]) + 30 + filename_len + extra_len
    compressed, data_receipts = fetch_logical_range(
        parts,
        logical_start=data_start,
        length=int(entry["compressed_size"]),
    )
    if entry["compression_method"] == 0:
        raw = compressed
    elif entry["compression_method"] == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"unsupported ZIP compression method {entry['compression_method']}")
    if len(raw) != int(entry["uncompressed_size"]):
        raise ValueError("decompressed script size differs from central-directory authority")
    observed_crc32 = binascii.crc32(raw) & 0xFFFFFFFF
    if observed_crc32 != int(entry["crc32"]):
        raise ValueError("decompressed script CRC32 differs from central-directory authority")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return {
        **entry,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "observed_crc32": observed_crc32,
        "range_receipts": fixed_receipts + data_receipts,
        "content_utf8": text,
    }


def probe_analysis_scripts(output_path: str | Path) -> dict[str, Any]:
    transport = fetch_transport_metadata()
    first, _ = fetch_suffix_range(
        transport["transport_url"],
        suffix_bytes=128 * 1024,
        max_bytes=MAX_TAIL_BYTES,
    )
    footer = _find_footer(first)
    required = required_suffix_bytes(payload_length=len(first), footer=footer)
    if required > MAX_TAIL_BYTES:
        raise ValueError("central directory exceeds frozen script-probe tail cap")
    full_tail, tail_receipt = fetch_suffix_range(
        transport["transport_url"],
        suffix_bytes=required,
        max_bytes=MAX_TAIL_BYTES,
    )
    footer = _find_footer(full_tail)
    targets = target_central_entries(full_tail, footer)
    parts = archive_part_inventory()
    reconstructed_size = sum(int(part["size_bytes"]) for part in parts)
    expected_size = int(footer["central_directory_offset"]) + int(footer["central_directory_size"]) + 98
    if reconstructed_size != expected_size:
        raise ValueError(
            f"archive-part sizes do not reconstruct ZIP64 logical size: {reconstructed_size} != {expected_size}"
        )
    scripts = [fetch_script_member(targets[path], parts=parts) for path in TARGET_SCRIPTS]
    payload: dict[str, Any] = {
        "schema": "fly-sniff-pfn-analysis-script-probe-v1",
        "status": "ANALYSIS_SCRIPTS_FETCHED_PENDING_ADJUDICATION",
        "targets": list(TARGET_SCRIPTS),
        "archive_logical_size_bytes": reconstructed_size,
        "central_directory_probe": tail_receipt,
        "archive_parts": [
            {
                key: part[key]
                for key in ("filename", "size_bytes", "published_md5", "logical_start", "logical_end")
            }
            for part in parts
        ],
        "scripts": scripts,
        "navigation_performance_used": False,
        "neural_recording_member_bytes_read": False,
        "allowed_member_content": list(TARGET_SCRIPTS),
        "boundary": (
            "This transient evidence contains only two small published MATLAB analysis scripts. "
            "It does not read any neural recording member, compute physiology, or promote the PFN source contract."
        ),
    }
    payload["probe_sha256"] = canonical_sha256(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, mode="w", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(output)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Range-fetch only the two tiny PFN Figure 4 MATLAB analysis scripts"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = probe_analysis_scripts(args.out)
    printable = {**result, "scripts": [{key: value for key, value in item.items() if key != "content_utf8"} for item in result["scripts"]]}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
