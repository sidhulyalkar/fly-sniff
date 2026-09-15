from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .freeze import canonical_sha256

ZENODO_RECORD_API = "https://zenodo.org/api/records/4407395"
ZENODO_RECORD = "https://zenodo.org/records/4407395"
TARGET_FILENAME = "Currier2020.zip.013"
TARGET_DRYAD_MD5 = "f2c15c70aec25f00d63282f87aee40d0"
DEFAULT_TRANSPORT_URL = (
    "https://zenodo.org/records/4407395/files/Currier2020.zip.013?download=1"
)
DRYAD_DOI = "10.5061/dryad.vq83bk3rh"
INITIAL_TAIL_BYTES = 128 * 1024
MAX_TAIL_BYTES = 16 * 1024 * 1024
METADATA_MAX_BYTES = 2 * 1024 * 1024
_USER_AGENT = "fly-sniff-pfn-archive-probe/1.0 (bounded provenance inspection)"

EOCD_SIG = b"PK\x05\x06"
ZIP64_EOCD_SIG = b"PK\x06\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
CENTRAL_FILE_SIG = b"PK\x01\x02"

_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def _bounded_get(url: str, *, max_bytes: int, headers: dict[str, str] | None = None) -> tuple[bytes, dict[str, str], int]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    request_headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"}
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=60.0) as response:
        status = int(getattr(response, "status", response.getcode()))
        final_url = str(response.geturl())
        if not final_url.startswith("https://"):
            raise ValueError("PFN archive transport redirected to a non-HTTPS URL")
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError(f"response exceeded frozen byte cap of {max_bytes}")
        response_headers = {
            "final_url": final_url,
            "content_length": str(response.headers.get("Content-Length", "")),
            "content_range": str(response.headers.get("Content-Range", "")),
            "accept_ranges": str(response.headers.get("Accept-Ranges", "")),
            "content_type": str(response.headers.get("Content-Type", "")),
            "etag": str(response.headers.get("ETag", "")),
        }
    return payload, response_headers, status


def fetch_transport_metadata(
    *,
    url: str = ZENODO_RECORD_API,
    max_bytes: int = METADATA_MAX_BYTES,
) -> dict[str, Any]:
    raw, headers, status = _bounded_get(url, max_bytes=max_bytes, headers={"Accept": "application/json"})
    if status != 200:
        raise ValueError(f"Zenodo record metadata returned HTTP {status}, expected 200")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Zenodo record metadata is not valid UTF-8 JSON") from exc
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise ValueError("Zenodo record metadata does not expose a files list")
    matches = [item for item in files if isinstance(item, dict) and item.get("key") == TARGET_FILENAME]
    if len(matches) != 1:
        raise ValueError(f"expected one Zenodo metadata entry for {TARGET_FILENAME}, found {len(matches)}")
    item = matches[0]
    checksum = str(item.get("checksum", ""))
    if checksum != f"md5:{TARGET_DRYAD_MD5}":
        raise ValueError("Zenodo transport metadata no longer matches Dryad's published MD5")
    size = int(item.get("size", 0))
    if size <= INITIAL_TAIL_BYTES:
        raise ValueError("final archive part size is implausibly small")
    links = item.get("links")
    if not isinstance(links, dict):
        links = {}
    content_url = str(links.get("content") or DEFAULT_TRANSPORT_URL)
    if not content_url.startswith("https://"):
        raise ValueError("Zenodo content link is not HTTPS")
    return {
        "record_url": ZENODO_RECORD,
        "record_api_url": url,
        "filename": TARGET_FILENAME,
        "size_bytes": size,
        "checksum": checksum,
        "transport_url": content_url,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "http": headers,
    }


def fetch_suffix_range(
    url: str,
    *,
    suffix_bytes: int,
    max_bytes: int = MAX_TAIL_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    if suffix_bytes <= 0 or suffix_bytes > max_bytes:
        raise ValueError(f"suffix_bytes must be in 1..{max_bytes}")
    payload, headers, status = _bounded_get(
        url,
        max_bytes=suffix_bytes,
        headers={"Accept": "application/octet-stream", "Range": f"bytes=-{suffix_bytes}"},
    )
    if status != 206:
        raise ValueError(
            f"archive transport ignored bounded Range request: HTTP {status}; refusing full-object transfer"
        )
    match = _CONTENT_RANGE_RE.fullmatch(headers["content_range"])
    if match is None:
        raise ValueError("HTTP 206 response lacks a valid Content-Range header")
    start, end, total = (int(value) for value in match.groups())
    if start < 0 or end < start or total <= end:
        raise ValueError("invalid Content-Range bounds")
    expected = end - start + 1
    if expected != len(payload):
        raise ValueError("Content-Range length does not match received suffix bytes")
    if expected > suffix_bytes:
        raise ValueError("server returned more bytes than the requested suffix cap")
    return payload, {
        "requested_suffix_bytes": suffix_bytes,
        "received_bytes": len(payload),
        "range_start": start,
        "range_end": end,
        "object_size_bytes": total,
        "suffix_sha256": hashlib.sha256(payload).hexdigest(),
        "http": headers,
    }


def _find_footer(payload: bytes) -> dict[str, Any]:
    eocd_pos = payload.rfind(EOCD_SIG)
    if eocd_pos < 0 or eocd_pos + 22 > len(payload):
        raise ValueError("ZIP end-of-central-directory record not found in bounded suffix")
    (
        signature,
        disk_number,
        cd_disk_number,
        entries_on_disk,
        entries_total,
        cd_size32,
        cd_offset32,
        comment_length,
    ) = struct.unpack_from("<4s4H2LH", payload, eocd_pos)
    if signature != EOCD_SIG:
        raise ValueError("invalid ZIP EOCD signature")
    if eocd_pos + 22 + comment_length > len(payload):
        raise ValueError("ZIP EOCD comment extends outside the bounded suffix")

    uses_zip64 = (
        entries_on_disk == 0xFFFF
        or entries_total == 0xFFFF
        or cd_size32 == 0xFFFFFFFF
        or cd_offset32 == 0xFFFFFFFF
    )
    footer: dict[str, Any] = {
        "format": "zip64" if uses_zip64 else "zip",
        "eocd_offset_in_suffix": eocd_pos,
        "disk_number": disk_number,
        "central_directory_disk_number": cd_disk_number,
        "entries_on_disk_32": entries_on_disk,
        "entries_total_32": entries_total,
        "central_directory_size_32": cd_size32,
        "central_directory_offset_32": cd_offset32,
        "comment_length": comment_length,
    }
    if not uses_zip64:
        footer.update(
            {
                "entries_total": entries_total,
                "central_directory_size": cd_size32,
                "central_directory_anchor_offset_in_suffix": eocd_pos,
            }
        )
        return footer

    locator_pos = payload.rfind(ZIP64_LOCATOR_SIG, 0, eocd_pos)
    if locator_pos < 0 or locator_pos + 20 > eocd_pos:
        raise ValueError("ZIP64 EOCD locator not found before EOCD")
    _, zip64_disk, zip64_offset, total_disks = struct.unpack_from("<4sLQL", payload, locator_pos)

    zip64_pos = payload.rfind(ZIP64_EOCD_SIG, 0, locator_pos)
    if zip64_pos < 0 or zip64_pos + 56 > locator_pos:
        raise ValueError("ZIP64 EOCD record not found in bounded suffix")
    (
        zip64_signature,
        record_size,
        version_made,
        version_needed,
        disk_number_64,
        cd_disk_number_64,
        entries_on_disk_64,
        entries_total_64,
        cd_size64,
        cd_offset64,
    ) = struct.unpack_from("<4sQ2H2L4Q", payload, zip64_pos)
    if zip64_signature != ZIP64_EOCD_SIG or record_size < 44:
        raise ValueError("invalid ZIP64 EOCD record")
    footer.update(
        {
            "zip64_locator_offset_in_suffix": locator_pos,
            "zip64_eocd_offset_in_suffix": zip64_pos,
            "zip64_eocd_advertised_global_offset": zip64_offset,
            "zip64_disk": zip64_disk,
            "zip64_total_disks": total_disks,
            "zip64_version_made": version_made,
            "zip64_version_needed": version_needed,
            "zip64_disk_number": disk_number_64,
            "zip64_central_directory_disk_number": cd_disk_number_64,
            "entries_on_disk": entries_on_disk_64,
            "entries_total": entries_total_64,
            "central_directory_size": cd_size64,
            "central_directory_offset": cd_offset64,
            "central_directory_anchor_offset_in_suffix": zip64_pos,
        }
    )
    return footer


def required_suffix_bytes(*, payload_length: int, footer: dict[str, Any]) -> int:
    anchor = int(footer["central_directory_anchor_offset_in_suffix"])
    cd_size = int(footer["central_directory_size"])
    if anchor < 0 or anchor > payload_length or cd_size <= 0:
        raise ValueError("invalid central-directory geometry")
    return cd_size + (payload_length - anchor)


def parse_central_directory(payload: bytes, footer: dict[str, Any]) -> list[dict[str, Any]]:
    anchor = int(footer["central_directory_anchor_offset_in_suffix"])
    cd_size = int(footer["central_directory_size"])
    start = anchor - cd_size
    if start < 0:
        raise ValueError("bounded suffix does not contain the complete central directory")
    cursor = start
    entries: list[dict[str, Any]] = []
    while cursor < anchor:
        if cursor + 46 > anchor:
            raise ValueError("truncated central-directory file header")
        values = struct.unpack_from("<4s6H3L5H2L", payload, cursor)
        if values[0] != CENTRAL_FILE_SIG:
            raise ValueError(f"unexpected central-directory signature at relative offset {cursor}")
        flags = int(values[3])
        compressed_size = int(values[8])
        uncompressed_size = int(values[9])
        filename_length = int(values[10])
        extra_length = int(values[11])
        comment_length = int(values[12])
        disk_start = int(values[13])
        local_header_offset = int(values[16])
        end = cursor + 46 + filename_length + extra_length + comment_length
        if end > anchor:
            raise ValueError("central-directory entry extends beyond declared directory size")
        raw_name = payload[cursor + 46 : cursor + 46 + filename_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        filename = raw_name.decode(encoding, errors="replace")
        entries.append(
            {
                "path": filename,
                "compressed_size_32": compressed_size,
                "uncompressed_size_32": uncompressed_size,
                "disk_start_32": disk_start,
                "local_header_offset_32": local_header_offset,
            }
        )
        cursor = end
    expected_entries = int(footer.get("entries_total", len(entries)))
    if expected_entries != len(entries):
        raise ValueError(
            f"central-directory entry count mismatch: footer={expected_entries}, parsed={len(entries)}"
        )
    return entries


def _is_ephys_path(path: str) -> bool:
    normalized = "/" + path.replace("\\", "/").strip("/").lower() + "/"
    return "/ephys/" in normalized


def probe_archive_directory(
    output_path: str | Path,
    *,
    metadata_url: str = ZENODO_RECORD_API,
    initial_tail_bytes: int = INITIAL_TAIL_BYTES,
    max_tail_bytes: int = MAX_TAIL_BYTES,
) -> dict[str, Any]:
    metadata = fetch_transport_metadata(url=metadata_url)
    first, first_receipt = fetch_suffix_range(
        metadata["transport_url"],
        suffix_bytes=initial_tail_bytes,
        max_bytes=max_tail_bytes,
    )
    if first_receipt["object_size_bytes"] != metadata["size_bytes"]:
        raise ValueError("range response object size differs from Zenodo record metadata")
    footer = _find_footer(first)
    required = required_suffix_bytes(payload_length=len(first), footer=footer)
    if required > max_tail_bytes:
        result: dict[str, Any] = {
            "schema": "fly-sniff-pfn-archive-directory-probe-v1",
            "status": "BLOCKED_CENTRAL_DIRECTORY_EXCEEDS_BYTE_CAP",
            "scientific_authority": {"repository": "Dryad", "doi": DRYAD_DOI},
            "transport": metadata,
            "initial_probe": first_receipt,
            "footer": footer,
            "required_suffix_bytes": required,
            "max_tail_bytes": max_tail_bytes,
            "navigation_performance_used": False,
            "neural_member_bytes_read": False,
            "archive_member_map_promoted": False,
            "figure4_recording_set_promoted": False,
        }
        result["probe_sha256"] = canonical_sha256(result)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    if required <= len(first):
        full_tail, full_receipt = first, first_receipt
        footer_full = footer
    else:
        full_tail, full_receipt = fetch_suffix_range(
            metadata["transport_url"],
            suffix_bytes=required,
            max_bytes=max_tail_bytes,
        )
        footer_full = _find_footer(full_tail)
        if required_suffix_bytes(payload_length=len(full_tail), footer=footer_full) > len(full_tail):
            raise ValueError("second bounded suffix still does not contain the declared central directory")

    entries = parse_central_directory(full_tail, footer_full)
    ephys_entries = [entry for entry in entries if _is_ephys_path(entry["path"])]
    result = {
        "schema": "fly-sniff-pfn-archive-directory-probe-v1",
        "status": "CENTRAL_DIRECTORY_RECOVERED_PENDING_RECORDING_ADJUDICATION",
        "scientific_authority": {
            "repository": "Dryad",
            "doi": DRYAD_DOI,
            "published_target_file_md5": TARGET_DRYAD_MD5,
        },
        "transport": metadata,
        "initial_probe": first_receipt,
        "directory_probe": full_receipt,
        "footer": footer_full,
        "required_suffix_bytes": required,
        "max_tail_bytes": max_tail_bytes,
        "archive_entry_count": len(entries),
        "ephys_entry_count": len(ephys_entries),
        "ephys_entries": ephys_entries,
        "navigation_performance_used": False,
        "neural_member_bytes_read": False,
        "archive_member_map_promoted": False,
        "figure4_recording_set_promoted": False,
        "boundary": (
            "This receipt exposes only ZIP directory metadata transported from the checksum-matched "
            "Zenodo copy. It does not authenticate individual member bytes, identify the published "
            "Figure 4 cohort by itself, compute physiology, or clear the PFN source blockers."
        ),
    }
    result["probe_sha256"] = canonical_sha256(result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, mode="w", encoding="utf-8", delete=False) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover PFN archive directory metadata using bounded suffix-range requests only"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-url", default=ZENODO_RECORD_API)
    parser.add_argument("--initial-tail-bytes", type=int, default=INITIAL_TAIL_BYTES)
    parser.add_argument("--max-tail-bytes", type=int, default=MAX_TAIL_BYTES)
    args = parser.parse_args(argv)
    result = probe_archive_directory(
        args.out,
        metadata_url=args.metadata_url,
        initial_tail_bytes=args.initial_tail_bytes,
        max_tail_bytes=args.max_tail_bytes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
