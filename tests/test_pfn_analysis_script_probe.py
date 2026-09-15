from __future__ import annotations

import binascii
import struct
import zlib

import pytest

from fly_sniff.pfn_analysis_script_probe import (
    MAX_TOTAL_MEMBER_BYTES,
    TARGET_SCRIPTS,
    _zip64_values,
    fetch_script_member,
)


def test_target_allowlist_contains_only_two_analysis_scripts() -> None:
    assert TARGET_SCRIPTS == (
        "Ephys/analyzeWindTuning.m",
        "Ephys/windTuningPFN.m",
    )
    assert MAX_TOTAL_MEMBER_BYTES <= 64 * 1024


def test_zip64_extra_resolves_only_sentinel_local_offset() -> None:
    logical_offset = 117_000_000_123
    extra = struct.pack("<HHQ", 0x0001, 8, logical_offset)
    values = _zip64_values(
        uncompressed32=4025,
        compressed32=1557,
        local_offset32=0xFFFFFFFF,
        disk_start32=0,
        extra=extra,
    )
    assert values == (4025, 1557, logical_offset, 0)


def test_zip64_sentinel_without_extra_blocks() -> None:
    with pytest.raises(ValueError, match="ZIP64 sentinel"):
        _zip64_values(
            uncompressed32=4025,
            compressed32=1557,
            local_offset32=0xFFFFFFFF,
            disk_start32=0,
            extra=b"",
        )


def test_script_member_verifies_local_header_decompression_and_crc(monkeypatch) -> None:
    raw = b"function y = tiny(x)\ny = x + 1;\nend\n"
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    name = b"Ephys/analyzeWindTuning.m"
    local_header = struct.pack(
        "<4s5H3L2H",
        b"PK\x03\x04",
        20,
        0,
        8,
        0,
        0,
        binascii.crc32(raw) & 0xFFFFFFFF,
        len(compressed),
        len(raw),
        len(name),
        0,
    )
    logical = local_header + name + compressed

    def fake_fetch(parts, *, logical_start: int, length: int):
        return logical[logical_start : logical_start + length], [
            {"filename": "synthetic", "range_start": logical_start, "range_end": logical_start + length - 1}
        ]

    monkeypatch.setattr("fly_sniff.pfn_analysis_script_probe.fetch_logical_range", fake_fetch)
    entry = {
        "path": "Ephys/analyzeWindTuning.m",
        "flags": 0,
        "compression_method": 8,
        "crc32": binascii.crc32(raw) & 0xFFFFFFFF,
        "compressed_size": len(compressed),
        "uncompressed_size": len(raw),
        "local_header_offset": 0,
        "disk_start": 0,
    }
    result = fetch_script_member(entry, parts=[])
    assert result["content_utf8"].startswith("function y")
    assert result["observed_crc32"] == entry["crc32"]


def test_script_member_rejects_encryption() -> None:
    entry = {
        "path": "Ephys/analyzeWindTuning.m",
        "flags": 1,
        "compression_method": 8,
        "crc32": 0,
        "compressed_size": 1,
        "uncompressed_size": 1,
        "local_header_offset": 0,
        "disk_start": 0,
    }
    with pytest.raises(ValueError, match="encrypted"):
        fetch_script_member(entry, parts=[])
