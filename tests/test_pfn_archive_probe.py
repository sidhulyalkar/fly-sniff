from __future__ import annotations

import io
import json
import struct
import zipfile

import pytest

import fly_sniff.pfn_archive_probe as probe


def _ordinary_zip() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Data and Code/Ephys/pfn_left.mat", b"left")
        archive.writestr("Data and Code/Ephys/pfn_right.mat", b"right")
        archive.writestr("Data and Code/Behavior/readme.txt", b"other")
    return stream.getvalue()


def _zip64_suffix() -> bytes:
    name = b"Data and Code/Ephys/pfn_zip64.mat"
    central = struct.pack(
        "<4s6H3L5H2L",
        probe.CENTRAL_FILE_SIG,
        45,
        45,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        len(name),
        0,
        0,
        0,
        0,
        0,
        0xFFFFFFFF,
    ) + name
    zip64 = struct.pack(
        "<4sQ2H2L4Q",
        probe.ZIP64_EOCD_SIG,
        44,
        45,
        45,
        0,
        0,
        1,
        1,
        len(central),
        0x1_0000_1234,
    )
    locator = struct.pack(
        "<4sLQL",
        probe.ZIP64_LOCATOR_SIG,
        0,
        0x1_0000_1234 + len(central),
        1,
    )
    eocd = struct.pack(
        "<4s4H2LH",
        probe.EOCD_SIG,
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    return b"prefix-not-part-of-central-directory" + central + zip64 + locator + eocd


def test_standard_zip_tail_recovers_only_directory_metadata() -> None:
    payload = _ordinary_zip()
    footer = probe._find_footer(payload)
    assert footer["format"] == "zip"
    assert probe.required_suffix_bytes(payload_length=len(payload), footer=footer) <= len(payload)
    entries = probe.parse_central_directory(payload, footer)
    paths = [entry["path"] for entry in entries]
    assert paths == [
        "Data and Code/Ephys/pfn_left.mat",
        "Data and Code/Ephys/pfn_right.mat",
        "Data and Code/Behavior/readme.txt",
    ]
    assert [path for path in paths if probe._is_ephys_path(path)] == paths[:2]
    assert all("contents" not in entry for entry in entries)


def test_zip64_footer_is_parsed_without_trusting_global_offsets() -> None:
    payload = _zip64_suffix()
    footer = probe._find_footer(payload)
    assert footer["format"] == "zip64"
    assert footer["entries_total"] == 1
    assert footer["central_directory_size"] > 0
    assert footer["central_directory_offset"] > 2**32
    entries = probe.parse_central_directory(payload, footer)
    assert [entry["path"] for entry in entries] == ["Data and Code/Ephys/pfn_zip64.mat"]


def test_range_fetch_refuses_server_that_ignores_range(monkeypatch) -> None:
    def fake_get(url: str, *, max_bytes: int, headers=None):
        return b"x" * 10, {
            "final_url": url,
            "content_length": "10",
            "content_range": "",
            "accept_ranges": "",
            "content_type": "application/octet-stream",
            "etag": "",
        }, 200

    monkeypatch.setattr(probe, "_bounded_get", fake_get)
    with pytest.raises(ValueError, match="ignored bounded Range request"):
        probe.fetch_suffix_range("https://example.test/archive", suffix_bytes=10, max_bytes=10)


def test_range_fetch_requires_exact_content_range(monkeypatch) -> None:
    def fake_get(url: str, *, max_bytes: int, headers=None):
        return b"abcd", {
            "final_url": url,
            "content_length": "4",
            "content_range": "bytes 96-99/100",
            "accept_ranges": "bytes",
            "content_type": "application/octet-stream",
            "etag": "",
        }, 206

    monkeypatch.setattr(probe, "_bounded_get", fake_get)
    data, receipt = probe.fetch_suffix_range(
        "https://example.test/archive", suffix_bytes=4, max_bytes=4
    )
    assert data == b"abcd"
    assert receipt["range_start"] == 96
    assert receipt["range_end"] == 99
    assert receipt["object_size_bytes"] == 100


def test_transport_metadata_must_reproduce_dryad_md5(monkeypatch) -> None:
    body = json.dumps(
        {
            "files": [
                {
                    "key": probe.TARGET_FILENAME,
                    "size": 1_067_714_867,
                    "checksum": f"md5:{probe.TARGET_DRYAD_MD5}",
                    "links": {"content": probe.DEFAULT_TRANSPORT_URL},
                }
            ]
        }
    ).encode()

    def fake_get(url: str, *, max_bytes: int, headers=None):
        return body, {
            "final_url": url,
            "content_length": str(len(body)),
            "content_range": "",
            "accept_ranges": "",
            "content_type": "application/json",
            "etag": "",
        }, 200

    monkeypatch.setattr(probe, "_bounded_get", fake_get)
    result = probe.fetch_transport_metadata()
    assert result["size_bytes"] == 1_067_714_867
    assert result["checksum"] == f"md5:{probe.TARGET_DRYAD_MD5}"


def test_transport_metadata_checksum_drift_blocks(monkeypatch) -> None:
    body = json.dumps(
        {
            "files": [
                {
                    "key": probe.TARGET_FILENAME,
                    "size": 1_067_714_867,
                    "checksum": "md5:" + "0" * 32,
                }
            ]
        }
    ).encode()

    def fake_get(url: str, *, max_bytes: int, headers=None):
        return body, {
            "final_url": url,
            "content_length": str(len(body)),
            "content_range": "",
            "accept_ranges": "",
            "content_type": "application/json",
            "etag": "",
        }, 200

    monkeypatch.setattr(probe, "_bounded_get", fake_get)
    with pytest.raises(ValueError, match="no longer matches Dryad"):
        probe.fetch_transport_metadata()
