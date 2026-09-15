from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import fly_sniff.dna02_source_bytes as source_bytes

ADJUDICATION = Path("authority/program-a-dna02-cohort-adjudication-v1.json")


class _Headers(dict):
    def get(self, key: str, default=None):
        return super().get(key, default)


class _ChunkedResponse:
    def __init__(self, payload: bytes, *, url: str = "https://example.test/file"):
        self.payload = payload
        self.offset = 0
        self.url = url
        self.headers = _Headers({"Content-Length": str(len(payload))})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def geturl(self) -> str:
        return self.url

    def read(self, count: int) -> bytes:
        if self.offset >= len(self.payload):
            return b""
        chunk = self.payload[self.offset : self.offset + count]
        self.offset += len(chunk)
        return chunk


def _spec(payload: bytes, *, alias: str = "a2_d_08") -> dict:
    return {
        "fly_alias": alias,
        "file_id": 123,
        "filename": f"{alias}.mat",
        "expected_size_bytes": len(payload),
        "expected_md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    }


def test_real_adjudication_yields_exact_four_frozen_source_specs() -> None:
    adjudication_sha, specs = source_bytes.source_specs(ADJUDICATION)
    assert adjudication_sha == "a7262fe9e82c1898a4ac6dc753aa38a854e05de558d64e466ffdde884dfdde53"
    assert [spec["fly_alias"] for spec in specs] == ["a2_d_08", "a2_d_12", "a2_d_13", "a2_d_14"]
    assert [spec["file_id"] for spec in specs] == [11634638, 11634639, 11634640, 11634641]
    assert sum(spec["expected_size_bytes"] for spec in specs) == 929441392


def test_stream_hash_verifies_size_md5_and_sha256(monkeypatch) -> None:
    raw = b"0123456789abcdef"
    spec = _spec(raw)
    response = _ChunkedResponse(raw)
    monkeypatch.setattr(source_bytes, "urlopen", lambda request, timeout: response)

    receipt = source_bytes.stream_hash_dataverse_file(spec, chunk_bytes=3)

    assert receipt["byte_count"] == len(raw)
    assert receipt["observed_md5"] == spec["expected_md5"]
    assert receipt["observed_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["raw_bytes_retained"] is False
    assert receipt["navigation_performance_used"] is False


def test_stream_hash_rejects_md5_mismatch(monkeypatch) -> None:
    raw = b"source bytes"
    spec = _spec(raw)
    spec["expected_md5"] = "0" * 32
    response = _ChunkedResponse(raw)
    monkeypatch.setattr(source_bytes, "urlopen", lambda request, timeout: response)

    with pytest.raises(ValueError, match="MD5 mismatch"):
        source_bytes.stream_hash_dataverse_file(spec)


def test_stream_hash_rejects_content_length_change(monkeypatch) -> None:
    raw = b"source bytes"
    spec = _spec(raw)
    response = _ChunkedResponse(raw)
    response.headers["Content-Length"] = str(len(raw) + 1)
    monkeypatch.setattr(source_bytes, "urlopen", lambda request, timeout: response)

    with pytest.raises(ValueError, match="Content-Length changed"):
        source_bytes.stream_hash_dataverse_file(spec)


def test_stream_hash_rejects_non_https_redirect(monkeypatch) -> None:
    raw = b"source bytes"
    spec = _spec(raw)
    response = _ChunkedResponse(raw, url="http://example.test/file")
    monkeypatch.setattr(source_bytes, "urlopen", lambda request, timeout: response)

    with pytest.raises(ValueError, match="non-HTTPS"):
        source_bytes.stream_hash_dataverse_file(spec)


def test_manifest_is_published_only_after_all_four_files_pass(tmp_path, monkeypatch) -> None:
    adjudication_sha, specs = source_bytes.source_specs(ADJUDICATION)
    receipts = []
    for spec in specs:
        receipts.append(
            {
                "fly_alias": spec["fly_alias"],
                "file_id": spec["file_id"],
                "filename": spec["filename"],
                "source_url": "https://example.test/source",
                "final_url": "https://example.test/source",
                "byte_count": spec["expected_size_bytes"],
                "published_md5": spec["expected_md5"],
                "observed_md5": spec["expected_md5"],
                "observed_sha256": hashlib.sha256(spec["fly_alias"].encode()).hexdigest(),
                "md5_matches_dataverse_authority": True,
                "raw_bytes_retained": False,
                "navigation_performance_used": False,
            }
        )
    receipt_iter = iter(receipts)
    monkeypatch.setattr(source_bytes, "stream_hash_dataverse_file", lambda spec, timeout_s: next(receipt_iter))

    out = tmp_path / "manifest.json"
    manifest = source_bytes.hash_adjudicated_sources(out, adjudication_path=ADJUDICATION)

    assert manifest["adjudication_sha256"] == adjudication_sha
    assert len(manifest["files"]) == 4
    assert manifest["raw_bytes_retained"] is False
    assert manifest["navigation_performance_used"] is False
    assert out.exists()


def test_failed_file_prevents_aggregate_manifest_publication(tmp_path, monkeypatch) -> None:
    calls = 0

    def fail_second(spec, *, timeout_s):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic source failure")
        return {"fly_alias": spec["fly_alias"]}

    monkeypatch.setattr(source_bytes, "stream_hash_dataverse_file", fail_second)
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="synthetic source failure"):
        source_bytes.hash_adjudicated_sources(out, adjudication_path=ADJUDICATION)
    assert not out.exists()


def test_source_hashing_never_accepts_navigation_inputs() -> None:
    assert "navigation" not in source_bytes.stream_hash_dataverse_file.__code__.co_varnames
    assert "navigation" not in source_bytes.hash_adjudicated_sources.__code__.co_varnames
