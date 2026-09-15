from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fly_sniff.physiology_source_fetch as source_fetch

DNA02_CONTRACT = Path("authority/program-a-dna02-source-contract-v1.json")


class _Headers(dict):
    def get(self, key: str, default=None):
        return super().get(key, default)


class _FakeResponse:
    def __init__(self, payload: bytes, *, url: str = "https://example.test/data", headers=None):
        self.payload = payload
        self.url = url
        self.headers = _Headers(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def geturl(self) -> str:
        return self.url

    def read(self, count: int) -> bytes:
        return self.payload[:count]


def _dataverse_payload() -> dict:
    return {
        "status": "OK",
        "data": {
            "id": 123,
            "latestVersion": {
                "versionNumber": 1,
                "versionMinorNumber": 2,
                "versionState": "RELEASED",
                "files": [
                    {
                        "restricted": False,
                        "dataFile": {
                            "id": 77,
                            "filename": "180410_gfp_3G_ss730_dual_08.mat",
                            "directoryLabel": "A2_dual",
                            "filesize": 100,
                            "contentType": "application/matlab-mat",
                            "checksum": {"type": "MD5", "value": "a" * 32},
                        },
                    },
                    {
                        "restricted": False,
                        "dataFile": {
                            "id": 88,
                            "filename": "180430_gfp_3G_ss730_dual_12.mat",
                            "directoryLabel": "A2_dual",
                            "filesize": 120,
                            "contentType": "application/matlab-mat",
                            "checksum": {"type": "MD5", "value": "b" * 32},
                        },
                    },
                    {
                        "restricted": False,
                        "dataFile": {
                            "id": 99,
                            "filename": "other_recording.mat",
                            "filesize": 50,
                            "contentType": "application/matlab-mat",
                            "checksum": {"type": "MD5", "value": "c" * 32},
                        },
                    },
                ],
            },
        },
    }


def test_bounded_get_rejects_advertised_oversize(monkeypatch) -> None:
    response = _FakeResponse(b"abc", headers={"Content-Length": "1000"})
    monkeypatch.setattr(source_fetch, "urlopen", lambda request, timeout: response)
    with pytest.raises(ValueError, match="exceeding cap"):
        source_fetch._bounded_get("https://example.test/data", max_bytes=10)


def test_bounded_get_rejects_stream_that_exceeds_cap(monkeypatch) -> None:
    response = _FakeResponse(b"01234567890")
    monkeypatch.setattr(source_fetch, "urlopen", lambda request, timeout: response)
    with pytest.raises(ValueError, match="exceeded byte cap"):
        source_fetch._bounded_get("https://example.test/data", max_bytes=10)


def test_bounded_get_rejects_non_https_redirect(monkeypatch) -> None:
    response = _FakeResponse(b"abc", url="http://example.test/data")
    monkeypatch.setattr(source_fetch, "urlopen", lambda request, timeout: response)
    with pytest.raises(ValueError, match="non-HTTPS"):
        source_fetch._bounded_get("https://example.test/data", max_bytes=10)


def test_pfn_readme_fetch_hashes_exact_bytes_and_writes_receipt(tmp_path, monkeypatch) -> None:
    raw = b"{\\rtf1 test Currier README}"
    expected_md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
    response = _FakeResponse(
        raw,
        url="https://datadryad.org/downloads/file_stream/536042",
        headers={"Content-Length": str(len(raw)), "Content-Type": "application/rtf"},
    )
    monkeypatch.setattr(source_fetch, "urlopen", lambda request, timeout: response)
    monkeypatch.setattr(source_fetch, "PFN_README_EXPECTED_MD5", expected_md5)

    receipt = source_fetch.fetch_pfn_readme(tmp_path)

    assert receipt["observed_md5"] == expected_md5
    assert receipt["observed_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["md5_matches_published_authority"] is True
    assert (tmp_path / source_fetch.PFN_README_FILENAME).read_bytes() == raw
    persisted = json.loads((tmp_path / "pfn-readme-fetch-receipt.json").read_text())
    assert persisted["navigation_performance_used"] is False


def test_pfn_readme_mismatch_refuses_to_publish_receipt(tmp_path, monkeypatch) -> None:
    response = _FakeResponse(b"wrong bytes")
    monkeypatch.setattr(source_fetch, "urlopen", lambda request, timeout: response)
    with pytest.raises(ValueError, match="do not match the published MD5"):
        source_fetch.fetch_pfn_readme(tmp_path)
    assert not (tmp_path / "pfn-readme-fetch-receipt.json").exists()


def test_dataverse_inventory_normalizes_ids_names_checksums_and_restriction() -> None:
    inventory = source_fetch.normalize_dataverse_inventory(_dataverse_payload())
    assert [item["file_id"] for item in inventory] == [99, 77, 88]
    by_id = {item["file_id"]: item for item in inventory}
    assert by_id[77]["filename"] == "180410_gfp_3G_ss730_dual_08.mat"
    assert by_id[77]["checksum_type"] == "MD5"
    assert by_id[77]["checksum_value"] == "a" * 32
    assert by_id[77]["restricted"] is False


def test_duplicate_dataverse_file_id_fails_closed() -> None:
    payload = _dataverse_payload()
    payload["data"]["latestVersion"]["files"][1]["dataFile"]["id"] = 77
    with pytest.raises(ValueError, match="duplicate Dataverse file id"):
        source_fetch.normalize_dataverse_inventory(payload)


def test_non_ok_dataverse_response_fails_closed() -> None:
    with pytest.raises(ValueError, match="status is not OK"):
        source_fetch.normalize_dataverse_inventory({"status": "ERROR", "data": {}})


def test_dna02_review_matches_known_sessions_without_promoting_cohort() -> None:
    payload = _dataverse_payload()
    raw_sha = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
    review = source_fetch.build_dna02_metadata_review(
        payload,
        contract_path=DNA02_CONTRACT,
        raw_sha256=raw_sha,
    )

    assert review["candidate_matches"]["a2_d_08"][0]["file_id"] == 77
    assert review["candidate_matches"]["a2_d_12"][0]["file_id"] == 88
    assert review["candidate_matches"]["a2_d_13"] == []
    assert review["candidate_matches"]["a2_d_14"] == []
    assert review["automatic_cohort_resolution"] is False
    assert review["automatic_file_map_promotion"] is False
    assert "discovery evidence only" in review["review_boundary"]


def test_dataverse_fetch_writes_raw_hash_and_review_only(tmp_path, monkeypatch) -> None:
    raw = json.dumps(_dataverse_payload(), sort_keys=True).encode("utf-8")
    response = _FakeResponse(
        raw,
        url=source_fetch.DNA02_DATAVERSE_API,
        headers={"Content-Length": str(len(raw)), "Content-Type": "application/json"},
    )
    monkeypatch.setattr(source_fetch, "urlopen", lambda request, timeout: response)

    review = source_fetch.fetch_dna02_dataverse_metadata(
        tmp_path,
        contract_path=DNA02_CONTRACT,
    )

    assert review["raw_metadata_sha256"] == hashlib.sha256(raw).hexdigest()
    assert review["file_count"] == 3
    assert (tmp_path / "dna02-dataverse-raw.json").read_bytes() == raw
    receipt = json.loads((tmp_path / "dna02-dataverse-fetch-receipt.json").read_text())
    assert receipt["navigation_performance_used"] is False
    assert receipt["raw_metadata_sha256"] == hashlib.sha256(raw).hexdigest()


def test_fetcher_defaults_are_metadata_scale_not_dataset_scale() -> None:
    assert source_fetch.PFN_README_MAX_BYTES <= 64 * 1024
    assert source_fetch.DNA02_METADATA_MAX_BYTES <= 16 * 1024 * 1024
    assert "file_stream/536042" in source_fetch.PFN_README_URL
    assert "api/datasets/:persistentId" in source_fetch.DNA02_DATAVERSE_API
