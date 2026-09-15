from __future__ import annotations

import hashlib
import json

import pytest

import fly_sniff.dna02_dataset_readme as readme_fetch


def test_dna02_readme_fetch_hashes_verified_utf8_bytes(tmp_path, monkeypatch) -> None:
    raw = b"# dataset\npaired DNa02 recordings\n"
    expected_md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()

    monkeypatch.setattr(readme_fetch, "DNA02_README_EXPECTED_MD5", expected_md5)
    monkeypatch.setattr(
        readme_fetch,
        "_bounded_get",
        lambda url, max_bytes: (
            raw,
            {
                "final_url": readme_fetch.DNA02_README_URL,
                "content_type": "text/markdown",
                "content_length": str(len(raw)),
                "etag": "",
                "last_modified": "",
            },
        ),
    )

    receipt = readme_fetch.fetch_dna02_readme(tmp_path)
    assert receipt["file_id"] == 11634832
    assert receipt["observed_md5"] == expected_md5
    assert receipt["observed_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["navigation_performance_used"] is False
    assert (tmp_path / "README.md").read_bytes() == raw
    persisted = json.loads((tmp_path / "dna02-readme-fetch-receipt.json").read_text())
    assert persisted["observed_sha256"] == hashlib.sha256(raw).hexdigest()


def test_dna02_readme_checksum_mismatch_publishes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        readme_fetch,
        "_bounded_get",
        lambda url, max_bytes: (
            b"wrong",
            {
                "final_url": readme_fetch.DNA02_README_URL,
                "content_type": "text/markdown",
                "content_length": "5",
                "etag": "",
                "last_modified": "",
            },
        ),
    )
    with pytest.raises(ValueError, match="do not match repository MD5"):
        readme_fetch.fetch_dna02_readme(tmp_path)
    assert not (tmp_path / "README.md").exists()
    assert not (tmp_path / "dna02-readme-fetch-receipt.json").exists()


def test_dna02_readme_rejects_non_utf8_after_checksum(monkeypatch, tmp_path) -> None:
    raw = b"\xff\xfe"
    expected_md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
    monkeypatch.setattr(readme_fetch, "DNA02_README_EXPECTED_MD5", expected_md5)
    monkeypatch.setattr(
        readme_fetch,
        "_bounded_get",
        lambda url, max_bytes: (
            raw,
            {
                "final_url": readme_fetch.DNA02_README_URL,
                "content_type": "text/markdown",
                "content_length": str(len(raw)),
                "etag": "",
                "last_modified": "",
            },
        ),
    )
    with pytest.raises(ValueError, match="not valid UTF-8"):
        readme_fetch.fetch_dna02_readme(tmp_path)


def test_dna02_readme_defaults_are_tiny_and_frozen() -> None:
    assert readme_fetch.DNA02_README_FILE_ID == 11634832
    assert readme_fetch.DNA02_README_EXPECTED_MD5 == "cfb0cb853be3a454b965d74eb5c772c0"
    assert readme_fetch.DNA02_README_MAX_BYTES <= 64 * 1024
