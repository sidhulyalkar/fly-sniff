from __future__ import annotations

import hashlib
import json
from urllib.error import HTTPError

import pytest

import fly_sniff.pfn_dataset_readme as pfn_readme


def _headers(url: str) -> dict[str, str]:
    return {
        "content_type": "application/rtf",
        "content_length": "24",
        "etag": "",
        "last_modified": "",
        "final_url": url,
    }


def _http_error(url: str, code: int) -> HTTPError:
    return HTTPError(url=url, code=code, msg="test", hdrs=None, fp=None)


def test_primary_dryad_success_does_not_touch_mirror(tmp_path, monkeypatch) -> None:
    raw = b"{\\rtf1 Currier README}"
    expected_md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
    calls: list[str] = []

    def fake_get(url: str, *, max_bytes: int):
        calls.append(url)
        return raw, _headers(url)

    monkeypatch.setattr(pfn_readme, "_bounded_get", fake_get)
    monkeypatch.setattr(pfn_readme, "PFN_README_EXPECTED_MD5", expected_md5)
    receipt = pfn_readme.fetch_pfn_readme(tmp_path)

    assert calls == [pfn_readme.PFN_README_PRIMARY_URL]
    assert receipt["retrieval_transport"] == "dryad_primary"
    assert receipt["md5_matches_dryad_authority"] is True
    assert receipt["observed_sha256"] == hashlib.sha256(raw).hexdigest()


def test_dryad_403_falls_back_to_checksum_gated_archive_mirror(tmp_path, monkeypatch) -> None:
    raw = b"{\\rtf1 mirror Currier README}"
    expected_md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
    calls: list[str] = []

    def fake_get(url: str, *, max_bytes: int):
        calls.append(url)
        if url == pfn_readme.PFN_README_PRIMARY_URL:
            raise _http_error(url, 403)
        return raw, _headers(url)

    monkeypatch.setattr(pfn_readme, "_bounded_get", fake_get)
    monkeypatch.setattr(pfn_readme, "PFN_README_EXPECTED_MD5", expected_md5)
    receipt = pfn_readme.fetch_pfn_readme(tmp_path)

    assert calls == [pfn_readme.PFN_README_PRIMARY_URL, pfn_readme.PFN_README_MIRROR_URL]
    assert receipt["retrieval_transport"] == "zenodo_dryad_archive_mirror"
    assert receipt["attempts"][0]["http_status"] == 403
    assert receipt["scientific_authority"]["repository"] == "Dryad"
    assert "transport-only" in receipt["transport_policy"]
    assert (tmp_path / pfn_readme.PFN_README_FILENAME).read_bytes() == raw


def test_dryad_500_does_not_silently_fallback(tmp_path, monkeypatch) -> None:
    def fake_get(url: str, *, max_bytes: int):
        raise _http_error(url, 500)

    monkeypatch.setattr(pfn_readme, "_bounded_get", fake_get)
    with pytest.raises(HTTPError) as exc_info:
        pfn_readme.fetch_pfn_readme(tmp_path)
    assert exc_info.value.code == 500
    assert not (tmp_path / "pfn-readme-fetch-receipt.json").exists()


def test_mirror_mismatch_refuses_receipt_and_bytes(tmp_path, monkeypatch) -> None:
    wrong = b"not authoritative bytes"

    def fake_get(url: str, *, max_bytes: int):
        if url == pfn_readme.PFN_README_PRIMARY_URL:
            raise _http_error(url, 403)
        return wrong, _headers(url)

    monkeypatch.setattr(pfn_readme, "_bounded_get", fake_get)
    with pytest.raises(ValueError, match="Dryad's published MD5"):
        pfn_readme.fetch_pfn_readme(tmp_path)
    assert not (tmp_path / "pfn-readme-fetch-receipt.json").exists()
    assert not (tmp_path / pfn_readme.PFN_README_FILENAME).exists()


def test_no_mirror_mode_preserves_dryad_403(tmp_path, monkeypatch) -> None:
    def fake_get(url: str, *, max_bytes: int):
        raise _http_error(url, 403)

    monkeypatch.setattr(pfn_readme, "_bounded_get", fake_get)
    with pytest.raises(HTTPError) as exc_info:
        pfn_readme.fetch_pfn_readme(tmp_path, mirror_url=None)
    assert exc_info.value.code == 403


def test_receipt_keeps_dryad_authority_even_when_mirror_transports_bytes(tmp_path, monkeypatch) -> None:
    raw = b"{\\rtf1 mirror evidence}"
    expected_md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()

    def fake_get(url: str, *, max_bytes: int):
        if url == pfn_readme.PFN_README_PRIMARY_URL:
            raise _http_error(url, 401)
        return raw, _headers(url)

    monkeypatch.setattr(pfn_readme, "_bounded_get", fake_get)
    monkeypatch.setattr(pfn_readme, "PFN_README_EXPECTED_MD5", expected_md5)
    pfn_readme.fetch_pfn_readme(tmp_path)
    receipt = json.loads((tmp_path / "pfn-readme-fetch-receipt.json").read_text())

    assert receipt["scientific_authority"]["doi"] == "10.5061/dryad.vq83bk3rh"
    assert receipt["scientific_authority"]["file_stream_id"] == 536042
    assert receipt["scientific_authority"]["published_md5"] == expected_md5
    assert receipt["retrieval_transport"] == "zenodo_dryad_archive_mirror"
    assert receipt["navigation_performance_used"] is False


def test_readme_fetch_remains_metadata_scale() -> None:
    assert pfn_readme.PFN_README_MAX_BYTES <= 64 * 1024
    assert pfn_readme.PFN_README_MIRROR_RECORD == "https://zenodo.org/records/4407395"
