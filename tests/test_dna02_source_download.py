from __future__ import annotations

import pytest

from fly_sniff.dna02_source_download import _verify_hashes, expected_sources


def test_real_expected_sources_are_exact_and_content_addressed() -> None:
    specs = expected_sources()
    assert [item["fly_alias"] for item in specs] == [
        "a2_d_08",
        "a2_d_12",
        "a2_d_13",
        "a2_d_14",
    ]
    assert sum(item["byte_count"] for item in specs) == 929_441_392
    assert all(len(item["md5"]) == 32 for item in specs)
    assert all(len(item["sha256"]) == 64 for item in specs)


def test_download_verification_rejects_every_identity_axis() -> None:
    expected = {
        "byte_count": 3,
        "md5": "900150983cd24fb0d6963f7d28e17f72",
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }
    _verify_hashes(
        filename="tiny.mat",
        size=3,
        md5=expected["md5"],
        sha256=expected["sha256"],
        expected=expected,
    )
    with pytest.raises(ValueError, match="byte-count mismatch"):
        _verify_hashes(
            filename="tiny.mat",
            size=2,
            md5=expected["md5"],
            sha256=expected["sha256"],
            expected=expected,
        )
    with pytest.raises(ValueError, match="MD5 mismatch"):
        _verify_hashes(
            filename="tiny.mat",
            size=3,
            md5="0" * 32,
            sha256=expected["sha256"],
            expected=expected,
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify_hashes(
            filename="tiny.mat",
            size=3,
            md5=expected["md5"],
            sha256="0" * 64,
            expected=expected,
        )
