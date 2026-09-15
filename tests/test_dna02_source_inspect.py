from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from fly_sniff.dna02_source_inspect import (
    DEFAULT_EVIDENCE,
    _hash_file,
    _load_byte_evidence,
    _match_local_paths,
    detect_mat_format,
    inspect_mat_schema,
)
from fly_sniff.freeze import canonical_sha256


def test_real_source_byte_evidence_is_canonical_and_resolved() -> None:
    payload = _load_byte_evidence(DEFAULT_EVIDENCE)
    claimed = payload["evidence_sha256"]
    without_hash = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    assert canonical_sha256(without_hash) == claimed
    assert payload["status"] == "SHA256_BYTE_IDENTITIES_RESOLVED"
    assert len(payload["files"]) == 4


def test_classic_mat_schema_is_inventoried_without_loading_values(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mat"
    savemat(
        path,
        {
            "yaw": np.arange(12, dtype=np.float64).reshape(3, 4),
            "ephys_A": np.arange(5, dtype=np.float32),
        },
    )

    assert detect_mat_format(path) == "matlab_v5"
    report = inspect_mat_schema(path)
    by_name = {item["name"]: item for item in report["variables"]}
    assert report["inventory_method"] == "scipy.io.whosmat"
    assert by_name["yaw"]["shape"] == [3, 4]
    assert by_name["ephys_A"]["shape"] in ([1, 5], [5, 1])
    assert "values" not in json.dumps(report)


def test_hash_file_reports_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"fly-sniff-source-inspection")
    size, md5, sha256 = _hash_file(path, chunk_bytes=5)
    assert size == len(b"fly-sniff-source-inspection")
    assert md5 == "9572c45b9278cff06bd29a062a09d757"
    assert sha256 == "40490f5190bb4f19ab878355c71197396a1961e228010945172baaf36e1d04bc"


def test_local_source_set_must_be_exact(tmp_path: Path) -> None:
    expected = {"a.mat", "b.mat"}
    a = tmp_path / "a.mat"
    a.write_bytes(b"a")
    with pytest.raises(ValueError, match="missing frozen DNa02 source files"):
        _match_local_paths([a], expected)

    surprise = tmp_path / "surprise.mat"
    surprise.write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected DNa02 source filename"):
        _match_local_paths([a, surprise], expected)


def test_tampered_byte_evidence_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    payload["files"][0]["byte_count"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical hash does not match"):
        _load_byte_evidence(path)
