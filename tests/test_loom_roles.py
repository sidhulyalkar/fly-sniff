import json

import pandas as pd
import pytest

from fly_sniff.loom_roles import (
    build_candidate_manifest,
    derive_loom_roles,
    git_blob_sha1,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5, 6, 7, 8],
            "type": ["LPLC2", "LPLC2", "DNp06", "DNp06", "LC4", "LC4", "DNp01", "DNp01"],
            "side": ["L", "R", "L", "R", "L", "R", "L", "R"],
        }
    )


def test_derives_exact_bilateral_primary_roles():
    roles, populations = derive_loom_roles(_frame())
    assert roles["loom_left"] == [1]
    assert roles["loom_right"] == [2]
    assert roles["escape_left"] == [3]
    assert roles["escape_right"] == [4]
    assert roles["loom_lc4_left"] == [5]
    assert roles["loom_lc4_right"] == [6]
    assert roles["escape_dnp01_left"] == [7]
    assert roles["escape_dnp01_right"] == [8]
    assert populations["LPLC2"] == {"L": [1], "R": [2]}


def test_role_derivation_requires_bilateral_lplc2_and_dnp06():
    with pytest.raises(ValueError, match="both sides"):
        derive_loom_roles(_frame().query("bodyId != 2"))
    with pytest.raises(ValueError, match="both sides"):
        derive_loom_roles(_frame().query("bodyId != 4"))


def test_role_derivation_rejects_unresolved_side():
    frame = _frame()
    frame.loc[frame.bodyId == 1, "side"] = None
    with pytest.raises(ValueError, match="without resolved L/R side"):
        derive_loom_roles(frame)


def test_candidate_manifest_verifies_git_blob_and_never_qualifies():
    rows = [
        {"bodyId": 1, "type": "LPLC2", "side": "left"},
        {"bodyId": 2, "type": "LPLC2", "side": "right"},
        {"bodyId": 3, "type": "DNp06", "side": "left"},
        {"bodyId": 4, "type": "DNp06", "side": "right"},
    ]
    raw = json.dumps(rows, separators=(",", ":")).encode()
    blob = git_blob_sha1(raw)
    manifest = build_candidate_manifest(
        raw,
        source="fixture.json",
        upstream_repository="example/repo",
        expected_blob_sha=blob,
    )
    assert manifest["git_blob_verified"] is True
    assert manifest["qualification_status"] == "candidate"
    assert manifest["scientific_claim_allowed"] is False
    assert manifest["edge_authority"] == "pending"
    assert manifest["roles"]["loom_left"] == [1]
    assert manifest["roles"]["escape_right"] == [4]


def test_candidate_manifest_rejects_source_drift():
    raw = b'[{"bodyId":1}]'
    with pytest.raises(ValueError, match="source drifted"):
        build_candidate_manifest(
            raw,
            source="fixture.json",
            upstream_repository="example/repo",
            expected_blob_sha="0" * 40,
        )
