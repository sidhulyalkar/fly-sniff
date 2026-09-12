from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from fly_sniff.r002_extract import build_r002_bundle


def _write_weights(path: Path) -> None:
    table = pa.table(
        {
            "body_pre": pa.array([1, 2, 3, 4, 5, 6, 1, 3], type=pa.int64()),
            "body_post": pa.array([5, 5, 6, 6, 1, 3, 2, 4], type=pa.int64()),
            "weight": pa.array([7, 3, 11, 5, 2, 4, 1, 1], type=pa.int64()),
        }
    )
    feather.write_feather(table, path)


def _write_nt(path: Path) -> None:
    pd.DataFrame(
        {
            "body": [1, 2, 3, 4, 5, 6],
            "consensus_nt": [
                "acetylcholine",
                "acetylcholine",
                "acetylcholine",
                "acetylcholine",
                "acetylcholine",
                "acetylcholine",
            ],
        }
    ).to_feather(path)


def _authority() -> dict:
    return {
        "schema": "fly-sniff-r002-authority-v1",
        "dataset": "male-cns:v1.0",
        "populations": {
            "LPLC2_L": [1],
            "LPLC2_R": [2],
            "LC4_L": [3],
            "LC4_R": [4],
            "DNp01_L": [5],
            "DNp01_R": [6],
        },
        "structural_summaries": [
            {
                "source_type": "LPLC2",
                "source_side": "L",
                "target_type": "DNp01",
                "target_side": "L",
                "aggregate_connections": 7,
            },
            {
                "source_type": "LPLC2",
                "source_side": "R",
                "target_type": "DNp01",
                "target_side": "R",
                "aggregate_connections": 0,
            },
            {
                "source_type": "LC4",
                "source_side": "L",
                "target_type": "DNp01",
                "target_side": "L",
                "aggregate_connections": 0,
            },
            {
                "source_type": "LC4",
                "source_side": "R",
                "target_type": "DNp01",
                "target_side": "R",
                "aggregate_connections": 5,
            },
        ],
    }


def test_extract_builds_escape_roles_and_signed_edges(tmp_path: Path) -> None:
    weights = tmp_path / "weights.feather"
    nt = tmp_path / "nt.feather"
    _write_weights(weights)
    _write_nt(nt)
    authority = _authority()

    nodes, edges, roles, manifest = build_r002_bundle(
        authority,
        weights_path=weights,
        neurotransmitters_path=nt,
    )

    assert len(nodes) == 6
    assert set(edges.columns) == {"source", "target", "weight", "source_nt", "sign"}
    assert set(edges.sign) == {1}
    assert roles["escape"] == [5, 6]
    assert "steer_left" not in roles
    assert manifest["behavioral_readout"]["directional_steering_claim"] is False
    assert manifest["qualification_status"] == "candidate"


def test_extract_fails_when_body_edges_disagree_with_type_summary(tmp_path: Path) -> None:
    weights = tmp_path / "weights.feather"
    nt = tmp_path / "nt.feather"
    _write_weights(weights)
    _write_nt(nt)
    authority = _authority()
    authority["structural_summaries"][0]["aggregate_connections"] = 8

    try:
        build_r002_bundle(authority, weights_path=weights, neurotransmitters_path=nt)
    except ValueError as exc:
        assert "structural cross-check failed" in str(exc)
    else:
        raise AssertionError("expected structural mismatch to fail closed")
