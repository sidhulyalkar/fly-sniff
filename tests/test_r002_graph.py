import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from fly_sniff.graph import GraphBundle
from fly_sniff.r002_graph import build_r002_graph


def _write_table(path: Path, data: dict):
    feather.write_feather(pa.table(data), path)


def test_r002_graph_reproduces_authority_sums_and_roles(tmp_path):
    authority = {
        "dataset": "male-cns:v1.0",
        "populations": {
            "LPLC2_L": [1, 2],
            "LPLC2_R": [3, 4],
            "LC4_L": [5],
            "LC4_R": [6],
            "DNp01_L": [7],
            "DNp01_R": [8],
        },
        "structural_summaries": [
            {"source_type": "LPLC2", "source_side": "L", "target_type": "DNp01", "target_side": "L", "aggregate_connections": 7},
            {"source_type": "LC4", "source_side": "L", "target_type": "DNp01", "target_side": "L", "aggregate_connections": 6},
            {"source_type": "LPLC2", "source_side": "R", "target_type": "DNp01", "target_side": "R", "aggregate_connections": 9},
            {"source_type": "LC4", "source_side": "R", "target_type": "DNp01", "target_side": "R", "aggregate_connections": 8},
        ],
    }
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority))

    annotations = tmp_path / "annotations.feather"
    rows = []
    for body, type_name, side in [
        (1, "LPLC2", "L"), (2, "LPLC2", "L"), (3, "LPLC2", "R"), (4, "LPLC2", "R"),
        (5, "LC4", "L"), (6, "LC4", "R"), (7, "DNp01", "L"), (8, "DNp01", "R"),
    ]:
        rows.append({"bodyId": body, "type": type_name, "somaSide": side})
    pd.DataFrame(rows).to_feather(annotations)

    neurotransmitters = tmp_path / "nt.feather"
    pd.DataFrame(
        {"body": list(range(1, 9)), "consensus_nt": ["acetylcholine"] * 8}
    ).to_feather(neurotransmitters)

    weights = tmp_path / "weights.feather"
    _write_table(
        weights,
        {
            "body_pre": [1, 2, 5, 3, 4, 6, 7],
            "body_post": [7, 7, 7, 8, 8, 8, 1],
            "weight": [3, 4, 6, 5, 4, 8, 2],
        },
    )

    output = tmp_path / "graph"
    manifest = build_r002_graph(
        authority_path,
        annotations,
        neurotransmitters,
        weights,
        output,
    )
    assert manifest["selected_body_count"] == 8
    assert manifest["selected_edge_count"] == 7
    assert all(row["exact_match"] for row in manifest["direct_edge_validation"])

    bundle = GraphBundle.load(output)
    bundle.validate(require_sign=True, require_qualified=False)
    assert bundle.roles["escape_left"] == [7]
    assert bundle.roles["escape_right"] == [8]
    assert set(bundle.edges["sign"]) == {1}
