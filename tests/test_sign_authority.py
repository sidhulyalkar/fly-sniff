import json

import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.sign_authority import build_sign_authority_report


def _authority(tmp_path):
    path = tmp_path / "authority.json"
    path.write_text(
        json.dumps(
            {
                "authority_kind": "type-level-transmitter-evidence",
                "dataset": "male-cns:v1.0",
                "model_sign_rule": {"ACh": 1, "GABA": -1, "Glu": 0, "other_or_unclear": 0},
                "types": {
                    "FB5AB": {
                        "predicted_neurotransmitter": "ACh",
                        "confidence": 0.9,
                        "source": "fixture",
                    }
                },
            }
        )
    )
    return path


def _bundle(unknown_sign=0):
    nodes = pd.DataFrame(
        [
            {"bodyId": 1, "type": "FB5AB"},
            {"bodyId": 2, "type": "Mystery"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": 1, "target": 2, "weight": 5.0, "sign": 1},
            {"source": 2, "target": 1, "weight": 3.0, "sign": unknown_sign},
        ]
    )
    return GraphBundle(nodes, edges, {}, {"dataset": "male-cns:v1.0"})


def test_unknown_transmitter_stays_unknown_and_zero_signed(tmp_path):
    report = build_sign_authority_report(_bundle(), [_authority(tmp_path)])
    records = {item["source_body_id"]: item for item in report["source_records"]}
    assert report["passed"] is True
    assert records[1]["transmitter"] == "ACh"
    assert records[1]["model_sign"] == 1
    assert records[2]["transmitter"] is None
    assert records[2]["confidence"] is None
    assert records[2]["model_sign"] == 0
    assert report["coverage"]["signed_edge_fraction"] == 0.5


def test_nonzero_sign_without_authority_is_rejected(tmp_path):
    report = build_sign_authority_report(_bundle(unknown_sign=1), [_authority(tmp_path)])
    assert report["passed"] is False
    assert report["coverage"]["edge_sign_mismatch_count"] == 1
    mismatch = report["edge_sign_mismatches"][0]
    assert mismatch["source_body_id"] == 2
    assert mismatch["authority_sign"] == 0
