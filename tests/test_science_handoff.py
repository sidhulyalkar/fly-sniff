import json

import pandas as pd

from fly_sniff.science_handoff import build_handoff


def test_science_handoff_collects_seed_anchor_and_depth_evidence(tmp_path):
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5],
            "type": ["Or42b", "PFNa", "PFL3", "DNa02", "FB5AB"],
            "instance": ["Or42b_L", "PFNa_L", "PFL3_L", "DNa02_L", "FB5AB_L"],
            "class": ["sensory", "CX", "CX", "descending", "FB"],
            "subclass": ["olfactory", "PFN", "PFL", "DN", "FB"],
        }
    )
    annotations_path = tmp_path / "annotations.feather"
    annotations.to_feather(annotations_path)

    weights = pd.DataFrame(
        {
            "body_pre": [1, 3],
            "body_post": [3, 4],
            "weight": [8.0, 9.0],
        }
    )
    weights_path = tmp_path / "weights.feather"
    weights.to_feather(weights_path)

    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    nodes = annotations[annotations.bodyId.isin([1, 3, 4])].copy()
    edges = pd.DataFrame(
        {
            "source": [1, 3],
            "target": [3, 4],
            "weight": [8.0, 9.0],
        }
    )
    provenance = pd.DataFrame(
        {
            "bodyId": [1, 3, 4],
            "forward_depth": [0, 1, 2],
            "reverse_depth": [2, 1, 0],
            "is_source_seed": [True, False, False],
            "is_target_seed": [False, False, True],
        }
    )
    nodes.to_parquet(trace_dir / "nodes.parquet", index=False)
    edges.to_parquet(trace_dir / "edges.parquet", index=False)
    provenance.to_csv(trace_dir / "path_provenance.csv", index=False)
    (trace_dir / "trace_report.json").write_text(
        json.dumps(
            {
                "source_seed_count": 1,
                "target_seed_count": 1,
                "input_source_seed_count": 1,
                "input_target_seed_count": 1,
                "retained_source_seed_count": 1,
                "retained_target_seed_count": 1,
                "corridor_nodes": 3,
                "corridor_edges": 2,
                "min_weight": 5.0,
                "max_hops": 2,
            }
        )
    )

    report = build_handoff(
        annotations_path,
        weights_path,
        trace_dir,
        max_examples=5,
    )

    assert report["trace_audit"]["passed"]
    assert report["olfactory_seed_hypothesis"]["union_count"] == 1
    assert report["anchor_populations"]["PFNa"]["exact_count"] == 1
    assert report["anchor_corridor_membership"]["PFNa"]["retained_in_corridor_count"] == 0
    assert report["anchor_corridor_membership"]["PFL3"]["retained_in_corridor_count"] == 1
    assert report["anchor_corridor_membership"]["DNa02"]["retained_in_corridor_count"] == 1
    transitions = report["corridor_profiles"]["forward_depth_edge_transitions"]
    assert [(row["source_forward_depth"], row["target_forward_depth"]) for row in transitions] == [
        (0.0, 1.0),
        (1.0, 2.0),
    ]
    assert len(report["inputs"]["annotations"]["sha256"]) == 64
    assert len(report["inputs"]["weights"]["sha256"]) == 64
