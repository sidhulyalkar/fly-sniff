import pandas as pd

from fly_sniff.trace_audit import audit_corridor


def _fixture():
    nodes = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "type": ["ORN", "PN", "PFL3", "DNa02"],
        }
    )
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3],
            "target": [2, 3, 4],
            "weight": [8.0, 10.0, 12.0],
        }
    )
    provenance = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "forward_depth": [0, 1, 2, 3],
            "reverse_depth": [3, 2, 1, 0],
            "is_source_seed": [True, False, False, False],
            "is_target_seed": [False, False, False, True],
        }
    )
    report = {
        "source_seed_count": 1,
        "target_seed_count": 1,
        "input_source_seed_count": 1,
        "input_target_seed_count": 1,
        "retained_source_seed_count": 1,
        "retained_target_seed_count": 1,
        "corridor_nodes": 4,
        "corridor_edges": 3,
        "min_weight": 5.0,
        "max_hops": 3,
    }
    return nodes, edges, provenance, report


def test_structural_audit_passes_closed_consistent_corridor():
    nodes, edges, provenance, trace_report = _fixture()
    report = audit_corridor(nodes, edges, provenance, trace_report, top_n=2)

    assert report["passed"]
    assert all(report["checks"].values())
    assert report["counts"] == {
        "nodes": 4,
        "edges": 3,
        "input_source_seeds": 1,
        "input_target_seeds": 1,
        "retained_source_seeds": 1,
        "retained_target_seeds": 1,
    }
    assert report["depths"]["bounded_path_length"] == {"3": 4}
    assert report["edge_geometry"]["shortest_layer_step_fraction"] == 1.0
    assert len(report["top_structural_hubs"]) == 2


def test_structural_audit_parses_csv_style_boolean_strings():
    nodes, edges, provenance, trace_report = _fixture()
    provenance["is_source_seed"] = provenance.is_source_seed.map({True: "True", False: "False"})
    provenance["is_target_seed"] = provenance.is_target_seed.map({True: "1", False: "0"})

    report = audit_corridor(nodes, edges, provenance, trace_report)

    assert report["passed"]
    assert report["counts"]["retained_source_seeds"] == 1
    assert report["counts"]["retained_target_seeds"] == 1


def test_structural_audit_allows_more_input_seeds_than_retained_seeds():
    nodes, edges, provenance, trace_report = _fixture()
    trace_report = dict(trace_report)
    trace_report["source_seed_count"] = 50
    trace_report["input_source_seed_count"] = 50

    report = audit_corridor(nodes, edges, provenance, trace_report)

    assert report["passed"]
    assert report["counts"]["input_source_seeds"] == 50
    assert report["counts"]["retained_source_seeds"] == 1


def test_structural_audit_fails_edge_endpoint_escape():
    nodes, edges, provenance, trace_report = _fixture()
    edges = pd.concat(
        [
            edges,
            pd.DataFrame({"source": [4], "target": [99], "weight": [7.0]}),
        ],
        ignore_index=True,
    )
    trace_report = dict(trace_report)
    trace_report["corridor_edges"] = 4

    report = audit_corridor(nodes, edges, provenance, trace_report)

    assert not report["passed"]
    assert not report["checks"]["edge_endpoint_closure"]


def test_structural_audit_fails_weight_below_trace_threshold():
    nodes, edges, provenance, trace_report = _fixture()
    edges.loc[0, "weight"] = 4.0

    report = audit_corridor(nodes, edges, provenance, trace_report)

    assert not report["passed"]
    assert not report["checks"]["trace_min_weight_respected"]


def test_structural_audit_fails_bounded_depth_above_trace_limit():
    nodes, edges, provenance, trace_report = _fixture()
    provenance.loc[1, "forward_depth"] = 4

    report = audit_corridor(nodes, edges, provenance, trace_report)

    assert not report["passed"]
    assert not report["checks"]["bounded_hop_limit_respected"]


def test_structural_audit_reports_count_mismatch():
    nodes, edges, provenance, trace_report = _fixture()
    trace_report = dict(trace_report)
    trace_report["corridor_nodes"] = 999

    report = audit_corridor(nodes, edges, provenance, trace_report)

    assert not report["passed"]
    assert not report["checks"]["trace_report_counts_match"]
    assert report["report_mismatches"]["corridor_nodes"] == {
        "trace_report": 999,
        "observed": 4,
    }
