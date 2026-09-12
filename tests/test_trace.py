import pandas as pd

from fly_sniff.trace import resolve_edge_columns, trace_corridor


def test_resolve_edge_columns_accepts_public_malecns_schema():
    weights = pd.DataFrame(
        {
            "body_pre": [1],
            "body_post": [2],
            "weight": [7],
        }
    )

    assert resolve_edge_columns(weights) == ("body_pre", "body_post", "weight")


def test_trace_corridor_keeps_only_nodes_on_bounded_source_target_paths():
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5, 9],
            "type": ["ORN", "PN", "hDeltaC", "PFL3", "DNa02", "dead_end"],
        }
    )
    weights = pd.DataFrame(
        {
            "source": [1, 2, 3, 4, 1],
            "target": [2, 3, 4, 5, 9],
            "weight": [9, 9, 9, 9, 99],
        }
    )
    nodes, edges, provenance = trace_corridor(
        annotations,
        weights,
        {1},
        {5},
        max_hops=4,
        min_weight=5,
        fanout_per_node=10,
    )
    assert set(nodes.bodyId) == {1, 2, 3, 4, 5}
    assert set(edges.target) == {2, 3, 4, 5}
    assert 9 not in set(provenance.bodyId)


def test_trace_corridor_accepts_public_malecns_schema():
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 2, 3],
            "type": ["ORN", "PN", "DNa02"],
        }
    )
    weights = pd.DataFrame(
        {
            "body_pre": [1, 2],
            "body_post": [2, 3],
            "weight": [8, 9],
        }
    )

    nodes, edges, provenance = trace_corridor(
        annotations,
        weights,
        {1},
        {3},
        max_hops=2,
        min_weight=5,
        fanout_per_node=10,
    )

    assert set(nodes.bodyId) == {1, 2, 3}
    assert list(edges.columns) == ["source", "target", "weight"]
    assert list(edges[["source", "target"]].itertuples(index=False, name=None)) == [(1, 2), (2, 3)]
    assert set(provenance.bodyId) == {1, 2, 3}
