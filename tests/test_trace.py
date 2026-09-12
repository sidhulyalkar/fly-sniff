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
    assert nodes.annotation_present.all()
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
    assert nodes.annotation_present.all()
    assert list(edges.columns) == ["source", "target", "weight"]
    assert list(edges[["source", "target"]].itertuples(index=False, name=None)) == [(1, 2), (2, 3)]
    assert set(provenance.bodyId) == {1, 2, 3}


def test_trace_corridor_preserves_unannotated_structural_intermediate():
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 3],
            "type": ["ORN", "DNa02"],
        }
    )
    weights = pd.DataFrame(
        {
            "source": [1, 2],
            "target": [2, 3],
            "weight": [8.0, 9.0],
        }
    )

    nodes, edges, provenance = trace_corridor(
        annotations,
        weights,
        {1},
        {3},
        max_hops=2,
        min_weight=5.0,
        fanout_per_node=10,
    )

    assert set(nodes.bodyId) == {1, 2, 3}
    missing = nodes.set_index("bodyId").loc[2]
    assert not bool(missing.annotation_present)
    assert pd.isna(missing.type)
    assert set(edges.source) | set(edges.target) == {1, 2, 3}
    assert set(provenance.bodyId) == {1, 2, 3}


def test_trace_persisted_edges_obey_min_weight_threshold():
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 2, 3],
            "type": ["ORN", "PN", "DNa02"],
        }
    )
    weights = pd.DataFrame(
        {
            "source": [1, 2, 1],
            "target": [2, 3, 3],
            "weight": [8.0, 9.0, 1.0],
        }
    )

    nodes, edges, _ = trace_corridor(
        annotations,
        weights,
        {1},
        {3},
        max_hops=2,
        min_weight=5.0,
        fanout_per_node=10,
    )

    assert set(nodes.bodyId) == {1, 2, 3}
    assert (edges.weight >= 5.0).all()
    assert (1, 3) not in set(edges[["source", "target"]].itertuples(index=False, name=None))
