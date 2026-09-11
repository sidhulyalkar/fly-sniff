import pandas as pd

from fly_sniff.trace import trace_corridor


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
