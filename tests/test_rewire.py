import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.rewire import degree_preserving_rewire


def _degree(edges, col):
    return edges.groupby(col).size().sort_index().to_dict()


def test_rewire_preserves_directed_degrees():
    nodes = pd.DataFrame({"bodyId": list(range(8))})
    edges = pd.DataFrame(
        {
            "source": [0, 0, 1, 1, 2, 3, 4, 5],
            "target": [2, 3, 4, 5, 6, 7, 6, 7],
            "weight": [3, 4, 5, 6, 2, 2, 3, 3],
        }
    )
    bundle = GraphBundle(nodes, edges, {"odor_left": [0], "steer_right": [7]})
    rewired = degree_preserving_rewire(bundle, seed=9, swaps_per_edge=2)
    assert _degree(edges, "source") == _degree(rewired.edges, "source")
    assert _degree(edges, "target") == _degree(rewired.edges, "target")
    assert len(set(zip(rewired.edges.source, rewired.edges.target))) == len(rewired.edges)
