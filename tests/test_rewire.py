import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.rewire import degree_preserving_rewire, lesion_incoming_to_roles


def _degree(edges, col):
    return edges.groupby(col).size().sort_index().to_dict()


def _source_attributes(edges):
    return {
        int(source): sorted(
            (float(row.weight), int(row.sign))
            for row in frame[["weight", "sign"]].itertuples(index=False)
        )
        for source, frame in edges.groupby("source")
    }


def test_rewire_preserves_directed_degrees_and_presynaptic_attributes():
    nodes = pd.DataFrame({"bodyId": list(range(8))})
    edges = pd.DataFrame(
        {
            "source": [0, 0, 1, 1, 2, 3, 4, 5],
            "target": [2, 3, 4, 5, 6, 7, 6, 7],
            "weight": [3, 4, 5, 6, 2, 2, 3, 3],
            "sign": [1, 1, -1, -1, 1, 1, -1, -1],
        }
    )
    bundle = GraphBundle(
        nodes,
        edges,
        {"odor_left": [0], "steer_right": [7]},
        {"qualification_status": "qualified", "graph_role": "malecns"},
    )
    rewired = degree_preserving_rewire(bundle, seed=9, swaps_per_edge=2)
    assert _degree(edges, "source") == _degree(rewired.edges, "source")
    assert _degree(edges, "target") == _degree(rewired.edges, "target")
    assert _source_attributes(edges) == _source_attributes(rewired.edges)
    assert len(set(zip(rewired.edges.source, rewired.edges.target, strict=True))) == len(rewired.edges)
    assert rewired.manifest["qualification_status"] == "qualified"
    assert rewired.manifest["graph_role"] == "degree-preserving-rewire"
    assert rewired.manifest["rewire"]["exact_in_out_degree_preserved"] is True


def test_role_input_lesion_removes_only_edges_into_frozen_roles():
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4]})
    edges = pd.DataFrame(
        {
            "source": [1, 1, 2, 3],
            "target": [2, 3, 4, 4],
            "weight": [5, 6, 7, 8],
            "sign": [1, 1, -1, 1],
        }
    )
    bundle = GraphBundle(
        nodes,
        edges,
        {
            "odor_left": [1],
            "odor_right": [2],
            "steer_left": [3],
            "steer_right": [4],
        },
        {"qualification_status": "qualified", "graph_role": "malecns"},
    )

    lesioned = lesion_incoming_to_roles(bundle, ["steer_left", "steer_right"])

    assert list(zip(lesioned.edges.source, lesioned.edges.target, strict=True)) == [(1, 2)]
    assert lesioned.manifest["qualification_status"] == "qualified"
    assert lesioned.manifest["graph_role"] == "role-input-lesion"
    assert lesioned.manifest["lesion"]["body_ids"] == [3, 4]
    assert lesioned.manifest["lesion"]["removed_edge_count"] == 3
