import pandas as pd

from fly_sniff.flyarc_prepare import (
    prepare_graph,
    retained_annotations,
    selected_neurotransmitters,
)


def _write_sources(tmp_path):
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5, 6, 99],
            "superclass": [
                "central",
                "central",
                "central",
                "central",
                "central",
                "central",
                "",
            ],
            "type": ["a", "b", "c", "d", "e", "f", "fragment"],
        }
    )
    connectivity = pd.DataFrame(
        {
            "body_pre": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 99, 1],
            "body_post": [2, 3, 4, 5, 6, 1, 4, 5, 6, 1, 2, 3, 1, 99],
            "weight": [9, 8, 7, 6, 5, 4, 7, 7, 6, 6, 5, 5, 100, 100],
        }
    )
    nts = pd.DataFrame(
        {
            "body": [1, 2, 3, 4, 5, 6, 99],
            "consensus_nt": [
                "acetylcholine",
                "gaba",
                "glutamate",
                "histamine",
                "unclear",
                "unclear",
                "acetylcholine",
            ],
            "predicted_nt": [
                "acetylcholine",
                "gaba",
                "glutamate",
                "histamine",
                "dopamine",
                "unclear",
                "acetylcholine",
            ],
            "predicted_nt_confidence": [0.9, 0.9, 0.9, 0.9, 0.8, 0.2, 0.9],
            "celltype_predicted_nt": [
                "acetylcholine",
                "gaba",
                "glutamate",
                "histamine",
                "dopamine",
                "acetylcholine",
                "acetylcholine",
            ],
        }
    )

    ann_path = tmp_path / "annotations.feather"
    edge_path = tmp_path / "weights.feather"
    nt_path = tmp_path / "nt.feather"
    annotations.to_feather(ann_path)
    connectivity.to_feather(edge_path)
    nts.to_feather(nt_path)
    return ann_path, edge_path, nt_path


def test_retention_uses_nonempty_superclass(tmp_path):
    annotations, _, _ = _write_sources(tmp_path)
    retained = retained_annotations(annotations)

    assert retained["bodyId"].tolist() == [1, 2, 3, 4, 5, 6]


def test_transmitter_policy_uses_explicit_fallbacks(tmp_path):
    _, _, nts = _write_sources(tmp_path)
    selected = selected_neurotransmitters(
        nts,
        [1, 2, 3, 4, 5, 6],
        unresolved_sign=0,
    ).set_index("bodyId")

    assert selected.loc[1, "flyarc_sign"] == 1
    assert selected.loc[2, "flyarc_sign"] == -1
    assert selected.loc[3, "flyarc_sign"] == -1
    assert selected.loc[4, "flyarc_sign"] == -1
    assert selected.loc[5, "flyarc_nt"] == "dopamine"
    assert selected.loc[5, "flyarc_nt_source"] == "predicted_nt>=0.5"
    assert selected.loc[6, "flyarc_nt"] == "acetylcholine"
    assert selected.loc[6, "flyarc_nt_source"] == "celltype_predicted_nt"


def test_prepare_graph_streams_only_retained_endpoints_and_writes_candidate(tmp_path):
    annotations, connectivity, nts = _write_sources(tmp_path)
    output = tmp_path / "graph"

    manifest = prepare_graph(
        annotations=annotations,
        connectivity=connectivity,
        neurotransmitters=nts,
        output=output,
        max_nodes=4,
        expected_neurons=6,
        expected_edges=12,
    )

    nodes = pd.read_parquet(output / "nodes.parquet")
    edges = pd.read_parquet(output / "edges.parquet")

    assert manifest["qualification_status"] == "candidate"
    assert manifest["retained_neurons_scanned"] == 6
    assert manifest["retained_edges_scanned"] == 12
    assert manifest["selected_neurons"] == 4
    assert len(nodes) == 4
    assert set(edges["source"]).issubset(set(nodes["bodyId"]))
    assert set(edges["target"]).issubset(set(nodes["bodyId"]))
    assert set(edges["sign"]).issubset({-1, 0, 1})
    assert (output / "roles.json").read_text() == "{}\n"
