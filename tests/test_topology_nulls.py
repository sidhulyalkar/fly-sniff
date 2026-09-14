from __future__ import annotations

import pandas as pd
import pytest

from fly_sniff.graph import GraphBundle
from fly_sniff.topology_nulls import NullFamily, generate_topology_null


def _bundle() -> GraphBundle:
    nodes = pd.DataFrame(
        {
            "bodyId": list(range(1, 9)),
            "somaSide": ["L", "L", "R", "R", "L", "L", "R", "R"],
            "type": ["S", "S", "S", "S", "T", "T", "T", "T"],
        }
    )
    edges = pd.DataFrame(
        {
            "source": [1, 2, 1, 2, 3, 4, 3, 4],
            "target": [5, 6, 7, 8, 5, 6, 7, 8],
            "weight": [11, 12, 13, 14, 15, 16, 17, 18],
            "sign": [1, -1, 1, -1, -1, 1, -1, 1],
        }
    )
    return GraphBundle(nodes=nodes, edges=edges, roles={"readout": [7, 8]})


def _degrees(edges: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    out_degree = edges.groupby("source").size().astype(int).to_dict()
    in_degree = edges.groupby("target").size().astype(int).to_dict()
    return out_degree, in_degree


def _source_attributes(edges: pd.DataFrame) -> dict[int, list[tuple[int, int]]]:
    result: dict[int, list[tuple[int, int]]] = {}
    for source, group in edges.groupby("source"):
        result[int(source)] = sorted(
            (int(row.weight), int(row.sign)) for row in group.itertuples(index=False)
        )
    return result


def _hemisphere_blocks(bundle: GraphBundle) -> dict[tuple[str, str], int]:
    side = bundle.nodes.set_index("bodyId")["somaSide"].to_dict()
    counts: dict[tuple[str, str], int] = {}
    for row in bundle.edges.itertuples(index=False):
        key = (str(side[int(row.source)]), str(side[int(row.target)]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_directed_null_preserves_degree_and_presynaptic_attributes() -> None:
    source = _bundle()
    result = generate_topology_null(
        source,
        family=NullFamily.DIRECTED_DEGREE,
        seed=17,
        swaps_per_edge=1,
    )

    assert _degrees(result.bundle.edges) == _degrees(source.edges)
    assert _source_attributes(result.bundle.edges) == _source_attributes(source.edges)
    assert result.receipt.exact_directed_degree_preserved
    assert result.receipt.source_weight_sign_multiset_preserved
    assert result.receipt.complete_requested_swap_budget
    assert not result.receipt.stationarity_claimed
    assert result.receipt.output_graph_sha256 != ""


def test_hemisphere_null_preserves_exact_source_target_side_blocks() -> None:
    source = _bundle()
    result = generate_topology_null(
        source,
        family=NullFamily.HEMISPHERE,
        seed=9,
        swaps_per_edge=1,
    )

    assert result.receipt.constraint_columns == ("somaSide",)
    assert result.receipt.constraint_block_counts_preserved
    assert _hemisphere_blocks(result.bundle) == _hemisphere_blocks(source)
    assert _degrees(result.bundle.edges) == _degrees(source.edges)


def test_cell_type_null_uses_type_metadata_without_role_mutation() -> None:
    source = _bundle()
    result = generate_topology_null(
        source,
        family=NullFamily.CELL_TYPE,
        seed=31,
        swaps_per_edge=1,
    )

    assert result.receipt.constraint_columns == ("type",)
    assert result.bundle.roles == source.roles
    assert result.bundle.nodes.equals(source.nodes)
    assert result.receipt.constraint_block_counts_preserved


def test_constrained_null_requires_complete_metadata() -> None:
    source = _bundle()
    source.nodes.loc[source.nodes.bodyId == 1, "somaSide"] = None
    with pytest.raises(ValueError, match="lacks complete metadata"):
        generate_topology_null(
            source,
            family=NullFamily.CONSTRAINED,
            seed=1,
            swaps_per_edge=1,
            constraint_columns=("somaSide",),
        )


def test_null_generation_fails_closed_when_swap_budget_cannot_complete() -> None:
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4]})
    edges = pd.DataFrame(
        {
            "source": [1, 1, 2, 2],
            "target": [3, 4, 3, 4],
            "weight": [1, 1, 1, 1],
            "sign": [1, 1, 1, 1],
        }
    )
    source = GraphBundle(nodes=nodes, edges=edges, roles={})
    with pytest.raises(RuntimeError, match="exhausted its preregistered attempt budget"):
        generate_topology_null(
            source,
            family=NullFamily.DIRECTED_DEGREE,
            seed=3,
            swaps_per_edge=1,
            max_attempt_multiplier=1,
        )
