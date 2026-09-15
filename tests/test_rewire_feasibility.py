from __future__ import annotations

import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.rewire_feasibility import maximum_changed_edge_fraction


def _bundle(n: int, edges: list[tuple[int, int]]) -> GraphBundle:
    return GraphBundle(
        nodes=pd.DataFrame({"bodyId": list(range(n))}),
        edges=pd.DataFrame(
            [
                {"source": source, "target": target, "weight": 1.0, "sign": 1}
                for source, target in edges
            ]
        ),
        roles={},
        manifest={"qualification_status": "candidate"},
    )


def test_cycle_can_be_completely_changed() -> None:
    bundle = _bundle(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    report = maximum_changed_edge_fraction(bundle)
    assert report["minimum_unavoidable_original_edges"] == 0
    assert report["maximum_changed_edge_fraction"] == 1.0


def test_complete_loopless_digraph_cannot_change() -> None:
    edges = [(source, target) for source in range(3) for target in range(3) if source != target]
    bundle = _bundle(3, edges)
    report = maximum_changed_edge_fraction(bundle)
    assert report["minimum_unavoidable_original_edges"] == len(edges)
    assert report["maximum_changed_edge_fraction"] == 0.0
