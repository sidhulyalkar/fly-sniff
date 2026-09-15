from __future__ import annotations

import pandas as pd
import pytest

from fly_sniff.graph import GraphBundle
from fly_sniff.null_generation import (
    audit_degree_null,
    exact_max_distance_degree_rewire,
)


def _cycle_bundle(n: int = 6) -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": list(range(n))})
    edges = pd.DataFrame(
        {
            "source": list(range(n)),
            "target": [(index + 1) % n for index in range(n)],
            "weight": [float(index + 1) for index in range(n)],
            "sign": [1 if index % 2 == 0 else -1 for index in range(n)],
        }
    )
    return GraphBundle(nodes, edges, {"all": list(range(n))}, {"protocol": "synthetic"})


def test_exact_degree_null_reaches_full_distance_when_feasible() -> None:
    intact = _cycle_bundle()
    first = exact_max_distance_degree_rewire(
        intact,
        seed=42,
        minimum_changed_edge_fraction=0.80,
    )
    second = exact_max_distance_degree_rewire(
        intact,
        seed=42,
        minimum_changed_edge_fraction=0.80,
    )
    report = audit_degree_null(intact, first)
    assert report["passed"] is True
    assert report["changed_edge_fraction"] == 1.0
    assert first.edges.equals(second.edges)
    assert sorted(first.edges.weight.astype(float)) == sorted(intact.edges.weight.astype(float))
    assert sorted(first.edges.sign.astype(int)) == sorted(intact.edges.sign.astype(int))
    assert first.manifest["rewire"]["selection_used_behavior_performance"] is False


def test_exact_degree_null_fails_closed_when_distance_floor_is_impossible() -> None:
    nodes = pd.DataFrame({"bodyId": [0, 1]})
    edges = pd.DataFrame(
        {
            "source": [0, 1],
            "target": [1, 0],
            "weight": [2.0, 3.0],
            "sign": [1, -1],
        }
    )
    bundle = GraphBundle(nodes, edges, {}, {"protocol": "synthetic"})
    with pytest.raises(ValueError, match="cannot satisfy frozen distance floor"):
        exact_max_distance_degree_rewire(
            bundle,
            seed=7,
            minimum_changed_edge_fraction=0.80,
        )


def test_exact_degree_null_refuses_unbounded_candidate_expansion() -> None:
    intact = _cycle_bundle(8)
    with pytest.raises(ValueError, match="candidate graph is too large"):
        exact_max_distance_degree_rewire(
            intact,
            seed=1,
            minimum_changed_edge_fraction=0.0,
            max_candidate_pairs=10,
        )
