from __future__ import annotations

import pandas as pd
import pytest

from fly_sniff.graph import GraphBundle
from fly_sniff.null_metadata import attach_null_metadata


def _bundle() -> GraphBundle:
    nodes = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "type": ["A", "A", "B", "B"],
            "somaSide": ["L", "R", "left", "right"],
        }
    )
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 4],
            "target": [2, 3, 4, 1],
            "weight": [1.0, 2.0, 3.0, 4.0],
            "sign": [1, 1, -1, -1],
        }
    )
    return GraphBundle(nodes, edges, {}, {"dataset": "male-cns:v1.0"})


def test_metadata_mapping_copies_type_and_normalizes_explicit_side_only() -> None:
    mapped, report = attach_null_metadata(_bundle())
    assert mapped.nodes["cell_type"].tolist() == ["A", "A", "B", "B"]
    assert mapped.nodes["hemisphere_class"].tolist() == ["L", "R", "L", "R"]
    assert report["mapping"]["spatial_bin"] == "unresolved-not-created"
    assert report["selection_used_navigation_performance"] is False


def test_unresolved_side_fails_closed() -> None:
    bundle = _bundle()
    bundle.nodes.loc[0, "somaSide"] = "midline"
    with pytest.raises(ValueError, match="unresolved somaSide"):
        attach_null_metadata(bundle)


def test_unresolved_cell_type_fails_closed() -> None:
    bundle = _bundle()
    bundle.nodes.loc[0, "type"] = None
    with pytest.raises(ValueError, match="unresolved MaleCNS type"):
        attach_null_metadata(bundle)
