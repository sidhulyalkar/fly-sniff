import pandas as pd

from fly_sniff.trace_enrichment import audit_enrichment, enrichment_for_column


def test_enrichment_prioritizes_overrepresented_annotation():
    annotations = pd.DataFrame(
        {
            "bodyId": list(range(1, 21)),
            "type": ["common"] * 15 + ["rare"] * 5,
            "class": ["A"] * 20,
        }
    )
    nodes = pd.DataFrame({"bodyId": [1, 16, 17, 18, 19]})

    report = audit_enrichment(annotations, nodes, columns=("type",), top_n=10)
    rows = report["columns"]["type"]["rows"]
    rare = next(row for row in rows if row["label"] == "rare")
    common = next(row for row in rows if row["label"] == "common")

    assert rare["corridor_count"] == 4
    assert rare["background_count"] == 5
    assert rare["fold_enrichment"] > 1.0
    assert rare["p_value"] < common["p_value"]
    assert rare["fdr_bh"] >= rare["p_value"]
    assert report["retained_node_count"] == 5


def test_enrichment_reports_missing_annotation_column_without_guessing():
    annotations = pd.DataFrame({"bodyId": [1, 2], "type": ["A", "B"]})
    result = enrichment_for_column(annotations, {1}, "subclass")

    assert not result["available"]
    assert result["rows"] == []
