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
    assert report["structural_retained_node_count"] == 5
    assert report["annotated_retained_node_count"] == 5
    assert report["unannotated_retained_node_count"] == 0
    assert report["columns"]["type"]["annotated_corridor_size"] == 5


def test_enrichment_excludes_unannotated_structural_ids_from_statistical_sample():
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "type": ["A", "A", "B", "B"],
        }
    )
    nodes = pd.DataFrame(
        {
            "bodyId": [1, 3, 99],
            "annotation_present": [True, True, False],
        }
    )

    report = audit_enrichment(annotations, nodes, columns=("type",), top_n=10)

    assert report["structural_retained_node_count"] == 3
    assert report["annotated_retained_node_count"] == 2
    assert report["unannotated_retained_node_count"] == 1
    assert report["unannotated_body_ids"] == [99]
    assert report["columns"]["type"]["corridor_size"] == 2
    assert report["columns"]["type"]["annotated_corridor_size"] == 2


def test_enrichment_reports_missing_annotation_column_without_guessing():
    annotations = pd.DataFrame({"bodyId": [1, 2], "type": ["A", "B"]})
    result = enrichment_for_column(annotations, {1}, "subclass")

    assert not result["available"]
    assert result["rows"] == []
