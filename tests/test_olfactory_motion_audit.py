from __future__ import annotations

import pandas as pd

from fly_sniff.olfactory_motion_audit import build_olfactory_motion_audit


def _annotations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"bodyId": 1, "type": "il3LN6", "instance": "il3LN6_L", "class": "AL local", "subclass": "", "somaSide": "L", "rootSide": "L"},
            {"bodyId": 2, "type": "il3LN6", "instance": "il3LN6_R", "class": "AL local", "subclass": "", "somaSide": "R", "rootSide": "R"},
            {"bodyId": 10, "type": "DA1_lPN", "instance": "DA1_lPN_L", "class": "ALPN", "subclass": "uniglomerular", "somaSide": "L", "rootSide": "L"},
            {"bodyId": 11, "type": "DA1_lPN", "instance": "DA1_lPN_R", "class": "ALPN", "subclass": "uniglomerular", "somaSide": "R", "rootSide": "R"},
            {"bodyId": 20, "type": "Or67d_ORN", "instance": "Or67d_ORN_L", "class": "ORN", "subclass": "DA1", "somaSide": "L", "rootSide": "L"},
            {"bodyId": 21, "type": "Or67d_ORN", "instance": "Or67d_ORN_R", "class": "ORN", "subclass": "DA1", "somaSide": "R", "rootSide": "R"},
            {"bodyId": 99, "type": "unrelated", "instance": "unrelated_R", "class": "other", "subclass": "", "somaSide": "R", "rootSide": "R"},
        ]
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"body_pre": 20, "body_post": 1, "weight": 5},
            {"body_pre": 21, "body_post": 2, "weight": 7},
            {"body_pre": 1, "body_post": 10, "weight": 9},
            {"body_pre": 2, "body_post": 11, "weight": 11},
            {"body_pre": 99, "body_post": 1, "weight": 100},
        ]
    )


def test_complete_motif_uses_only_annotation_resolved_body_ids(tmp_path):
    annotations = tmp_path / "annotations.feather"
    weights = tmp_path / "weights.feather"
    _annotations().to_feather(annotations)
    _weights().to_feather(weights)

    report, edges = build_olfactory_motion_audit(annotations, weights)

    assert report["status"] == "candidate_audit_pass_complete_structural_motif"
    assert report["candidate_counts"] == {"il3LN6": 2, "DA1_lPN": 2, "DA1_ORN": 2}
    assert report["selected_body_count"] == 6
    assert report["selected_edge_count"] == 4
    assert set(edges["source"]) == {1, 2, 20, 21}
    assert 99 not in set(edges["source"])
    assert report["controller_access"] is False
    assert report["navigation_performance_used"] is False
    assert report["functional_motion_claim_allowed"] is False
    assert report["navigation_claim_allowed"] is False


def test_family_connectivity_is_aggregated_only_after_body_selection(tmp_path):
    annotations = tmp_path / "annotations.feather"
    weights = tmp_path / "weights.feather"
    _annotations().to_feather(annotations)
    _weights().to_feather(weights)

    report, _ = build_olfactory_motion_audit(annotations, weights)
    summary = {
        (row["source_family"], row["target_family"]): (row["edge_count"], row["aggregate_weight"])
        for row in report["family_connectivity"]
    }
    assert summary[("DA1_ORN", "il3LN6")] == (2, 12)
    assert summary[("il3LN6", "DA1_lPN")] == (2, 20)


def test_missing_il3ln6_blocks_instead_of_substituting_neighbor(tmp_path):
    annotations = tmp_path / "annotations.feather"
    frame = _annotations()
    frame = frame[frame["type"] != "il3LN6"].copy()
    frame.loc[len(frame)] = {
        "bodyId": 50,
        "type": "il3LN5",
        "instance": "il3LN5_R",
        "class": "AL local",
        "subclass": "",
        "somaSide": "R",
        "rootSide": "R",
    }
    frame.to_feather(annotations)

    report, _ = build_olfactory_motion_audit(annotations)

    assert report["status"] == "blocked_missing_required_candidate_family"
    assert "il3LN6" in report["missing_required_candidate_families"]
    assert report["candidate_counts"]["il3LN6"] == 0
    assert all(row["bodyId"] != 50 for row in report["candidates"]["il3LN6"])


def test_missing_da1_orn_keeps_candidate_audit_but_marks_motif_incomplete(tmp_path):
    annotations = tmp_path / "annotations.feather"
    frame = _annotations()
    frame = frame[~frame["type"].str.contains("Or67d")].copy()
    frame.to_feather(annotations)

    report, _ = build_olfactory_motion_audit(annotations)

    assert report["status"] == "candidate_audit_pass_incomplete_motif"
    assert report["missing_required_candidate_families"] == []
    assert report["missing_complete_motif_families"] == ["DA1_ORN"]


def test_report_records_exact_selected_ids_from_input_table(tmp_path):
    annotations = tmp_path / "annotations.feather"
    frame = _annotations()
    frame.loc[frame["type"] == "il3LN6", "bodyId"] = [7001, 7002]
    frame.to_feather(annotations)

    report, _ = build_olfactory_motion_audit(annotations)

    resolved = {row["bodyId"] for row in report["candidates"]["il3LN6"]}
    assert resolved == {7001, 7002}


def test_anchored_patterns_match_instance_independently_of_type_column(tmp_path):
    annotations = tmp_path / "annotations.feather"
    frame = _annotations()
    frame.loc[frame["bodyId"] == 1, "type"] = ""
    frame.loc[frame["bodyId"] == 10, "type"] = ""
    frame.to_feather(annotations)

    report, _ = build_olfactory_motion_audit(annotations)

    assert report["candidate_counts"]["il3LN6"] == 2
    assert report["candidate_counts"]["DA1_lPN"] == 2
    il3 = {row["bodyId"]: row["matched_patterns"] for row in report["candidates"]["il3LN6"]}
    da1 = {row["bodyId"]: row["matched_patterns"] for row in report["candidates"]["DA1_lPN"]}
    assert any(match["column"] == "instance" for match in il3[1])
    assert any(match["column"] == "instance" for match in da1[10])
