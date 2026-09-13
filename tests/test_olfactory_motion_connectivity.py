from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from fly_sniff.olfactory_motion_audit import build_olfactory_motion_audit
from fly_sniff.olfactory_motion_connectivity import (
    _canonical_sha,
    build_connectivity_audit,
    load_connectivity_config,
)


def _annotations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"bodyId": 1, "type": "il3LN6", "instance": "il3LN6_L", "class": "ALLN", "subclass": None, "somaSide": "L", "rootSide": None},
            {"bodyId": 2, "type": "il3LN6", "instance": "il3LN6_R", "class": "ALLN", "subclass": None, "somaSide": "R", "rootSide": None},
            {"bodyId": 10, "type": "DA1_lPN", "instance": "DA1_lPN_L", "class": "ALPN", "subclass": None, "somaSide": "L", "rootSide": None},
            {"bodyId": 11, "type": "DA1_lPN", "instance": "DA1_lPN_R", "class": "ALPN", "subclass": None, "somaSide": "R", "rootSide": None},
            {"bodyId": 20, "type": "ORN_DA1", "instance": "ORN_DA1_L", "class": "olfactory", "subclass": None, "somaSide": None, "rootSide": "L"},
            {"bodyId": 21, "type": "ORN_DA1", "instance": "ORN_DA1_R", "class": "olfactory", "subclass": None, "somaSide": None, "rootSide": "R"},
            {"bodyId": 22, "type": "ORN_DA1", "instance": "ORN_DA1_unknown", "class": "olfactory", "subclass": None, "somaSide": None, "rootSide": "unknown"},
            {"bodyId": 99, "type": "unrelated", "instance": "unrelated_R", "class": "other", "subclass": None, "somaSide": "R", "rootSide": "R"},
        ]
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"body_pre": 20, "body_post": 10, "weight": 3},
            {"body_pre": 21, "body_post": 11, "weight": 4},
            {"body_pre": 20, "body_post": 2, "weight": 5},
            {"body_pre": 21, "body_post": 1, "weight": 7},
            {"body_pre": 22, "body_post": 1, "weight": 2},
            {"body_pre": 1, "body_post": 10, "weight": 11},
            {"body_pre": 2, "body_post": 11, "weight": 13},
            {"body_pre": 99, "body_post": 1, "weight": 1000},
        ]
    )


def _frozen_fixture(tmp_path):
    annotations = tmp_path / "annotations.feather"
    weights = tmp_path / "weights.feather"
    gate_path = tmp_path / "gate-a.json"
    config_path = tmp_path / "connectivity.json"
    _annotations().to_feather(annotations)
    _weights().to_feather(weights)
    gate_a, _ = build_olfactory_motion_audit(annotations)
    gate_path.write_text(json.dumps(gate_a, indent=2, sort_keys=True) + "\n")

    config = copy.deepcopy(load_connectivity_config())
    config["gate_a"]["report_sha256"] = gate_a["report_sha256"]
    config["gate_a"]["annotation_sha256"] = gate_a["annotation_sha256"]
    config["gate_a"]["selected_body_count"] = gate_a["selected_body_count"]
    config["gate_a"]["candidate_counts"] = gate_a["candidate_counts"]
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return annotations, weights, gate_path, config_path, gate_a


def test_gate_b_uses_exact_frozen_candidates_and_prespecified_paths(tmp_path):
    annotations, weights, gate_path, config_path, gate_a = _frozen_fixture(tmp_path)
    report, edges = build_connectivity_audit(gate_path, annotations, weights, config_path)

    assert report["gate_a_report_sha256"] == gate_a["report_sha256"]
    assert report["status"] == "candidate_structural_motif_present"
    assert report["all_required_structural_paths_present"] is True
    assert report["selected_body_count"] == 7
    assert 99 not in set(edges["source"]) | set(edges["target"])
    paths = {(row["source"], row["target"]): row for row in report["prespecified_paths"]}
    assert paths[("DA1_ORN", "DA1_lPN")]["edge_count"] == 2
    assert paths[("DA1_ORN", "il3LN6")]["edge_count"] == 3
    assert paths[("il3LN6", "DA1_lPN")]["edge_count"] == 2
    assert report["neurotransmitter_sign_inferred"] is False
    assert report["delay_inferred"] is False
    assert report["functional_motion_claim_allowed"] is False
    assert report["navigation_claim_allowed"] is False


def test_laterality_keeps_unknown_orn_side_unknown(tmp_path):
    annotations, weights, gate_path, config_path, _ = _frozen_fixture(tmp_path)
    report, edges = build_connectivity_audit(gate_path, annotations, weights, config_path)

    unknown_edge = edges[(edges["source"] == 22) & (edges["target"] == 1)].iloc[0]
    assert unknown_edge["source_side"] == "unknown"
    assert unknown_edge["source_side_basis"] == "rootSide"
    assert unknown_edge["target_side_basis"] == "somaSide"
    assert unknown_edge["side_relation"] == "unknown"
    assert report["laterality_contract"]["relations"] == [
        "same_annotation_side",
        "opposite_annotation_side",
        "unknown",
    ]


def test_gate_b_rejects_tampered_gate_a_without_rehash(tmp_path):
    annotations, weights, gate_path, config_path, _ = _frozen_fixture(tmp_path)
    gate = json.loads(gate_path.read_text())
    gate["candidates"]["il3LN6"][0]["bodyId"] = 777
    gate_path.write_text(json.dumps(gate))

    with pytest.raises(ValueError, match="self-hash mismatch"):
        build_connectivity_audit(gate_path, annotations, weights, config_path)


def test_gate_b_rejects_rehashed_gate_a_candidate_injection(tmp_path):
    annotations, weights, gate_path, config_path, _ = _frozen_fixture(tmp_path)
    gate = json.loads(gate_path.read_text())
    injected = dict(gate["candidates"]["il3LN6"][0])
    injected["bodyId"] = 99
    gate["candidates"]["il3LN6"].append(injected)
    gate["candidate_counts"]["il3LN6"] += 1
    gate["selected_body_count"] += 1
    payload = dict(gate)
    payload.pop("report_sha256")
    gate["report_sha256"] = _canonical_sha(payload)
    gate_path.write_text(json.dumps(gate))

    with pytest.raises(ValueError, match="differs from frozen Gate B contract"):
        build_connectivity_audit(gate_path, annotations, weights, config_path)


def test_gate_b_rejects_different_annotation_bytes(tmp_path):
    annotations, weights, gate_path, config_path, _ = _frozen_fixture(tmp_path)
    frame = pd.read_feather(annotations)
    frame.loc[frame["bodyId"] == 10, "instance"] = "DA1_lPN_CHANGED"
    frame.to_feather(annotations)

    with pytest.raises(ValueError, match="annotation bytes"):
        build_connectivity_audit(gate_path, annotations, weights, config_path)


def test_missing_required_structural_path_is_reported_not_repaired(tmp_path):
    annotations, weights, gate_path, config_path, _ = _frozen_fixture(tmp_path)
    frame = pd.read_feather(weights)
    frame = frame[~((frame["body_pre"].isin([1, 2])) & (frame["body_post"].isin([10, 11])))]
    frame.to_feather(weights)

    report, _ = build_connectivity_audit(gate_path, annotations, weights, config_path)
    assert report["status"] == "candidate_structural_motif_incomplete"
    assert report["all_required_structural_paths_present"] is False
    path = next(
        row for row in report["prespecified_paths"]
        if row["source"] == "il3LN6" and row["target"] == "DA1_lPN"
    )
    assert path["present"] is False
    assert path["edge_count"] == 0
