from __future__ import annotations

import copy

import pytest

from fly_sniff.integration_topography_audit import build_integration_topography_audit


def _audit() -> dict:
    return {
        "populations": {
            "SRC": {
                "count": 2,
                "rows": [
                    {"bodyId": 1, "type": "SRC", "instance": "SRC_L2_C1", "somaSide": "L"},
                    {"bodyId": 2, "type": "SRC", "instance": "SRC", "somaSide": "R"},
                ],
            },
            "PFL3": {
                "count": 2,
                "rows": [
                    {
                        "bodyId": 3,
                        "type": "PFL3",
                        "instance": "PFL3(PB12c)_R2_C4",
                        "somaSide": "L",
                    },
                    {
                        "bodyId": 4,
                        "type": "PFL3",
                        "instance": "PFL3(PB12c)_L5_C7",
                        "somaSide": "R",
                    },
                ],
            },
        },
        "direct_predictions": [
            {
                "source": "SRC",
                "target": "PFL3",
                "observed_edge_pairs": 2,
                "observed_weight_sum": 15.0,
                "edges": [
                    {
                        "source": 1,
                        "target": 3,
                        "source_instance": "SRC_L2_C1",
                        "target_instance": "PFL3(PB12c)_R2_C4",
                        "source_somaSide": "L",
                        "target_somaSide": "L",
                        "weight": 10.0,
                    },
                    {
                        "source": 2,
                        "target": 4,
                        "source_instance": "SRC",
                        "target_instance": "PFL3(PB12c)_L5_C7",
                        "source_somaSide": "R",
                        "target_somaSide": "R",
                        "weight": 5.0,
                    },
                ],
            }
        ],
    }


def _config() -> dict:
    return {
        "protocol": "topography-test",
        "dataset": "male-cns:v1.0",
        "thresholds": [1, 10],
        "population_instance_rules": {
            "SRC": {"column_regex": r"_C(\d+)(?:_|$)"},
            "PFL3": {
                "column_regex": r"_C(\d+)(?:_|$)",
                "side_label_regex": r"\)_([LR])\d+_C",
            },
        },
        "edge_families": [{"source": "SRC", "target": "PFL3"}],
        "interpretation_rules": ["instance side is descriptive only"],
        "claim_boundary": "test",
    }


def test_topography_parses_columns_without_imputing_missing_instances() -> None:
    report = build_integration_topography_audit(_audit(), _config())

    assert report["directional_role_status"] == "unresolved"
    assert report["populations"]["SRC"]["column_parsed_count"] == 1
    assert report["populations"]["SRC"]["unparsed_column_body_ids"] == [2]
    assert report["populations"]["PFL3"]["columns"] == [4, 7]


def test_pfl3_side_label_is_reported_from_instance_not_soma_side() -> None:
    report = build_integration_topography_audit(_audit(), _config())
    population = report["populations"]["PFL3"]

    assert population["body_side_labels"] == {"3": "R", "4": "L"}
    assert population["side_label_counts"] == {"L": 1, "R": 1}

    family = report["edge_families"][0]
    side = family["threshold_sweep"][0]["target_instance_side"]
    assert side["target_body_ids"] == {"L": [4], "R": [3]}


def test_topography_keeps_fixed_threshold_sweep() -> None:
    report = build_integration_topography_audit(_audit(), _config())
    sweep = report["edge_families"][0]["threshold_sweep"]

    assert [row["min_weight"] for row in sweep] == [1.0, 10.0]
    assert [row["edge_pairs"] for row in sweep] == [2, 1]
    assert sweep[0]["both_column_parse_fraction"] == 0.5
    assert sweep[1]["both_column_parse_fraction"] == 1.0


def test_topography_rejects_truncated_direct_edge_detail() -> None:
    audit = copy.deepcopy(_audit())
    audit["direct_predictions"][0]["observed_edge_pairs"] = 3

    with pytest.raises(ValueError, match="is truncated"):
        build_integration_topography_audit(audit, _config())
