from __future__ import annotations

import pandas as pd

from fly_sniff.integration_route_audit import build_integration_route_audit


def _annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId": list(range(1, 10)),
            "type": [
                "FB5AB",
                "PFNa",
                "PFNm_a",
                "PFNp_c",
                "hDeltaC",
                "hDeltaG",
                "PFL3",
                "PFL2",
                "xPFNm_a",
            ],
            "instance": [f"cell_{i}" for i in range(1, 10)],
        }
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": [1, 2, 3, 4, 5, 6, 6, 9],
            "target": [5, 5, 5, 5, 6, 7, 8, 5],
            "weight": [12.0, 8.0, 6.0, 4.0, 7.0, 9.0, 5.0, 100.0],
        }
    )


def _config() -> dict:
    return {
        "protocol": "malecns-integration-route-audit-v1",
        "dataset": "male-cns:v1.0",
        "thresholds": [1, 3, 5, 10],
        "population_selectors": {
            "FB5AB": {"column": "type", "exact": "FB5AB"},
            "PFNa_family": {"column": "type", "regex": "^PFNa$"},
            "PFNm_family": {"column": "type", "regex": "^PFNm(?:_|$)"},
            "PFNp_family": {"column": "type", "regex": "^PFNp(?:_|$)"},
            "hDeltaC": {"column": "type", "exact": "hDeltaC"},
            "hDeltaG": {"column": "type", "exact": "hDeltaG"},
            "PFL3": {"column": "type", "exact": "PFL3"},
            "PFL2": {"column": "type", "exact": "PFL2"},
        },
        "direct_predictions": [
            {"source": "PFNm_family", "target": "hDeltaC"},
            {"source": "hDeltaC", "target": "hDeltaG"},
            {"source": "hDeltaG", "target": "PFL3"},
        ],
        "three_hop_predictions": [
            {
                "source": "PFNm_family",
                "via1": "hDeltaC",
                "via2": "hDeltaG",
                "target": "PFL3",
            },
            {
                "source": "FB5AB",
                "via1": "hDeltaC",
                "via2": "hDeltaG",
                "target": "PFL2",
            },
        ],
    }


def test_type_anchored_family_selector_excludes_misleading_prefix() -> None:
    report = build_integration_route_audit(_annotations(), _weights(), _config())
    assert report["populations"]["PFNm_family"]["count"] == 1
    assert report["populations"]["PFNm_family"]["rows"][0]["bodyId"] == 3
    direct = report["direct_predictions"][0]
    assert direct["observed_edge_pairs"] == 1
    assert direct["observed_weight_sum"] == 6.0


def test_three_hop_thresholds_require_every_edge_to_pass() -> None:
    report = build_integration_route_audit(_annotations(), _weights(), _config())
    pf_nm = report["three_hop_predictions"][0]
    sweep = {row["min_weight_each_edge"]: row for row in pf_nm["threshold_sweep"]}
    assert pf_nm["observed_path_count"] == 1
    assert sweep[5.0]["path_count"] == 1
    assert sweep[10.0]["path_count"] == 0

    odor_speed = report["three_hop_predictions"][1]
    odor_sweep = {
        row["min_weight_each_edge"]: row for row in odor_speed["threshold_sweep"]
    }
    assert odor_sweep[5.0]["path_count"] == 1
    assert odor_sweep[10.0]["path_count"] == 0
