from __future__ import annotations

import pytest

from fly_sniff.heading_topography_review import review_heading_topography


def test_heading_topography_review_tracks_pb_alignment() -> None:
    audit = {
        "protocol": "malecns-heading-route-audit-v1",
        "dataset": "male-cns:v1.0",
        "thresholds": [1, 5, 10],
        "direct_predictions": [
            {
                "source": "EPG",
                "target": "PFL3",
                "edges": [
                    {
                        "source": 1,
                        "target": 10,
                        "source_instance": "EPG(PB08)_R4",
                        "target_instance": "PFL3(PB12c)_R4_C4",
                        "weight": 12.0,
                    },
                    {
                        "source": 2,
                        "target": 11,
                        "source_instance": "EPG(PB08)_L3",
                        "target_instance": "PFL3(PB12c)_L3_C5",
                        "weight": 7.0,
                    },
                    {
                        "source": 3,
                        "target": 12,
                        "source_instance": "EPG(PB08)_R4",
                        "target_instance": "PFL3(PB12c)_R3_C5",
                        "weight": 14.0,
                    },
                ],
            }
        ],
    }
    report = review_heading_topography(audit)
    threshold_10 = report["threshold_reports"]["10"]
    assert threshold_10["retained_parsed_edges"] == 2
    assert threshold_10["matching_pb_edges"] == 1
    assert threshold_10["matching_pb_weight_fraction"] == 12.0 / 26.0
    assert len(threshold_10["unmatched_edges"]) == 1
    assert "physical heading angle" in report["interpretation"]


def test_heading_topography_review_rejects_protocol_mismatch() -> None:
    with pytest.raises(ValueError, match="unexpected heading route audit protocol"):
        review_heading_topography({"protocol": "not-heading"})
