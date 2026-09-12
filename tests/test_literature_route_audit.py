import pandas as pd

from fly_sniff.literature_route_audit import build_route_audit


def _fixture():
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "type": [
                "ORN_DA1",
                "olf_bilateral",
                "FB5AB",
                "hDeltaC",
                "PFL3",
                "PFL2",
                "DNa02",
                "DNa03",
                "LAL010",
                "MBON32",
            ],
            "instance": [
                "ORN_DA1_R",
                "olf_bilateral_R",
                "FB5AB_L",
                "hDeltaC_01",
                "PFL3_R",
                "PFL2_R",
                "DNa02_R",
                "DNa03_R",
                "LAL010_R",
                "MBON32_R",
            ],
            "class": [
                "olfactory",
                "olfactory",
                "CX",
                "CX",
                "CX",
                "CX",
                None,
                None,
                None,
                "MBON",
            ],
            "hemibrainType": [
                "ORN_DA1",
                "olf_bilateral",
                "FB5AB",
                "hDeltaC",
                "PFL3",
                "PFL2",
                "DNa02",
                "DNa03",
                "LAL010",
                "MBON32",
            ],
            "flywireType": [
                "ORN_DA1",
                "olf_bilateral",
                "FB5AB",
                "hDeltaC",
                "PFL3",
                "PFL2",
                "DNa02",
                "DNa03",
                "LAL010",
                "MBON32",
            ],
            "somaSide": ["R", "R", "L", "L", "R", "R", "R", "R", "R", "R"],
            "receptorType": [
                "Or42b",
                "Ir64a",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        }
    )
    weights = pd.DataFrame(
        {
            "source": [3, 4, 5, 5, 5, 8, 9, 6, 6, 10],
            "target": [4, 5, 7, 8, 9, 7, 7, 8, 9, 7],
            "weight": [4, 7, 2, 8, 6, 9, 11, 5, 3, 12],
        }
    )
    config = {
        "protocol": "test-route-audit",
        "dataset": "toy",
        "thresholds": [1, 5, 10],
        "populations": [
            "FB5AB",
            "hDeltaC",
            "PFL3",
            "PFL2",
            "DNa02",
            "DNa03",
            "LAL010",
            "MBON32",
        ],
        "direct_predictions": [
            {"source": "FB5AB", "target": "hDeltaC"},
            {"source": "hDeltaC", "target": "PFL3"},
            {"source": "PFL3", "target": "DNa02"},
        ],
        "two_hop_predictions": [
            {"source": "PFL3", "via": "DNa03", "target": "DNa02"},
            {"source": "PFL3", "via": "LAL010", "target": "DNa02"},
        ],
    }
    return annotations, weights, config


def test_route_audit_uses_exact_named_populations_and_preserves_body_edges():
    annotations, weights, config = _fixture()
    report = build_route_audit(annotations, weights, config)

    assert report["populations"]["FB5AB"]["count"] == 1
    assert report["populations"]["DNa02"]["count"] == 1

    direct = {
        (row["source"], row["target"]): row for row in report["direct_predictions"]
    }
    assert direct[("FB5AB", "hDeltaC")]["observed_weight_sum"] == 4.0
    assert direct[("hDeltaC", "PFL3")]["observed_weight_sum"] == 7.0
    assert direct[("PFL3", "DNa02")]["observed_weight_sum"] == 2.0
    assert direct[("PFL3", "DNa02")]["threshold_sweep"][1]["edge_pairs"] == 0


def test_route_audit_reports_preregistered_two_hop_support():
    annotations, weights, config = _fixture()
    report = build_route_audit(annotations, weights, config)

    paths = {
        (row["source"], row["via"], row["target"]): row
        for row in report["two_hop_predictions"]
    }
    via_dna03 = paths[("PFL3", "DNa03", "DNa02")]
    via_lal010 = paths[("PFL3", "LAL010", "DNa02")]

    assert via_dna03["observed_path_count"] == 1
    assert via_dna03["threshold_sweep"][1]["path_count"] == 1
    assert via_lal010["observed_path_count"] == 1
    assert via_lal010["threshold_sweep"][1]["path_count"] == 1
    assert via_lal010["threshold_sweep"][2]["path_count"] == 0


def test_source_population_audit_exposes_legacy_regex_semantics():
    annotations, weights, config = _fixture()
    report = build_route_audit(annotations, weights, config)
    source = report["source_population_audit"]

    assert source["olfactory_class_count"] == 2
    assert source["orn_type_prefix_count"] == 1
    assert source["olfactory_not_orn_count"] == 1
    assert source["receptor_type_nonnull_in_olfactory"] == 2
    assert source["legacy_regex_counts"] == {"ORN": 1, "^Or": 1, "^Ir": 0}
    assert source["legacy_regex_overlaps"]["ORN_and_^Or"] == 1
