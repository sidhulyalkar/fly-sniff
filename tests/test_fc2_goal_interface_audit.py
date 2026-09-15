from fly_sniff.fc2_goal_interface_audit import build_report


def test_fc2_interface_audit_smoke() -> None:
    pops = {
        "FC2A": {"count": 1, "rows": [{"bodyId": 1, "instance": "FC2A_R1_C1"}]},
        "FC2B": {"count": 1, "rows": [{"bodyId": 2, "instance": "FC2B_R1_C2"}]},
        "FC2C": {"count": 1, "rows": [{"bodyId": 3, "instance": "FC2C_R1_C3"}]},
        "PFL3": {"count": 1, "rows": [{"bodyId": 10, "instance": "PFL3_R1_C1"}]},
    }
    direct = []
    for name, body_id in (("FC2A", 1), ("FC2B", 2), ("FC2C", 3)):
        direct.append({
            "source": name,
            "target": "PFL3",
            "observed_edge_pairs": 1,
            "observed_weight_sum": 12.0,
            "edges": [{"source": body_id, "target": 10, "weight": 12.0}],
        })
    audit = {
        "protocol": "malecns-goal-relay-audit-v1",
        "dataset": "male-cns:v1.0",
        "populations": pops,
        "direct_predictions": direct,
    }
    config = {
        "protocol": "malecns-fc2-goal-interface-audit-v1",
        "dataset": "male-cns:v1.0",
        "thresholds": [1, 10],
        "source_populations": ["FC2A", "FC2B", "FC2C"],
        "target_population": "PFL3",
        "instance_column_regex": "_C([0-9]+)",
    }
    signs = {
        "dataset": "male-cns:v1.0",
        "predictions": {
            "FC2A": {"neuron_count": 1, "modeled_sign": 1},
            "FC2B": {"neuron_count": 1, "modeled_sign": 1},
            "FC2C": {"neuron_count": 1, "modeled_sign": 1},
        },
    }
    report = build_report(audit, config, signs)
    assert report["phase_mapping_status"] == "unresolved"
    assert report["interfaces"]["FC2A"]["threshold_reports"]["10"]["target_coverage"] == "1/1"
    assert report["populations"]["FC2A"]["instance_columns"]["parse_fraction"] == 1.0
