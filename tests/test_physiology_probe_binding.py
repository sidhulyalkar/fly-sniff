from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.evidence.bootstrap import build_ledger
from fly_sniff.physiology_probe_binding import bind_probe_authorities

CALIBRATION = Path("configs/physiology_calibration_v1.json")
BOOTSTRAP = Path("configs/evidence_bootstrap_v1.json")


def test_published_evidence_binds_five_probes_but_not_temporal_scale() -> None:
    calibration = json.loads(CALIBRATION.read_text())
    ledger = build_ledger(json.loads(BOOTSTRAP.read_text()))
    report = bind_probe_authorities(calibration, ledger)

    assert report["sealed_probe_count"] == 5
    assert report["probe_count"] == 6
    assert report["ready_to_define_fit_objective"] is False
    assert report["unresolved_probes"] == ["temporal_response_scale"]


def test_binding_report_retains_only_independent_physiology_authorities() -> None:
    calibration = json.loads(CALIBRATION.read_text())
    ledger = build_ledger(json.loads(BOOTSTRAP.read_text()))
    report = bind_probe_authorities(calibration, ledger)

    bound = {
        row["probe_id"]: row
        for row in report["probe_bindings"]
        if row["authority_status"] == "sealed"
    }
    assert "PFN_airflow_tuning_geometry" in bound
    assert "CROSS_DATASET_PRIOR" in bound["PFN_airflow_tuning_geometry"]["evidence_classes"]
    assert "PFL3_lateralized_steering_relationship" in bound
    assert "CROSS_DATASET_PRIOR" in bound["PFL3_lateralized_steering_relationship"]["evidence_classes"]
    assert "MODELED_STATE" not in bound["PFL3_lateralized_steering_relationship"]["evidence_classes"]
