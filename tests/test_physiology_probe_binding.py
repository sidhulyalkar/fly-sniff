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
    assert report["unresolved_probes"] == ["temporal_response_scale"]
    assert report["ready_to_define_fit_objective"] is False

    by_id = {row["id"]: row for row in report["probe_reports"]}
    assert by_id["PFN_airflow_tuning_geometry"]["status"] == "sealed"
    assert by_id["PFN_mirrored_directional_response"]["status"] == "sealed"
    assert by_id["odor_context_gating"]["status"] == "sealed"
    assert by_id["PFL3_lateralized_steering_relationship"]["status"] == "sealed"
    assert by_id["DNa02_turn_relationship"]["status"] == "sealed"
    assert by_id["temporal_response_scale"]["status"] == "unresolved"
    assert by_id["temporal_response_scale"]["rejected_candidate_facts"]


def test_modeled_state_cannot_satisfy_physiology_probe() -> None:
    calibration = json.loads(CALIBRATION.read_text())
    bootstrap = json.loads(BOOTSTRAP.read_text())
    steering = next(
        row for row in bootstrap["records"] if row["entity"]["name"] == "steering-scaffold-model"
    )
    steering["facts"][0]["metadata"] = {
        "probe_ids": ["temporal_response_scale"],
        "calibration_eligible": True,
    }
    ledger = build_ledger(bootstrap)
    report = bind_probe_authorities(calibration, ledger)
    temporal = next(row for row in report["probe_reports"] if row["id"] == "temporal_response_scale")
    assert temporal["status"] == "unresolved"
    assert any(
        row["reason"] == "evidence_class_not_permitted"
        for row in temporal["rejected_candidate_facts"]
    )
