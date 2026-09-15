from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.evidence.bootstrap import build_ledger
from fly_sniff.experiment_protocol import ExperimentSpec
from fly_sniff.science_readiness import build_readiness_report


def test_readiness_exposes_real_scientific_and_artifact_blockers(tmp_path: Path) -> None:
    bootstrap = json.loads(Path("configs/evidence_bootstrap_v1.json").read_text())
    ledger = build_ledger(bootstrap)
    ledger_path = ledger.save(tmp_path / "evidence-ledger.json")

    report = build_readiness_report(
        zero_shot_spec=ExperimentSpec.load("configs/zero_shot_latent_wiring_v1.json"),
        physiology_config=json.loads(Path("configs/physiology_calibration_v1.json").read_text()),
        null_config=json.loads(Path("configs/null_factory_v2.json").read_text()),
        necessity_config=json.loads(Path("configs/connectome_necessity_v1.json").read_text()),
        artifact_paths={
            "evidence_ledger": ledger_path,
            "null_factory_protocol": Path("configs/null_factory_v2.json"),
            "connectome_necessity_protocol": Path("configs/connectome_necessity_v1.json"),
        },
    )

    assert report["ready_to_seal_zero_shot"] is False
    assert "temporal_response_scale" in report["physiology_probe_binding"]["unresolved_probes"]
    details = [row["detail"] for row in report["blockers"]]
    assert any("physiology_calibrated_model" in detail for detail in details)
    assert any("experimental_plume_receipt" in detail for detail in details)
    assert any("olfactory_motion_structural_audit" in detail for detail in details)
    assert any("confirmatory null families not ready" in detail for detail in details)


def test_readiness_protocols_are_independently_valid_even_when_artifacts_are_missing() -> None:
    report = build_readiness_report(
        zero_shot_spec=ExperimentSpec.load("configs/zero_shot_latent_wiring_v1.json"),
        physiology_config=json.loads(Path("configs/physiology_calibration_v1.json").read_text()),
        null_config=json.loads(Path("configs/null_factory_v2.json").read_text()),
        necessity_config=json.loads(Path("configs/connectome_necessity_v1.json").read_text()),
        artifact_paths={},
    )
    assert report["protocol_reports"]["zero_shot"]["valid_for_preregistration"] is True
    assert report["protocol_reports"]["physiology_calibration"]["valid_for_preregistration"] is True
    assert report["protocol_reports"]["null_factory"]["valid_for_preregistration"] is True
    assert report["protocol_reports"]["connectome_necessity"]["valid_for_preregistration"] is True
