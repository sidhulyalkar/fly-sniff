from fly_sniff.olfactory_cli import study_status


def test_study_status_is_blocked_without_promoting_unresolved_evidence() -> None:
    report = study_status(".")
    assert report["program_id"] == "olfactory-computation-v0"
    assert report["study_status"] == "blocked"
    assert report["unresolved_count"] == 9
    assert report["readiness"]["O003_flagship"]["confirmatory_allowed"] is False
    assert report["readiness"]["E006_door_ingestion"]["status"] == (
        "source_frozen_ingestion_not_yet_executed"
    )


def test_status_preserves_e001_and_e002_epistemic_state() -> None:
    report = study_status(".")
    o001 = report["readiness"]["O001_geosmin_calibration"]
    assert o001["E001"] == "development_evidence_not_qualified"
    assert o001["E002"] == "blocked_incomplete_body_id_adjudication"
    assert o001["status"] == "blocked_pending_qualified_evidence"
