from __future__ import annotations

from fly_sniff.zero_shot_artifacts import (
    validate_plume_evidence_envelope,
    validate_structural_evidence_envelope,
)


def _plume() -> dict:
    return {
        "protocol": "zero-shot-experimental-plume-evidence-v1",
        "source_protocol": "experimental-plume-sensory-validation-v3",
        "status": "qualified-for-sensory-evaluation",
        "controller_access": False,
        "navigation_performance_used": False,
        "source_bytes_verified": True,
        "native_time_basis_verified": True,
        "published_cue_reference_comparison_passed": True,
        "physical_pixel_to_fly_mapping_frozen": True,
        "bilateral_sensor_geometry_frozen": True,
        "receipt_sha256": {
            "source": "11" * 32,
            "archive": "22" * 32,
            "cue_reference": "33" * 32,
            "physical_geometry": "44" * 32,
        },
    }


def _structure() -> dict:
    return {
        "protocol": "zero-shot-olfactory-structure-evidence-v1",
        "source_protocol": "olfactory-motion-structural-audit-v1",
        "source_status": "candidate_audit_pass_complete_structural_motif",
        "controller_access": False,
        "navigation_performance_used": False,
        "body_ids_selected_from_navigation": False,
        "external_connectome_body_ids_imported": False,
        "functional_motion_claim_allowed": False,
        "navigation_claim_allowed": False,
        "source_audit_sha256": "55" * 32,
    }


def test_plume_envelope_requires_every_physical_validation_gate() -> None:
    assert validate_plume_evidence_envelope(_plume())["valid"] is True
    broken = _plume()
    broken["physical_pixel_to_fly_mapping_frozen"] = False
    report = validate_plume_evidence_envelope(broken)
    assert report["valid"] is False
    gate = next(row for row in report["gates"] if row["name"] == "physical_coordinate_mapping_frozen")
    assert gate["passed"] is False


def test_structure_envelope_cannot_promote_incomplete_motif() -> None:
    assert validate_structural_evidence_envelope(_structure())["valid"] is True
    broken = _structure()
    broken["source_status"] = "candidate_audit_pass_incomplete_motif"
    report = validate_structural_evidence_envelope(broken)
    assert report["valid"] is False
    gate = next(row for row in report["gates"] if row["name"] == "complete_structural_motif")
    assert gate["passed"] is False


def test_structure_envelope_preserves_functional_claim_boundary() -> None:
    broken = _structure()
    broken["functional_motion_claim_allowed"] = True
    assert validate_structural_evidence_envelope(broken)["valid"] is False
