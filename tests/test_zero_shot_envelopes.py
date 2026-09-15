from __future__ import annotations

import pytest

from fly_sniff.zero_shot_envelopes import build_plume_envelope, build_structural_envelope

SHA = "ab" * 32


def test_complete_structural_audit_can_be_wrapped_without_promoting_function() -> None:
    audit = {
        "protocol": "olfactory-motion-structural-audit-v1",
        "dataset": "male-cns:v1.0",
        "status": "candidate_audit_pass_complete_structural_motif",
        "weights_sha256": SHA,
        "annotation_sha256": "cd" * 32,
        "selected_body_count": 6,
        "selected_edge_count": 4,
        "controller_access": False,
        "navigation_performance_used": False,
        "functional_motion_claim_allowed": False,
        "navigation_claim_allowed": False,
    }
    envelope = build_structural_envelope(audit, audit_sha256=SHA)
    assert envelope["source_status"] == "candidate_audit_pass_complete_structural_motif"
    assert envelope["functional_motion_claim_allowed"] is False
    assert envelope["navigation_claim_allowed"] is False


def test_incomplete_structural_audit_is_rejected() -> None:
    audit = {
        "protocol": "olfactory-motion-structural-audit-v1",
        "dataset": "male-cns:v1.0",
        "status": "candidate_audit_pass_incomplete_motif",
    }
    with pytest.raises(ValueError, match="not complete"):
        build_structural_envelope(audit, audit_sha256=SHA)


def test_plume_envelope_requires_every_independent_receipt() -> None:
    source = {
        "protocol": "experimental-plume-sensory-validation-v3",
        "source_bytes_verified": True,
    }
    archive = {
        "protocol": "experimental-plume-archive-inspection-v3",
        "status": "archive_qualified",
        "controller_access": False,
        "navigation_performance_used": False,
        "native_time_basis": {"kind": "fixed_rate", "fps": 15.0},
    }
    cue = {
        "protocol": "experimental-plume-cue-reference-comparison-v1",
        "status": "passed_reference_comparison",
        "passed": True,
        "controller_access": False,
        "navigation_performance_used": False,
    }
    geometry = {
        "protocol": "experimental-plume-physical-geometry-v1",
        "status": "qualified_physical_geometry",
        "pixel_to_fly_mapping_frozen": True,
        "bilateral_sensor_geometry_frozen": True,
        "controller_access": False,
        "navigation_performance_used": False,
    }
    envelope = build_plume_envelope(
        source_receipt=source,
        archive_receipt=archive,
        cue_reference_receipt=cue,
        physical_geometry_receipt=geometry,
        source_sha256=SHA,
        archive_sha256=SHA,
        cue_reference_sha256=SHA,
        physical_geometry_sha256=SHA,
    )
    assert envelope["status"] == "qualified-for-sensory-evaluation"

    bad_geometry = dict(geometry)
    bad_geometry["bilateral_sensor_geometry_frozen"] = False
    with pytest.raises(ValueError, match="bilateral sensor geometry"):
        build_plume_envelope(
            source_receipt=source,
            archive_receipt=archive,
            cue_reference_receipt=cue,
            physical_geometry_receipt=bad_geometry,
            source_sha256=SHA,
            archive_sha256=SHA,
            cue_reference_sha256=SHA,
            physical_geometry_sha256=SHA,
        )
