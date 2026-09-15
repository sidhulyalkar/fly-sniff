from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .connectome_necessity import validate_protocol as validate_connectome_necessity
from .evidence import EvidenceLedger
from .null_factory_v2 import validate_protocol as validate_null_factory

PLUME_ENVELOPE_PROTOCOL = "zero-shot-experimental-plume-evidence-v1"
STRUCTURE_ENVELOPE_PROTOCOL = "zero-shot-olfactory-structure-evidence-v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"artifact must contain a JSON object: {path}")
    return payload


def _require_sha256(value: Any, *, field: str) -> None:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a concrete SHA-256 digest")


def validate_plume_evidence_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    gates = [
        {
            "name": "protocol",
            "passed": payload.get("protocol") == PLUME_ENVELOPE_PROTOCOL,
        },
        {
            "name": "source_protocol",
            "passed": payload.get("source_protocol") == "experimental-plume-sensory-validation-v3",
        },
        {
            "name": "qualified_status",
            "passed": payload.get("status") == "qualified-for-sensory-evaluation",
        },
        {
            "name": "no_controller_or_navigation_selection",
            "passed": payload.get("controller_access") is False
            and payload.get("navigation_performance_used") is False,
        },
        {
            "name": "source_bytes_verified",
            "passed": payload.get("source_bytes_verified") is True,
        },
        {
            "name": "native_time_basis_verified",
            "passed": payload.get("native_time_basis_verified") is True,
        },
        {
            "name": "published_cue_reference_comparison_passed",
            "passed": payload.get("published_cue_reference_comparison_passed") is True,
        },
        {
            "name": "physical_coordinate_mapping_frozen",
            "passed": payload.get("physical_pixel_to_fly_mapping_frozen") is True,
        },
        {
            "name": "bilateral_sensor_geometry_frozen",
            "passed": payload.get("bilateral_sensor_geometry_frozen") is True,
        },
    ]
    receipts = dict(payload.get("receipt_sha256", {}))
    required_receipts = {"source", "archive", "cue_reference", "physical_geometry"}
    receipts_ok = set(receipts) == required_receipts
    if receipts_ok:
        try:
            for name, digest in receipts.items():
                _require_sha256(digest, field=f"receipt_sha256.{name}")
        except ValueError:
            receipts_ok = False
    gates.append({"name": "all_receipts_hash_bound", "passed": receipts_ok})
    return {
        "artifact": "experimental_plume_receipt",
        "valid": all(bool(row["passed"]) for row in gates),
        "gates": gates,
    }


def validate_structural_evidence_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    gates = [
        {
            "name": "protocol",
            "passed": payload.get("protocol") == STRUCTURE_ENVELOPE_PROTOCOL,
        },
        {
            "name": "source_protocol",
            "passed": payload.get("source_protocol") == "olfactory-motion-structural-audit-v1",
        },
        {
            "name": "complete_structural_motif",
            "passed": payload.get("source_status")
            == "candidate_audit_pass_complete_structural_motif",
        },
        {
            "name": "selection_independent_of_navigation",
            "passed": payload.get("controller_access") is False
            and payload.get("navigation_performance_used") is False
            and payload.get("body_ids_selected_from_navigation") is False
            and payload.get("external_connectome_body_ids_imported") is False,
        },
        {
            "name": "functional_claim_remains_forbidden",
            "passed": payload.get("functional_motion_claim_allowed") is False
            and payload.get("navigation_claim_allowed") is False,
        },
    ]
    audit_sha_ok = True
    try:
        _require_sha256(payload.get("source_audit_sha256"), field="source_audit_sha256")
    except ValueError:
        audit_sha_ok = False
    gates.append({"name": "source_audit_hash_bound", "passed": audit_sha_ok})
    return {
        "artifact": "olfactory_motion_structural_audit",
        "valid": all(bool(row["passed"]) for row in gates),
        "gates": gates,
    }


def validate_zero_shot_artifact(name: str, path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)

    if name == "evidence_ledger":
        ledger = EvidenceLedger.load(artifact_path)
        return {
            "artifact": name,
            "valid": True,
            "summary": {
                "records": len(ledger.records),
                "ledger_sha256": ledger.to_dict()["ledger_sha256"],
            },
        }

    payload = _load_json(artifact_path)
    if name == "physiology_calibrated_model":
        valid = (
            payload.get("protocol") == "physiology-calibrated-model-v1"
            and payload.get("fit_status") == "accepted_under_preregistered_objectives"
            and isinstance(payload.get("parameters"), dict)
            and bool(payload["parameters"])
            and isinstance(payload.get("probe_results"), dict)
            and bool(payload["probe_results"])
        )
        return {"artifact": name, "valid": valid}
    if name == "experimental_plume_receipt":
        return validate_plume_evidence_envelope(payload)
    if name == "olfactory_motion_structural_audit":
        return validate_structural_evidence_envelope(payload)
    if name == "null_factory_protocol":
        report = validate_null_factory(payload)
        valid = bool(report["valid_for_preregistration"])
        return {"artifact": name, "valid": valid, "summary": report}
    if name == "connectome_necessity_protocol":
        report = validate_connectome_necessity(payload)
        valid = bool(report["valid_for_preregistration"])
        return {"artifact": name, "valid": valid, "summary": report}
    raise ValueError(f"unknown zero-shot artifact role: {name}")
