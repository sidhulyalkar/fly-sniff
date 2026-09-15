from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .zero_shot_artifacts import (
    PLUME_ENVELOPE_PROTOCOL,
    STRUCTURE_ENVELOPE_PROTOCOL,
    validate_plume_evidence_envelope,
    validate_structural_evidence_envelope,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def build_structural_envelope(audit: dict[str, Any], *, audit_sha256: str) -> dict[str, Any]:
    if audit.get("protocol") != "olfactory-motion-structural-audit-v1":
        raise ValueError("unexpected olfactory-motion structural audit protocol")
    if audit.get("dataset") != "male-cns:v1.0":
        raise ValueError("zero-shot structural evidence requires male-cns:v1.0")
    if audit.get("status") != "candidate_audit_pass_complete_structural_motif":
        raise ValueError(f"structural motif is not complete: {audit.get('status')}")
    if audit.get("weights_sha256") in {None, ""}:
        raise ValueError("structural audit must be bound to the MaleCNS weight table")
    if audit.get("controller_access") is not False:
        raise ValueError("structural audit granted controller access")
    if audit.get("navigation_performance_used") is not False:
        raise ValueError("structural audit used navigation performance")
    if audit.get("functional_motion_claim_allowed") is not False:
        raise ValueError("structural audit improperly permits a functional motion claim")
    if audit.get("navigation_claim_allowed") is not False:
        raise ValueError("structural audit improperly permits a navigation claim")

    envelope = {
        "protocol": STRUCTURE_ENVELOPE_PROTOCOL,
        "dataset": "male-cns:v1.0",
        "source_protocol": audit["protocol"],
        "source_status": audit["status"],
        "source_audit_sha256": audit_sha256,
        "controller_access": False,
        "navigation_performance_used": False,
        "body_ids_selected_from_navigation": False,
        "external_connectome_body_ids_imported": False,
        "functional_motion_claim_allowed": False,
        "navigation_claim_allowed": False,
        "selected_body_count": int(audit.get("selected_body_count", 0)),
        "selected_edge_count": int(audit.get("selected_edge_count", 0)),
        "annotation_sha256": audit.get("annotation_sha256"),
        "weights_sha256": audit.get("weights_sha256"),
        "claim_boundary": (
            "This envelope carries a complete annotation-resolved MaleCNS structural motif into "
            "the zero-shot experiment lock. It does not promote the motif to a functional odor-"
            "motion circuit or navigation mechanism."
        ),
    }
    validation = validate_structural_evidence_envelope(envelope)
    if not validation["valid"]:
        failed = [row["name"] for row in validation["gates"] if not row["passed"]]
        raise ValueError(f"constructed structural evidence envelope failed validation: {failed}")
    return envelope


def _require_passed_receipt(
    payload: dict[str, Any],
    *,
    protocol: str,
    status_values: set[str],
    label: str,
) -> None:
    if payload.get("protocol") != protocol:
        raise ValueError(f"unexpected {label} protocol: {payload.get('protocol')!r}")
    if str(payload.get("status")) not in status_values:
        raise ValueError(f"{label} is not qualified: {payload.get('status')!r}")
    if payload.get("controller_access") not in {False, None}:
        raise ValueError(f"{label} improperly grants controller access")
    if payload.get("navigation_performance_used") not in {False, None}:
        raise ValueError(f"{label} improperly uses navigation performance")


def build_plume_envelope(
    *,
    source_receipt: dict[str, Any],
    archive_receipt: dict[str, Any],
    cue_reference_receipt: dict[str, Any],
    physical_geometry_receipt: dict[str, Any],
    source_sha256: str,
    archive_sha256: str,
    cue_reference_sha256: str,
    physical_geometry_sha256: str,
) -> dict[str, Any]:
    """Aggregate already-qualified plume evidence without manufacturing missing evidence."""
    if source_receipt.get("protocol") != "experimental-plume-sensory-validation-v3":
        raise ValueError("unexpected experimental-plume source protocol")
    if source_receipt.get("source_bytes_verified") is not True:
        raise ValueError("experimental plume source bytes have not been verified")

    _require_passed_receipt(
        archive_receipt,
        protocol="experimental-plume-archive-inspection-v3",
        status_values={"archive_qualified", "archive_qualified_for_publication_reproduction"},
        label="archive receipt",
    )
    native_time = archive_receipt.get("native_time_basis", {})
    if not isinstance(native_time, dict) or native_time.get("kind") == "unresolved":
        raise ValueError("experimental plume native time basis is unresolved")

    _require_passed_receipt(
        cue_reference_receipt,
        protocol="experimental-plume-cue-reference-comparison-v1",
        status_values={"passed_reference_comparison"},
        label="cue reference receipt",
    )
    if cue_reference_receipt.get("passed") is not True:
        raise ValueError("published cue reference comparison did not pass")

    _require_passed_receipt(
        physical_geometry_receipt,
        protocol="experimental-plume-physical-geometry-v1",
        status_values={"qualified_physical_geometry"},
        label="physical geometry receipt",
    )
    if physical_geometry_receipt.get("pixel_to_fly_mapping_frozen") is not True:
        raise ValueError("physical pixel-to-fly mapping is not frozen")
    if physical_geometry_receipt.get("bilateral_sensor_geometry_frozen") is not True:
        raise ValueError("bilateral sensor geometry is not frozen")

    envelope = {
        "protocol": PLUME_ENVELOPE_PROTOCOL,
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
            "source": source_sha256,
            "archive": archive_sha256,
            "cue_reference": cue_reference_sha256,
            "physical_geometry": physical_geometry_sha256,
        },
        "claim_boundary": (
            "This envelope qualifies a frozen experimental sensory substrate for zero-shot "
            "evaluation only. It does not establish a biological olfactory-motion circuit, "
            "connectome dynamics, or navigation performance."
        ),
    }
    validation = validate_plume_evidence_envelope(envelope)
    if not validation["valid"]:
        failed = [row["name"] for row in validation["gates"] if not row["passed"]]
        raise ValueError(f"constructed plume evidence envelope failed validation: {failed}")
    return envelope


def _write(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence envelope: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fail-closed zero-shot evidence envelopes")
    sub = parser.add_subparsers(dest="command", required=True)

    structure = sub.add_parser("structure")
    structure.add_argument("audit")
    structure.add_argument("--output", default="authority/zero-shot-olfactory-structure-evidence-v1.json")

    plume = sub.add_parser("plume")
    plume.add_argument("--source", required=True)
    plume.add_argument("--archive", required=True)
    plume.add_argument("--cue-reference", required=True)
    plume.add_argument("--physical-geometry", required=True)
    plume.add_argument("--output", default="authority/zero-shot-experimental-plume-evidence-v1.json")

    args = parser.parse_args()
    if args.command == "structure":
        payload = build_structural_envelope(
            _load(args.audit),
            audit_sha256=file_sha256(args.audit),
        )
    else:
        payload = build_plume_envelope(
            source_receipt=_load(args.source),
            archive_receipt=_load(args.archive),
            cue_reference_receipt=_load(args.cue_reference),
            physical_geometry_receipt=_load(args.physical_geometry),
            source_sha256=file_sha256(args.source),
            archive_sha256=file_sha256(args.archive),
            cue_reference_sha256=file_sha256(args.cue_reference),
            physical_geometry_sha256=file_sha256(args.physical_geometry),
        )
    output = _write(args.output, payload)
    print(output)
    print(file_sha256(output))


if __name__ == "__main__":
    main()
