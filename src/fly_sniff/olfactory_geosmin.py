from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_CLAIMS = {
    "G001_ab4B_geosmin_specificity",
    "G002_ab4B_geosmin_sensitivity",
    "G003_DA2_glomerulus_selectivity",
    "G004_DA2_PN_tuning",
    "G005_Or56a_input_necessary_for_aversion",
    "G006_DA2_activation_overrides_attraction",
    "G007_geosmin_suppresses_vinegar_attraction",
}


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError("E001 evidence must be a JSON object")
    return payload


def validate_geosmin_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("E001 evidence requires schema_version=1")
    if payload.get("authority_id") != "E001_geosmin_receptor_physiology":
        raise ValueError("E001 authority id changed")
    if payload.get("program_id") != "olfactory-computation-v0":
        raise ValueError("E001 program id changed")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("doi") != "10.1016/j.cell.2012.09.046":
        raise ValueError("E001 must retain the primary Stensmyr 2012 source authority")
    if source.get("archive_manifest") != "authority/geosmin-e001-source-v1.json":
        raise ValueError("E001 must retain the frozen institutional archive manifest")
    if source.get("primary_archive_status") != "institutional_uri_frozen_local_sha256_pending":
        raise ValueError("E001 primary archive provenance status changed")

    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise TypeError("E001 claims must be a list")
    by_id = {claim.get("claim_id"): claim for claim in claims if isinstance(claim, dict)}
    if set(by_id) != REQUIRED_CLAIMS:
        raise ValueError("E001 claim set changed")
    if len(by_id) != len(claims):
        raise ValueError("E001 claim ids must be unique")
    if any(claim.get("numeric_model_parameter_allowed") is not False for claim in claims):
        raise ValueError("published E001 summary claims cannot become numeric fitted parameters in v0")

    if by_id["G001_ab4B_geosmin_specificity"].get("stimulus_dilution") != 0.01:
        raise ValueError("G001 specificity-screen dilution changed")
    sensitivity = by_id["G002_ab4B_geosmin_sensitivity"]
    if sensitivity.get("reported_detectable_dilution") != 1e-8:
        raise ValueError("G002 reported sensitivity dilution changed")
    if sensitivity.get("reported_stimulus_pipette_mass_pg") != 100.0:
        raise ValueError("G002 reported pipette mass changed")
    if "not equivalent to receptor-site concentration" not in sensitivity.get("caveat", ""):
        raise ValueError("G002 lost the pipette-versus-receptor concentration caveat")

    pn = by_id["G004_DA2_PN_tuning"]
    if pn.get("odor_panel_size") != 17 or pn.get("geosmin_dilution") != 0.001:
        raise ValueError("G004 DA2 PN tuning conditions changed")
    lesion = by_id["G005_Or56a_input_necessary_for_aversion"]
    if lesion.get("o001_use") != "required_perturbation_direction_check":
        raise ValueError("G005 must remain an O001 perturbation-direction check")
    ectopic = by_id["G006_DA2_activation_overrides_attraction"]
    if ectopic.get("stimulus") != "ethyl_butyrate" or ectopic.get("stimulus_dilution") != 1e-5:
        raise ValueError("G006 chemical-identity-independent DA2 activation check changed")
    conflict = by_id["G007_geosmin_suppresses_vinegar_attraction"]
    if conflict.get("o001_use") != "context_only_reserved_for_O003_behavioral_conflict_design":
        raise ValueError("G007 conflict behavior may not be used to fit O001")

    policy = payload.get("calibration_policy")
    if not isinstance(policy, dict) or any(value is not True for value in policy.values()):
        raise ValueError("E001 calibration anti-leakage policy may not be weakened")
    blockers = payload.get("qualification_blockers")
    if not isinstance(blockers, list) or len(blockers) < 4:
        raise ValueError("E001 qualification blockers are incomplete")
    if payload.get("status") != "development_evidence_not_qualified":
        raise ValueError("E001 v0 must remain unqualified until blockers are resolved in a new artifact")

    return {
        "status": payload["status"],
        "claim_count": len(claims),
        "numeric_fit_claims": 0,
        "o003_conflict_reserved": True,
        "qualification_blockers": len(blockers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate geosmin O001 published-evidence constraints")
    parser.add_argument("evidence")
    args = parser.parse_args()
    print(json.dumps(validate_geosmin_evidence(_load(args.evidence)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
