from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_AIMS = ("O001", "O002", "O003", "O004", "O005")
REQUIRED_COMPARATORS = {
    "receptor_only",
    "static_biological_topology",
    "biological_topology_plus_independent_physiology",
    "capacity_matched_generic_model",
    "matched_topology_nulls",
}
REQUIRED_NULL_FAMILIES = {
    "directed_degree_preserving",
    "cell_type_constrained",
    "hemisphere_constrained",
    "glomerulus_output_identity_shuffle",
}
REQUIRED_EVIDENCE_CLASSES = {
    "measured_structure",
    "measured_physiology",
    "predicted_annotation",
    "cross_dataset_prior",
    "model_assumption",
    "fitted_parameter",
    "modeled_state",
    "behavioral_output",
}


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def validate_program(program: dict[str, Any]) -> None:
    if program.get("schema_version") != 1 or program.get("program_id") != "olfactory-computation-v0":
        raise ValueError("unsupported olfactory computation program schema")
    if program.get("status") != "pre_data_preregistration":
        raise ValueError("v0 must remain a pre-data preregistration until explicitly versioned")
    rules = program.get("program_a_rules")
    if not isinstance(rules, dict):
        raise TypeError("program_a_rules must be an object")
    if rules.get("behavioral_reward_allowed_for_calibration") is not False:
        raise ValueError("Program A calibration must not consume behavioral reward")
    if rules.get("topology_specific_fit_allowed") is not False:
        raise ValueError("Program A forbids topology-specific fitting")
    if rules.get("whole_graph_backprop_allowed") is not False:
        raise ValueError("whole-graph backprop is outside the v0 biological claim boundary")
    if rules.get("equal_budget_across_topologies") is not True:
        raise ValueError("topology comparisons require equal fitting budgets")
    if rules.get("negative_results_retained") is not True:
        raise ValueError("negative results must be retained")

    aims = program.get("aims")
    if not isinstance(aims, list):
        raise TypeError("aims must be a list")
    by_id = {aim.get("id"): aim for aim in aims if isinstance(aim, dict)}
    if tuple(sorted(by_id)) != REQUIRED_AIMS:
        raise ValueError(f"program must define exactly aims {REQUIRED_AIMS}")
    if by_id["O001"].get("headline_topology_claim_allowed") is not False:
        raise ValueError("O001 geosmin specialist calibration cannot be the headline topology claim")
    if by_id["O001"].get("role") != "calibration":
        raise ValueError("O001 must remain a calibration aim")
    if by_id["O005"].get("headline_topology_claim_allowed") is not False:
        raise ValueError("engineering transfer cannot establish the biological topology claim")
    if by_id["O005"].get("role") != "engineering_translation":
        raise ValueError("O005 must remain explicitly engineering translation")

    comparators = set(program.get("required_model_comparators", []))
    if not REQUIRED_COMPARATORS <= comparators:
        raise ValueError(f"missing required model comparators: {sorted(REQUIRED_COMPARATORS - comparators)}")
    nulls = set(program.get("required_topology_null_families", []))
    if not REQUIRED_NULL_FAMILIES <= nulls:
        raise ValueError(f"missing required topology null families: {sorted(REQUIRED_NULL_FAMILIES - nulls)}")

    policy = program.get("confirmatory_policy")
    if not isinstance(policy, dict):
        raise TypeError("confirmatory_policy must be an object")
    if int(policy.get("minimum_topology_null_count", -1)) < 31:
        raise ValueError("confirmatory topology claims require at least 31 topology nulls")
    if int(policy.get("preferred_topology_null_count", -1)) < 63:
        raise ValueError("flagship v0 should preserve the preferred 63-null target")
    required_true = (
        "paired_conditions_required",
        "hidden_final_entropy_required",
        "one_way_final_required",
        "topology_level_inference_required",
        "episode_level_pseudoreplication_forbidden",
        "heldout_conditions_frozen_before_final",
    )
    if any(policy.get(key) is not True for key in required_true):
        raise ValueError("confirmatory policy weakened a frozen anti-leakage/statistical requirement")
    if policy.get("prospective_validation_target") != "at_least_one_nontrivial_novel_prediction":
        raise ValueError("publication program requires a prospective novel-prediction target")

    boundaries = program.get("claim_boundaries")
    if not isinstance(boundaries, list) or len(boundaries) < 5:
        raise ValueError("claim boundaries are incomplete")
    text = " ".join(str(item).lower() for item in boundaries)
    if "connectivity alone" not in text or "engineering transfer" not in text:
        raise ValueError("claim boundary must separate structure, physiology, and engineering utility")


def validate_evidence_requirements(evidence: dict[str, Any]) -> list[str]:
    if evidence.get("schema_version") != 1 or evidence.get("program_id") != "olfactory-computation-v0":
        raise ValueError("unsupported olfactory evidence registry")
    authorities = evidence.get("authorities")
    if not isinstance(authorities, list) or not authorities:
        raise ValueError("evidence registry requires authorities")
    ids: set[str] = set()
    unresolved: list[str] = []
    for record in authorities:
        if not isinstance(record, dict):
            raise TypeError("authority record must be an object")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("authority requires a stable id")
        if record_id in ids:
            raise ValueError(f"duplicate evidence authority {record_id}")
        ids.add(record_id)
        evidence_class = record.get("evidence_class")
        if evidence_class not in REQUIRED_EVIDENCE_CLASSES:
            raise ValueError(f"unsupported evidence class {evidence_class!r} for {record_id}")
        if not record.get("required_for") or not record.get("requirements"):
            raise ValueError(f"authority {record_id} is missing aim scope or requirements")
        status = record.get("status")
        if status not in {"unresolved", "qualified"}:
            raise ValueError(f"unsupported authority status {status!r}")
        if status == "unresolved":
            unresolved.append(record_id)
        elif not isinstance(record.get("artifact_sha256"), str) or len(record["artifact_sha256"]) != 64:
            raise ValueError(f"qualified authority {record_id} requires a content-addressed artifact")
    rules = evidence.get("rules")
    if not isinstance(rules, dict) or any(value is not True for value in rules.values()):
        raise ValueError("evidence anti-promotion rules may not be weakened")
    expected_status = "blocked_pending_evidence" if unresolved else "ready_for_confirmatory_lock"
    if evidence.get("confirmatory_execution_status") != expected_status:
        raise ValueError("confirmatory execution status does not match unresolved evidence authorities")
    return sorted(unresolved)


def validate_source_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != 1 or registry.get("program_id") != "olfactory-computation-v0":
        raise ValueError("unsupported olfactory source registry")
    sources = registry.get("sources")
    if not isinstance(sources, list) or len(sources) < 7:
        raise ValueError("source registry is incomplete")
    ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("source record must be an object")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in ids:
            raise ValueError("source ids must be unique and non-empty")
        ids.add(source_id)
        if source.get("status") != "source_identified_not_frozen":
            raise ValueError("v0 source records remain candidates until exact bytes are frozen")
        if not source.get("citation") or not source.get("role") or not source.get("evidence_use"):
            raise ValueError(f"source {source_id} lacks citation, evidence role, or use boundary")
    rules = registry.get("rules")
    if not isinstance(rules, dict) or any(value is not True for value in rules.values()):
        raise ValueError("source-provenance rules may not be weakened")


def validate_o001(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("experiment_id") != "O001-geosmin-calibration-v0":
        raise ValueError("unsupported O001 protocol")
    if protocol.get("role") != "calibration_not_headline_topology_claim":
        raise ValueError("O001 cannot be promoted to the headline topology claim")
    forbidden = set(protocol.get("forbidden_calibration_targets", []))
    required_forbidden = {"O003_heldout_behavior_score", "O004_navigation_reward", "source_finding_success"}
    if not required_forbidden <= forbidden:
        raise ValueError("O001 failed to exclude downstream behavioral/task objectives")
    comparators = set(protocol.get("required_comparators", []))
    if "receptor_only" not in comparators or "audited_biological_pathway" not in comparators:
        raise ValueError("O001 requires receptor-only and audited-pathway comparators")
    perturbations = set(protocol.get("required_perturbation_checks", []))
    if not {"remove_or56a_input", "remove_or_isolate_da2_projection_path"} <= perturbations:
        raise ValueError("O001 requires prespecified Or56a and DA2 perturbation checks")
    if protocol.get("promotion_rule") != "A new version is required for any change informed by downstream performance.":
        raise ValueError("O001 downstream-performance versioning rule changed")


def validate_study(
    program_path: str | Path,
    evidence_path: str | Path,
    *,
    source_registry_path: str | Path | None = None,
    o001_path: str | Path | None = None,
) -> dict[str, Any]:
    program = _load(program_path)
    evidence = _load(evidence_path)
    validate_program(program)
    unresolved = validate_evidence_requirements(evidence)
    if source_registry_path is not None:
        validate_source_registry(_load(source_registry_path))
    if o001_path is not None:
        validate_o001(_load(o001_path))
    return {
        "status": "blocked" if unresolved else "ready_for_confirmatory_lock",
        "program_id": program["program_id"],
        "unresolved_authorities": unresolved,
        "unresolved_count": len(unresolved),
        "message": (
            "Development planning may proceed, but confirmatory execution is blocked until every required authority is qualified."
            if unresolved
            else "All declared authorities are qualified; construct a content-addressed ExperimentSpec/Lock."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the olfactory computation study contract")
    parser.add_argument("program")
    parser.add_argument("evidence")
    parser.add_argument("--source-registry")
    parser.add_argument("--o001")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_study(
                args.program,
                args.evidence,
                source_registry_path=args.source_registry,
                o001_path=args.o001,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
