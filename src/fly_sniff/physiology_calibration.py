from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROTOCOL = "physiology-calibration-v1"

_NAVIGATION_TERMS = {
    "navigation_reward",
    "source_coordinates",
    "distance_to_source",
    "episode_success",
    "SPL",
    "path_efficiency",
    "time_to_source",
    "intact_vs_rewire_separation",
    "final_test_performance",
}

_REQUIRED_PERMITTED_PARAMETERS = {
    "global_neural_time_constant",
    "global_recurrent_gain",
    "global_activation_scale",
    "sensory_transduction_gain",
}

_REQUIRED_FORBIDDEN_PARAMETERS = {
    "individual_edge_weights",
    "body_id_membership",
    "left_right_independent_gains",
    "variant_specific_parameters",
    "navigation_policy_parameters",
}


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected PhysiologyCalibration protocol")

    forbidden_signals = {str(x) for x in config["forbidden_calibration_signals"]}
    permitted_parameters = {str(x) for x in config["permitted_parameters"]}
    forbidden_parameters = {str(x) for x in config["forbidden_parameters"]}
    sharing = dict(config["parameter_sharing"])
    probe_contract = dict(config["probe_contract"])
    fit = dict(config["fit_contract"])
    probes = [dict(row) for row in config["probes"]]

    gates = [
        {
            "name": "navigation_signals_all_forbidden",
            "passed": _NAVIGATION_TERMS.issubset(forbidden_signals),
        },
        {
            "name": "permitted_parameter_set_is_minimal_and_global",
            "passed": permitted_parameters == _REQUIRED_PERMITTED_PARAMETERS,
        },
        {
            "name": "dangerous_parameter_freedom_forbidden",
            "passed": _REQUIRED_FORBIDDEN_PARAMETERS.issubset(forbidden_parameters),
        },
        {
            "name": "parameters_shared_across_topology_and_acute_lesions",
            "passed": all(bool(value) for value in sharing.values()),
        },
        {
            "name": "probe_objective_frozen_before_fit",
            "passed": probe_contract.get("objective_must_be_defined_before_fit") is True,
        },
        {
            "name": "probe_authority_required",
            "passed": probe_contract.get("all_probes_require_evidence_ledger_authority") is True,
        },
        {
            "name": "cross_dataset_label_preserved",
            "passed": probe_contract.get("cross_dataset_probes_must_remain_labeled_cross_dataset") is True,
        },
        {
            "name": "failed_probes_cannot_be_dropped",
            "passed": probe_contract.get("failed_probes_may_not_be_dropped_after_fit") is True,
        },
        {
            "name": "navigation_weight_exactly_zero",
            "passed": float(fit.get("navigation_objective_weight", float("nan"))) == 0.0,
        },
        {
            "name": "fit_blinded_to_topology_and_final_evaluation",
            "passed": fit.get("topology_variant_visible_during_fit") is False
            and fit.get("final_evaluation_visible_during_fit") is False,
        },
        {
            "name": "deterministic_fit_seed_required",
            "passed": fit.get("deterministic_fit_seed_required") is True,
        },
        {
            "name": "probe_set_nonempty_and_unique",
            "passed": bool(probes)
            and len({str(row["id"]) for row in probes}) == len(probes),
        },
    ]

    unresolved = [str(row["id"]) for row in probes if row.get("authority_status") != "sealed"]
    preregistration_valid = all(bool(row["passed"]) for row in gates)
    return {
        "protocol": PROTOCOL,
        "status": config.get("status"),
        "valid_for_preregistration": preregistration_valid,
        "ready_to_fit": preregistration_valid and not unresolved,
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(row["passed"]) for row in gates),
        "gates": gates,
        "unresolved_probe_authorities": unresolved,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PhysiologyCalibration v1")
    parser.add_argument("config", nargs="?", default="configs/physiology_calibration_v1.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    report = validate_protocol(config)
    print(
        f"valid_for_preregistration={report['valid_for_preregistration']} "
        f"ready_to_fit={report['ready_to_fit']} "
        f"gates={report['passed_gate_count']}/{report['gate_count']}"
    )
    if report["unresolved_probe_authorities"]:
        print("unresolved=" + ",".join(report["unresolved_probe_authorities"]))


if __name__ == "__main__":
    main()
