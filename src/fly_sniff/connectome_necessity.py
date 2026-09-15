from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROTOCOL = "connectome-necessity-v1"

_REQUIRED_COUNTERFACTUALS = {
    "degree_preserving_rewire",
    "type_constrained_rewire",
    "hemisphere_constrained_rewire",
    "weight_shuffle",
    "sign_shuffle",
    "sensory_temporal_shuffle",
    "odor_removed",
    "wind_removed",
    "PFN_acute_lesion",
    "FB5AB_input_acute_lesion",
    "PFL_DN_acute_lesion",
    "generic_random_reservoir",
}


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected ConnectomeNecessity protocol")
    training = dict(config["training_contract"])
    statistics = dict(config["statistics"])
    null_hierarchy = list(config["null_hierarchy"])
    counterfactuals = {str(x) for x in config["counterfactuals"]}
    upstream = dict(config["required_upstream_artifacts"])
    dynamics_uncertainty = dict(config["dynamics_uncertainty"])
    intervention = dict(config["intervention_contract"])

    gates = [
        {
            "name": "latent_wiring_program",
            "passed": config.get("program") == "latent-wiring",
        },
        {
            "name": "navigation_training_forbidden",
            "passed": training.get("navigation_reward_allowed") is False,
        },
        {
            "name": "navigation_oracles_forbidden",
            "passed": all(
                training.get(name) is False
                for name in (
                    "source_coordinates_allowed",
                    "episode_success_allowed_during_calibration",
                    "body_id_selection_from_navigation_performance_allowed",
                    "individual_edge_weight_fitting_allowed",
                    "variant_specific_parameter_fitting_allowed",
                )
            ),
        },
        {
            "name": "required_upstream_artifacts_frozen",
            "passed": all(value == "required-before-run" for value in upstream.values()),
        },
        {
            "name": "core_counterfactuals_present",
            "passed": _REQUIRED_COUNTERFACTUALS.issubset(counterfactuals),
        },
        {
            "name": "null_hierarchy_is_progressively_constrained",
            "passed": [int(row["level"]) for row in null_hierarchy]
            == list(range(1, len(null_hierarchy) + 1)),
        },
        {
            "name": "confirmatory_null_count_at_least_63",
            "passed": bool(null_hierarchy)
            and all(int(row["minimum_topologies"]) >= 63 for row in null_hierarchy)
            and int(statistics["minimum_confirmatory_topology_nulls"]) >= 63,
        },
        {
            "name": "topology_level_inference_required",
            "passed": statistics.get("topology_is_unit_of_randomization") is True
            and statistics.get("episode_count_must_not_be_treated_as_connectome_count") is True
            and statistics.get("empirical_randomization_p") is True,
        },
        {
            "name": "acute_lesions_primary_and_no_recalibration",
            "passed": intervention.get("primary_necessity_claim_uses_acute_interventions") is True
            and intervention.get("lesions_may_not_trigger_recalibration") is True
            and intervention.get("acute_and_adaptive_results_must_be_separate") is True,
        },
        {
            "name": "dynamics_uncertainty_required",
            "passed": dynamics_uncertainty.get("required") is True
            and len(dynamics_uncertainty.get("families", [])) >= 3
            and dynamics_uncertainty.get("topology_conclusion_must_report_family_sensitivity") is True,
        },
        {
            "name": "power_analysis_required",
            "passed": statistics.get("power_analysis_required_before_final") is True,
        },
    ]
    return {
        "protocol": PROTOCOL,
        "status": config.get("status"),
        "valid_for_preregistration": all(bool(row["passed"]) for row in gates),
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(row["passed"]) for row in gates),
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ConnectomeNecessity v1 before running it")
    parser.add_argument("config", nargs="?", default="configs/connectome_necessity_v1.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    report = validate_protocol(config)
    print(
        f"valid_for_preregistration={report['valid_for_preregistration']} "
        f"gates={report['passed_gate_count']}/{report['gate_count']}"
    )
    for gate in report["gates"]:
        print(f"  {'PASS' if gate['passed'] else 'FAIL'}  {gate['name']}")


if __name__ == "__main__":
    main()
