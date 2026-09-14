from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROTOCOL = "null-factory-v2"

_REQUIRED_FAMILY_ORDER = [
    "degree_preserving",
    "degree_sign_preserving",
    "degree_sign_hemisphere_preserving",
    "within_cell_type_rewire",
    "spatially_constrained_rewire",
]

_REQUIRED_GLOBAL_INVARIANTS = {
    "node_identity_preserved",
    "edge_count_preserved",
    "same_dynamics_parameters_as_intact",
    "same_sensory_inputs_as_intact",
    "same_episode_seed_stream_as_intact",
}


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected NullFactory v2 protocol")

    families = [dict(row) for row in config["families"]]
    seed_contract = dict(config["seed_contract"])
    invariants = dict(config["global_invariants"])
    minimum = int(config["minimum_confirmatory_topologies_per_family"])

    family_names = [str(row["name"]) for row in families]
    family_levels = [int(row["level"]) for row in families]
    required_metadata = {
        str(row["name"]): tuple(str(x) for x in row.get("required_node_metadata", []))
        for row in families
    }

    gates = [
        {
            "name": "confirmatory_null_floor_at_least_63",
            "passed": minimum >= 63,
        },
        {
            "name": "null_hierarchy_order_frozen",
            "passed": family_names == _REQUIRED_FAMILY_ORDER
            and family_levels == list(range(1, len(_REQUIRED_FAMILY_ORDER) + 1)),
        },
        {
            "name": "performance_based_seed_selection_forbidden",
            "passed": seed_contract.get("performance_based_seed_selection_allowed") is False,
        },
        {
            "name": "null_discarding_forbidden",
            "passed": seed_contract.get("failed_or_inconvenient_nulls_may_be_discarded") is False,
        },
        {
            "name": "commit_reveal_required_for_final",
            "passed": seed_contract.get("source") == "commit-reveal-required-before-final",
        },
        {
            "name": "global_fairness_invariants_frozen",
            "passed": all(invariants.get(name) is True for name in _REQUIRED_GLOBAL_INVARIANTS),
        },
        {
            "name": "simple_digraph_constraints_frozen",
            "passed": invariants.get("self_loops_allowed") is False
            and invariants.get("duplicate_edges_allowed") is False,
        },
        {
            "name": "hemisphere_family_requires_metadata",
            "passed": "hemisphere_class"
            in required_metadata["degree_sign_hemisphere_preserving"],
        },
        {
            "name": "type_family_requires_metadata",
            "passed": "cell_type" in required_metadata["within_cell_type_rewire"],
        },
        {
            "name": "spatial_family_requires_type_and_space_metadata",
            "passed": set(required_metadata["spatially_constrained_rewire"])
            >= {"cell_type", "spatial_bin"},
        },
    ]

    implementation_status = {
        str(row["name"]): str(row["implementation_status"])
        for row in families
    }
    ready_families = [
        name
        for name, status in implementation_status.items()
        if status not in {"not-implemented", "blocked"}
    ]
    return {
        "protocol": PROTOCOL,
        "status": config.get("status"),
        "valid_for_preregistration": all(bool(row["passed"]) for row in gates),
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(row["passed"]) for row in gates),
        "gates": gates,
        "ready_families": ready_families,
        "implementation_status": implementation_status,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NullFactory v2 hierarchy")
    parser.add_argument("config", nargs="?", default="configs/null_factory_v2.json")
    args = parser.parse_args()
    report = validate_protocol(json.loads(Path(args.config).read_text()))
    print(
        f"valid_for_preregistration={report['valid_for_preregistration']} "
        f"gates={report['passed_gate_count']}/{report['gate_count']}"
    )
    print("ready_families=" + ",".join(report["ready_families"]))


if __name__ == "__main__":
    main()
