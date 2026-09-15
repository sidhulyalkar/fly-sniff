from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .experiment_protocol import ExperimentSpec, seal_experiment

PROGRAM = "latent-wiring"
REQUIRED_NULL_FAMILIES = {
    "degree_preserving",
    "degree_sign_preserving",
    "degree_sign_hemisphere_preserving",
    "within_cell_type_rewire",
    "spatially_constrained_rewire",
}
REQUIRED_ACUTE = {
    "PFN_lesion",
    "FB5AB_input_lesion",
    "PFL_DN_lesion",
    "odor_removed",
    "wind_removed",
    "sensory_temporal_shuffle",
}


def validate_zero_shot_spec(spec: ExperimentSpec) -> dict[str, Any]:
    spec.validate()
    training = spec.training
    final = spec.final
    nulls = spec.nulls
    metrics = spec.metrics
    dynamics = spec.dynamics
    circuit = spec.circuit
    environment = spec.environment
    interventions = {str(row["name"]): str(row["mode"]) for row in spec.interventions}
    required_artifacts = set(str(x) for x in spec.metadata.get("required_artifacts", []))

    gates = [
        {"name": "latent_wiring_program", "passed": training.get("program") == PROGRAM},
        {
            "name": "navigation_training_forbidden",
            "passed": training.get("navigation_reward_allowed") is False
            and training.get("topology_specific_fit_allowed") is False
            and training.get("whole_graph_backprop_allowed") is False,
        },
        {
            "name": "body_id_selection_from_behavior_forbidden",
            "passed": circuit.get("body_id_selection_from_navigation_performance_allowed") is False,
        },
        {
            "name": "independent_dynamics_calibration_required",
            "passed": dynamics.get("calibration_protocol") == "physiology-calibration-v1"
            and dynamics.get("navigation_reward_used_during_calibration") is False
            and dynamics.get("same_parameters_for_intact_and_nulls") is True
            and dynamics.get("same_parameters_for_acute_lesions") is True,
        },
        {
            "name": "environment_validated_before_neural_evaluation",
            "passed": environment.get("environment_validated_before_neural_evaluation") is True,
        },
        {
            "name": "controller_oracles_forbidden",
            "passed": {
                "source_coordinates",
                "distance_to_source",
                "viewer_plume_image",
                "episode_success",
                "future_sensory_state",
            }.issubset(set(environment.get("controller_forbidden_inputs", []))),
        },
        {
            "name": "confirmatory_null_hierarchy_and_count_frozen",
            "passed": nulls.get("protocol") == "null-factory-v2"
            and int(nulls.get("minimum_confirmatory_topologies_per_family", 0)) >= 63
            and REQUIRED_NULL_FAMILIES.issubset(set(nulls.get("families", [])))
            and nulls.get("generic_random_reservoir_required") is True
            and nulls.get("nulls_may_not_be_selected_or_discarded_by_behavior") is True,
        },
        {
            "name": "acute_intervention_suite_present",
            "passed": REQUIRED_ACUTE.issubset(interventions)
            and all(interventions[name].startswith("acute") for name in REQUIRED_ACUTE),
        },
        {
            "name": "topology_level_statistics_required",
            "passed": metrics.get("primary") == "SPL"
            and metrics.get("topology_is_unit_of_randomization") is True
            and metrics.get("episode_count_must_not_be_treated_as_connectome_count") is True
            and metrics.get("empirical_randomization_p_required") is True
            and metrics.get("power_analysis_required_before_final") is True,
        },
        {
            "name": "one_way_commit_reveal_final_required",
            "passed": final.get("phase") == "confirmatory"
            and final.get("blinded") is True
            and final.get("hidden_final_entropy") == "commit-reveal"
            and final.get("one_way_final") is True
            and final.get("negative_results_retained") is True
            and final.get("paired_episode_conditions") is True
            and final.get("final_results_may_not_trigger_refit") is True,
        },
        {
            "name": "required_lock_artifacts_declared",
            "passed": required_artifacts
            == {
                "evidence_ledger",
                "physiology_calibrated_model",
                "experimental_plume_receipt",
                "olfactory_motion_structural_audit",
                "null_factory_protocol",
                "connectome_necessity_protocol",
            },
        },
    ]
    return {
        "experiment_id": spec.experiment_id,
        "valid_for_preregistration": all(bool(row["passed"]) for row in gates),
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(row["passed"]) for row in gates),
        "gates": gates,
        "required_artifacts": sorted(required_artifacts),
    }


def _parse_artifacts(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("artifact arguments must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if name in result:
            raise ValueError(f"duplicate artifact name: {name}")
        result[name] = Path(raw_path)
    return result


def _git_state() -> tuple[str, list[str]]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = [
        line
        for line in subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines()
        if line.strip()
    ]
    return sha, dirty


def _runtime_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "pandas", "networkx", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def seal_zero_shot(
    spec: ExperimentSpec,
    *,
    artifact_paths: dict[str, Path],
    code_ref: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_zero_shot_spec(spec)
    if not validation["valid_for_preregistration"]:
        failed = [row["name"] for row in validation["gates"] if not row["passed"]]
        raise ValueError(f"zero-shot experiment spec failed preregistration gates: {failed}")
    expected = set(validation["required_artifacts"])
    observed = set(artifact_paths)
    if observed != expected:
        raise ValueError(
            f"zero-shot lock artifact set mismatch: missing={sorted(expected - observed)} "
            f"unexpected={sorted(observed - expected)}"
        )
    missing_paths = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"required zero-shot artifacts missing: {missing_paths}")
    lock = seal_experiment(
        spec,
        artifact_paths=artifact_paths,
        code_ref=code_ref,
        runtime=runtime,
    )
    return lock.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or seal zero-shot latent-wiring experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("spec", nargs="?", default="configs/zero_shot_latent_wiring_v1.json")

    seal = sub.add_parser("seal")
    seal.add_argument("spec", nargs="?", default="configs/zero_shot_latent_wiring_v1.json")
    seal.add_argument("--artifact", action="append", default=[], help="NAME=PATH")
    seal.add_argument("--output", default="manifests/zero-shot-latent-wiring-v1.lock.json")

    args = parser.parse_args()
    spec = ExperimentSpec.load(args.spec)
    report = validate_zero_shot_spec(spec)
    if args.command == "validate":
        print(
            f"valid_for_preregistration={report['valid_for_preregistration']} "
            f"gates={report['passed_gate_count']}/{report['gate_count']}"
        )
        for gate in report["gates"]:
            print(f"  {'PASS' if gate['passed'] else 'FAIL'}  {gate['name']}")
        return

    sha, dirty = _git_state()
    if dirty:
        raise RuntimeError(f"refusing to seal zero-shot experiment from dirty git tree: {dirty}")
    payload = seal_zero_shot(
        spec,
        artifact_paths=_parse_artifacts(args.artifact),
        code_ref=sha,
        runtime=_runtime_manifest(),
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"lock={output}")
    print(f"sha256={payload['lock_sha256']}")


if __name__ == "__main__":
    main()
