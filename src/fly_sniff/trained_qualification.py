from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import Observation
from .graph import GraphBundle
from .rewire import lesion_incoming_to_roles
from .runtime_provenance import verify_runtime_environment_receipt
from .training import (
    FINAL_TEST_MAX_SEED,
    DynamicsParameters,
    TaskOptimizedMaleCNSController,
    _development_gate_from_summaries,
    _validate_model_role_partition,
    canonical_sha256,
    default_parameters,
    load_training_config,
    make_training_seed_split,
    optimizer_budget_receipt,
    validate_parameters,
)
from .training_budget_redteam import reconstruct_optimizer_execution

REQUIRED_ROLES = (
    "odor_context_left",
    "odor_context_right",
    "wind_basis_left",
    "wind_basis_right",
    "steer_left",
    "steer_right",
)


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    value: float | int | str | None
    criterion: str


@dataclass(frozen=True)
class TrainedProbeResult:
    odor_on_downwind_left_turn: float
    odor_on_downwind_right_turn: float
    odor_off_downwind_left_turn: float
    odor_off_downwind_right_turn: float
    odor_on_wind_separation: float
    odor_off_wind_separation: float
    odor_gating_separation_delta: float
    upwind_laterality_correct: bool
    odor_laterality_error: float
    lesioned_peak_turn: float
    deterministic_error: float


def _observation(
    *,
    left_odor: float,
    right_odor: float,
    wind_y: float,
) -> Observation:
    return Observation(
        left_odor=float(left_odor),
        right_odor=float(right_odor),
        mean_odor=0.5 * (left_odor + right_odor),
        odor_delta=float(right_odor - left_odor),
        wind_x_body=0.0,
        wind_y_body=float(wind_y),
        heading=0.0,
    )


def _rollout(
    bundle: GraphBundle,
    parameters: DynamicsParameters,
    observation: Observation,
    *,
    steps: int,
    seed: int,
) -> np.ndarray:
    controller = TaskOptimizedMaleCNSController(
        bundle,
        parameters,
        require_qualified=False,
    )
    controller.reset(seed)
    return np.asarray([controller.act(observation).turn for _ in range(steps)], dtype=float)


def _late_mean(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    width = max(4, len(values) // 4)
    return float(np.mean(values[-width:]))


def _reachable(edges, starts: set[int]) -> set[int]:
    adjacency: dict[int, list[int]] = {}
    for row in edges[["source", "target", "weight"]].itertuples(index=False):
        if float(row.weight) <= 0.0:
            continue
        adjacency.setdefault(int(row.source), []).append(int(row.target))
    seen = {int(x) for x in starts}
    queue = deque(seen)
    while queue:
        node = queue.popleft()
        for target in adjacency.get(node, []):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def probe_trained_candidate(
    bundle: GraphBundle,
    parameters: DynamicsParameters,
    config: dict[str, Any],
) -> TrainedProbeResult:
    probe = config["trained_e002"]
    seed = int(probe["probe_seed"])
    steps = int(probe["probe_steps"])
    odor = float(probe["odor_level"])
    wind = float(probe["crosswind_magnitude"])
    if steps < 4 or not 0.0 < odor <= 1.0 or wind <= 0.0:
        raise ValueError("invalid frozen trained-E002 probe settings")

    odor_on_left = _observation(left_odor=odor, right_odor=odor, wind_y=wind)
    odor_on_right = _observation(left_odor=odor, right_odor=odor, wind_y=-wind)
    odor_off_left = _observation(left_odor=0.0, right_odor=0.0, wind_y=wind)
    odor_off_right = _observation(left_odor=0.0, right_odor=0.0, wind_y=-wind)

    on_left = _rollout(bundle, parameters, odor_on_left, steps=steps, seed=seed)
    on_right = _rollout(bundle, parameters, odor_on_right, steps=steps, seed=seed)
    off_left = _rollout(bundle, parameters, odor_off_left, steps=steps, seed=seed)
    off_right = _rollout(bundle, parameters, odor_off_right, steps=steps, seed=seed)

    on_left_turn = _late_mean(on_left)
    on_right_turn = _late_mean(on_right)
    off_left_turn = _late_mean(off_left)
    off_right_turn = _late_mean(off_right)
    on_separation = abs(on_right_turn - on_left_turn)
    off_separation = abs(off_right_turn - off_left_turn)

    left_only = _observation(left_odor=odor, right_odor=0.0, wind_y=wind)
    right_only = _observation(left_odor=0.0, right_odor=odor, wind_y=wind)
    left_only_turns = _rollout(bundle, parameters, left_only, steps=steps, seed=seed)
    right_only_turns = _rollout(bundle, parameters, right_only, steps=steps, seed=seed)
    laterality_error = float(np.max(np.abs(left_only_turns - right_only_turns)))

    replay = _rollout(bundle, parameters, odor_on_left, steps=steps, seed=seed)
    deterministic_error = float(np.max(np.abs(on_left - replay)))

    lesioned = lesion_incoming_to_roles(bundle, ["steer_left", "steer_right"])
    lesion_left = _rollout(lesioned, parameters, odor_on_left, steps=steps, seed=seed)
    lesion_right = _rollout(lesioned, parameters, odor_on_right, steps=steps, seed=seed)
    lesioned_peak = float(
        max(
            np.max(np.abs(lesion_left)) if len(lesion_left) else 0.0,
            np.max(np.abs(lesion_right)) if len(lesion_right) else 0.0,
        )
    )

    laterality_correct = bool(on_left_turn < 0.0 and on_right_turn > 0.0)
    return TrainedProbeResult(
        odor_on_downwind_left_turn=on_left_turn,
        odor_on_downwind_right_turn=on_right_turn,
        odor_off_downwind_left_turn=off_left_turn,
        odor_off_downwind_right_turn=off_right_turn,
        odor_on_wind_separation=on_separation,
        odor_off_wind_separation=off_separation,
        odor_gating_separation_delta=on_separation - off_separation,
        upwind_laterality_correct=laterality_correct,
        odor_laterality_error=laterality_error,
        lesioned_peak_turn=lesioned_peak,
        deterministic_error=deterministic_error,
    )


def _require_equal_float(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training report {name} is missing or invalid") from exc
    if not np.isfinite(value) or not np.isclose(value, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"training report {name} does not match the value recomputed from frozen evidence"
        )


def _verify_seed_receipts(report: dict[str, Any], config: dict[str, Any]) -> tuple[list[int], list[int]]:
    train, validation = make_training_seed_split(config)
    observed_train = [int(x) for x in report.get("train_seeds", [])]
    observed_validation = [int(x) for x in report.get("validation_seeds", [])]
    if observed_train != train or observed_validation != validation:
        raise ValueError("training report seed split/receipt does not match the frozen config")
    if report.get("train_seed_sha256") != canonical_sha256(train):
        raise ValueError("training report train seed hash mismatch")
    if report.get("validation_seed_sha256") != canonical_sha256(validation):
        raise ValueError("training report validation seed hash mismatch")
    if int(report.get("train_seed_count", -1)) != len(train):
        raise ValueError("training report train seed count mismatch")
    if int(report.get("validation_seed_count", -1)) != len(validation):
        raise ValueError("training report validation seed count mismatch")
    if min(train + validation) <= FINAL_TEST_MAX_SEED:
        raise ValueError("frozen development seed split overlaps the final-test namespace")
    if report.get("final_test_namespace_touched") is not False:
        raise ValueError("training report claims final-test namespace access")
    return train, validation


def _verify_budget_receipt(
    report: dict[str, Any],
    config: dict[str, Any],
    *,
    train_count: int,
    validation_count: int,
) -> dict[str, Any]:
    expected = optimizer_budget_receipt(
        config,
        train_seed_count=train_count,
        validation_seed_count=validation_count,
    )
    if report.get("optimizer_budget") != expected:
        raise ValueError("training report optimizer compute-budget receipt mismatch")
    expected_hash = canonical_sha256(expected)
    if report.get("optimizer_budget_sha256") != expected_hash:
        raise ValueError("training report optimizer budget hash mismatch")
    try:
        reconstructed = reconstruct_optimizer_execution(report, config)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "training report optimizer history does not reconstruct to the frozen execution budget"
        ) from exc
    if reconstructed != expected:
        raise ValueError("training report reconstructed optimizer execution budget mismatch")
    return expected


def _verify_runtime_receipt(report: dict[str, Any]) -> tuple[str, str]:
    receipt = report.get("runtime_environment")
    if not isinstance(receipt, dict):
        raise ValueError("training report is missing the sealed numerical runtime receipt")
    runtime_sha256 = canonical_sha256(receipt)
    if report.get("runtime_environment_sha256") != runtime_sha256:
        raise ValueError("training report runtime environment hash mismatch")
    try:
        numerical_sha256 = verify_runtime_environment_receipt(
            receipt,
            require_current_numerical_match=True,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "training report numerical runtime is incompatible with the current promotion runtime"
        ) from exc
    if report.get("numerical_runtime_sha256") != numerical_sha256:
        raise ValueError("training report numerical runtime hash mismatch")
    return runtime_sha256, numerical_sha256


def _verify_development_gate(report: dict[str, Any], config: dict[str, Any]) -> None:
    baseline_validation = report.get("baseline_validation")
    trained_validation = report.get("trained_validation")
    if not isinstance(baseline_validation, dict) or not isinstance(trained_validation, dict):
        raise ValueError("training report is missing validation summaries needed to recompute gate")
    objective_delta, success_delta, passed = _development_gate_from_summaries(
        baseline_validation,
        trained_validation,
        config,
    )
    _require_equal_float(
        "validation_objective_delta",
        report.get("validation_objective_delta"),
        objective_delta,
    )
    _require_equal_float(
        "validation_success_rate_delta",
        report.get("validation_success_rate_delta"),
        success_delta,
    )
    if report.get("development_gate_passed") is not passed:
        raise ValueError("training report development gate contradicts frozen validation evidence")


def parameters_from_training_report(
    report: dict[str, Any],
    bundle: GraphBundle,
    config: dict[str, Any],
) -> DynamicsParameters:
    if report.get("protocol") != config["protocol"]:
        raise ValueError("training report protocol does not match frozen config")
    if report.get("training_config_sha256") != canonical_sha256(config):
        raise ValueError("training report was produced under a different training config")
    if report.get("graph_sha256") != bundle.replay_fingerprint():
        raise ValueError("training report graph fingerprint does not match candidate bundle")
    if report.get("sensory_interface") != config["connectome_sensory_interface"]:
        raise ValueError("training report sensory-interface receipt does not match frozen config")

    _validate_model_role_partition(bundle)
    train, validation = _verify_seed_receipts(report, config)
    budget = _verify_budget_receipt(
        report,
        config,
        train_count=len(train),
        validation_count=len(validation),
    )
    runtime_sha256, numerical_sha256 = _verify_runtime_receipt(report)
    _verify_development_gate(report, config)

    baseline = default_parameters(config).to_dict()
    if report.get("baseline_parameters") != baseline:
        raise ValueError("training report baseline parameters do not match frozen defaults")

    parameters = DynamicsParameters.from_mapping(report["trained_parameters"])
    validate_parameters(parameters, config)
    parameter_sha = canonical_sha256(parameters.to_dict())
    if report.get("trained_parameter_sha256") != parameter_sha:
        raise ValueError("trained parameter hash mismatch")

    audit_payload = {
        "graph_sha256": bundle.replay_fingerprint(),
        "training_config_sha256": canonical_sha256(config),
        "train_seed_sha256": canonical_sha256(train),
        "validation_seed_sha256": canonical_sha256(validation),
        "trained_parameter_sha256": parameter_sha,
        "optimizer_budget_sha256": canonical_sha256(budget),
        "runtime_environment_sha256": runtime_sha256,
        "numerical_runtime_sha256": numerical_sha256,
    }
    if report.get("audit_receipt_sha256") != canonical_sha256(audit_payload):
        raise ValueError("training report audit receipt hash mismatch")
    return parameters


def qualify_trained_candidate(
    bundle: GraphBundle,
    training_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    bundle.validate(require_sign=True, require_qualified=False)
    parameters = parameters_from_training_report(training_report, bundle, config)
    frozen = config["trained_e002"]
    roles = {name: {int(x) for x in ids} for name, ids in bundle.roles.items()}
    ids = set(bundle.nodes.bodyId.astype(int))
    missing_roles = [role for role in REQUIRED_ROLES if not roles.get(role)]
    role_body_ids = set().union(*(roles.get(role, set()) for role in REQUIRED_ROLES))
    role_closure = role_body_ids.issubset(ids)
    signed_fraction = float(bundle.edges.sign.astype(int).ne(0).mean()) if len(bundle.edges) else 0.0

    sensory = set().union(
        roles.get("odor_context_left", set()),
        roles.get("odor_context_right", set()),
        roles.get("wind_basis_left", set()),
        roles.get("wind_basis_right", set()),
    )
    reached = _reachable(bundle.edges, sensory)
    steering_reached = bool(
        roles.get("steer_left", set()) & reached
        and roles.get("steer_right", set()) & reached
    )
    steering_disjoint = roles.get("steer_left", set()).isdisjoint(
        roles.get("steer_right", set())
    )

    probe = probe_trained_candidate(bundle, parameters, config)
    gates = [
        Gate(
            "training_development_gate",
            bool(training_report.get("development_gate_passed")),
            int(bool(training_report.get("development_gate_passed"))),
            "recomputed training report passed its frozen development-improvement gate",
        ),
        Gate(
            "required_roles",
            not missing_roles,
            ",".join(missing_roles) if missing_roles else "complete",
            "bilateral FB5AB context, PFN basis, and DNa02 steering roles are non-empty",
        ),
        Gate(
            "role_body_id_closure",
            role_closure,
            len(role_body_ids),
            "all required role body IDs exist in the reviewed graph",
        ),
        Gate(
            "steering_role_disjointness",
            steering_disjoint,
            int(steering_disjoint),
            "left and right steering role body IDs do not overlap",
        ),
        Gate(
            "signed_edge_fraction",
            signed_fraction >= float(frozen["minimum_signed_edge_fraction"]),
            signed_fraction,
            f">= {float(frozen['minimum_signed_edge_fraction']):.2f}",
        ),
        Gate(
            "structural_reachability",
            steering_reached,
            int(steering_reached),
            "FB5AB-context/PFN-basis roles structurally reach both DNa02 steering roles",
        ),
        Gate(
            "odor_on_wind_separation",
            probe.odor_on_wind_separation
            >= float(frozen["minimum_odor_on_wind_turn_separation"]),
            probe.odor_on_wind_separation,
            f">= {float(frozen['minimum_odor_on_wind_turn_separation']):.3f}",
        ),
        Gate(
            "upwind_laterality",
            probe.upwind_laterality_correct,
            int(probe.upwind_laterality_correct),
            "mirrored crosswind inputs with odor produce opposite turns toward upwind",
        ),
        Gate(
            "odor_gates_pfn_basis_response",
            probe.odor_gating_separation_delta
            >= float(frozen["minimum_odor_gating_separation_delta"]),
            probe.odor_gating_separation_delta,
            f">= {float(frozen['minimum_odor_gating_separation_delta']):.3f}",
        ),
        Gate(
            "odor_direction_invariance",
            probe.odor_laterality_error <= float(frozen["maximum_odor_laterality_error"]),
            probe.odor_laterality_error,
            f"<= {float(frozen['maximum_odor_laterality_error']):.3g}",
        ),
        Gate(
            "steering_lesion_dependency",
            probe.lesioned_peak_turn <= float(frozen["maximum_lesioned_peak_turn"]),
            probe.lesioned_peak_turn,
            f"<= {float(frozen['maximum_lesioned_peak_turn']):.3f}",
        ),
        Gate(
            "deterministic_replay",
            probe.deterministic_error
            <= float(frozen["maximum_deterministic_replay_error"]),
            probe.deterministic_error,
            f"<= {float(frozen['maximum_deterministic_replay_error']):.3g}",
        ),
    ]
    passed = all(gate.passed for gate in gates)
    return {
        "protocol": frozen["protocol"],
        "passed": passed,
        "dataset": (bundle.manifest or {}).get("dataset", "unknown"),
        "graph_sha256": bundle.replay_fingerprint(),
        "training_config_sha256": canonical_sha256(config),
        "training_audit_receipt_sha256": training_report["audit_receipt_sha256"],
        "runtime_environment_sha256": training_report["runtime_environment_sha256"],
        "numerical_runtime_sha256": training_report["numerical_runtime_sha256"],
        "trained_parameter_sha256": canonical_sha256(parameters.to_dict()),
        "trained_parameters": parameters.to_dict(),
        "gates": [asdict(gate) for gate in gates],
        "passed_gate_count": sum(gate.passed for gate in gates),
        "gate_count": len(gates),
        "probe": asdict(probe),
        "coordinate_convention": {
            "positive_turn": "left/counterclockwise",
            "negative_turn": "right/clockwise",
            "wind_vector": "direction air travels (downwind)",
            "pfn_basis_vector": "direction airflow arrives from",
            "wind_y_positive": "air travels left, arrives from right",
            "wind_y_negative": "air travels right, arrives from left",
        },
        "memory_policy": frozen["memory_policy"],
        "warning": (
            "Passing trained E002 supports internal consistency of the explicit modeled dynamics, "
            "the supplied training receipts, reconstructed represented optimizer execution, and "
            "an exact numerical-runtime match to training. It does not prove no external final-test "
            "peeking, measured physiology, peripheral sensory transduction, or final navigation "
            "superiority over matched trained topology controls."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained E002 odor-gated PFN-basis qualification"
    )
    parser.add_argument("bundle", help="reviewed signed candidate GraphBundle")
    parser.add_argument("training_report", help="runtime-sealed task-optimization report")
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument("--output", default="results/e002/trained-qualification-v1.json")
    args = parser.parse_args()

    config = load_training_config(args.config)
    bundle = GraphBundle.load(args.bundle)
    training_report = json.loads(Path(args.training_report).read_text())
    report = qualify_trained_candidate(bundle, training_report, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
