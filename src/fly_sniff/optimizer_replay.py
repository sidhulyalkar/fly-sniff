from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .runtime_provenance import verify_runtime_environment_receipt
from .training import (
    PARAMETER_NAMES,
    _decode_parameters,
    _parameter_bounds,
    canonical_sha256,
    load_training_config,
    make_training_seed_split,
)

PROTOCOL = "task-optimization-cem-replay-v1"
FLOAT_ATOL = 1e-12


def _require_close(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"optimizer replay {name} is missing or invalid") from exc
    if not np.isfinite(value) or not np.isclose(value, expected, rtol=0.0, atol=FLOAT_ATOL):
        raise RuntimeError(
            f"optimizer replay {name} mismatch: observed={value}, expected={expected}"
        )


def _require_parameter_mapping(
    name: str,
    observed: Any,
    expected: dict[str, float],
) -> None:
    if not isinstance(observed, dict) or set(observed) != set(PARAMETER_NAMES):
        raise RuntimeError(f"optimizer replay {name} parameter mapping is malformed")
    for parameter in PARAMETER_NAMES:
        _require_close(
            f"{name}.{parameter}",
            observed[parameter],
            float(expected[parameter]),
        )


def replay_single_optimizer_history(
    report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Replay deterministic CEM mechanics from config, RNG seed, and recorded objectives.

    The replay verifies seed-batch selection, candidate generation, parameter hashes,
    deterministic tie-breaking, elite selection, distribution updates, and the final
    trained parameter hash. It deliberately does not recompute episode objectives.
    """
    if report.get("protocol") != config.get("protocol"):
        raise ValueError("optimizer replay report/config protocol mismatch")
    runtime = report.get("runtime_environment")
    if not isinstance(runtime, dict):
        raise TypeError("optimizer replay requires a runtime-sealed training report")
    numerical_sha = verify_runtime_environment_receipt(
        runtime,
        require_current_numerical_match=True,
    )
    if report.get("numerical_runtime_sha256") != numerical_sha:
        raise RuntimeError("optimizer replay numerical runtime hash mismatch")

    history = report.get("history")
    if not isinstance(history, list):
        raise TypeError("optimizer replay requires generation history")
    history_sha = canonical_sha256(history)
    if report.get("optimizer_history_sha256") != history_sha:
        raise RuntimeError("optimizer replay history hash mismatch")

    train_seeds, _ = make_training_seed_split(config)
    optimizer = config["optimizer"]
    population_size = int(optimizer["population"])
    generations = int(optimizer["generations"])
    elite_fraction = float(optimizer["elite_fraction"])
    episodes_per_candidate = int(optimizer["episodes_per_candidate"])
    update_rate = float(optimizer["update_rate"])
    initial_sigma_fraction = float(optimizer["initial_sigma_fraction_of_log_range"])
    minimum_sigma = float(optimizer["minimum_log_sigma"])
    if len(history) != generations:
        raise RuntimeError("optimizer replay generation count mismatch")

    low, mean, high = _parameter_bounds(config)
    sigma = np.maximum((high - low) * initial_sigma_fraction, minimum_sigma)
    rng = np.random.default_rng(int(optimizer["optimizer_seed"]))
    elite_count = max(2, math.ceil(population_size * elite_fraction))

    replay_generations: list[dict[str, Any]] = []
    for generation_index, generation in enumerate(history):
        if not isinstance(generation, dict):
            raise TypeError("optimizer replay generation entry must be a mapping")
        if int(generation.get("generation", -1)) != generation_index:
            raise RuntimeError("optimizer replay generation index mismatch")

        expected_seeds = [
            int(value)
            for value in rng.choice(
                np.asarray(train_seeds, dtype=np.int64),
                size=episodes_per_candidate,
                replace=False,
            )
        ]
        observed_seeds = [int(value) for value in generation.get("seed_batch", [])]
        if observed_seeds != expected_seeds:
            raise RuntimeError(
                f"optimizer replay seed batch mismatch in generation {generation_index}"
            )
        if generation.get("seed_batch_sha256") != canonical_sha256(expected_seeds):
            raise RuntimeError("optimizer replay seed-batch hash mismatch")

        population = rng.normal(
            loc=mean,
            scale=sigma,
            size=(population_size, len(PARAMETER_NAMES)),
        )
        population = np.clip(population, low, high)
        population[0] = mean

        receipts = generation.get("candidate_receipts")
        if not isinstance(receipts, list) or len(receipts) != population_size:
            raise RuntimeError("optimizer replay candidate receipt count mismatch")
        scores = np.empty(population_size, dtype=float)
        for candidate_index, candidate in enumerate(population):
            receipt = receipts[candidate_index]
            if not isinstance(receipt, dict) or int(receipt.get("index", -1)) != candidate_index:
                raise RuntimeError("optimizer replay candidate index mismatch")
            parameters = _decode_parameters(candidate, config)
            expected_parameter_hash = canonical_sha256(parameters.to_dict())
            if receipt.get("parameter_sha256") != expected_parameter_hash:
                raise RuntimeError(
                    "optimizer replay candidate parameter hash mismatch in "
                    f"generation {generation_index}, candidate {candidate_index}"
                )
            score = float(receipt.get("objective", float("nan")))
            if not np.isfinite(score):
                raise RuntimeError("optimizer replay encountered a nonfinite candidate objective")
            scores[candidate_index] = score

        if (
            generation.get("candidate_tie_break")
            != "descending_objective_then_ascending_candidate_index"
        ):
            raise RuntimeError("optimizer replay tie-break contract mismatch")
        order = np.lexsort((np.arange(population_size, dtype=int), -scores))
        elite = population[order[:elite_count]]
        elite_mean = np.mean(elite, axis=0)
        elite_sigma = np.std(elite, axis=0)
        next_mean = (1.0 - update_rate) * mean + update_rate * elite_mean
        next_sigma = np.maximum(
            (1.0 - update_rate) * sigma + update_rate * elite_sigma,
            minimum_sigma,
        )
        best_index = int(order[0])

        _require_close(
            f"generation[{generation_index}].population_mean_objective",
            generation.get("population_mean_objective"),
            float(np.mean(scores)),
        )
        _require_close(
            f"generation[{generation_index}].elite_mean_objective",
            generation.get("elite_mean_objective"),
            float(np.mean(scores[order[:elite_count]])),
        )
        _require_close(
            f"generation[{generation_index}].best_objective",
            generation.get("best_objective"),
            float(scores[best_index]),
        )
        _require_parameter_mapping(
            f"generation[{generation_index}].best_parameters",
            generation.get("best_parameters"),
            _decode_parameters(population[best_index], config).to_dict(),
        )
        _require_parameter_mapping(
            f"generation[{generation_index}].distribution_mean_parameters",
            generation.get("distribution_mean_parameters"),
            _decode_parameters(next_mean, config).to_dict(),
        )
        sigma_mapping = generation.get("distribution_log_sigma")
        if not isinstance(sigma_mapping, dict) or set(sigma_mapping) != set(PARAMETER_NAMES):
            raise RuntimeError("optimizer replay distribution sigma mapping is malformed")
        for parameter, expected_sigma in zip(PARAMETER_NAMES, next_sigma, strict=True):
            _require_close(
                f"generation[{generation_index}].distribution_log_sigma.{parameter}",
                sigma_mapping[parameter],
                float(expected_sigma),
            )

        replay_generations.append(
            {
                "generation": generation_index,
                "seed_batch_sha256": canonical_sha256(expected_seeds),
                "population_parameter_sha256": canonical_sha256(
                    [receipt["parameter_sha256"] for receipt in receipts]
                ),
                "best_candidate_index": best_index,
                "best_parameter_sha256": receipts[best_index]["parameter_sha256"],
            }
        )
        mean = next_mean
        sigma = next_sigma

    final_parameters = _decode_parameters(mean, config)
    final_mapping = final_parameters.to_dict()
    _require_parameter_mapping("trained_parameters", report.get("trained_parameters"), final_mapping)
    final_sha = canonical_sha256(final_mapping)
    if report.get("trained_parameter_sha256") != final_sha:
        raise RuntimeError("optimizer replay final trained-parameter hash mismatch")

    return {
        "protocol": PROTOCOL,
        "status": "deterministic_cem_mechanics_replayed",
        "optimizer_seed": int(optimizer["optimizer_seed"]),
        "generation_count": generations,
        "population": population_size,
        "optimizer_history_sha256": history_sha,
        "numerical_runtime_sha256": numerical_sha,
        "trained_parameter_sha256": final_sha,
        "generations": replay_generations,
        "claim_boundary": (
            "This replay proves deterministic optimizer mechanics under the sealed numerical "
            "runtime given the recorded candidate objectives. It does not independently recompute "
            "those objectives from simulator episodes or prove absence of external experiments."
        ),
    }


def replay_matched_optimizer_history(
    matched_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if matched_report.get("protocol") != "matched-task-optimization-controls-v1":
        raise ValueError("not a matched task-optimization v1 report")
    results = matched_report.get("results")
    if not isinstance(results, dict):
        raise TypeError("matched optimizer replay is missing results")
    intact = results.get("intact")
    lesion = results.get("lesion")
    rewires = results.get("rewires")
    if not isinstance(intact, dict) or not isinstance(lesion, dict):
        raise TypeError("matched optimizer replay is missing intact or lesion report")
    if not isinstance(rewires, dict):
        raise TypeError("matched optimizer replay is missing rewire reports")

    topology_reports: list[tuple[str, dict[str, Any]]] = [("intact", intact)]
    topology_reports.extend(
        (f"rewire:{seed}", report) for seed, report in sorted(rewires.items())
    )
    topology_reports.append(("lesion", lesion))
    replays = {
        label: replay_single_optimizer_history(report, config)
        for label, report in topology_reports
    }
    numerical_hashes = {item["numerical_runtime_sha256"] for item in replays.values()}
    if len(numerical_hashes) != 1:
        raise RuntimeError("matched optimizer replay found multiple numerical runtimes")
    return {
        "protocol": "matched-task-optimization-cem-replay-v1",
        "status": "all_topology_optimizer_mechanics_replayed",
        "matched_report_sha256": canonical_sha256(matched_report),
        "training_config_sha256": canonical_sha256(config),
        "topology_count": len(replays),
        "numerical_runtime_sha256": next(iter(numerical_hashes)),
        "topologies": replays,
        "claim_boundary": (
            "Every topology's CEM mechanics reproduce from the frozen optimizer seed under the "
            "sealed numerical runtime and recorded objectives. Episode objective recomputation "
            "remains a separate audit layer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay deterministic CEM mechanics from a runtime-sealed training artifact"
    )
    parser.add_argument("training_report", help="single or matched runtime-sealed training report")
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument(
        "--output",
        default="results/training-redteam/cem-replay-v1.json",
    )
    args = parser.parse_args()

    config = load_training_config(args.config)
    report = json.loads(Path(args.training_report).read_text())
    if report.get("protocol") == "matched-task-optimization-controls-v1":
        replay = replay_matched_optimizer_history(report, config)
    else:
        replay = replay_single_optimizer_history(report, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(output)
    print(canonical_sha256(replay))


if __name__ == "__main__":
    main()
