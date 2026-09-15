from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "topology-power-analysis-v1"


def empirical_randomization_p(intact: float, null_values: np.ndarray) -> float:
    nulls = np.asarray(null_values, dtype=float)
    if nulls.ndim != 1 or len(nulls) == 0:
        raise ValueError("null_values must be a non-empty 1D array")
    return float((1 + np.sum(nulls >= float(intact))) / (len(nulls) + 1))


def estimate_power(
    pilot_null_values: list[float],
    *,
    assumed_intact_advantage: float,
    candidate_null_counts: list[int],
    trials: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    pilot = np.asarray(pilot_null_values, dtype=float)
    if pilot.ndim != 1 or len(pilot) < 2:
        raise ValueError("at least two pilot topology-null values are required")
    if trials <= 0:
        raise ValueError("trials must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if any(count <= 0 for count in candidate_null_counts):
        raise ValueError("candidate null counts must be positive")

    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for count in candidate_null_counts:
        significant = 0
        p_values = np.empty(trials, dtype=float)
        deltas = np.empty(trials, dtype=float)
        for trial in range(trials):
            nulls = rng.choice(pilot, size=count, replace=True)
            reference = float(rng.choice(pilot))
            intact = reference + float(assumed_intact_advantage)
            p_value = empirical_randomization_p(intact, nulls)
            p_values[trial] = p_value
            deltas[trial] = intact - float(nulls.mean())
            significant += int(p_value <= alpha)
        results.append(
            {
                "null_topology_count": count,
                "minimum_attainable_empirical_p": float(1.0 / (count + 1)),
                "estimated_power": float(significant / trials),
                "median_empirical_p": float(np.median(p_values)),
                "median_intact_minus_mean_null": float(np.median(deltas)),
                "delta_ci95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
            }
        )

    return {
        "protocol": PROTOCOL,
        "topology_is_sampling_unit": True,
        "episode_count_is_not_a_power_parameter": True,
        "pilot_null_topology_count": len(pilot),
        "pilot_null_mean": float(pilot.mean()),
        "pilot_null_std": float(pilot.std(ddof=1)),
        "assumed_intact_advantage": float(assumed_intact_advantage),
        "alpha": float(alpha),
        "trials": int(trials),
        "seed": int(seed),
        "results": results,
        "claim_boundary": (
            "This simulation estimates statistical power under an explicit assumed effect and "
            "pilot topology distribution. It is a design tool, not evidence that the intact "
            "connectome has the assumed advantage."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate topology-level randomization-test power")
    parser.add_argument("pilot")
    parser.add_argument("--assumed-advantage", type=float, required=True)
    parser.add_argument("--null-counts", default="31,63,127")
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", default="results/statistics/topology-power-v1.json")
    args = parser.parse_args()

    pilot_payload = json.loads(Path(args.pilot).read_text())
    if isinstance(pilot_payload, dict):
        values = pilot_payload.get("null_values")
    else:
        values = pilot_payload
    if not isinstance(values, list):
        raise ValueError("pilot file must be a JSON list or contain null_values")
    counts = [int(x) for x in args.null_counts.split(",") if x.strip()]
    report = estimate_power(
        [float(x) for x in values],
        assumed_intact_advantage=args.assumed_advantage,
        candidate_null_counts=counts,
        trials=args.trials,
        alpha=args.alpha,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
