from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "topology-power-analysis-v1"


def _finite_vector(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("pilot null topology means must contain at least two values")
    if not np.isfinite(array).all():
        raise ValueError("pilot null topology means contain non-finite values")
    return array


def simulate_topology_power(
    pilot_null_topology_means: list[float] | np.ndarray,
    *,
    assumed_intact_advantage: float,
    null_counts: tuple[int, ...] = (31, 63, 127),
    alpha: float = 0.05,
    trials: int = 10_000,
    seed: int = 55109,
) -> dict[str, Any]:
    """Estimate randomization-test power with topology as the sampling unit.

    The empirical pilot distribution is resampled to represent topology-to-topology
    variability. A simulated intact topology is drawn from that same distribution and
    shifted by ``assumed_intact_advantage``. Episode replication is intentionally absent
    from this calculation because episodes are not independent connectomes.
    """
    pilot = _finite_vector(pilot_null_topology_means)
    if not np.isfinite(float(assumed_intact_advantage)):
        raise ValueError("assumed_intact_advantage must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if trials <= 0:
        raise ValueError("trials must be > 0")
    if not null_counts or any(int(count) < 1 for count in null_counts):
        raise ValueError("null_counts must contain positive integers")

    rng = np.random.default_rng(int(seed))
    results: list[dict[str, Any]] = []
    for raw_count in null_counts:
        count = int(raw_count)
        significant = 0
        p_values = np.empty(trials, dtype=float)
        deltas = np.empty(trials, dtype=float)
        for trial in range(trials):
            nulls = rng.choice(pilot, size=count, replace=True)
            intact = float(rng.choice(pilot)) + float(assumed_intact_advantage)
            exceed = int(np.sum(nulls >= intact))
            p_value = float((1 + exceed) / (count + 1))
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
        "pilot_null_topology_count": int(len(pilot)),
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
    parser = argparse.ArgumentParser(description="Simulate topology-level randomization-test power")
    parser.add_argument("pilot", help="JSON containing null_topology_means=[...]")
    parser.add_argument("--advantage", type=float, required=True)
    parser.add_argument("--null-count", action="append", type=int, default=[])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=55109)
    parser.add_argument("--output", default="results/statistics/topology-power-v1.json")
    args = parser.parse_args()

    pilot = json.loads(Path(args.pilot).read_text())
    counts = tuple(args.null_count) if args.null_count else (31, 63, 127)
    report = simulate_topology_power(
        pilot["null_topology_means"],
        assumed_intact_advantage=args.advantage,
        null_counts=counts,
        alpha=args.alpha,
        trials=args.trials,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{output}")
    for row in report["results"]:
        print(
            f"n_null={row['null_topology_count']} power={row['estimated_power']:.4f} "
            f"min_p={row['minimum_attainable_empirical_p']:.6f}"
        )


if __name__ == "__main__":
    main()
