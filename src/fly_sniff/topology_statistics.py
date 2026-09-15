from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "topology-randomization-inference-v1"


def _as_finite_vector(values: list[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def topology_randomization_report(
    intact_values: list[float] | np.ndarray,
    null_values: dict[str, list[float] | np.ndarray],
    *,
    bootstrap_seed: int = 73031,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Compare one intact topology with an ensemble of topology nulls.

    Episodes are paired within topology, but the unit of randomization for the
    topology claim is the null graph. Replicating episodes therefore improves each
    topology's mean estimate without increasing the reported topology sample size.
    """
    intact = _as_finite_vector(intact_values, name="intact_values")
    if len(null_values) < 1:
        raise ValueError("at least one null topology is required")
    null_arrays: dict[str, np.ndarray] = {}
    for name, values in sorted(null_values.items()):
        array = _as_finite_vector(values, name=f"null[{name}]")
        if len(array) != len(intact):
            raise ValueError("paired topology comparison requires identical episode counts")
        null_arrays[name] = array

    null_names = list(null_arrays)
    null_means = np.asarray([float(null_arrays[name].mean()) for name in null_names])
    intact_mean = float(intact.mean())
    null_mean = float(null_means.mean())
    delta = float(intact_mean - null_mean)
    exceed = int(np.sum(null_means >= intact_mean))
    empirical_p = float((1 + exceed) / (len(null_means) + 1))
    percentile = float(100.0 * np.mean(null_means < intact_mean))

    paired_delta_means = np.asarray(
        [float(np.mean(intact - null_arrays[name])) for name in null_names],
        dtype=float,
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be > 0")
    indices = rng.integers(0, len(null_means), size=(int(bootstrap_samples), len(null_means)))
    sampled_null_means = null_means[indices].mean(axis=1)
    sampled_delta = intact_mean - sampled_null_means
    ci_low, ci_high = np.quantile(sampled_delta, [0.025, 0.975])

    return {
        "protocol": PROTOCOL,
        "topology_is_unit_of_randomization": True,
        "episode_count_must_not_be_treated_as_connectome_count": True,
        "topology_counts": {"intact": 1, "null": len(null_means)},
        "paired_episode_count_per_topology": len(intact),
        "intact_mean": intact_mean,
        "null_topology_means": {
            name: float(value) for name, value in zip(null_names, null_means, strict=True)
        },
        "mean_null_topology_mean": null_mean,
        "intact_minus_mean_null": delta,
        "intact_percentile_among_nulls": percentile,
        "empirical_randomization_p_greater": empirical_p,
        "null_topologies_at_or_above_intact": exceed,
        "minimum_attainable_empirical_p": float(1.0 / (len(null_means) + 1)),
        "paired_delta_by_topology": {
            name: float(value)
            for name, value in zip(null_names, paired_delta_means, strict=True)
        },
        "topology_bootstrap": {
            "seed": int(bootstrap_seed),
            "samples": int(bootstrap_samples),
            "intact_minus_mean_null_ci95": [float(ci_low), float(ci_high)],
            "resampling_unit": "null_topology",
        },
        "claim_boundary": (
            "This report performs topology-level randomization inference for the supplied metric. "
            "It does not by itself establish biological generalization, valid sensory modeling, "
            "or correct dynamics calibration."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topology-level randomization inference")
    parser.add_argument("input", help="JSON with intact=[...] and nulls={name:[...]} arrays")
    parser.add_argument("--bootstrap-seed", type=int, default=73031)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", default="results/statistics/topology-randomization-v1.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    report = topology_randomization_report(
        payload["intact"],
        payload["nulls"],
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{output}")
    print(
        f"delta={report['intact_minus_mean_null']:.6f} "
        f"p={report['empirical_randomization_p_greater']:.6f} "
        f"null_topologies={report['topology_counts']['null']}"
    )


if __name__ == "__main__":
    main()
