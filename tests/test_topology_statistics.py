from __future__ import annotations

import numpy as np

from fly_sniff.topology_statistics import topology_randomization_report


def test_63_nulls_give_expected_minimum_empirical_p() -> None:
    intact = [1.0, 1.0, 1.0, 1.0]
    nulls = {f"null-{index:02d}": [0.0, 0.0, 0.0, 0.0] for index in range(63)}
    report = topology_randomization_report(
        intact,
        nulls,
        bootstrap_seed=1,
        bootstrap_samples=100,
    )
    assert report["topology_counts"] == {"intact": 1, "null": 63}
    assert report["empirical_randomization_p_greater"] == 1 / 64
    assert report["minimum_attainable_empirical_p"] == 1 / 64
    assert report["intact_percentile_among_nulls"] == 100.0
    assert report["topology_bootstrap"]["resampling_unit"] == "null_topology"


def test_episode_replication_does_not_change_topology_sample_count() -> None:
    nulls_short = {"a": [0.0, 0.2], "b": [0.1, 0.1]}
    short = topology_randomization_report(
        [0.5, 0.5],
        nulls_short,
        bootstrap_samples=50,
    )
    nulls_long = {
        name: np.repeat(values, 50).tolist()
        for name, values in nulls_short.items()
    }
    long = topology_randomization_report(
        np.repeat([0.5, 0.5], 50).tolist(),
        nulls_long,
        bootstrap_samples=50,
    )
    assert short["topology_counts"] == long["topology_counts"] == {"intact": 1, "null": 2}
    assert short["paired_episode_count_per_topology"] == 2
    assert long["paired_episode_count_per_topology"] == 100
    assert short["empirical_randomization_p_greater"] == long["empirical_randomization_p_greater"]


def test_paired_episode_lengths_must_match() -> None:
    try:
        topology_randomization_report([1.0, 2.0], {"bad": [1.0]}, bootstrap_samples=10)
    except ValueError as exc:
        assert "identical episode counts" in str(exc)
    else:
        raise AssertionError("mismatched paired episodes were accepted")
