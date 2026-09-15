from __future__ import annotations

from fly_sniff.topology_power import simulate_topology_power


def test_power_report_uses_topologies_not_episodes() -> None:
    report = simulate_topology_power(
        [0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
        assumed_intact_advantage=0.10,
        null_counts=(63,),
        trials=500,
        seed=7,
    )
    assert report["topology_is_sampling_unit"] is True
    assert report["episode_count_is_not_a_power_parameter"] is True
    row = report["results"][0]
    assert row["null_topology_count"] == 63
    assert row["minimum_attainable_empirical_p"] == 1 / 64
    assert 0.0 <= row["estimated_power"] <= 1.0


def test_larger_assumed_effect_increases_power_for_same_seeded_design() -> None:
    pilot = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    low = simulate_topology_power(
        pilot,
        assumed_intact_advantage=0.01,
        null_counts=(63,),
        trials=1000,
        seed=19,
    )
    high = simulate_topology_power(
        pilot,
        assumed_intact_advantage=0.20,
        null_counts=(63,),
        trials=1000,
        seed=19,
    )
    assert high["results"][0]["estimated_power"] > low["results"][0]["estimated_power"]
