import pandas as pd
import pytest

from fly_sniff.config import ArenaConfig, PlumeConfig, SensorConfig
from fly_sniff.controllers import RandomWalkController
from fly_sniff.evaluate import evaluate, paired_success_report


def test_evaluate_accepts_frozen_environment_configs():
    arena = ArenaConfig(max_steps=2)
    plume = PlumeConfig(warmup_s=0.0, emission_rate_hz=0.0)
    sensors = SensorConfig()
    frame = evaluate(
        {"random": RandomWalkController},
        [1, 2],
        arena=arena,
        plume=plume,
        sensors=sensors,
    )
    assert len(frame) == 2
    assert set(frame.label) == {"random"}


def test_paired_success_report_uses_seed_matched_discordant_pairs():
    frame = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2, 3, 3, 4, 4],
            "label": ["a", "b"] * 4,
            "success": [True, False, True, True, False, False, True, False],
        }
    )

    report = paired_success_report(frame, "a", "b", bootstrap_seed=7, n_boot=1000)

    assert report["n"] == 4
    assert report["a_success_rate"] == pytest.approx(0.75)
    assert report["b_success_rate"] == pytest.approx(0.25)
    assert report["mean_delta"] == pytest.approx(0.50)
    assert report["a_only_successes"] == 2
    assert report["b_only_successes"] == 0
    assert report["discordant_pairs"] == 2
    assert report["mcnemar_exact_p"] == pytest.approx(0.5)
