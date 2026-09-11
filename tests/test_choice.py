import pytest

from fly_sniff.choice import benchmark_choice, choice_observation, run_choice
from fly_sniff.controllers import BilateralProxyController


def test_choice_observation_is_mirrored():
    left = choice_observation("left")
    right = choice_observation("right")
    assert left.left_odor == right.right_odor
    assert left.right_odor == right.left_odor
    assert left.odor_delta == -right.odor_delta


def test_choice_rejects_invalid_side():
    with pytest.raises(ValueError, match="source_side"):
        choice_observation("up")


def test_proxy_turns_toward_mirrored_odor_sources():
    left = run_choice(BilateralProxyController(), "left", seed=7)
    right = run_choice(BilateralProxyController(), "right", seed=7)
    assert left.correct
    assert right.correct
    assert left.mean_turn > 0
    assert right.mean_turn < 0


def test_balanced_proxy_benchmark_is_deterministic():
    a = benchmark_choice(BilateralProxyController(), trials=20, seed=11)
    b = benchmark_choice(BilateralProxyController(), trials=20, seed=11)
    assert a["accuracy"] == b["accuracy"]
    assert a["mean_signed_turn_margin"] == b["mean_signed_turn_margin"]
    assert a["accuracy"] > 0.5
