import pytest

from fly_sniff.choice import benchmark_choice, choice_observation, run_choice
from fly_sniff.controllers import BilateralProxyController, RandomWalkController


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
    assert left.correct and left.committed
    assert right.correct and right.committed
    assert left.mean_turn > 0
    assert right.mean_turn < 0


def test_proxy_motor_diagnostics_follow_turn_side():
    controller = BilateralProxyController()
    controller.reset(7)
    left_action = controller.act(choice_observation("left"))
    left_diag = controller.diagnostics()
    assert left_action.turn > 0
    assert left_diag["dn_left"] > left_diag["dn_right"]

    controller.reset(7)
    right_action = controller.act(choice_observation("right"))
    right_diag = controller.diagnostics()
    assert right_action.turn < 0
    assert right_diag["dn_right"] > right_diag["dn_left"]


def test_balanced_proxy_benchmark_is_deterministic():
    a = benchmark_choice(BilateralProxyController(), trials=20, seed=11)
    b = benchmark_choice(BilateralProxyController(), trials=20, seed=11)
    assert a["accuracy"] == b["accuracy"]
    assert a["commitment_rate"] == b["commitment_rate"]
    assert a["mean_signed_turn_margin"] == b["mean_signed_turn_margin"]
    assert a["accuracy"] > 0.5
    assert a["commitment_rate"] == 1.0


def test_random_forced_choice_stays_near_fifty_percent():
    report = benchmark_choice(RandomWalkController(), trials=1000, seed=13013)
    assert report["chance_accuracy"] == 0.5
    assert 0.45 <= report["accuracy"] <= 0.55
    assert abs(report["mean_signed_turn_margin"]) < 0.03
