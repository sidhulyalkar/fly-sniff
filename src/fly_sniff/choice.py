from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np

from .controllers import (
    BilateralProxyController,
    CastSurgeController,
    Controller,
    RandomWalkController,
)
from .env import Observation


@dataclass(frozen=True)
class ChoiceResult:
    seed: int
    source_side: str
    mean_turn: float
    correct: bool
    committed: bool


def choice_observation(source_side: str, *, strong: float = 0.75, weak: float = 0.20) -> Observation:
    """Return a minimal bilateral odor observation for the E002A steering gate.

    The agent is already facing upwind (heading=pi), so the assay isolates the
    left/right sensory-to-steering mapping rather than rewarding a generic
    upwind turn. This is a modeled sensory pulse, not a plume-navigation result.
    """
    if source_side not in {"left", "right"}:
        raise ValueError("source_side must be 'left' or 'right'")
    left, right = (strong, weak) if source_side == "left" else (weak, strong)
    return Observation(
        left_odor=float(left),
        right_odor=float(right),
        mean_odor=0.5 * (left + right),
        odor_delta=float(right - left),
        wind_x_body=0.0,
        wind_y_body=0.0,
        heading=float(np.pi),
    )


def run_choice(
    controller: Controller,
    source_side: str,
    *,
    seed: int,
    pulse_steps: int = 24,
    commitment_threshold: float = 0.05,
) -> ChoiceResult:
    """Ask whether a controller turns toward a lateral odor pulse.

    Correctness is a forced left/right choice based only on turn sign, so an
    odor-blind symmetric controller has a 50% chance baseline. Turn magnitude is
    reported separately as commitment rather than silently changing the null.
    """
    if pulse_steps < 4:
        raise ValueError("pulse_steps must be >= 4")
    if commitment_threshold < 0.0:
        raise ValueError("commitment_threshold must be >= 0")
    obs = choice_observation(source_side)
    controller.reset(seed + 101)
    turns = np.asarray([controller.act(obs).turn for _ in range(pulse_steps)], dtype=float)
    tail = turns[-max(4, pulse_steps // 4) :]
    mean_turn = float(np.mean(tail))
    expected_sign = 1.0 if source_side == "left" else -1.0
    correct = bool(expected_sign * mean_turn > 0.0)
    committed = bool(abs(mean_turn) > commitment_threshold)
    return ChoiceResult(
        seed=seed,
        source_side=source_side,
        mean_turn=mean_turn,
        correct=correct,
        committed=committed,
    )


def benchmark_choice(
    controller: Controller,
    *,
    trials: int = 100,
    seed: int = 13013,
) -> dict:
    """Run balanced left/right E002A trials and summarize accuracy."""
    if trials < 2:
        raise ValueError("trials must be >= 2")
    results: list[ChoiceResult] = []
    for index in range(trials):
        side = "left" if index % 2 == 0 else "right"
        trial_seed = seed + index * 9973
        results.append(run_choice(controller, side, seed=trial_seed))
    accuracy = float(np.mean([result.correct for result in results]))
    commitment_rate = float(np.mean([result.committed for result in results]))
    margin = float(
        np.mean(
            [
                (1.0 if result.source_side == "left" else -1.0) * result.mean_turn
                for result in results
            ]
        )
    )
    return {
        "protocol": "E002A-two-choice-sniff-v1",
        "controller": controller.name,
        "trials": trials,
        "accuracy": accuracy,
        "commitment_rate": commitment_rate,
        "mean_signed_turn_margin": margin,
        "chance_accuracy": 0.5,
        "claim_status": "development-only",
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the development E002A two-choice sniff assay")
    parser.add_argument("--controller", choices=["proxy", "classical", "random"], default="proxy")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13013)
    args = parser.parse_args()
    factories = {
        "proxy": BilateralProxyController,
        "classical": CastSurgeController,
        "random": RandomWalkController,
    }
    report = benchmark_choice(factories[args.controller](), trials=args.trials, seed=args.seed)
    print(
        f"{report['controller']}: accuracy={report['accuracy']:.3f} "
        f"committed={report['commitment_rate']:.3f} "
        f"margin={report['mean_signed_turn_margin']:+.3f} trials={report['trials']}"
    )
