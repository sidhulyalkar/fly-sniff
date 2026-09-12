from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .env import FlySniffEnv, Observation
from .graph import GraphBundle
from .metrics import shortest_path_to_goal_region, spl
from .trained_qualification import parameters_from_training_report
from .training import (
    FINAL_TEST_MAX_SEED,
    TaskOptimizedMaleCNSController,
    canonical_sha256,
    make_training_seed_split,
)

DIAGNOSTIC_PROTOCOL = "task-optimization-shortcut-redteam-v1"
DIAGNOSTIC_SEED_FLOOR = 2_200_000_000
DEFAULT_DIAGNOSTIC_EPISODES = 32


def make_diagnostic_seeds(
    config: dict[str, Any],
    *,
    count: int = DEFAULT_DIAGNOSTIC_EPISODES,
) -> list[int]:
    """Create a deterministic namespace disjoint from train/validation/final seeds."""
    if count < 1:
        raise ValueError("diagnostic seed count must be positive")
    train, validation = make_training_seed_split(config)
    namespace = config["seed_namespace"]
    development_end = int(namespace["base"]) + int(namespace["span"]) - 1
    start = max(DIAGNOSTIC_SEED_FLOOR, development_end + 1)
    seeds = list(range(start, start + int(count)))
    if min(seeds) <= FINAL_TEST_MAX_SEED:
        raise RuntimeError("diagnostic seeds overlap the final-test namespace")
    if set(seeds) & (set(train) | set(validation)):
        raise RuntimeError("diagnostic seeds overlap training/development seeds")
    return seeds


def mask_observation(
    observation: Observation,
    *,
    odor_scale: float = 1.0,
    wind_scale: float = 1.0,
) -> Observation:
    """Apply an explicit sensory ablation at the controller boundary."""
    if not np.isfinite(odor_scale) or not 0.0 <= odor_scale <= 1.0:
        raise ValueError("odor_scale must be finite and in [0, 1]")
    if not np.isfinite(wind_scale) or not 0.0 <= wind_scale <= 1.0:
        raise ValueError("wind_scale must be finite and in [0, 1]")
    return Observation(
        left_odor=float(observation.left_odor * odor_scale),
        right_odor=float(observation.right_odor * odor_scale),
        mean_odor=float(observation.mean_odor * odor_scale),
        odor_delta=float(observation.odor_delta * odor_scale),
        wind_x_body=float(observation.wind_x_body * wind_scale),
        wind_y_body=float(observation.wind_y_body * wind_scale),
        heading=float(observation.heading),
    )


def predict_wall_contact(
    env: FlySniffEnv,
    *,
    turn_command: float,
    speed_scale: float,
) -> tuple[bool, bool]:
    """Predict whether the commanded physical step crosses an x or y boundary.

    This exactly mirrors the kinematic part of ``FlySniffEnv.step`` up to, but not
    including, wall reflection. It is used only for post-hoc diagnostic telemetry;
    no wall distance/contact signal is exposed to the controller.
    """
    arena = env.arena
    agent = env.agent
    turn = float(np.clip(turn_command, -1.0, 1.0)) * arena.max_turn_rate
    heading = float((agent.heading + turn * arena.dt + np.pi) % (2 * np.pi) - np.pi)
    speed = arena.speed * float(np.clip(speed_scale, 0.0, 1.5))
    proposed_x = float(agent.x + np.cos(heading) * speed * arena.dt)
    proposed_y = float(agent.y + np.sin(heading) * speed * arena.dt)
    x_contact = bool(proposed_x < 0.0 or proposed_x > arena.width)
    y_contact = bool(proposed_y < 0.0 or proposed_y > arena.height)
    return x_contact, y_contact


def _run_diagnostic_episode(
    bundle: GraphBundle,
    parameters,
    seed: int,
    *,
    arena: ArenaConfig,
    plume: PlumeConfig,
    sensors: SensorConfig,
    odor_scale: float,
    wind_scale: float,
) -> dict[str, Any]:
    env = FlySniffEnv(seed=seed, arena=arena, plume=plume, sensors=sensors)
    controller = TaskOptimizedMaleCNSController(
        bundle,
        parameters,
        model_dt_s=float(arena.dt),
        require_qualified=False,
    )
    controller.reset(int(seed) + 101)
    initial_distance = env.distance_to_source
    shortest = shortest_path_to_goal_region(initial_distance, arena.source_radius)
    observation = mask_observation(
        env.observe(),
        odor_scale=odor_scale,
        wind_scale=wind_scale,
    )
    done = False
    wall_contact_steps = 0
    x_wall_contact_steps = 0
    y_wall_contact_steps = 0
    first_wall_contact_step: int | None = None
    while not done:
        action = controller.act(observation)
        x_contact, y_contact = predict_wall_contact(
            env,
            turn_command=action.turn,
            speed_scale=action.speed,
        )
        if x_contact or y_contact:
            wall_contact_steps += 1
            if first_wall_contact_step is None:
                first_wall_contact_step = int(env.agent.steps + 1)
        x_wall_contact_steps += int(x_contact)
        y_wall_contact_steps += int(y_contact)
        raw_observation, done = env.step(action.turn, action.speed)
        observation = mask_observation(
            raw_observation,
            odor_scale=odor_scale,
            wind_scale=wind_scale,
        )
    steps = int(env.agent.steps)
    return {
        "seed": int(seed),
        "success": bool(env.agent.found),
        "spl": float(spl(env.agent.found, shortest, env.agent.path_length)),
        "path_length": float(env.agent.path_length),
        "final_distance": float(env.distance_to_source),
        "steps": steps,
        "wall_contact_steps": int(wall_contact_steps),
        "x_wall_contact_steps": int(x_wall_contact_steps),
        "y_wall_contact_steps": int(y_wall_contact_steps),
        "wall_contact_fraction_steps": float(wall_contact_steps / max(steps, 1)),
        "first_wall_contact_step": first_wall_contact_step,
    }


def _conditional_success_rate(
    rows: list[dict[str, Any]],
    *,
    has_wall_contact: bool,
) -> float | None:
    selected = [row for row in rows if bool(row["wall_contact_steps"] > 0) is has_wall_contact]
    if not selected:
        return None
    return float(np.mean([bool(row["success"]) for row in selected]))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("shortcut diagnostic requires at least one episode")
    contact_rows = [row for row in rows if int(row["wall_contact_steps"]) > 0]
    successful_rows = [row for row in rows if bool(row["success"])]
    successful_with_wall = [
        row for row in successful_rows if int(row["wall_contact_steps"]) > 0
    ]
    first_contacts = [
        int(row["first_wall_contact_step"])
        for row in contact_rows
        if row["first_wall_contact_step"] is not None
    ]
    return {
        "n": len(rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "mean_spl": float(np.mean([float(row["spl"]) for row in rows])),
        "mean_path_length": float(np.mean([float(row["path_length"]) for row in rows])),
        "mean_final_distance": float(np.mean([float(row["final_distance"]) for row in rows])),
        "episodes_with_wall_contact_rate": float(len(contact_rows) / len(rows)),
        "mean_wall_contact_steps": float(
            np.mean([int(row["wall_contact_steps"]) for row in rows])
        ),
        "mean_wall_contact_fraction_steps": float(
            np.mean([float(row["wall_contact_fraction_steps"]) for row in rows])
        ),
        "mean_first_wall_contact_step_if_any": (
            float(np.mean(first_contacts)) if first_contacts else None
        ),
        "success_rate_without_wall_contact": _conditional_success_rate(
            rows,
            has_wall_contact=False,
        ),
        "success_rate_with_wall_contact": _conditional_success_rate(
            rows,
            has_wall_contact=True,
        ),
        "successful_episodes_with_wall_contact_rate": (
            float(len(successful_with_wall) / len(successful_rows)) if successful_rows else None
        ),
    }


def run_shortcut_redteam(
    bundle: GraphBundle,
    training_report: dict[str, Any],
    config: dict[str, Any],
    *,
    episode_count: int = DEFAULT_DIAGNOSTIC_EPISODES,
) -> dict[str, Any]:
    """Measure sensory/geometry shortcuts without changing or selecting parameters.

    These are deliberately diagnostics, not qualification gates. Adding a threshold
    after inspecting their outcomes would convert a red-team measurement into a new
    post-hoc model-selection channel.
    """
    parameters = parameters_from_training_report(training_report, bundle, config)
    seeds = make_diagnostic_seeds(config, count=episode_count)
    arena = ArenaConfig()
    plume = PlumeConfig()
    sensors = SensorConfig()

    mirrored_arena = replace(
        arena,
        source_x=float(arena.width - arena.source_x),
        start_x=float(arena.width - arena.start_x),
    )
    mirrored_plume = replace(plume, wind_speed=-float(plume.wind_speed))
    scenarios = {
        "normal": (arena, plume, 1.0, 1.0),
        "odor_clamped": (arena, plume, 0.0, 1.0),
        "wind_clamped": (arena, plume, 1.0, 0.0),
        "odor_and_wind_clamped": (arena, plume, 0.0, 0.0),
        "source_crosswind_low": (
            replace(arena, source_y=float(0.25 * arena.height)),
            plume,
            1.0,
            1.0,
        ),
        "source_crosswind_high": (
            replace(arena, source_y=float(0.75 * arena.height)),
            plume,
            1.0,
            1.0,
        ),
        "mirrored_world": (mirrored_arena, mirrored_plume, 1.0, 1.0),
    }

    scenario_reports: dict[str, Any] = {}
    for name, (scenario_arena, scenario_plume, odor_scale, wind_scale) in scenarios.items():
        rows = [
            _run_diagnostic_episode(
                bundle,
                parameters,
                seed,
                arena=scenario_arena,
                plume=scenario_plume,
                sensors=sensors,
                odor_scale=odor_scale,
                wind_scale=wind_scale,
            )
            for seed in seeds
        ]
        scenario_reports[name] = {
            "arena": asdict(scenario_arena),
            "plume": asdict(scenario_plume),
            "odor_scale": float(odor_scale),
            "wind_scale": float(wind_scale),
            "summary": _summarize(rows),
        }

    normal = scenario_reports["normal"]["summary"]
    odor_clamped = scenario_reports["odor_clamped"]["summary"]
    wind_clamped = scenario_reports["wind_clamped"]["summary"]
    both_clamped = scenario_reports["odor_and_wind_clamped"]["summary"]
    comparisons = {
        "normal_minus_odor_clamped_success_rate": float(
            normal["success_rate"] - odor_clamped["success_rate"]
        ),
        "normal_minus_odor_clamped_mean_spl": float(
            normal["mean_spl"] - odor_clamped["mean_spl"]
        ),
        "normal_minus_wind_clamped_success_rate": float(
            normal["success_rate"] - wind_clamped["success_rate"]
        ),
        "normal_minus_wind_clamped_mean_spl": float(
            normal["mean_spl"] - wind_clamped["mean_spl"]
        ),
        "normal_minus_both_clamped_success_rate": float(
            normal["success_rate"] - both_clamped["success_rate"]
        ),
        "normal_minus_both_clamped_mean_spl": float(
            normal["mean_spl"] - both_clamped["mean_spl"]
        ),
        "normal_minus_odor_clamped_wall_contact_rate": float(
            normal["episodes_with_wall_contact_rate"]
            - odor_clamped["episodes_with_wall_contact_rate"]
        ),
        "normal_minus_both_clamped_wall_contact_rate": float(
            normal["episodes_with_wall_contact_rate"]
            - both_clamped["episodes_with_wall_contact_rate"]
        ),
        "normal_minus_both_clamped_mean_wall_contact_fraction_steps": float(
            normal["mean_wall_contact_fraction_steps"]
            - both_clamped["mean_wall_contact_fraction_steps"]
        ),
    }

    return {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "status": "diagnostic_only_not_a_gate",
        "graph_sha256": bundle.replay_fingerprint(),
        "training_config_sha256": canonical_sha256(config),
        "training_audit_receipt_sha256": training_report["audit_receipt_sha256"],
        "trained_parameter_sha256": training_report["trained_parameter_sha256"],
        "diagnostic_seeds": seeds,
        "diagnostic_seed_sha256": canonical_sha256(seeds),
        "scenarios": scenario_reports,
        "comparisons": comparisons,
        "wall_telemetry": {
            "status": "diagnostic_only_not_controller_input",
            "method": (
                "For each action, predict whether the commanded physical step crosses an arena "
                "boundary before FlySniffEnv applies reflection. No wall distance/contact signal "
                "is provided to the controller."
            ),
        },
        "selection_policy": (
            "These outcomes must not alter v1 parameters, role membership, training thresholds, "
            "or final-test policy. They diagnose shortcut dependence only."
        ),
        "claim_boundary": (
            "A large intact score alone does not establish odor-source navigation if odor-clamped, "
            "geometry-counterfactual, or wall-reflection-dependent runs retain comparable "
            "performance. Conversely, these diagnostics are not preregistered pass/fail gates "
            "and must not be thresholded after observing the result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diagnostic-only shortcut attacks on a frozen trained controller"
    )
    parser.add_argument("bundle", help="reviewed signed GraphBundle directory")
    parser.add_argument("training_report", help="frozen single-graph training report")
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_DIAGNOSTIC_EPISODES,
        help="diagnostic episodes per counterfactual; does not affect training",
    )
    parser.add_argument(
        "--output",
        default="results/training-redteam/shortcut-diagnostics-v1.json",
    )
    args = parser.parse_args()

    from .training import load_training_config

    config = load_training_config(args.config)
    bundle = GraphBundle.load(args.bundle)
    training_report = json.loads(Path(args.training_report).read_text())
    report = run_shortcut_redteam(
        bundle,
        training_report,
        config,
        episode_count=args.episodes,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(output)
    print(canonical_sha256(report))


if __name__ == "__main__":
    main()
