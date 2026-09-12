from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .controllers import Action
from .env import FlySniffEnv, Observation
from .freeze import make_rewire_seeds
from .graph import GraphBundle, MaleCNSRateController
from .metrics import shortest_path_to_goal_region, spl
from .rewire import degree_preserving_rewire, lesion_incoming_to_roles

PARAMETER_NAMES = (
    "tau_s",
    "activation_gain",
    "recurrent_gain",
    "odor_gain",
    "wind_forward_gain",
    "wind_backward_gain",
    "wind_cross_gain",
    "turn_gain",
)
FINAL_TEST_MAX_SEED = 1_999_999_999


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DynamicsParameters:
    tau_s: float
    activation_gain: float
    recurrent_gain: float
    odor_gain: float
    wind_forward_gain: float
    wind_backward_gain: float
    wind_cross_gain: float
    turn_gain: float

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> DynamicsParameters:
        missing = [name for name in PARAMETER_NAMES if name not in values]
        extra = sorted(set(values) - set(PARAMETER_NAMES))
        if missing or extra:
            raise ValueError(f"parameter mismatch: missing={missing}, extra={extra}")
        return cls(**{name: float(values[name]) for name in PARAMETER_NAMES})

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in PARAMETER_NAMES}


class TaskOptimizedMaleCNSController(MaleCNSRateController):
    """Fixed-topology model with eight task-optimized global parameters.

    The connectome, roles, structural synapse counts, and edge signs are immutable.
    Odor enters this v1 model as a nondirectional presence signal, while directional
    information comes from body-frame airflow. This mirrors the primary functional
    prior for FB5AB/PFN/hDeltaC rather than inventing odor laterality at that stage.
    """

    name = "malecns-rate-task-optimized-v1"

    def __init__(
        self,
        bundle: GraphBundle,
        parameters: DynamicsParameters,
        *,
        model_dt_s: float = 0.05,
        require_qualified: bool = True,
    ):
        self.parameters = parameters
        self.recurrent_gain = float(parameters.recurrent_gain)
        self.odor_gain = float(parameters.odor_gain)
        self.wind_forward_gain = float(parameters.wind_forward_gain)
        self.wind_backward_gain = float(parameters.wind_backward_gain)
        self.wind_cross_gain = float(parameters.wind_cross_gain)
        for name in (
            "recurrent_gain",
            "odor_gain",
            "wind_forward_gain",
            "wind_backward_gain",
            "wind_cross_gain",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        super().__init__(
            bundle,
            tau_s=parameters.tau_s,
            gain=parameters.activation_gain,
            model_dt_s=model_dt_s,
            turn_gain=parameters.turn_gain,
            require_qualified=require_qualified,
        )
        self.parameter_sha256 = canonical_sha256(parameters.to_dict())

    def act(self, obs: Observation) -> Action:
        # Matheson et al. support odor-sensitive but non-directional FB tangential
        # input and directional PFN airflow input. Preserve physical bilateral
        # antenna sensing in the recording, but do not convert that asymmetry into
        # an invented directional MB/FB drive in this connectome controller.
        odor_presence = float(obs.mean_odor)
        raw_drive = {
            "odor_left": odor_presence,
            "odor_right": odor_presence,
            "wind_forward": float(max(obs.wind_x_body, 0.0)),
            "wind_backward": float(max(-obs.wind_x_body, 0.0)),
            "wind_left": float(max(obs.wind_y_body, 0.0)),
            "wind_right": float(max(-obs.wind_y_body, 0.0)),
        }
        role_gain = {
            "odor_left": self.odor_gain,
            "odor_right": self.odor_gain,
            "wind_forward": self.wind_forward_gain,
            "wind_backward": self.wind_backward_gain,
            "wind_left": self.wind_cross_gain,
            "wind_right": self.wind_cross_gain,
        }
        role_drive = {
            role: float(value * role_gain[role]) for role, value in raw_drive.items()
        }
        drive = np.zeros_like(self.activity)
        for role, value in role_drive.items():
            self._inject(role, value, drive)

        self._input_snapshot = {
            "signal_kind": "task_optimized_modeled_role_drive",
            "odor_interface": "mean_bilateral_nondirectional",
            "direction_interface": "body_frame_wind",
            "antenna_context": {
                "left_odor": float(obs.left_odor),
                "right_odor": float(obs.right_odor),
                "mean_odor": odor_presence,
            },
            "interface_status": (self.bundle.manifest or {}).get(
                "sensory_interface_status",
                "modeled_interface_not_peripheral_sensory_qualification",
            ),
            "graph_sha256": self.graph_sha256,
            "parameter_sha256": self.parameter_sha256,
            "roles": [
                {
                    "role": role,
                    "raw_value": raw_drive[role],
                    "gain": role_gain[role],
                    "value": role_drive[role],
                    "body_ids": [int(x) for x in self.bundle.roles.get(role, [])],
                }
                for role in raw_drive
            ],
            "warning": (
                "Task-optimized role drives are modeled interface values, not receptor currents, "
                "ORN spikes, or fitted physiological measurements."
            ),
        }

        recurrent = self.recurrent_gain * (self.w @ self.activity)
        proposal = np.tanh(self.gain * (recurrent + drive))
        self.activity = (
            self.retention * self.activity + (1.0 - self.retention) * proposal
        )
        left = self._role_mean("steer_left")
        right = self._role_mean("steer_right")
        turn = float(np.tanh(self.turn_gain * (left - right)))
        self._diag = {
            "dn_left": left,
            "dn_right": right,
            "activity_mean": float(np.abs(self.activity).mean()),
            "modeled_dynamics": 1.0,
            "task_optimized": 1.0,
            "rate_tau_s": self.tau_s,
            "rate_dt_s": self.model_dt_s,
            "rate_retention": self.retention,
            "activation_gain": self.gain,
            "recurrent_gain": self.recurrent_gain,
            "odor_gain": self.odor_gain,
            "wind_forward_gain": self.wind_forward_gain,
            "wind_backward_gain": self.wind_backward_gain,
            "wind_cross_gain": self.wind_cross_gain,
            "turn_gain": self.turn_gain,
            "signed_edge_fraction": self.signed_fraction,
        }
        return Action(turn=turn, speed=1.0)


@dataclass(frozen=True)
class TrainingEpisode:
    seed: int
    objective: float
    success: bool
    spl: float
    terminal_progress: float
    path_length: float
    final_distance: float


def load_training_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    if config.get("protocol") != "task-optimized-connectome-dynamics-v1":
        raise ValueError("unsupported task-optimization protocol")
    interface = config.get("connectome_sensory_interface", {})
    if interface.get("odor_mode") != "mean_bilateral_nondirectional":
        raise ValueError("v1 requires nondirectional mean-odor connectome drive")
    if interface.get("direction_source") != "body_frame_wind":
        raise ValueError("v1 requires body-frame wind as directional input")
    specs = config.get("trainable_parameters", {})
    if set(specs) != set(PARAMETER_NAMES):
        raise ValueError(
            f"trainable parameters must be exactly {list(PARAMETER_NAMES)}; "
            f"found {sorted(specs)}"
        )
    for name in PARAMETER_NAMES:
        spec = specs[name]
        low = float(spec["min"])
        default = float(spec["default"])
        high = float(spec["max"])
        if not (0.0 < low <= default <= high):
            raise ValueError(f"invalid positive bounds/default for {name}")
    objective = config.get("objective", {})
    weights = [
        float(objective["success_weight"]),
        float(objective["spl_weight"]),
        float(objective["terminal_progress_weight"]),
    ]
    if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("task objective weights must sum to one")
    namespace = config.get("seed_namespace", {})
    if int(namespace["base"]) <= FINAL_TEST_MAX_SEED:
        raise ValueError("training seed namespace overlaps the final-test seed population")
    return config


def default_parameters(config: dict[str, Any]) -> DynamicsParameters:
    return DynamicsParameters.from_mapping(
        {
            name: float(config["trainable_parameters"][name]["default"])
            for name in PARAMETER_NAMES
        }
    )


def validate_parameters(
    parameters: DynamicsParameters,
    config: dict[str, Any],
) -> None:
    for name, value in parameters.to_dict().items():
        spec = config["trainable_parameters"][name]
        low = float(spec["min"])
        high = float(spec["max"])
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name}={value} is outside frozen bounds [{low}, {high}]")


def make_training_seed_split(config: dict[str, Any]) -> tuple[list[int], list[int]]:
    namespace = config["seed_namespace"]
    base = int(namespace["base"])
    span = int(namespace["span"])
    n_train = int(namespace["train_episodes"])
    n_validation = int(namespace["validation_episodes"])
    if min(span, n_train, n_validation) <= 0 or n_train + n_validation > span:
        raise ValueError("invalid training seed namespace or split sizes")
    if base <= FINAL_TEST_MAX_SEED:
        raise ValueError("training seeds must be outside the final-test namespace")
    rng = np.random.default_rng(int(namespace["split_seed"]))
    offsets = rng.choice(span, size=n_train + n_validation, replace=False)
    seeds = [int(base + offset) for offset in offsets]
    train = seeds[:n_train]
    validation = seeds[n_train:]
    if set(train) & set(validation):
        raise RuntimeError("training and validation seeds overlap")
    if min(seeds) <= FINAL_TEST_MAX_SEED:
        raise RuntimeError("development seed leaked into final-test namespace")
    return train, validation


def _episode_objective(
    *,
    success: bool,
    episode_spl: float,
    terminal_progress: float,
    config: dict[str, Any],
) -> float:
    objective = config["objective"]
    return float(
        float(objective["success_weight"]) * float(success)
        + float(objective["spl_weight"]) * episode_spl
        + float(objective["terminal_progress_weight"]) * terminal_progress
    )


def run_training_episode(
    controller: TaskOptimizedMaleCNSController,
    seed: int,
    config: dict[str, Any],
    *,
    arena: ArenaConfig | None = None,
    plume: PlumeConfig | None = None,
    sensors: SensorConfig | None = None,
) -> TrainingEpisode:
    env = FlySniffEnv(seed=seed, arena=arena, plume=plume, sensors=sensors)
    controller.reset(seed + 101)
    initial_distance = env.distance_to_source
    shortest = shortest_path_to_goal_region(initial_distance, env.arena.source_radius)
    obs = env.observe()
    done = False
    while not done:
        action = controller.act(obs)
        obs, done = env.step(action.turn, action.speed)
    final_distance = env.distance_to_source
    progress = float(
        np.clip(
            (initial_distance - final_distance) / max(shortest, 1e-12),
            -1.0,
            1.0,
        )
    )
    episode_spl = float(spl(env.agent.found, shortest, env.agent.path_length))
    objective = _episode_objective(
        success=env.agent.found,
        episode_spl=episode_spl,
        terminal_progress=progress,
        config=config,
    )
    return TrainingEpisode(
        seed=int(seed),
        objective=objective,
        success=bool(env.agent.found),
        spl=episode_spl,
        terminal_progress=progress,
        path_length=float(env.agent.path_length),
        final_distance=float(final_distance),
    )


def evaluate_parameters(
    bundle: GraphBundle,
    parameters: DynamicsParameters,
    seeds: list[int],
    config: dict[str, Any],
    *,
    require_qualified: bool = True,
    arena: ArenaConfig | None = None,
    plume: PlumeConfig | None = None,
    sensors: SensorConfig | None = None,
) -> dict[str, float | int]:
    if not seeds:
        raise ValueError("parameter evaluation requires at least one seed")
    validate_parameters(parameters, config)
    model_dt = float((arena or ArenaConfig()).dt)
    controller = TaskOptimizedMaleCNSController(
        bundle,
        parameters,
        model_dt_s=model_dt,
        require_qualified=require_qualified,
    )
    episodes = [
        run_training_episode(
            controller,
            seed,
            config,
            arena=arena,
            plume=plume,
            sensors=sensors,
        )
        for seed in seeds
    ]
    return {
        "n": len(episodes),
        "objective": float(np.mean([x.objective for x in episodes])),
        "success_rate": float(np.mean([x.success for x in episodes])),
        "mean_spl": float(np.mean([x.spl for x in episodes])),
        "mean_terminal_progress": float(
            np.mean([x.terminal_progress for x in episodes])
        ),
        "mean_path_length": float(np.mean([x.path_length for x in episodes])),
        "mean_final_distance": float(np.mean([x.final_distance for x in episodes])),
    }


def _parameter_bounds(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    specs = config["trainable_parameters"]
    low = np.log([float(specs[name]["min"]) for name in PARAMETER_NAMES])
    default = np.log([float(specs[name]["default"]) for name in PARAMETER_NAMES])
    high = np.log([float(specs[name]["max"]) for name in PARAMETER_NAMES])
    return low, default, high


def _decode_parameters(values: np.ndarray, config: dict[str, Any]) -> DynamicsParameters:
    low, _, high = _parameter_bounds(config)
    clipped = np.clip(np.asarray(values, dtype=float), low, high)
    decoded = np.exp(clipped)
    return DynamicsParameters.from_mapping(
        {name: float(value) for name, value in zip(PARAMETER_NAMES, decoded, strict=True)}
    )


def optimize_dynamics(
    bundle: GraphBundle,
    config: dict[str, Any],
    *,
    require_qualified: bool = True,
) -> dict[str, Any]:
    """Fit eight global dynamics parameters with deterministic log-space CEM."""
    train_seeds, validation_seeds = make_training_seed_split(config)
    baseline_parameters = default_parameters(config)
    baseline_train = evaluate_parameters(
        bundle,
        baseline_parameters,
        train_seeds,
        config,
        require_qualified=require_qualified,
    )
    baseline_validation = evaluate_parameters(
        bundle,
        baseline_parameters,
        validation_seeds,
        config,
        require_qualified=require_qualified,
    )

    optimizer = config["optimizer"]
    population_size = int(optimizer["population"])
    generations = int(optimizer["generations"])
    elite_fraction = float(optimizer["elite_fraction"])
    episodes_per_candidate = int(optimizer["episodes_per_candidate"])
    update_rate = float(optimizer["update_rate"])
    initial_sigma_fraction = float(optimizer["initial_sigma_fraction_of_log_range"])
    minimum_sigma = float(optimizer["minimum_log_sigma"])
    if population_size < 4 or generations < 1:
        raise ValueError("CEM requires population >= 4 and at least one generation")
    if not 0.0 < elite_fraction <= 0.5:
        raise ValueError("elite_fraction must be in (0, 0.5]")
    if not 0.0 < update_rate <= 1.0:
        raise ValueError("update_rate must be in (0, 1]")
    if not 1 <= episodes_per_candidate <= len(train_seeds):
        raise ValueError("episodes_per_candidate must fit the frozen training seed pool")

    low, mean, high = _parameter_bounds(config)
    sigma = np.maximum((high - low) * initial_sigma_fraction, minimum_sigma)
    rng = np.random.default_rng(int(optimizer["optimizer_seed"]))
    elite_count = max(2, math.ceil(population_size * elite_fraction))
    history: list[dict[str, Any]] = []

    for generation in range(generations):
        generation_seeds = [
            int(x)
            for x in rng.choice(
                np.asarray(train_seeds, dtype=np.int64),
                size=episodes_per_candidate,
                replace=False,
            )
        ]
        population = rng.normal(
            loc=mean,
            scale=sigma,
            size=(population_size, len(PARAMETER_NAMES)),
        )
        population = np.clip(population, low, high)
        population[0] = mean
        scores = np.empty(population_size, dtype=float)
        summaries: list[dict[str, float | int]] = []
        for index, candidate in enumerate(population):
            parameters = _decode_parameters(candidate, config)
            summary = evaluate_parameters(
                bundle,
                parameters,
                generation_seeds,
                config,
                require_qualified=require_qualified,
            )
            summaries.append(summary)
            scores[index] = float(summary["objective"])

        order = np.argsort(scores)[::-1]
        elite = population[order[:elite_count]]
        elite_mean = np.mean(elite, axis=0)
        elite_sigma = np.std(elite, axis=0)
        mean = (1.0 - update_rate) * mean + update_rate * elite_mean
        sigma = np.maximum(
            (1.0 - update_rate) * sigma + update_rate * elite_sigma,
            minimum_sigma,
        )
        best_index = int(order[0])
        history.append(
            {
                "generation": generation,
                "seed_batch": generation_seeds,
                "population_mean_objective": float(np.mean(scores)),
                "elite_mean_objective": float(np.mean(scores[order[:elite_count]])),
                "best_objective": float(scores[best_index]),
                "best_summary": summaries[best_index],
                "best_parameters": _decode_parameters(
                    population[best_index], config
                ).to_dict(),
                "distribution_mean_parameters": _decode_parameters(mean, config).to_dict(),
                "distribution_log_sigma": {
                    name: float(value)
                    for name, value in zip(PARAMETER_NAMES, sigma, strict=True)
                },
            }
        )

    trained_parameters = _decode_parameters(mean, config)
    trained_train = evaluate_parameters(
        bundle,
        trained_parameters,
        train_seeds,
        config,
        require_qualified=require_qualified,
    )
    trained_validation = evaluate_parameters(
        bundle,
        trained_parameters,
        validation_seeds,
        config,
        require_qualified=require_qualified,
    )
    gate = config["development_gate"]
    objective_delta = float(
        trained_validation["objective"] - baseline_validation["objective"]
    )
    success_delta = float(
        trained_validation["success_rate"] - baseline_validation["success_rate"]
    )
    minimum_delta = float(
        gate["minimum_validation_objective_delta_vs_own_untrained_default"]
    )
    maximum_success_drop = float(
        gate["maximum_validation_success_rate_drop_vs_own_untrained_default"]
    )
    development_passed = bool(
        objective_delta >= minimum_delta and success_delta >= -maximum_success_drop
    )

    return {
        "protocol": config["protocol"],
        "graph_sha256": bundle.replay_fingerprint(),
        "dataset": (bundle.manifest or {}).get("dataset", "unspecified"),
        "graph_role": (bundle.manifest or {}).get("graph_role", "unspecified"),
        "qualification_status": (bundle.manifest or {}).get(
            "qualification_status", "candidate"
        ),
        "sensory_interface": config["connectome_sensory_interface"],
        "training_config_sha256": canonical_sha256(config),
        "train_seed_sha256": canonical_sha256(train_seeds),
        "validation_seed_sha256": canonical_sha256(validation_seeds),
        "train_seed_count": len(train_seeds),
        "validation_seed_count": len(validation_seeds),
        "final_test_namespace_touched": False,
        "baseline_parameters": baseline_parameters.to_dict(),
        "trained_parameters": trained_parameters.to_dict(),
        "trained_parameter_sha256": canonical_sha256(trained_parameters.to_dict()),
        "baseline_train": baseline_train,
        "baseline_validation": baseline_validation,
        "trained_train": trained_train,
        "trained_validation": trained_validation,
        "validation_objective_delta": objective_delta,
        "validation_success_rate_delta": success_delta,
        "development_gate_passed": development_passed,
        "history": history,
        "claim_boundary": config["claim_boundary"],
        "training_signal_warning": config["objective"]["privileged_training_signal"],
    }


def _require_complete_rewire(bundle: GraphBundle) -> None:
    rewire = (bundle.manifest or {}).get("rewire", {})
    if not rewire.get("mixing_complete"):
        raise RuntimeError(
            "rewire failed to reach the frozen directed double-edge-swap target; "
            "refusing to train an under-mixed null topology"
        )


def train_matched_control_cohort(
    bundle: GraphBundle,
    config: dict[str, Any],
    *,
    require_qualified: bool = True,
    rewire_count: int = 8,
    swaps_per_edge: int = 8,
) -> dict[str, Any]:
    """Train intact, rewired, and lesioned topologies with identical budgets."""
    if rewire_count < 2:
        raise ValueError("matched control cohort requires at least two rewires")
    results: dict[str, Any] = {}
    results["intact"] = optimize_dynamics(
        bundle,
        config,
        require_qualified=require_qualified,
    )

    rewire_results: list[dict[str, Any]] = []
    for seed in make_rewire_seeds(n=rewire_count):
        rewired = degree_preserving_rewire(
            bundle,
            seed=seed,
            swaps_per_edge=swaps_per_edge,
        )
        _require_complete_rewire(rewired)
        report = optimize_dynamics(
            rewired,
            config,
            require_qualified=require_qualified,
        )
        report["rewire_seed"] = int(seed)
        report["rewire_manifest"] = (rewired.manifest or {}).get("rewire")
        rewire_results.append(report)
    results["rewires"] = {str(item["rewire_seed"]): item for item in rewire_results}

    lesioned = lesion_incoming_to_roles(bundle, ["steer_left", "steer_right"])
    results["lesion"] = optimize_dynamics(
        lesioned,
        config,
        require_qualified=require_qualified,
    )

    intact_validation = results["intact"]["trained_validation"]
    rewire_validation_objective = np.asarray(
        [item["trained_validation"]["objective"] for item in rewire_results],
        dtype=float,
    )
    rewire_validation_success = np.asarray(
        [item["trained_validation"]["success_rate"] for item in rewire_results],
        dtype=float,
    )
    rewire_validation_spl = np.asarray(
        [item["trained_validation"]["mean_spl"] for item in rewire_results],
        dtype=float,
    )
    development_comparison = {
        "intact_validation_objective": float(intact_validation["objective"]),
        "mean_trained_rewire_validation_objective": float(
            rewire_validation_objective.mean()
        ),
        "intact_minus_mean_rewire_validation_objective": float(
            intact_validation["objective"] - rewire_validation_objective.mean()
        ),
        "intact_validation_success_rate": float(intact_validation["success_rate"]),
        "mean_trained_rewire_validation_success_rate": float(
            rewire_validation_success.mean()
        ),
        "intact_validation_mean_spl": float(intact_validation["mean_spl"]),
        "mean_trained_rewire_validation_spl": float(rewire_validation_spl.mean()),
        "lesion_validation": results["lesion"]["trained_validation"],
        "interpretation": (
            "Development evidence only. Every topology was separately optimized with the same "
            "budget. Final held-out/OOD seeds remain unopened."
        ),
    }
    return {
        "protocol": "matched-task-optimization-controls-v1",
        "training_config_sha256": canonical_sha256(config),
        "intact_graph_sha256": bundle.replay_fingerprint(),
        "rewire_count": rewire_count,
        "swaps_per_edge": swaps_per_edge,
        "results": results,
        "development_comparison": development_comparison,
        "claim_boundary": (
            "This is development/model-selection evidence only. It cannot support the public "
            "connectome navigation claim until E002 is rerun on frozen trained parameters and "
            "the one-way final held-out/OOD benchmark is executed."
        ),
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task-optimize eight global dynamics parameters on a fixed graph"
    )
    parser.add_argument("bundle", help="signed GraphBundle directory")
    parser.add_argument(
        "--config",
        default="configs/task_optimization_v1.json",
        help="frozen task-optimization JSON contract",
    )
    parser.add_argument("--output", default="results/training/intact-v1.json")
    parser.add_argument(
        "--exploratory-candidate",
        action="store_true",
        help="allow an unqualified graph for plumbing only; result remains candidate evidence",
    )
    parser.add_argument(
        "--matched-controls",
        action="store_true",
        help="also train the sealed rewire ensemble and steering-input lesion",
    )
    parser.add_argument("--rewire-count", type=int, default=8)
    args = parser.parse_args()

    config = load_training_config(args.config)
    bundle = GraphBundle.load(args.bundle)
    require_qualified = not args.exploratory_candidate
    if args.matched_controls:
        report = train_matched_control_cohort(
            bundle,
            config,
            require_qualified=require_qualified,
            rewire_count=args.rewire_count,
        )
    else:
        report = optimize_dynamics(
            bundle,
            config,
            require_qualified=require_qualified,
        )
    path = write_report(args.output, report)
    print(path)
    print(canonical_sha256(report))


if __name__ == "__main__":
    main()
