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
    "wind_basis_gain",
    "turn_gain",
)
REQUIRED_MODEL_ROLES = (
    "odor_context_left",
    "odor_context_right",
    "wind_basis_left",
    "wind_basis_right",
    "steer_left",
    "steer_right",
)
SENSORY_DRIVE_ROLES = (
    "odor_context_left",
    "odor_context_right",
    "wind_basis_left",
    "wind_basis_right",
)
STEERING_ROLES = ("steer_left", "steer_right")
FINAL_TEST_MAX_SEED = 1_999_999_999
FROZEN_V1_REWIRE_COUNT = 8
FROZEN_V1_SWAPS_PER_EDGE = 8
INV_SQRT_2 = float(1.0 / np.sqrt(2.0))


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_model_role_partition(bundle: GraphBundle) -> None:
    """Require each modeled role to name a distinct neuron population.

    This is stricter than merely requiring left/right steering disjointness. A body
    ID that is both a directly driven sensory role and a steering role would let
    modeled drive bypass the prespecified incoming-edge lesion, invalidating the
    lesion as a dependency control.
    """
    missing = [role for role in REQUIRED_MODEL_ROLES if not bundle.roles.get(role)]
    if missing:
        raise ValueError(f"task-optimized controller requires non-empty roles: {missing}")

    owner: dict[int, str] = {}
    overlap: list[tuple[int, str, str]] = []
    for role in REQUIRED_MODEL_ROLES:
        for raw_id in bundle.roles.get(role, []):
            body_id = int(raw_id)
            prior = owner.get(body_id)
            if prior is not None and prior != role:
                overlap.append((body_id, prior, role))
            else:
                owner[body_id] = role
    if overlap:
        preview = ", ".join(
            f"{body_id}:{first}/{second}" for body_id, first, second in overlap[:5]
        )
        raise ValueError(
            "modeled role populations must be pairwise disjoint; overlapping sensory/steering "
            f"roles can bypass the lesion ({preview})"
        )


def _assert_graph_identity(bundle: GraphBundle, expected_sha256: str) -> None:
    observed = bundle.replay_fingerprint()
    if observed != expected_sha256:
        raise RuntimeError(
            "graph topology, structural weights/signs, body IDs, or role membership mutated "
            "during task optimization"
        )


@dataclass(frozen=True)
class DynamicsParameters:
    tau_s: float
    activation_gain: float
    recurrent_gain: float
    odor_gain: float
    wind_basis_gain: float
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
    """Fixed-topology model with six task-optimized global parameters.

    Odor is a nondirectional contextual drive. Airflow direction is represented by
    two signed, orthogonal PFN-basis drives whose preferred arrival directions are
    approximately 45 degrees left and right of the fly midline. Connectivity,
    body IDs, role membership, structural weights, and edge signs never train.
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
        _validate_model_role_partition(bundle)
        self.parameters = parameters
        self.recurrent_gain = float(parameters.recurrent_gain)
        self.odor_gain = float(parameters.odor_gain)
        self.wind_basis_gain = float(parameters.wind_basis_gain)
        for name in ("recurrent_gain", "odor_gain", "wind_basis_gain"):
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

    @staticmethod
    def pfn_basis_raw_drive(obs: Observation) -> tuple[float, float]:
        """Project airflow-arrival direction onto +/-45 degree PFN bases."""
        arrival_x = -float(obs.wind_x_body)
        arrival_y = -float(obs.wind_y_body)
        left = (arrival_x + arrival_y) * INV_SQRT_2
        right = (arrival_x - arrival_y) * INV_SQRT_2
        return float(left), float(right)

    def act(self, obs: Observation) -> Action:
        odor_presence = float(obs.mean_odor)
        raw_left_basis, raw_right_basis = self.pfn_basis_raw_drive(obs)
        raw_drive = {
            "odor_context_left": odor_presence,
            "odor_context_right": odor_presence,
            "wind_basis_left": raw_left_basis,
            "wind_basis_right": raw_right_basis,
        }
        role_gain = {
            "odor_context_left": self.odor_gain,
            "odor_context_right": self.odor_gain,
            "wind_basis_left": self.wind_basis_gain,
            "wind_basis_right": self.wind_basis_gain,
        }
        role_drive = {
            role: float(raw_drive[role] * role_gain[role]) for role in raw_drive
        }
        drive = np.zeros_like(self.activity)
        for role, value in role_drive.items():
            self._inject(role, value, drive)

        self._input_snapshot = {
            "signal_kind": "task_optimized_modeled_role_drive",
            "odor_interface": "mean_bilateral_nondirectional",
            "direction_interface": "signed_pfn_basis_from_body_frame_airflow_arrival",
            "airflow_arrival": {
                "x": float(-obs.wind_x_body),
                "y": float(-obs.wind_y_body),
            },
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
                "These task-optimized drives are explicit modeled interface values, not receptor "
                "currents, antennal mechanoreceptor spikes, PFN recordings, or fitted physiology."
            ),
        }

        recurrent = self.recurrent_gain * (self.w @ self.activity)
        proposal = np.tanh(self.gain * (recurrent + drive))
        self.activity = self.retention * self.activity + (1.0 - self.retention) * proposal
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
            "wind_basis_gain": self.wind_basis_gain,
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
    if interface.get("direction_source") != "signed_pfn_basis_from_body_frame_airflow_arrival":
        raise ValueError("v1 requires the signed +/-45 degree PFN airflow-basis interface")
    if interface.get("odor_roles") != ["odor_context_left", "odor_context_right"]:
        raise ValueError("v1 odor-role contract changed")
    if interface.get("wind_roles") != ["wind_basis_left", "wind_basis_right"]:
        raise ValueError("v1 PFN-basis role contract changed")

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
        if not all(np.isfinite(x) for x in (low, default, high)):
            raise ValueError(f"nonfinite bounds/default for {name}")
        if not (0.0 < low <= default <= high):
            raise ValueError(f"invalid positive bounds/default for {name}")

    objective = config.get("objective", {})
    weights = [
        float(objective["success_weight"]),
        float(objective["spl_weight"]),
        float(objective["terminal_progress_weight"]),
    ]
    if not all(np.isfinite(x) and x >= 0.0 for x in weights):
        raise ValueError("task objective weights must be finite and nonnegative")
    if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("task objective weights must sum to one")

    namespace = config.get("seed_namespace", {})
    base = int(namespace["base"])
    span = int(namespace["span"])
    if base <= FINAL_TEST_MAX_SEED:
        raise ValueError("training seed namespace overlaps the final-test seed population")
    if span <= 0 or base + span - 1 > np.iinfo(np.int64).max:
        raise ValueError("invalid development seed namespace span")

    optimizer = config.get("optimizer", {})
    if optimizer.get("kind") != "log-space cross-entropy method":
        raise ValueError("v1 optimizer kind changed")
    if optimizer.get("common_random_numbers") is not True:
        raise ValueError("v1 requires common random numbers within every CEM generation")

    return config


def default_parameters(config: dict[str, Any]) -> DynamicsParameters:
    return DynamicsParameters.from_mapping(
        {
            name: float(config["trainable_parameters"][name]["default"])
            for name in PARAMETER_NAMES
        }
    )


def validate_parameters(parameters: DynamicsParameters, config: dict[str, Any]) -> None:
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
    spl_value = float(episode_spl)
    progress_value = float(terminal_progress)
    if not np.isfinite(spl_value) or not 0.0 <= spl_value <= 1.0:
        raise ValueError(f"SPL objective component must be finite and in [0, 1]; got {spl_value}")
    if not np.isfinite(progress_value) or not -1.0 <= progress_value <= 1.0:
        raise ValueError(
            "terminal progress objective component must be finite and in [-1, 1]; "
            f"got {progress_value}"
        )
    objective = config["objective"]
    value = float(
        float(objective["success_weight"]) * float(bool(success))
        + float(objective["spl_weight"]) * spl_value
        + float(objective["terminal_progress_weight"]) * progress_value
    )
    if not np.isfinite(value):
        raise ValueError("objective became nonfinite")
    return value


def run_training_episode(
    controller: TaskOptimizedMaleCNSController,
    seed: int,
    config: dict[str, Any],
    *,
    arena: ArenaConfig | None = None,
    plume: PlumeConfig | None = None,
    sensors: SensorConfig | None = None,
) -> TrainingEpisode:
    if int(seed) <= FINAL_TEST_MAX_SEED:
        raise ValueError("task-optimization episode attempted to use the final-test seed namespace")
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
    values = (env.agent.path_length, final_distance, shortest)
    if not all(np.isfinite(float(x)) and float(x) >= 0.0 for x in values):
        raise RuntimeError("simulator produced a nonfinite or negative path/distance metric")
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
    if any(int(seed) <= FINAL_TEST_MAX_SEED for seed in seeds):
        raise ValueError("task optimization cannot evaluate final-test namespace seeds")
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
        "mean_terminal_progress": float(np.mean([x.terminal_progress for x in episodes])),
        "mean_path_length": float(np.mean([x.path_length for x in episodes])),
        "mean_final_distance": float(np.mean([x.final_distance for x in episodes])),
    }


def _validate_evaluation_summary(summary: dict[str, Any], expected_n: int) -> None:
    if int(summary.get("n", -1)) != int(expected_n):
        raise RuntimeError("evaluation summary episode count does not match requested seed batch")
    finite_fields = (
        "objective",
        "success_rate",
        "mean_spl",
        "mean_terminal_progress",
        "mean_path_length",
        "mean_final_distance",
    )
    for field in finite_fields:
        value = float(summary[field])
        if not np.isfinite(value):
            raise RuntimeError(f"evaluation summary {field} is nonfinite")
    if not 0.0 <= float(summary["success_rate"]) <= 1.0:
        raise RuntimeError("evaluation success_rate is outside [0, 1]")
    if not 0.0 <= float(summary["mean_spl"]) <= 1.0:
        raise RuntimeError("evaluation mean_spl is outside [0, 1]")
    if not -1.0 <= float(summary["mean_terminal_progress"]) <= 1.0:
        raise RuntimeError("evaluation mean_terminal_progress is outside [-1, 1]")
    if float(summary["mean_path_length"]) < 0.0 or float(summary["mean_final_distance"]) < 0.0:
        raise RuntimeError("evaluation path/distance summaries must be nonnegative")


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


def optimizer_budget_receipt(
    config: dict[str, Any],
    *,
    train_seed_count: int,
    validation_seed_count: int,
) -> dict[str, int | str]:
    optimizer = config["optimizer"]
    population = int(optimizer["population"])
    generations = int(optimizer["generations"])
    episodes_per_candidate = int(optimizer["episodes_per_candidate"])
    candidate_evaluations = population * generations
    candidate_episodes = candidate_evaluations * episodes_per_candidate
    full_pool_evaluations = 4
    full_pool_episodes = 2 * (int(train_seed_count) + int(validation_seed_count))
    return {
        "protocol": "task-optimization-budget-v1",
        "population": population,
        "generations": generations,
        "episodes_per_candidate": episodes_per_candidate,
        "candidate_evaluations": candidate_evaluations,
        "candidate_episodes": candidate_episodes,
        "full_pool_evaluations": full_pool_evaluations,
        "full_pool_episodes": full_pool_episodes,
        "total_parameter_evaluations": candidate_evaluations + full_pool_evaluations,
        "total_episode_evaluations": candidate_episodes + full_pool_episodes,
    }


def _guarded_evaluate(
    bundle: GraphBundle,
    expected_graph_sha256: str,
    parameters: DynamicsParameters,
    seeds: list[int],
    config: dict[str, Any],
    *,
    require_qualified: bool,
) -> dict[str, float | int]:
    _assert_graph_identity(bundle, expected_graph_sha256)
    if any(int(seed) <= FINAL_TEST_MAX_SEED for seed in seeds):
        raise RuntimeError("optimizer attempted to evaluate a final-test namespace seed")
    summary = evaluate_parameters(
        bundle,
        parameters,
        seeds,
        config,
        require_qualified=require_qualified,
    )
    _assert_graph_identity(bundle, expected_graph_sha256)
    _validate_evaluation_summary(summary, len(seeds))
    return summary


def _development_gate_from_summaries(
    baseline_validation: dict[str, Any],
    trained_validation: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, float, bool]:
    objective_delta = float(
        float(trained_validation["objective"]) - float(baseline_validation["objective"])
    )
    success_delta = float(
        float(trained_validation["success_rate"])
        - float(baseline_validation["success_rate"])
    )
    gate = config["development_gate"]
    passed = bool(
        objective_delta
        >= float(gate["minimum_validation_objective_delta_vs_own_untrained_default"])
        and success_delta
        >= -float(gate["maximum_validation_success_rate_drop_vs_own_untrained_default"])
    )
    return objective_delta, success_delta, passed


def optimize_dynamics(
    bundle: GraphBundle,
    config: dict[str, Any],
    *,
    require_qualified: bool = True,
) -> dict[str, Any]:
    """Fit six global dynamics parameters with deterministic log-space CEM."""
    bundle.validate(require_sign=True, require_qualified=require_qualified)
    _validate_model_role_partition(bundle)
    graph_sha256 = bundle.replay_fingerprint()

    train_seeds, validation_seeds = make_training_seed_split(config)
    baseline_parameters = default_parameters(config)
    baseline_train = _guarded_evaluate(
        bundle,
        graph_sha256,
        baseline_parameters,
        train_seeds,
        config,
        require_qualified=require_qualified,
    )
    baseline_validation = _guarded_evaluate(
        bundle,
        graph_sha256,
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
    if initial_sigma_fraction <= 0.0 or minimum_sigma <= 0.0:
        raise ValueError("CEM sigma settings must be positive")

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
        candidate_receipts: list[dict[str, Any]] = []
        for index, candidate in enumerate(population):
            parameters = _decode_parameters(candidate, config)
            summary = _guarded_evaluate(
                bundle,
                graph_sha256,
                parameters,
                generation_seeds,
                config,
                require_qualified=require_qualified,
            )
            summaries.append(summary)
            scores[index] = float(summary["objective"])
            candidate_receipts.append(
                {
                    "index": int(index),
                    "parameter_sha256": canonical_sha256(parameters.to_dict()),
                    "objective": float(summary["objective"]),
                }
            )

        order = np.lexsort((np.arange(population_size, dtype=int), -scores))
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
                "seed_batch_sha256": canonical_sha256(generation_seeds),
                "candidate_tie_break": "descending_objective_then_ascending_candidate_index",
                "candidate_receipts": candidate_receipts,
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
    trained_train = _guarded_evaluate(
        bundle,
        graph_sha256,
        trained_parameters,
        train_seeds,
        config,
        require_qualified=require_qualified,
    )
    trained_validation = _guarded_evaluate(
        bundle,
        graph_sha256,
        trained_parameters,
        validation_seeds,
        config,
        require_qualified=require_qualified,
    )
    objective_delta, success_delta, development_passed = _development_gate_from_summaries(
        baseline_validation,
        trained_validation,
        config,
    )
    _assert_graph_identity(bundle, graph_sha256)

    budget = optimizer_budget_receipt(
        config,
        train_seed_count=len(train_seeds),
        validation_seed_count=len(validation_seeds),
    )
    report = {
        "report_schema": "task-optimization-report-v2-redteam-hardened",
        "protocol": config["protocol"],
        "graph_sha256": graph_sha256,
        "dataset": (bundle.manifest or {}).get("dataset", "unspecified"),
        "graph_role": (bundle.manifest or {}).get("graph_role", "unspecified"),
        "qualification_status": (bundle.manifest or {}).get(
            "qualification_status", "candidate"
        ),
        "sensory_interface": config["connectome_sensory_interface"],
        "training_config_sha256": canonical_sha256(config),
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "train_seed_sha256": canonical_sha256(train_seeds),
        "validation_seed_sha256": canonical_sha256(validation_seeds),
        "train_seed_count": len(train_seeds),
        "validation_seed_count": len(validation_seeds),
        "development_seed_namespace_min": min(train_seeds + validation_seeds),
        "final_test_namespace_touched": False,
        "optimizer_budget": budget,
        "optimizer_budget_sha256": canonical_sha256(budget),
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
    report["audit_receipt_sha256"] = canonical_sha256(
        {
            "graph_sha256": graph_sha256,
            "training_config_sha256": report["training_config_sha256"],
            "train_seed_sha256": report["train_seed_sha256"],
            "validation_seed_sha256": report["validation_seed_sha256"],
            "trained_parameter_sha256": report["trained_parameter_sha256"],
            "optimizer_budget_sha256": report["optimizer_budget_sha256"],
        }
    )
    return report


def _require_complete_rewire(bundle: GraphBundle) -> None:
    rewire = (bundle.manifest or {}).get("rewire", {})
    try:
        accepted = int(rewire["accepted_swaps"])
        target = int(rewire["target_swaps"])
        attempted = int(rewire["attempted_swaps"])
        swaps_per_edge = int(rewire["swaps_per_edge"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("rewire manifest is missing auditable swap receipts") from exc

    expected_target = swaps_per_edge * len(bundle.edges)
    if rewire.get("mixing_complete") is not True:
        raise RuntimeError("rewire did not complete its frozen swap target")
    if target != expected_target or accepted != target:
        raise RuntimeError(
            "rewire mixing/swap receipt is inconsistent with the graph and requested swap budget"
        )
    if attempted < accepted:
        raise RuntimeError("rewire attempted_swaps cannot be smaller than accepted_swaps")
    if rewire.get("exact_in_out_degree_preserved") is not True:
        raise RuntimeError("rewire manifest does not attest exact directed in/out degree preservation")


def _degree_counts(bundle: GraphBundle) -> tuple[dict[int, int], dict[int, int]]:
    ids = [int(x) for x in bundle.nodes.bodyId]
    incoming = {body_id: 0 for body_id in ids}
    outgoing = {body_id: 0 for body_id in ids}
    for row in bundle.edges[["source", "target"]].itertuples(index=False):
        outgoing[int(row.source)] += 1
        incoming[int(row.target)] += 1
    return incoming, outgoing


def _assert_rewire_matches_original(original: GraphBundle, rewired: GraphBundle) -> None:
    original.validate(require_sign=True)
    rewired.validate(require_sign=True)
    if sorted(int(x) for x in original.nodes.bodyId) != sorted(
        int(x) for x in rewired.nodes.bodyId
    ):
        raise RuntimeError("rewire changed the node/body-ID set")
    if original.roles != rewired.roles:
        raise RuntimeError("rewire changed role membership")
    if len(original.edges) != len(rewired.edges):
        raise RuntimeError("rewire changed edge count")
    if _degree_counts(original) != _degree_counts(rewired):
        raise RuntimeError("rewire failed exact directed in/out-degree preservation")
    _require_complete_rewire(rewired)


def _require_frozen_v1_matched_control_budget(
    *,
    rewire_count: int,
    swaps_per_edge: int,
) -> None:
    if int(rewire_count) != FROZEN_V1_REWIRE_COUNT:
        raise ValueError(
            f"frozen v1 matched control requires exactly {FROZEN_V1_REWIRE_COUNT} rewires; "
            f"got rewire_count={rewire_count}"
        )
    if int(swaps_per_edge) != FROZEN_V1_SWAPS_PER_EDGE:
        raise ValueError(
            f"frozen v1 matched control requires swaps_per_edge={FROZEN_V1_SWAPS_PER_EDGE}; "
            f"got {swaps_per_edge}"
        )


def _require_identical_optimizer_budgets(reports: list[dict[str, Any]]) -> str:
    hashes: list[str] = []
    for report in reports:
        budget = report.get("optimizer_budget")
        if not isinstance(budget, dict):
            raise RuntimeError("matched-control report is missing optimizer_budget receipt")
        observed_hash = canonical_sha256(budget)
        if report.get("optimizer_budget_sha256") != observed_hash:
            raise RuntimeError("optimizer budget hash mismatch")
        hashes.append(observed_hash)
    if len(set(hashes)) != 1:
        raise RuntimeError("intact, rewire, and lesion runs received unequal compute budgets")
    return hashes[0]


def train_matched_control_cohort(
    bundle: GraphBundle,
    config: dict[str, Any],
    *,
    require_qualified: bool = True,
    rewire_count: int = FROZEN_V1_REWIRE_COUNT,
    swaps_per_edge: int = FROZEN_V1_SWAPS_PER_EDGE,
) -> dict[str, Any]:
    """Train intact, rewired, and lesioned topologies with identical frozen budgets."""
    _require_frozen_v1_matched_control_budget(
        rewire_count=rewire_count,
        swaps_per_edge=swaps_per_edge,
    )
    bundle.validate(require_sign=True, require_qualified=require_qualified)
    _validate_model_role_partition(bundle)
    intact_sha256 = bundle.replay_fingerprint()

    results: dict[str, Any] = {}
    results["intact"] = optimize_dynamics(
        bundle,
        config,
        require_qualified=require_qualified,
    )
    _assert_graph_identity(bundle, intact_sha256)

    rewire_results: list[dict[str, Any]] = []
    rewire_seeds = make_rewire_seeds(n=FROZEN_V1_REWIRE_COUNT)
    if len(set(rewire_seeds)) != FROZEN_V1_REWIRE_COUNT:
        raise RuntimeError("frozen rewire seed generator returned duplicate topology seeds")
    for seed in rewire_seeds:
        rewired = degree_preserving_rewire(
            bundle,
            seed=seed,
            swaps_per_edge=FROZEN_V1_SWAPS_PER_EDGE,
        )
        _assert_rewire_matches_original(bundle, rewired)
        report = optimize_dynamics(
            rewired,
            config,
            require_qualified=require_qualified,
        )
        report["rewire_seed"] = int(seed)
        report["rewire_manifest"] = (rewired.manifest or {}).get("rewire")
        rewire_results.append(report)
        _assert_graph_identity(bundle, intact_sha256)
    results["rewires"] = {str(item["rewire_seed"]): item for item in rewire_results}

    lesioned = lesion_incoming_to_roles(bundle, list(STEERING_ROLES))
    _validate_model_role_partition(lesioned)
    results["lesion"] = optimize_dynamics(
        lesioned,
        config,
        require_qualified=require_qualified,
    )
    _assert_graph_identity(bundle, intact_sha256)

    all_reports = [results["intact"], *rewire_results, results["lesion"]]
    budget_sha256 = _require_identical_optimizer_budgets(all_reports)

    intact_validation = results["intact"]["trained_validation"]
    rewire_objectives = np.asarray(
        [item["trained_validation"]["objective"] for item in rewire_results], dtype=float
    )
    rewire_success = np.asarray(
        [item["trained_validation"]["success_rate"] for item in rewire_results], dtype=float
    )
    rewire_spl = np.asarray(
        [item["trained_validation"]["mean_spl"] for item in rewire_results], dtype=float
    )
    comparison = {
        "intact_validation_objective": float(intact_validation["objective"]),
        "mean_trained_rewire_validation_objective": float(rewire_objectives.mean()),
        "intact_minus_mean_rewire_validation_objective": float(
            intact_validation["objective"] - rewire_objectives.mean()
        ),
        "intact_validation_success_rate": float(intact_validation["success_rate"]),
        "mean_trained_rewire_validation_success_rate": float(rewire_success.mean()),
        "intact_validation_mean_spl": float(intact_validation["mean_spl"]),
        "mean_trained_rewire_validation_spl": float(rewire_spl.mean()),
        "lesion_validation": results["lesion"]["trained_validation"],
        "interpretation": (
            "Development evidence only. Every topology was separately optimized with the same "
            "machine-checked budget. Final held-out/OOD seeds are not used by this training run."
        ),
    }
    return {
        "protocol": "matched-task-optimization-controls-v1",
        "training_config_sha256": canonical_sha256(config),
        "intact_graph_sha256": intact_sha256,
        "rewire_count": FROZEN_V1_REWIRE_COUNT,
        "rewire_seeds": rewire_seeds,
        "swaps_per_edge": FROZEN_V1_SWAPS_PER_EDGE,
        "optimizer_budget_sha256": budget_sha256,
        "results": results,
        "development_comparison": comparison,
        "claim_boundary": (
            "Development/model-selection evidence only. It cannot support the public connectome "
            "navigation claim until trained E002 passes and the one-way final benchmark runs."
        ),
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task-optimize six global dynamics parameters on a fixed graph"
    )
    parser.add_argument("bundle", help="reviewed signed GraphBundle directory")
    parser.add_argument(
        "--config",
        default="configs/task_optimization_v1.json",
        help="frozen task-optimization JSON contract",
    )
    parser.add_argument("--output", default="results/training/intact-v1.json")
    parser.add_argument(
        "--exploratory-candidate",
        action="store_true",
        help="allow an unqualified graph for development only; result remains candidate evidence",
    )
    parser.add_argument(
        "--matched-controls",
        action="store_true",
        help="also train the frozen eight-rewire ensemble and steering-input lesion",
    )
    parser.add_argument(
        "--rewire-count",
        type=int,
        default=FROZEN_V1_REWIRE_COUNT,
        help="protocol-identity check; frozen v1 requires exactly 8",
    )
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
