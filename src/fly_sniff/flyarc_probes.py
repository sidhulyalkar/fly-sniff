from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .flyarc import (
    ARCFrameEncoder,
    CoreSelection,
    _fixed_input_projection,
    _graph_artifact_hashes,
    _normalized_weight_matrix,
    _sha256,
    degree_preserving_rewire,
    random_edges_like,
    select_structural_core,
)
from .flyarc_render import validate_receipt
from .graph import GraphBundle

PROBE_SCHEMA = "flyarc-representation-probe-v1"
PROBE_VARIANTS = ("intact", "rewire", "random", "leak_only", "stateless")
DEFAULT_LAGS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class ProbeConfig:
    max_nodes: int = 4096
    pool: int = 8
    colors: int = 16
    input_fanout: int = 4
    leak: float = 0.82
    gain: float = 1.6
    projection_seed: int = 1701
    topology_seed: int = 2903
    target_sketch_seed: int = 8801
    target_sketch_dim: int = 128
    ridge_alpha: float = 0.1
    lags: tuple[int, ...] = DEFAULT_LAGS
    health_neurons: int = 256
    blocked_folds: int = 5
    purge: int = 32


@dataclass(frozen=True)
class SharedTrajectory:
    frames: np.ndarray
    shapes: np.ndarray
    segment_ids: np.ndarray
    reset_indices: tuple[int, ...]
    source_variant: str
    source_frame_sha256: str
    source_receipt_sha256: str
    source_run_dir: str


class ProbeReservoir:
    """Replay-only reservoir with controls that separate recurrence from leak memory."""

    def __init__(
        self,
        core: CoreSelection,
        *,
        variant: str,
        feature_dim: int,
        projection_seed: int,
        topology_seed: int,
        input_fanout: int,
        leak: float,
        gain: float,
    ):
        if variant not in PROBE_VARIANTS:
            raise ValueError(f"unknown probe variant {variant!r}")
        self.variant = variant
        self.body_ids = core.body_ids
        self.leak = float(leak)
        self.gain = float(gain)
        self.activity = np.zeros(len(self.body_ids), dtype=np.float32)
        self.projection = _fixed_input_projection(
            state_dim=len(self.body_ids),
            feature_dim=feature_dim,
            fanout=input_fanout,
            seed=projection_seed,
        )

        edges = core.edges
        self.rewire_swaps = 0
        if variant == "rewire":
            edges, self.rewire_swaps = degree_preserving_rewire(
                edges,
                seed=topology_seed,
            )
        elif variant == "random":
            edges = random_edges_like(
                edges,
                body_ids=self.body_ids,
                seed=topology_seed,
            )

        self.edge_count = len(edges)
        if variant in {"stateless", "leak_only"}:
            self.w: sparse.csr_matrix | None = None
        else:
            self.w = _normalized_weight_matrix(self.body_ids, edges)

    def reset(self) -> None:
        self.activity.fill(0.0)

    def step(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        drive = np.asarray(self.projection @ values).ravel()
        if self.variant == "stateless":
            self.activity = np.tanh(self.gain * drive).astype(np.float32, copy=False)
            return self.activity.copy()

        recurrent = 0.0 if self.w is None else self.w @ self.activity
        proposal = np.tanh(self.gain * (recurrent + drive))
        updated = self.leak * self.activity + (1.0 - self.leak) * proposal
        self.activity = np.asarray(updated, dtype=np.float32)
        return self.activity.copy()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_shared_trajectory(run_dir: str | Path) -> SharedTrajectory:
    """Load a first-light trajectory only when all recorded conditions saw identical frames."""

    root = Path(run_dir)
    receipt = validate_receipt(root)
    files = receipt.get("files", {})
    frame_entries = {
        path: digest
        for path, digest in files.items()
        if path.count("/") == 1 and path.endswith("/frames.npz")
    }
    if len(frame_entries) < 2:
        raise ValueError("probe source must contain at least two hashed condition frame streams")
    frame_hashes = set(frame_entries.values())
    if len(frame_hashes) != 1:
        raise ValueError(
            "probe source is not an identical-stimulus run: condition frame hashes differ"
        )

    variants = sorted(path.split("/", 1)[0] for path in frame_entries)
    source_variant = "intact" if "intact" in variants else variants[0]
    frame_path = root / source_variant / "frames.npz"
    step_path = root / source_variant / "steps.jsonl"
    rows = _read_jsonl(step_path)

    with np.load(frame_path) as archive:
        frames = archive["frames"].copy()
        shapes = archive["shapes"].copy()
    if len(frames) != len(shapes):
        raise ValueError("frames/shapes length mismatch")

    action_observations = {int(row["observation_index"]) for row in rows}
    if 0 in action_observations:
        raise ValueError("observation index 0 must be the initial/reset observation")
    reset_indices = tuple(
        index for index in range(len(frames)) if index == 0 or index not in action_observations
    )
    if not reset_indices or reset_indices[0] != 0:
        raise ValueError("trajectory must begin with a reset/initial observation")

    segment_ids = np.empty(len(frames), dtype=np.int32)
    segment = -1
    reset_set = set(reset_indices)
    for index in range(len(frames)):
        if index in reset_set:
            segment += 1
        segment_ids[index] = segment

    expected_resets = sum(1 for row in rows if row.get("state") == "GAME_OVER")
    observed_extra_resets = len(reset_indices) - 1
    if observed_extra_resets != expected_resets:
        raise ValueError(
            "reset reconstruction mismatch: "
            f"GAME_OVER rows={expected_resets}, reset frames={observed_extra_resets}"
        )

    return SharedTrajectory(
        frames=frames,
        shapes=shapes,
        segment_ids=segment_ids,
        reset_indices=reset_indices,
        source_variant=source_variant,
        source_frame_sha256=next(iter(frame_hashes)),
        source_receipt_sha256=_sha256(root / "receipt.json"),
        source_run_dir=str(root),
    )


def _unpack_frame(trajectory: SharedTrajectory, index: int) -> np.ndarray:
    height, width = (int(x) for x in trajectory.shapes[index])
    return trajectory.frames[index, :height, :width]


def encode_trajectory(
    trajectory: SharedTrajectory,
    *,
    pool: int,
    colors: int,
) -> np.ndarray:
    encoder = ARCFrameEncoder(pool=pool, colors=colors)
    encoded: list[np.ndarray] = []
    reset_set = set(trajectory.reset_indices)
    for index in range(len(trajectory.frames)):
        if index in reset_set:
            encoder.reset()
        encoded.append(encoder.encode(_unpack_frame(trajectory, index)))
    return np.stack(encoded).astype(np.float32, copy=False)


def replay_states(
    core: CoreSelection,
    trajectory: SharedTrajectory,
    features: np.ndarray,
    *,
    variant: str,
    config: ProbeConfig,
) -> tuple[np.ndarray, ProbeReservoir]:
    reservoir = ProbeReservoir(
        core,
        variant=variant,
        feature_dim=features.shape[1],
        projection_seed=config.projection_seed,
        topology_seed=config.topology_seed,
        input_fanout=config.input_fanout,
        leak=config.leak,
        gain=config.gain,
    )
    reset_set = set(trajectory.reset_indices)
    states: list[np.ndarray] = []
    for index, feature in enumerate(features):
        if index in reset_set:
            reservoir.reset()
        states.append(reservoir.step(feature))
    return np.stack(states).astype(np.float32, copy=False), reservoir


def frame_target_sketch(
    features: np.ndarray,
    *,
    seed: int,
    output_dim: int,
) -> np.ndarray:
    if output_dim <= 0:
        raise ValueError("target sketch dimension must be positive")
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(features.shape[1], output_dim),
    )
    signs /= math.sqrt(float(output_dim))
    return (np.asarray(features, dtype=np.float32) @ signs).astype(np.float32)


def _variance_weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    center = y.mean(axis=0, keepdims=True)
    sst = float(np.square(y - center).sum())
    if sst <= 1e-15:
        return float("nan")
    sse = float(np.square(y - pred).sum())
    return 1.0 - sse / sst


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    feature_scale = math.sqrt(float(train.shape[1]))
    train /= feature_scale
    test /= feature_scale

    y_mean = y_train.mean(axis=0, keepdims=True)
    target = y_train - y_mean
    gram = train @ train.T
    gram.flat[:: len(gram) + 1] += alpha
    dual = np.linalg.solve(gram, target)
    return test @ train.T @ dual + y_mean


def _blocked_splits(
    sample_times: np.ndarray,
    sample_segments: np.ndarray,
    *,
    folds: int,
    purge: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_segments = np.unique(sample_segments)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    if len(unique_segments) >= 2:
        for segment in unique_segments:
            test = np.flatnonzero(sample_segments == segment)
            train = np.flatnonzero(sample_segments != segment)
            if len(test) >= 4 and len(train) >= 12:
                splits.append((train, test))
        if len(splits) >= 2:
            return splits

    order = np.argsort(sample_times)
    fold_count = max(2, min(folds, len(order) // 8))
    for test in np.array_split(order, fold_count):
        if len(test) < 4:
            continue
        test_times = sample_times[test]
        test_min, test_max = int(test_times.min()), int(test_times.max())
        keep = np.ones(len(sample_times), dtype=bool)
        keep[test] = False
        same_segment = np.isin(sample_segments, np.unique(sample_segments[test]))
        near = (sample_times >= test_min - purge) & (sample_times <= test_max + purge)
        keep &= ~(same_segment & near)
        train = np.flatnonzero(keep)
        if len(train) >= 12:
            splits.append((train, test))
    if len(splits) < 2:
        raise ValueError("insufficient samples for blocked probe cross-validation")
    return splits


def cross_validated_r2(
    x: np.ndarray,
    y: np.ndarray,
    sample_times: np.ndarray,
    sample_segments: np.ndarray,
    *,
    alpha: float,
    folds: int,
    purge: int,
) -> float:
    predictions = np.zeros_like(y, dtype=np.float64)
    covered = np.zeros(len(y), dtype=bool)
    for train, test in _blocked_splits(
        sample_times,
        sample_segments,
        folds=folds,
        purge=purge,
    ):
        predictions[test] = _ridge_predict(
            x[train],
            y[train],
            x[test],
            alpha=alpha,
        )
        covered[test] = True
    if covered.sum() < 8:
        raise ValueError("probe cross-validation covered fewer than eight samples")
    return _variance_weighted_r2(y[covered], predictions[covered])


def _valid_lag_indices(segment_ids: np.ndarray, lag: int) -> np.ndarray:
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if lag == 0:
        return np.arange(len(segment_ids), dtype=int)
    index = np.arange(lag, len(segment_ids), dtype=int)
    return index[segment_ids[index] == segment_ids[index - lag]]


def _valid_future_indices(segment_ids: np.ndarray) -> np.ndarray:
    index = np.arange(0, len(segment_ids) - 1, dtype=int)
    return index[segment_ids[index] == segment_ids[index + 1]]


def _health_indices(body_ids: tuple[int, ...], count: int) -> np.ndarray:
    keyed = []
    for index, body_id in enumerate(body_ids):
        digest = hashlib.sha256(str(body_id).encode()).digest()
        keyed.append((int.from_bytes(digest[:8], "big"), index))
    keyed.sort()
    return np.asarray([index for _, index in keyed[: min(count, len(keyed))]], dtype=int)


def state_health(
    states: np.ndarray,
    *,
    body_ids: tuple[int, ...],
    segment_ids: np.ndarray,
    health_neurons: int,
) -> dict[str, float | int]:
    values = np.asarray(states, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(values))))
    mean_abs = float(np.mean(np.abs(values)))
    saturation = float(np.mean(np.abs(values) >= 0.95))

    valid_pairs = segment_ids[1:] == segment_ids[:-1]
    if valid_pairs.any():
        previous = values[:-1][valid_pairs]
        current = values[1:][valid_pairs]
        delta_rms = float(np.sqrt(np.mean(np.square(current - previous))))
        numerator = np.sum(previous * current, axis=1)
        denominator = np.linalg.norm(previous, axis=1) * np.linalg.norm(current, axis=1)
        cosine = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        lag1_cosine = float(np.mean(cosine))
    else:
        delta_rms = float("nan")
        lag1_cosine = float("nan")

    subset = values[:, _health_indices(body_ids, health_neurons)]
    subset = subset - subset.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(subset, full_matrices=False, compute_uv=False)
    eigen = np.square(singular)
    total = float(eigen.sum())
    if total <= 1e-15:
        participation = 0.0
        rank95 = 0
    else:
        participation = total * total / float(np.square(eigen).sum())
        cumulative = np.cumsum(eigen) / total
        rank95 = int(np.searchsorted(cumulative, 0.95) + 1)

    return {
        "state_rms": rms,
        "state_mean_abs": mean_abs,
        "saturation_fraction_abs_ge_0_95": saturation,
        "state_delta_rms": delta_rms,
        "lag1_state_cosine": lag1_cosine,
        "effective_dimension_participation": participation,
        "rank_95_variance": rank95,
        "health_neuron_subsample": int(subset.shape[1]),
    }


def compute_input_baselines(
    trajectory: SharedTrajectory,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    config: ProbeConfig,
) -> dict[str, Any]:
    memory: dict[str, float] = {}
    for lag in config.lags:
        index = _valid_lag_indices(trajectory.segment_ids, lag)
        if len(index) < 24:
            memory[str(lag)] = float("nan")
            continue
        memory[str(lag)] = cross_validated_r2(
            features[index],
            targets[index - lag],
            index,
            trajectory.segment_ids[index],
            alpha=config.ridge_alpha,
            folds=config.blocked_folds,
            purge=max(config.purge, lag),
        )

    future_index = _valid_future_indices(trajectory.segment_ids)
    future_r2 = cross_validated_r2(
        features[future_index],
        targets[future_index + 1],
        future_index,
        trajectory.segment_ids[future_index],
        alpha=config.ridge_alpha,
        folds=config.blocked_folds,
        purge=config.purge,
    )
    return {
        "memory": memory,
        "future_r2": future_r2,
    }


def run_probe(
    bundle: GraphBundle,
    trajectory: SharedTrajectory,
    *,
    variant: str,
    config: ProbeConfig,
    core: CoreSelection | None = None,
    features: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    input_baselines: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if variant not in PROBE_VARIANTS:
        raise ValueError(f"unknown probe variant {variant!r}")
    if core is None:
        core = select_structural_core(bundle, max_nodes=config.max_nodes)
    if features is None:
        features = encode_trajectory(
            trajectory,
            pool=config.pool,
            colors=config.colors,
        )
    if targets is None:
        targets = frame_target_sketch(
            features,
            seed=config.target_sketch_seed,
            output_dim=config.target_sketch_dim,
        )
    if input_baselines is None:
        input_baselines = compute_input_baselines(
            trajectory,
            features,
            targets,
            config=config,
        )

    states, reservoir = replay_states(
        core,
        trajectory,
        features,
        variant=variant,
        config=config,
    )

    health = state_health(
        states,
        body_ids=core.body_ids,
        segment_ids=trajectory.segment_ids,
        health_neurons=config.health_neurons,
    )

    memory: dict[str, dict[str, float | int]] = {}
    for lag in config.lags:
        index = _valid_lag_indices(trajectory.segment_ids, lag)
        if len(index) < 24:
            memory[str(lag)] = {
                "samples": len(index),
                "reservoir_r2": float("nan"),
                "current_input_r2": float("nan"),
                "excess_r2": float("nan"),
            }
            continue
        y = targets[index - lag]
        sample_segments = trajectory.segment_ids[index]
        reservoir_r2 = cross_validated_r2(
            states[index],
            y,
            index,
            sample_segments,
            alpha=config.ridge_alpha,
            folds=config.blocked_folds,
            purge=max(config.purge, lag),
        )
        current_r2 = float(input_baselines["memory"][str(lag)])
        memory[str(lag)] = {
            "samples": int(len(index)),
            "reservoir_r2": reservoir_r2,
            "current_input_r2": current_r2,
            "excess_r2": reservoir_r2 - current_r2,
        }

    current_index = _valid_lag_indices(trajectory.segment_ids, 0)
    current_r2 = cross_validated_r2(
        states[current_index],
        targets[current_index],
        current_index,
        trajectory.segment_ids[current_index],
        alpha=config.ridge_alpha,
        folds=config.blocked_folds,
        purge=config.purge,
    )

    future_index = _valid_future_indices(trajectory.segment_ids)
    future_y = targets[future_index + 1]
    future_reservoir_r2 = cross_validated_r2(
        states[future_index],
        future_y,
        future_index,
        trajectory.segment_ids[future_index],
        alpha=config.ridge_alpha,
        folds=config.blocked_folds,
        purge=config.purge,
    )
    future_current_r2 = float(input_baselines["future_r2"])

    excess_values = [
        value["excess_r2"]
        for value in memory.values()
        if np.isfinite(value["excess_r2"])
    ]
    memory_excess_mean = (
        float(np.mean(excess_values)) if excess_values else float("nan")
    )

    return {
        "schema": PROBE_SCHEMA,
        "variant": variant,
        "projection_seed": config.projection_seed,
        "topology_seed": config.topology_seed,
        "state_dim": len(core.body_ids),
        "edge_count": reservoir.edge_count,
        "rewire_swaps": int(reservoir.rewire_swaps),
        "trajectory_frames": len(trajectory.frames),
        "trajectory_segments": int(trajectory.segment_ids.max() + 1),
        "source_frame_sha256": trajectory.source_frame_sha256,
        "source_receipt_sha256": trajectory.source_receipt_sha256,
        "config": asdict(config),
        "health": health,
        "current_frame_sketch_r2": current_r2,
        "future_prediction": {
            "samples": len(future_index),
            "reservoir_r2": future_reservoir_r2,
            "current_input_r2": future_current_r2,
            "excess_r2": future_reservoir_r2 - future_current_r2,
        },
        "memory": memory,
        "memory_excess_mean_r2": memory_excess_mean,
    }


def _output_name(variant: str, projection_seed: int, topology_seed: int) -> str:
    return f"{variant}-p{projection_seed}-t{topology_seed}.json"


def run_probe_to_file(
    graph_root: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    variant: str,
    config: ProbeConfig,
    overwrite: bool = False,
) -> Path:
    graph_root = Path(graph_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _output_name(
        variant,
        config.projection_seed,
        config.topology_seed,
    )
    if output_path.exists() and not overwrite:
        return output_path

    bundle = GraphBundle.load(graph_root)
    bundle.validate(require_sign=True, require_qualified=False)
    trajectory = load_shared_trajectory(run_dir)
    result = run_probe(
        bundle,
        trajectory,
        variant=variant,
        config=config,
    )
    result["graph_artifacts"] = _graph_artifact_hashes(graph_root)
    temp = output_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temp.replace(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one identical ARC trajectory through a FlyARC topology and quantify "
            "memory/prediction/state geometry"
        )
    )
    parser.add_argument("graph")
    parser.add_argument("source_run")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=PROBE_VARIANTS, required=True)
    parser.add_argument("--projection-seed", type=int, default=1701)
    parser.add_argument("--topology-seed", type=int, default=2903)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--target-sketch-seed", type=int, default=8801)
    parser.add_argument("--target-sketch-dim", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = ProbeConfig(
        max_nodes=args.max_nodes,
        projection_seed=args.projection_seed,
        topology_seed=args.topology_seed,
        target_sketch_seed=args.target_sketch_seed,
        target_sketch_dim=args.target_sketch_dim,
        ridge_alpha=args.ridge_alpha,
    )
    path = run_probe_to_file(
        args.graph,
        args.source_run,
        args.output_dir,
        variant=args.variant,
        config=config,
        overwrite=args.overwrite,
    )
    print(path)


if __name__ == "__main__":
    main()
