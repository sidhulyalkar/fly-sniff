from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .graph import GraphBundle
from .rewire import degree_preserving_rewire
from .runtime import ConnectomeRuntime


@dataclass(frozen=True)
class LoomConfig:
    """Physically grounded looming geometry for R002."""

    steps: int = 96
    dt: float = 0.02
    onset_steps: int = 8
    object_radius: float = 0.012
    initial_distance: float = 0.30
    speed: float = 0.20
    miss_offset: float = 0.060
    size_scale_rad: float = 1.20
    velocity_scale_rad_s: float = 7.0

    def validate(self) -> None:
        if self.steps < 16:
            raise ValueError("steps must be >= 16")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if not 0 <= self.onset_steps < self.steps - 4:
            raise ValueError("onset_steps must leave at least four active frames")
        if self.object_radius <= 0.0:
            raise ValueError("object_radius must be > 0")
        if self.initial_distance <= self.object_radius:
            raise ValueError("initial_distance must exceed object_radius")
        if self.speed <= 0.0:
            raise ValueError("speed must be > 0")
        if self.miss_offset <= self.object_radius:
            raise ValueError("miss_offset must exceed object_radius")
        if self.size_scale_rad <= 0.0 or self.velocity_scale_rad_s <= 0.0:
            raise ValueError("loom normalization scales must be > 0")


@dataclass(frozen=True)
class LoomFrame:
    frame: int
    t: float
    side: str
    trajectory: str
    axial_distance: float
    distance: float
    angular_size_rad: float
    angular_velocity_rad_s: float
    inputs: dict[str, float]


@dataclass(frozen=True)
class LoomTrialResult:
    seed: int
    side: str
    trajectory: str
    mean_turn: float | None
    away_correct: bool | None
    peak_escape: float
    escape_auc: float
    escape_latency_s: float | None
    peak_activity: float
    peak_size_drive: float
    peak_velocity_drive: float


def _geometry(config: LoomConfig, trajectory: str, active_t: float) -> tuple[float, float, float]:
    if trajectory not in {"direct-hit", "near-miss"}:
        raise ValueError("trajectory must be 'direct-hit' or 'near-miss'")
    axial = config.initial_distance - config.speed * active_t
    lateral = 0.0 if trajectory == "direct-hit" else config.miss_offset
    distance = float(np.hypot(axial, lateral))
    distance = max(distance, config.object_radius * 0.20)
    angular_size = float(2.0 * np.arctan2(config.object_radius, distance))
    return float(axial), distance, angular_size


def generate_loom_frames(
    side: str,
    trajectory: str,
    *,
    config: LoomConfig | None = None,
) -> list[LoomFrame]:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    cfg = config or LoomConfig()
    cfg.validate()

    frames: list[LoomFrame] = []
    previous_size = 0.0
    for index in range(cfg.steps):
        active_index = max(0, index - cfg.onset_steps)
        active_t = active_index * cfg.dt
        if index < cfg.onset_steps:
            axial = cfg.initial_distance
            distance = cfg.initial_distance
            angular_size = 0.0
        else:
            axial, distance, angular_size = _geometry(cfg, trajectory, active_t)

        angular_velocity = max((angular_size - previous_size) / cfg.dt, 0.0)
        previous_size = angular_size
        size_drive = float(np.clip(angular_size / cfg.size_scale_rad, 0.0, 1.0))
        velocity_drive = float(np.clip(angular_velocity / cfg.velocity_scale_rad_s, 0.0, 1.0))
        inputs = {
            "loom_size_left": size_drive if side == "left" else 0.0,
            "loom_size_right": size_drive if side == "right" else 0.0,
            "loom_velocity_left": velocity_drive if side == "left" else 0.0,
            "loom_velocity_right": velocity_drive if side == "right" else 0.0,
        }
        frames.append(
            LoomFrame(
                frame=index,
                t=index * cfg.dt,
                side=side,
                trajectory=trajectory,
                axial_distance=axial,
                distance=distance,
                angular_size_rad=angular_size,
                angular_velocity_rad_s=angular_velocity,
                inputs=inputs,
            )
        )
    return frames


def write_stimulus(path: str | Path, frames: list[LoomFrame]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for frame in frames:
            handle.write(json.dumps(asdict(frame), sort_keys=True) + "\n")


def _jitter_config(base: LoomConfig, seed: int) -> LoomConfig:
    rng = np.random.default_rng(seed)
    return replace(
        base,
        object_radius=base.object_radius * float(rng.uniform(0.85, 1.15)),
        initial_distance=base.initial_distance * float(rng.uniform(0.92, 1.08)),
        speed=base.speed * float(rng.uniform(0.82, 1.18)),
        miss_offset=base.miss_offset * float(rng.uniform(0.90, 1.20)),
    )


def run_loom_trial(
    bundle: GraphBundle,
    side: str,
    trajectory: str,
    *,
    seed: int,
    config: LoomConfig | None = None,
    require_qualified: bool = True,
    disabled_inputs: set[str] | None = None,
) -> LoomTrialResult:
    cfg = _jitter_config(config or LoomConfig(), seed)
    frames = generate_loom_frames(side, trajectory, config=cfg)
    runtime = ConnectomeRuntime(bundle, require_qualified=require_qualified)
    runtime.reset()
    disabled_inputs = disabled_inputs or set()

    has_steering = "steer_left" in bundle.roles and "steer_right" in bundle.roles
    turns: list[float] = []
    escapes: list[float] = []
    activity: list[float] = []
    size_drive: list[float] = []
    velocity_drive: list[float] = []

    for frame in frames:
        inputs = {
            key: (0.0 if key in disabled_inputs else value)
            for key, value in frame.inputs.items()
        }
        readouts = ["escape"]
        if has_steering:
            readouts.extend(["steer_left", "steer_right"])
        snapshot = runtime.step(inputs, readouts=readouts)
        if has_steering:
            left = snapshot.readouts.get("steer_left", 0.0)
            right = snapshot.readouts.get("steer_right", 0.0)
            turns.append(float(np.tanh(2.4 * (right - left))))
        escapes.append(float(snapshot.readouts["escape"]))
        activity.append(snapshot.activity_max)
        size_drive.append(max(inputs["loom_size_left"], inputs["loom_size_right"]))
        velocity_drive.append(max(inputs["loom_velocity_left"], inputs["loom_velocity_right"]))

    response_start = cfg.onset_steps + max(4, (cfg.steps - cfg.onset_steps) // 2)
    mean_turn: float | None = None
    away_correct: bool | None = None
    if has_steering:
        mean_turn = float(np.mean(turns[response_start:]))
        expected_sign = -1.0 if side == "left" else 1.0
        away_correct = bool(expected_sign * mean_turn > 0.0)

    baseline = float(np.mean(escapes[: cfg.onset_steps])) if cfg.onset_steps else 0.0
    peak_escape = float(max(escapes))
    # A deterministic response threshold based on the trial's dynamic range, not a biological firing threshold.
    response_threshold = baseline + 0.25 * max(peak_escape - baseline, 0.0)
    latency = None
    for index in range(cfg.onset_steps, len(escapes)):
        if escapes[index] > response_threshold and peak_escape > baseline:
            latency = float((index - cfg.onset_steps) * cfg.dt)
            break

    return LoomTrialResult(
        seed=seed,
        side=side,
        trajectory=trajectory,
        mean_turn=mean_turn,
        away_correct=away_correct,
        peak_escape=peak_escape,
        escape_auc=float(np.trapezoid(escapes, dx=cfg.dt)),
        escape_latency_s=latency,
        peak_activity=float(max(activity)),
        peak_size_drive=float(max(size_drive)),
        peak_velocity_drive=float(max(velocity_drive)),
    )


def benchmark_loom(
    bundle: GraphBundle,
    *,
    trials: int = 40,
    seed: int = 24017,
    config: LoomConfig | None = None,
    require_qualified: bool = True,
    disabled_inputs: set[str] | None = None,
) -> dict:
    if trials < 4:
        raise ValueError("trials must be >= 4")
    base = config or LoomConfig()
    direct: list[LoomTrialResult] = []
    near: list[LoomTrialResult] = []
    for index in range(trials):
        side = "left" if index % 2 == 0 else "right"
        trial_seed = seed + index * 7919
        direct.append(
            run_loom_trial(
                bundle,
                side,
                "direct-hit",
                seed=trial_seed,
                config=base,
                require_qualified=require_qualified,
                disabled_inputs=disabled_inputs,
            )
        )
        near.append(
            run_loom_trial(
                bundle,
                side,
                "near-miss",
                seed=trial_seed,
                config=base,
                require_qualified=require_qualified,
                disabled_inputs=disabled_inputs,
            )
        )

    steering_available = all(row.away_correct is not None for row in direct)
    away_accuracy: float | None = None
    signed_margin: float | None = None
    if steering_available:
        away_accuracy = float(np.mean([bool(row.away_correct) for row in direct]))
        signed_margin = float(
            np.mean(
                [
                    (-1.0 if row.side == "left" else 1.0) * float(row.mean_turn)
                    for row in direct
                ]
            )
        )

    peak_separation = float(
        np.mean([hit.peak_escape - miss.peak_escape for hit, miss in zip(direct, near, strict=True)])
    )
    auc_separation = float(
        np.mean([hit.escape_auc - miss.escape_auc for hit, miss in zip(direct, near, strict=True)])
    )
    hit_latencies = [row.escape_latency_s for row in direct if row.escape_latency_s is not None]

    return {
        "protocol": "R002-loom-escape-v2",
        "trials_per_trajectory": trials,
        "primary_endpoint": "escape_direct_minus_near_miss",
        "escape_direct_minus_near_miss": peak_separation,
        "escape_auc_direct_minus_near_miss": auc_separation,
        "median_direct_hit_latency_s": (
            float(np.median(hit_latencies)) if hit_latencies else None
        ),
        "directional_steering_evaluated": steering_available,
        "away_accuracy_direct_hit": away_accuracy,
        "mean_signed_away_margin": signed_margin,
        "disabled_inputs": sorted(disabled_inputs or set()),
        "claim_status": (
            "qualified-malecns-modeled-dynamics"
            if bundle.manifest and bundle.manifest.get("qualification_status") == "qualified"
            else "candidate-malecns-development"
        ),
        "direct_hit": [asdict(row) for row in direct],
        "near_miss": [asdict(row) for row in near],
    }


def compare_with_controls(
    bundle: GraphBundle,
    *,
    trials: int,
    seed: int,
    rewire_seed: int,
    allow_candidate: bool,
) -> dict:
    require_qualified = not allow_candidate
    intact = benchmark_loom(bundle, trials=trials, seed=seed, require_qualified=require_qualified)
    rewired_bundle = degree_preserving_rewire(bundle, seed=rewire_seed)
    rewired = benchmark_loom(
        rewired_bundle,
        trials=trials,
        seed=seed,
        require_qualified=require_qualified,
    )
    size_lesion = benchmark_loom(
        bundle,
        trials=trials,
        seed=seed,
        require_qualified=require_qualified,
        disabled_inputs={"loom_size_left", "loom_size_right"},
    )
    velocity_lesion = benchmark_loom(
        bundle,
        trials=trials,
        seed=seed,
        require_qualified=require_qualified,
        disabled_inputs={"loom_velocity_left", "loom_velocity_right"},
    )
    endpoint = "escape_direct_minus_near_miss"
    return {
        "protocol": "R002-loom-escape-controls-v2",
        "primary_endpoint": endpoint,
        "rewire_seed": rewire_seed,
        "intact": intact,
        "degree_preserving_rewire": rewired,
        "size_channel_lesion": size_lesion,
        "velocity_channel_lesion": velocity_lesion,
        "delta_intact_minus_rewire": intact[endpoint] - rewired[endpoint],
        "delta_intact_minus_size_lesion": intact[endpoint] - size_lesion[endpoint],
        "delta_intact_minus_velocity_lesion": intact[endpoint] - velocity_lesion[endpoint],
        "directional_steering_is_primary_claim": False,
    }


def compare_with_rewire(
    bundle: GraphBundle,
    *,
    trials: int,
    seed: int,
    rewire_seed: int,
    allow_candidate: bool,
) -> dict:
    """Backward-compatible alias for the stronger R002 control pack."""
    return compare_with_controls(
        bundle,
        trials=trials,
        seed=seed,
        rewire_seed=rewire_seed,
        allow_candidate=allow_candidate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or emit the R002 looming/escape assay")
    parser.add_argument("--circuit", help="GraphBundle directory for connectome evaluation")
    parser.add_argument("--allow-candidate", action="store_true")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=24017)
    parser.add_argument("--rewire-seed", type=int, default=24018)
    parser.add_argument("--emit-stimulus", help="write one JSONL stimulus instead of evaluating")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--trajectory", choices=["direct-hit", "near-miss"], default="direct-hit")
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()

    if args.emit_stimulus:
        frames = generate_loom_frames(args.side, args.trajectory)
        write_stimulus(args.emit_stimulus, frames)
        print(f"wrote {len(frames)} frames to {args.emit_stimulus}")
        return

    if not args.circuit:
        parser.error("provide --circuit or --emit-stimulus")
    bundle = GraphBundle.load(args.circuit)
    status = (bundle.manifest or {}).get("qualification_status")
    if status != "qualified" and not args.allow_candidate:
        raise SystemExit(
            "loom evaluation requires a qualified graph; pass --allow-candidate only for "
            "development runs that are not eligible for MaleCNS claims"
        )
    report = compare_with_controls(
        bundle,
        trials=args.trials,
        seed=args.seed,
        rewire_seed=args.rewire_seed,
        allow_candidate=args.allow_candidate,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
