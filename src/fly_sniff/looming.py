from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LoomingStimulusConfig:
    frames: int = 120
    dt: float = 1.0 / 60.0
    object_radius: float = 0.12
    initial_distance: float = 2.5
    speed: float = 1.0
    min_distance: float = 0.08
    azimuth_deg: float = 0.0
    expansion_scale: float = 3.0
    mode: str = "approach"

    def validate(self) -> None:
        if self.frames < 2:
            raise ValueError("frames must be >= 2")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.object_radius <= 0.0:
            raise ValueError("object_radius must be > 0")
        if self.initial_distance <= self.min_distance:
            raise ValueError("initial_distance must exceed min_distance")
        if self.speed <= 0.0:
            raise ValueError("speed must be > 0")
        if self.min_distance <= 0.0:
            raise ValueError("min_distance must be > 0")
        if not -90.0 <= self.azimuth_deg <= 90.0:
            raise ValueError("azimuth_deg must be between -90 and +90 degrees")
        if self.expansion_scale <= 0.0:
            raise ValueError("expansion_scale must be > 0")
        if self.mode not in {"approach", "recede"}:
            raise ValueError("mode must be 'approach' or 'recede'")


def _distance(config: LoomingStimulusConfig, t: float) -> float:
    if config.mode == "approach":
        value = config.initial_distance - config.speed * t
    else:
        value = min(config.initial_distance, config.min_distance + config.speed * t)
    return float(max(config.min_distance, value))


def _side_gains(azimuth_deg: float) -> tuple[float, float]:
    """Return an explicit engineering split for left/right loom channels.

    Negative azimuth is left visual field, positive is right. This is not a
    retinotopic retinal model. It only creates a deterministic lateralized drive
    for development and candidate-circuit probing.
    """

    lateral = float(np.sin(np.deg2rad(azimuth_deg)))
    left = float(np.clip(0.5 * (1.0 - lateral), 0.0, 1.0))
    right = float(np.clip(0.5 * (1.0 + lateral), 0.0, 1.0))
    return left, right


def generate_looming_frames(config: LoomingStimulusConfig) -> list[dict]:
    """Generate a deterministic abstract looming-channel stimulus.

    The adapter computes angular expansion from simple collision geometry and
    maps positive expansion into ``loom_left`` / ``loom_right`` roles. It does
    not model photoreceptors, T4/T5 motion detectors, or LPLC2 radial motion
    opponency. Those are a later retinal-adapter layer.
    """

    config.validate()
    left_gain, right_gain = _side_gains(config.azimuth_deg)
    frames: list[dict] = []
    previous_angle: float | None = None

    for index in range(config.frames):
        t = index * config.dt
        distance = _distance(config, t)
        angular_size = float(2.0 * np.arctan2(config.object_radius, distance))
        if previous_angle is None:
            expansion_rate = 0.0
        else:
            expansion_rate = max(0.0, (angular_size - previous_angle) / config.dt)
        previous_angle = angular_size
        loom_drive = float(np.tanh(config.expansion_scale * expansion_rate))

        frames.append(
            {
                "t": float(t),
                "inputs": {
                    "loom_left": loom_drive * left_gain,
                    "loom_right": loom_drive * right_gain,
                },
                "stimulus": {
                    "mode": config.mode,
                    "distance": distance,
                    "angular_size_rad": angular_size,
                    "angular_expansion_rate": expansion_rate,
                    "azimuth_deg": config.azimuth_deg,
                },
            }
        )
    return frames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stimulus(path: str | Path, frames: list[dict]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for frame in frames:
            handle.write(json.dumps(frame, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic abstract looming channels for MaleCNS replay"
    )
    parser.add_argument("--output", default="artifacts/loom-approach.jsonl")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--radius", type=float, default=0.12)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--min-distance", type=float, default=0.08)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--expansion-scale", type=float, default=3.0)
    parser.add_argument("--mode", choices=["approach", "recede"], default="approach")
    args = parser.parse_args()

    config = LoomingStimulusConfig(
        frames=args.frames,
        dt=args.dt,
        object_radius=args.radius,
        initial_distance=args.distance,
        speed=args.speed,
        min_distance=args.min_distance,
        azimuth_deg=args.azimuth,
        expansion_scale=args.expansion_scale,
        mode=args.mode,
    )
    frames = generate_looming_frames(config)
    output = write_stimulus(args.output, frames)
    receipt = {
        "contract": "abstract-loom-channel-v0",
        "claim_status": "adapter-development-only",
        "config": asdict(config),
        "frames": len(frames),
        "stimulus": str(output),
        "stimulus_sha256": _sha256(output),
        "warning": (
            "loom_left/right are engineered angular-expansion channels, not a retinal or "
            "LPLC2 physiology model"
        ),
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
