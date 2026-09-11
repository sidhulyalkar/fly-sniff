from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .graph import GraphBundle
from .looming import LoomConfig, generate_loom_frames
from .rewire import degree_preserving_rewire
from .runtime import ConnectomeRuntime


@dataclass(frozen=True)
class EscapeProbeTrial:
    seed: int
    side: str
    direct_peak_target: float
    direct_peak_opposite: float
    near_peak_target: float
    near_peak_opposite: float
    direct_minus_near_target: float
    direct_lateralization: float
    direct_stronger_than_near: bool


def _peak_escape(
    bundle: GraphBundle,
    side: str,
    trajectory: str,
    *,
    seed: int,
    config: LoomConfig,
    require_qualified: bool,
) -> tuple[float, float]:
    runtime = ConnectomeRuntime(bundle, require_qualified=require_qualified)
    runtime.reset()
    target = "escape_left" if side == "left" else "escape_right"
    opposite = "escape_right" if side == "left" else "escape_left"
    if target not in bundle.roles or opposite not in bundle.roles:
        raise ValueError("R002 escape probe requires escape_left and escape_right roles")

    # Jitter is intentionally controlled by the public looming benchmark rather
    # than hidden inside the runtime. A small deterministic perturbation keeps
    # paired direct-hit/near-miss trials from collapsing to one geometry.
    rng = np.random.default_rng(seed)
    cfg = LoomConfig(
        **{
            **asdict(config),
            "object_radius": config.object_radius * float(rng.uniform(0.85, 1.15)),
            "initial_distance": config.initial_distance * float(rng.uniform(0.92, 1.08)),
            "speed": config.speed * float(rng.uniform(0.82, 1.18)),
            "miss_offset": config.miss_offset * float(rng.uniform(0.90, 1.20)),
        }
    )
    frames = generate_loom_frames(side, trajectory, config=cfg)
    target_values: list[float] = []
    opposite_values: list[float] = []
    for frame in frames:
        snapshot = runtime.step(frame.inputs, readouts=(target, opposite))
        target_values.append(snapshot.readouts[target])
        opposite_values.append(snapshot.readouts[opposite])
    return float(max(target_values)), float(max(opposite_values))


def run_escape_probe(
    bundle: GraphBundle,
    *,
    trials: int = 40,
    seed: int = 24017,
    config: LoomConfig | None = None,
    require_qualified: bool = True,
) -> dict:
    if trials < 4:
        raise ValueError("trials must be >= 4")
    cfg = config or LoomConfig()
    rows: list[EscapeProbeTrial] = []
    for index in range(trials):
        side = "left" if index % 2 == 0 else "right"
        trial_seed = seed + index * 7919
        direct_target, direct_opposite = _peak_escape(
            bundle,
            side,
            "direct-hit",
            seed=trial_seed,
            config=cfg,
            require_qualified=require_qualified,
        )
        near_target, near_opposite = _peak_escape(
            bundle,
            side,
            "near-miss",
            seed=trial_seed,
            config=cfg,
            require_qualified=require_qualified,
        )
        rows.append(
            EscapeProbeTrial(
                seed=trial_seed,
                side=side,
                direct_peak_target=direct_target,
                direct_peak_opposite=direct_opposite,
                near_peak_target=near_target,
                near_peak_opposite=near_opposite,
                direct_minus_near_target=direct_target - near_target,
                direct_lateralization=direct_target - direct_opposite,
                direct_stronger_than_near=bool(direct_target > near_target),
            )
        )

    return {
        "protocol": "R002-escape-readout-v1",
        "trials": trials,
        "direct_stronger_than_near_rate": float(
            np.mean([row.direct_stronger_than_near for row in rows])
        ),
        "mean_direct_minus_near_target": float(
            np.mean([row.direct_minus_near_target for row in rows])
        ),
        "mean_direct_lateralization": float(
            np.mean([row.direct_lateralization for row in rows])
        ),
        "claim_status": (
            "qualified-malecns"
            if bundle.manifest and bundle.manifest.get("qualification_status") == "qualified"
            else "candidate-malecns-development"
        ),
        "interpretation": (
            "Modeled DNp01/GF escape activity only. This protocol does not infer a turn command, "
            "behavioral latency, perception, or a complete biological escape response."
        ),
        "trials_detail": [asdict(row) for row in rows],
    }


def compare_escape_with_rewire(
    bundle: GraphBundle,
    *,
    trials: int = 40,
    seed: int = 24017,
    rewire_seed: int = 24018,
    allow_candidate: bool = False,
) -> dict:
    require_qualified = not allow_candidate
    intact = run_escape_probe(
        bundle,
        trials=trials,
        seed=seed,
        require_qualified=require_qualified,
    )
    rewired_bundle = degree_preserving_rewire(bundle, seed=rewire_seed)
    rewired = run_escape_probe(
        rewired_bundle,
        trials=trials,
        seed=seed,
        require_qualified=require_qualified,
    )
    return {
        "protocol": "R002-escape-topology-pair-v1",
        "rewire_seed": rewire_seed,
        "intact": intact,
        "degree_preserving_rewire": rewired,
        "delta_direct_minus_near_target": (
            intact["mean_direct_minus_near_target"]
            - rewired["mean_direct_minus_near_target"]
        ),
        "delta_lateralization": (
            intact["mean_direct_lateralization"] - rewired["mean_direct_lateralization"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe DNp01/GF escape readouts in R002")
    parser.add_argument("circuit")
    parser.add_argument("--allow-candidate", action="store_true")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=24017)
    parser.add_argument("--rewire-seed", type=int, default=24018)
    parser.add_argument("--output")
    args = parser.parse_args()

    bundle = GraphBundle.load(args.circuit)
    status = (bundle.manifest or {}).get("qualification_status")
    if status != "qualified" and not args.allow_candidate:
        raise SystemExit(
            "R002 escape probing requires a qualified graph; pass --allow-candidate only for "
            "development output that is not eligible for MaleCNS claims"
        )
    report = compare_escape_with_rewire(
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
