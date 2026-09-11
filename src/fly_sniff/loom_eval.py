from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .graph import GraphBundle
from .looming import LoomingStimulusConfig, generate_looming_frames
from .rewire import degree_preserving_rewire
from .runtime import ConnectomeRuntime

DEFAULT_AZIMUTHS = (-60.0, -30.0, 0.0, 30.0, 60.0)
DEFAULT_THRESHOLD = 0.10


def _bundle_digest(bundle: GraphBundle) -> str:
    payload = {
        "nodes": sorted(bundle.nodes.bodyId.astype(int).tolist()),
        "edges": sorted(
            (
                int(row.source),
                int(row.target),
                float(row.weight),
                int(row.sign),
            )
            for row in bundle.edges.itertuples(index=False)
        ),
        "roles": {key: sorted(int(x) for x in value) for key, value in sorted(bundle.roles.items())},
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def lesion_roles(bundle: GraphBundle, role_names: tuple[str, ...], *, label: str) -> GraphBundle:
    """Silence role populations by removing their incident edges and role assignments."""

    missing = [role for role in role_names if role not in bundle.roles]
    if missing:
        raise ValueError(f"cannot lesion missing roles: {missing}")
    body_ids = {int(x) for role in role_names for x in bundle.roles[role]}
    edges = bundle.edges.loc[
        ~bundle.edges.source.astype(int).isin(body_ids)
        & ~bundle.edges.target.astype(int).isin(body_ids)
    ].copy()
    roles = {key: list(value) for key, value in bundle.roles.items()}
    for role in role_names:
        roles[role] = []
    manifest = copy.deepcopy(bundle.manifest) if bundle.manifest else {}
    manifest["graph_role"] = label
    manifest["lesion"] = {
        "roles": list(role_names),
        "body_ids": sorted(body_ids),
        "incident_edges_removed": int(len(bundle.edges) - len(edges)),
    }
    return GraphBundle(bundle.nodes.copy(), edges.reset_index(drop=True), roles, manifest)


def stimulus_grid(
    *,
    azimuths: tuple[float, ...] = DEFAULT_AZIMUTHS,
    frames: int = 120,
    dt: float = 1.0 / 60.0,
    object_radius: float = 0.12,
    initial_distance: float = 2.5,
    speed: float = 1.0,
    min_distance: float = 0.08,
    expansion_scale: float = 3.0,
) -> list[LoomingStimulusConfig]:
    configs: list[LoomingStimulusConfig] = []
    for azimuth in azimuths:
        for mode in ("approach", "recede"):
            configs.append(
                LoomingStimulusConfig(
                    frames=frames,
                    dt=dt,
                    object_radius=object_radius,
                    initial_distance=initial_distance,
                    speed=speed,
                    min_distance=min_distance,
                    azimuth_deg=float(azimuth),
                    expansion_scale=expansion_scale,
                    mode=mode,
                )
            )
    return configs


def evaluate_stimulus(
    bundle: GraphBundle,
    config: LoomingStimulusConfig,
    *,
    require_qualified: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    leak: float = 0.82,
    gain: float = 1.6,
) -> dict[str, Any]:
    runtime = ConnectomeRuntime(
        bundle,
        leak=leak,
        gain=gain,
        require_qualified=require_qualified,
    )
    frames = generate_looming_frames(config)
    magnitudes: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []

    for frame in frames:
        snapshot = runtime.step(
            frame["inputs"],
            readouts=("escape_left", "escape_right"),
        )
        left = float(snapshot.readouts["escape_left"])
        right = float(snapshot.readouts["escape_right"])
        left_values.append(left)
        right_values.append(right)
        magnitudes.append(0.5 * (abs(left) + abs(right)))

    magnitude = np.asarray(magnitudes, dtype=float)
    peak_index = int(np.argmax(magnitude))
    peak_left = left_values[peak_index]
    peak_right = right_values[peak_index]
    denom = abs(peak_left) + abs(peak_right) + 1e-12
    crossings = np.flatnonzero(magnitude >= threshold)
    latency = float(crossings[0] * config.dt) if len(crossings) else None

    return {
        "mode": config.mode,
        "azimuth_deg": float(config.azimuth_deg),
        "peak_escape_magnitude": float(magnitude[peak_index]),
        "escape_auc": float(np.trapezoid(magnitude, dx=config.dt)),
        "latency_s_at_fixed_threshold": latency,
        "threshold": float(threshold),
        "peak_escape_left": float(peak_left),
        "peak_escape_right": float(peak_right),
        "peak_lateralization_index": float((peak_right - peak_left) / denom),
        "stimulus": asdict(config),
    }


def summarize_approach_selectivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["azimuth_deg"], row["mode"]): row for row in rows}
    summary: list[dict[str, Any]] = []
    for azimuth in sorted({float(row["azimuth_deg"]) for row in rows}):
        approach = by_key.get((azimuth, "approach"))
        recede = by_key.get((azimuth, "recede"))
        if approach is None or recede is None:
            continue
        ap = float(approach["peak_escape_magnitude"])
        rp = float(recede["peak_escape_magnitude"])
        aa = float(approach["escape_auc"])
        ra = float(recede["escape_auc"])
        summary.append(
            {
                "azimuth_deg": azimuth,
                "peak_approach_selectivity": float((ap - rp) / (ap + rp + 1e-12)),
                "auc_approach_selectivity": float((aa - ra) / (aa + ra + 1e-12)),
                "approach_peak": ap,
                "recede_peak": rp,
                "approach_auc": aa,
                "recede_auc": ra,
            }
        )
    return summary


def evaluate_bundle(
    bundle: GraphBundle,
    *,
    label: str,
    require_qualified: bool = True,
    configs: list[LoomingStimulusConfig] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    leak: float = 0.82,
    gain: float = 1.6,
) -> dict[str, Any]:
    configs = stimulus_grid() if configs is None else configs
    rows = [
        evaluate_stimulus(
            bundle,
            config,
            require_qualified=require_qualified,
            threshold=threshold,
            leak=leak,
            gain=gain,
        )
        for config in configs
    ]
    selectivity = summarize_approach_selectivity(rows)
    return {
        "label": label,
        "graph_sha256": _bundle_digest(bundle),
        "rows": rows,
        "approach_selectivity": selectivity,
        "mean_peak_approach_selectivity": (
            float(np.mean([row["peak_approach_selectivity"] for row in selectivity]))
            if selectivity
            else None
        ),
        "mean_auc_approach_selectivity": (
            float(np.mean([row["auc_approach_selectivity"] for row in selectivity]))
            if selectivity
            else None
        ),
    }


def evaluate_control_suite(
    biological: GraphBundle,
    *,
    require_qualified: bool = True,
    rewire_seed: int = 20260911,
    swaps_per_edge: int = 8,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    biological.validate(require_sign=True, require_qualified=require_qualified)
    controls = {
        "intact": biological,
        "degree-preserving-rewire": degree_preserving_rewire(
            biological,
            seed=rewire_seed,
            swaps_per_edge=swaps_per_edge,
        ),
        "lplc2-lesion": lesion_roles(
            biological,
            ("loom_left", "loom_right"),
            label="lplc2-role-lesion",
        ),
        "dnp06-lesion": lesion_roles(
            biological,
            ("escape_left", "escape_right"),
            label="dnp06-role-lesion",
        ),
    }
    configs = stimulus_grid()
    evaluations = {
        label: evaluate_bundle(
            bundle,
            label=label,
            require_qualified=require_qualified,
            configs=configs,
            threshold=threshold,
        )
        for label, bundle in controls.items()
    }
    return {
        "contract": "loom-escape-eval-v0",
        "claim_status": "qualified-graph-evaluation" if require_qualified else "development-candidate",
        "sensory_model": "abstract-positive-angular-expansion-v0",
        "motor_semantics": "DNp06 population readout only; no turn direction asserted",
        "stimulus_grid": [asdict(config) for config in configs],
        "rewire": {"seed": rewire_seed, "swaps_per_edge": swaps_per_edge},
        "fixed_threshold": float(threshold),
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate looming response across intact, rewired, and lesioned connectome graphs"
    )
    parser.add_argument("graph", help="GraphBundle directory with loom/escape roles")
    parser.add_argument("--output", default="results/loom-escape-v0.json")
    parser.add_argument("--rewire-seed", type=int, default=20260911)
    parser.add_argument("--swaps-per-edge", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help="development only: evaluate an unqualified graph; receipt remains non-evidence",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite looming evaluation: {output}")
    bundle = GraphBundle.load(args.graph)
    report = evaluate_control_suite(
        bundle,
        require_qualified=not args.allow_candidate,
        rewire_seed=args.rewire_seed,
        swaps_per_edge=args.swaps_per_edge,
        threshold=args.threshold,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "claim_status": report["claim_status"]}, indent=2))


if __name__ == "__main__":
    main()
