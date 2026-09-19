from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .flyarc import _git_head, _graph_artifact_hashes, _sha256, select_structural_core
from .flyarc_probes import (
    DEFAULT_LAGS,
    PROBE_SCHEMA,
    ProbeConfig,
    encode_trajectory,
    frame_target_sketch,
    load_shared_trajectory,
    run_probe,
)
from .graph import GraphBundle

MARATHON_SCHEMA = "flyarc-representation-marathon-v1"


def _json_config(config: MarathonConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config)))


@dataclass(frozen=True)
class MarathonConfig:
    hours: float = 6.0
    projection_count: int = 4
    rewires_per_projection: int = 16
    randoms_per_projection: int = 8
    base_projection_seed: int = 1701
    base_topology_seed: int = 2903
    max_nodes: int = 4096
    target_sketch_seed: int = 8801
    target_sketch_dim: int = 128
    ridge_alpha: float = 0.1
    lags: tuple[int, ...] = DEFAULT_LAGS
    input_fanout: int = 4
    leak: float = 0.82
    gain: float = 1.6
    health_neurons: int = 256
    blocked_folds: int = 5
    purge: int = 32


@dataclass(frozen=True)
class ProbeJob:
    variant: str
    projection_seed: int
    topology_seed: int

    @property
    def job_id(self) -> str:
        return f"{self.variant}-p{self.projection_seed}-t{self.topology_seed}"


def projection_seed(config: MarathonConfig, index: int) -> int:
    return config.base_projection_seed + 10_007 * index


def topology_seed(config: MarathonConfig, projection_index: int, replicate: int, offset: int) -> int:
    return (
        config.base_topology_seed
        + offset
        + 100_003 * projection_index
        + 997 * replicate
    )


def build_schedule(config: MarathonConfig) -> list[ProbeJob]:
    if config.projection_count < 1:
        raise ValueError("projection_count must be at least one")
    if config.rewires_per_projection < 0 or config.randoms_per_projection < 0:
        raise ValueError("null counts must be non-negative")

    jobs: list[ProbeJob] = []

    for projection_index in range(config.projection_count):
        pseed = projection_seed(config, projection_index)
        anchor_seed = topology_seed(config, projection_index, 0, 0)
        for variant in ("intact", "leak_only", "stateless"):
            jobs.append(ProbeJob(variant, pseed, anchor_seed))

    max_replicates = max(config.rewires_per_projection, config.randoms_per_projection)
    for replicate in range(max_replicates):
        for projection_index in range(config.projection_count):
            pseed = projection_seed(config, projection_index)
            if replicate < config.rewires_per_projection:
                jobs.append(
                    ProbeJob(
                        "rewire",
                        pseed,
                        topology_seed(config, projection_index, replicate, 1_000_000),
                    )
                )
            if replicate < config.randoms_per_projection:
                jobs.append(
                    ProbeJob(
                        "random",
                        pseed,
                        topology_seed(config, projection_index, replicate, 2_000_000),
                    )
                )
    return jobs


def probe_config(config: MarathonConfig, job: ProbeJob) -> ProbeConfig:
    return ProbeConfig(
        max_nodes=config.max_nodes,
        input_fanout=config.input_fanout,
        leak=config.leak,
        gain=config.gain,
        projection_seed=job.projection_seed,
        topology_seed=job.topology_seed,
        target_sketch_seed=config.target_sketch_seed,
        target_sketch_dim=config.target_sketch_dim,
        ridge_alpha=config.ridge_alpha,
        lags=config.lags,
        health_neurons=config.health_neurons,
        blocked_folds=config.blocked_folds,
        purge=config.purge,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"unexpected probe schema in {path}")
    return payload


def result_path(results_dir: Path, job: ProbeJob) -> Path:
    return results_dir / f"{job.job_id}.json"


def _flatten_result(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": payload["variant"],
        "projection_seed": payload["projection_seed"],
        "topology_seed": payload["topology_seed"],
        "state_dim": payload["state_dim"],
        "edge_count": payload["edge_count"],
        "rewire_swaps": payload["rewire_swaps"],
        "memory_excess_mean_r2": payload["memory_excess_mean_r2"],
        "current_frame_sketch_r2": payload["current_frame_sketch_r2"],
        "future_reservoir_r2": payload["future_prediction"]["reservoir_r2"],
        "future_current_input_r2": payload["future_prediction"]["current_input_r2"],
        "future_excess_r2": payload["future_prediction"]["excess_r2"],
    }
    row.update(payload["health"])
    for lag, metrics in payload["memory"].items():
        row[f"memory_lag_{lag}_reservoir_r2"] = metrics["reservoir_r2"]
        row[f"memory_lag_{lag}_current_input_r2"] = metrics["current_input_r2"]
        row[f"memory_lag_{lag}_excess_r2"] = metrics["excess_r2"]
    return row


def _summary_stats(values: pd.Series) -> dict[str, float | int]:
    data = pd.to_numeric(values, errors="coerce").dropna()
    if data.empty:
        return {"n": 0}
    return {
        "n": len(data),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "median": float(data.median()),
        "q05": float(data.quantile(0.05)),
        "q25": float(data.quantile(0.25)),
        "q75": float(data.quantile(0.75)),
        "q95": float(data.quantile(0.95)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def aggregate_results(
    output_dir: str | Path,
    *,
    config: MarathonConfig,
    source_run: str | Path,
    graph_root: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir)
    results_dir = root / "results"
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise ValueError("no completed FlyARC probe jobs to aggregate")

    payloads = [_load_result(path) for path in paths]
    rows = [_flatten_result(payload) for payload in payloads]
    frame = pd.DataFrame(rows).sort_values(
        ["projection_seed", "variant", "topology_seed"],
        kind="mergesort",
    )
    frame.to_csv(root / "summary.csv", index=False)

    primary = "memory_excess_mean_r2"
    groups: dict[str, Any] = {}
    for variant, group in frame.groupby("variant", sort=True):
        groups[str(variant)] = {
            "primary": _summary_stats(group[primary]),
            "future_excess_r2": _summary_stats(group["future_excess_r2"]),
            "state_rms": _summary_stats(group["state_rms"]),
            "effective_dimension_participation": _summary_stats(
                group["effective_dimension_participation"]
            ),
        }

    projection_comparisons: list[dict[str, Any]] = []
    for pseed, group in frame.groupby("projection_seed", sort=True):
        intact = group[group["variant"] == "intact"]
        rewires = group[group["variant"] == "rewire"]
        randoms = group[group["variant"] == "random"]
        if intact.empty:
            continue
        intact_value = float(intact.iloc[0][primary])
        record: dict[str, Any] = {
            "projection_seed": int(pseed),
            "intact_memory_excess_mean_r2": intact_value,
            "rewire_n": len(rewires),
            "random_n": len(randoms),
        }
        if not rewires.empty:
            rewire_values = rewires[primary].astype(float)
            record.update(
                {
                    "rewire_median": float(rewire_values.median()),
                    "intact_minus_rewire_median": intact_value
                    - float(rewire_values.median()),
                    "intact_percentile_among_rewires": float(
                        np.mean(rewire_values.to_numpy() <= intact_value)
                    ),
                }
            )
        if not randoms.empty:
            random_values = randoms[primary].astype(float)
            record.update(
                {
                    "random_median": float(random_values.median()),
                    "intact_minus_random_median": intact_value
                    - float(random_values.median()),
                    "intact_percentile_among_randoms": float(
                        np.mean(random_values.to_numpy() <= intact_value)
                    ),
                }
            )
        projection_comparisons.append(record)

    aggregate = {
        "schema": MARATHON_SCHEMA,
        "status": "development-descriptive",
        "completed_jobs": len(frame),
        "configured_jobs": len(build_schedule(config)),
        "config": _json_config(config),
        "source_run": str(source_run),
        "source_receipt_sha256": _sha256(Path(source_run) / "receipt.json"),
        "graph_artifacts": _graph_artifact_hashes(graph_root),
        "groups": groups,
        "projection_comparisons": projection_comparisons,
        "claim_boundary": (
            "These are developmental descriptive statistics. They are not a confirmatory "
            "topology advantage claim."
        ),
    }
    _atomic_json(root / "aggregate.json", aggregate)
    _write_report(root, aggregate)
    _plot_memory(root, frame, config)
    _plot_primary(root, frame)
    _write_receipt(root)
    return aggregate


def _write_report(root: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# FlyARC representation marathon v1",
        "",
        (
            f"- Completed jobs: **{aggregate['completed_jobs']}** / "
            f"**{aggregate['configured_jobs']}**"
        ),
        "- Status: **development-descriptive**",
        "- Primary developmental metric: mean excess memory R² across frozen lags",
        "",
        (
            "Excess memory R² is the linear decodability of past frame information from reservoir "
            "state minus the decodability available from the current frame alone."
        ),
        "",
        "## Variant summaries",
        "",
        "| variant | n | memory excess mean | median | future excess mean | state RMS | eff. dim |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in sorted(aggregate["groups"]):
        group = aggregate["groups"][variant]
        primary = group["primary"]
        future = group["future_excess_r2"]
        rms = group["state_rms"]
        dim = group["effective_dimension_participation"]
        lines.append(
            f"| {variant} | {primary.get('n', 0)} | "
            f"{primary.get('mean', float('nan')):.5f} | "
            f"{primary.get('median', float('nan')):.5f} | "
            f"{future.get('mean', float('nan')):.5f} | "
            f"{rms.get('mean', float('nan')):.5f} | "
            f"{dim.get('mean', float('nan')):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-projection intact versus topology nulls",
            "",
            "| projection | intact | rewire median | Δ intact-rewire | intact rewire percentile |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for record in aggregate["projection_comparisons"]:
        lines.append(
            f"| {record['projection_seed']} | "
            f"{record['intact_memory_excess_mean_r2']:.5f} | "
            f"{record.get('rewire_median', float('nan')):.5f} | "
            f"{record.get('intact_minus_rewire_median', float('nan')):.5f} | "
            f"{record.get('intact_percentile_among_rewires', float('nan')):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            (
                "Do not select a new leak, gain, projection, reservoir size, or target sketch "
                "because it makes intact topology look better. Any confirmatory follow-up must "
                "be frozen in a new authority manifest before inspecting its outcome."
            ),
            "",
            (
                "See memory_curve.png, primary_metric_distribution.png, summary.csv, and "
                "aggregate.json for machine-readable detail."
            ),
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines))


def _plot_memory(root: Path, frame: pd.DataFrame, config: MarathonConfig) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    lags = list(config.lags)
    for variant, group in frame.groupby("variant", sort=True):
        matrix = np.asarray(
            [
                [
                    float(row[f"memory_lag_{lag}_excess_r2"])
                    for lag in lags
                ]
                for _, row in group.iterrows()
            ],
            dtype=float,
        )
        mean = np.nanmean(matrix, axis=0)
        axis.plot(lags, mean, marker="o", label=str(variant))
        if len(matrix) >= 4 and variant in {"rewire", "random"}:
            low = np.nanquantile(matrix, 0.10, axis=0)
            high = np.nanquantile(matrix, 0.90, axis=0)
            axis.fill_between(lags, low, high, alpha=0.15)

    axis.axhline(0.0, linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set_xticks(lags, labels=[str(lag) for lag in lags])
    axis.set_xlabel("steps into the past")
    axis.set_ylabel("excess memory R² over current-frame baseline")
    axis.set_title("FlyARC identical-trajectory memory probe")
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "memory_curve.png", dpi=170)
    plt.close(figure)


def _plot_primary(root: Path, frame: pd.DataFrame) -> None:
    order = [
        variant
        for variant in ("intact", "rewire", "random", "leak_only", "stateless")
        if variant in set(frame["variant"])
    ]
    values = [
        frame.loc[frame["variant"] == variant, "memory_excess_mean_r2"]
        .astype(float)
        .to_numpy()
        for variant in order
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.boxplot(values, labels=order, showmeans=True)
    axis.axhline(0.0, linewidth=1)
    axis.set_ylabel("mean excess memory R²")
    axis.set_title("FlyARC developmental topology-null distribution")
    figure.tight_layout()
    figure.savefig(root / "primary_metric_distribution.png", dpi=170)
    plt.close(figure)


def _write_receipt(root: Path) -> None:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"receipt.json", "console.log"}
        and not path.name.endswith(".tmp")
    ]
    receipt = {
        "schema": MARATHON_SCHEMA,
        "files": {
            str(path.relative_to(root)): _sha256(path)
            for path in sorted(paths)
        },
    }
    _atomic_json(root / "receipt.json", receipt)


def _manifest(
    graph_root: Path,
    source_run: Path,
    config: MarathonConfig,
    schedule: list[ProbeJob],
) -> dict[str, Any]:
    return {
        "schema": MARATHON_SCHEMA,
        "status": "development-preregistered",
        "git_head": _git_head(),
        "config": _json_config(config),
        "configured_jobs": len(schedule),
        "schedule": [asdict(job) | {"job_id": job.job_id} for job in schedule],
        "graph_artifacts": _graph_artifact_hashes(graph_root),
        "source_run": str(source_run),
        "source_receipt_sha256": _sha256(source_run / "receipt.json"),
        "primary_developmental_metric": "memory_excess_mean_r2",
        "claim_boundaries": [
            "All topologies receive the exact same recorded ARC observation stream.",
            (
                "The input projection changes only between frozen projection blocks and is "
                "shared within each block."
            ),
            "Excess memory is measured relative to current-frame decodability.",
            "Marathon results are developmental and descriptive, not confirmatory.",
            "A confirmatory topology test requires a separately frozen multi-game manifest.",
        ],
    }


def run_marathon(
    graph_root: str | Path,
    source_run: str | Path,
    output_dir: str | Path,
    *,
    config: MarathonConfig,
) -> dict[str, Any]:
    graph_root = Path(graph_root)
    source_run = Path(source_run)
    root = Path(output_dir)
    results_dir = root / "results"
    root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(config)
    manifest_path = root / "manifest.json"
    current_manifest = _manifest(graph_root, source_run, config, schedule)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior != current_manifest:
            raise ValueError(
                "marathon manifest mismatch; use a new output directory rather than mixing "
                "different experimental contracts"
            )
    else:
        _atomic_json(manifest_path, current_manifest)

    bundle = GraphBundle.load(graph_root)
    bundle.validate(require_sign=True, require_qualified=False)
    trajectory = load_shared_trajectory(source_run)
    core = select_structural_core(bundle, max_nodes=config.max_nodes)

    base_probe = ProbeConfig(
        max_nodes=config.max_nodes,
        target_sketch_seed=config.target_sketch_seed,
        target_sketch_dim=config.target_sketch_dim,
    )
    features = encode_trajectory(
        trajectory,
        pool=base_probe.pool,
        colors=base_probe.colors,
    )
    targets = frame_target_sketch(
        features,
        seed=config.target_sketch_seed,
        output_dim=config.target_sketch_dim,
    )

    deadline = None if config.hours <= 0 else time.monotonic() + config.hours * 3600.0
    completed_this_session = 0
    started = time.monotonic()
    interrupted = False

    try:
        for index, job in enumerate(schedule):
            path = result_path(results_dir, job)
            if path.exists():
                _load_result(path)
                continue
            if deadline is not None and time.monotonic() >= deadline:
                break

            job_started = time.monotonic()
            pconfig = probe_config(config, job)
            payload = run_probe(
                bundle,
                trajectory,
                variant=job.variant,
                config=pconfig,
                core=core,
                features=features,
                targets=targets,
            )
            payload["job_id"] = job.job_id
            payload["job_index"] = index
            payload["runtime_seconds"] = time.monotonic() - job_started
            payload["graph_artifacts"] = _graph_artifact_hashes(graph_root)
            _atomic_json(path, payload)
            completed_this_session += 1

            progress = {
                "schema": MARATHON_SCHEMA,
                "completed_total": len(list(results_dir.glob("*.json"))),
                "completed_this_session": completed_this_session,
                "configured_jobs": len(schedule),
                "last_job": job.job_id,
                "last_job_runtime_seconds": payload["runtime_seconds"],
                "session_elapsed_seconds": time.monotonic() - started,
            }
            _atomic_json(root / "progress.json", progress)
    except KeyboardInterrupt:
        interrupted = True

    aggregate = aggregate_results(
        root,
        config=config,
        source_run=source_run,
        graph_root=graph_root,
    )
    aggregate["session_interrupted"] = interrupted
    aggregate["session_elapsed_seconds"] = time.monotonic() - started
    aggregate["completed_this_session"] = completed_this_session
    _atomic_json(root / "aggregate.json", aggregate)
    _write_receipt(root)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable multi-hour FlyARC identical-trajectory topology-null marathon"
        )
    )
    parser.add_argument("graph")
    parser.add_argument("source_run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--projection-count", type=int, default=4)
    parser.add_argument("--rewires-per-projection", type=int, default=16)
    parser.add_argument("--randoms-per-projection", type=int, default=8)
    parser.add_argument("--base-projection-seed", type=int, default=1701)
    parser.add_argument("--base-topology-seed", type=int, default=2903)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--target-sketch-dim", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    args = parser.parse_args()

    config = MarathonConfig(
        hours=args.hours,
        projection_count=args.projection_count,
        rewires_per_projection=args.rewires_per_projection,
        randoms_per_projection=args.randoms_per_projection,
        base_projection_seed=args.base_projection_seed,
        base_topology_seed=args.base_topology_seed,
        max_nodes=args.max_nodes,
        target_sketch_dim=args.target_sketch_dim,
        ridge_alpha=args.ridge_alpha,
    )
    aggregate = run_marathon(
        args.graph,
        args.source_run,
        args.output,
        config=config,
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
