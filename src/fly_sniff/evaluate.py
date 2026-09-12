from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .controllers import Controller
from .env import FlySniffEnv
from .metrics import EpisodeMetrics, paired_bootstrap_delta, shortest_path_to_goal_region, spl


def run_episode(
    controller: Controller,
    seed: int,
    arena: ArenaConfig | None = None,
    plume: PlumeConfig | None = None,
    sensors: SensorConfig | None = None,
) -> EpisodeMetrics:
    env = FlySniffEnv(seed=seed, arena=arena, plume=plume, sensors=sensors)
    controller.reset(seed + 101)
    initial_center_distance = env.distance_to_source
    shortest_path = shortest_path_to_goal_region(
        initial_center_distance,
        env.arena.source_radius,
    )
    obs = env.observe()
    done = False
    while not done:
        action = controller.act(obs)
        obs, done = env.step(action.turn, action.speed)
    a = env.agent
    return EpisodeMetrics(
        seed=seed,
        controller=controller.name,
        success=a.found,
        steps=a.steps,
        elapsed_s=a.steps * env.arena.dt,
        path_length=a.path_length,
        shortest_path=shortest_path,
        spl=spl(a.found, shortest_path, a.path_length),
        final_distance=env.distance_to_source,
    )


def evaluate(
    factories: dict[str, Callable[[], Controller]],
    seeds: list[int],
    arena: ArenaConfig | None = None,
    plume: PlumeConfig | None = None,
    sensors: SensorConfig | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in seeds:
        for label, factory in factories.items():
            metric = run_episode(
                factory(),
                seed=seed,
                arena=arena,
                plume=plume,
                sensors=sensors,
            )
            row = metric.as_dict()
            row["label"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("label", as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_spl=("spl", "mean"),
            mean_path=("path_length", "mean"),
        )
        .sort_values("mean_spl", ascending=False)
    )


def paired_metric_report(
    frame: pd.DataFrame,
    a: str,
    b: str,
    metric: str,
    *,
    bootstrap_seed: int = 13013,
    n_boot: int = 10000,
) -> dict[str, float | int | str]:
    if metric not in frame.columns:
        raise ValueError(f"metric {metric!r} is not present in evaluation frame")
    pivot = frame.pivot(index="seed", columns="label", values=metric).dropna(subset=[a, b])
    mean, lo, hi = paired_bootstrap_delta(
        pivot[a].to_numpy(),
        pivot[b].to_numpy(),
        seed=bootstrap_seed,
        n_boot=n_boot,
    )
    return {
        "metric": metric,
        "mean_delta": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "n": int(len(pivot)),
    }


def paired_spl_report(frame: pd.DataFrame, a: str, b: str) -> dict[str, float | int | str]:
    return paired_metric_report(frame, a, b, "spl")


def paired_success_report(
    frame: pd.DataFrame,
    a: str,
    b: str,
    *,
    bootstrap_seed: int = 13013,
    n_boot: int = 10000,
) -> dict[str, float | int | str]:
    pivot = frame.pivot(index="seed", columns="label", values="success").dropna(subset=[a, b])
    a_success = pivot[a].astype(bool)
    b_success = pivot[b].astype(bool)
    a_only = int((a_success & ~b_success).sum())
    b_only = int((~a_success & b_success).sum())
    discordant = a_only + b_only
    exact_p = float(binomtest(a_only, discordant, p=0.5).pvalue) if discordant else 1.0
    mean, lo, hi = paired_bootstrap_delta(
        a_success.astype(float).to_numpy(),
        b_success.astype(float).to_numpy(),
        seed=bootstrap_seed,
        n_boot=n_boot,
    )
    return {
        "metric": "success",
        "a_success_rate": float(a_success.mean()),
        "b_success_rate": float(b_success.mean()),
        "mean_delta": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "a_only_successes": a_only,
        "b_only_successes": b_only,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": exact_p,
        "n": int(len(pivot)),
    }


def manifest_digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def sealed_manifest_digest(manifest: dict) -> str:
    expected = manifest.get("manifest_sha256")
    if expected:
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256")
        actual = manifest_digest(unsigned)
        if actual != expected:
            raise ValueError(
                f"manifest self-hash mismatch: expected {expected}, recomputed {actual}"
            )
        return str(expected)
    return manifest_digest(manifest)


def write_receipt(path: str | Path, manifest: dict, summary: dict) -> None:
    receipt = {
        "manifest_sha256": sealed_manifest_digest(manifest),
        "manifest": manifest,
        "summary": summary,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
