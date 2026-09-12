from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

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


def paired_spl_report(frame: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    pivot = frame.pivot(index="seed", columns="label", values="spl").dropna(subset=[a, b])
    mean, lo, hi = paired_bootstrap_delta(pivot[a].to_numpy(), pivot[b].to_numpy())
    return {"mean_delta": mean, "ci95_low": lo, "ci95_high": hi, "n": len(pivot)}


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
