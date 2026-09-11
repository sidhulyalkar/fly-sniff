from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .choice import benchmark_choice
from .controllers import BilateralProxyController, RandomWalkController
from .party_social import PROXY_CLAIM_LABEL, render_party_proxy


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_demo_manifest(
    *,
    seed: int,
    trials: int,
    seconds: int,
    fps: int,
    proxy_report: dict[str, Any],
    random_report: dict[str, Any],
    clip_name: str,
) -> dict[str, Any]:
    """Build a machine-readable contract for a non-claim-bearing social demo."""
    return {
        "schema_version": 1,
        "demo": "who-farted-development-proxy",
        "claim_status": "development-only",
        "claim_label": PROXY_CLAIM_LABEL,
        "scientific_scope": (
            "Visualization and plumbing demo only. This does not report a qualified MaleCNS result. "
            "The scene tests simulated odor-source localization, not human identity chemistry."
        ),
        "git_sha": os.getenv("GITHUB_SHA"),
        "seed": seed,
        "choice_trials_per_controller": trials,
        "render_seconds": seconds,
        "render_fps": fps,
        "artifacts": {
            "clip": clip_name,
            "proxy_choice": "choice-proxy.json",
            "random_choice": "choice-random.json",
        },
        "controllers": {
            "proxy": proxy_report["controller"],
            "random": random_report["controller"],
        },
        "choice_summary": {
            "chance_accuracy": 0.5,
            "proxy_accuracy": proxy_report["accuracy"],
            "proxy_commitment_rate": proxy_report["commitment_rate"],
            "random_accuracy": random_report["accuracy"],
            "random_commitment_rate": random_report["commitment_rate"],
        },
    }


def build_demo_pack(
    output_dir: str | Path,
    *,
    seed: int = 13013,
    trials: int = 100,
    seconds: int = 8,
    fps: int = 12,
    clip_format: str = "gif",
) -> Path:
    """Create the smallest auditable package needed to demo fly-sniff today."""
    if clip_format not in {"gif", "mp4"}:
        raise ValueError("clip_format must be 'gif' or 'mp4'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_report = benchmark_choice(
        BilateralProxyController(),
        trials=trials,
        seed=seed,
        claim_status="development-only",
    )
    random_report = benchmark_choice(
        RandomWalkController(),
        trials=trials,
        seed=seed,
        claim_status="development-only",
    )
    proxy_path = output_dir / "choice-proxy.json"
    random_path = output_dir / "choice-random.json"
    _write_json(proxy_path, proxy_report)
    _write_json(random_path, random_report)

    clip_name = f"who-farted-proxy.{clip_format}"
    render_party_proxy(output_dir / clip_name, seed=seed, seconds=seconds, fps=fps)

    manifest = make_demo_manifest(
        seed=seed,
        trials=trials,
        seconds=seconds,
        fps=fps,
        proxy_report=proxy_report,
        random_report=random_report,
        clip_name=clip_name,
    )
    manifest_path = output_dir / "demo-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the auditable fly-sniff development demo pack"
    )
    parser.add_argument("--output-dir", default="artifacts/demo")
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    args = parser.parse_args()
    print(
        build_demo_pack(
            args.output_dir,
            seed=args.seed,
            trials=args.trials,
            seconds=args.seconds,
            fps=args.fps,
            clip_format=args.format,
        )
    )
