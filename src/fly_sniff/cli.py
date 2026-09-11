from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controllers import BilateralProxyController, CastSurgeController, RandomWalkController
from .evaluate import evaluate, summarize, write_receipt
from .social import render_social_video


def demo_main() -> None:
    parser = argparse.ArgumentParser(description="Render FlyBrain Plume Hunt development visualization")
    parser.add_argument("--output", default="artifacts/fly-sniff-proxy.mp4")
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--seconds", type=int, default=24)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    path = render_social_video(args.output, seed=args.seed, seconds=args.seconds, fps=args.fps)
    print(path)


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(description="Run the development plume benchmark")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--output", default="results/dev-v0")
    args = parser.parse_args()
    seeds = [args.seed + i * 9973 for i in range(args.episodes)]
    factories = {
        "proxy": BilateralProxyController,
        "cast-surge": CastSurgeController,
        "random": RandomWalkController,
    }
    frame = evaluate(factories, seeds)
    summary = summarize(frame)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "episodes.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    manifest = {"mode": "development", "seed_base": args.seed, "episodes": args.episodes, "controllers": list(factories)}
    write_receipt(out / "receipt.json", manifest, {"rows": summary.to_dict(orient="records")})
    print(summary.to_string(index=False))
