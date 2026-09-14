from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark import load_benchmark
from .metrics import summarize_metrics
from .registry import load_registry
from .schema import SampleWindow
from .splits import validate_animal_disjoint_splits


def _load_samples(path: str | Path) -> list[SampleWindow]:
    payload = json.loads(Path(path).read_text())
    rows = payload["samples"] if isinstance(payload, dict) else payload
    return [SampleWindow.from_dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fly video-to-neural benchmark utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    registry = sub.add_parser("validate-registry")
    registry.add_argument("registry")

    benchmark = sub.add_parser("validate-benchmark")
    benchmark.add_argument("benchmark")

    split = sub.add_parser("validate-splits")
    split.add_argument("samples")
    split.add_argument("splits")

    score = sub.add_parser("score")
    score.add_argument("truth")
    score.add_argument("prediction")
    score.add_argument("--output")

    args = parser.parse_args()
    if args.command == "validate-registry":
        document = load_registry(args.registry)
        print(json.dumps({"status": "valid", "datasets": len(document["datasets"])}, sort_keys=True))
        return
    if args.command == "validate-benchmark":
        document = load_benchmark(args.benchmark)
        print(json.dumps({"status": "valid", "benchmark_id": document["benchmark_id"]}, sort_keys=True))
        return
    if args.command == "validate-splits":
        samples = _load_samples(args.samples)
        assignment = json.loads(Path(args.splits).read_text())["animal_to_split"]
        validate_animal_disjoint_splits(samples, assignment)
        print(json.dumps({"status": "valid", "samples": len(samples)}, sort_keys=True))
        return
    truth = np.load(args.truth)
    prediction = np.load(args.prediction)
    report = summarize_metrics(truth, prediction)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
