from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .alignment import materialize_session_windows
from .benchmark import load_benchmark
from .mc2p import build_manifest
from .metrics import summarize_metrics
from .registry import load_registry
from .schema import SampleWindow
from .split_lock import build_split_lock, load_window_manifests
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

    mc2p = sub.add_parser("inspect-mc2p")
    mc2p.add_argument("root")
    mc2p.add_argument("--output")

    windows = sub.add_parser("make-mc2p-windows")
    windows.add_argument("session")
    windows.add_argument("alignment")
    windows.add_argument("--output", required=True)

    lock = sub.add_parser("make-split-lock")
    lock.add_argument("windows", nargs="+")
    lock.add_argument("--output", required=True)

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
    if args.command == "inspect-mc2p":
        report = build_manifest(args.root)
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(text)
        print(text, end="")
        return
    if args.command == "make-mc2p-windows":
        report = materialize_session_windows(args.session, args.alignment)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "valid", "windows": report["window_count"]}, sort_keys=True))
        return
    if args.command == "make-split-lock":
        samples = load_window_manifests(args.windows)
        report = build_split_lock(samples)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "locked", "sha256": report["split_lock_sha256"]}, sort_keys=True))
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
