from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ANNOTATIONS_PATH = Path("data/raw/body-annotations-male-cns-v1.0.feather")
WEIGHTS_PATH = Path("data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather")


def diagnose(
    *,
    root: str | Path = ".",
    environ: Mapping[str, str] | None = None,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    root = Path(root)
    env = os.environ if environ is None else environ
    ffmpeg = shutil.which("ffmpeg") if ffmpeg_path is None else ffmpeg_path
    annotations = root / ANNOTATIONS_PATH
    weights = root / WEIGHTS_PATH
    python_ok = sys.version_info >= (3, 11)
    token_present = bool(env.get("NEUPRINT_TOKEN"))
    return {
        "python": {
            "version": ".".join(str(x) for x in sys.version_info[:3]),
            "ok": python_ok,
            "required": ">=3.11",
        },
        "ffmpeg": {"path": ffmpeg, "ok": bool(ffmpeg)},
        "neuprint_token": {"present": token_present},
        "data": {
            "annotations": {"path": str(annotations), "present": annotations.exists()},
            "weights": {"path": str(weights), "present": weights.exists()},
        },
        "ready": {
            "gif_demo": python_ok,
            "mp4_showcase": python_ok and bool(ffmpeg),
            "live_malecns": python_ok and token_present,
            "offline_full_graph_trace": python_ok and annotations.exists() and weights.exists(),
        },
    }


def _print_human(report: dict[str, Any]) -> None:
    yes = "READY"
    no = "MISSING"
    print(f"Python {report['python']['version']} ({yes if report['python']['ok'] else no}; requires >=3.11)")
    print(f"ffmpeg: {report['ffmpeg']['path'] or no}")
    print(f"NEUPRINT_TOKEN: {yes if report['neuprint_token']['present'] else no}")
    print(
        "annotations: "
        f"{yes if report['data']['annotations']['present'] else no} "
        f"({report['data']['annotations']['path']})"
    )
    print(
        "full weights: "
        f"{yes if report['data']['weights']['present'] else no} "
        f"({report['data']['weights']['path']})"
    )
    print("\nCapabilities")
    for name, ready in report["ready"].items():
        print(f"  {name}: {yes if ready else no}")
    if not report["ready"]["mp4_showcase"]:
        print("\nInstall ffmpeg for MP4 output, or render GIF immediately.")
    if not report["data"]["annotations"]["present"]:
        print("Run: fly-sniff-download-annotations")
    if not report["neuprint_token"]["present"]:
        print("Set NEUPRINT_TOKEN to enable live male-cns:v1.0 extraction.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check fly-sniff demo and MaleCNS prerequisites")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    report = diagnose()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
