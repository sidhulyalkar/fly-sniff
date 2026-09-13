from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .odor_events import (
    OdorEventConfig,
    build_diagnostics_bundle,
    write_diagnostics,
)
from .odor_motion import OdorMotionConfig, load_analysis, load_odor_motion_config
from .recording import load_recording


def _json_normalized(value: Any) -> Any:
    """Normalize tuples and other JSON-compatible containers for exact comparison."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def verify_motion_config_binding(
    motion_bundle: dict[str, Any],
    motion_config: OdorMotionConfig,
) -> None:
    analysis = motion_bundle.get("analysis")
    if not isinstance(analysis, dict):
        raise TypeError("invalid odor-motion analysis bundle")
    embedded = analysis.get("estimator")
    expected = asdict(motion_config)
    if _json_normalized(embedded) != _json_normalized(expected):
        raise ValueError(
            "odor-event config does not match the estimator settings embedded in the "
            "odor-motion sidecar; rerun with the exact timing config that produced the sidecar"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive hysteretic odor events, behavioral responses, and temporal "
            "counterfactuals from a hash-bound fly-sniff replay"
        )
    )
    parser.add_argument("recording")
    parser.add_argument("analysis")
    parser.add_argument("--config")
    parser.add_argument(
        "--output",
        default="artifacts/showcase/odor-events-v2.json",
    )
    args = parser.parse_args()

    recording_bundle = load_recording(args.recording)
    motion_bundle = load_analysis(args.analysis)
    config_document, motion_config = load_odor_motion_config(args.config)
    verify_motion_config_binding(motion_bundle, motion_config)
    event_config = OdorEventConfig.from_document(config_document)
    diagnostics = build_diagnostics_bundle(
        recording_bundle,
        motion_bundle,
        motion_config=motion_config,
        event_config=event_config,
    )
    output = write_diagnostics(Path(args.output), diagnostics)
    print(output)
    print(diagnostics["diagnostics_sha256"])


if __name__ == "__main__":
    main()
