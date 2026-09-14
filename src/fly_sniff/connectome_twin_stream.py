from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .recording import load_recording

STREAM_PROTOCOL = "fly-sniff-connectome-twin-stream-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_stream(
    recording_path: str | Path,
    *,
    connectome_assets_path: str | Path | None = None,
    mechanism_report_path: str | Path | None = None,
) -> dict[str, Any]:
    bundle = load_recording(recording_path)
    recording = bundle["recording"]
    arena = recording["arena"]
    behavior_frames: list[dict[str, Any]] = []
    for frame in recording["frames"]:
        agents = []
        for agent in frame["agents"]:
            obs = agent["observation"]
            agents.append(
                {
                    "label": agent["label"],
                    "position": [float(agent["x"]), float(agent["y"])],
                    "heading_rad": float(agent["heading"]),
                    "found": bool(agent["found"]),
                    "sensors": {
                        "odor_left": float(obs["left_odor"]),
                        "odor_right": float(obs["right_odor"]),
                        "wind_body": [float(obs["wind_x_body"]), float(obs["wind_y_body"])],
                    },
                    "command": {
                        "turn": float(agent["action"]["turn"]),
                        "speed": float(agent["action"]["speed"]),
                    },
                }
            )
        behavior_frames.append(
            {
                "t_s": float(frame["t"]),
                "plume_t_s": float(frame["plume_t"]),
                "plume_snapshot_complete": bool(frame["plume_snapshot"]["complete"]),
                "plume_components": frame["plume"],
                "agents": agents,
            }
        )

    assets: dict[str, Any] = {}
    if connectome_assets_path is not None:
        asset_path = Path(connectome_assets_path)
        if not asset_path.exists():
            raise FileNotFoundError(asset_path)
        assets["connectome"] = {
            "path": str(asset_path),
            "sha256": _sha256(asset_path),
            "evidence_class": "MEASURED_CONNECTIVITY",
        }

    mechanism: dict[str, Any] | None = None
    if mechanism_report_path is not None:
        path = Path(mechanism_report_path)
        report = json.loads(path.read_text())
        mechanism = {
            "clock": "independent-mechanism-probe",
            "synchronized_with_behavior": False,
            "source_path": str(path),
            "source_sha256": _sha256(path),
            "protocol": report.get("protocol"),
            "passed": bool(report.get("passed", False)),
            "primary_structural_threshold": report.get("primary_structural_threshold"),
            "threshold_reports": report.get("threshold_reports"),
            "evidence_class": "MODELED_ACTIVITY",
        }

    return {
        "protocol": STREAM_PROTOCOL,
        "dataset": "male-cns:v1.0",
        "evidence_classes": {
            "MEASURED_CONNECTIVITY": "dataset-derived graph/morphology only",
            "MODELED_ACTIVITY": "explicit model state; never measured firing",
            "BEHAVIORAL_STATE": "body, sensor, plume and command replay",
            "VIEWER_ONLY_WORLD": "world truth available to the renderer but forbidden to the controller",
        },
        "assets": assets,
        "timelines": {
            "behavior": {
                "clock": "recorded-behavior",
                "recording_sha256": bundle["recording_sha256"],
                "claim_boundary": recording["claim_boundary"],
                "world": {
                    "width": float(arena["width"]),
                    "height": float(arena["height"]),
                    "source_for_viewer_only": [
                        float(arena["source_x"]),
                        float(arena["source_y"]),
                    ],
                    "source_radius": float(arena["source_radius"]),
                    "evidence_class": "VIEWER_ONLY_WORLD",
                },
                "frames": behavior_frames,
                "evidence_class": "BEHAVIORAL_STATE",
            },
            "mechanism": mechanism,
        },
        "synchronization_contract": (
            "Timelines may be rendered beside each other but must not be presented as synchronized unless a future "
            "closed-loop recording explicitly records neural state and body state on one shared clock."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export evidence-aware replay data for the connectome twin")
    parser.add_argument("recording")
    parser.add_argument("--connectome-assets")
    parser.add_argument("--mechanism-report")
    parser.add_argument("--output", default="artifacts/connectome-twin/replay-stream-v1.json")
    args = parser.parse_args()

    payload = build_stream(
        args.recording,
        connectome_assets_path=args.connectome_assets,
        mechanism_report_path=args.mechanism_report,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
