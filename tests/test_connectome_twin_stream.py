from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.connectome_twin_stream import STREAM_PROTOCOL, build_stream
from fly_sniff.recording import build_recording, recording_sha256, write_recording


def test_stream_keeps_behavior_and_mechanism_clocks_separate(tmp_path: Path) -> None:
    recording_path = tmp_path / "recording.json"
    write_recording(recording_path, build_recording(sim_seconds=0.1, plume_points=220))
    mechanism_path = tmp_path / "mechanism.json"
    mechanism_path.write_text(
        json.dumps(
            {
                "protocol": "E002e-pfl3-descending-steering-v1",
                "passed": True,
                "primary_structural_threshold": 5,
                "threshold_reports": {"5": {"phase_reports": {}}},
            }
        )
    )

    stream = build_stream(recording_path, mechanism_report_path=mechanism_path)
    assert stream["protocol"] == STREAM_PROTOCOL
    assert stream["timelines"]["behavior"]["clock"] == "recorded-behavior"
    assert stream["timelines"]["mechanism"]["clock"] == "independent-mechanism-probe"
    assert stream["timelines"]["mechanism"]["synchronized_with_behavior"] is False
    assert "must not be presented as synchronized" in stream["synchronization_contract"]


def test_behavior_stream_excludes_source_coordinates_from_agent_state(tmp_path: Path) -> None:
    recording_path = tmp_path / "recording.json"
    write_recording(recording_path, build_recording(sim_seconds=0.1, plume_points=220))
    stream = build_stream(recording_path)
    first = stream["timelines"]["behavior"]["frames"][0]["agents"][0]
    assert "distance_to_source" not in first
    assert "source_x" not in first
    assert set(first["sensors"]) == {"odor_left", "odor_right", "wind_body"}


def test_stream_preserves_unknown_plume_completeness_for_legacy_recording(tmp_path: Path) -> None:
    recording_path = tmp_path / "legacy-recording.json"
    bundle = build_recording(sim_seconds=0.1, plume_points=220)
    recording = bundle["recording"]
    for frame in recording["frames"]:
        frame.pop("plume_snapshot", None)
    bundle["recording_sha256"] = recording_sha256(recording)
    write_recording(recording_path, bundle)

    stream = build_stream(recording_path)
    first = stream["timelines"]["behavior"]["frames"][0]
    assert first["plume_snapshot_complete"] is None
    assert first["plume_snapshot_metadata_available"] is False
    assert first["plume_components"]


def test_stream_reports_modern_plume_completeness_metadata(tmp_path: Path) -> None:
    recording_path = tmp_path / "recording.json"
    write_recording(recording_path, build_recording(sim_seconds=0.1, plume_points=220))

    stream = build_stream(recording_path)
    first = stream["timelines"]["behavior"]["frames"][0]
    assert isinstance(first["plume_snapshot_complete"], bool)
    assert first["plume_snapshot_metadata_available"] is True
