from __future__ import annotations

from fly_sniff.odor_events import (
    OdorEventConfig,
    build_diagnostics_bundle,
    write_diagnostics,
)
from fly_sniff.odor_motion import (
    analyze_recording_bundle,
    load_odor_motion_config,
    write_analysis,
)
from fly_sniff.odor_motion_showcase import (
    load_replay_context,
    render_temporal_showcase_frame,
)
from fly_sniff.recording import build_recording, write_recording


def test_temporal_social_frame_uses_bound_recording_analysis_and_events(tmp_path):
    source = build_recording(seed=49, sim_seconds=0.50, plume_points=16)
    config_document, motion_config = load_odor_motion_config()
    motion = analyze_recording_bundle(
        source,
        config_document=config_document,
        config=motion_config,
    )
    diagnostics = build_diagnostics_bundle(
        source,
        motion,
        motion_config=motion_config,
        event_config=OdorEventConfig.from_document(config_document),
    )

    recording_path = write_recording(tmp_path / "episode.json", source)
    motion_path = write_analysis(tmp_path / "motion.json", motion)
    event_path = write_diagnostics(tmp_path / "events.json", diagnostics)
    context = load_replay_context(recording_path, motion_path, event_path)

    assert context.label == source["recording"]["controllers"][0]["label"]
    assert len(context.motion_samples) == len(source["recording"]["frames"])

    output = render_temporal_showcase_frame(
        recording_path,
        motion_path,
        event_path,
        tmp_path / "showcase-frame.png",
        frame_index=0,
    )
    assert output.exists()
    assert output.stat().st_size > 0
