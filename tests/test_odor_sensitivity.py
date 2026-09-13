from __future__ import annotations

import copy
import json

import pytest

from fly_sniff.odor_events import OdorEventConfig, build_diagnostics_bundle
from fly_sniff.odor_motion import analyze_recording_bundle, load_odor_motion_config
from fly_sniff.odor_sensitivity import (
    SensitivitySpec,
    build_sensitivity_bundle,
    load_sensitivity,
    write_sensitivity,
)
from fly_sniff.recording import build_recording


def _build_all(seed: int = 51):
    source = build_recording(seed=seed, sim_seconds=0.60, plume_points=16)
    document, motion_config = load_odor_motion_config()
    primary = OdorEventConfig.from_document(document)
    motion = analyze_recording_bundle(
        source,
        config_document=document,
        config=motion_config,
    )
    diagnostics = build_diagnostics_bundle(
        source,
        motion,
        motion_config=motion_config,
        event_config=primary,
    )
    spec = SensitivitySpec.from_document(document, primary)
    sensitivity = build_sensitivity_bundle(
        source,
        motion,
        diagnostics,
        config_document=document,
        motion_config=motion_config,
        primary_event_config=primary,
        spec=spec,
    )
    return source, motion, diagnostics, sensitivity, document, primary


def test_sensitivity_grid_contains_exact_primary_and_cannot_select_thresholds():
    _, _, diagnostics, sensitivity, _, _ = _build_all()
    payload = sensitivity["sensitivity"]

    assert payload["status"] == "secondary_robustness_only"
    assert payload["may_select_or_redefine_primary_thresholds"] is False
    assert payload["controller_access_in_v1"] is False

    for agent in payload["agents"]:
        primary_odor = [row for row in agent["odor_event_grid"] if row["is_primary"]]
        primary_timing = [row for row in agent["timing_event_grid"] if row["is_primary"]]
        assert len(primary_odor) == 1
        assert len(primary_timing) == 1
        source_summary = next(
            item["summary"]
            for item in diagnostics["diagnostics"]["agents"]
            if item["label"] == agent["label"]
        )
        assert primary_odor[0]["loss_count"] == source_summary["loss_count"]
        assert primary_odor[0]["reacquisition_count"] == source_summary["reacquisition_count"]

        primary_fraction = primary_odor[0]["reacquisition_within_horizon_fraction"]
        source_fraction = source_summary["reacquisition_within_horizon_fraction"]
        if source_fraction is None:
            assert primary_fraction is None
        else:
            assert primary_fraction == pytest.approx(source_fraction)

        primary_latency = primary_odor[0]["median_reacquisition_latency_within_horizon_s"]
        source_latency = source_summary["median_reacquisition_latency_s"]
        if source_latency is None:
            assert primary_latency is None
        else:
            assert primary_latency == pytest.approx(source_latency)
        assert primary_timing[0]["timing_event_count"] == source_summary["timing_event_count"]


def test_sensitivity_grid_is_fixed_instead_of_generated_from_performance():
    _, _, _, _, document, primary = _build_all(seed=52)
    spec = SensitivitySpec.from_document(document, primary)

    assert [item.label for item in spec.hysteresis] == ["lower", "primary", "higher"]
    assert spec.dwell_s == (0.05, 0.10, 0.20)
    assert spec.timing_confidence == (0.20, 0.35, 0.50)

    bad = copy.deepcopy(document)
    bad["sensitivity_analysis"]["may_select_or_redefine_primary_thresholds"] = True
    with pytest.raises(ValueError, match="must not be allowed"):
        SensitivitySpec.from_document(bad, primary)


def test_sensitivity_receipt_is_hash_bound_and_rejects_tampering(tmp_path):
    source, motion, diagnostics, sensitivity, _, _ = _build_all(seed=53)
    payload = sensitivity["sensitivity"]
    assert payload["source_recording_sha256"] == source["recording_sha256"]
    assert payload["source_odor_motion_sha256"] == motion["analysis_sha256"]
    assert payload["source_diagnostics_sha256"] == diagnostics["diagnostics_sha256"]

    path = write_sensitivity(tmp_path / "sensitivity.json", sensitivity)
    loaded = load_sensitivity(path)
    assert loaded["sensitivity_sha256"] == sensitivity["sensitivity_sha256"]

    raw = json.loads(path.read_text())
    raw["sensitivity"]["agents"][0]["odor_event_grid"][0]["loss_count"] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="sensitivity SHA-256 mismatch"):
        load_sensitivity(path)


def test_sensitivity_receipt_has_no_best_or_recommended_threshold_fields():
    _, _, _, sensitivity, _, _ = _build_all(seed=54)
    text = json.dumps(sensitivity["sensitivity"], sort_keys=True).lower()

    assert '"best"' not in text
    assert '"recommended"' not in text
    assert '"selected"' not in text
