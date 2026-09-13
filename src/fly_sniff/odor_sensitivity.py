from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .odor_event_cli import _json_normalized, verify_motion_config_binding
from .odor_events import (
    OdorEventConfig,
    detect_motion_events,
    detect_odor_state_events,
    load_diagnostics,
)
from .odor_motion import OdorMotionConfig, load_analysis, load_odor_motion_config
from .recording import load_recording

PROTOCOL = "odor-threshold-sensitivity-v2"
SCHEMA_VERSION = 1


def sensitivity_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HysteresisSetting:
    label: str
    odor_on_threshold: float
    odor_off_threshold: float


@dataclass(frozen=True)
class SensitivitySpec:
    hysteresis: tuple[HysteresisSetting, ...]
    dwell_s: tuple[float, ...]
    timing_confidence: tuple[float, ...]

    @classmethod
    def from_document(
        cls,
        document: dict[str, Any],
        primary: OdorEventConfig,
    ) -> SensitivitySpec:
        raw = document.get("sensitivity_analysis")
        if not isinstance(raw, dict):
            raise TypeError("odor-motion config is missing sensitivity_analysis settings")
        if raw.get("status") != "secondary_robustness_only":
            raise ValueError("sensitivity analysis must remain secondary robustness only")
        if raw.get("may_select_or_redefine_primary_thresholds") is not False:
            raise ValueError("sensitivity grid must not be allowed to redefine primary thresholds")

        hysteresis_raw = raw.get("odor_hysteresis_grid")
        dwell_raw = raw.get("minimum_state_duration_s_grid")
        confidence_raw = raw.get("timing_confidence_grid")
        if not isinstance(hysteresis_raw, list) or not hysteresis_raw:
            raise TypeError("sensitivity odor_hysteresis_grid must be a nonempty list")
        if not isinstance(dwell_raw, list) or not dwell_raw:
            raise TypeError("sensitivity minimum_state_duration_s_grid must be a nonempty list")
        if not isinstance(confidence_raw, list) or not confidence_raw:
            raise TypeError("sensitivity timing_confidence_grid must be a nonempty list")

        hysteresis = tuple(
            HysteresisSetting(
                label=str(item["label"]),
                odor_on_threshold=float(item["odor_on_threshold"]),
                odor_off_threshold=float(item["odor_off_threshold"]),
            )
            for item in hysteresis_raw
        )
        dwell = tuple(float(value) for value in dwell_raw)
        confidence = tuple(float(value) for value in confidence_raw)
        spec = cls(
            hysteresis=hysteresis,
            dwell_s=dwell,
            timing_confidence=confidence,
        )
        return spec.validated(primary)

    def validated(self, primary: OdorEventConfig) -> SensitivitySpec:
        labels = [item.label for item in self.hysteresis]
        if len(set(labels)) != len(labels):
            raise ValueError("sensitivity hysteresis labels must be unique")
        if labels.count("primary") != 1:
            raise ValueError("sensitivity hysteresis grid requires exactly one primary row")
        for item in self.hysteresis:
            if not 0.0 <= item.odor_off_threshold < item.odor_on_threshold <= 1.0:
                raise ValueError("sensitivity hysteresis rows must satisfy 0 <= off < on <= 1")
        for name, values in (
            ("minimum_state_duration_s_grid", self.dwell_s),
            ("timing_confidence_grid", self.timing_confidence),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} values must be unique")
            if tuple(sorted(values)) != values:
                raise ValueError(f"{name} values must be strictly increasing")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.dwell_s):
            raise ValueError("sensitivity dwell values must be finite and > 0")
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.timing_confidence
        ):
            raise ValueError("sensitivity confidence values must be finite and in [0, 1]")

        primary_row = next(item for item in self.hysteresis if item.label == "primary")
        if not np.isclose(primary_row.odor_on_threshold, primary.odor_on_threshold):
            raise ValueError("primary odor-on threshold is missing from sensitivity grid")
        if not np.isclose(primary_row.odor_off_threshold, primary.odor_off_threshold):
            raise ValueError("primary odor-off threshold is missing from sensitivity grid")
        if not any(np.isclose(value, primary.minimum_state_duration_s) for value in self.dwell_s):
            raise ValueError("primary dwell time is missing from sensitivity grid")
        if not any(
            np.isclose(value, primary.high_confidence_threshold)
            for value in self.timing_confidence
        ):
            raise ValueError("primary timing confidence is missing from sensitivity grid")
        return self


def _agent(payload: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [item for item in payload.get("agents", []) if item.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one agent labelled {label!r}")
    return matches[0]


def _event_summary(
    events: list[dict[str, Any]],
    *,
    horizon_s: float,
) -> dict[str, Any]:
    losses = [event for event in events if event["kind"] == "loss"]
    reacquisitions = [event for event in events if event["kind"] == "reacquisition"]
    within_latencies: list[float] = []
    all_latencies = [
        float(event["latency_s"])
        for event in reacquisitions
        if event.get("latency_s") is not None
    ]
    for loss in losses:
        next_reacquisition = next(
            (
                event
                for event in reacquisitions
                if float(event["t"]) > float(loss["t"])
            ),
            None,
        )
        if next_reacquisition is None:
            continue
        latency = float(next_reacquisition["t"]) - float(loss["t"])
        if latency <= horizon_s:
            within_latencies.append(latency)

    return {
        "encounter_count": sum(event["kind"] == "encounter" for event in events),
        "loss_count": len(losses),
        "reacquisition_count": len(reacquisitions),
        "reacquisition_within_horizon_fraction": (
            float(len(within_latencies) / len(losses)) if losses else None
        ),
        "median_reacquisition_latency_all_s": (
            float(np.median(all_latencies)) if all_latencies else None
        ),
        "median_reacquisition_latency_within_horizon_s": (
            float(np.median(within_latencies)) if within_latencies else None
        ),
    }


def _range(values: list[float | int | None]) -> dict[str, float | int | None]:
    present = [value for value in values if value is not None]
    if not present:
        return {"min": None, "max": None}
    return {"min": min(present), "max": max(present)}


def _verify_diagnostics_binding(
    diagnostics_bundle: dict[str, Any],
    *,
    recording_sha: str,
    motion_sha: str,
    primary_event_config: OdorEventConfig,
) -> None:
    payload = diagnostics_bundle.get("diagnostics")
    if not isinstance(payload, dict):
        raise TypeError("invalid odor-event diagnostics bundle")
    if payload.get("source_recording_sha256") != recording_sha:
        raise ValueError("odor-event diagnostics do not belong to this recording")
    if payload.get("source_odor_motion_sha256") != motion_sha:
        raise ValueError("odor-event diagnostics do not belong to this odor-motion analysis")
    embedded = payload.get("config", {}).get("event_detection")
    if _json_normalized(embedded) != _json_normalized(asdict(primary_event_config)):
        raise ValueError("odor-event diagnostics do not match the primary event config")


def build_sensitivity_bundle(
    recording_bundle: dict[str, Any],
    motion_bundle: dict[str, Any],
    diagnostics_bundle: dict[str, Any],
    *,
    config_document: dict[str, Any],
    motion_config: OdorMotionConfig,
    primary_event_config: OdorEventConfig,
    spec: SensitivitySpec,
) -> dict[str, Any]:
    verify_motion_config_binding(motion_bundle, motion_config)
    recording = recording_bundle["recording"]
    motion = motion_bundle["analysis"]
    recording_sha = recording_bundle["recording_sha256"]
    motion_sha = motion_bundle["analysis_sha256"]
    diagnostics_sha = diagnostics_bundle["diagnostics_sha256"]
    if motion.get("source_recording_sha256") != recording_sha:
        raise ValueError("odor-motion analysis does not belong to this recording")
    _verify_diagnostics_binding(
        diagnostics_bundle,
        recording_sha=recording_sha,
        motion_sha=motion_sha,
        primary_event_config=primary_event_config,
    )
    dt = float(recording["dt"])
    agents: list[dict[str, Any]] = []

    for controller in recording["controllers"]:
        label = str(controller["label"])
        samples = _agent(motion, label)["samples"]
        primary_summary = _agent(diagnostics_bundle["diagnostics"], label)["summary"]

        odor_rows: list[dict[str, Any]] = []
        for hysteresis in spec.hysteresis:
            for dwell_s in spec.dwell_s:
                candidate = replace(
                    primary_event_config,
                    odor_on_threshold=hysteresis.odor_on_threshold,
                    odor_off_threshold=hysteresis.odor_off_threshold,
                    minimum_state_duration_s=dwell_s,
                ).validated()
                events = detect_odor_state_events(samples, dt=dt, config=candidate)
                summary = _event_summary(
                    events,
                    horizon_s=primary_event_config.reacquisition_horizon_s,
                )
                odor_rows.append(
                    {
                        "hysteresis_label": hysteresis.label,
                        "odor_on_threshold": hysteresis.odor_on_threshold,
                        "odor_off_threshold": hysteresis.odor_off_threshold,
                        "minimum_state_duration_s": dwell_s,
                        "is_primary": bool(
                            hysteresis.label == "primary"
                            and np.isclose(
                                dwell_s,
                                primary_event_config.minimum_state_duration_s,
                            )
                        ),
                        **summary,
                    }
                )

        timing_rows: list[dict[str, Any]] = []
        for threshold in spec.timing_confidence:
            candidate = replace(
                primary_event_config,
                high_confidence_threshold=threshold,
            ).validated()
            events = detect_motion_events(samples, dt=dt, config=candidate)
            timing_rows.append(
                {
                    "high_confidence_threshold": threshold,
                    "is_primary": bool(
                        np.isclose(
                            threshold,
                            primary_event_config.high_confidence_threshold,
                        )
                    ),
                    "timing_event_count": len(events),
                }
            )

        primary_odor = next(row for row in odor_rows if row["is_primary"])
        primary_timing = next(row for row in timing_rows if row["is_primary"])
        if primary_odor["loss_count"] != primary_summary["loss_count"]:
            raise ValueError("sensitivity primary odor row does not reproduce diagnostics loss count")
        if primary_odor["reacquisition_count"] != primary_summary["reacquisition_count"]:
            raise ValueError(
                "sensitivity primary odor row does not reproduce diagnostics reacquisition count"
            )
        if primary_timing["timing_event_count"] != primary_summary["timing_event_count"]:
            raise ValueError(
                "sensitivity primary timing row does not reproduce diagnostics timing-event count"
            )

        agents.append(
            {
                "label": label,
                "primary_summary": primary_summary,
                "odor_event_grid": odor_rows,
                "timing_event_grid": timing_rows,
                "robustness_ranges": {
                    "loss_count": _range([row["loss_count"] for row in odor_rows]),
                    "reacquisition_count": _range(
                        [row["reacquisition_count"] for row in odor_rows]
                    ),
                    "reacquisition_within_horizon_fraction": _range(
                        [row["reacquisition_within_horizon_fraction"] for row in odor_rows]
                    ),
                    "timing_event_count": _range(
                        [row["timing_event_count"] for row in timing_rows]
                    ),
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": config_document["sensitivity_analysis"]["status"],
        "source_recording_sha256": recording_sha,
        "source_odor_motion_sha256": motion_sha,
        "source_diagnostics_sha256": diagnostics_sha,
        "controller_access_in_v1": False,
        "may_select_or_redefine_primary_thresholds": False,
        "claim_boundary": (
            "This is a secondary robustness sweep over nearby engineering thresholds. "
            "It cannot select, optimize, or redefine the frozen primary thresholds and cannot "
            "be used as controller input or as evidence of a neural threshold."
        ),
        "grid": config_document["sensitivity_analysis"],
        "fixed_across_grid": {
            "reacquisition_horizon_s": primary_event_config.reacquisition_horizon_s,
            "turn_response_horizon_s": primary_event_config.turn_response_horizon_s,
            "timing_estimator": asdict(motion_config),
        },
        "agents": agents,
    }
    return {
        "sensitivity": payload,
        "sensitivity_sha256": sensitivity_sha256(payload),
    }


def write_sensitivity(path: str | Path, bundle: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def load_sensitivity(path: str | Path) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text())
    payload = bundle.get("sensitivity")
    expected = bundle.get("sensitivity_sha256")
    if not isinstance(payload, dict) or not isinstance(expected, str):
        raise TypeError("invalid odor-threshold sensitivity bundle")
    actual = sensitivity_sha256(payload)
    if actual != expected:
        raise ValueError(
            f"odor-threshold sensitivity SHA-256 mismatch: expected {expected}, got {actual}"
        )
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported sensitivity protocol {payload.get('protocol')!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported sensitivity schema {payload.get('schema_version')!r}")
    if payload.get("may_select_or_redefine_primary_thresholds") is not False:
        raise ValueError("sensitivity receipt illegally permits primary-threshold selection")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen secondary robustness sweep over odor-event and timing-event "
            "thresholds without selecting a best setting"
        )
    )
    parser.add_argument("recording")
    parser.add_argument("analysis")
    parser.add_argument("diagnostics")
    parser.add_argument("--config")
    parser.add_argument(
        "--output",
        default="artifacts/showcase/odor-sensitivity-v2.json",
    )
    args = parser.parse_args()

    recording_bundle = load_recording(args.recording)
    motion_bundle = load_analysis(args.analysis)
    diagnostics_bundle = load_diagnostics(args.diagnostics)
    config_document, motion_config = load_odor_motion_config(args.config)
    primary_event_config = OdorEventConfig.from_document(config_document)
    spec = SensitivitySpec.from_document(config_document, primary_event_config)
    bundle = build_sensitivity_bundle(
        recording_bundle,
        motion_bundle,
        diagnostics_bundle,
        config_document=config_document,
        motion_config=motion_config,
        primary_event_config=primary_event_config,
        spec=spec,
    )
    output = write_sensitivity(args.output, bundle)
    print(output)
    print(bundle["sensitivity_sha256"])


if __name__ == "__main__":
    main()
