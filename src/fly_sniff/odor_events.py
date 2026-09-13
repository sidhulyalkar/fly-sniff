from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .odor_motion import (
    BilateralOdorMotionEstimator,
    OdorMotionConfig,
    load_analysis,
    load_odor_motion_config,
)
from .recording import load_recording

PROTOCOL = "odor-event-diagnostics-v2"
SCHEMA_VERSION = 1


def diagnostics_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OdorEventConfig:
    odor_on_threshold: float = 0.16
    odor_off_threshold: float = 0.08
    minimum_state_duration_s: float = 0.10
    reacquisition_horizon_s: float = 2.0
    turn_response_horizon_s: float = 0.50
    high_confidence_threshold: float = 0.35
    high_confidence_refractory_s: float = 0.20
    temporal_shift_s: float = 0.75

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> OdorEventConfig:
        event = document.get("event_detection")
        counterfactual = document.get("counterfactuals")
        if not isinstance(event, dict):
            raise TypeError("odor-motion config is missing event_detection settings")
        if not isinstance(counterfactual, dict):
            raise TypeError("odor-motion config is missing counterfactual settings")
        return cls(
            odor_on_threshold=float(event["odor_on_threshold"]),
            odor_off_threshold=float(event["odor_off_threshold"]),
            minimum_state_duration_s=float(event["minimum_state_duration_s"]),
            reacquisition_horizon_s=float(event["reacquisition_horizon_s"]),
            turn_response_horizon_s=float(event["turn_response_horizon_s"]),
            high_confidence_threshold=float(event["high_confidence_threshold"]),
            high_confidence_refractory_s=float(event["high_confidence_refractory_s"]),
            temporal_shift_s=float(counterfactual["right_channel_shift_s"]),
        ).validated()

    def validated(self) -> OdorEventConfig:
        if not 0.0 <= self.odor_off_threshold < self.odor_on_threshold <= 1.0:
            raise ValueError("odor thresholds must satisfy 0 <= off < on <= 1")
        for name in (
            "minimum_state_duration_s",
            "reacquisition_horizon_s",
            "turn_response_horizon_s",
            "high_confidence_refractory_s",
            "temporal_shift_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if (
            not np.isfinite(self.high_confidence_threshold)
            or not 0.0 <= self.high_confidence_threshold <= 1.0
        ):
            raise ValueError("high_confidence_threshold must be finite and in [0, 1]")
        return self


def _motion_agent(payload: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [agent for agent in payload.get("agents", []) if agent.get("label") == label]
    if len(matches) != 1:
        raise ValueError(
            f"odor-motion sidecar must contain exactly one agent labelled {label!r}"
        )
    return matches[0]


def _recording_agent(frame: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [agent for agent in frame.get("agents", []) if agent.get("label") == label]
    if len(matches) != 1:
        raise ValueError(
            f"recording frame must contain exactly one agent labelled {label!r}"
        )
    return matches[0]


def _angle_delta(first: float, second: float) -> float:
    return float(np.arctan2(np.sin(second - first), np.cos(second - first)))


def _window_end(index: int, *, steps: int, length: int) -> int:
    return min(length - 1, index + max(1, steps))


def detect_odor_state_events(
    samples: list[dict[str, Any]],
    *,
    dt: float,
    config: OdorEventConfig,
) -> list[dict[str, Any]]:
    if not samples:
        return []
    min_steps = max(1, round(config.minimum_state_duration_s / dt))
    present = False
    candidate_start: int | None = None
    candidate_count = 0
    seen_encounter = False
    last_loss: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        left = float(sample["left_response"])
        right = float(sample["right_response"])
        strength = max(left, right)
        qualifies = (
            strength <= config.odor_off_threshold
            if present
            else strength >= config.odor_on_threshold
        )
        if qualifies:
            if candidate_start is None:
                candidate_start = index
                candidate_count = 1
            else:
                candidate_count += 1
        else:
            candidate_start = None
            candidate_count = 0

        if candidate_count < min_steps or candidate_start is None:
            continue

        event_index = candidate_start
        event_sample = samples[event_index]
        event_strength = max(
            float(event_sample["left_response"]),
            float(event_sample["right_response"]),
        )
        if present:
            event = {
                "kind": "loss",
                "index": int(event_index),
                "step": int(event_sample["step"]),
                "t": float(event_sample["t"]),
                "odor_strength": float(event_strength),
            }
            events.append(event)
            last_loss = event
            present = False
        else:
            kind = "reacquisition" if seen_encounter else "encounter"
            event = {
                "kind": kind,
                "index": int(event_index),
                "step": int(event_sample["step"]),
                "t": float(event_sample["t"]),
                "odor_strength": float(event_strength),
            }
            if kind == "reacquisition" and last_loss is not None:
                event["loss_step"] = int(last_loss["step"])
                event["latency_s"] = float(event["t"] - float(last_loss["t"]))
            events.append(event)
            seen_encounter = True
            present = True
        candidate_start = None
        candidate_count = 0

    return events


def detect_motion_events(
    samples: list[dict[str, Any]],
    *,
    dt: float,
    config: OdorEventConfig,
) -> list[dict[str, Any]]:
    refractory_steps = max(1, round(config.high_confidence_refractory_s / dt))
    last_event_index = -refractory_steps
    previous_direction: str | None = None
    previous_qualified = False
    events: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        direction = str(sample["direction"])
        qualified = (
            direction in {"left_to_right", "right_to_left"}
            and float(sample["confidence"]) >= config.high_confidence_threshold
        )
        is_onset = qualified and (
            not previous_qualified or direction != previous_direction
        )
        refractory_ok = index - last_event_index >= refractory_steps
        if is_onset and refractory_ok:
            events.append(
                {
                    "kind": "timing",
                    "index": int(index),
                    "step": int(sample["step"]),
                    "t": float(sample["t"]),
                    "direction": direction,
                    "evidence": float(sample["evidence"]),
                    "confidence": float(sample["confidence"]),
                    "dominant_delay_s": (
                        None
                        if sample["dominant_delay_s"] is None
                        else float(sample["dominant_delay_s"])
                    ),
                }
            )
            last_event_index = index
        previous_qualified = qualified
        previous_direction = direction if qualified else None

    return events


def _annotate_behavior(
    events: list[dict[str, Any]],
    *,
    recording: dict[str, Any],
    label: str,
    dt: float,
    config: OdorEventConfig,
) -> list[dict[str, Any]]:
    frames = recording["frames"]
    agents = [_recording_agent(frame, label) for frame in frames]
    turns = [float(agent["action"]["turn"]) for agent in agents]
    headings = [float(agent["heading"]) for agent in agents]
    response_steps = max(1, round(config.turn_response_horizon_s / dt))
    reacquisition_steps = max(1, round(config.reacquisition_horizon_s / dt))
    annotated: list[dict[str, Any]] = []

    for source_event in events:
        event = dict(source_event)
        index = int(event["index"])
        response_end = _window_end(index, steps=response_steps, length=len(frames))
        turn_window = turns[index : response_end + 1]
        event["behavior"] = {
            "mean_turn_command": float(np.mean(turn_window)) if turn_window else 0.0,
            "mean_abs_turn_command": (
                float(np.mean(np.abs(turn_window))) if turn_window else 0.0
            ),
            "peak_abs_turn_command": (
                float(np.max(np.abs(turn_window))) if turn_window else 0.0
            ),
            "heading_delta_rad": _angle_delta(headings[index], headings[response_end]),
            "response_horizon_s": float((response_end - index) * dt),
        }
        if event["kind"] == "loss":
            cutoff = min(len(frames) - 1, index + reacquisition_steps)
            reacquisition = next(
                (
                    candidate
                    for candidate in events
                    if candidate["kind"] == "reacquisition"
                    and int(candidate["index"]) > index
                    and int(candidate["index"]) <= cutoff
                ),
                None,
            )
            event["reacquisition"] = {
                "within_horizon": reacquisition is not None,
                "horizon_s": float(config.reacquisition_horizon_s),
                "latency_s": (
                    None
                    if reacquisition is None
                    else float(reacquisition["t"]) - float(event["t"])
                ),
            }
        annotated.append(event)
    return annotated


def _run_estimator(
    left: list[float],
    right: list[float],
    *,
    dt: float,
    motion_config: OdorMotionConfig,
) -> list[dict[str, Any]]:
    estimator = BilateralOdorMotionEstimator(dt=dt, config=motion_config)
    output: list[dict[str, Any]] = []
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
        estimate = estimator.update(
            step=index,
            t=index * dt,
            left_response=left_value,
            right_response=right_value,
        )
        output.append(
            {
                "evidence": float(estimate.evidence),
                "confidence": float(estimate.confidence),
                "direction": estimate.direction,
            }
        )
    return output


def _counterfactual_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = np.asarray([float(item["evidence"]) for item in samples], dtype=float)
    confidence = np.asarray([float(item["confidence"]) for item in samples], dtype=float)
    directional = [
        item for item in samples if item["direction"] in {"left_to_right", "right_to_left"}
    ]
    return {
        "mean_abs_evidence": float(np.mean(np.abs(evidence))) if len(evidence) else 0.0,
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else 0.0,
        "directional_fraction": float(len(directional) / len(samples)) if samples else 0.0,
    }


def build_counterfactuals(
    samples: list[dict[str, Any]],
    *,
    dt: float,
    motion_config: OdorMotionConfig,
    event_config: OdorEventConfig,
) -> dict[str, Any]:
    left = [float(item["left_response"]) for item in samples]
    right = [float(item["right_response"]) for item in samples]
    baseline = [
        {
            "evidence": float(item["evidence"]),
            "confidence": float(item["confidence"]),
            "direction": str(item["direction"]),
        }
        for item in samples
    ]
    mirrored = _run_estimator(
        right,
        left,
        dt=dt,
        motion_config=motion_config,
    )
    shift_steps = max(1, round(event_config.temporal_shift_s / dt))
    if right:
        shift_steps %= len(right)
    shifted_right = right[-shift_steps:] + right[:-shift_steps] if shift_steps else list(right)
    shifted = _run_estimator(
        left,
        shifted_right,
        dt=dt,
        motion_config=motion_config,
    )
    zeroed = [
        {"evidence": 0.0, "confidence": 0.0, "direction": "ablated"}
        for _ in samples
    ]

    mirror_error = (
        max(
            abs(float(first["evidence"]) + float(second["evidence"]))
            for first, second in zip(baseline, mirrored, strict=True)
        )
        if baseline
        else 0.0
    )
    return {
        "baseline": _counterfactual_summary(baseline),
        "mirror_left_right": {
            **_counterfactual_summary(mirrored),
            "max_sign_reversal_error": float(mirror_error),
        },
        "zero_temporal_evidence": _counterfactual_summary(zeroed),
        "right_channel_circular_shift": {
            **_counterfactual_summary(shifted),
            "requested_shift_s": float(event_config.temporal_shift_s),
            "realized_shift_s": float(shift_steps * dt),
            "preserves_right_channel_marginal_exactly": bool(
                sorted(right) == sorted(shifted_right)
            ),
            "analysis_only_noncausal_transform": True,
        },
    }


def build_diagnostics_bundle(
    recording_bundle: dict[str, Any],
    motion_bundle: dict[str, Any],
    *,
    motion_config: OdorMotionConfig,
    event_config: OdorEventConfig,
) -> dict[str, Any]:
    recording = recording_bundle["recording"]
    motion = motion_bundle["analysis"]
    recording_sha = recording_bundle["recording_sha256"]
    analysis_sha = motion_bundle["analysis_sha256"]
    if motion["source_recording_sha256"] != recording_sha:
        raise ValueError("odor-motion analysis does not belong to this recording")
    dt = float(recording["dt"])
    agents: list[dict[str, Any]] = []

    for controller in recording["controllers"]:
        label = str(controller["label"])
        samples = _motion_agent(motion, label)["samples"]
        if len(samples) != len(recording["frames"]):
            raise ValueError("odor-motion sample count does not match source recording")

        odor_events = detect_odor_state_events(samples, dt=dt, config=event_config)
        timing_events = detect_motion_events(samples, dt=dt, config=event_config)
        all_events = sorted(
            [*odor_events, *timing_events],
            key=lambda item: (int(item["index"]), str(item["kind"])),
        )
        annotated = _annotate_behavior(
            all_events,
            recording=recording,
            label=label,
            dt=dt,
            config=event_config,
        )
        losses = [event for event in annotated if event["kind"] == "loss"]
        timing = [event for event in annotated if event["kind"] == "timing"]
        reacquired = [
            event
            for event in losses
            if event["reacquisition"]["within_horizon"]
        ]
        agents.append(
            {
                "label": label,
                "events": annotated,
                "summary": {
                    "encounter_count": sum(
                        event["kind"] == "encounter" for event in annotated
                    ),
                    "loss_count": len(losses),
                    "reacquisition_count": sum(
                        event["kind"] == "reacquisition" for event in annotated
                    ),
                    "reacquisition_within_horizon_fraction": (
                        float(len(reacquired) / len(losses)) if losses else None
                    ),
                    "median_reacquisition_latency_s": (
                        float(
                            np.median(
                                [
                                    event["reacquisition"]["latency_s"]
                                    for event in reacquired
                                    if event["reacquisition"]["latency_s"] is not None
                                ]
                            )
                        )
                        if reacquired
                        else None
                    ),
                    "timing_event_count": len(timing),
                    "mean_abs_heading_response_rad": (
                        float(
                            np.mean(
                                [
                                    abs(event["behavior"]["heading_delta_rad"])
                                    for event in timing
                                ]
                            )
                        )
                        if timing
                        else None
                    ),
                },
                "counterfactuals": build_counterfactuals(
                    samples,
                    dt=dt,
                    motion_config=motion_config,
                    event_config=event_config,
                ),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_recording_sha256": recording_sha,
        "source_odor_motion_sha256": analysis_sha,
        "controller_access_in_v1": False,
        "claim_boundary": (
            "Event and counterfactual diagnostics are post-recording analysis only. "
            "They do not modify the sealed v1 controller and do not establish a neural mechanism."
        ),
        "event_definition": {
            "signal": "max(left_response, right_response)",
            "hysteresis": {
                "odor_on_threshold": event_config.odor_on_threshold,
                "odor_off_threshold": event_config.odor_off_threshold,
                "minimum_state_duration_s": event_config.minimum_state_duration_s,
            },
        },
        "behavior_contract": {
            "turn_command": "recorded controller command; positive is left turn",
            "heading_response": "wrapped heading change over the frozen response horizon",
            "reacquisition": "first hysteresis-qualified odor return after a loss",
        },
        "counterfactual_contract": {
            "mirror_left_right": "causal re-evaluation after swapping antenna identities",
            "zero_temporal_evidence": "analysis-only ablation preserving raw antenna samples",
            "right_channel_circular_shift": (
                "analysis-only noncausal temporal misalignment preserving the right-channel marginal"
            ),
        },
        "config": {
            "event_detection": asdict(event_config),
            "motion_estimator": asdict(motion_config),
        },
        "agents": agents,
    }
    return {
        "diagnostics": payload,
        "diagnostics_sha256": diagnostics_sha256(payload),
    }


def write_diagnostics(path: str | Path, bundle: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def load_diagnostics(path: str | Path) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text())
    payload = bundle.get("diagnostics")
    expected = bundle.get("diagnostics_sha256")
    if not isinstance(payload, dict) or not isinstance(expected, str):
        raise TypeError("invalid odor-event diagnostics bundle")
    actual = diagnostics_sha256(payload)
    if actual != expected:
        raise ValueError(
            f"odor-event diagnostics SHA-256 mismatch: expected {expected}, got {actual}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported odor-event diagnostics schema {payload.get('schema_version')!r}"
        )
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(
            f"unsupported odor-event diagnostics protocol {payload.get('protocol')!r}"
        )
    return bundle


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
    event_config = OdorEventConfig.from_document(config_document)
    diagnostics = build_diagnostics_bundle(
        recording_bundle,
        motion_bundle,
        motion_config=motion_config,
        event_config=event_config,
    )
    output = write_diagnostics(args.output, diagnostics)
    print(output)
    print(diagnostics["diagnostics_sha256"])


if __name__ == "__main__":
    main()
