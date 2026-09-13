from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from .recording import load_recording, recording_sha256

PROTOCOL = "bilateral-odor-motion-analysis-v2"
SCHEMA_VERSION = 1
DEFAULT_CONFIG_RESOURCE = "configs/odor_motion_v2.json"


def analysis_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OdorMotionConfig:
    delays_s: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    delay_decay_tau_s: float = 0.15
    signal_scale: float = 0.08
    minimum_absolute_evidence: float = 0.12
    minimum_confidence: float = 0.08

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> OdorMotionConfig:
        return cls(
            delays_s=tuple(float(value) for value in mapping["delays_s"]),
            delay_decay_tau_s=float(mapping["delay_decay_tau_s"]),
            signal_scale=float(mapping["signal_scale"]),
            minimum_absolute_evidence=float(mapping["minimum_absolute_evidence"]),
            minimum_confidence=float(mapping["minimum_confidence"]),
        ).validated()

    def validated(self) -> OdorMotionConfig:
        if not self.delays_s:
            raise ValueError("odor-motion estimator requires at least one causal delay")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.delays_s):
            raise ValueError("odor-motion delays must be finite and > 0")
        if tuple(sorted(set(self.delays_s))) != self.delays_s:
            raise ValueError("odor-motion delays must be unique and strictly increasing")
        for name in ("delay_decay_tau_s", "signal_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("minimum_absolute_evidence", "minimum_confidence"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        return self


@dataclass(frozen=True)
class DelayEvidence:
    delay_s: float
    delay_steps: int
    evidence: float
    signal_gate: float
    weight: float


@dataclass(frozen=True)
class OdorMotionEstimate:
    step: int
    t: float
    left_response: float
    right_response: float
    evidence: float
    confidence: float
    direction: str
    dominant_delay_s: float | None
    delay_evidence: tuple[DelayEvidence, ...]


class BilateralOdorMotionEstimator:
    """Causal temporal comparator over modeled bilateral antenna responses.

    For each fixed lag d, the estimator compares L(t-d)R(t) with R(t-d)L(t).
    The operation is deliberately simple, auditable, and causal. It is an
    engineering adapter for testing whether temporal bilateral information helps
    navigation, not a claim that a specific MaleCNS circuit implements this rule.
    """

    def __init__(self, *, dt: float, config: OdorMotionConfig | None = None):
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and > 0")
        self.config = (config or OdorMotionConfig()).validated()
        self.delay_steps = tuple(
            max(1, int(round(delay_s / self.dt))) for delay_s in self.config.delays_s
        )
        if len(set(self.delay_steps)) != len(self.delay_steps):
            raise ValueError(
                "configured delays collapse onto duplicate sample lags at this recording dt"
            )
        self.realized_delays_s = tuple(step * self.dt for step in self.delay_steps)
        self.left_history: list[float] = []
        self.right_history: list[float] = []

    @staticmethod
    def _clean_response(value: float, name: str) -> float:
        clean = float(value)
        if not np.isfinite(clean) or clean < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return clean

    def update(
        self,
        *,
        step: int,
        t: float,
        left_response: float,
        right_response: float,
    ) -> OdorMotionEstimate:
        left = self._clean_response(left_response, "left_response")
        right = self._clean_response(right_response, "right_response")
        sample_t = float(t)
        if not np.isfinite(sample_t):
            raise ValueError("t must be finite")

        self.left_history.append(left)
        self.right_history.append(right)
        current_index = len(self.left_history) - 1
        components: list[DelayEvidence] = []

        for delay_steps, realized_delay in zip(
            self.delay_steps,
            self.realized_delays_s,
            strict=True,
        ):
            delayed_index = current_index - delay_steps
            if delayed_index < 0:
                continue
            left_delayed = self.left_history[delayed_index]
            right_delayed = self.right_history[delayed_index]
            left_then_right = left_delayed * right
            right_then_left = right_delayed * left
            denominator = abs(left_then_right) + abs(right_then_left) + 1e-12
            evidence = float((left_then_right - right_then_left) / denominator)
            mean_signal = 0.25 * (left_delayed + right_delayed + left + right)
            signal_gate = float(1.0 - np.exp(-mean_signal / self.config.signal_scale))
            decay = float(np.exp(-realized_delay / self.config.delay_decay_tau_s))
            weight = float(decay * signal_gate)
            components.append(
                DelayEvidence(
                    delay_s=float(realized_delay),
                    delay_steps=int(delay_steps),
                    evidence=evidence,
                    signal_gate=signal_gate,
                    weight=weight,
                )
            )

        if not components:
            return OdorMotionEstimate(
                step=int(step),
                t=sample_t,
                left_response=left,
                right_response=right,
                evidence=0.0,
                confidence=0.0,
                direction="insufficient_history",
                dominant_delay_s=None,
                delay_evidence=(),
            )

        weighted_sum = sum(item.weight * item.evidence for item in components)
        weight_total = sum(item.weight for item in components)
        evidence = float(weighted_sum / weight_total) if weight_total > 1e-12 else 0.0
        coverage = len(components) / len(self.delay_steps)
        max_possible_weight = sum(
            np.exp(-delay / self.config.delay_decay_tau_s)
            for delay in self.realized_delays_s
            if current_index >= int(round(delay / self.dt))
        )
        signal_strength = float(weight_total / max_possible_weight) if max_possible_weight else 0.0
        confidence = float(np.clip(coverage * signal_strength * abs(evidence), 0.0, 1.0))

        if signal_strength < 1e-6:
            direction = "no_signal"
        elif (
            abs(evidence) < self.config.minimum_absolute_evidence
            or confidence < self.config.minimum_confidence
        ):
            direction = "ambiguous"
        elif evidence > 0.0:
            direction = "left_to_right"
        else:
            direction = "right_to_left"

        dominant = max(components, key=lambda item: abs(item.weight * item.evidence))
        dominant_delay = (
            float(dominant.delay_s)
            if direction in {"left_to_right", "right_to_left"}
            else None
        )
        return OdorMotionEstimate(
            step=int(step),
            t=sample_t,
            left_response=left,
            right_response=right,
            evidence=evidence,
            confidence=confidence,
            direction=direction,
            dominant_delay_s=dominant_delay,
            delay_evidence=tuple(components),
        )


def _default_config_text() -> str:
    resource = files("fly_sniff").joinpath(DEFAULT_CONFIG_RESOURCE)
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    source_path = Path(__file__).resolve().parents[2] / "configs" / "odor_motion_v2.json"
    return source_path.read_text()


def load_odor_motion_config(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], OdorMotionConfig]:
    text = Path(path).read_text() if path is not None else _default_config_text()
    raw = json.loads(text)
    if raw.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported odor-motion protocol {raw.get('protocol')!r}")
    if raw.get("controller_access_in_v1") is not False:
        raise ValueError("v2 analysis contract must not grant the sealed v1 controller new inputs")
    estimator = raw.get("estimator")
    if not isinstance(estimator, dict):
        raise TypeError("odor-motion config is missing estimator settings")
    return raw, OdorMotionConfig.from_mapping(estimator)


def _agent_sequence(
    recording: dict[str, Any],
    label: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    sequence: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for frame in recording.get("frames", []):
        matches = [agent for agent in frame.get("agents", []) if agent.get("label") == label]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one agent labelled {label!r} in every recording frame"
            )
        sequence.append((frame, matches[0]))
    return sequence


def analyze_recording_bundle(
    bundle: dict[str, Any],
    *,
    config_document: dict[str, Any],
    config: OdorMotionConfig,
) -> dict[str, Any]:
    recording = bundle.get("recording")
    source_sha = bundle.get("recording_sha256")
    if not isinstance(recording, dict) or not isinstance(source_sha, str):
        raise TypeError("invalid recording bundle")
    if recording_sha256(recording) != source_sha:
        raise ValueError("source recording SHA-256 mismatch")
    if recording.get("model_contract", {}).get("sensor_model") != "bilateral-phenomenological-v1":
        raise ValueError("odor-motion v2 requires the audited bilateral sensor recording contract")

    dt = float(recording["dt"])
    controllers = recording.get("controllers", [])
    labels = [str(item["label"]) for item in controllers]
    analyses: list[dict[str, Any]] = []

    for label in labels:
        estimator = BilateralOdorMotionEstimator(dt=dt, config=config)
        samples: list[dict[str, Any]] = []
        for frame, agent in _agent_sequence(recording, label):
            trace = agent.get("sensor_trace", {})
            if trace.get("signal_kind") != "modeled_antenna_transduction":
                raise ValueError("odor-motion analysis requires modeled antenna transduction")
            estimate = estimator.update(
                step=int(frame["step"]),
                t=float(frame["t"]),
                left_response=float(trace["left"]["response"]),
                right_response=float(trace["right"]["response"]),
            )
            sample = asdict(estimate)
            sample["delay_evidence"] = [asdict(item) for item in estimate.delay_evidence]
            samples.append(sample)
        analyses.append({"label": label, "samples": samples})

    analysis_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_recording_sha256": source_sha,
        "claim_boundary": config_document["claim_boundary"],
        "controller_access_in_v1": False,
        "input_contract": {
            "used_fields": [
                "recording.dt",
                "frames[].step",
                "frames[].t",
                "frames[].agents[].label",
                "frames[].agents[].sensor_trace.left.response",
                "frames[].agents[].sensor_trace.right.response",
            ],
            "forbidden_privileged_inputs": [
                "source coordinates",
                "distance to source",
                "plume particle positions",
                "culprit identity",
                "future antenna samples",
            ],
        },
        "direction_convention": config_document["direction_convention"],
        "visualization_contract": config_document["visualization_contract"],
        "dt": dt,
        "estimator": asdict(config),
        "agents": analyses,
    }
    return {
        "analysis": analysis_payload,
        "analysis_sha256": analysis_sha256(analysis_payload),
    }


def write_analysis(path: str | Path, bundle: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def load_analysis(path: str | Path) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text())
    payload = bundle.get("analysis")
    expected = bundle.get("analysis_sha256")
    if not isinstance(payload, dict) or not isinstance(expected, str):
        raise TypeError("invalid odor-motion analysis bundle")
    actual = analysis_sha256(payload)
    if actual != expected:
        raise ValueError(f"odor-motion SHA-256 mismatch: expected {expected}, got {actual}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported odor-motion schema {payload.get('schema_version')!r}")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported odor-motion protocol {payload.get('protocol')!r}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a causal bilateral odor-motion sidecar from an auditable recording"
    )
    parser.add_argument("recording")
    parser.add_argument("--config")
    parser.add_argument(
        "--output",
        default="artifacts/showcase/odor-motion-v2.json",
    )
    args = parser.parse_args()

    source_bundle = load_recording(args.recording)
    config_document, config = load_odor_motion_config(args.config)
    analysis = analyze_recording_bundle(
        source_bundle,
        config_document=config_document,
        config=config,
    )
    output = write_analysis(args.output, analysis)
    print(output)
    print(analysis["analysis_sha256"])


if __name__ == "__main__":
    main()
