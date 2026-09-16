from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class EvidenceClass(StrEnum):
    MEASURED_VIDEO = "measured_video"
    MEASURED_POSE = "measured_pose"
    MEASURED_NEURAL_ACTIVITY = "measured_neural_activity"
    MEASURED_CONNECTIVITY = "measured_connectivity"
    INFERRED_BEHAVIOR_STATE = "inferred_behavior_state"
    MODELED_LATENT_NEURAL_STATE = "modeled_latent_neural_state"
    VIEWER_ONLY = "viewer_only"


@dataclass(frozen=True)
class SampleWindow:
    dataset_id: str
    sample_id: str
    animal_id: str
    session_id: str
    input_start_s: float
    input_end_s: float
    target_start_s: float
    target_end_s: float
    video_fps: float
    neural_rate_hz: float | None = None
    pose_available: bool = False
    context_available: bool = False

    def __post_init__(self) -> None:
        for name in ("dataset_id", "sample_id", "animal_id", "session_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.input_start_s < 0 or self.input_end_s <= self.input_start_s:
            raise ValueError("input interval must be positive and ordered")
        if self.target_end_s <= self.target_start_s:
            raise ValueError("target interval must be positive and ordered")
        if self.target_start_s < self.input_end_s:
            raise ValueError("future-neural benchmark forbids target/input temporal overlap")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if self.neural_rate_hz is not None and self.neural_rate_hz <= 0:
            raise ValueError("neural_rate_hz must be positive when provided")

    @property
    def input_duration_s(self) -> float:
        return self.input_end_s - self.input_start_s

    @property
    def target_duration_s(self) -> float:
        return self.target_end_s - self.target_start_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SampleWindow:
        return cls(**payload)
