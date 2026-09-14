from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .mc2p import MC2PSession, inspect_session
from .schema import SampleWindow


@dataclass(frozen=True)
class AlignedWindow:
    sample: SampleWindow
    input_behavior_frames: tuple[int, int]
    target_behavior_frames: tuple[int, int]
    target_neural_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "input_behavior_frames": list(self.input_behavior_frames),
            "target_behavior_frames": list(self.target_behavior_frames),
            "target_neural_indices": list(self.target_neural_indices),
        }


def load_safe_alignment(path: str | Path) -> np.ndarray:
    alignment_path = Path(path)
    if alignment_path.suffix != ".npy":
        raise ValueError(
            "window materialization accepts only non-pickled .npy alignment arrays; "
            "convert trusted upstream pickle explicitly before use"
        )
    values = np.load(alignment_path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("alignment must be a one-dimensional array with at least two frames")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.issubdtype(values.dtype, np.floating) or not np.all(values == np.floor(values)):
            raise ValueError("alignment values must be integer neural frame indices")
    alignment = np.asarray(values, dtype=np.int64)
    if (alignment < 0).any():
        raise ValueError("alignment cannot contain negative neural frame indices")
    if (np.diff(alignment) < 0).any():
        raise ValueError("behavior-to-neural alignment must be monotonic non-decreasing")
    return alignment


def materialize_windows(
    session: MC2PSession,
    alignment: np.ndarray,
    *,
    video_fps: float = 100.0,
    history_s: float = 3.0,
    horizon_s: float = 0.5,
    stride_s: float = 0.5,
) -> list[AlignedWindow]:
    if video_fps <= 0 or history_s <= 0 or horizon_s <= 0 or stride_s <= 0:
        raise ValueError("fps, history, horizon, and stride must be positive")
    history_frames = round(video_fps * history_s)
    horizon_frames = round(video_fps * horizon_s)
    stride_frames = round(video_fps * stride_s)
    if min(history_frames, horizon_frames, stride_frames) < 1:
        raise ValueError("temporal settings must map to at least one behavior frame")
    if history_frames + horizon_frames > len(alignment):
        return []
    rows: list[AlignedWindow] = []
    target_start = history_frames
    final_start = len(alignment) - horizon_frames
    for current_target_start in range(target_start, final_start + 1, stride_frames):
        input_start = current_target_start - history_frames
        input_end = current_target_start
        target_end = current_target_start + horizon_frames
        neural_indices = tuple(
            int(index) for index in np.unique(alignment[current_target_start:target_end])
        )
        if not neural_indices:
            continue
        sample = SampleWindow(
            dataset_id="mc2p_v1",
            sample_id=f"{session.session_id}:{input_start}-{target_end}",
            animal_id=session.animal_id,
            session_id=session.session_id,
            input_start_s=input_start / video_fps,
            input_end_s=input_end / video_fps,
            target_start_s=current_target_start / video_fps,
            target_end_s=target_end / video_fps,
            video_fps=video_fps,
            neural_rate_hz=None,
            pose_available=session.pose is not None,
            context_available=True,
        )
        rows.append(
            AlignedWindow(
                sample=sample,
                input_behavior_frames=(input_start, input_end),
                target_behavior_frames=(current_target_start, target_end),
                target_neural_indices=neural_indices,
            )
        )
    return rows


def materialize_session_windows(
    session_path: str | Path,
    alignment_path: str | Path,
    *,
    video_fps: float = 100.0,
    history_s: float = 3.0,
    horizon_s: float = 0.5,
    stride_s: float = 0.5,
) -> dict[str, Any]:
    session = inspect_session(session_path)
    alignment = load_safe_alignment(alignment_path)
    windows = materialize_windows(
        session,
        alignment,
        video_fps=video_fps,
        history_s=history_s,
        horizon_s=horizon_s,
        stride_s=stride_s,
    )
    return {
        "schema_version": 1,
        "dataset_id": "mc2p_v1",
        "session": session.to_dict(),
        "alignment_path": str(Path(alignment_path).resolve()),
        "alignment_length_behavior_frames": int(len(alignment)),
        "history_s": history_s,
        "horizon_s": horizon_s,
        "stride_s": stride_s,
        "window_count": len(windows),
        "windows": [window.to_dict() for window in windows],
    }
