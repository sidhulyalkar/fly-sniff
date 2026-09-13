from __future__ import annotations

import numpy as np


def crosswind_gradient(frame: np.ndarray) -> np.ndarray:
    """Centered dI/dy with one-sided boundary differences for (y, x) frames."""
    value = np.asarray(frame, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError(f"expected a 2-D frame with at least two y rows; got {value.shape}")
    output = np.empty_like(value, dtype=np.float32)
    output[0] = value[1] - value[0]
    output[-1] = value[-1] - value[-2]
    output[1:-1] = 0.5 * (value[2:] - value[:-2])
    return output


def published_motion_cue(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
) -> np.ndarray:
    """Published Figure-1 cue: -(dI/dy)(dI/dt), centered by one frame."""
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    following = np.asarray(following, dtype=np.float32)
    if previous.shape != current.shape or current.shape != following.shape:
        raise ValueError("previous, current, and following plume frames must share one shape")
    temporal = 0.5 * (following - previous)
    return np.asarray(-crosswind_gradient(current) * temporal, dtype=np.float32)


def cue_triplet(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen published gradient and motion definitions for one frame triplet."""
    return crosswind_gradient(current), published_motion_cue(previous, current, following)
