from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    seed: int
    controller: str
    success: bool
    steps: int
    elapsed_s: float
    path_length: float
    shortest_path: float
    spl: float
    final_distance: float

    def as_dict(self) -> dict:
        return asdict(self)


def shortest_path_to_goal_region(center_distance: float, goal_radius: float) -> float:
    """Shortest Euclidean path from a point to a circular success region."""
    center_distance = float(center_distance)
    goal_radius = float(goal_radius)
    if not np.isfinite(center_distance) or not np.isfinite(goal_radius):
        raise ValueError("center distance and goal radius must be finite")
    if center_distance < 0.0 or goal_radius < 0.0:
        raise ValueError("center distance and goal radius must be nonnegative")
    return float(max(center_distance - goal_radius, 0.0))


def spl(success: bool, shortest_path: float, path_length: float) -> float:
    shortest_path = float(shortest_path)
    path_length = float(path_length)
    if not np.isfinite(shortest_path) or not np.isfinite(path_length):
        raise ValueError("SPL distances must be finite")
    if shortest_path < 0.0 or path_length < 0.0:
        raise ValueError("SPL distances must be nonnegative")
    if not success:
        return 0.0
    denom = max(shortest_path, path_length, 1e-12)
    return float(shortest_path / denom)


def paired_bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    seed: int = 13013,
    n_boot: int = 10000,
) -> tuple[float, float, float]:
    """Paired percentile-bootstrap CI for mean(a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("paired arrays must be non-empty one-dimensional arrays of equal shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired arrays must contain only finite values")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
