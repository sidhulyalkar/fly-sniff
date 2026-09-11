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


def spl(success: bool, shortest_path: float, path_length: float) -> float:
    if not success:
        return 0.0
    denom = max(float(shortest_path), float(path_length), 1e-12)
    return float(shortest_path / denom)


def paired_bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    seed: int = 13013,
    n_boot: int = 10000,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("paired arrays must be non-empty one-dimensional arrays of equal shape")
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
