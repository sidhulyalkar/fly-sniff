from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureTargetBatch:
    sample_ids: np.ndarray
    features: np.ndarray
    targets: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.sample_ids).astype(str)
        x = np.asarray(self.features, dtype=float)
        y = np.asarray(self.targets, dtype=float)
        if ids.ndim != 1:
            raise ValueError("sample_ids must be one-dimensional")
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("features and targets must be two-dimensional")
        if len(ids) != x.shape[0] or len(ids) != y.shape[0]:
            raise ValueError("sample_ids, features, and targets must share the sample axis")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("sample_ids must be unique")
        if x.shape[1] < 1 or y.shape[1] < 1:
            raise ValueError("batch needs at least one feature and one target")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("features and targets must be finite")
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "features", x)
        object.__setattr__(self, "targets", y)


def load_batch(path: str | Path) -> FeatureTargetBatch:
    with np.load(path, allow_pickle=False) as payload:
        missing = {"sample_ids", "features", "targets"} - set(payload.files)
        if missing:
            raise ValueError(f"feature-target batch missing arrays: {sorted(missing)}")
        return FeatureTargetBatch(
            sample_ids=payload["sample_ids"],
            features=payload["features"],
            targets=payload["targets"],
        )
