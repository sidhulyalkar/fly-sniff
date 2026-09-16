from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .batch import FeatureTargetBatch
from .metrics import MeanTargetBaseline, metric_for_selection, summarize_metrics


class RidgeDecoder:
    def __init__(self, alpha: float) -> None:
        if alpha <= 0:
            raise ValueError("ridge alpha must be positive")
        self.alpha = float(alpha)
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RidgeDecoder:
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("ridge fit requires aligned 2D features and targets")
        if x.shape[0] < 2:
            raise ValueError("ridge fit requires at least two samples")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("ridge fit requires finite arrays")
        self.x_mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        self.x_scale_ = scale
        self.y_mean_ = y.mean(axis=0)
        xs = (x - self.x_mean_) / self.x_scale_
        yc = y - self.y_mean_
        gram = xs.T @ xs + self.alpha * np.eye(xs.shape[1])
        self.weights_ = np.linalg.solve(gram, xs.T @ yc)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if any(value is None for value in (self.x_mean_, self.x_scale_, self.y_mean_, self.weights_)):
            raise RuntimeError("ridge decoder must be fit before predict")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.x_mean_.shape[0]:
            raise ValueError("prediction features do not match fitted feature dimension")
        xs = (x - self.x_mean_) / self.x_scale_
        return xs @ self.weights_ + self.y_mean_


def _indices(batch: FeatureTargetBatch, split_lock: dict[str, Any], split: str) -> np.ndarray:
    mapping = split_lock["sample_to_split"]
    ids = batch.sample_ids.tolist()
    if set(ids) != set(mapping):
        missing = sorted(set(mapping) - set(ids))
        extra = sorted(set(ids) - set(mapping))
        raise ValueError(f"batch/split-lock sample mismatch; missing={missing[:5]} extra={extra[:5]}")
    return np.array([mapping[sample_id] == split for sample_id in ids], dtype=bool)


def run_ridge_baseline(
    batch: FeatureTargetBatch,
    split_lock: dict[str, Any],
    *,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    consume_test: bool = False,
) -> dict[str, Any]:
    train = _indices(batch, split_lock, "train")
    validation = _indices(batch, split_lock, "validation")
    test = _indices(batch, split_lock, "test")
    if min(train.sum(), validation.sum(), test.sum()) < 1:
        raise ValueError("train, validation, and test samples must all be present")
    candidates: list[dict[str, Any]] = []
    for alpha in alphas:
        model = RidgeDecoder(alpha).fit(batch.features[train], batch.targets[train])
        metrics = summarize_metrics(batch.targets[validation], model.predict(batch.features[validation]))
        candidates.append({"alpha": float(alpha), "validation": metrics})
    selected = max(
        candidates,
        key=lambda row: metric_for_selection(row["validation"], "median_pearson_r"),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "mc2p_future_neural_v0",
        "model": "ridge_decoder",
        "selection_metric": "median_pearson_r",
        "alpha_candidates": candidates,
        "selected_alpha": selected["alpha"],
        "test_status": "locked_not_consumed",
        "test_metrics": None,
        "mean_baseline_test_metrics": None,
    }
    if not consume_test:
        return report
    development = train | validation
    model = RidgeDecoder(selected["alpha"]).fit(
        batch.features[development], batch.targets[development]
    )
    mean = MeanTargetBaseline().fit(batch.targets[development])
    report["test_status"] = "consumed_explicitly"
    report["test_metrics"] = summarize_metrics(
        batch.targets[test], model.predict(batch.features[test])
    )
    report["mean_baseline_test_metrics"] = summarize_metrics(
        batch.targets[test], mean.predict(int(test.sum()))
    )
    return report


def load_split_lock(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
