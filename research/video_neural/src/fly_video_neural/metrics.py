from __future__ import annotations

from typing import Any

import numpy as np


def _arrays(truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(truth, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("truth and prediction must have identical [samples, targets] shape")
    if y.shape[0] < 2 or y.shape[1] < 1:
        raise ValueError("metrics require at least two samples and one target")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("metrics require finite arrays")
    return y, p


def pearson_per_target(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    y, p = _arrays(truth, prediction)
    y0 = y - y.mean(axis=0)
    p0 = p - p.mean(axis=0)
    denom = np.sqrt((y0 * y0).sum(axis=0) * (p0 * p0).sum(axis=0))
    out = np.full(y.shape[1], np.nan, dtype=float)
    valid = denom > 0
    out[valid] = (y0[:, valid] * p0[:, valid]).sum(axis=0) / denom[valid]
    return out


def r2_per_target(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    y, p = _arrays(truth, prediction)
    residual = ((y - p) ** 2).sum(axis=0)
    total = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    out = np.full(y.shape[1], np.nan, dtype=float)
    valid = total > 0
    out[valid] = 1.0 - residual[valid] / total[valid]
    return out


def _finite_summary(values: np.ndarray) -> tuple[int, float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, None, None
    return int(finite.size), float(np.mean(finite)), float(np.median(finite))


def summarize_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int | None]:
    corr = pearson_per_target(truth, prediction)
    r2 = r2_per_target(truth, prediction)
    valid_corr, mean_corr, median_corr = _finite_summary(corr)
    valid_r2, mean_r2, median_r2 = _finite_summary(r2)
    return {
        "targets": int(corr.size),
        "valid_correlation_targets": valid_corr,
        "valid_r2_targets": valid_r2,
        "mean_pearson_r": mean_corr,
        "median_pearson_r": median_corr,
        "mean_r2": mean_r2,
        "median_r2": median_r2,
    }


def metric_for_selection(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        return float("-inf")
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("-inf")


class MeanTargetBaseline:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, y_train: np.ndarray) -> MeanTargetBaseline:
        y = np.asarray(y_train, dtype=float)
        if y.ndim != 2 or not np.isfinite(y).all():
            raise ValueError("training targets must be finite [samples, targets]")
        self.mean_ = y.mean(axis=0)
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("baseline must be fit before predict")
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        return np.repeat(self.mean_[None, :], n_samples, axis=0)
