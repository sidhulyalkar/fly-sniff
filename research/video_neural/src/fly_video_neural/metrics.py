from __future__ import annotations

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


def summarize_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    corr = pearson_per_target(truth, prediction)
    r2 = r2_per_target(truth, prediction)
    return {
        "targets": int(corr.size),
        "valid_correlation_targets": int(np.isfinite(corr).sum()),
        "mean_pearson_r": float(np.nanmean(corr)),
        "median_pearson_r": float(np.nanmedian(corr)),
        "mean_r2": float(np.nanmean(r2)),
        "median_r2": float(np.nanmedian(r2)),
    }


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
