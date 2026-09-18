from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .olfactory_door import sha256_file
from .olfactory_e006_audit import _canonical_sha
from .olfactory_o002 import _balanced_accuracy
from .olfactory_o002_robustness import (
    _eligible_labels,
    _load_inputs,
    _row_l2_normalize,
    _validate_v1_receipt,
)

HOLDOUT_DRAWS = 512
HOLDOUT_SEED = 24020
PCA_COMPONENTS = (1, 2, 3, 5, 8, 12, 16, 20, 24)


def _validate_v2_receipt(path: Path, v1_receipt_sha256: str) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if receipt.get("protocol") != "o002-coding-robustness-development-v2":
        raise ValueError("O002 stability v3 requires an o002-coding-robustness-development-v2 receipt")
    observed = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if observed != _canonical_sha(unhashed):
        raise ValueError("O002 v2 canonical receipt hash mismatch")
    if receipt.get("development_only") is not True:
        raise ValueError("O002 v2 receipt must remain development-only")
    if receipt.get("confirmatory_use_allowed") is not False:
        raise ValueError("O002 v2 receipt unexpectedly permits confirmatory use")
    if str(receipt["input"]["v1_receipt_sha256"]) != v1_receipt_sha256:
        raise ValueError("O002 v2 does not point to the supplied O002 v1 receipt")
    return receipt


def _predict_nearest_centroid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    train_z = (x_train - mean) / std
    test_z = (x_test - mean) / std
    labels = sorted(set(y_train.tolist()))
    centroids = {label: train_z[y_train == label].mean(axis=0) for label in labels}
    return np.asarray(
        [
            min(
                labels,
                key=lambda label: float(np.square(row - centroids[label]).sum()),
            )
            for row in test_z
        ],
        dtype=object,
    )


def _loo_predictions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions: list[str] = []
    for test_index in range(len(y)):
        train = np.ones(len(y), dtype=bool)
        train[test_index] = False
        prediction = _predict_nearest_centroid(
            x[train],
            y[train],
            x[test_index : test_index + 1],
        )
        predictions.append(str(prediction[0]))
    return np.asarray(predictions, dtype=object)


def _class_diagnostics(
    y: np.ndarray,
    predictions: np.ndarray,
    classes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    confusion = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    rows: list[dict[str, Any]] = []
    for label in classes:
        mask = y == label
        pred = predictions[mask]
        recall = float(np.mean(pred == label))
        rows.append(
            {
                "odor_class": label,
                "count": int(mask.sum()),
                "recall": recall,
            }
        )
        for predicted in classes:
            confusion.loc[label, predicted] = int(np.sum(pred == predicted))

    recalls = np.asarray([row["recall"] for row in rows], dtype=float)
    sensitivity: dict[str, float] = {}
    for excluded in classes:
        keep = y != excluded
        sensitivity[excluded] = _balanced_accuracy(y[keep], predictions[keep])

    return (
        pd.DataFrame(rows),
        confusion,
        {
            "minimum_recall": float(recalls.min()),
            "median_recall": float(np.median(recalls)),
            "maximum_recall": float(recalls.max()),
            "macro_accuracy_excluding_each_class": sensitivity,
            "minimum_macro_accuracy_after_exclusion": float(min(sensitivity.values())),
            "maximum_macro_accuracy_after_exclusion": float(max(sensitivity.values())),
        },
    )


def _balanced_holdout_stability(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    direction = _row_l2_normalize(x)
    sorted_profile = np.sort(x, axis=1)
    rng = np.random.default_rng(HOLDOUT_SEED)
    classes = sorted(set(y.tolist()))
    rows: list[dict[str, Any]] = []

    class_indices = {label: np.flatnonzero(y == label) for label in classes}
    if any(len(indices) < 2 for indices in class_indices.values()):
        raise ValueError("balanced holdout requires at least two odors in every class")

    for draw in range(HOLDOUT_DRAWS):
        test_indices = np.asarray(
            [rng.choice(class_indices[label]) for label in classes],
            dtype=int,
        )
        train = np.ones(len(y), dtype=bool)
        train[test_indices] = False
        test_y = y[test_indices]

        full_pred = _predict_nearest_centroid(x[train], y[train], x[test_indices])
        direction_pred = _predict_nearest_centroid(
            direction[train],
            y[train],
            direction[test_indices],
        )
        sorted_pred = _predict_nearest_centroid(
            sorted_profile[train],
            y[train],
            sorted_profile[test_indices],
        )
        full = float(np.mean(full_pred == test_y))
        direction_score = float(np.mean(direction_pred == test_y))
        sorted_score = float(np.mean(sorted_pred == test_y))
        rows.append(
            {
                "draw": draw,
                "full_pattern_accuracy": full,
                "direction_only_accuracy": direction_score,
                "identity_erased_sorted_accuracy": sorted_score,
                "direction_minus_full": direction_score - full,
                "full_minus_sorted": full - sorted_score,
            }
        )

    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "draws": HOLDOUT_DRAWS,
        "seed": HOLDOUT_SEED,
        "test_design": "one randomly selected odor per eligible class per draw; paired transforms use identical test odors",
    }
    for column in (
        "full_pattern_accuracy",
        "direction_only_accuracy",
        "identity_erased_sorted_accuracy",
        "direction_minus_full",
        "full_minus_sorted",
    ):
        values = frame[column].to_numpy(dtype=float)
        summary[column] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    summary["fraction_direction_greater_than_full"] = float(
        np.mean(frame["direction_minus_full"].to_numpy(dtype=float) > 0)
    )
    summary["fraction_full_greater_than_sorted"] = float(
        np.mean(frame["full_minus_sorted"].to_numpy(dtype=float) > 0)
    )
    return frame, summary


def _loo_pca_accuracy(x: np.ndarray, y: np.ndarray, components: int) -> float:
    predictions: list[str] = []
    for test_index in range(len(y)):
        train = np.ones(len(y), dtype=bool)
        train[test_index] = False
        x_train = x[train]
        y_train = y[train]
        x_test = x[test_index : test_index + 1]

        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        train_z = (x_train - mean) / std
        test_z = (x_test - mean) / std

        pca_mean = train_z.mean(axis=0)
        train_centered = train_z - pca_mean
        test_centered = test_z - pca_mean
        _, _, vt = np.linalg.svd(train_centered, full_matrices=False)
        k = min(components, vt.shape[0], vt.shape[1])
        basis = vt[:k].T
        train_pc = train_centered @ basis
        test_pc = test_centered @ basis

        labels = sorted(set(y_train.tolist()))
        centroids = {
            label: train_pc[y_train == label].mean(axis=0)
            for label in labels
        }
        prediction = min(
            labels,
            key=lambda label: float(np.square(test_pc[0] - centroids[label]).sum()),
        )
        predictions.append(prediction)
    return _balanced_accuracy(y, np.asarray(predictions, dtype=object))


def _subspace_curve(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    direction = _row_l2_normalize(x)
    rows: list[dict[str, Any]] = []
    for components in PCA_COMPONENTS:
        rows.append(
            {
                "components": components,
                "full_pattern_balanced_accuracy": _loo_pca_accuracy(x, y, components),
                "direction_only_balanced_accuracy": _loo_pca_accuracy(
                    direction,
                    y,
                    components,
                ),
            }
        )
    return pd.DataFrame(rows)


def run_o002_stability(
    o002_dir: str | Path,
    robustness_dir: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(o002_dir).expanduser().resolve()
    robustness = Path(robustness_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty O002 stability output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    v1_path = source / "o002-development-receipt.json"
    v2_path = robustness / "o002-robustness-receipt.json"
    if not v1_path.is_file():
        raise FileNotFoundError(f"missing O002 v1 receipt: {v1_path}")
    if not v2_path.is_file():
        raise FileNotFoundError(f"missing O002 v2 receipt: {v2_path}")

    v1 = _validate_v1_receipt(v1_path)
    v2 = _validate_v2_receipt(v2_path, str(v1["receipt_sha256"]))
    matrix, metadata = _load_inputs(source, v1)
    x, y, classes, class_counts = _eligible_labels(matrix, metadata)

    full_predictions = _loo_predictions(x, y)
    direction_x = _row_l2_normalize(x)
    direction_predictions = _loo_predictions(direction_x, y)

    full_class, full_confusion, full_class_summary = _class_diagnostics(
        y,
        full_predictions,
        classes,
    )
    direction_class, direction_confusion, direction_class_summary = _class_diagnostics(
        y,
        direction_predictions,
        classes,
    )
    holdout_frame, holdout_summary = _balanced_holdout_stability(x, y)
    subspace = _subspace_curve(x, y)

    full_class_path = output / "o002-v3-full-pattern-per-class.csv"
    direction_class_path = output / "o002-v3-direction-only-per-class.csv"
    full_confusion_path = output / "o002-v3-full-pattern-confusion.csv"
    direction_confusion_path = output / "o002-v3-direction-only-confusion.csv"
    holdout_path = output / "o002-v3-balanced-holdout.csv"
    subspace_path = output / "o002-v3-subspace-curve.csv"

    full_class.to_csv(full_class_path, index=False)
    direction_class.to_csv(direction_class_path, index=False)
    full_confusion.to_csv(full_confusion_path)
    direction_confusion.to_csv(direction_confusion_path)
    holdout_frame.to_csv(holdout_path, index=False)
    subspace.to_csv(subspace_path, index=False)

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "o002-subspace-stability-development-v3",
        "program_id": "olfactory-computation-v0",
        "aim": "O002_multi_odor_representation",
        "status": "development_complete_not_confirmatory",
        "development_only": True,
        "confirmatory_use_allowed": False,
        "o003_accessed": False,
        "frozen_question": (
            "Is the direction-dominant within-study chemical-class signal stable across balanced odor "
            "resampling, distributed across classes, and preserved in a low-dimensional population subspace?"
        ),
        "frozen_before_v3_outcomes": True,
        "input": {
            "v1_receipt_sha256": v1["receipt_sha256"],
            "v1_receipt_file_sha256": sha256_file(v1_path),
            "v2_receipt_sha256": v2["receipt_sha256"],
            "v2_receipt_file_sha256": sha256_file(v2_path),
            "matrix_sha256": v1["outputs"]["matrix"]["sha256"],
            "metadata_sha256": v1["outputs"]["odor_metadata"]["sha256"],
        },
        "sample": {
            "study_id": v1["selected_study"]["study_id"],
            "eligible_odors": len(y),
            "responding_units": x.shape[1],
            "eligible_classes": classes,
            "class_counts": class_counts,
        },
        "class_diagnostics": {
            "full_pattern": full_class_summary,
            "direction_only": direction_class_summary,
        },
        "balanced_holdout_stability": holdout_summary,
        "subspace_curve": [
            {
                "components": int(row["components"]),
                "full_pattern_balanced_accuracy": float(
                    row["full_pattern_balanced_accuracy"]
                ),
                "direction_only_balanced_accuracy": float(
                    row["direction_only_balanced_accuracy"]
                ),
            }
            for _, row in subspace.iterrows()
        ],
        "outputs": {
            "full_pattern_per_class": {
                "path": str(full_class_path),
                "sha256": sha256_file(full_class_path),
            },
            "direction_only_per_class": {
                "path": str(direction_class_path),
                "sha256": sha256_file(direction_class_path),
            },
            "full_pattern_confusion": {
                "path": str(full_confusion_path),
                "sha256": sha256_file(full_confusion_path),
            },
            "direction_only_confusion": {
                "path": str(direction_confusion_path),
                "sha256": sha256_file(direction_confusion_path),
            },
            "balanced_holdout": {
                "path": str(holdout_path),
                "sha256": sha256_file(holdout_path),
            },
            "subspace_curve": {
                "path": str(subspace_path),
                "sha256": sha256_file(subspace_path),
            },
        },
        "claim_boundary": (
            "This v3 analysis is a frozen-after-v2 development stability analysis. It can characterize "
            "whether the within-study chemical-class signal is stable to balanced odor resampling, broadly "
            "distributed across classes, and retained under low-rank projection. The repeated holdouts are "
            "stability diagnostics, not independent biological replicates or confirmatory p-values. It does "
            "not establish odor identity, valence, receptor-specific mechanism, cross-study generalization, "
            "connectome topology effects, or authorize O003/O004."
        ),
    }
    report["receipt_sha256"] = _canonical_sha(report)
    receipt_path = output / "o002-stability-receipt.json"
    receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    best_direction = max(
        report["subspace_curve"],
        key=lambda row: row["direction_only_balanced_accuracy"],
    )
    lines = [
        "O002 SUBSPACE STABILITY DEVELOPMENT V3",
        f"status: {report['status']}",
        f"study: {report['sample']['study_id']}",
        f"eligible_odors: {report['sample']['eligible_odors']}",
        f"responding_units: {report['sample']['responding_units']}",
        (
            "balanced_holdout_full_mean: "
            f"{holdout_summary['full_pattern_accuracy']['mean']:.6f}"
        ),
        (
            "balanced_holdout_direction_mean: "
            f"{holdout_summary['direction_only_accuracy']['mean']:.6f}"
        ),
        (
            "balanced_holdout_direction_minus_full_mean: "
            f"{holdout_summary['direction_minus_full']['mean']:.6f}"
        ),
        (
            "balanced_holdout_fraction_direction_gt_full: "
            f"{holdout_summary['fraction_direction_greater_than_full']:.6f}"
        ),
        (
            "balanced_holdout_fraction_full_gt_sorted: "
            f"{holdout_summary['fraction_full_greater_than_sorted']:.6f}"
        ),
        (
            "full_pattern_min_class_recall: "
            f"{full_class_summary['minimum_recall']:.6f}"
        ),
        (
            "direction_only_min_class_recall: "
            f"{direction_class_summary['minimum_recall']:.6f}"
        ),
        (
            "best_direction_subspace_components: "
            f"{best_direction['components']}"
        ),
        (
            "best_direction_subspace_balanced_accuracy: "
            f"{best_direction['direction_only_balanced_accuracy']:.6f}"
        ),
        f"receipt_sha256: {report['receipt_sha256']}",
        "",
        "CLAIM BOUNDARY",
        report["claim_boundary"],
    ]
    (output / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    return report
