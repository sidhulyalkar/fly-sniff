from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .olfactory_door import sha256_file
from .olfactory_e006_audit import _canonical_sha
from .olfactory_o002 import (
    MIN_CLASS_COUNT,
    _balanced_accuracy,
    _loo_nearest_centroid,
)

ROBUSTNESS_SEED = 24018
CHANNEL_SHUFFLE_COUNT = 256
FEATURE_SUBSET_DRAWS = 64
FEATURE_SUBSET_SIZES = (4, 8, 12, 16, 20)


def _validate_v1_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if receipt.get("protocol") != "o002-within-study-development-v1":
        raise ValueError("O002 robustness v2 requires an o002-within-study-development-v1 receipt")
    observed = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if observed != _canonical_sha(unhashed):
        raise ValueError("O002 v1 canonical receipt hash mismatch")
    if receipt.get("development_only") is not True:
        raise ValueError("O002 v1 receipt must remain development-only")
    if receipt.get("confirmatory_use_allowed") is not False:
        raise ValueError("O002 v1 receipt unexpectedly permits confirmatory use")
    if receipt.get("feature_identity") != "source_responding_unit":
        raise ValueError("O002 robustness v2 requires source responding-unit identity")
    return receipt


def _load_inputs(o002_dir: Path, receipt: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_path = o002_dir / "o002-complete-response-matrix.csv"
    metadata_path = o002_dir / "o002-odor-metadata.csv"
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("O002 v1 matrix or odor metadata is missing")

    expected_matrix = str(receipt["outputs"]["matrix"]["sha256"])
    expected_metadata = str(receipt["outputs"]["odor_metadata"]["sha256"])
    if sha256_file(matrix_path) != expected_matrix:
        raise ValueError("O002 v1 matrix hash mismatch")
    if sha256_file(metadata_path) != expected_metadata:
        raise ValueError("O002 v1 odor metadata hash mismatch")

    matrix = pd.read_csv(matrix_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)
    matrix.index = matrix.index.astype(str)
    metadata.index = metadata.index.astype(str)
    if not matrix.index.equals(metadata.index):
        raise ValueError("O002 v1 matrix and metadata source-row order differ")
    if matrix.isna().any().any():
        raise ValueError("O002 v2 refuses matrices with missing values")
    return matrix, metadata


def _eligible_labels(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    labels = metadata["odor_class"].fillna("").astype(str).str.strip()
    counts = labels[labels.ne("")].value_counts()
    eligible = sorted(counts[counts.ge(MIN_CLASS_COUNT)].index.tolist())
    mask = labels.isin(eligible)
    x = matrix.loc[mask].to_numpy(dtype=float)
    y = labels.loc[mask].to_numpy(dtype=object)
    if len(eligible) < 2 or len(y) < 10:
        raise ValueError("O002 robustness requires at least two eligible classes and ten odors")
    return x, y, eligible, {label: int(counts[label]) for label in eligible}


def _score(x: np.ndarray, y: np.ndarray) -> float:
    return _balanced_accuracy(y, _loo_nearest_centroid(x, y))


def _row_l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    safe = np.where(norm > 0, norm, 1.0)
    return x / safe


def _amplitude_only(x: np.ndarray) -> dict[str, float]:
    mean = x.mean(axis=1, keepdims=True)
    l2 = np.linalg.norm(x, axis=1, keepdims=True)
    peak_to_peak = np.ptp(x, axis=1, keepdims=True)
    return {
        "mean_response_balanced_accuracy": float("nan"),
        "l2_norm_balanced_accuracy": float("nan"),
        "peak_to_peak_balanced_accuracy": float("nan"),
        "_mean": mean,
        "_l2": l2,
        "_peak_to_peak": peak_to_peak,
    }


def _channel_shuffle_null(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(ROBUSTNESS_SEED)
    scores = np.empty(CHANNEL_SHUFFLE_COUNT, dtype=float)
    for iteration in range(CHANNEL_SHUFFLE_COUNT):
        shuffled = np.empty_like(x)
        for row in range(x.shape[0]):
            shuffled[row] = x[row, rng.permutation(x.shape[1])]
        scores[iteration] = _score(shuffled, y)
    return {
        "count": CHANNEL_SHUFFLE_COUNT,
        "seed": ROBUSTNESS_SEED,
        "mean": float(scores.mean()),
        "q05": float(np.quantile(scores, 0.05)),
        "median": float(np.median(scores)),
        "q95": float(np.quantile(scores, 0.95)),
        "maximum": float(scores.max()),
        "scores": [float(value) for value in scores],
    }


def _leave_one_unit(
    x: np.ndarray,
    y: np.ndarray,
    unit_names: list[str],
    full_score: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, unit in enumerate(unit_names):
        reduced = np.delete(x, index, axis=1)
        accuracy = _score(reduced, y)
        rows.append(
            {
                "responding_unit": unit,
                "balanced_accuracy": accuracy,
                "delta_from_full": accuracy - full_score,
            }
        )
    values = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
    worst = min(rows, key=lambda row: row["balanced_accuracy"])
    best = max(rows, key=lambda row: row["balanced_accuracy"])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "worst_removed_unit": worst,
        "best_removed_unit": best,
        "per_unit": rows,
    }


def _feature_subsets(
    x: np.ndarray,
    y: np.ndarray,
    unit_names: list[str],
) -> dict[str, Any]:
    rng = np.random.default_rng(ROBUSTNESS_SEED + 1)
    result: dict[str, Any] = {
        "seed": ROBUSTNESS_SEED + 1,
        "draws_per_size": FEATURE_SUBSET_DRAWS,
        "sizes": {},
    }
    for size in FEATURE_SUBSET_SIZES:
        if size >= x.shape[1]:
            continue
        rows: list[dict[str, Any]] = []
        for draw in range(FEATURE_SUBSET_DRAWS):
            indices = np.sort(rng.choice(x.shape[1], size=size, replace=False))
            accuracy = _score(x[:, indices], y)
            rows.append(
                {
                    "draw": draw,
                    "balanced_accuracy": accuracy,
                    "responding_units": [unit_names[index] for index in indices],
                }
            )
        values = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
        result["sizes"][str(size)] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "draws": rows,
        }
    return result


def run_o002_robustness(
    o002_dir: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(o002_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty O002 robustness output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    receipt_path = source / "o002-development-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing O002 v1 receipt: {receipt_path}")
    v1 = _validate_v1_receipt(receipt_path)
    matrix, metadata = _load_inputs(source, v1)
    x, y, classes, class_counts = _eligible_labels(matrix, metadata)
    unit_names = [str(unit) for unit in matrix.columns]

    full_accuracy = _score(x, y)

    amplitudes = _amplitude_only(x)
    mean_accuracy = _score(amplitudes.pop("_mean"), y)
    l2_accuracy = _score(amplitudes.pop("_l2"), y)
    peak_to_peak_accuracy = _score(amplitudes.pop("_peak_to_peak"), y)
    amplitudes["mean_response_balanced_accuracy"] = mean_accuracy
    amplitudes["l2_norm_balanced_accuracy"] = l2_accuracy
    amplitudes["peak_to_peak_balanced_accuracy"] = peak_to_peak_accuracy

    direction_accuracy = _score(_row_l2_normalize(x), y)
    sorted_profile_accuracy = _score(np.sort(x, axis=1), y)
    shuffle = _channel_shuffle_null(x, y)
    shuffle_tail = float(
        (1 + sum(value >= full_accuracy for value in shuffle["scores"]))
        / (CHANNEL_SHUFFLE_COUNT + 1)
    )
    leave_one = _leave_one_unit(x, y, unit_names, full_accuracy)
    subsets = _feature_subsets(x, y, unit_names)

    leave_one_path = output / "o002-v2-leave-one-unit.csv"
    pd.DataFrame(leave_one["per_unit"]).to_csv(leave_one_path, index=False)

    subset_rows: list[dict[str, Any]] = []
    for size, block in subsets["sizes"].items():
        for row in block["draws"]:
            subset_rows.append(
                {
                    "feature_count": int(size),
                    "draw": row["draw"],
                    "balanced_accuracy": row["balanced_accuracy"],
                    "responding_units": ",".join(row["responding_units"]),
                }
            )
    subset_path = output / "o002-v2-feature-subsets.csv"
    pd.DataFrame(subset_rows).to_csv(subset_path, index=False)

    shuffle_path = output / "o002-v2-channel-shuffle-null.csv"
    pd.DataFrame(
        {
            "iteration": np.arange(CHANNEL_SHUFFLE_COUNT),
            "balanced_accuracy": shuffle["scores"],
        }
    ).to_csv(shuffle_path, index=False)

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "o002-coding-robustness-development-v2",
        "program_id": "olfactory-computation-v0",
        "aim": "O002_multi_odor_representation",
        "status": "development_complete_not_confirmatory",
        "development_only": True,
        "confirmatory_use_allowed": False,
        "o003_accessed": False,
        "frozen_question": (
            "Does the observed chemical-class signal require stable responding-unit identity beyond "
            "global response magnitude/profile shape, and is that signal distributed across the "
            "24-channel source representation?"
        ),
        "frozen_before_v2_outcomes": True,
        "input": {
            "v1_receipt_sha256": v1["receipt_sha256"],
            "v1_receipt_file_sha256": sha256_file(receipt_path),
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
        "full_pattern": {
            "balanced_accuracy": full_accuracy,
            "classifier": "leave-one-odor-out nearest centroid after training-fold z-scoring",
        },
        "amplitude_only": amplitudes,
        "direction_only": {
            "transform": "odor-wise L2 normalization before the frozen classifier",
            "balanced_accuracy": direction_accuracy,
        },
        "identity_erased_sorted_profile": {
            "transform": (
                "sort each odor response vector ascending, preserving its exact response multiset "
                "while erasing responding-unit identity"
            ),
            "balanced_accuracy": sorted_profile_accuracy,
        },
        "channel_identity_shuffle_null": {
            key: value for key, value in shuffle.items() if key != "scores"
        }
        | {
            "full_pattern_empirical_tail_fraction": shuffle_tail,
            "null_definition": (
                "independently permute responding-unit positions within each odor; preserves every "
                "odor's exact response multiset while destroying consistent unit identity"
            ),
        },
        "leave_one_unit": {
            key: value for key, value in leave_one.items() if key != "per_unit"
        },
        "feature_subset_robustness": {
            "seed": subsets["seed"],
            "draws_per_size": subsets["draws_per_size"],
            "sizes": {
                size: {key: value for key, value in block.items() if key != "draws"}
                for size, block in subsets["sizes"].items()
            },
        },
        "contrasts": {
            "full_minus_mean_amplitude": full_accuracy - mean_accuracy,
            "full_minus_l2_amplitude": full_accuracy - l2_accuracy,
            "full_minus_direction_only": full_accuracy - direction_accuracy,
            "full_minus_sorted_profile": full_accuracy - sorted_profile_accuracy,
            "full_minus_shuffle_null_mean": full_accuracy - shuffle["mean"],
        },
        "outputs": {
            "leave_one_unit": {
                "path": str(leave_one_path),
                "sha256": sha256_file(leave_one_path),
            },
            "feature_subsets": {
                "path": str(subset_path),
                "sha256": sha256_file(subset_path),
            },
            "channel_shuffle_null": {
                "path": str(shuffle_path),
                "sha256": sha256_file(shuffle_path),
            },
        },
        "claim_boundary": (
            "This v2 analysis is a frozen-after-v1 development robustness decomposition of the observed "
            "within-study chemical-class signal. It can distinguish dependence on overall response "
            "magnitude, response-profile shape, stable responding-unit identity, and distributed feature "
            "support. It is not a new confirmatory test, does not validate odor identity or valence, does "
            "not establish receptor-specific biology, and does not authorize O003/O004."
        ),
    }
    report["receipt_sha256"] = _canonical_sha(report)
    report_path = output / "o002-robustness-receipt.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "O002 CODING ROBUSTNESS DEVELOPMENT V2",
        f"status: {report['status']}",
        f"study: {report['sample']['study_id']}",
        f"eligible_odors: {report['sample']['eligible_odors']}",
        f"responding_units: {report['sample']['responding_units']}",
        f"full_pattern_balanced_accuracy: {full_accuracy:.6f}",
        f"mean_amplitude_only_accuracy: {mean_accuracy:.6f}",
        f"l2_amplitude_only_accuracy: {l2_accuracy:.6f}",
        f"peak_to_peak_only_accuracy: {peak_to_peak_accuracy:.6f}",
        f"direction_only_accuracy: {direction_accuracy:.6f}",
        f"identity_erased_sorted_profile_accuracy: {sorted_profile_accuracy:.6f}",
        f"channel_shuffle_null_mean: {shuffle['mean']:.6f}",
        f"channel_shuffle_null_q95: {shuffle['q95']:.6f}",
        f"full_pattern_shuffle_tail_fraction: {shuffle_tail:.6f}",
        f"leave_one_unit_min_accuracy: {leave_one['minimum']:.6f}",
        f"leave_one_unit_mean_accuracy: {leave_one['mean']:.6f}",
        f"receipt_sha256: {report['receipt_sha256']}",
        "",
        "CLAIM BOUNDARY",
        report["claim_boundary"],
    ]
    (output / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    return report
