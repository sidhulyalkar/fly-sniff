from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .olfactory_door import sha256_file
from .olfactory_e006_audit import _canonical_sha

MIN_COMPLETE_ODORS = 25
MIN_CLASS_COUNT = 3
PERMUTATION_COUNT = 512
PERMUTATION_SEED = 24017


def _validate_audit(path: Path) -> dict[str, Any]:
    audit = json.loads(path.read_text())
    if audit.get("protocol") != "door-e006-audit-v1":
        raise ValueError("O002 development requires a door-e006-audit-v1 artifact")
    observed = str(audit.get("audit_sha256", ""))
    unhashed = dict(audit)
    unhashed.pop("audit_sha256", None)
    if observed != _canonical_sha(unhashed):
        raise ValueError("E006 audit canonical hash mismatch")
    gate = audit.get("gate", {})
    if gate.get("within_study_development_subset_allowed") is not True:
        raise ValueError("E006 audit does not allow a within-study development subset")
    if gate.get("confirmatory_use_allowed") is not False:
        raise ValueError("E006 audit unexpectedly permits confirmatory use")
    if gate.get("development_feature_identity") != "source_responding_unit":
        raise ValueError("O002 v1 requires frozen source responding-unit feature identity")
    return audit


def _select_subset(audit: dict[str, Any], study_id: str | None) -> dict[str, Any]:
    subsets = audit["development_subsets"]
    candidates = subsets.get("candidates", [])
    by_id = {str(row["study_id"]): row for row in candidates}
    if study_id is None:
        selected = subsets.get("default_development_subset")
        if not selected:
            raise ValueError("E006 audit has no default development subset")
        study_id = str(selected["study_id"])
    if study_id not in by_id:
        raise ValueError(
            f"study {study_id!r} is not a frozen performance-blind development candidate"
        )
    return dict(by_id[study_id])


def _load_study_matrix(
    long_path: Path,
    study_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(long_path)
    required = {
        "responding_unit",
        "source_row_id",
        "odor_class",
        "odor_name",
        "inchikey",
        "cid",
        "cas",
        "study_id",
        "response_status",
        "response_value",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"E006 long-form artifact lacks required O002 columns: {missing}")

    study = df[df["study_id"].eq(study_id)].copy()
    if study.empty:
        raise ValueError(f"selected O002 study is absent from E006: {study_id}")

    observed = study[
        study["response_status"].eq("observed")
        & study["response_value"].notna()
        & study["odor_name"].notna()
    ].copy()
    observed["odor_name_normalized"] = observed["odor_name"].astype(str).str.casefold()
    sfr_rows = observed["odor_name_normalized"].eq("sfr")
    sfr_observation_cells = int(sfr_rows.sum())
    observed = observed[~sfr_rows].copy()
    observed["response_value"] = pd.to_numeric(observed["response_value"], errors="raise")

    duplicate = observed.duplicated(["source_row_id", "responding_unit"], keep=False)
    if duplicate.any():
        examples = observed.loc[
            duplicate, ["source_row_id", "responding_unit", "odor_name"]
        ].head(10)
        raise ValueError(
            "O002 study contains duplicate source-row/responding-unit observations: "
            + examples.to_dict(orient="records").__repr__()
        )

    matrix = observed.pivot(
        index="source_row_id",
        columns="responding_unit",
        values="response_value",
    ).sort_index(axis=0).sort_index(axis=1)

    metadata_fields = ["source_row_id", "odor_class", "odor_name", "inchikey", "cid", "cas"]
    metadata = observed[metadata_fields].drop_duplicates()
    consistency = metadata.groupby("source_row_id").size()
    inconsistent = consistency[consistency.ne(1)]
    if not inconsistent.empty:
        raise ValueError(
            "odor metadata are inconsistent across responding units for source rows: "
            + ", ".join(map(str, inconsistent.index.tolist()[:10]))
        )
    metadata = metadata.set_index("source_row_id").reindex(matrix.index)

    complete = matrix.dropna(axis=0, how="any")
    if len(complete) < MIN_COMPLETE_ODORS:
        raise ValueError(
            f"O002 requires at least {MIN_COMPLETE_ODORS} complete odor rows; got {len(complete)}"
        )
    metadata = metadata.reindex(complete.index)

    diagnostics = {
        "study_id": study_id,
        "source_cells": len(study),
        "observed_cells_including_sfr": int(
            (
                study["response_status"].eq("observed")
                & study["response_value"].notna()
                & study["odor_name"].notna()
            ).sum()
        ),
        "sfr_observation_cells_excluded": sfr_observation_cells,
        "responding_units": int(matrix.shape[1]),
        "odor_rows_with_any_observation_excluding_sfr": int(matrix.shape[0]),
        "complete_odor_rows_excluding_sfr": int(complete.shape[0]),
        "incomplete_odor_rows_excluding_sfr": int(matrix.shape[0] - complete.shape[0]),
        "complete_matrix_cells": int(complete.shape[0] * complete.shape[1]),
        "complete_matrix_missing_cells": int(complete.isna().sum().sum()),
        "feature_identity": "source_responding_unit",
        "missing_value_policy": "complete_odor_rows_only; no imputation and no zero filling",
        "sfr_policy": "exclude spontaneous-firing-rate pseudo-odor from odor representation analysis",
    }
    return complete, metadata, diagnostics


def _pca_metrics(matrix: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    x = matrix.to_numpy(dtype=float)
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    eigen = singular**2
    total = float(eigen.sum())
    explained = eigen / total if total > 0 else np.zeros_like(eigen)
    effective_rank = (
        float(total**2 / np.square(eigen).sum())
        if total > 0 and float(np.square(eigen).sum()) > 0
        else 0.0
    )
    scores = centered @ vt.T
    columns = [f"PC{i + 1}" for i in range(scores.shape[1])]
    coords = pd.DataFrame(scores, index=matrix.index, columns=columns)
    return (
        {
            "centered_not_zscored": True,
            "components": len(explained),
            "explained_variance_ratio_top10": [
                float(value) for value in explained[:10]
            ],
            "cumulative_variance_top2": float(explained[:2].sum()),
            "cumulative_variance_top5": float(explained[:5].sum()),
            "effective_rank_participation_ratio": effective_rank,
        },
        coords,
    )


def _geometry_stability(matrix: pd.DataFrame) -> dict[str, Any]:
    x = matrix.to_numpy(dtype=float)
    if x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("O002 geometry stability requires >=3 odors and >=2 responding units")
    full = pdist(x, metric="euclidean")
    correlations: list[dict[str, Any]] = []
    for index, unit in enumerate(matrix.columns):
        reduced = np.delete(x, index, axis=1)
        distance = pdist(reduced, metric="euclidean")
        rho = spearmanr(full, distance).statistic
        correlations.append({"responding_unit": str(unit), "spearman_rho": float(rho)})
    values = np.array([row["spearman_rho"] for row in correlations], dtype=float)
    worst = min(correlations, key=lambda row: row["spearman_rho"])
    best = max(correlations, key=lambda row: row["spearman_rho"])
    return {
        "metric": "Spearman correlation of full vs leave-one-responding-unit-out Euclidean odor distances",
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "worst_unit": worst,
        "best_unit": best,
        "per_unit": correlations,
    }


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls: list[float] = []
    for label in sorted(set(y_true.tolist())):
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _loo_nearest_centroid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    prediction: list[str] = []
    for test_index in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_index] = False
        x_train = x[train_mask]
        y_train = y[train_mask]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        train_z = (x_train - mean) / std
        test_z = (x[test_index] - mean) / std
        labels = sorted(set(y_train.tolist()))
        centroids = {
            label: train_z[y_train == label].mean(axis=0)
            for label in labels
        }
        prediction.append(
            min(
                labels,
                key=lambda label: float(np.square(test_z - centroids[label]).sum()),
            )
        )
    return np.asarray(prediction, dtype=object)


def _chemical_class_probe(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    labels = metadata["odor_class"].fillna("").astype(str).str.strip()
    counts = labels[labels.ne("")].value_counts()
    eligible_classes = sorted(counts[counts.ge(MIN_CLASS_COUNT)].index.tolist())
    mask = labels.isin(eligible_classes)
    x = matrix.loc[mask].to_numpy(dtype=float)
    y = labels.loc[mask].to_numpy(dtype=object)
    if len(eligible_classes) < 2 or len(y) < 10:
        return {
            "status": "blocked_insufficient_source_labels",
            "eligible_classes": eligible_classes,
            "eligible_odors": len(y),
        }

    prediction = _loo_nearest_centroid(x, y)
    observed_score = _balanced_accuracy(y, prediction)
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.empty(PERMUTATION_COUNT, dtype=float)
    for i in range(PERMUTATION_COUNT):
        permuted = rng.permutation(y)
        null_prediction = _loo_nearest_centroid(x, permuted)
        null[i] = _balanced_accuracy(permuted, null_prediction)

    empirical_tail = float((1 + np.sum(null >= observed_score)) / (PERMUTATION_COUNT + 1))
    return {
        "status": "exploratory_development_only",
        "target": "source odor_class metadata; not odor identity and not behavioral valence",
        "classifier": "leave-one-odor-out nearest centroid after training-fold z-scoring",
        "minimum_class_count": MIN_CLASS_COUNT,
        "eligible_classes": eligible_classes,
        "class_counts": {label: int(counts[label]) for label in eligible_classes},
        "eligible_odors": int(len(y)),
        "balanced_accuracy": observed_score,
        "permutation_count": PERMUTATION_COUNT,
        "permutation_seed": PERMUTATION_SEED,
        "permutation_null_mean": float(null.mean()),
        "permutation_null_q95": float(np.quantile(null, 0.95)),
        "empirical_tail_fraction": empirical_tail,
        "claim_boundary": (
            "This exploratory probe asks whether source chemical-class labels are geometrically "
            "decodable within one assay. It is not an odor-identity, valence, behavior, topology, "
            "or confirmatory biological-mechanism result."
        ),
    }


def run_o002_development(
    artifact_dir: str | Path,
    *,
    audit_path: str | Path,
    output_dir: str | Path,
    study_id: str | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact_dir).expanduser().resolve()
    audit_file = Path(audit_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty O002 development output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    audit = _validate_audit(audit_file)
    selected = _select_subset(audit, study_id)
    long_path = artifact / "door-responses-long.csv"
    if not long_path.is_file():
        raise FileNotFoundError(f"missing E006 long-form artifact: {long_path}")
    expected_long_hash = str(audit["input_receipt"]["long_form_sha256"])
    if sha256_file(long_path) != expected_long_hash:
        raise ValueError("O002 E006 long-form hash does not match the audited input")

    matrix, metadata, matrix_diagnostics = _load_study_matrix(
        long_path, str(selected["study_id"])
    )
    pca, coordinates = _pca_metrics(matrix)
    geometry = _geometry_stability(matrix)
    class_probe = _chemical_class_probe(matrix, metadata)

    matrix_path = output / "o002-complete-response-matrix.csv"
    metadata_path = output / "o002-odor-metadata.csv"
    coordinates_path = output / "o002-pca-coordinates.csv"
    matrix.to_csv(matrix_path)
    metadata.to_csv(metadata_path)
    coordinates.to_csv(coordinates_path)

    limitations = {
        "odor_identity_decoding": (
            "not estimated: the selected DoOR study supplies one aggregate response vector per odor, "
            "not independent replicate response vectors for held-out identity decoding"
        ),
        "concentration_generalization": (
            "not estimated: the selected development subset has one nominal source concentration "
            f"({selected['concentration']})"
        ),
        "valence_decoding": "not estimated: behavioral valence labels are not part of this source authority",
        "cross_study_generalization": "not estimated: E006 cross-study comparability remains unqualified",
        "receptor_specific_claims": (
            "not allowed: features retain source responding-unit IDs and ambiguous receptor mappings "
            "are not collapsed"
        ),
    }

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "o002-within-study-development-v1",
        "program_id": "olfactory-computation-v0",
        "aim": "O002_multi_odor_representation",
        "status": "development_complete_not_confirmatory",
        "development_only": True,
        "confirmatory_use_allowed": False,
        "o003_accessed": False,
        "selected_study": selected,
        "selection_policy": (
            "Study selection was frozen from E006 source coverage and assay metadata before any O002 "
            "representation metric was computed."
        ),
        "feature_identity": "source_responding_unit",
        "input": {
            "e006_audit_sha256": audit["audit_sha256"],
            "e006_audit_file_sha256": sha256_file(audit_file),
            "e006_long_form_sha256": expected_long_hash,
        },
        "matrix": matrix_diagnostics,
        "pca": pca,
        "geometry_stability": geometry,
        "chemical_class_probe": class_probe,
        "limitations": limitations,
        "outputs": {
            "matrix": {"path": str(matrix_path), "sha256": sha256_file(matrix_path)},
            "odor_metadata": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
            "pca_coordinates": {
                "path": str(coordinates_path),
                "sha256": sha256_file(coordinates_path),
            },
        },
        "claim_boundary": (
            "This is an exploratory, within-study O002 representation analysis using a source-coverage-"
            "selected electrophysiology dataset. It can characterize geometry and robustness of the "
            "measured response matrix. It cannot qualify E006 globally, establish receptor identity for "
            "ambiguous units, support cross-study normalization, establish behavioral valence, test "
            "connectome topology, or authorize O003/O004 confirmatory claims."
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path = output / "o002-development-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    summary = [
        "O002 WITHIN-STUDY DEVELOPMENT V1",
        f"status: {receipt['status']}",
        f"study: {selected['study_id']}",
        f"nominal_concentration: {selected['concentration']}",
        f"responding_units: {matrix_diagnostics['responding_units']}",
        f"complete_odor_rows_excluding_sfr: {matrix_diagnostics['complete_odor_rows_excluding_sfr']}",
        f"incomplete_odor_rows_excluding_sfr: {matrix_diagnostics['incomplete_odor_rows_excluding_sfr']}",
        f"pca_top2_variance: {pca['cumulative_variance_top2']:.6f}",
        f"pca_effective_rank: {pca['effective_rank_participation_ratio']:.6f}",
        f"geometry_leave_one_unit_mean_rho: {geometry['mean']:.6f}",
        f"chemical_class_probe_status: {class_probe['status']}",
    ]
    if class_probe.get("status") == "exploratory_development_only":
        summary.extend(
            [
                f"chemical_class_balanced_accuracy: {class_probe['balanced_accuracy']:.6f}",
                f"chemical_class_null_q95: {class_probe['permutation_null_q95']:.6f}",
                f"chemical_class_empirical_tail_fraction: {class_probe['empirical_tail_fraction']:.6f}",
            ]
        )
    summary.extend(
        [
            f"receipt_sha256: {receipt['receipt_sha256']}",
            "",
            "LIMITATIONS",
            *[f"- {key}: {value}" for key, value in limitations.items()],
            "",
            "CLAIM BOUNDARY",
            receipt["claim_boundary"],
        ]
    )
    (output / "SUMMARY.txt").write_text("\n".join(summary) + "\n")
    return receipt
