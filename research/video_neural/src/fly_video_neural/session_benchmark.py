from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import MeanTargetBaseline, metric_for_selection, summarize_metrics
from .ridge import RidgeDecoder

SPLIT_SEED = 2701
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class SessionBenchmarkBatch:
    sample_ids: np.ndarray
    animal_ids: np.ndarray
    session_ids: np.ndarray
    features: np.ndarray
    targets: np.ndarray

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids).astype(str)
        animal_ids = np.asarray(self.animal_ids).astype(str)
        session_ids = np.asarray(self.session_ids).astype(str)
        features = np.asarray(self.features, dtype=float)
        targets = np.asarray(self.targets, dtype=float)
        n_samples = len(sample_ids)
        if any(array.ndim != 1 for array in (sample_ids, animal_ids, session_ids)):
            raise ValueError("sample, animal, and session ids must be one-dimensional")
        if len(animal_ids) != n_samples or len(session_ids) != n_samples:
            raise ValueError("metadata arrays must share the sample axis")
        if (
            features.ndim != 2
            or targets.ndim != 2
            or features.shape[0] != n_samples
            or targets.shape[0] != n_samples
        ):
            raise ValueError("features and targets must be aligned 2D arrays")
        if len(set(sample_ids.tolist())) != n_samples:
            raise ValueError("sample ids must be unique")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("features and targets must be finite")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "animal_ids", animal_ids)
        object.__setattr__(self, "session_ids", session_ids)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_session_batches(
    paths: list[str | Path],
) -> tuple[SessionBenchmarkBatch, list[dict[str, str]]]:
    if not paths:
        raise ValueError("at least one session batch is required")
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    file_receipts: list[dict[str, str]] = []
    feature_dim: int | None = None
    target_dim: int | None = None
    names = ("sample_ids", "animal_ids", "session_ids", "features", "targets")
    for path in paths:
        batch_path = Path(path)
        with np.load(batch_path, allow_pickle=False) as payload:
            missing = set(names) - set(payload.files)
            if missing:
                raise ValueError(f"session batch missing arrays: {sorted(missing)}")
            chunk = tuple(np.asarray(payload[name]) for name in names)
        _, animals, sessions, features, targets = chunk
        if len(set(np.asarray(animals).astype(str).tolist())) != 1:
            raise ValueError(f"batch {batch_path} must contain exactly one animal")
        if len(set(np.asarray(sessions).astype(str).tolist())) != 1:
            raise ValueError(f"batch {batch_path} must contain exactly one session")
        feature_dim = feature_dim or int(features.shape[1])
        target_dim = target_dim or int(targets.shape[1])
        if features.shape[1] != feature_dim or targets.shape[1] != target_dim:
            raise ValueError("all session batches must share feature and target dimensions")
        chunks.append(chunk)
        file_receipts.append({"path": str(batch_path.resolve()), "sha256": _file_sha(batch_path)})
    batch = SessionBenchmarkBatch(
        sample_ids=np.concatenate([chunk[0] for chunk in chunks]),
        animal_ids=np.concatenate([chunk[1] for chunk in chunks]),
        session_ids=np.concatenate([chunk[2] for chunk in chunks]),
        features=np.concatenate([chunk[3] for chunk in chunks]),
        targets=np.concatenate([chunk[4] for chunk in chunks]),
    )
    return batch, sorted(file_receipts, key=lambda row: row["path"])


def build_session_split_lock(
    batch: SessionBenchmarkBatch,
    *,
    source_batches: list[dict[str, str]] | None = None,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    if seed != SPLIT_SEED:
        raise ValueError(f"v1 session split seed is frozen to {SPLIT_SEED}")
    session_owner: dict[str, str] = {}
    for animal_value, session_value in zip(batch.animal_ids, batch.session_ids, strict=True):
        animal = str(animal_value)
        session = str(session_value)
        previous = session_owner.setdefault(session, animal)
        if previous != animal:
            raise ValueError(f"session {session!r} appears under multiple animals")
    animal_sessions: dict[str, list[str]] = {}
    for session, animal in session_owner.items():
        animal_sessions.setdefault(animal, []).append(session)
    assignments: dict[str, str] = {}
    by_animal: dict[str, dict[str, list[str]]] = {}
    for animal, sessions in sorted(animal_sessions.items()):
        if len(sessions) < 3:
            raise ValueError(f"animal {animal!r} needs at least three sessions")
        ranked = sorted(
            sessions,
            key=lambda session: hashlib.sha256(f"{seed}:{animal}:{session}".encode()).hexdigest(),
        )
        validation = ranked[0]
        test = ranked[1]
        train = ranked[2:]
        by_animal[animal] = {
            "train": sorted(train),
            "validation": [validation],
            "test": [test],
        }
        assignments.update({session: "train" for session in train})
        assignments[validation] = "validation"
        assignments[test] = "test"
    sample_to_split = {
        str(sample): assignments[str(session)]
        for sample, session in zip(batch.sample_ids, batch.session_ids, strict=True)
    }
    sample_to_animal = {
        str(sample): str(animal)
        for sample, animal in zip(batch.sample_ids, batch.animal_ids, strict=True)
    }
    source_rows = sorted(
        (
            {"sample_id": str(sample), "animal_id": str(animal), "session_id": str(session)}
            for sample, animal, session in zip(
                batch.sample_ids, batch.animal_ids, batch.session_ids, strict=True
            )
        ),
        key=lambda row: row["sample_id"],
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "mc2p_future_neural_v1",
        "split_unit": "session_id_within_animal",
        "split_seed": seed,
        "source_samples_sha256": _sha(source_rows),
        "source_batches": sorted(source_batches or [], key=lambda row: row["path"]),
        "animal_sessions": by_animal,
        "sample_to_split": dict(sorted(sample_to_split.items())),
        "sample_to_animal": dict(sorted(sample_to_animal.items())),
        "validation_sessions_per_animal": 1,
        "test_sessions_per_animal": 1,
        "test_sessions_for_hyperparameter_selection": False,
    }
    payload["split_lock_sha256"] = _sha(payload)
    return payload


def _verify_lock_self_hash(lock: dict[str, Any]) -> None:
    supplied = dict(lock)
    claimed = supplied.pop("split_lock_sha256", None)
    if claimed != _sha(supplied):
        raise ValueError("session split lock self-hash mismatch")


def verify_session_split_lock(
    lock: dict[str, Any],
    batch: SessionBenchmarkBatch,
    *,
    source_batches: list[dict[str, str]] | None = None,
) -> None:
    _verify_lock_self_hash(lock)
    supplied = dict(lock)
    supplied.pop("split_lock_sha256", None)
    rebuilt = build_session_split_lock(
        batch,
        source_batches=source_batches,
        seed=lock["split_seed"],
    )
    expected = dict(rebuilt)
    expected.pop("split_lock_sha256")
    if expected != supplied:
        raise ValueError("session split lock does not match batch inputs or assignments")


def verify_development_projection(
    lock: dict[str, Any],
    batch: SessionBenchmarkBatch,
    *,
    source_batches: list[dict[str, str]] | None = None,
) -> None:
    """Verify an exact train+validation projection without requiring test arrays in memory."""
    _verify_lock_self_hash(lock)
    sample_to_split = lock["sample_to_split"]
    sample_to_animal = lock["sample_to_animal"]
    expected_samples = {
        sample for sample, split_name in sample_to_split.items() if split_name in {"train", "validation"}
    }
    actual_samples = set(batch.sample_ids.tolist())
    if actual_samples != expected_samples:
        missing = sorted(expected_samples - actual_samples)
        extra = sorted(actual_samples - expected_samples)
        raise ValueError(
            "development projection does not exactly match frozen train+validation samples; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    expected_sessions: set[str] = set()
    for animal, assignments in lock["animal_sessions"].items():
        expected_sessions.update(assignments["train"])
        expected_sessions.update(assignments["validation"])
        if set(assignments["test"]) & set(batch.session_ids.tolist()):
            raise ValueError("development projection contains a held-out test session")
    if set(batch.session_ids.tolist()) != expected_sessions:
        raise ValueError("development projection session set does not match frozen train+validation sessions")

    for sample, animal, session in zip(
        batch.sample_ids, batch.animal_ids, batch.session_ids, strict=True
    ):
        sample_id = str(sample)
        animal_id = str(animal)
        session_id = str(session)
        split_name = sample_to_split.get(sample_id)
        if split_name not in {"train", "validation"}:
            raise ValueError(f"development projection contains non-development sample {sample_id!r}")
        if sample_to_animal.get(sample_id) != animal_id:
            raise ValueError(f"development sample {sample_id!r} changed animal identity")
        allowed_sessions = set(lock["animal_sessions"][animal_id][split_name])
        if session_id not in allowed_sessions:
            raise ValueError(f"development sample {sample_id!r} changed session identity")

    if source_batches is not None:
        frozen_sources = {
            (row["path"], row["sha256"]) for row in lock.get("source_batches", [])
        }
        supplied_sources = {(row["path"], row["sha256"]) for row in source_batches}
        if not supplied_sources <= frozen_sources:
            raise ValueError("development source-batch identity is not a subset of the frozen split lock")
        if len(source_batches) != len(expected_sessions):
            raise ValueError("development source-batch count does not match development session count")


def subset_development_batch(
    batch: SessionBenchmarkBatch,
    lock: dict[str, Any],
) -> SessionBenchmarkBatch:
    _verify_lock_self_hash(lock)
    sample_to_split = lock["sample_to_split"]
    mask = np.asarray(
        [sample_to_split[str(sample)] in {"train", "validation"} for sample in batch.sample_ids],
        dtype=bool,
    )
    return SessionBenchmarkBatch(
        sample_ids=batch.sample_ids[mask],
        animal_ids=batch.animal_ids[mask],
        session_ids=batch.session_ids[mask],
        features=batch.features[mask],
        targets=batch.targets[mask],
    )


def _finite_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        metrics = row["test_metrics"]
        value = None if metrics is None else metrics.get(metric)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return values


def _aggregate(values: list[float], operation: str) -> float | None:
    if not values:
        return None
    if operation == "median":
        return float(np.median(values))
    if operation == "mean":
        return float(np.mean(values))
    raise ValueError(f"unsupported aggregate operation {operation!r}")


def run_within_animal_ridge(
    batch: SessionBenchmarkBatch,
    lock: dict[str, Any],
    *,
    source_batches: list[dict[str, str]] | None = None,
    consume_test: bool = False,
) -> dict[str, Any]:
    if consume_test:
        verify_session_split_lock(lock, batch, source_batches=source_batches)
    else:
        verify_development_projection(lock, batch, source_batches=source_batches)
    sample_to_split = lock["sample_to_split"]
    sample_to_animal = lock["sample_to_animal"]
    animals = sorted(lock["animal_sessions"])
    rows: list[dict[str, Any]] = []
    for animal in animals:
        animal_mask = np.array(
            [sample_to_animal[str(sample)] == animal for sample in batch.sample_ids], dtype=bool
        )
        train = animal_mask & np.array(
            [sample_to_split[str(sample)] == "train" for sample in batch.sample_ids], dtype=bool
        )
        validation = animal_mask & np.array(
            [sample_to_split[str(sample)] == "validation" for sample in batch.sample_ids],
            dtype=bool,
        )
        test = animal_mask & np.array(
            [sample_to_split[str(sample)] == "test" for sample in batch.sample_ids], dtype=bool
        )
        if min(train.sum(), validation.sum()) < 2:
            raise ValueError(f"animal {animal!r} needs at least two windows in train and validation")
        if consume_test and test.sum() < 2:
            raise ValueError(f"animal {animal!r} needs at least two windows in test")
        candidates: list[dict[str, Any]] = []
        for alpha in RIDGE_ALPHAS:
            model = RidgeDecoder(alpha).fit(batch.features[train], batch.targets[train])
            metrics = summarize_metrics(
                batch.targets[validation], model.predict(batch.features[validation])
            )
            candidates.append({"alpha": alpha, "validation": metrics})
        selected = max(
            candidates,
            key=lambda row: metric_for_selection(row["validation"], "median_pearson_r"),
        )
        animal_report: dict[str, Any] = {
            "animal_id": animal,
            "selected_alpha": selected["alpha"],
            "selected_validation_metrics": selected["validation"],
            "alpha_candidates": candidates,
            "test_status": "locked_not_consumed",
            "test_metrics": None,
            "mean_baseline_test_metrics": None,
        }
        if consume_test:
            development = train | validation
            model = RidgeDecoder(selected["alpha"]).fit(
                batch.features[development], batch.targets[development]
            )
            mean = MeanTargetBaseline().fit(batch.targets[development])
            animal_report["test_status"] = "consumed_explicitly"
            animal_report["test_metrics"] = summarize_metrics(
                batch.targets[test], model.predict(batch.features[test])
            )
            animal_report["mean_baseline_test_metrics"] = summarize_metrics(
                batch.targets[test], mean.predict(int(test.sum()))
            )
        rows.append(animal_report)
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "mc2p_future_neural_v1",
        "model": "within_animal_ridge_decoder",
        "split_lock_sha256": lock["split_lock_sha256"],
        "selection_metric": "median_pearson_r",
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "test_status": "consumed_explicitly" if consume_test else "locked_not_consumed",
        "animals": rows,
        "aggregate_test_metrics": None,
        "claim_boundary": (
            "Models are fit independently within each animal. "
            "No raw neural pixel correspondence is assumed across animals."
        ),
    }
    if consume_test:
        correlations = _finite_values(rows, "median_pearson_r")
        r2_values = _finite_values(rows, "median_r2")
        report["aggregate_test_metrics"] = {
            "animal_count": len(rows),
            "scorable_correlation_animals": len(correlations),
            "scorable_r2_animals": len(r2_values),
            "median_of_animal_median_pearson_r": _aggregate(correlations, "median"),
            "mean_of_animal_median_pearson_r": _aggregate(correlations, "mean"),
            "median_of_animal_median_r2": _aggregate(r2_values, "median"),
            "mean_of_animal_median_r2": _aggregate(r2_values, "mean"),
        }
    return report
