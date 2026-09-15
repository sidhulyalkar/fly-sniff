from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import summarize_metrics
from .ridge import RidgeDecoder
from .session_benchmark import (
    RIDGE_ALPHAS,
    SessionBenchmarkBatch,
    load_session_batches,
    verify_session_split_lock,
)

SAMPLE_START = re.compile(r".+:(?P<start>\d+)-\d+$")
NULL_NAME = "circular_quartile_session_feature_shift_ensemble"
NULL_FRACTIONS = (0.25, 0.5, 0.75)
NULL_SELECTION_RULE = "strongest_validation_median_pearson_r_per_animal"


def _sample_start(sample_id: str) -> int:
    match = SAMPLE_START.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"sample id does not expose behavior-frame start: {sample_id!r}")
    return int(match.group("start"))


def circular_session_fraction_shift(
    batch: SessionBenchmarkBatch,
    mask: np.ndarray,
    fraction: float,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(batch.sample_ids),):
        raise ValueError("alignment-null mask must match batch sample axis")
    if fraction not in NULL_FRACTIONS:
        raise ValueError(f"alignment-null fraction must be one of {NULL_FRACTIONS}")
    indices = np.flatnonzero(mask)
    shifted = np.empty((len(indices), batch.features.shape[1]), dtype=float)
    output_position = {int(index): position for position, index in enumerate(indices)}
    for session in sorted(set(batch.session_ids[mask].tolist())):
        session_indices = indices[batch.session_ids[indices] == session]
        if len(session_indices) < 2:
            raise ValueError(f"alignment null requires at least two windows in session {session!r}")
        starts = np.asarray([_sample_start(str(batch.sample_ids[index])) for index in session_indices])
        chronological = session_indices[np.argsort(starts)]
        offset = int(round(len(chronological) * fraction))
        offset = min(max(1, offset), len(chronological) - 1)
        source = np.roll(chronological, offset)
        for destination_index, source_index in zip(chronological, source, strict=True):
            shifted[output_position[int(destination_index)]] = batch.features[int(source_index)]
    return shifted


def circular_half_session_shift(
    batch: SessionBenchmarkBatch,
    mask: np.ndarray,
) -> np.ndarray:
    return circular_session_fraction_shift(batch, mask, 0.5)


def _mask_for(
    batch: SessionBenchmarkBatch,
    lock: dict[str, Any],
    animal: str,
    split_name: str,
) -> np.ndarray:
    sample_to_split = lock["sample_to_split"]
    sample_to_animal = lock["sample_to_animal"]
    return np.asarray(
        [
            sample_to_animal[str(sample)] == animal
            and sample_to_split[str(sample)] == split_name
            for sample in batch.sample_ids
        ],
        dtype=bool,
    )


def _run_fraction(
    batch: SessionBenchmarkBatch,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    fraction: float,
    *,
    consume_test: bool,
) -> dict[str, Any]:
    train_x = circular_session_fraction_shift(batch, train, fraction)
    validation_x = circular_session_fraction_shift(batch, validation, fraction)
    train_y = batch.targets[train]
    validation_y = batch.targets[validation]
    candidates: list[dict[str, Any]] = []
    for alpha in RIDGE_ALPHAS:
        model = RidgeDecoder(alpha).fit(train_x, train_y)
        metrics = summarize_metrics(validation_y, model.predict(validation_x))
        candidates.append({"alpha": alpha, "validation": metrics})
    selected = max(candidates, key=lambda row: row["validation"]["median_pearson_r"])
    row: dict[str, Any] = {
        "fraction": fraction,
        "selected_alpha": selected["alpha"],
        "selected_validation_metrics": selected["validation"],
        "alpha_candidates": candidates,
        "test_status": "locked_not_consumed",
        "test_metrics": None,
    }
    if consume_test:
        development = train | validation
        development_x = circular_session_fraction_shift(batch, development, fraction)
        test_x = circular_session_fraction_shift(batch, test, fraction)
        model = RidgeDecoder(selected["alpha"]).fit(
            development_x,
            batch.targets[development],
        )
        row["test_status"] = "consumed_explicitly"
        row["test_metrics"] = summarize_metrics(
            batch.targets[test],
            model.predict(test_x),
        )
    return row


def run_alignment_null(
    batch: SessionBenchmarkBatch,
    lock: dict[str, Any],
    *,
    source_batches: list[dict[str, str]] | None = None,
    consume_test: bool = False,
) -> dict[str, Any]:
    verify_session_split_lock(lock, batch, source_batches=source_batches)
    rows: list[dict[str, Any]] = []
    for animal in sorted(set(batch.animal_ids.tolist())):
        train = _mask_for(batch, lock, animal, "train")
        validation = _mask_for(batch, lock, animal, "validation")
        test = _mask_for(batch, lock, animal, "test")
        if min(train.sum(), validation.sum(), test.sum()) < 2:
            raise ValueError(f"animal {animal!r} needs at least two windows in every split")
        variants = [
            _run_fraction(
                batch,
                train,
                validation,
                test,
                fraction,
                consume_test=consume_test,
            )
            for fraction in NULL_FRACTIONS
        ]
        selected = max(
            variants,
            key=lambda row: row["selected_validation_metrics"]["median_pearson_r"],
        )
        rows.append(
            {
                "animal_id": animal,
                "selected_null_fraction": selected["fraction"],
                "selected_alpha": selected["selected_alpha"],
                "selected_validation_metrics": selected["selected_validation_metrics"],
                "alpha_candidates": selected["alpha_candidates"],
                "null_candidates": variants,
                "test_status": selected["test_status"],
                "test_metrics": selected["test_metrics"],
            }
        )
    return {
        "schema_version": 1,
        "benchmark_id": "mc2p_future_neural_v1",
        "model": "within_animal_ridge_alignment_null_ensemble",
        "null_name": NULL_NAME,
        "null_fractions": list(NULL_FRACTIONS),
        "null_selection_rule": NULL_SELECTION_RULE,
        "split_lock_sha256": lock["split_lock_sha256"],
        "selection_metric": "median_pearson_r",
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "test_status": "consumed_explicitly" if consume_test else "locked_not_consumed",
        "animals": rows,
        "claim_boundary": (
            "Pose features are circularly shifted by 25%, 50%, and 75% within each session. "
            "Each shift preserves the session-level feature multiset while breaking contemporaneous "
            "behavior-neural pairing. The primary null fraction for each animal is selected only on "
            "validation median Pearson r; final scoring never chooses a null fraction from test scores."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the development-only frozen within-session alignment-null ensemble"
    )
    parser.add_argument("split_lock")
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    batch, source_batches = load_session_batches(args.batches)
    lock = json.loads(Path(args.split_lock).read_text())
    report = run_alignment_null(
        batch,
        lock,
        source_batches=source_batches,
        consume_test=False,
    )
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": report["test_status"], "animals": len(report["animals"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
