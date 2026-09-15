from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .alignment_null import run_alignment_null
from .final_inference import evaluate_final_inference
from .mc2p_legacy import sha256_file
from .provenance import implementation_fingerprint, runtime_fingerprint
from .session_benchmark import (
    load_session_batches,
    run_within_animal_ridge,
    verify_session_split_lock,
)
from .validation_gate import build_validation_unlock, load_acceptance_config

FINAL_LOCK_NAME = "FINAL_TEST_CONSUMED.json"


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _verify_self_hash(document: dict[str, Any], key: str) -> str:
    supplied = dict(document)
    claimed = supplied.pop(key, None)
    if not isinstance(claimed, str) or claimed != _sha(supplied):
        raise ValueError(f"{key} self-hash mismatch")
    return claimed


def _fresh_output_dir(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty final directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _load_development_bundle(directory: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(directory)
    filenames = {
        "receipt": "development-receipt.json",
        "qc": "development-qc.json",
        "aligned": "aligned-ridge-development.json",
        "null": "temporal-null-development.json",
        "unlock": "validation-unlock.json",
    }
    missing = [filename for filename in filenames.values() if not (root / filename).is_file()]
    if missing:
        raise ValueError(f"development directory is missing required artifacts: {missing}")
    return {name: _load_json(root / filename) for name, filename in filenames.items()}


def validate_final_authorization(
    development_dir: str | Path,
    split_lock: dict[str, Any],
    acceptance_config: dict[str, Any],
) -> dict[str, Any]:
    bundle = _load_development_bundle(development_dir)
    receipt = bundle["receipt"]
    qc = bundle["qc"]
    saved_unlock = bundle["unlock"]
    _verify_self_hash(receipt, "receipt_sha256")
    _verify_self_hash(qc, "report_sha256")
    _verify_self_hash(saved_unlock, "report_sha256")
    if receipt.get("test_consumption_capability") is not False:
        raise ValueError("development receipt unexpectedly had test-consumption capability")
    if receipt.get("test_metrics_present") is not False:
        raise ValueError("development receipt indicates test metrics were present")
    if receipt.get("test_target_arrays_deserialized") is not False:
        raise ValueError("development receipt did not preserve the no-test-deserialization boundary")
    if qc.get("test_target_arrays_deserialized") is not False:
        raise ValueError("development QC did not preserve the no-test-deserialization boundary")
    if receipt.get("split_lock_sha256") != split_lock.get("split_lock_sha256"):
        raise ValueError("development receipt does not bind the supplied split lock")
    rebuilt = build_validation_unlock(
        qc,
        bundle["aligned"],
        bundle["null"],
        acceptance_config,
    )
    if rebuilt != saved_unlock:
        raise ValueError("saved validation unlock does not reconstruct from development evidence")
    if receipt.get("validation_unlock_report_sha256") != saved_unlock["report_sha256"]:
        raise ValueError("development receipt does not bind the saved validation unlock")
    if saved_unlock.get("status") != "unlocked_for_single_test_consumption":
        raise ValueError("development validation did not unlock final test consumption")
    if saved_unlock.get("test_consumption_allowed") is not True:
        raise ValueError("validation receipt does not authorize final test consumption")
    eligible_ids = saved_unlock.get("eligible_animal_ids")
    if not isinstance(eligible_ids, list) or len(eligible_ids) < acceptance_config["minimum_eligible_animals"]:
        raise ValueError("validation receipt does not freeze a sufficient confirmatory animal population")
    if len(eligible_ids) != len(set(eligible_ids)):
        raise ValueError("validation receipt contains duplicate confirmatory animal ids")
    return bundle


def _source_batch_receipts(paths: list[str | Path]) -> list[dict[str, str]]:
    return sorted(
        [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for path in paths
        ],
        key=lambda row: row["path"],
    )


def _verify_preload_contract(
    bundle: dict[str, dict[str, Any]],
    *,
    split_lock_path: str | Path,
    batch_paths: list[str | Path],
    acceptance_config_path: str | Path,
) -> list[dict[str, str]]:
    receipt = bundle["receipt"]
    source_batches = _source_batch_receipts(batch_paths)
    if receipt.get("source_batches") != source_batches:
        raise ValueError("final batch bytes do not match the frozen development receipt")
    if receipt.get("split_lock_file_sha256") != sha256_file(split_lock_path):
        raise ValueError("final split-lock bytes do not match the frozen development receipt")
    if receipt.get("acceptance_config_file_sha256") != sha256_file(acceptance_config_path):
        raise ValueError("final acceptance-config bytes do not match the frozen development receipt")
    if receipt.get("implementation_fingerprint") != implementation_fingerprint():
        raise ValueError("final implementation bytes do not match the frozen development runtime")
    if receipt.get("runtime_fingerprint") != runtime_fingerprint():
        raise ValueError("final Python/NumPy runtime does not match the frozen development runtime")
    return source_batches


def _development_paths_from_receipt(
    receipt: dict[str, Any],
    batch_paths: list[str | Path],
) -> tuple[list[str | Path], list[dict[str, str]]]:
    development_sources = receipt.get("development_deserialized_batches")
    if not isinstance(development_sources, list) or not development_sources:
        raise ValueError("development receipt is missing its deserialized train+validation batch identity")
    path_by_resolved = {str(Path(path).resolve()): path for path in batch_paths}
    if len(path_by_resolved) != len(batch_paths):
        raise ValueError("final batch paths contain duplicate resolved paths")
    selected: list[str | Path] = []
    seen: set[str] = set()
    for source in development_sources:
        if not isinstance(source, dict):
            raise TypeError("development batch receipt entry is malformed")
        path = source.get("path")
        digest = source.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise TypeError("development batch receipt entry is incomplete")
        if path in seen:
            raise ValueError("development receipt contains duplicate batch paths")
        seen.add(path)
        supplied = path_by_resolved.get(path)
        if supplied is None:
            raise ValueError("a frozen development batch path is missing from final inputs")
        selected.append(supplied)
    return selected, sorted(development_sources, key=lambda row: row["path"])


def _reproduce_development_evidence_preconsumption(
    bundle: dict[str, dict[str, Any]],
    split_lock: dict[str, Any],
    batch_paths: list[str | Path],
) -> None:
    development_paths, expected_sources = _development_paths_from_receipt(
        bundle["receipt"],
        batch_paths,
    )
    development_batch, current_sources = load_session_batches(development_paths)
    if current_sources != expected_sources:
        raise ValueError("development batch identity changed before final consumption")
    current_aligned = run_within_animal_ridge(
        development_batch,
        split_lock,
        source_batches=current_sources,
        consume_test=False,
    )
    current_null = run_alignment_null(
        development_batch,
        split_lock,
        source_batches=current_sources,
        consume_test=False,
    )
    if current_aligned != bundle["aligned"]:
        raise ValueError("aligned development evidence no longer reproduces before final consumption")
    if current_null != bundle["null"]:
        raise ValueError("temporal-null development evidence no longer reproduces before final consumption")


def _write_consumption_lock(
    development_dir: str | Path,
    *,
    split_lock_sha256: str,
    development_receipt_sha256: str,
    validation_unlock_sha256: str,
) -> dict[str, Any]:
    path = Path(development_dir) / FINAL_LOCK_NAME
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "mc2p-v1-one-way-final-consumption-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "state": "final_evaluator_started_test_namespace_consumed",
        "split_lock_sha256": split_lock_sha256,
        "development_receipt_sha256": development_receipt_sha256,
        "validation_unlock_sha256": validation_unlock_sha256,
        "reopen_after_failure_allowed": False,
        "prepared_test_batch_reopen_allowed_after_this_marker_only": True,
        "development_evidence_reproduced_before_marker": True,
    }
    payload["consumption_lock_sha256"] = _sha(payload)
    try:
        with path.open("x") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise ValueError(
            f"final test namespace already consumed; one-way lock exists at {path}"
        ) from exc
    return payload


def _finite_metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _test_effects(
    aligned: dict[str, Any],
    null: dict[str, Any],
) -> list[dict[str, Any]]:
    aligned_rows = {row["animal_id"]: row for row in aligned["animals"]}
    null_rows = {row["animal_id"]: row for row in null["animals"]}
    if set(aligned_rows) != set(null_rows):
        raise RuntimeError("final aligned/null reports cover different animals")
    effects: list[dict[str, Any]] = []
    for animal in sorted(aligned_rows):
        aligned_metrics = aligned_rows[animal]["test_metrics"]
        null_metrics = null_rows[animal]["test_metrics"]
        if aligned_metrics is None or null_metrics is None:
            raise RuntimeError("final report is missing test metrics after consumption")
        aligned_r = _finite_metric(aligned_metrics, "median_pearson_r")
        null_r = _finite_metric(null_metrics, "median_pearson_r")
        scorable = aligned_r is not None and null_r is not None
        effects.append(
            {
                "animal_id": animal,
                "test_metric_computable": scorable,
                "aligned_test_median_pearson_r": aligned_r,
                "validation_selected_null_fraction": null_rows[animal]["selected_null_fraction"],
                "selected_null_test_median_pearson_r": null_r,
                "paired_alignment_effect": aligned_r - null_r if scorable else None,
            }
        )
    return effects


def run_final_protocol(
    split_lock_path: str | Path,
    batch_paths: list[str | Path],
    development_dir: str | Path,
    output_dir: str | Path,
    *,
    acceptance_config_path: str | Path,
) -> dict[str, Any]:
    output = _fresh_output_dir(output_dir)
    split_lock = _load_json(split_lock_path)
    acceptance_config = load_acceptance_config(acceptance_config_path)
    bundle = validate_final_authorization(development_dir, split_lock, acceptance_config)
    confirmatory_animal_ids = list(bundle["unlock"]["eligible_animal_ids"])
    expected_source_batches = _verify_preload_contract(
        bundle,
        split_lock_path=split_lock_path,
        batch_paths=batch_paths,
        acceptance_config_path=acceptance_config_path,
    )
    _reproduce_development_evidence_preconsumption(bundle, split_lock, batch_paths)

    consumption_lock = _write_consumption_lock(
        development_dir,
        split_lock_sha256=split_lock["split_lock_sha256"],
        development_receipt_sha256=bundle["receipt"]["receipt_sha256"],
        validation_unlock_sha256=bundle["unlock"]["report_sha256"],
    )

    batch, source_batches = load_session_batches(batch_paths)
    if source_batches != expected_source_batches:
        raise RuntimeError("batch identity changed after final consumption lock")
    verify_session_split_lock(split_lock, batch, source_batches=source_batches)

    aligned_final = run_within_animal_ridge(
        batch,
        split_lock,
        source_batches=source_batches,
        consume_test=True,
    )
    null_final = run_alignment_null(
        batch,
        split_lock,
        source_batches=source_batches,
        consume_test=True,
    )
    (output / "aligned-ridge-final.json").write_text(
        json.dumps(aligned_final, indent=2, sort_keys=True) + "\n"
    )
    (output / "temporal-null-final.json").write_text(
        json.dumps(null_final, indent=2, sort_keys=True) + "\n"
    )
    effects = _test_effects(aligned_final, null_final)
    final_inference = evaluate_final_inference(
        effects,
        acceptance_config,
        confirmatory_animal_ids=confirmatory_animal_ids,
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "mc2p-v1-final-evaluation-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "split_lock_sha256": split_lock["split_lock_sha256"],
        "split_lock_file_sha256": sha256_file(split_lock_path),
        "development_receipt_sha256": bundle["receipt"]["receipt_sha256"],
        "validation_unlock_sha256": bundle["unlock"]["report_sha256"],
        "consumption_lock_sha256": consumption_lock["consumption_lock_sha256"],
        "development_evidence_reproduced_before_consumption_lock": True,
        "test_status": "consumed_once",
        "prepared_test_batches_first_reopened_post_split_after_consumption_lock": True,
        "primary_null_selection": "fraction_selected_on_validation_before_test",
        "confirmatory_animal_ids": confirmatory_animal_ids,
        "animal_effects": effects,
        "final_inference": final_inference,
        "median_paired_alignment_effect": final_inference["median_paired_effect"],
        "positive_effect_animals": final_inference["positive_effect_animals"],
        "eligible_animals": final_inference["confirmatory_population_size"],
        "scorable_test_animals": final_inference["scorable_animals"],
        "total_test_animals_reported": len(effects),
        "primary_metric_scope": "all_measured_dff_pixels_with_finite_correlation",
        "claim_boundary": (
            "This receipt reports the prespecified held-out evaluation against the null fraction selected "
            "on validation for each animal. PREPARE necessarily materializes all deterministic session "
            "batches before the split is evaluated. After that split is frozen, DEVELOPMENT authenticates "
            "held-out batch bytes without reopening their arrays. FINAL replays the exact train+validation "
            "development evidence using only its authenticated development batches before writing the one-way "
            "consumption marker; only then may it reopen prepared held-out arrays. The confirmatory population "
            "is exactly the validation-eligible animal IDs frozen before test. Validation-ineligible animals "
            "cannot enter final inference, and missing/non-computable effects inside the frozen confirmatory "
            "population cannot shrink the denominator. Pixels and overlapping windows are not independent "
            "biological replicates."
        ),
    }
    receipt["receipt_sha256"] = _sha(receipt)
    (output / "final-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume the MC2P v1 final test namespace exactly once")
    parser.add_argument("split_lock")
    parser.add_argument("development_dir")
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--acceptance-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_final_protocol(
        args.split_lock,
        args.batches,
        args.development_dir,
        args.output,
        acceptance_config_path=args.acceptance_config,
    )
    print(
        json.dumps(
            {
                "status": report["test_status"],
                "inference": report["final_inference"]["status"],
                "animals": report["eligible_animals"],
                "sha256": report["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
