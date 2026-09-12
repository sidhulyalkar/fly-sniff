from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .training import (
    FINAL_TEST_MAX_SEED,
    canonical_sha256,
    load_training_config,
    make_training_seed_split,
    optimizer_budget_receipt,
)

PROTOCOL = "task-optimization-execution-audit-v1"


def _require_summary_n(report: dict[str, Any], key: str, expected_n: int) -> int:
    summary = report.get(key)
    if not isinstance(summary, dict):
        raise TypeError(f"training report is missing {key} summary")
    try:
        observed_n = int(summary["n"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"training report {key} summary is missing an episode count") from exc
    if observed_n != int(expected_n):
        raise RuntimeError(
            f"training report {key} executed {observed_n} episodes; expected {expected_n}"
        )
    return observed_n


def reconstruct_optimizer_execution(
    report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, int | str]:
    """Reconstruct optimizer work from persisted generation and evaluation receipts.

    This deliberately does not trust the top-level expected-budget object. Candidate
    evaluations are counted from generation history and candidate receipts, while
    full-pool evaluations are counted from the four persisted evaluation summaries.
    """
    train_seeds, validation_seeds = make_training_seed_split(config)
    train_set = set(train_seeds)
    optimizer = config["optimizer"]
    population = int(optimizer["population"])
    generations = int(optimizer["generations"])
    episodes_per_candidate = int(optimizer["episodes_per_candidate"])

    history = report.get("history")
    if not isinstance(history, list):
        raise TypeError("training report is missing optimizer generation history")
    if len(history) != generations:
        raise RuntimeError(
            f"optimizer history contains {len(history)} generations; expected {generations}"
        )

    candidate_evaluations = 0
    candidate_episodes = 0
    for expected_generation, generation in enumerate(history):
        if not isinstance(generation, dict):
            raise TypeError("optimizer generation history entries must be mappings")
        if int(generation.get("generation", -1)) != expected_generation:
            raise RuntimeError("optimizer generation history is missing, duplicated, or reordered")

        seed_batch = [int(seed) for seed in generation.get("seed_batch", [])]
        if len(seed_batch) != episodes_per_candidate or len(set(seed_batch)) != len(seed_batch):
            raise RuntimeError("optimizer generation seed batch has the wrong size or duplicates")
        if not set(seed_batch).issubset(train_set):
            raise RuntimeError("optimizer generation used a seed outside the frozen training split")
        if any(seed <= FINAL_TEST_MAX_SEED for seed in seed_batch):
            raise RuntimeError("optimizer generation touched the final-test seed namespace")
        if generation.get("seed_batch_sha256") != canonical_sha256(seed_batch):
            raise RuntimeError("optimizer generation seed-batch hash mismatch")

        receipts = generation.get("candidate_receipts")
        if not isinstance(receipts, list) or len(receipts) != population:
            raise RuntimeError("optimizer generation candidate receipt count does not match population")
        indices = [int(item.get("index", -1)) for item in receipts if isinstance(item, dict)]
        if indices != list(range(population)):
            raise RuntimeError("optimizer candidate indices are missing, duplicated, or reordered")
        for item in receipts:
            if not isinstance(item, dict):
                raise TypeError("optimizer candidate receipt must be a mapping")
            parameter_sha = str(item.get("parameter_sha256", ""))
            if len(parameter_sha) != 64:
                raise RuntimeError("optimizer candidate receipt has an invalid parameter hash")
            try:
                int(parameter_sha, 16)
            except ValueError as exc:
                raise RuntimeError("optimizer candidate parameter hash is not hexadecimal") from exc
            objective = float(item.get("objective", float("nan")))
            if not np.isfinite(objective):
                raise RuntimeError("optimizer candidate receipt contains a nonfinite objective")

        candidate_evaluations += len(receipts)
        candidate_episodes += len(receipts) * len(seed_batch)

    baseline_train_n = _require_summary_n(report, "baseline_train", len(train_seeds))
    baseline_validation_n = _require_summary_n(
        report, "baseline_validation", len(validation_seeds)
    )
    trained_train_n = _require_summary_n(report, "trained_train", len(train_seeds))
    trained_validation_n = _require_summary_n(
        report, "trained_validation", len(validation_seeds)
    )
    full_pool_evaluations = 4
    full_pool_episodes = (
        baseline_train_n + baseline_validation_n + trained_train_n + trained_validation_n
    )

    reconstructed = {
        "protocol": "task-optimization-budget-v1",
        "population": population,
        "generations": generations,
        "episodes_per_candidate": episodes_per_candidate,
        "candidate_evaluations": candidate_evaluations,
        "candidate_episodes": candidate_episodes,
        "full_pool_evaluations": full_pool_evaluations,
        "full_pool_episodes": full_pool_episodes,
        "total_parameter_evaluations": candidate_evaluations + full_pool_evaluations,
        "total_episode_evaluations": candidate_episodes + full_pool_episodes,
    }
    expected = optimizer_budget_receipt(
        config,
        train_seed_count=len(train_seeds),
        validation_seed_count=len(validation_seeds),
    )
    if reconstructed != expected:
        raise RuntimeError(
            "reconstructed optimizer execution does not match the frozen expected budget"
        )
    if report.get("optimizer_budget") != reconstructed:
        raise RuntimeError("top-level optimizer budget disagrees with reconstructed execution")
    if report.get("optimizer_budget_sha256") != canonical_sha256(reconstructed):
        raise RuntimeError("top-level optimizer budget hash disagrees with reconstructed execution")
    return reconstructed


def audit_matched_optimizer_execution(
    matched_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Verify actual optimizer work for intact, every rewire, and lesion."""
    if matched_report.get("protocol") != "matched-task-optimization-controls-v1":
        raise ValueError("not a matched task-optimization v1 report")
    results = matched_report.get("results")
    if not isinstance(results, dict):
        raise TypeError("matched report is missing results")
    intact = results.get("intact")
    lesion = results.get("lesion")
    rewires = results.get("rewires")
    if not isinstance(intact, dict) or not isinstance(lesion, dict):
        raise TypeError("matched report is missing intact or lesion training evidence")
    if not isinstance(rewires, dict):
        raise TypeError("matched report is missing rewire training evidence")

    reports: list[tuple[str, dict[str, Any]]] = [("intact", intact)]
    reports.extend((f"rewire:{seed}", report) for seed, report in sorted(rewires.items()))
    reports.append(("lesion", lesion))

    audits: dict[str, Any] = {}
    hashes: list[str] = []
    for label, report in reports:
        if not isinstance(report, dict):
            raise TypeError(f"matched optimizer report {label} is not a mapping")
        reconstructed = reconstruct_optimizer_execution(report, config)
        receipt_hash = canonical_sha256(reconstructed)
        hashes.append(receipt_hash)
        audits[label] = {
            "reconstructed_budget": reconstructed,
            "reconstructed_budget_sha256": receipt_hash,
            "history_sha256": canonical_sha256(report["history"]),
        }

    if len(set(hashes)) != 1:
        raise RuntimeError("matched topologies did not execute identical optimizer budgets")
    if matched_report.get("optimizer_budget_sha256") != hashes[0]:
        raise RuntimeError("matched top-level budget hash disagrees with executed topology budgets")

    return {
        "protocol": PROTOCOL,
        "status": "actual_execution_receipts_verified",
        "training_config_sha256": canonical_sha256(config),
        "matched_report_sha256": canonical_sha256(matched_report),
        "topology_count": len(reports),
        "shared_reconstructed_budget_sha256": hashes[0],
        "topologies": audits,
        "claim_boundary": (
            "This audit verifies work represented by the persisted optimizer history and evaluation "
            "summaries. It does not prove that an external process performed no additional hidden "
            "experiments or that human model selection never occurred outside this artifact."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct actual task-optimization work from persisted receipts"
    )
    parser.add_argument("matched_report", help="matched task-optimization report JSON")
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument(
        "--output",
        default="results/training-redteam/execution-budget-audit-v1.json",
    )
    args = parser.parse_args()

    config = load_training_config(args.config)
    matched_report = json.loads(Path(args.matched_report).read_text())
    audit = audit_matched_optimizer_execution(matched_report, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(output)
    print(canonical_sha256(audit))


if __name__ == "__main__":
    main()
