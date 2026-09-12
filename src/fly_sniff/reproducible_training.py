from __future__ import annotations

import argparse
from typing import Any

from .graph import GraphBundle
from .runtime_provenance import (
    runtime_environment_receipt,
    verify_runtime_environment_receipt,
)
from .training import (
    FROZEN_V1_REWIRE_COUNT,
    canonical_sha256,
    load_training_config,
    optimize_dynamics,
    train_matched_control_cohort,
    write_report,
)

RUNTIME_SEALED_REPORT_SCHEMA = "task-optimization-report-v3-runtime-sealed"


def _seal_single_report_runtime(
    report: dict[str, Any],
    runtime_receipt: dict[str, Any],
) -> None:
    if report.get("protocol") != "task-optimized-connectome-dynamics-v1":
        raise ValueError("cannot runtime-seal an unexpected training report protocol")
    history = report.get("history")
    if not isinstance(history, list):
        raise TypeError("cannot runtime-seal a training report without optimizer history")
    runtime_sha256 = canonical_sha256(runtime_receipt)
    numerical_sha256 = verify_runtime_environment_receipt(
        runtime_receipt,
        require_current_numerical_match=True,
    )
    history_sha256 = canonical_sha256(history)
    report["report_schema"] = RUNTIME_SEALED_REPORT_SCHEMA
    report["runtime_environment"] = runtime_receipt
    report["runtime_environment_sha256"] = runtime_sha256
    report["numerical_runtime_sha256"] = numerical_sha256
    report["optimizer_history_sha256"] = history_sha256
    report["audit_receipt_sha256"] = canonical_sha256(
        {
            "graph_sha256": report["graph_sha256"],
            "training_config_sha256": report["training_config_sha256"],
            "train_seed_sha256": report["train_seed_sha256"],
            "validation_seed_sha256": report["validation_seed_sha256"],
            "trained_parameter_sha256": report["trained_parameter_sha256"],
            "optimizer_budget_sha256": report["optimizer_budget_sha256"],
            "optimizer_history_sha256": history_sha256,
            "runtime_environment_sha256": runtime_sha256,
            "numerical_runtime_sha256": numerical_sha256,
        }
    )


def seal_training_runtime(
    report: dict[str, Any],
    runtime_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach one immutable runtime receipt to all training reports in an artifact."""
    receipt = runtime_receipt or runtime_environment_receipt()
    verify_runtime_environment_receipt(receipt, require_current_numerical_match=True)

    protocol = report.get("protocol")
    if protocol == "task-optimized-connectome-dynamics-v1":
        _seal_single_report_runtime(report, receipt)
        return report
    if protocol != "matched-task-optimization-controls-v1":
        raise ValueError("unsupported training artifact protocol for runtime sealing")

    results = report.get("results")
    if not isinstance(results, dict):
        raise TypeError("matched training artifact is missing results")
    intact = results.get("intact")
    lesion = results.get("lesion")
    rewires = results.get("rewires")
    if not isinstance(intact, dict) or not isinstance(lesion, dict):
        raise TypeError("matched training artifact is missing intact or lesion report")
    if not isinstance(rewires, dict):
        raise TypeError("matched training artifact is missing rewire reports")

    _seal_single_report_runtime(intact, receipt)
    for seed, rewire_report in rewires.items():
        if not isinstance(rewire_report, dict):
            raise TypeError(f"matched rewire report {seed} is not a mapping")
        _seal_single_report_runtime(rewire_report, receipt)
    _seal_single_report_runtime(lesion, receipt)

    runtime_sha256 = canonical_sha256(receipt)
    numerical_sha256 = verify_runtime_environment_receipt(
        receipt,
        require_current_numerical_match=True,
    )
    report["runtime_environment"] = receipt
    report["runtime_environment_sha256"] = runtime_sha256
    report["numerical_runtime_sha256"] = numerical_sha256
    report["runtime_contract"] = (
        "All intact, eight rewire, and lesion optimization reports were produced within one "
        "numerical runtime identity and sealed before artifact persistence."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task-optimize six global dynamics parameters with sealed runtime provenance"
    )
    parser.add_argument("bundle", help="reviewed signed GraphBundle directory")
    parser.add_argument(
        "--config",
        default="configs/task_optimization_v1.json",
        help="frozen task-optimization JSON contract",
    )
    parser.add_argument("--output", default="results/training/intact-v1.json")
    parser.add_argument(
        "--exploratory-candidate",
        action="store_true",
        help="allow an unqualified graph for development only; result remains candidate evidence",
    )
    parser.add_argument(
        "--matched-controls",
        action="store_true",
        help="also train the frozen eight-rewire ensemble and steering-input lesion",
    )
    parser.add_argument(
        "--rewire-count",
        type=int,
        default=FROZEN_V1_REWIRE_COUNT,
        help="protocol-identity check; frozen v1 requires exactly 8",
    )
    args = parser.parse_args()

    runtime_before = runtime_environment_receipt()
    verify_runtime_environment_receipt(runtime_before, require_current_numerical_match=True)
    config = load_training_config(args.config)
    bundle = GraphBundle.load(args.bundle)
    require_qualified = not args.exploratory_candidate
    if args.matched_controls:
        report = train_matched_control_cohort(
            bundle,
            config,
            require_qualified=require_qualified,
            rewire_count=args.rewire_count,
        )
    else:
        report = optimize_dynamics(
            bundle,
            config,
            require_qualified=require_qualified,
        )

    runtime_after = runtime_environment_receipt()
    if runtime_after["numerical_compatibility_sha256"] != runtime_before[
        "numerical_compatibility_sha256"
    ]:
        raise RuntimeError("numerical runtime changed during task optimization")
    report = seal_training_runtime(report, runtime_before)
    path = write_report(args.output, report)
    print(path)
    print(canonical_sha256(report))


if __name__ == "__main__":
    main()
