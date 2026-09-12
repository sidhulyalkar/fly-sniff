from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .candidate_seal import canonical_sha256 as candidate_sha256
from .candidate_seal import verify_candidate_manifest
from .evaluate import manifest_digest
from .freeze import current_git_ref
from .graph import GraphBundle
from .reproducible_training import seal_training_runtime
from .runtime_provenance import runtime_environment_receipt, verify_runtime_environment_receipt
from .trained_final import (
    build_trained_final_manifest,
    main as trained_final_main,
    verify_trained_final_manifest,
)
from .trained_qualification import qualify_trained_candidate
from .training import (
    canonical_sha256,
    load_training_config,
    train_matched_control_cohort,
    write_report,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _candidate_manifest(
    bundle: str | Path,
    manifest: str | Path,
    config: str | Path,
) -> dict[str, Any]:
    return verify_candidate_manifest(
        bundle,
        manifest,
        task_config_path=config,
        require_training_ready=True,
    )


def _bind_matched_report(
    report: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if report.get("protocol") != "matched-task-optimization-controls-v1":
        raise ValueError("sealed experiment requires the complete matched-control training report")
    core_sha256 = canonical_sha256(report)
    binding = {
        "protocol": "sealed-candidate-training-binding-v1",
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "candidate_graph_sha256": candidate["graph_sha256"],
        "candidate_task_config_sha256": candidate["task_config_sha256"],
        "matched_training_core_sha256": core_sha256,
        "sensory_drive_normalization": candidate["sensory_drive_normalization"],
    }
    report["candidate_seal"] = binding
    report["candidate_seal_sha256"] = canonical_sha256(binding)
    return report


def _verify_matched_binding(
    report: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    binding = report.get("candidate_seal")
    if not isinstance(binding, dict):
        raise ValueError("matched training report is not bound to a sealed candidate manifest")
    if report.get("candidate_seal_sha256") != canonical_sha256(binding):
        raise ValueError("matched training candidate-seal receipt hash mismatch")
    if binding.get("candidate_manifest_sha256") != candidate["manifest_sha256"]:
        raise ValueError("matched training used a different candidate manifest")
    if binding.get("candidate_graph_sha256") != candidate["graph_sha256"]:
        raise ValueError("matched training candidate graph fingerprint mismatch")
    if binding.get("candidate_task_config_sha256") != candidate["task_config_sha256"]:
        raise ValueError("matched training candidate task-config hash mismatch")
    core = dict(report)
    core.pop("candidate_seal", None)
    core.pop("candidate_seal_sha256", None)
    if binding.get("matched_training_core_sha256") != canonical_sha256(core):
        raise ValueError("matched training report changed after candidate binding")


def train_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched intact/rewire/lesion optimization from one sealed candidate"
    )
    parser.add_argument("bundle")
    parser.add_argument("candidate_manifest")
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument("--output", default="results/sealed-experiment/matched-training-v1.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite matched training artifact: {output}")

    candidate = _candidate_manifest(args.bundle, args.candidate_manifest, args.config)
    config = load_training_config(args.config)
    bundle = GraphBundle.load(args.bundle)
    bundle.validate(require_sign=True, require_qualified=True)
    if bundle.replay_fingerprint() != candidate["graph_sha256"]:
        raise ValueError("candidate manifest graph changed before training")

    runtime_before = runtime_environment_receipt()
    verify_runtime_environment_receipt(runtime_before, require_current_numerical_match=True)
    report = train_matched_control_cohort(bundle, config, require_qualified=True)
    runtime_after = runtime_environment_receipt()
    if runtime_after["numerical_compatibility_sha256"] != runtime_before[
        "numerical_compatibility_sha256"
    ]:
        raise RuntimeError("numerical runtime changed during matched task optimization")
    report = seal_training_runtime(report, runtime_before)
    report = _bind_matched_report(report, candidate)
    write_report(output, report)
    print(output)
    print(canonical_sha256(report))


def qualify_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained E002 while preserving the sealed candidate binding"
    )
    parser.add_argument("bundle")
    parser.add_argument("candidate_manifest")
    parser.add_argument("matched_training_report")
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument(
        "--output",
        default="results/sealed-experiment/trained-e002-v1.json",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite trained E002 artifact: {output}")

    candidate = _candidate_manifest(args.bundle, args.candidate_manifest, args.config)
    matched = _load_json(args.matched_training_report)
    _verify_matched_binding(matched, candidate)
    config = load_training_config(args.config)
    bundle = GraphBundle.load(args.bundle)
    intact = matched.get("results", {}).get("intact")
    if not isinstance(intact, dict):
        raise TypeError("matched training report is missing intact evidence")
    report = qualify_trained_candidate(bundle, intact, config)
    report["candidate_manifest_sha256"] = candidate["manifest_sha256"]
    report["matched_training_report_sha256"] = canonical_sha256(matched)
    report["candidate_binding_sha256"] = canonical_sha256(
        {
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "matched_training_report_sha256": report["matched_training_report_sha256"],
            "trained_e002_core_sha256": canonical_sha256(
                {key: value for key, value in report.items() if not key.startswith("candidate_")}
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(output)
    print(canonical_sha256(report))
    if report.get("passed") is not True:
        raise SystemExit(2)


def freeze_final_main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze final seeds only after the sealed candidate and trained E002 are fixed"
    )
    parser.add_argument("bundle")
    parser.add_argument("candidate_manifest")
    parser.add_argument("matched_training_report")
    parser.add_argument("trained_e002")
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument("--output", default="manifests/sealed-trained-final-v1.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite trained final manifest: {output}")

    candidate = _candidate_manifest(args.bundle, args.candidate_manifest, args.config)
    matched = _load_json(args.matched_training_report)
    _verify_matched_binding(matched, candidate)
    trained_e002 = _load_json(args.trained_e002)
    if trained_e002.get("candidate_manifest_sha256") != candidate["manifest_sha256"]:
        raise ValueError("trained E002 is not bound to the same candidate manifest")
    if trained_e002.get("matched_training_report_sha256") != canonical_sha256(matched):
        raise ValueError("trained E002 is not bound to the supplied matched training report")

    bundle = GraphBundle.load(args.bundle)
    config = load_training_config(args.config)
    from .qualified_eval import circuit_digest

    manifest = build_trained_final_manifest(
        bundle=bundle,
        circuit_sha256=circuit_digest(args.bundle),
        config=config,
        matched_report=matched,
        trained_e002=trained_e002,
        code_ref=current_git_ref(),
    )
    manifest.pop("manifest_sha256", None)
    manifest["candidate_manifest_sha256"] = candidate["manifest_sha256"]
    manifest["candidate_graph_sha256"] = candidate["graph_sha256"]
    manifest["candidate_manifest_file_sha256"] = candidate_sha256(candidate)
    manifest["sensory_drive_normalization"] = candidate["sensory_drive_normalization"]
    manifest["candidate_freeze_rule"] = candidate["performance_freeze"]
    manifest["manifest_sha256"] = manifest_digest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"sealed {output}: {manifest['manifest_sha256']}")


def final_main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume the frozen final namespace exactly once for a sealed candidate"
    )
    parser.add_argument("bundle")
    parser.add_argument("candidate_manifest")
    parser.add_argument("final_manifest")
    parser.add_argument("matched_training_report")
    parser.add_argument("trained_e002")
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument("--output", default="results/sealed-experiment/final-v1")
    parser.add_argument(
        "--final-lock",
        default="manifests/final-run-consumed-v1.json",
        help="one-way receipt created before final seeds are evaluated",
    )
    parser.add_argument(
        "--arm-final",
        action="store_true",
        help="required acknowledgement that this consumes the v1 final-test namespace",
    )
    args = parser.parse_args()
    if not args.arm_final:
        raise SystemExit("refusing final evaluation without --arm-final")

    candidate = _candidate_manifest(args.bundle, args.candidate_manifest, args.config)
    matched = _load_json(args.matched_training_report)
    _verify_matched_binding(matched, candidate)
    final_manifest = _load_json(args.final_manifest)
    verify_trained_final_manifest(final_manifest)
    if final_manifest.get("candidate_manifest_sha256") != candidate["manifest_sha256"]:
        raise ValueError("final manifest is not bound to the supplied candidate manifest")
    if final_manifest.get("candidate_graph_sha256") != candidate["graph_sha256"]:
        raise ValueError("final manifest is not bound to the supplied candidate graph")

    final_lock = Path(args.final_lock)
    if final_lock.exists():
        raise SystemExit(
            f"final-test namespace was already consumed according to {final_lock}; "
            "do not rerun v1 under a different output path"
        )
    final_lock.parent.mkdir(parents=True, exist_ok=True)
    lock_payload = {
        "protocol": "one-way-final-consumption-v1",
        "status": "FINAL_NAMESPACE_CONSUMED_STARTED",
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "final_manifest_sha256": final_manifest["manifest_sha256"],
        "matched_training_report_sha256": canonical_sha256(matched),
        "trained_e002_sha256": canonical_sha256(_load_json(args.trained_e002)),
        "code_ref": current_git_ref(),
        "rule": "Presence of this file forbids another v1 final evaluation even if execution fails after final seeds become observable.",
    }
    lock_payload["receipt_sha256"] = canonical_sha256(lock_payload)
    final_lock.write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n")

    saved_argv = sys.argv
    try:
        sys.argv = [
            "fly-sniff-trained-final",
            str(args.bundle),
            str(args.final_manifest),
            str(args.matched_training_report),
            str(args.trained_e002),
            "--output",
            str(args.output),
        ]
        trained_final_main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    train_main()
