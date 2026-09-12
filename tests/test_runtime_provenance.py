import copy

import pytest

from fly_sniff.reproducible_training import seal_training_runtime
from fly_sniff.runtime_provenance import (
    NUMERICAL_DISTRIBUTIONS,
    numerical_compatibility_sha256,
    runtime_environment_receipt,
    verify_runtime_environment_receipt,
)
from fly_sniff.training import canonical_sha256


def _minimal_training_report():
    return {
        "protocol": "task-optimized-connectome-dynamics-v1",
        "graph_sha256": "graph",
        "training_config_sha256": "config",
        "train_seed_sha256": "train",
        "validation_seed_sha256": "validation",
        "trained_parameter_sha256": "parameters",
        "optimizer_budget_sha256": "budget",
    }


def test_runtime_receipt_binds_exact_numerical_versions_but_not_platform_diagnostics():
    receipt = runtime_environment_receipt()
    numerical_hash = verify_runtime_environment_receipt(receipt)
    assert numerical_hash == numerical_compatibility_sha256()
    assert set(receipt["numerical_compatibility"]["packages"]) == set(NUMERICAL_DISTRIBUTIONS)

    changed_platform = copy.deepcopy(receipt)
    changed_platform["platform_diagnostics"]["machine"] = "diagnostic-only-other-machine"
    assert verify_runtime_environment_receipt(changed_platform) == numerical_hash


def test_runtime_receipt_rejects_forged_numerical_version():
    receipt = runtime_environment_receipt()
    receipt["numerical_compatibility"]["packages"]["numpy"] = "0.0.0"
    with pytest.raises(ValueError, match="compatibility hash"):
        verify_runtime_environment_receipt(receipt)


def test_runtime_seal_rehashes_training_audit_receipt():
    report = _minimal_training_report()
    sealed = seal_training_runtime(report)
    assert sealed["report_schema"] == "task-optimization-report-v3-runtime-sealed"
    assert sealed["runtime_environment_sha256"] == canonical_sha256(
        sealed["runtime_environment"]
    )
    assert sealed["numerical_runtime_sha256"] == numerical_compatibility_sha256()
    expected_audit = canonical_sha256(
        {
            "graph_sha256": "graph",
            "training_config_sha256": "config",
            "train_seed_sha256": "train",
            "validation_seed_sha256": "validation",
            "trained_parameter_sha256": "parameters",
            "optimizer_budget_sha256": "budget",
            "runtime_environment_sha256": sealed["runtime_environment_sha256"],
            "numerical_runtime_sha256": sealed["numerical_runtime_sha256"],
        }
    )
    assert sealed["audit_receipt_sha256"] == expected_audit


def test_matched_runtime_seal_uses_one_identity_for_every_topology():
    matched = {
        "protocol": "matched-task-optimization-controls-v1",
        "results": {
            "intact": _minimal_training_report(),
            "rewires": {
                "1": _minimal_training_report(),
                "2": _minimal_training_report(),
            },
            "lesion": _minimal_training_report(),
        },
    }
    sealed = seal_training_runtime(matched)
    expected = sealed["numerical_runtime_sha256"]
    reports = [
        sealed["results"]["intact"],
        *sealed["results"]["rewires"].values(),
        sealed["results"]["lesion"],
    ]
    assert all(report["numerical_runtime_sha256"] == expected for report in reports)
    assert len({report["runtime_environment_sha256"] for report in reports}) == 1
