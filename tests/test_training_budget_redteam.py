import copy

import pytest

from fly_sniff.training import (
    canonical_sha256,
    load_training_config,
    make_training_seed_split,
    optimizer_budget_receipt,
)
from fly_sniff.training_budget_redteam import (
    audit_matched_optimizer_execution,
    reconstruct_optimizer_execution,
)


def _config():
    config = copy.deepcopy(load_training_config("configs/task_optimization_v1.json"))
    config["optimizer"].update(
        {
            "population": 4,
            "generations": 2,
            "episodes_per_candidate": 3,
        }
    )
    return config


def _single_report(config):
    train, validation = make_training_seed_split(config)
    history = []
    for generation in range(int(config["optimizer"]["generations"])):
        batch = train[generation : generation + int(config["optimizer"]["episodes_per_candidate"])]
        receipts = [
            {
                "index": index,
                "parameter_sha256": canonical_sha256({"generation": generation, "index": index}),
                "objective": 0.25 + 0.01 * index,
            }
            for index in range(int(config["optimizer"]["population"]))
        ]
        history.append(
            {
                "generation": generation,
                "seed_batch": batch,
                "seed_batch_sha256": canonical_sha256(batch),
                "candidate_receipts": receipts,
            }
        )
    budget = optimizer_budget_receipt(
        config,
        train_seed_count=len(train),
        validation_seed_count=len(validation),
    )
    return {
        "history": history,
        "baseline_train": {"n": len(train)},
        "baseline_validation": {"n": len(validation)},
        "trained_train": {"n": len(train)},
        "trained_validation": {"n": len(validation)},
        "optimizer_budget": budget,
        "optimizer_budget_sha256": canonical_sha256(budget),
    }


def test_reconstruct_optimizer_execution_counts_history_not_just_expected_formula():
    config = _config()
    report = _single_report(config)
    reconstructed = reconstruct_optimizer_execution(report, config)
    assert reconstructed["candidate_evaluations"] == 8
    assert reconstructed["candidate_episodes"] == 24
    assert reconstructed == report["optimizer_budget"]


def test_reconstruct_optimizer_execution_rejects_missing_candidate_receipt():
    config = _config()
    report = _single_report(config)
    report["history"][0]["candidate_receipts"].pop()
    with pytest.raises(RuntimeError, match="candidate receipt count|population"):
        reconstruct_optimizer_execution(report, config)


def test_reconstruct_optimizer_execution_rejects_hidden_validation_seed_in_cem():
    config = _config()
    report = _single_report(config)
    _, validation = make_training_seed_split(config)
    report["history"][0]["seed_batch"][0] = validation[0]
    report["history"][0]["seed_batch_sha256"] = canonical_sha256(
        report["history"][0]["seed_batch"]
    )
    with pytest.raises(RuntimeError, match="outside the frozen training split"):
        reconstruct_optimizer_execution(report, config)


def test_reconstruct_optimizer_execution_rejects_mismatched_full_pool_episode_count():
    config = _config()
    report = _single_report(config)
    report["trained_validation"]["n"] -= 1
    with pytest.raises(RuntimeError, match="trained_validation executed"):
        reconstruct_optimizer_execution(report, config)


def test_matched_execution_audit_requires_all_topologies_to_match():
    config = _config()
    intact = _single_report(config)
    lesion = copy.deepcopy(intact)
    rewires = {str(seed): copy.deepcopy(intact) for seed in range(8)}
    matched = {
        "protocol": "matched-task-optimization-controls-v1",
        "optimizer_budget_sha256": intact["optimizer_budget_sha256"],
        "results": {
            "intact": intact,
            "rewires": rewires,
            "lesion": lesion,
        },
    }
    audit = audit_matched_optimizer_execution(matched, config)
    assert audit["status"] == "actual_execution_receipts_verified"
    assert audit["topology_count"] == 10

    matched["results"]["rewires"]["3"]["history"][1]["candidate_receipts"].pop()
    with pytest.raises(RuntimeError, match="candidate receipt count|population"):
        audit_matched_optimizer_execution(matched, config)
