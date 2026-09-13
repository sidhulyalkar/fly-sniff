import copy

import pytest

from fly_sniff.seed_plan import build_seed_plan, verify_seed_plan
from fly_sniff.training import load_training_config


def test_seed_plan_materializes_disjoint_development_and_final_populations():
    config = load_training_config("configs/task_optimization_v1.json")
    plan = build_seed_plan(config)
    train = plan["development"]["train_seeds"]
    validation = plan["development"]["validation_seeds"]
    heldout = plan["final"]["heldout_seeds"]
    ood = plan["final"]["ood_seeds"]

    assert set(train).isdisjoint(validation)
    assert set(train + validation).isdisjoint(heldout + ood)
    assert min(train + validation) > 1_999_999_999
    assert max(heldout + ood) <= 1_999_999_999
    assert len(plan["rewire"]["seeds"]) == 8
    verify_seed_plan(plan, config)


def test_seed_plan_rejects_posthoc_seed_mutation():
    config = load_training_config("configs/task_optimization_v1.json")
    plan = build_seed_plan(config)
    tampered = copy.deepcopy(plan)
    tampered["final"]["heldout_seeds"][0] += 1
    with pytest.raises(ValueError, match="seed-plan hash mismatch"):
        verify_seed_plan(tampered, config)
