from __future__ import annotations

import dataclasses

import pytest

from fly_sniff.evaluation_contract import (
    AcuteIntervention,
    ConditionFamily,
    ConditionRegime,
    InterventionKind,
    ProgramAEvaluationContract,
    assert_paired_condition_ids,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _family(family_id: str, regime: ConditionRegime, namespace: str) -> ConditionFamily:
    return ConditionFamily(
        family_id=family_id,
        regime=regime,
        scientific_rationale=f"prespecified {regime.value} evaluation family",
        source_rule="sample source from frozen physical support",
        start_rule="sample start pose from frozen support",
        wind_rule="use frozen measured/qualified wind condition",
        plume_rule="sample only from bound plume contract",
        sensor_rule="sample through bound bilateral sensor geometry",
        episode_rule="fixed horizon and dynamics step from preregistration",
        success_rule="fixed source-entry criterion from preregistration",
        hidden_seed_namespace=namespace,
    )


def _lesion(**overrides: object) -> AcuteIntervention:
    values: dict[str, object] = {
        "intervention_id": "steering-input-lesion",
        "kind": InterventionKind.CIRCUIT_LESION,
        "transformation": "remove prespecified steering-input edges without changing parameters",
        "scientific_rationale": "test acute dependence on the preregistered steering route",
        "affected_entities": ("steering_input",),
    }
    values.update(overrides)
    return AcuteIntervention(**values)  # type: ignore[arg-type]


def _contract(**overrides: object) -> ProgramAEvaluationContract:
    values: dict[str, object] = {
        "contract_id": "program-a-evaluation-v1",
        "plume_contract_sha256": _sha(1),
        "bilateral_sensor_geometry_sha256": _sha(2),
        "condition_families": (
            _family("id-main", ConditionRegime.ID, "program-a-final-id"),
            _family("ood-wind", ConditionRegime.OOD, "program-a-final-ood-wind"),
        ),
        "interventions": (_lesion(),),
    }
    values.update(overrides)
    return ProgramAEvaluationContract(**values)  # type: ignore[arg-type]


def test_complete_contract_is_hash_stable_and_roundtrips() -> None:
    contract = _contract()
    contract.validate()
    payload = contract.to_dict()
    restored = ProgramAEvaluationContract.from_dict(payload)

    assert restored.sha256 == contract.sha256
    assert payload["contract_sha256"] == contract.sha256


def test_condition_ids_are_deterministic_and_family_specific() -> None:
    contract = _contract()
    first = contract.condition_id(family_id="id-main", final_seed=1234)
    again = contract.condition_id(family_id="id-main", final_seed=1234)
    other_family = contract.condition_id(family_id="ood-wind", final_seed=1234)
    other_seed = contract.condition_id(family_id="id-main", final_seed=1235)

    assert first == again
    assert first != other_family
    assert first != other_seed
    assert first.startswith("id-main:")


def test_paired_condition_ids_require_exact_ordered_identity() -> None:
    contract = _contract()
    ids = tuple(
        contract.condition_id(family_id="id-main", final_seed=seed)
        for seed in (11, 12, 13)
    )
    assert_paired_condition_ids(ids, ids, ids)

    with pytest.raises(ValueError, match="exact ordered paired"):
        assert_paired_condition_ids(ids, tuple(reversed(ids)))


def test_contract_requires_both_id_and_ood_families() -> None:
    id_only = (_family("id-main", ConditionRegime.ID, "program-a-final-id"),)
    with pytest.raises(ValueError, match="at least one OOD"):
        _contract(condition_families=id_only).validate()

    ood_only = (_family("ood-main", ConditionRegime.OOD, "program-a-final-ood"),)
    with pytest.raises(ValueError, match="at least one ID"):
        _contract(condition_families=ood_only).validate()


def test_condition_families_require_distinct_final_namespaces() -> None:
    families = (
        _family("id-main", ConditionRegime.ID, "same-final-namespace"),
        _family("ood-main", ConditionRegime.OOD, "same-final-namespace"),
    )
    with pytest.raises(ValueError, match="distinct hidden seed namespaces"):
        _contract(condition_families=families).validate()


def test_development_seed_namespace_is_forbidden() -> None:
    family = _family("id-main", ConditionRegime.ID, "development-seeds")
    with pytest.raises(ValueError, match="development seed namespaces"):
        family.validate()


@pytest.mark.parametrize(
    "field",
    [
        "source_coordinates_visible_to_controller",
        "final_seeds_materialized_before_entropy_reveal",
        "topology_specific_condition_selection_allowed",
        "outcome_driven_ood_selection_allowed",
        "acute_intervention_retraining_allowed",
    ],
)
def test_program_a_evaluation_forbidden_freedoms_fail_closed(field: str) -> None:
    with pytest.raises(ValueError):
        _contract(**{field: True}).validate()


@pytest.mark.parametrize(
    "field",
    [
        "parameter_refit_allowed",
        "topology_reselection_allowed",
        "role_reselection_allowed",
    ],
)
def test_acute_lesion_cannot_adapt_after_intervention(field: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _lesion(**{field: True}).validate()


def test_contract_hash_changes_when_condition_rule_changes() -> None:
    contract = _contract()
    changed_family = dataclasses.replace(
        contract.condition_families[0],
        source_rule="different frozen source support",
    )
    changed = dataclasses.replace(
        contract,
        condition_families=(changed_family, contract.condition_families[1]),
    )

    assert changed.sha256 != contract.sha256


def test_contract_hash_changes_when_intervention_changes() -> None:
    contract = _contract()
    changed = dataclasses.replace(
        contract,
        interventions=(
            dataclasses.replace(
                contract.interventions[0],
                affected_entities=("different_role",),
            ),
        ),
    )
    assert changed.sha256 != contract.sha256


def test_unknown_condition_family_cannot_generate_id() -> None:
    with pytest.raises(ValueError, match="unknown condition family_id"):
        _contract().condition_id(family_id="not-frozen", final_seed=7)


def test_contract_does_not_store_materialized_final_seeds() -> None:
    payload = _contract().to_dict()
    assert "final_seeds" not in payload
    assert payload["final_seeds_materialized_before_entropy_reveal"] is False
