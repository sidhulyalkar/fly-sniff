from __future__ import annotations

import copy

import pytest

from fly_sniff import experiment as exp

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _artifact(name: str = "connectome") -> exp.ArtifactRef:
    return exp.ArtifactRef(
        name=name,
        kind="scientific_input",
        sha256=SHA_A,
        uri=f"artifact://{name}",
    )


def _training(**overrides: object) -> exp.TrainingPolicy:
    values: dict[str, object] = {
        "navigation_reward_allowed": False,
        "topology_specific_fit_allowed": False,
        "equal_budget_across_topologies": True,
        "whole_graph_backprop_allowed": False,
        "calibration_targets": ("pfn_airflow_tuning",),
        "plasticity_scope": (),
    }
    values.update(overrides)
    return exp.TrainingPolicy(**values)  # type: ignore[arg-type]


def _final(**overrides: object) -> exp.FinalPolicy:
    values: dict[str, object] = {
        "topology_claim": False,
        "topology_null_count": 0,
        "paired_episodes_required": False,
        "hidden_final_entropy_required": False,
        "negative_results_retained": True,
        "one_way_final": False,
    }
    values.update(overrides)
    return exp.FinalPolicy(**values)  # type: ignore[arg-type]


def _spec(
    *,
    program: exp.ExperimentProgram = exp.ExperimentProgram.LATENT_WIRING,
    phase: exp.ExperimentPhase = exp.ExperimentPhase.DEVELOPMENT,
    training_policy: exp.TrainingPolicy | None = None,
    final_policy: exp.FinalPolicy | None = None,
    null_families: tuple[str, ...] = (),
) -> exp.ExperimentSpec:
    return exp.ExperimentSpec(
        experiment_id="test-experiment",
        program=program,
        phase=phase,
        scientific_question="Does biological topology matter?",
        evidence_ledger_sha256=SHA_B,
        artifacts=(_artifact(),),
        null_families=null_families,
        interventions=("intact",),
        primary_metrics=("spl",),
        training_policy=training_policy or _training(),
        final_policy=final_policy or _final(),
        claim_boundary=("development only",),
    )


def test_latent_wiring_development_spec_is_content_addressed() -> None:
    spec = _spec()
    spec.validate()
    payload = spec.to_dict()
    assert payload["spec_sha256"] == spec.sha256
    assert exp.ExperimentSpec.from_dict(payload).sha256 == spec.sha256


def test_latent_wiring_forbids_navigation_reward() -> None:
    spec = _spec(training_policy=_training(navigation_reward_allowed=True))
    with pytest.raises(ValueError, match="forbid navigation reward"):
        spec.validate()


def test_latent_wiring_forbids_topology_specific_fitting() -> None:
    spec = _spec(training_policy=_training(topology_specific_fit_allowed=True))
    with pytest.raises(ValueError, match="topology-specific fitting"):
        spec.validate()


def test_all_v1_programs_forbid_whole_graph_backprop() -> None:
    spec = _spec(training_policy=_training(whole_graph_backprop_allowed=True))
    with pytest.raises(ValueError, match="whole-graph backpropagation"):
        spec.validate()


def test_latent_wiring_requires_independent_calibration_targets() -> None:
    spec = _spec(training_policy=_training(calibration_targets=()))
    with pytest.raises(ValueError, match="independent calibration targets"):
        spec.validate()


def test_confirmatory_topology_claim_requires_credible_null_ensemble() -> None:
    final = _final(
        topology_claim=True,
        topology_null_count=30,
        paired_episodes_required=True,
        hidden_final_entropy_required=True,
        one_way_final=True,
    )
    spec = _spec(
        phase=exp.ExperimentPhase.CONFIRMATORY,
        final_policy=final,
        null_families=("degree_preserving",),
    )
    with pytest.raises(ValueError, match="at least 31"):
        spec.validate()


def test_confirmatory_topology_claim_requires_hidden_entropy_and_pairing() -> None:
    unblinded = _final(
        topology_claim=True,
        topology_null_count=63,
        paired_episodes_required=True,
        hidden_final_entropy_required=False,
        one_way_final=True,
    )
    with pytest.raises(ValueError, match="hidden final entropy"):
        _spec(
            phase=exp.ExperimentPhase.CONFIRMATORY,
            final_policy=unblinded,
            null_families=("degree_preserving",),
        ).validate()

    unpaired = _final(
        topology_claim=True,
        topology_null_count=63,
        paired_episodes_required=False,
        hidden_final_entropy_required=True,
        one_way_final=True,
    )
    with pytest.raises(ValueError, match="paired episode"):
        _spec(
            phase=exp.ExperimentPhase.CONFIRMATORY,
            final_policy=unpaired,
            null_families=("degree_preserving",),
        ).validate()


def test_confirmatory_runs_must_be_one_way_and_retain_negative_results() -> None:
    non_one_way = _final(
        hidden_final_entropy_required=True,
        negative_results_retained=True,
        one_way_final=False,
    )
    with pytest.raises(ValueError, match="one-way final"):
        _spec(
            phase=exp.ExperimentPhase.CONFIRMATORY,
            final_policy=non_one_way,
        ).validate()

    discards_negative = _final(
        hidden_final_entropy_required=True,
        negative_results_retained=False,
        one_way_final=True,
    )
    with pytest.raises(ValueError, match="retain negative results"):
        _spec(
            phase=exp.ExperimentPhase.CONFIRMATORY,
            final_policy=discards_negative,
        ).validate()


def test_inductive_bias_topology_fitting_requires_equal_budgets() -> None:
    training = _training(
        navigation_reward_allowed=True,
        topology_specific_fit_allowed=True,
        equal_budget_across_topologies=False,
        calibration_targets=(),
    )
    spec = _spec(
        program=exp.ExperimentProgram.TOPOLOGY_INDUCTIVE_BIAS,
        training_policy=training,
    )
    with pytest.raises(ValueError, match="equal optimization budgets"):
        spec.validate()


def test_biological_learning_requires_explicit_plasticity_scope() -> None:
    training = _training(calibration_targets=(), plasticity_scope=())
    spec = _spec(
        program=exp.ExperimentProgram.BIOLOGICAL_LEARNING,
        training_policy=training,
    )
    with pytest.raises(ValueError, match="plasticity_scope"):
        spec.validate()


def test_spec_lock_and_receipt_detect_tampering() -> None:
    spec = _spec()
    spec_payload = spec.to_dict()
    tampered_spec = copy.deepcopy(spec_payload)
    tampered_spec["scientific_question"] = "A different question"
    with pytest.raises(ValueError, match="spec hash mismatch"):
        exp.ExperimentSpec.from_dict(tampered_spec)

    lock = exp.ExperimentLock(spec=spec, code_ref="deadbeef", runtime_sha256=SHA_C)
    lock_payload = lock.to_dict()
    tampered_lock = copy.deepcopy(lock_payload)
    tampered_lock["runtime_sha256"] = SHA_A
    with pytest.raises(ValueError, match="lock hash mismatch"):
        exp.ExperimentLock.from_dict(tampered_lock)

    result = exp.ArtifactRef(
        name="results",
        kind="result",
        sha256=SHA_B,
        uri="artifact://results",
    )
    receipt = exp.RunReceipt(
        run_id="run-1",
        lock_sha256=lock.sha256,
        status=exp.RunStatus.COMPLETED,
        result_artifacts=(result,),
        metric_summary=(("spl", 0.42),),
    )
    receipt_payload = receipt.to_dict()
    tampered_receipt = copy.deepcopy(receipt_payload)
    tampered_receipt["metric_summary"]["spl"] = 0.99
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        exp.RunReceipt.from_dict(tampered_receipt)


def test_completed_run_requires_content_addressed_result() -> None:
    spec = _spec()
    lock = exp.ExperimentLock(spec=spec, code_ref="deadbeef", runtime_sha256=SHA_C)
    receipt = exp.RunReceipt(
        run_id="run-1",
        lock_sha256=lock.sha256,
        status=exp.RunStatus.COMPLETED,
    )
    with pytest.raises(ValueError, match="content-addressed result artifact"):
        receipt.validate()
