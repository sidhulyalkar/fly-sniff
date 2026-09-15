from __future__ import annotations

import dataclasses

import pytest

from fly_sniff.experiment import (
    ArtifactRef,
    ExperimentPhase,
    ExperimentProgram,
    ExperimentSpec,
    FinalPolicy,
    TrainingPolicy,
)
from fly_sniff.program_a_preflight import (
    REQUIRED_PROGRAM_A_ARTIFACT_KINDS,
    DependencyGate,
    DependencyManifest,
    DependencyStatus,
    assemble_program_a_lock,
    build_readiness_report,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _artifacts() -> tuple[ArtifactRef, ...]:
    return tuple(
        ArtifactRef(
            name=f"artifact-{index}-{kind}",
            kind=kind,
            sha256=_sha(index + 1),
            uri=f"artifact://{kind}/{index}",
        )
        for index, kind in enumerate(REQUIRED_PROGRAM_A_ARTIFACT_KINDS)
    )


def _manifest(
    artifacts: tuple[ArtifactRef, ...],
    *,
    blocked_name: str | None = None,
) -> DependencyManifest:
    return DependencyManifest(
        dependencies=tuple(
            DependencyGate(
                artifact=artifact,
                status=(
                    DependencyStatus.BLOCKED
                    if artifact.name == blocked_name
                    else DependencyStatus.PASS
                ),
                notes=("scientific gate not satisfied" if artifact.name == blocked_name else None),
            )
            for artifact in artifacts
        )
    )


def _spec(
    artifacts: tuple[ArtifactRef, ...],
    **overrides: object,
) -> ExperimentSpec:
    values: dict[str, object] = {
        "experiment_id": "program-a-zero-shot-v1",
        "program": ExperimentProgram.LATENT_WIRING,
        "phase": ExperimentPhase.CONFIRMATORY,
        "scientific_question": "Does intact MaleCNS wiring outperform frozen topology nulls without navigation training?",
        "evidence_ledger_sha256": _sha(100),
        "artifacts": artifacts,
        "null_families": ("directed_degree",),
        "interventions": ("steering_input_lesion",),
        "primary_metrics": ("mean_paired_spl",),
        "training_policy": TrainingPolicy(
            navigation_reward_allowed=False,
            topology_specific_fit_allowed=False,
            equal_budget_across_topologies=True,
            whole_graph_backprop_allowed=False,
            calibration_targets=("pfn_airflow_geometry",),
        ),
        "final_policy": FinalPolicy(
            topology_claim=True,
            topology_null_count=63,
            paired_episodes_required=True,
            hidden_final_entropy_required=True,
            negative_results_retained=True,
            one_way_final=True,
        ),
        "claim_boundary": ("modeled dynamics are not measured firing",),
    }
    values.update(overrides)
    return ExperimentSpec(**values)  # type: ignore[arg-type]


def test_complete_program_a_contract_is_ready_and_can_lock() -> None:
    artifacts = _artifacts()
    manifest = _manifest(artifacts)
    spec = _spec(artifacts)

    readiness = build_readiness_report(spec, manifest)
    assert readiness.ready_to_lock is True
    assert readiness.blockers == ()

    lock = assemble_program_a_lock(
        spec,
        manifest,
        readiness,
        code_ref="5f938b058a44a613d1ba17262551cdf4fef1d6ab",
        runtime_sha256=_sha(999),
    )
    assert lock.spec.sha256 == spec.sha256


def test_missing_dependency_kind_blocks() -> None:
    artifacts = _artifacts()
    manifest = _manifest(artifacts[:-1])
    spec = _spec(artifacts)

    readiness = build_readiness_report(spec, manifest)
    assert readiness.ready_to_lock is False
    assert "missing_dependency_kind:final_entropy_commitment" in readiness.blockers


def test_explicitly_blocked_dependency_blocks_even_when_bound() -> None:
    artifacts = _artifacts()
    manifest = _manifest(artifacts, blocked_name=artifacts[5].name)
    readiness = build_readiness_report(_spec(artifacts), manifest)

    assert f"dependency_blocked:{artifacts[5].name}" in readiness.blockers


def test_dependency_must_be_exactly_bound_in_spec() -> None:
    artifacts = _artifacts()
    mutated = dataclasses.replace(artifacts[0], sha256=_sha(500))
    manifest = _manifest((mutated,) + artifacts[1:])
    readiness = build_readiness_report(_spec(artifacts), manifest)

    assert f"dependency_not_exactly_bound_in_spec:{mutated.name}" in readiness.blockers


def test_multiple_headline_metrics_block() -> None:
    artifacts = _artifacts()
    spec = _spec(artifacts, primary_metrics=("mean_paired_spl", "success_rate"))
    readiness = build_readiness_report(spec, _manifest(artifacts))

    assert "exactly_one_headline_primary_metric_required" in readiness.blockers


def test_wrong_program_and_phase_block() -> None:
    artifacts = _artifacts()
    spec = _spec(
        artifacts,
        program=ExperimentProgram.TOPOLOGY_INDUCTIVE_BIAS,
        phase=ExperimentPhase.DEVELOPMENT,
    )
    readiness = build_readiness_report(spec, _manifest(artifacts))

    assert "program_must_be_latent_wiring" in readiness.blockers
    assert "phase_must_be_confirmatory" in readiness.blockers


def test_too_few_topology_nulls_block_without_relaxing_threshold() -> None:
    artifacts = _artifacts()
    final = dataclasses.replace(_spec(artifacts).final_policy, topology_null_count=30)
    readiness = build_readiness_report(
        _spec(artifacts, final_policy=final),
        _manifest(artifacts),
    )

    assert "at_least_31_topology_nulls_required" in readiness.blockers


def test_navigation_reward_and_topology_specific_fit_cannot_pass() -> None:
    artifacts = _artifacts()
    policy = dataclasses.replace(
        _spec(artifacts).training_policy,
        navigation_reward_allowed=True,
        topology_specific_fit_allowed=True,
    )
    readiness = build_readiness_report(
        _spec(artifacts, training_policy=policy),
        _manifest(artifacts),
    )

    assert "navigation_reward_forbidden" in readiness.blockers
    assert "topology_specific_fit_forbidden" in readiness.blockers
    with pytest.raises(ValueError, match="BLOCKED"):
        assemble_program_a_lock(
            _spec(artifacts, training_policy=policy),
            _manifest(artifacts),
            readiness,
            code_ref="main",
            runtime_sha256=_sha(999),
        )


def test_dependency_hash_mutation_changes_readiness_identity() -> None:
    artifacts = _artifacts()
    spec = _spec(artifacts)
    manifest = _manifest(artifacts)
    readiness = build_readiness_report(spec, manifest)

    mutated = dataclasses.replace(artifacts[0], sha256=_sha(777))
    mutated_manifest = _manifest((mutated,) + artifacts[1:])
    mutated_readiness = build_readiness_report(spec, mutated_manifest)

    assert mutated_readiness.sha256 != readiness.sha256
    with pytest.raises(ValueError, match="does not match"):
        assemble_program_a_lock(
            spec,
            mutated_manifest,
            readiness,
            code_ref="main",
            runtime_sha256=_sha(999),
        )


def test_preflight_module_does_not_import_navigation_evaluator() -> None:
    import fly_sniff.program_a_preflight as module

    imported_names = set(module.__dict__)
    assert "training" not in imported_names
    assert "qualified_eval" not in imported_names
    assert "sealed_experiment" not in imported_names
