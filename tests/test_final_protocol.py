from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.artifact_io import file_sha256, write_typed_artifact
from fly_sniff.experiment import (
    ArtifactRef,
    ExperimentLock,
    ExperimentPhase,
    ExperimentProgram,
    ExperimentSpec,
    FinalPolicy,
    TrainingPolicy,
)
from fly_sniff.final_entropy import FinalEntropyCommitment, FinalSeedReceipt
from fly_sniff.final_protocol import main, verify_commitment_binding

SECRET = b"final-secret-for-tests-32-bytes!!"
RUNTIME_SHA = "c" * 64
EVIDENCE_SHA = "d" * 64


def _write_commitment(path: Path, commitment: FinalEntropyCommitment) -> None:
    path.write_text(
        json.dumps(commitment.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _lock_with_commitment(tmp_path: Path) -> tuple[FinalEntropyCommitment, Path, ExperimentLock, Path]:
    commitment = FinalEntropyCommitment.build(
        experiment_id="odor-zero-shot-v1",
        secret=SECRET,
    )
    commitment_path = tmp_path / "final-entropy-commitment.json"
    _write_commitment(commitment_path, commitment)

    spec = ExperimentSpec(
        experiment_id="odor-zero-shot-v1",
        program=ExperimentProgram.LATENT_WIRING,
        phase=ExperimentPhase.CONFIRMATORY,
        scientific_question="Does intact biological wiring matter without navigation training?",
        evidence_ledger_sha256=EVIDENCE_SHA,
        artifacts=(
            ArtifactRef(
                name="final_entropy_commitment",
                kind="final_entropy_commitment",
                sha256=file_sha256(commitment_path),
                uri=str(commitment_path),
            ),
        ),
        null_families=(),
        interventions=("intact",),
        primary_metrics=("spl",),
        training_policy=TrainingPolicy(
            navigation_reward_allowed=False,
            topology_specific_fit_allowed=False,
            equal_budget_across_topologies=True,
            whole_graph_backprop_allowed=False,
            calibration_targets=("pfn_airflow_tuning",),
        ),
        final_policy=FinalPolicy(
            topology_claim=False,
            topology_null_count=0,
            paired_episodes_required=True,
            hidden_final_entropy_required=True,
            negative_results_retained=True,
            one_way_final=True,
        ),
    )
    lock = ExperimentLock(spec=spec, code_ref="test-code-ref", runtime_sha256=RUNTIME_SHA)
    lock_path = tmp_path / "experiment-lock.json"
    write_typed_artifact(lock_path, lock)
    return commitment, commitment_path, lock, lock_path


def test_commitment_file_must_be_exactly_bound_to_lock(tmp_path) -> None:
    commitment, commitment_path, lock, _ = _lock_with_commitment(tmp_path)
    verify_commitment_binding(
        lock=lock,
        commitment=commitment,
        commitment_file_sha256=file_sha256(commitment_path),
    )

    commitment_path.write_text(commitment_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact commitment file bytes"):
        verify_commitment_binding(
            lock=lock,
            commitment=commitment,
            commitment_file_sha256=file_sha256(commitment_path),
        )


def test_commitment_binding_rejects_missing_reference(tmp_path) -> None:
    commitment = FinalEntropyCommitment.build(experiment_id="odor-zero-shot-v1", secret=SECRET)
    spec = ExperimentSpec(
        experiment_id="odor-zero-shot-v1",
        program=ExperimentProgram.LATENT_WIRING,
        phase=ExperimentPhase.DEVELOPMENT,
        scientific_question="Development-only contract",
        evidence_ledger_sha256=EVIDENCE_SHA,
        artifacts=(),
        null_families=(),
        interventions=("intact",),
        primary_metrics=("spl",),
        training_policy=TrainingPolicy(
            navigation_reward_allowed=False,
            topology_specific_fit_allowed=False,
            equal_budget_across_topologies=True,
            whole_graph_backprop_allowed=False,
            calibration_targets=("pfn_airflow_tuning",),
        ),
        final_policy=FinalPolicy(
            topology_claim=False,
            topology_null_count=0,
            paired_episodes_required=False,
            hidden_final_entropy_required=False,
            negative_results_retained=True,
            one_way_final=False,
        ),
    )
    lock = ExperimentLock(spec=spec, code_ref="test-code-ref", runtime_sha256=RUNTIME_SHA)
    with pytest.raises(ValueError, match="exactly one final_entropy_commitment"):
        verify_commitment_binding(
            lock=lock,
            commitment=commitment,
            commitment_file_sha256="e" * 64,
        )


def test_registered_protocol_cli_derives_and_verifies_lock_bound_seeds(tmp_path, capsys) -> None:
    secret_path = tmp_path / "secret.bin"
    secret_path.write_bytes(SECRET)
    _, commitment_path, lock, lock_path = _lock_with_commitment(tmp_path)
    seed_path = tmp_path / "final-seeds.json"

    assert (
        main(
            [
                "derive",
                "--secret-file",
                str(secret_path),
                "--commitment",
                str(commitment_path),
                "--lock",
                str(lock_path),
                "--namespace",
                "heldout",
                "--count",
                "48",
                "--output",
                str(seed_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "derived_lock_bound_final_seed_receipt"
    assert output["lock_sha256"] == lock.sha256

    receipt = FinalSeedReceipt.from_dict(json.loads(seed_path.read_text()))
    assert receipt.lock_sha256 == lock.sha256
    assert len(receipt.seeds) == 48

    assert (
        main(
            [
                "verify",
                "--secret-file",
                str(secret_path),
                "--commitment",
                str(commitment_path),
                "--lock",
                str(lock_path),
                "--seed-receipt",
                str(seed_path),
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "verified_lock_bound_final_entropy_reveal"


def test_commit_cli_never_prints_or_writes_secret(tmp_path, capsys) -> None:
    secret_path = tmp_path / "secret.bin"
    secret_path.write_bytes(SECRET)
    commitment_path = tmp_path / "commitment.json"

    assert (
        main(
            [
                "commit",
                "--secret-file",
                str(secret_path),
                "--experiment-id",
                "odor-zero-shot-v1",
                "--output",
                str(commitment_path),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert SECRET.hex() not in stdout
    assert SECRET.decode("ascii") not in stdout
    assert SECRET.hex() not in commitment_path.read_text()
    assert SECRET.decode("ascii") not in commitment_path.read_text()
