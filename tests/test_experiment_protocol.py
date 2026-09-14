from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.experiment_protocol import (
    ExperimentLock,
    ExperimentSpec,
    RunReceipt,
    canonical_sha256,
    seal_experiment,
)


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="test-zero-shot-v1",
        dataset={"connectome": "male-cns:v1.0"},
        circuit={"hypothesis": "odor-navigation"},
        evidence={"ledger": "authority/evidence-ledger.json"},
        dynamics={"model": "physiology-calibrated-rate-v1"},
        environment={"sensory_source": "experimental-plume-v1"},
        interventions=[{"name": "intact"}, {"name": "PFL3_output_cut"}],
        nulls={"family": "type-constrained", "n": 63},
        metrics={"primary": "SPL"},
        training={"navigation_reward": "forbidden"},
        final={"blinded": True},
    )


def test_experiment_lock_hashes_spec_and_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"ok": True}))
    spec = _spec()
    lock = seal_experiment(
        spec,
        artifact_paths={"artifact": artifact},
        code_ref="abc123",
        runtime={"python": "3.11"},
    )
    payload = lock.to_dict()
    assert payload["spec_sha256"] == canonical_sha256(spec.to_dict())
    assert ExperimentLock.from_dict(payload).to_dict()["lock_sha256"] == payload["lock_sha256"]


def test_tampered_lock_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    lock = seal_experiment(
        _spec(),
        artifact_paths={"artifact": artifact},
        code_ref="abc123",
        runtime={"python": "3.11"},
    )
    payload = lock.to_dict()
    payload["code_ref"] = "tampered"
    try:
        ExperimentLock.from_dict(payload)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered experiment lock was accepted")


def test_run_receipt_roundtrip() -> None:
    receipt = RunReceipt(
        run_id="run-001",
        experiment_lock_sha256="a" * 64,
        variant="intact",
        seeds=[11, 12],
        result_artifact_sha256={"metrics": "b" * 64},
        runtime={"code_ref": "abc123"},
    )
    payload = receipt.to_dict()
    loaded = RunReceipt.from_dict(payload)
    assert loaded.run_id == "run-001"
    assert loaded.to_dict()["receipt_sha256"] == payload["receipt_sha256"]
