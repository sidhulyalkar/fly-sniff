import json
import sys

import pytest

from fly_sniff import sealed_experiment


def _write_json(path, payload):
    path.write_text(json.dumps(payload))


def test_candidate_manifest_requires_exact_sealing_checkout(monkeypatch):
    candidate = {"code_ref": "sealed-commit"}
    monkeypatch.setattr(
        sealed_experiment,
        "verify_candidate_manifest",
        lambda bundle, manifest, task_config_path, require_training_ready: candidate,
    )
    monkeypatch.setattr(sealed_experiment, "current_git_ref", lambda: "different-commit")
    with pytest.raises(RuntimeError, match="differs from the code commit sealed before performance"):
        sealed_experiment._candidate_manifest("bundle", "candidate", "config")


def test_candidate_manifest_accepts_exact_sealing_checkout(monkeypatch):
    candidate = {"code_ref": "sealed-commit"}
    monkeypatch.setattr(
        sealed_experiment,
        "verify_candidate_manifest",
        lambda bundle, manifest, task_config_path, require_training_ready: candidate,
    )
    monkeypatch.setattr(sealed_experiment, "current_git_ref", lambda: "sealed-commit")
    assert sealed_experiment._candidate_manifest("bundle", "candidate", "config") is candidate


def test_final_lock_precedes_evaluation_and_survives_failed_final(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    candidate_path = tmp_path / "candidate.json"
    final_manifest_path = tmp_path / "final.json"
    matched_path = tmp_path / "matched.json"
    trained_e002_path = tmp_path / "trained-e002.json"
    final_lock = tmp_path / "final-consumed.json"
    output = tmp_path / "final-output"

    _write_json(candidate_path, {"fixture": "candidate"})
    _write_json(
        final_manifest_path,
        {
            "manifest_sha256": "final-manifest-sha",
            "candidate_manifest_sha256": "candidate-sha",
            "candidate_graph_sha256": "graph-sha",
            "candidate_manifest_file_sha256": "candidate-file-sha",
        },
    )
    _write_json(matched_path, {"fixture": "matched"})
    _write_json(trained_e002_path, {"fixture": "trained-e002"})

    candidate = {
        "manifest_sha256": "candidate-sha",
        "graph_sha256": "graph-sha",
    }
    monkeypatch.setattr(
        sealed_experiment,
        "_candidate_manifest",
        lambda bundle_path, manifest_path, config_path: candidate,
    )
    monkeypatch.setattr(
        sealed_experiment,
        "_verify_matched_binding",
        lambda report, sealed_candidate: None,
    )
    monkeypatch.setattr(
        sealed_experiment,
        "candidate_file_sha256",
        lambda path: "candidate-file-sha",
    )
    monkeypatch.setattr(sealed_experiment, "current_git_ref", lambda: "fixture-code-ref")
    monkeypatch.setattr(
        sealed_experiment.trained_final,
        "verify_trained_final_manifest",
        lambda manifest: None,
    )

    evaluator_started = {"value": False}

    def fail_after_observing_lock():
        assert final_lock.exists()
        receipt = json.loads(final_lock.read_text())
        assert receipt["status"] == "FINAL_NAMESPACE_CONSUMED_STARTED"
        assert receipt["candidate_manifest_sha256"] == "candidate-sha"
        assert receipt["final_manifest_sha256"] == "final-manifest-sha"
        evaluator_started["value"] = True
        raise RuntimeError("simulated failure after final seeds become observable")

    monkeypatch.setattr(sealed_experiment.trained_final, "main", fail_after_observing_lock)
    argv = [
        "fly-sniff-sealed-final",
        str(bundle),
        str(candidate_path),
        str(final_manifest_path),
        str(matched_path),
        str(trained_e002_path),
        "--output",
        str(output),
        "--final-lock",
        str(final_lock),
        "--arm-final",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError, match="simulated failure"):
        sealed_experiment.final_main()
    assert evaluator_started["value"] is True
    assert final_lock.exists()

    with pytest.raises(SystemExit, match="already consumed"):
        sealed_experiment.final_main()
