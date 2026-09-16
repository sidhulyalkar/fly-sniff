from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

import fly_sniff.dna02_threshold_audit as audit
from fly_sniff.freeze import canonical_sha256


def test_prominence_summary_is_neural_only_and_deterministic() -> None:
    trace = np.zeros(1000, dtype=float)
    trace[[100, 300, 700]] = [1.0, 2.0, 4.0]
    result = audit._prominence_summary(trace, 100.0)
    assert result["sample_count"] == 1000
    assert result["duration_s"] == 10.0
    assert result["positive_prominence_count"] == 3
    assert result["candidate_sweep"][-1]["events_at_or_above"] == 1


def test_loader_requests_only_neural_allowlist_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "source.mat"
    savemat(
        path,
        {
            "ephys_SR": np.array([[10_000.0]]),
            "ephys_A": np.arange(10.0),
            "ephys_B": np.arange(10.0),
            "yaw": np.arange(10.0),
        },
    )
    seen: list[list[str]] = []
    real_loadmat = audit.loadmat

    def guarded_loadmat(*args, **kwargs):
        fields = list(kwargs.get("variable_names", []))
        seen.append(fields)
        assert "yaw" not in fields
        assert "fwd" not in fields
        assert "lat" not in fields
        return real_loadmat(*args, **kwargs)

    monkeypatch.setattr(audit, "loadmat", guarded_loadmat)
    fs, trace = audit._load_neural_channel(path, "ephys_A")
    assert fs == 10_000.0
    assert trace.size == 10
    assert seen == [["ephys_SR", "ephys_A"]]


def test_loader_rejects_behavior_field_even_if_present(tmp_path: Path) -> None:
    path = tmp_path / "source.mat"
    savemat(path, {"ephys_SR": [[10_000.0]], "yaw": np.arange(10.0)})
    with pytest.raises(ValueError, match="not allowlisted"):
        audit._load_neural_channel(path, "yaw")


def test_audit_hash_contract_shape() -> None:
    payload = {
        "schema": "fly-sniff-dna02-prominence-audit-v1",
        "yaw_loaded": False,
        "navigation_performance_used": False,
        "thresholds_frozen": False,
    }
    digest = canonical_sha256(payload)
    assert len(digest) == 64
    assert json.loads(json.dumps(payload))["yaw_loaded"] is False
