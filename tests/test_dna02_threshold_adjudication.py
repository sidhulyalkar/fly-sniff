from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from scipy.signal import find_peaks

import fly_sniff.dna02_threshold_adjudication as adjudication
from fly_sniff.dna02_threshold_manifest import freeze_manifest
from fly_sniff.freeze import canonical_sha256


def test_channel_loader_never_requests_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.mat"
    savemat(
        path,
        {
            "ephys_SR": np.array([[10_000.0]]),
            "ephys_A": np.arange(100.0),
            "ephys_B": np.arange(100.0),
            "yaw": np.arange(100.0),
            "fwd": np.arange(100.0),
            "lat": np.arange(100.0),
        },
    )
    seen: list[list[str]] = []
    real_loadmat = adjudication.loadmat

    def guarded_loadmat(*args, **kwargs):
        fields = list(kwargs.get("variable_names", []))
        seen.append(fields)
        assert "yaw" not in fields
        assert "fwd" not in fields
        assert "lat" not in fields
        return real_loadmat(*args, **kwargs)

    monkeypatch.setattr(adjudication, "loadmat", guarded_loadmat)
    fs, trace = adjudication._load_channel(path, "ephys_A")
    assert fs == 10_000.0
    assert trace.size == 100
    assert seen == [["ephys_SR", "ephys_A"]]


def test_channel_loader_rejects_behavior_field(tmp_path: Path) -> None:
    path = tmp_path / "source.mat"
    savemat(path, {"ephys_SR": [[10_000.0]], "yaw": np.arange(10.0)})
    with pytest.raises(ValueError, match="not allowlisted"):
        adjudication._load_channel(path, "yaw")


def test_nested_candidate_qc_uses_one_peak_set() -> None:
    trace = np.zeros(4000, dtype=float)
    locations = np.array([200, 700, 1200, 1800, 2500, 3300])
    trace[locations] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    peaks, properties = find_peaks(trace, prominence=(1.0, None))
    prominences = np.asarray(properties["prominences"], dtype=float)

    low, low_peaks = adjudication._candidate_qc(
        trace,
        peaks,
        prominences,
        fs=1000.0,
        quantile=0.95,
        threshold=2.0,
        seed_text="synthetic",
    )
    high, high_peaks = adjudication._candidate_qc(
        trace,
        peaks,
        prominences,
        fs=1000.0,
        quantile=0.999,
        threshold=5.0,
        seed_text="synthetic",
    )

    assert low["event_count"] == 5
    assert high["event_count"] == 2
    assert set(high_peaks).issubset(set(low_peaks))
    assert low["waveform"]["sampled_waveform_count"] == 5
    assert high["waveform"]["sampled_waveform_count"] == 2


def test_bilateral_coincidence_detects_zero_lag_artifact() -> None:
    left = {0.99: np.array([100, 500, 900], dtype=np.int64)}
    right = {0.99: np.array([101, 700, 901], dtype=np.int64)}
    result = adjudication._bilateral_coincidence(left, right, fs=10_000.0)
    assert len(result) == 1
    assert result[0]["left_fraction_with_right_within_0.2ms"] == pytest.approx(2 / 3)
    assert result[0]["right_fraction_with_left_within_0.2ms"] == pytest.approx(2 / 3)


def _synthetic_qc() -> dict:
    files = []
    aliases = ("a2_d_08", "a2_d_12", "a2_d_13", "a2_d_14")
    for index, alias in enumerate(aliases):
        channels = []
        for source_field, side in (("ephys_A", "L"), ("ephys_B", "R")):
            channels.append(
                {
                    "source_field": source_field,
                    "soma_side": side,
                    "minimum_prominence_computed": 1.0,
                    "candidate_qc": [
                        {
                            "prominence_quantile": 0.99,
                            "threshold": 3.0 + index,
                            "event_count": 100,
                            "event_rate_hz": 5.0,
                            "refractory_violation_fraction_lt_1ms": 0.001,
                            "short_isi_fraction_lt_2ms": 0.002,
                            "block_rate_cv": 0.1,
                            "waveform": {
                                "median_template_correlation": 0.95,
                                "fwhm_ms": 0.8,
                            },
                        }
                    ],
                }
            )
        files.append(
            {
                "fly_alias": alias,
                "filename": f"{alias}.mat",
                "source_sha256": f"{index + 1:064x}",
                "ephys_sampling_rate_hz": 10_000.0,
                "channels": channels,
                "bilateral_candidate_coincidence": [],
            }
        )

    payload = {
        "schema": "fly-sniff-dna02-threshold-adjudication-qc-v1",
        "status": "NEURAL_QC_COMPLETE_PENDING_HUMAN_THRESHOLD_DECISIONS",
        "source_contract_sha256": "1" * 64,
        "source_byte_evidence_sha256": "2" * 64,
        "prominence_audit_sha256": "3" * 64,
        "candidate_quantiles": [0.99],
        "input_fields_allowlist": ["ephys_SR", "ephys_A", "ephys_B"],
        "behavior_fields_loaded": False,
        "yaw_loaded": False,
        "navigation_performance_used": False,
        "figure3c_statistic_computed": False,
        "thresholds_frozen": False,
        "automatic_threshold_selection": False,
        "selection_basis_allowlist": ["waveform_stereotypy"],
        "files": files,
        "next_allowed_action": "review ephys only",
        "forbidden_interpretation": [],
    }
    payload["qc_sha256"] = canonical_sha256(payload)
    return payload


def _synthetic_decisions(qc: dict) -> dict:
    decisions = []
    for file_item in qc["files"]:
        for channel in file_item["channels"]:
            candidate = channel["candidate_qc"][0]
            decisions.append(
                {
                    "fly_alias": file_item["fly_alias"],
                    "filename": file_item["filename"],
                    "source_sha256": file_item["source_sha256"],
                    "source_field": channel["source_field"],
                    "soma_side": channel["soma_side"],
                    "candidate_thresholds": [
                        {
                            "prominence_quantile": candidate["prominence_quantile"],
                            "threshold": candidate["threshold"],
                        }
                    ],
                    "selected_prominence_quantile": candidate["prominence_quantile"],
                    "selected_threshold": candidate["threshold"],
                    "selection_basis": [
                        "raw_trace_review",
                        "waveform_stereotypy",
                        "refractory_violations",
                    ],
                    "rationale": (
                        "Selected from neural-only waveform and refractory diagnostics "
                        "before behavior review."
                    ),
                }
            )
    return {
        "schema": "fly-sniff-dna02-threshold-decisions-v1",
        "qc_sha256": qc["qc_sha256"],
        "prominence_audit_sha256": qc["prominence_audit_sha256"],
        "adjudication_method": "manual_ephys_only_review",
        "reviewer": "test-reviewer",
        "behavior_fields_reviewed": False,
        "yaw_reviewed": False,
        "navigation_performance_used": False,
        "figure3c_statistic_reviewed": False,
        "allowed_selection_basis": [
            "raw_trace_review",
            "waveform_stereotypy",
            "refractory_violations",
        ],
        "decisions": decisions,
    }


def test_manifest_freezer_requires_complete_eight_channel_decisions(tmp_path: Path) -> None:
    qc = _synthetic_qc()
    decisions = _synthetic_decisions(qc)
    qc_path = tmp_path / "qc.json"
    decisions_path = tmp_path / "decisions.json"
    output_path = tmp_path / "manifest.json"
    qc_path.write_text(json.dumps(qc))
    decisions_path.write_text(json.dumps(decisions))

    result = freeze_manifest(
        qc_path=qc_path,
        decisions_path=decisions_path,
        output_path=output_path,
        code_ref="a" * 40,
    )
    assert result["status"] == "THRESHOLDS_FROZEN_BEFORE_BEHAVIOR"
    assert result["threshold_count"] == 8
    assert result["behavior_fields_reviewed"] is False
    assert result["yaw_reviewed"] is False
    assert result["navigation_performance_used"] is False
    assert len(result["manifest_sha256"]) == 64

    decisions["decisions"] = decisions["decisions"][:-1]
    decisions_path.write_text(json.dumps(decisions))
    with pytest.raises(ValueError, match="exactly eight"):
        freeze_manifest(
            qc_path=qc_path,
            decisions_path=decisions_path,
            output_path=output_path,
            code_ref="a" * 40,
        )


def test_manifest_freezer_rejects_unaudited_or_rate_based_choice(tmp_path: Path) -> None:
    qc = _synthetic_qc()
    decisions = _synthetic_decisions(qc)
    qc_path = tmp_path / "qc.json"
    decisions_path = tmp_path / "decisions.json"
    output_path = tmp_path / "manifest.json"
    qc_path.write_text(json.dumps(qc))

    decisions["decisions"][0]["selected_threshold"] = 999.0
    decisions_path.write_text(json.dumps(decisions))
    with pytest.raises(ValueError, match="not an exact audited candidate"):
        freeze_manifest(
            qc_path=qc_path,
            decisions_path=decisions_path,
            output_path=output_path,
            code_ref="a" * 40,
        )

    decisions = _synthetic_decisions(qc)
    decisions["decisions"][0]["selection_basis"] = ["event_rate"]
    decisions_path.write_text(json.dumps(decisions))
    with pytest.raises(ValueError, match="forbidden values"):
        freeze_manifest(
            qc_path=qc_path,
            decisions_path=decisions_path,
            output_path=output_path,
            code_ref="a" * 40,
        )
