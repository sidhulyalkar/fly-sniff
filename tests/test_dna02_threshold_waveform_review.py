from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.dna02_threshold_waveform_review import _validated_qc, render_waveform_review
from fly_sniff.freeze import canonical_sha256


def _qc_payload() -> dict:
    files = []
    for fly_index, alias in enumerate(("a2_d_08", "a2_d_12", "a2_d_13", "a2_d_14")):
        channels = []
        for side, field in (("L", "ephys_A"), ("R", "ephys_B")):
            candidates = []
            for index, quantile in enumerate((0.995, 0.999, 0.9995, 0.9999)):
                amplitude = 1.0 + index + fly_index * 0.1
                candidates.append(
                    {
                        "prominence_quantile": quantile,
                        "threshold": amplitude,
                        "event_count": 100 - index,
                        "event_rate_hz": 10.0 - index,
                        "refractory_violation_fraction_lt_1ms": 0.01,
                        "short_isi_fraction_lt_2ms": 0.02,
                        "block_rate_cv": 0.1,
                        "waveform": {
                            "time_ms": [-1.0, 0.0, 1.0],
                            "median_waveform": [0.0, amplitude, 0.0],
                            "mad_waveform": [0.1, 0.1, 0.1],
                            "median_peak_amplitude": amplitude,
                            "median_template_correlation": 0.95,
                        },
                    }
                )
            channels.append(
                {
                    "soma_side": side,
                    "source_field": field,
                    "candidate_qc": candidates,
                }
            )
        files.append(
            {
                "fly_alias": alias,
                "filename": f"{alias}.mat",
                "source_sha256": f"{fly_index + 1:064x}",
                "channels": channels,
            }
        )

    payload = {
        "schema": "fly-sniff-dna02-threshold-adjudication-qc-v1",
        "status": "NEURAL_QC_COMPLETE_PENDING_HUMAN_THRESHOLD_DECISIONS",
        "prominence_audit_sha256": "a" * 64,
        "behavior_fields_loaded": False,
        "yaw_loaded": False,
        "navigation_performance_used": False,
        "figure3c_statistic_computed": False,
        "automatic_threshold_selection": False,
        "thresholds_frozen": False,
        "files": files,
    }
    payload["qc_sha256"] = canonical_sha256(payload)
    return payload


def test_waveform_review_renders_eight_channels_without_selection(tmp_path: Path) -> None:
    qc_path = tmp_path / "qc.json"
    qc_path.write_text(json.dumps(_qc_payload()))
    out = tmp_path / "review"

    receipt = render_waveform_review(qc_path, output_dir=out)

    assert receipt["status"] == "HIGH_THRESHOLD_WAVEFORMS_RENDERED_NO_SELECTION"
    assert receipt["automatic_threshold_selection"] is False
    assert receipt["thresholds_frozen"] is False
    assert len(receipt["plots"]) == 8
    assert all((out / item["plot"]).is_file() for item in receipt["plots"])
    unhashed = dict(receipt)
    observed = unhashed.pop("review_sha256")
    assert observed == canonical_sha256(unhashed)


def test_waveform_review_rejects_behavior_contamination(tmp_path: Path) -> None:
    payload = _qc_payload()
    payload["behavior_fields_loaded"] = True
    payload.pop("qc_sha256")
    payload["qc_sha256"] = canonical_sha256(payload)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="behavior_fields_loaded"):
        _validated_qc(path)


def test_waveform_review_rejects_tampered_hash(tmp_path: Path) -> None:
    payload = _qc_payload()
    payload["status"] = "tampered"
    path = tmp_path / "bad-hash.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hash mismatch"):
        _validated_qc(path)
