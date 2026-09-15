from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from fly_sniff.complex_odor_benchmark import FAST_PROTOCOL, run_fast_preflight
from fly_sniff.odor_authority import build_authority_from_long_csv, validate_authority


def _authority() -> dict:
    return {
        "protocol": "door-response-authority-v1",
        "source": {"name": "toy", "version": "test"},
        "receptors": ["R1", "R2", "R3"],
        "odorants": {
            "ethyl acetate": [1.0, 0.1, 0.0],
            "acetic acid": [0.0, 1.0, 0.1],
            "ethanol": [0.1, 0.0, 1.0],
            "1-pentanol": [0.2, 0.8, 0.2],
        },
    }


def _config() -> dict:
    return {
        "protocol": FAST_PROTOCOL,
        "dataset": "male-cns:v1.0",
        "seed": 17,
        "trials": 32,
        "arena": {"width": 10.0, "height": 6.0},
        "antenna_separation": 0.28,
        "odorants": {
            "target": "ethyl acetate",
            "distractors": ["acetic acid", "ethanol", "1-pentanol"],
        },
        "emission_range": [0.4, 1.0],
        "distractor_count_range": [1, 2],
        "plume": {"sigma_cross": 1.0, "sigma_upwind": 0.7, "downwind_decay": 4.0},
        "mixture_models": [
            "independent-saturating-receptors",
            "competitive-binding-sensitivity",
        ],
    }


def test_fast_preflight_is_deterministic_and_reports_both_models() -> None:
    a = run_fast_preflight(_config(), _authority())
    b = run_fast_preflight(_config(), _authority())
    assert a == b
    assert a["status"] == "engineering-preflight-not-E003b-qualification"
    assert set(a["model_reports"]) == {
        "independent-saturating-receptors",
        "competitive-binding-sensitivity",
    }
    for report in a["model_reports"].values():
        assert report["trials"] == 32
        assert 0.0 <= report["AUROC_external_template_readout"] <= 1.0
        assert 0.0 <= report["AUPRC_external_template_readout"] <= 1.0


def test_fast_preflight_refuses_missing_configured_odor() -> None:
    authority = _authority()
    authority["odorants"].pop("1-pentanol")
    with pytest.raises(ValueError, match="missing configured odorants"):
        run_fast_preflight(_config(), authority)


def test_authority_rejects_incomplete_odorants(tmp_path: Path) -> None:
    path = tmp_path / "responses.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["receptor", "odorant", "response"])
        writer.writeheader()
        writer.writerows(
            [
                {"receptor": "R1", "odorant": "A", "response": 1.0},
                {"receptor": "R2", "odorant": "A", "response": 0.0},
                {"receptor": "R1", "odorant": "B", "response": 0.5},
            ]
        )
    with pytest.raises(ValueError, match="not silently imputed"):
        build_authority_from_long_csv(path, source_name="toy", source_version="1")


def test_authority_builder_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "responses.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["receptor", "odorant", "response"])
        writer.writeheader()
        for receptor, values in {"A": [1.0, 0.0], "B": [0.2, 0.8]}.items():
            for index, response in enumerate(values, start=1):
                writer.writerow(
                    {"receptor": f"R{index}", "odorant": receptor, "response": response}
                )
    authority = build_authority_from_long_csv(path, source_name="toy", source_version="1")
    validate_authority(authority)
    assert authority["receptors"] == ["R1", "R2"]
    assert np.allclose(authority["odorants"]["A"], [1.0, 0.0])
    # Ensure the payload remains ordinary JSON data.
    json.dumps(authority)
