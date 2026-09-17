from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fly_sniff.olfactory_door import sha256_file
from fly_sniff.olfactory_e006_audit import _canonical_sha
from fly_sniff.olfactory_o002 import (
    _load_study_matrix,
    _select_subset,
    run_o002_development,
)


def _audit_payload(long_sha: str) -> dict:
    payload = {
        "schema_version": 1,
        "protocol": "door-e006-audit-v1",
        "authority_id": "E006_odor_panel_receptor_responses",
        "input_receipt": {"long_form_sha256": long_sha},
        "development_subsets": {
            "candidates": [
                {
                    "study_id": "Study.Dev",
                    "observed_cells": 120,
                    "responding_units": 4,
                    "odor_names": 30,
                    "technique": "electrophysiology",
                    "data_type": "spikes",
                    "concentration": "10^-2",
                    "doi": "10.example/dev",
                    "selection_basis": "source_coverage_and_metadata_only",
                }
            ],
            "default_development_subset": {
                "study_id": "Study.Dev",
                "observed_cells": 120,
                "responding_units": 4,
                "odor_names": 30,
                "technique": "electrophysiology",
                "data_type": "spikes",
                "concentration": "10^-2",
                "doi": "10.example/dev",
                "selection_basis": "source_coverage_and_metadata_only",
            },
        },
        "gate": {
            "within_study_development_subset_allowed": True,
            "confirmatory_use_allowed": False,
            "development_feature_identity": "source_responding_unit",
        },
    }
    payload["audit_sha256"] = _canonical_sha(payload)
    return payload


def _write_long_form(path: Path, *, include_incomplete: bool = True) -> None:
    rows: list[dict] = []
    units = ["Or1", "Or2", "Or3", "Or4"]

    for unit_index, unit in enumerate(units):
        rows.append(
            {
                "responding_unit": unit,
                "source_row_id": 1,
                "odor_class": None,
                "odor_name": "sfr",
                "inchikey": "SFR",
                "cid": "SFR",
                "cas": "SFR",
                "study_id": "Study.Dev",
                "response_status": "observed",
                "response_value": float(unit_index),
            }
        )

    for odor_index in range(30):
        for unit_index, unit in enumerate(units):
            missing = include_incomplete and odor_index == 29 and unit == "Or4"
            rows.append(
                {
                    "responding_unit": unit,
                    "source_row_id": odor_index + 2,
                    "odor_class": "ester" if odor_index % 2 == 0 else "alcohol",
                    "odor_name": f"odor-{odor_index:02d}",
                    "inchikey": f"IK{odor_index:02d}",
                    "cid": str(1000 + odor_index),
                    "cas": f"CAS-{odor_index:02d}",
                    "study_id": "Study.Dev",
                    "response_status": "missing" if missing else "observed",
                    "response_value": None
                    if missing
                    else float((odor_index + 1) * (unit_index + 1)),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_select_subset_rejects_nonfrozen_study() -> None:
    audit = _audit_payload("abc")
    with pytest.raises(ValueError, match="not a frozen performance-blind development candidate"):
        _select_subset(audit, "Study.NotFrozen")


def test_load_study_matrix_excludes_sfr_and_does_not_impute(tmp_path: Path) -> None:
    long_path = tmp_path / "door-responses-long.csv"
    _write_long_form(long_path)
    matrix, metadata, diagnostics = _load_study_matrix(long_path, "Study.Dev")

    assert diagnostics["sfr_observation_cells_excluded"] == 4
    assert diagnostics["odor_rows_with_any_observation_excluding_sfr"] == 30
    assert diagnostics["complete_odor_rows_excluding_sfr"] == 29
    assert diagnostics["incomplete_odor_rows_excluding_sfr"] == 1
    assert diagnostics["complete_matrix_missing_cells"] == 0
    assert matrix.shape == (29, 4)
    assert metadata.shape[0] == 29
    assert "sfr" not in set(metadata["odor_name"])


def test_o002_run_is_development_only_and_content_addressed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    long_path = artifact / "door-responses-long.csv"
    _write_long_form(long_path, include_incomplete=False)

    audit = _audit_payload(sha256_file(long_path))
    audit_path = tmp_path / "e006-audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    output = tmp_path / "o002"
    report = run_o002_development(
        artifact,
        audit_path=audit_path,
        output_dir=output,
    )

    assert report["status"] == "development_complete_not_confirmatory"
    assert report["development_only"] is True
    assert report["confirmatory_use_allowed"] is False
    assert report["o003_accessed"] is False
    assert report["feature_identity"] == "source_responding_unit"
    assert report["matrix"]["complete_odor_rows_excluding_sfr"] == 30
    assert report["limitations"]["concentration_generalization"].startswith("not estimated")
    assert report["limitations"]["odor_identity_decoding"].startswith("not estimated")
    assert report["receipt_sha256"]
    assert (output / "o002-development-receipt.json").is_file()
    assert (output / "SUMMARY.txt").is_file()


def test_o002_refuses_to_overwrite_output(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    long_path = artifact / "door-responses-long.csv"
    _write_long_form(long_path, include_incomplete=False)
    audit = _audit_payload(sha256_file(long_path))
    audit_path = tmp_path / "e006-audit.json"
    audit_path.write_text(json.dumps(audit) + "\n")

    output = tmp_path / "o002"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite\n")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        run_o002_development(
            artifact,
            audit_path=audit_path,
            output_dir=output,
        )
