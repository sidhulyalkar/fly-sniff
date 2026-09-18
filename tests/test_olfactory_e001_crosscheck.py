from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fly_sniff.olfactory_door import sha256_file
from fly_sniff.olfactory_e001_crosscheck import (
    EXPECTED_DOOR_COMMIT,
    crosscheck_e001_door,
)


def _artifact(root: Path, *, response: float = 146.4) -> Path:
    root.mkdir()
    long_path = root / "door-responses-long.csv"
    frame = pd.DataFrame(
        [
            {
                "responding_unit": "ab4B",
                "source_row_id": "240",
                "odor_class": "alcohol",
                "odor_name": "geosmin",
                "inchikey": "JLPUXFOGCDVKGO-GRYCIOLGSA-N",
                "cid": "15559490",
                "cas": "16423-19-1",
                "study_id": "Stensmyr.2012.WT",
                "response_status": "observed",
                "response_value": response,
            },
            {
                "responding_unit": "ab4B",
                "source_row_id": "4",
                "odor_class": "amine",
                "odor_name": "putrescine",
                "inchikey": "KIDHWZJUCRJVML-UHFFFAOYSA-N",
                "cid": "1045",
                "cas": "110-60-1",
                "study_id": "Stensmyr.2012.WT",
                "response_status": "observed",
                "response_value": 0.4,
            },
            {
                "responding_unit": "Or7a",
                "source_row_id": "240",
                "odor_class": "alcohol",
                "odor_name": "geosmin",
                "inchikey": "JLPUXFOGCDVKGO-GRYCIOLGSA-N",
                "cid": "15559490",
                "cas": "16423-19-1",
                "study_id": "Stensmyr.2012.WT",
                "response_status": "missing",
                "response_value": None,
            },
        ]
    )
    frame.to_csv(long_path, index=False)
    receipt = {
        "protocol": "door-e006-source-resolved-ingestion-v0",
        "source_commit": EXPECTED_DOOR_COMMIT,
        "long_form_artifact": {
            "path": str(long_path),
            "sha256": sha256_file(long_path),
        },
    }
    (root / "door-e006-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return root


def test_e001_door_crosscheck_matches_frozen_ab4b_geosmin(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "e006")
    output = tmp_path / "out"
    report = crosscheck_e001_door(artifact, output_dir=output)

    assert report["status"] == (
        "source_transcription_crosscheck_complete_not_numeric_calibration"
    )
    assert report["input"]["door_commit"] == EXPECTED_DOOR_COMMIT
    assert report["geosmin"]["observed_cells"] == 1
    assert report["geosmin"]["ab4B_raw_response"] == 146.4
    assert report["geosmin"]["ab4B_exact_source_transcription_match"] is True
    assert report["interpretation"]["numeric_model_parameter_allowed"] is False
    assert report["interpretation"]["cross_study_scale_comparability_established"] is False
    assert report["interpretation"]["raw_trial_data_established"] is False
    assert (output / "e001-door-crosscheck-receipt.json").is_file()
    assert (output / "SUMMARY.txt").is_file()


def test_e001_door_crosscheck_rejects_changed_raw_source_value(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "e006", response=145.0)
    with pytest.raises(ValueError, match="raw response changed"):
        crosscheck_e001_door(artifact, output_dir=tmp_path / "out")


def test_e001_door_crosscheck_rejects_wrong_source_commit(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "e006")
    receipt_path = artifact / "door-e006-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["source_commit"] = "wrong"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="exact frozen DoOR source commit"):
        crosscheck_e001_door(artifact, output_dir=tmp_path / "out")


def test_e001_door_crosscheck_rejects_tampered_long_form(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "e006")
    long_path = artifact / "door-responses-long.csv"
    long_path.write_text(long_path.read_text() + "\n")
    with pytest.raises(ValueError, match="long-form sha256 mismatch"):
        crosscheck_e001_door(artifact, output_dir=tmp_path / "out")


def test_e001_door_crosscheck_refuses_overwrite(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "e006")
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        crosscheck_e001_door(artifact, output_dir=output)
