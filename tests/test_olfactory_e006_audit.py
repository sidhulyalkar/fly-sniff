import json
from pathlib import Path

import pandas as pd
import pytest

from fly_sniff.olfactory_door import sha256_file
from fly_sniff.olfactory_e006_audit import (
    _canonical_sha,
    _coverage_tables,
    _dataset_metadata,
    _development_subset_candidates,
    _mapping_summary,
    _validate_receipt,
)


def test_mapping_summary_preserves_one_to_many_identity() -> None:
    mapping = {
        "Or7a": [{"receptor": "Or7a", "glomerulus": "DL5"}],
        "ab4B": [
            {"receptor": "ab4B", "glomerulus": "DA2"},
            {"receptor": "Or33a", "glomerulus": "DA2"},
            {"receptor": "Or56a", "glomerulus": "DA2"},
        ],
    }
    report = _mapping_summary(mapping)
    assert report["candidate_count_histogram"] == {"1": 1, "3": 1}
    assert report["zero_mapping_units"] == []
    assert report["multiple_mapping_units"] == ["ab4B"]
    assert report["multiple_mapping_count"] == 1
    assert len(report["multiple_mapping_records"]["ab4B"]) == 3


def test_dataset_metadata_joins_on_dataset_id_not_citation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "door_dataset_info.csv").write_text(
        '"dataset";"study";"technique";"data.type";"concentration";"DOI"\n'
        '"1";"Hallem.2006.EN";"Hallem et.al. 2006";"electrophysiology";'
        '"spikes";"10^-2";"10.1016/j.cell.2006.01.050"\n'
    )
    study = pd.DataFrame(
        [
            {
                "study_id": "Hallem.2006.EN",
                "observed_cells": 2664,
                "responding_units": 24,
                "odor_names": 111,
            }
        ]
    ).set_index("study_id")
    joined, report = _dataset_metadata(tmp_path, study)
    assert report["complete_join"] is True
    assert report["missing_metadata_for_response_studies"] == []
    assert joined.loc[0, "dataset"] == "Hallem.2006.EN"
    assert joined.loc[0, "study"] == "Hallem et.al. 2006"


def test_coverage_counts_only_observed_odors_and_units() -> None:
    df = pd.DataFrame(
        [
            {
                "responding_unit": "Or1",
                "odor_name": "odor-a",
                "study_id": "Study.A",
                "response_status": "observed",
                "response_value": 1.0,
            },
            {
                "responding_unit": "Or1",
                "odor_name": "odor-b",
                "study_id": "Study.A",
                "response_status": "missing",
                "response_value": None,
            },
            {
                "responding_unit": "Or2",
                "odor_name": "odor-a",
                "study_id": "Study.A",
                "response_status": "observed",
                "response_value": 2.0,
            },
            {
                "responding_unit": "Or2",
                "odor_name": "odor-b",
                "study_id": "Study.A",
                "response_status": "missing",
                "response_value": None,
            },
        ]
    )
    study, units, odors = _coverage_tables(df)
    assert study.loc["Study.A", "odor_names"] == 1
    assert study.loc["Study.A", "responding_units"] == 2
    assert units.loc["Or1", "odor_names"] == 1
    assert odors.loc["odor-b", "observed_cells"] == 0


def test_dataset_metadata_resolves_only_frozen_muench_aliases(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "door_dataset_info.csv").write_text(
        '"dataset";"study";"technique";"data.type";"concentration";"DOI"\n'
        '"1";"Muench.2015.AntGC1";"Muench et al. 2016";"calcium imaging";'
        '"mean deltaF/F";"10^-2 -  (vol/vol)";""\n'
        '"2";"Muench.2015.AntGC3";"Muench et al. 2016";"calcium imaging";'
        '"mean deltaF/F";"10^-2 -  (vol/vol)";""\n'
    )
    study = pd.DataFrame(
        [
            {
                "study_id": "Muench.2016.AntGC1",
                "observed_cells": 426,
                "responding_units": 4,
                "odor_names": 113,
            },
            {
                "study_id": "Muench.2016.AntGC3",
                "observed_cells": 108,
                "responding_units": 1,
                "odor_names": 108,
            },
        ]
    ).set_index("study_id")
    joined, report = _dataset_metadata(tmp_path, study)
    assert report["complete_join"] is True
    assert report["missing_metadata_for_response_studies"] == []
    assert report["resolved_source_aliases"] == {
        "Muench.2016.AntGC1": "Muench.2015.AntGC1",
        "Muench.2016.AntGC3": "Muench.2015.AntGC3",
    }
    assert set(joined["response_study_id"]) == {
        "Muench.2016.AntGC1",
        "Muench.2016.AntGC3",
    }


def test_development_subset_selection_is_performance_blind() -> None:
    joined = pd.DataFrame(
        [
            {
                "study_id": "Hallem.2006.EN",
                "observed_cells": 2664,
                "responding_units": 24,
                "odor_names": 111,
                "technique": "electrophysiology",
                "data.type": "spikes",
                "concentration": "10^-2",
                "DOI": "10.1016/j.cell.2006.01.050",
            },
            {
                "study_id": "Large.But.No.Concentration",
                "observed_cells": 9999,
                "responding_units": 50,
                "odor_names": 500,
                "technique": "electrophysiology",
                "data.type": "spikes",
                "concentration": "",
                "DOI": "",
            },
        ]
    )
    report = _development_subset_candidates(joined)
    assert report["rule"]["performance_blind"] is True
    assert report["default_development_subset"]["study_id"] == "Hallem.2006.EN"
    assert [row["study_id"] for row in report["candidates"]] == ["Hallem.2006.EN"]


def _write_receipt(tmp_path: Path) -> Path:
    long_path = tmp_path / "door-responses-long.csv"
    mapping_path = tmp_path / "door-unit-mappings.json"
    long_path.write_text(
        "responding_unit,odor_name,study_id,response_status,response_value\n"
        "ab4B,geosmin,Stensmyr.2012.WT,observed,146.4\n"
    )
    mapping_path.write_text('{"ab4B": [{"receptor": "Or56a", "glomerulus": "DA2"}]}\n')
    receipt = {
        "protocol": "door-e006-source-resolved-ingestion-v0",
        "authority_id": "E006_odor_panel_receptor_responses",
        "source_commit": "db323a496577c4b4a72b5c2fcd1859e07521ffb5",
        "source_tree": "2b673e851cd0760715c1b74d6b3a6035c1af72cd",
        "normalization_applied": False,
        "aggregation_applied": False,
        "missingness_preserved": True,
        "study_identity_preserved": True,
        "long_form_artifact": {"sha256": sha256_file(long_path)},
        "mapping_artifact": {"sha256": sha256_file(mapping_path)},
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    path = tmp_path / "door-e006-receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def test_validate_receipt_accepts_canonical_self_hash(tmp_path: Path) -> None:
    _write_receipt(tmp_path)
    report = _validate_receipt(tmp_path)
    assert report["receipt_sha256"]


def test_validate_receipt_fails_closed_on_artifact_tamper(tmp_path: Path) -> None:
    _write_receipt(tmp_path)
    (tmp_path / "door-unit-mappings.json").write_text("{}\n")
    with pytest.raises(ValueError, match="mapping artifact hash mismatch"):
        _validate_receipt(tmp_path)


def test_validate_receipt_fails_closed_on_receipt_tamper(tmp_path: Path) -> None:
    path = _write_receipt(tmp_path)
    receipt = json.loads(path.read_text())
    receipt["aggregation_applied"] = True
    path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(ValueError, match="canonical receipt hash mismatch"):
        _validate_receipt(tmp_path)
