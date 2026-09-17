from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff import olfactory_door


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _authority(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "authority_id": "E006_odor_panel_receptor_responses",
        "source_id": "S002_door_data",
        "repository": "ropensci/DoOR.data",
        "commit": olfactory_door.EXPECTED_COMMIT,
        "tree": olfactory_door.EXPECTED_TREE,
        "checkout_policy": "clean_git_checkout_at_exact_commit",
        "required_index_git_blobs": dict(olfactory_door.INDEX_GIT_BLOBS),
        "expected_responding_units": olfactory_door.EXPECTED_UNITS,
        "normalization_policy": "none_during_ingestion",
        "aggregation_policy": "none_during_ingestion",
        "status": "source_frozen_ingestion_not_yet_executed",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_source_authority_is_frozen_and_cannot_drift(tmp_path: Path):
    authority_path = _authority(tmp_path / "authority.json")
    payload = json.loads(authority_path.read_text())
    olfactory_door.validate_source_authority(payload)
    payload["commit"] = "0" * 40
    with pytest.raises(ValueError, match="commit/tree"):
        olfactory_door.validate_source_authority(payload)


def test_r_csv2_requires_unnamed_rowname_offset(tmp_path: Path):
    good = tmp_path / "good.csv"
    _write(good, '"A";"B"\n"1";"x";"y"\n')
    header, rows = olfactory_door.read_r_csv2(good)
    assert header == [".row_id", "A", "B"]
    assert rows == [{".row_id": "1", "A": "x", "B": "y"}]

    bad = tmp_path / "bad.csv"
    _write(bad, '"A";"B"\n"x";"y"\n')
    with pytest.raises(ValueError, match="row-name offset"):
        olfactory_door.read_r_csv2(bad)


def test_response_parser_preserves_study_identity_and_missingness(tmp_path: Path):
    source = tmp_path / "ab4B.csv"
    _write(
        source,
        '"Class";"Name";"InChIKey";"CID";"CAS";"Stensmyr.2012.WT";"Other.Assay"\n'
        '"1";"other";"geosmin";"KEY";"123";"456";"146.4";NA\n',
    )
    rows, studies = olfactory_door.parse_response_unit(source, "ab4B")
    assert studies == ["Stensmyr.2012.WT", "Other.Assay"]
    observed, missing = rows
    assert observed["responding_unit"] == "ab4B"
    assert observed["odor_name"] == "geosmin"
    assert observed["study_id"] == "Stensmyr.2012.WT"
    assert observed["raw_response"] == "146.4"
    assert observed["response_value"] == pytest.approx(146.4)
    assert observed["response_status"] == "observed"
    assert missing["response_status"] == "missing"
    assert missing["response_value"] is None
    assert missing["raw_response"] is None


def test_mapping_candidates_preserve_ab4b_da2_identity_without_forcing_or56a_row():
    rows = [
        {
            "receptor": "ab4B",
            "sensillum": "ab4",
            "OSN": "ab4B",
            "glomerulus": "DA2",
            "code": "DA2",
            "code.OSN": "ab4B",
        },
        {
            "receptor": "Or56a",
            "sensillum": "ab4",
            "OSN": "ab4B",
            "glomerulus": "DA2",
            "code": "",
            "code.OSN": "",
        },
    ]
    candidates = olfactory_door.mapping_candidates("ab4B", rows)
    assert len(candidates) == 2
    assert {row["glomerulus"] for row in candidates} == {"DA2"}
    assert {row["receptor"] for row in candidates} == {"ab4B", "Or56a"}


def _make_fake_door_checkout(root: Path) -> None:
    data = root / "data"
    data.mkdir(parents=True)
    units = ["ab4B", *[f"U{index:02d}" for index in range(1, 78)]]
    ors = ['"OR"'] + [f'"{index}";"{unit}"' for index, unit in enumerate(units, start=1)]
    _write(data / "ORs.csv", "\n".join(ors) + "\n")
    _write(
        data / "door_mappings.csv",
        '"receptor";"sensillum";"OSN";"glomerulus";"co.receptor";"coexpressing";'
        '"related1";"related2";"related3";"related4";"related5";"related6";"Ors";'
        '"sensillum.type";"adult";"larva";"dataset.existing";"comment";"code";"code.OSN"\n'
        '"1";"ab4B";"ab4";"ab4B";"DA2";"Orco";"Or33a+56a";"Or33a";"Or56a";'
        '"";"";"";"";"Or33a+56a";"antennal basiconic";TRUE;NA;TRUE;"";"DA2";"ab4B"\n',
    )
    template = (
        '"Class";"Name";"InChIKey";"CID";"CAS";"StudyA";"StudyB"\n'
        '"1";"other";"geosmin";"KEY";"123";"456";"10";NA\n'
    )
    for unit in units:
        _write(data / f"{unit}.csv", template)


def test_ingestion_is_lossless_and_content_addressed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "door"
    _make_fake_door_checkout(source)
    authority_path = _authority(tmp_path / "authority.json")

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD^{commit}"):
            return olfactory_door.EXPECTED_COMMIT
        if args == ("rev-parse", "HEAD^{tree}"):
            return olfactory_door.EXPECTED_TREE
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(olfactory_door, "_git", fake_git)
    monkeypatch.setattr(olfactory_door, "INDEX_GIT_BLOBS", {})
    authority = json.loads(authority_path.read_text())
    authority["required_index_git_blobs"] = {}
    authority_path.write_text(json.dumps(authority))
    receipt = olfactory_door.ingest_door(
        source,
        tmp_path / "out",
        authority_path=authority_path,
    )
    assert receipt["responding_unit_count"] == 78
    assert receipt["response_cells"] == 156
    assert receipt["observed_response_cells"] == 78
    assert receipt["missing_response_cells"] == 78
    assert receipt["geosmin_observed_cells"] == 78
    assert receipt["normalization_applied"] is False
    assert receipt["aggregation_applied"] is False
    assert receipt["missingness_preserved"] is True
    assert len(receipt["receipt_sha256"]) == 64
    assert len(receipt["source_authority_sha256"]) == 64
    assert (tmp_path / "out" / "door-responses-long.csv").is_file()
    assert (tmp_path / "out" / "door-unit-mappings.json").is_file()


def test_dirty_source_checkout_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "door"
    source.mkdir()

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD^{commit}"):
            return olfactory_door.EXPECTED_COMMIT
        if args == ("rev-parse", "HEAD^{tree}"):
            return olfactory_door.EXPECTED_TREE
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return " M data/ab4B.csv"
        raise AssertionError(args)

    monkeypatch.setattr(olfactory_door, "_git", fake_git)
    with pytest.raises(ValueError, match="must be clean"):
        olfactory_door.verify_checkout(source)
