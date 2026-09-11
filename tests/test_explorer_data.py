import json

import pytest

from fly_sniff.explorer_data import (
    discover_candidates,
    import_explorer,
    normalize_explorer_document,
)


def test_normalizes_list_records_and_conservative_side_inference():
    document = [
        {"bodyId": 10, "type": "DNa02", "instance": "DNa02_L", "predicted_nt": "acetylcholine"},
        {"bodyId": "11", "cellType": "DNa02", "instance": "DNa02_R"},
    ]
    frame = normalize_explorer_document(document)
    assert frame.bodyId.tolist() == [10, 11]
    assert frame.side.tolist() == ["L", "R"]
    assert frame.body_id_source_field.tolist() == ["bodyId", "bodyId"]


def test_normalizes_numeric_id_keyed_map():
    document = {
        "1001": {"type": "PFL3", "instance": "PFL3_L"},
        "1002": {"type": "PFL3", "instance": "PFL3_R"},
    }
    frame = normalize_explorer_document(document)
    assert frame.bodyId.tolist() == [1001, 1002]
    assert frame.body_id_source_field.tolist() == ["json-map-key", "json-map-key"]


def test_normalizes_nested_container_and_discovers_regex():
    document = {
        "metadata": {"dataset": "male-cns:v1.0"},
        "neurons": [
            {"root_id": 7, "cell_type": "hDeltaC", "hemisphere": "left"},
            {"root_id": 8, "cell_type": "PFNa", "hemisphere": "right"},
            {"root_id": 9, "cell_type": "unrelated"},
        ],
    }
    frame = normalize_explorer_document(document)
    selected = discover_candidates(frame, r"hDeltaC|PFN")
    assert selected.bodyId.tolist() == [7, 8]
    assert selected.side.tolist() == ["L", "R"]


def test_rejects_duplicate_or_missing_body_ids():
    with pytest.raises(ValueError, match="duplicate body IDs"):
        normalize_explorer_document(
            [{"bodyId": 1, "type": "a"}, {"bodyId": 1, "type": "b"}]
        )
    with pytest.raises(ValueError, match="numeric unique body-ID"):
        normalize_explorer_document({"neurons": [{"type": "DNa02"}]})


def test_generic_id_field_is_not_body_authority():
    with pytest.raises(ValueError, match="numeric unique body-ID"):
        normalize_explorer_document({"neurons": [{"id": 123, "type": "DNa02"}]})


def test_import_hashes_exact_source_and_keeps_candidate_status(tmp_path):
    source = tmp_path / "neurons.json"
    source.write_text(json.dumps([{"body_id": 42, "type": "DNa02", "side": "left"}]))
    frame, manifest = import_explorer(
        str(source),
        pattern="DNa02",
        upstream_repository="example/repo",
        upstream_blob_sha="abc123",
    )
    assert frame.bodyId.tolist() == [42]
    assert manifest["qualification_status"] == "candidate"
    assert manifest["selected_body_ids"] == [42]
    assert len(manifest["source_sha256"]) == 64
    assert manifest["upstream_blob_sha"] == "abc123"
