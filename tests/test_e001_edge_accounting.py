import pandas as pd
import pytest

from fly_sniff.e001_evidence import _edge_context


def _write_stage(root, name, weight):
    stage = root / name
    stage.mkdir(parents=True)
    pd.DataFrame(
        [
            {"bodyId": 1, "type": "FB5AB", "instance": "FB5AB_L"},
            {"bodyId": 2, "type": "hDeltaC", "instance": "hDeltaC_1"},
        ]
    ).to_parquet(stage / "nodes.parquet", index=False)
    pd.DataFrame([{"source": 1, "target": 2, "weight": weight}]).to_parquet(
        stage / "edges.parquet", index=False
    )


def test_same_biological_edge_is_counted_once_with_stage_provenance(tmp_path):
    _write_stage(tmp_path, "a", 7)
    _write_stage(tmp_path, "b", 7)
    report = {"stages": [{"name": "a"}, {"name": "b"}]}
    nodes = {
        1: {"type": "FB5AB", "instance": "FB5AB_L"},
        2: {"type": "hDeltaC", "instance": "hDeltaC_1"},
    }
    context = _edge_context(tmp_path, report, {1, 2}, nodes)
    assert context[1]["outgoing_edge_count"] == 1
    assert context[1]["outgoing_weight"] == 7.0
    assert context[2]["incoming_edge_count"] == 1
    assert context[2]["incoming_weight"] == 7.0
    assert context[1]["top_outgoing"][0]["stages_seen"] == ["a", "b"]
    assert context[1]["accounting"]["duplicate_stage_occurrences_collapsed_global"] == 1


def test_duplicate_stage_edge_with_different_weight_is_rejected(tmp_path):
    _write_stage(tmp_path, "a", 7)
    _write_stage(tmp_path, "b", 8)
    report = {"stages": [{"name": "a"}, {"name": "b"}]}
    with pytest.raises(ValueError, match="inconsistent MaleCNS structural weight"):
        _edge_context(tmp_path, report, {1, 2}, {})
