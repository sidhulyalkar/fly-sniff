from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fly_sniff.morphology_context import _parse_swc_segments, build_soma_context


def test_soma_context_exports_only_measured_xyz(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.feather"
    frame = pd.DataFrame(
        {
            "bodyId": [3, 1, 2, 4],
            "status": ["Traced", "Traced", "Traced", "Orphan"],
            "somaLocation": [
                [30.0, 31.0, 32.0],
                [10.0, 11.0, 12.0],
                None,
                [40.0, 41.0, 42.0],
            ],
            "type": ["C", "A", "B", "D"],
            "superclass": ["central", "optic", "central", "other"],
        }
    )
    frame.to_feather(annotations)

    output = tmp_path / "context.json"
    payload = build_soma_context(annotations, output, max_points=10)

    assert payload["geometry_kind"] == "measured_soma_xyz"
    assert payload["selection"]["valid_soma_rows"] == 2
    assert payload["selection"]["exported_points"] == 2
    assert [row["body_id"] for row in payload["points"]] == [1, 3]
    assert payload["bounds"]["min"] == [10.0, 11.0, 12.0]
    assert payload["bounds"]["max"] == [30.0, 31.0, 32.0]
    assert json.loads(output.read_text())["dataset"] == "male-cns:v1.0"


def test_soma_context_bounded_sampling_is_deterministic(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.feather"
    frame = pd.DataFrame(
        {
            "bodyId": list(range(20)),
            "status": ["Traced"] * 20,
            "somaLocation": [[float(i), float(i + 1), float(i + 2)] for i in range(20)],
        }
    )
    frame.to_feather(annotations)

    first = build_soma_context(annotations, tmp_path / "a.json", max_points=5)
    second = build_soma_context(annotations, tmp_path / "b.json", max_points=5)

    assert first["points"] == second["points"]
    assert len(first["points"]) == 5


def test_soma_context_refuses_missing_position_column(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.feather"
    pd.DataFrame({"bodyId": [1]}).to_feather(annotations)

    with pytest.raises(ValueError, match="missing required columns"):
        build_soma_context(annotations, tmp_path / "out.json")


def test_swc_parser_preserves_parent_child_geometry() -> None:
    raw = (
        b"# synthetic\n"
        b"1 1 10 20 30 1 -1\n"
        b"2 3 11 21 31 1 1\n"
        b"3 3 12 22 32 1 2\n"
    )
    segments = _parse_swc_segments(raw)

    assert segments == [
        [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
        [11.0, 21.0, 31.0, 12.0, 22.0, 32.0],
    ]
