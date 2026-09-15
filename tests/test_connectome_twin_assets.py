from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fly_sniff.connectome_twin_assets import pack_swc_directory, read_swc


def _write_swc(path: Path) -> None:
    path.write_text(
        "# id type x y z radius parent\n"
        "1 1 0 0 0 1 -1\n"
        "2 3 10 0 0 1 1\n"
        "3 3 20 0 0 1 2\n"
        "4 3 20 10 0 1 2\n"
    )


def test_read_swc_preserves_centerline_topology(tmp_path: Path) -> None:
    path = tmp_path / "123.swc"
    _write_swc(path)
    nodes, edges = read_swc(path)
    assert nodes.shape == (4, 3)
    assert edges.tolist() == [[0, 1], [1, 2], [1, 3]]


def test_pack_swc_directory_converts_8nm_units_to_um(tmp_path: Path) -> None:
    _write_swc(tmp_path / "123.swc")
    packed = pack_swc_directory(tmp_path, [123], max_segments_per_neuron=10)
    neuron = packed["neurons"][0]
    assert neuron["body_id"] == 123
    assert neuron["lod_is_exact"] is True
    vertices = np.asarray(neuron["line_vertices_um"], dtype=float)
    assert np.isclose(vertices[:, 0].max(), 0.16)
    assert np.isclose(vertices[:, 1].max(), 0.08)


def test_pack_swc_directory_decimation_is_explicit(tmp_path: Path) -> None:
    _write_swc(tmp_path / "123.swc")
    packed = pack_swc_directory(tmp_path, [123], max_segments_per_neuron=2)
    neuron = packed["neurons"][0]
    assert neuron["source_segment_count"] == 3
    assert neuron["render_segment_count"] == 2
    assert neuron["lod_is_exact"] is False


def test_missing_body_id_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing MaleCNS skeleton"):
        pack_swc_directory(tmp_path, [999], max_segments_per_neuron=10)
