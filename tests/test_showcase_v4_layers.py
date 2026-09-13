from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fly_sniff.showcase_v4_layers import (
    density_rgba,
    pfl3_population_frame,
    social_display_frame_limit,
    viewer_density_field,
)


def test_viewer_density_requires_complete_snapshot() -> None:
    payload = {
        "arena": {"width": 10.0, "height": 6.0},
        "config": {"plume": {"puff_mass": 1.0}},
    }
    current = {
        "plume_snapshot": {"complete": False},
        "plume": [[2.0, 3.0, 0.5]],
    }
    assert viewer_density_field(current, payload) is None


def test_viewer_density_reconstructs_complete_snapshot() -> None:
    payload = {
        "arena": {"width": 10.0, "height": 6.0},
        "config": {"plume": {"puff_mass": 1.0}},
    }
    current = {
        "plume_snapshot": {"complete": True},
        "plume": [[2.0, 3.0, 0.5]],
    }
    field = viewer_density_field(current, payload, nx=21, ny=13)
    assert field is not None
    density, extent = field
    assert density.shape == (13, 21)
    assert extent == (0.0, 10.0, 0.0, 6.0)
    assert float(np.max(density)) > 0.0


def test_density_rgba_is_transparent_where_density_is_zero() -> None:
    rgba = density_rgba(np.array([[0.0, 1.0]], dtype=float))
    assert rgba.shape == (1, 2, 4)
    assert rgba[0, 0, 3] == 0.0
    assert rgba[0, 1, 3] > 0.0


def test_social_display_window_ends_shortly_after_first_success() -> None:
    payload = {
        "frames": [
            {
                "t": float(index),
                "agents": [
                    {"found": index >= 10},
                    {"found": False},
                ],
            }
            for index in range(46)
        ]
    }
    assert social_display_frame_limit(payload, reveal_hold_s=2.6) == 12


def test_social_display_window_has_bounded_no_success_fallback() -> None:
    payload = {
        "frames": [
            {"t": float(index), "agents": [{"found": False}, {"found": False}]}
            for index in range(46)
        ]
    }
    assert social_display_frame_limit(payload, no_success_window_s=18.0) == 18


def test_pfl3_population_frame_keeps_all_24_cells() -> None:
    e002c_path = Path("results/e002/pfl3-convergence-v1.json")
    fc2_path = Path("results/route/fc2-goal-interface-audit-v1.json")
    if not e002c_path.exists() or not fc2_path.exists():
        pytest.skip("real local evidence artifacts are not committed")
    e002c = json.loads(e002c_path.read_text())
    fc2 = json.loads(fc2_path.read_text())
    body_ids, columns, values, scale = pfl3_population_frame(
        e002c, fc2, threshold=5, probe_index=0
    )
    assert len(body_ids) == 24
    assert len(columns) == 24
    assert set(columns) == set(range(1, 10))
    assert all(value.shape == (24,) for value in values.values())
    assert scale > 0.0
