from __future__ import annotations

import pytest

from fly_sniff.pfl3_descending_probe import _swap_side_drive


def _records() -> list[dict[str, object]]:
    return [
        {
            "body_id": 1,
            "readout_side": "turn_drive_left",
            "column": 1,
            "pb_label": "L1",
        },
        {
            "body_id": 2,
            "readout_side": "turn_drive_left",
            "column": 2,
            "pb_label": "L2",
        },
        {
            "body_id": 3,
            "readout_side": "turn_drive_right",
            "column": 1,
            "pb_label": "R1",
        },
        {
            "body_id": 4,
            "readout_side": "turn_drive_right",
            "column": 2,
            "pb_label": "R2",
        },
    ]


def test_side_swap_handles_unequal_active_subsets() -> None:
    swapped = _swap_side_drive(_records(), {1: 0.4, 2: 0.8, 3: 0.2})
    assert swapped == pytest.approx({1: 0.2, 3: 0.4, 4: 0.8})


def test_side_swap_rejects_drive_outside_frozen_groups() -> None:
    with pytest.raises(ValueError, match="outside frozen side groups"):
        _swap_side_drive(_records(), {99: 1.0})
