from __future__ import annotations

from fly_sniff.pfl3_descending_probe_v2 import _side_swap_confirmation


def _protocol() -> dict:
    return {
        "mirrored_phase_pairs_deg": [[-30, 30], [-60, 60]],
    }


def test_v2_side_swap_ignores_zero_phase_for_reversal_gate() -> None:
    rows = {
        "-60": {"turn": 0.4, "side_swap_turn": -0.3},
        "-30": {"turn": 0.2, "side_swap_turn": -0.1},
        "0": {"turn": -0.02, "side_swap_turn": -0.01},
        "30": {"turn": -0.2, "side_swap_turn": 0.1},
        "60": {"turn": -0.4, "side_swap_turn": 0.3},
    }
    passed, detail = _side_swap_confirmation(rows, _protocol())
    assert passed is True
    assert detail["required_phase_count"] == 4
    assert detail["reversed_phase_count"] == 4
    assert detail["zero_phase"]["gate_role"] == "descriptive_baseline_only"


def test_v2_side_swap_fails_any_nonzero_phase_without_reversal() -> None:
    rows = {
        "-60": {"turn": 0.4, "side_swap_turn": -0.3},
        "-30": {"turn": 0.2, "side_swap_turn": 0.1},
        "0": {"turn": 0.0, "side_swap_turn": 0.0},
        "30": {"turn": -0.2, "side_swap_turn": 0.1},
        "60": {"turn": -0.4, "side_swap_turn": 0.3},
    }
    passed, detail = _side_swap_confirmation(rows, _protocol())
    assert passed is False
    assert detail["reversed_phase_count"] == 3
