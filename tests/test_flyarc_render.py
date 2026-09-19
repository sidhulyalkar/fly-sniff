import hashlib
import json

import numpy as np
import pytest

from fly_sniff.flyarc import _pack_frames
from fly_sniff.flyarc_render import _state_grid, _unpack_frame, validate_receipt


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pack_frames_round_trips_variable_arc_shapes():
    frames = [
        np.array([[1, 2], [3, 4]], dtype=np.uint8),
        np.array([[5, 6, 7]], dtype=np.uint8),
    ]
    packed, shapes = _pack_frames(frames)

    assert packed.shape == (2, 2, 3)
    assert np.array_equal(_unpack_frame(packed, shapes, 0), frames[0])
    assert np.array_equal(_unpack_frame(packed, shapes, 1), frames[1])


def test_state_grid_preserves_values_and_pads_nan():
    state = np.arange(5, dtype=float)
    grid = _state_grid(state)

    assert grid.shape == (3, 3)
    assert np.array_equal(grid.ravel()[:5], state)
    assert np.isnan(grid.ravel()[5:]).all()


def test_renderer_receipt_validation_is_fail_closed(tmp_path):
    payload = tmp_path / "comparison.json"
    payload.write_text('{"ok": true}\n')
    receipt = {
        "experiment": "flyarc-v1",
        "files": {"comparison.json": _sha(payload)},
        "paired_version_check": True,
    }
    (tmp_path / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n"
    )

    validated = validate_receipt(tmp_path)
    assert validated["paired_version_check"] is True

    payload.write_text('{"ok": false}\n')
    with pytest.raises(ValueError, match="receipt validation failed"):
        validate_receipt(tmp_path)
