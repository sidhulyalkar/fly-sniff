from __future__ import annotations

from copy import deepcopy

import pytest

from fly_video_neural.schema import SampleWindow
from fly_video_neural.split_lock import build_split_lock, verify_split_lock


def _windows() -> list[SampleWindow]:
    rows = []
    for animal in range(8):
        for trial in range(2):
            rows.append(
                SampleWindow(
                    dataset_id="mc2p_v1",
                    sample_id=f"fly{animal}-trial{trial}",
                    animal_id=f"fly{animal}",
                    session_id=f"fly{animal}_00{trial}",
                    input_start_s=0,
                    input_end_s=3,
                    target_start_s=3,
                    target_end_s=3.5,
                    video_fps=100,
                )
            )
    return rows


def test_eight_animal_lock_is_five_one_two_and_verifies():
    rows = _windows()
    lock = build_split_lock(rows)
    assert lock["animal_counts"] == {"test": 2, "train": 5, "validation": 1}
    verify_split_lock(lock, rows)


def test_split_lock_is_deterministic():
    rows = _windows()
    assert build_split_lock(rows)["split_lock_sha256"] == build_split_lock(reversed(rows))["split_lock_sha256"]


def test_rehashed_semantic_tamper_still_fails_assignment_check():
    rows = _windows()
    lock = build_split_lock(rows)
    tampered = deepcopy(lock)
    sample = next(iter(tampered["sample_to_split"]))
    tampered["sample_to_split"][sample] = "test"
    from fly_video_neural.split_lock import _sha

    payload = dict(tampered)
    payload.pop("split_lock_sha256")
    tampered["split_lock_sha256"] = _sha(payload)
    with pytest.raises(ValueError, match="sample assignments"):
        verify_split_lock(tampered, rows)


def test_window_change_invalidates_lock():
    rows = _windows()
    lock = build_split_lock(rows)
    changed = rows[:-1]
    with pytest.raises(ValueError, match="does not match"):
        verify_split_lock(lock, changed)
