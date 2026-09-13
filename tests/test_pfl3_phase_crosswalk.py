from __future__ import annotations

import pytest

from fly_sniff.pfl3_phase_crosswalk import build_crosswalk


def _fixtures():
    body_ids = list(range(100, 124))
    columns = {body_id: (index % 9) + 1 for index, body_id in enumerate(body_ids)}
    pb_labels = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    rows = []
    for index, body_id in enumerate(body_ids):
        pb = pb_labels[index % len(pb_labels)]
        side, number = pb[0], pb[1:]
        suffix = "_irreg" if index == 0 else ""
        rows.append(
            {
                "bodyId": body_id,
                "instance": f"PFL3(PB12c)_{side}{number}_C{columns[body_id]}{suffix}",
            }
        )
    config = {
        "protocol": "E002d-pfl3-relative-phase-comparison-v3",
        "mapping": {
            "nine_column_goal_sensitivity_degrees": [0, -45, -90, -135, -180, 135, 90, 45, 0],
            "heading_glomerulus_sequence_degrees_left_to_right": [-22.5, 22.5, 67.5, 112.5, 157.5, -157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5, -157.5, -112.5, -67.5, -22.5, 22.5],
        },
    }
    e002c = {"passed": True}
    fc2 = {
        "protocol": "malecns-fc2-goal-interface-audit-v1",
        "phase_mapping_status": "unresolved",
        "populations": {"PFL3": {"instance_columns": {"body_columns": {str(k): v for k, v in columns.items()}}}},
    }
    heading = {"protocol": "malecns-heading-route-audit-v1", "populations": {"PFL3": {"rows": rows}}}
    steering = {
        "protocol": "malecns-steering-scaffold-candidate-v1",
        "candidate_roles": {
            "turn_drive_left": body_ids[:12],
            "turn_drive_right": body_ids[12:],
        },
    }
    return config, e002c, fc2, heading, steering


def test_crosswalk_covers_exact_24_and_keeps_instance_side_separate() -> None:
    report = build_crosswalk(*_fixtures())
    assert report["ready_for_phase_probe"]
    assert report["counts"]["PFL3"] == 24
    assert report["counts"]["turn_drive_left"] == 12
    assert report["counts"]["turn_drive_right"] == 12
    first = next(row for row in report["records"] if row["body_id"] == 100)
    assert first["instance_pb_side"] == "R"
    assert first["readout_side"] == "turn_drive_left"
    assert first["irregular_instance"] is True


def test_crosswalk_rejects_column_drift() -> None:
    config, e002c, fc2, heading, steering = _fixtures()
    fc2["populations"]["PFL3"]["instance_columns"]["body_columns"]["100"] = 9
    with pytest.raises(ValueError, match="column mismatch"):
        build_crosswalk(config, e002c, fc2, heading, steering)


def test_crosswalk_rejects_overlapping_steering_groups() -> None:
    config, e002c, fc2, heading, steering = _fixtures()
    steering["candidate_roles"]["turn_drive_right"][0] = 100
    with pytest.raises(ValueError, match="disjoint 12\\+12"):
        build_crosswalk(config, e002c, fc2, heading, steering)
