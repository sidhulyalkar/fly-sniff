from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.freeze import canonical_sha256

CONFIRMATION = Path("authority/program-a-dna02-field-map-confirmation-v1.json")


def _load() -> dict:
    payload = json.loads(CONFIRMATION.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_real_schema_confirmation_is_content_addressed() -> None:
    payload = _load()
    claimed = payload.pop("confirmation_sha256")
    assert canonical_sha256(payload) == claimed
    assert claimed == "f38dfbabdbfa51c0629f763cb942999bb7d0d71749c845ffd0b51855b9ee6893"


def test_real_schema_confirmation_keeps_science_boundary() -> None:
    payload = _load()
    assert payload["status"] == "FIELD_MAP_CONFIRMED_PENDING_NUMERIC_EXTRACTION_REVIEW"
    assert payload["source_inspection_sha256"] == "68a4b5692f9515c4308c7a86e9e88d4eafb95d2d4a93f0f1b706990ce26817b7"
    assert payload["raw_values_read"] is False
    assert payload["physiology_statistic_computed"] is False
    assert payload["navigation_performance_used"] is False


def test_real_schema_confirmation_has_four_consistent_source_schemas() -> None:
    payload = _load()
    assert [item["fly_alias"] for item in payload["files"]] == [
        "a2_d_08",
        "a2_d_12",
        "a2_d_13",
        "a2_d_14",
    ]
    required = set(payload["required_raw_fields"])
    for item in payload["files"]:
        fields = item["required_field_metadata"]
        assert set(fields) == required
        assert item["mat_format"] == "matlab_v5"
        assert fields["ephys_A"]["shape"] == fields["ephys_B"]["shape"]
        assert fields["yaw"]["shape"] == fields["fwd"]["shape"] == fields["lat"]["shape"]
        assert fields["t_ball"]["shape"] == fields["yaw"]["shape"]
        assert fields["t_ephys"]["shape"] == fields["ephys_A"]["shape"]
        assert fields["stim"]["shape"][1] == fields["ephys_A"]["shape"][0]
        assert fields["ephys_A"]["shape"][0] == 100 * fields["yaw"]["shape"][0]
