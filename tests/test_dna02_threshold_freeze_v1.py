from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.freeze import canonical_sha256


DECISIONS = Path("authority/program-a-dna02-threshold-decisions-v1.json")
MANIFEST = Path("authority/program-a-dna02-threshold-manifest-v1.json")

EXPECTED = {
    ("a2_d_08", "L"): (0.999, 6.637119885669186),
    ("a2_d_08", "R"): (0.999, 6.765934524457742),
    ("a2_d_12", "L"): (0.995, 5.413398798564633),
    ("a2_d_12", "R"): (0.995, 5.661359850533993),
    ("a2_d_13", "L"): (0.9995, 5.32161236591439),
    ("a2_d_13", "R"): (0.999, 8.736763473032937),
    ("a2_d_14", "L"): (0.999, 5.497124009234522),
    ("a2_d_14", "R"): (0.999, 5.620147750325609),
}


def test_decisions_and_manifest_are_content_addressed_and_behavior_blind() -> None:
    decisions = json.loads(DECISIONS.read_text())
    manifest = json.loads(MANIFEST.read_text())

    assert canonical_sha256(decisions) == manifest["threshold_decisions_sha256"]

    declared_manifest_hash = manifest.pop("manifest_sha256")
    assert canonical_sha256(manifest) == declared_manifest_hash

    assert manifest["status"] == "THRESHOLDS_FROZEN_BEFORE_BEHAVIOR"
    assert manifest["threshold_count"] == 8
    assert manifest["thresholds_frozen"] is True
    assert manifest["behavior_fields_reviewed"] is False
    assert manifest["yaw_reviewed"] is False
    assert manifest["figure3c_statistic_reviewed"] is False
    assert manifest["navigation_performance_used"] is False

    observed = {
        (item["fly_alias"], item["soma_side"]): (
            item["selected_prominence_quantile"],
            item["selected_threshold"],
        )
        for item in manifest["thresholds"]
    }
    assert observed == EXPECTED
