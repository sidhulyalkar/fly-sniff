from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.dna02_source import READY, load_contract
from fly_sniff.freeze import canonical_sha256

EVIDENCE_PATH = Path("authority/program-a-dna02-source-byte-evidence-v1.json")
SOURCE_PATH = Path("authority/program-a-dna02-source-contract-v1.json")

EXPECTED_SHA256 = {
    "a2_d_08": "f02a345effb2fe722800dd1f43dce2c876878424307cb1fa486b5974f3d5922b",
    "a2_d_12": "0f9a9251262f528f0b415d67cdb169bc1236a5157029ffeebace6d86136f60f7",
    "a2_d_13": "a460eb313b503b039b3eeec554efd902b54119aadfd783c72210efaee2daee24",
    "a2_d_14": "6e5aabc4bc45d76fd24baa81a9351cc6d29530c72e697038b78f07a0da36623f",
}


def _evidence() -> dict:
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_source_byte_evidence_is_canonical_and_binds_network_manifest() -> None:
    payload = _evidence()
    claimed = payload.pop("evidence_sha256")
    assert canonical_sha256(payload) == claimed
    assert claimed == "d2e8f2c2bd41ff18b9be6a00c088f6b5c5f61b64bc6bfe42de70975a5dd93d67"
    assert payload["network_manifest_sha256"] == (
        "d957f1015f1b4d9724ab553f4da1762d817ffced4bd581f02ece6108a97ed2eb"
    )
    assert payload["cohort_adjudication_sha256"] == (
        "a7262fe9e82c1898a4ac6dc753aa38a854e05de558d64e466ffdde884dfdde53"
    )
    assert payload["raw_bytes_retained"] is False
    assert payload["navigation_performance_used"] is False


def test_all_four_md5s_reproduced_before_sha256_promotion() -> None:
    payload = _evidence()
    files = payload["files"]
    assert len(files) == 4
    assert sum(item["byte_count"] for item in files) == 929441392
    for item in files:
        assert item["published_md5"] == item["observed_md5"]
        assert item["observed_sha256"] == EXPECTED_SHA256[item["fly_alias"]]


def test_source_contract_exact_map_matches_frozen_byte_evidence() -> None:
    evidence = _evidence()
    by_alias = {item["fly_alias"]: item for item in evidence["files"]}
    contract = load_contract(SOURCE_PATH)
    assert contract.status == READY
    assert contract.blockers == ()
    for ref in contract.data_file_map:
        evidence_ref = by_alias[ref.fly_alias]
        assert ref.file_id == evidence_ref["file_id"]
        assert ref.filename == evidence_ref["filename"]
        assert ref.sha256 == evidence_ref["observed_sha256"]


def test_source_ready_scope_stops_before_physiology_and_navigation() -> None:
    payload = _evidence()
    unresolved = payload["scope"]["does_not_resolve"]
    assert "physiology summary statistics" in unresolved
    assert "model calibration parameters" in unresolved
    assert "odor-navigation performance" in unresolved
