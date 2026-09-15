from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.freeze import canonical_sha256
from fly_sniff.pfn_source import BLOCKED_MEMBER_MAP, BLOCKED_RECORDINGS, load_contract

EVIDENCE_PATH = Path("authority/program-a-pfn-archive-directory-evidence-v1.json")
ADJUDICATION_PATH = Path("authority/program-a-pfn-figure4-fly-cohort-adjudication-v1.json")
SOURCE_PATH = Path("authority/program-a-pfn-source-contract-v1.json")


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_archive_directory_evidence_is_content_addressed() -> None:
    payload = _load(EVIDENCE_PATH)
    claimed = payload.pop("evidence_sha256")
    assert canonical_sha256(payload) == claimed
    assert claimed == "0de9287c264c6b49750b469b5a71363476fad4f81f2652adc2ee5306e35eb4ba"


def test_bounded_probe_read_no_neural_member_bytes() -> None:
    payload = _load(EVIDENCE_PATH)
    probe = payload["bounded_probe"]
    assert probe["archive_format"] == "zip64"
    assert probe["required_suffix_bytes"] == 4_587_719
    assert probe["required_suffix_bytes"] < probe["max_tail_bytes"]
    assert probe["archive_entry_count"] == 42_708
    assert probe["ephys_entry_count"] == 41_979
    assert probe["neural_member_bytes_read"] is False
    assert payload["navigation_performance_used"] is False


def test_figure4_candidate_set_is_not_silently_promoted() -> None:
    payload = _load(EVIDENCE_PATH)
    structure = payload["ss02255_directory_structure"]
    aliases = [item["recording_alias"] for item in structure["wind_tuning_candidates"]]
    assert structure["wind_tuning_candidate_count"] == 13
    assert aliases == [
        "WTC02", "WTC03", "WTC04", "WTC05", "WTC06", "WTC07", "WTC08",
        "WTC09", "WTC11", "WTC12", "WTC13", "WTC14", "WTC15",
    ]
    assert payload["figure4_paper_constraint"]["published_recording_count"] == 12
    boundary = payload["adjudication_boundary"]
    assert boundary["automatic_recording_set_promotion"] is False
    assert boundary["automatic_archive_member_map_promotion"] is False


def test_trial_complete_metadata_resolves_exact_twelve_fly_cohort() -> None:
    payload = _load(ADJUDICATION_PATH)
    claimed = payload.pop("adjudication_sha256")
    assert canonical_sha256(payload) == claimed
    assert claimed == "50c410adc3fa5c66dbde7e8a389d4d4621dfddc3cfa19c297cef04e9c3fecf29"
    assert payload["status"] == "RESOLVED_FLY_COHORT_EXPERIMENT_MAP_PENDING"
    assert len(payload["resolved_figure4_fly_aliases"]) == 12
    assert "WTC09" not in payload["resolved_figure4_fly_aliases"]
    assert payload["excluded_candidate"]["fly_alias"] == "WTC09"
    assert payload["archive_trial_inventory"]["WTC09"][0]["trial_count"] == 21
    assert all(
        any(exp["complete_40_trial_experiment"] for exp in payload["archive_trial_inventory"][alias])
        for alias in payload["resolved_figure4_fly_aliases"]
    )
    assert payload["decision_rule"]["navigation_performance_used"] is False


def test_wtc02_experiment_ambiguity_stays_explicit() -> None:
    payload = _load(ADJUDICATION_PATH)
    wtc02 = payload["archive_trial_inventory"]["WTC02"]
    assert [item["experiment"] for item in wtc02] == ["E1", "E2"]
    assert all(item["complete_40_trial_experiment"] for item in wtc02)
    assert any("WTC02" in blocker for blocker in payload["remaining_blockers"])
    assert payload["automatic_recording_set_promotion"] is False


def test_directory_evidence_does_not_clear_pfn_source_contract() -> None:
    contract = load_contract(SOURCE_PATH)
    assert contract.status == "BLOCKED"
    assert contract.blockers == (BLOCKED_MEMBER_MAP, BLOCKED_RECORDINGS)
    assert len(contract.archive_member_map) == 0
    assert len(contract.recording_set) == 0


def test_tiny_analysis_scripts_are_the_only_next_member_targets() -> None:
    payload = _load(EVIDENCE_PATH)
    members = payload["ss02255_directory_structure"]["analysis_members"]
    assert [item["path"] for item in members] == [
        "Ephys/analyzeWindTuning.m",
        "Ephys/windTuningPFN.m",
    ]
    assert sum(item["compressed_size_32"] for item in members) < 8_000
