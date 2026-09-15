from __future__ import annotations

import json
from pathlib import Path

from fly_sniff.freeze import canonical_sha256
from fly_sniff.pfn_source import BLOCKED_MEMBER_MAP, BLOCKED_RECORDINGS, load_contract

EVIDENCE_PATH = Path("authority/program-a-pfn-readme-evidence-v1.json")
SOURCE_PATH = Path("authority/program-a-pfn-source-contract-v1.json")
README_SHA256 = "6313f97ff216f29e5457502f80524edbc3ec0801dc0cb3b7fd57d08cebc31692"


def _evidence() -> dict:
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_readme_evidence_is_content_addressed_and_matches_dryad_authority() -> None:
    payload = _evidence()
    claimed = payload.pop("evidence_sha256")
    assert canonical_sha256(payload) == claimed
    assert claimed == "31756df446359b73585fc80432f707f1b08d0fbe0a9f60aa211839c811651915"
    assert payload["scientific_authority"]["repository"] == "Dryad"
    assert payload["scientific_authority"]["doi"] == "10.5061/dryad.vq83bk3rh"
    assert payload["retrieval"]["observed_md5"] == payload["scientific_authority"]["published_md5"]
    assert payload["retrieval"]["observed_sha256"] == README_SHA256
    assert payload["retrieval"]["navigation_performance_used"] is False


def test_mirror_is_transport_only_and_source_contract_uses_observed_sha() -> None:
    payload = _evidence()
    assert payload["retrieval"]["transport"] == "zenodo_dryad_archive_mirror"
    assert payload["retrieval"]["primary_result"] == "HTTP 403"

    contract = load_contract(SOURCE_PATH)
    assert contract.readme_sha256 == README_SHA256
    assert contract.blockers == (BLOCKED_MEMBER_MAP, BLOCKED_RECORDINGS)
    assert contract.status == "BLOCKED"
    assert any("transport mirror" in text for text in contract.forbidden_interpretation)


def test_readme_narrows_figure4_analysis_without_claiming_member_resolution() -> None:
    payload = _evidence()
    findings = payload["readme_findings"]
    assert findings["figure4_ephys_location"] == "Ephys directory"
    assert findings["single_fly_analysis_script"] == "analyzeWindTuning"
    assert findings["group_analysis_script"] == "windTuningPFN"
    unresolved = payload["scope"]["does_not_resolve"]
    assert any("multipart archive container" in item for item in unresolved)
    assert any("exact 12 Figure 4 recording files" in item for item in unresolved)
