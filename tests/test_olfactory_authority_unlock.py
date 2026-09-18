from __future__ import annotations

import json
from pathlib import Path

import pytest

import fly_sniff.olfactory_authority_unlock as unlock


def _annotation_bytes() -> bytes:
    header = (
        "supervoxel_id\troot_id\themobrain_type\tside\tvfb_id\tfbbt_id\tstatus\ttop_nt\ttop_nt_conf\n"
    )
    rows = [
        "1\t100\tORN_DA2\tright\tfw1\tFBbt_00067063\t\tacetylcholine\t0.9",
        "2\t101\tORN_DA2\tleft\tfw2\tFBbt_00067063\toutlier_bio\tacetylcholine\t0.8",
        "3\t102\tORN_DA2\tna\tfw3\tFBbt_00067063\t\tacetylcholine\t0.7",
        "4\t200\tDA2_lPN\tright\tfw4\tFBbt_00110882\t\tacetylcholine\t0.9",
    ]
    return (header + "\n".join(rows) + "\n").encode()


def test_scan_e002_freezes_discrepancy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = _annotation_bytes()
    monkeypatch.setattr(unlock, "ANNOTATION_BLOB", unlock._git_blob_sha1(raw))
    monkeypatch.setattr(
        unlock,
        "_bounded_get",
        lambda url, max_bytes: (
            raw,
            {
                "content_type": "text/tab-separated-values",
                "content_length": str(len(raw)),
                "etag": "",
                "last_modified": "",
                "final_url": url,
            },
        ),
    )

    report = unlock.scan_e002(tmp_path / "e002")
    assert report["status"] == "blocked_annotation_publication_discrepancy"
    assert report["automatic_authority_promotion_allowed"] is False
    assert report["annotation_observation"]["ORN_DA2_count"] == 3
    assert report["annotation_observation"]["ORN_DA2_side_counts"] == {
        "left": 1,
        "na": 1,
        "right": 1,
    }
    assert report["annotation_observation"]["DA2_lPN_count"] == 1
    assert report["adjudication"]["current_table_can_define_complete_41_cell_cohort"] is False
    assert (tmp_path / "e002" / "e002-flywire-v783-discovery.json").is_file()


def test_freeze_e001_rejects_html(tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("<html>blocked</html>")
    with pytest.raises(ValueError, match="not a PDF"):
        unlock.freeze_e001_paper(paper, tmp_path / "out")


def test_freeze_e001_hashes_pdf_without_promoting_claims(tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.7\nsynthetic-test-only\n%%EOF\n")
    report = unlock.freeze_e001_paper(paper, tmp_path / "out")
    assert report["status"] == "primary_paper_bytes_frozen_content_adjudication_pending"
    assert report["pdf_magic_verified"] is True
    assert report["numeric_parameterization_allowed"] is False
    receipt = json.loads((tmp_path / "out" / "e001-primary-paper-byte-freeze.json").read_text())
    assert receipt["sha256"] == report["sha256"]
