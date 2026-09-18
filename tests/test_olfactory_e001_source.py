from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.olfactory_e001_source import freeze_e001_source

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "authority" / "geosmin-e001-source-v1.json"


def _fake_pdf(path: Path, size: int = 120000) -> Path:
    payload = b"%PDF-1.7\n" + b"0" * (size - 9)
    path.write_bytes(payload)
    return path


def test_e001_source_freeze_hashes_exact_pdf_bytes(tmp_path: Path) -> None:
    source = _fake_pdf(tmp_path / "source.pdf")
    output = tmp_path / "frozen"
    report = freeze_e001_source(
        MANIFEST,
        output_dir=output,
        source_file=source,
    )

    assert report["status"] == "primary_archive_frozen_not_numeric_qualified"
    assert report["source"]["doi"] == "10.1016/j.cell.2012.09.046"
    assert report["source"]["supplemental_information_in_same_pdf"] is True
    assert report["acquisition"]["pdf_magic_verified"] is True
    assert report["acquisition"]["pdf_bytes"] == 120000
    assert len(report["acquisition"]["pdf_sha256"]) == 64
    assert report["policy"]["published_pdf_is_raw_trial_data"] is False
    assert report["policy"]["figure_pixel_numeric_fitting_allowed"] is False
    assert (output / "stensmyr2012-cell-with-supplement.pdf").is_file()
    assert (output / "e001-source-receipt.json").is_file()
    assert (output / "SUMMARY.txt").is_file()


def test_e001_source_refuses_non_pdf(tmp_path: Path) -> None:
    source = tmp_path / "not.pdf"
    source.write_bytes(b"hello" * 30000)
    with pytest.raises(ValueError, match="PDF header"):
        freeze_e001_source(
            MANIFEST,
            output_dir=tmp_path / "out",
            source_file=source,
        )


def test_e001_source_refuses_tiny_pdf(tmp_path: Path) -> None:
    source = _fake_pdf(tmp_path / "tiny.pdf", size=1000)
    with pytest.raises(ValueError, match="unexpectedly small"):
        freeze_e001_source(
            MANIFEST,
            output_dir=tmp_path / "out",
            source_file=source,
        )


def test_e001_source_refuses_overwrite(tmp_path: Path) -> None:
    source = _fake_pdf(tmp_path / "source.pdf")
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        freeze_e001_source(
            MANIFEST,
            output_dir=output,
            source_file=source,
        )


def test_e001_source_manifest_cannot_change_archive_uri(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["source"]["url"] = "https://example.invalid/paper.pdf"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    source = _fake_pdf(tmp_path / "source.pdf")
    with pytest.raises(ValueError, match="archive URL changed"):
        freeze_e001_source(
            manifest,
            output_dir=tmp_path / "out",
            source_file=source,
        )
