from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .olfactory_e006_audit import _canonical_sha

EXPECTED_DOI = "10.1016/j.cell.2012.09.046"
EXPECTED_URL = (
    "https://pure.mpg.de/rest/items/item_1577953_11/component/"
    "file_3157174/content?download=true"
)


def _load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("E001 source manifest requires schema_version=1")
    if payload.get("authority_id") != "E001_geosmin_receptor_physiology_source_archive":
        raise ValueError("E001 source authority id changed")
    if payload.get("program_id") != "olfactory-computation-v0":
        raise ValueError("E001 source program id changed")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise TypeError("E001 source manifest source must be an object")
    if source.get("doi") != EXPECTED_DOI:
        raise ValueError("E001 source DOI changed")
    if source.get("url") != EXPECTED_URL:
        raise ValueError("E001 institutional archive URL changed")
    policy = payload.get("acquisition_policy")
    if not isinstance(policy, dict):
        raise TypeError("E001 source acquisition_policy must be an object")
    required_true = {
        "freeze_exact_bytes_once",
        "require_pdf_magic",
        "record_sha256",
        "do_not_treat_published_pdf_as_raw_trial_data",
        "do_not_fit_numeric_model_parameters_from_figure_pixels",
        "o003_conflict_behavior_reserved",
    }
    if any(policy.get(key) is not True for key in required_true):
        raise ValueError("E001 source acquisition policy may not be weakened")
    minimum_bytes = policy.get("minimum_bytes")
    if not isinstance(minimum_bytes, int) or minimum_bytes < 100000:
        raise ValueError("E001 source minimum_bytes guard changed")
    if payload.get("status") != "source_uri_frozen_hash_pending":
        raise ValueError("E001 source manifest status changed")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pdf(path: Path, *, minimum_bytes: int) -> None:
    size = path.stat().st_size
    if size < minimum_bytes:
        raise ValueError(
            f"E001 archive download is unexpectedly small: {size} < {minimum_bytes} bytes"
        )
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError("E001 archive does not have a PDF header")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 fly-sniff-e001-source-freezer/1.0 "
                "(scientific provenance acquisition)"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def freeze_e001_source(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    source_file: str | Path | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty E001 source output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_file)
    source = manifest["source"]
    policy = manifest["acquisition_policy"]
    pdf_path = output / "stensmyr2012-cell-with-supplement.pdf"

    if source_file is None:
        _download(str(source["url"]), pdf_path)
        acquisition_mode = "institutional_archive_download"
    else:
        local = Path(source_file).expanduser().resolve()
        if not local.is_file():
            raise FileNotFoundError(f"E001 source file does not exist: {local}")
        shutil.copyfile(local, pdf_path)
        acquisition_mode = "local_file_test_or_mirror"

    _validate_pdf(pdf_path, minimum_bytes=int(policy["minimum_bytes"]))
    pdf_sha = _sha256(pdf_path)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "e001-primary-archive-freeze-v1",
        "authority_id": manifest["authority_id"],
        "program_id": manifest["program_id"],
        "status": "primary_archive_frozen_not_numeric_qualified",
        "source": {
            "citation": source["citation"],
            "doi": source["doi"],
            "institutional_repository": source["institutional_repository"],
            "item_id": source["item_id"],
            "component_id": source["component_id"],
            "url": source["url"],
            "observed_page_count_from_archive_metadata": source["observed_page_count"],
            "supplemental_information_in_same_pdf": source[
                "supplemental_information_in_same_pdf"
            ],
            "supplemental_figures_reported": source["supplemental_figures_reported"],
        },
        "acquisition": {
            "mode": acquisition_mode,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "pdf_path": str(pdf_path),
            "pdf_bytes": pdf_path.stat().st_size,
            "pdf_sha256": pdf_sha,
            "pdf_magic_verified": True,
        },
        "policy": {
            "published_pdf_is_raw_trial_data": False,
            "figure_pixel_numeric_fitting_allowed": False,
            "o003_conflict_behavior_reserved": True,
        },
        "claim_boundary": (
            "This receipt freezes exact bytes for the institutional Stensmyr 2012 paper-plus-"
            "supplement archive. It establishes citable source provenance only. Published figures "
            "and summary statistics remain distinct from raw trial data, and this receipt does not "
            "authorize numeric O001 parameter fitting or O003/O004 confirmatory execution."
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path = output / "e001-source-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (output / "SUMMARY.txt").write_text(
        "\n".join(
            [
                "E001 PRIMARY ARCHIVE FREEZE V1",
                f"status: {receipt['status']}",
                f"doi: {source['doi']}",
                f"pdf_bytes: {receipt['acquisition']['pdf_bytes']}",
                f"pdf_sha256: {pdf_sha}",
                f"receipt_sha256: {receipt['receipt_sha256']}",
                "",
                "CLAIM BOUNDARY",
                receipt["claim_boundary"],
            ]
        )
        + "\n"
    )
    return receipt


def verify_e001_source(
    output_dir: str | Path,
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    receipt_path = output / "e001-source-receipt.json"
    pdf_path = output / "stensmyr2012-cell-with-supplement.pdf"
    if not receipt_path.is_file() or not pdf_path.is_file():
        raise FileNotFoundError("E001 frozen archive receipt or PDF is missing")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("protocol") != "e001-primary-archive-freeze-v1":
        raise ValueError("E001 frozen archive protocol changed")
    observed = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if observed != _canonical_sha(unhashed):
        raise ValueError("E001 frozen archive receipt canonical hash mismatch")
    if receipt.get("source", {}).get("doi") != manifest["source"]["doi"]:
        raise ValueError("E001 frozen archive DOI does not match manifest")
    if receipt.get("source", {}).get("url") != manifest["source"]["url"]:
        raise ValueError("E001 frozen archive URL does not match manifest")
    minimum_bytes = int(manifest["acquisition_policy"]["minimum_bytes"])
    _validate_pdf(pdf_path, minimum_bytes=minimum_bytes)
    observed_pdf_sha = _sha256(pdf_path)
    if observed_pdf_sha != receipt.get("acquisition", {}).get("pdf_sha256"):
        raise ValueError("E001 frozen archive PDF sha256 mismatch")
    if pdf_path.stat().st_size != receipt.get("acquisition", {}).get("pdf_bytes"):
        raise ValueError("E001 frozen archive PDF byte count mismatch")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze exact institutional Stensmyr 2012 E001 source bytes"
    )
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-file")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify an existing immutable archive instead of downloading",
    )
    args = parser.parse_args()
    if args.verify_existing:
        if args.source_file:
            parser.error("--source-file cannot be combined with --verify-existing")
        report = verify_e001_source(args.output, manifest_path=args.manifest)
    else:
        report = freeze_e001_source(
            args.manifest,
            output_dir=args.output,
            source_file=args.source_file,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
