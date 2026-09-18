from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ANNOTATION_COMMIT = "ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0"
ANNOTATION_BLOB = "1a3168731618ee62a47392252d3af7664e739e9e"
ANNOTATION_PATH = "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
ANNOTATION_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    f"{ANNOTATION_COMMIT}/{ANNOTATION_PATH}"
)
ANNOTATION_MAX_BYTES = 64 * 1024 * 1024

E001_DOI = "10.1016/j.cell.2012.09.046"
E001_PII = "S0092867412013578"
E001_OPEN_ACCESS_URL = f"https://www.cell.com/article/{E001_PII}/pdf"

PUBLISHED_OR56A_FLYWIRE_ANNOTATED = 40
PUBLISHED_OR56A_TOTAL = 41
PUBLISHED_RIGHT_DA2_IPSILATERAL = 22
PUBLISHED_RIGHT_DA2_CONTRALATERAL = 19

_USER_AGENT = "fly-sniff-olfactory-authority-ingress/1.0"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp = Path(handle.name)
    os.replace(temp, path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(prefix + payload).hexdigest()


def _bounded_get(url: str, max_bytes: int) -> tuple[bytes, dict[str, str]]:
    request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
    with urlopen(request, timeout=60.0) as response:
        final_url = str(response.geturl())
        if not final_url.startswith("https://"):
            raise ValueError("source redirected to non-HTTPS URL")
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(f"source exceeds byte cap {max_bytes}")
        headers = {
            "content_type": str(response.headers.get("Content-Type", "")),
            "content_length": str(response.headers.get("Content-Length", "")),
            "etag": str(response.headers.get("ETag", "")),
            "last_modified": str(response.headers.get("Last-Modified", "")),
            "final_url": final_url,
        }
    return raw, headers


def _side_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        side = row.get("side", "")
        result[side] = result.get(side, 0) + 1
    return dict(sorted(result.items()))


def _compact_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "root_id": row.get("root_id", ""),
            "side": row.get("side", ""),
            "vfb_id": row.get("vfb_id", ""),
            "fbbt_id": row.get("fbbt_id", ""),
            "status": row.get("status", ""),
            "hemibrain_type": row.get("hemibrain_type", ""),
            "top_nt": row.get("top_nt", ""),
            "top_nt_conf": row.get("top_nt_conf", ""),
        }
        for row in rows
    ]


def scan_e002(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    raw, headers = _bounded_get(ANNOTATION_URL, ANNOTATION_MAX_BYTES)
    observed_blob = _git_blob_sha1(raw)
    if observed_blob != ANNOTATION_BLOB:
        raise ValueError(
            "FlyWire annotation bytes do not match the pinned Git blob: "
            f"{observed_blob} != {ANNOTATION_BLOB}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("FlyWire annotation table is not valid UTF-8") from exc

    reader = csv.DictReader(StringIO(text), delimiter="\t")
    rows = [dict(row) for row in reader]
    orn = [row for row in rows if row.get("hemibrain_type") == "ORN_DA2"]
    pn = [row for row in rows if row.get("hemibrain_type") == "DA2_lPN"]

    orn_sides = _side_counts(orn)
    pn_sides = _side_counts(pn)
    anomalous_orn = [
        row
        for row in orn
        if row.get("side") not in {"left", "right"} or bool(row.get("status"))
    ]

    count_matches_published_annotation = len(orn) == PUBLISHED_OR56A_FLYWIRE_ANNOTATED
    side_complete = all(row.get("side") in {"left", "right"} for row in orn)
    auto_promote = (
        count_matches_published_annotation
        and side_complete
        and orn_sides.get("right", 0) == PUBLISHED_RIGHT_DA2_IPSILATERAL
        and orn_sides.get("left", 0) == PUBLISHED_RIGHT_DA2_CONTRALATERAL
    )

    report: dict[str, Any] = {
        "schema": "fly-sniff-e002-flywire-v783-discovery-v1",
        "status": (
            "candidate_table_matches_published_count_constraints"
            if auto_promote
            else "blocked_annotation_publication_discrepancy"
        ),
        "automatic_authority_promotion_allowed": False,
        "source": {
            "repository": "flyconnectome/flywire_annotations",
            "commit": ANNOTATION_COMMIT,
            "path": ANNOTATION_PATH,
            "expected_git_blob_sha1": ANNOTATION_BLOB,
            "observed_git_blob_sha1": observed_blob,
            "sha256": _sha256_bytes(raw),
            "byte_count": len(raw),
            "url": ANNOTATION_URL,
            "http": headers,
        },
        "publication_constraints": {
            "or56a_flywire_annotated": PUBLISHED_OR56A_FLYWIRE_ANNOTATED,
            "or56a_total_after_added_afferent": PUBLISHED_OR56A_TOTAL,
            "right_DA2_ipsilateral": PUBLISHED_RIGHT_DA2_IPSILATERAL,
            "right_DA2_contralateral": PUBLISHED_RIGHT_DA2_CONTRALATERAL,
            "publication": "Acharya et al., Frontiers in Cellular Neuroscience (2025)",
            "doi": "10.3389/fncel.2025.1579821",
        },
        "annotation_observation": {
            "ORN_DA2_count": len(orn),
            "ORN_DA2_side_counts": orn_sides,
            "ORN_DA2_rows": _compact_rows(orn),
            "ORN_DA2_anomalous_rows": _compact_rows(anomalous_orn),
            "DA2_lPN_count": len(pn),
            "DA2_lPN_side_counts": pn_sides,
            "DA2_lPN_rows": _compact_rows(pn),
        },
        "adjudication": {
            "annotation_count_matches_published_40": count_matches_published_annotation,
            "all_ORN_DA2_sides_resolved": side_complete,
            "current_table_can_define_complete_41_cell_cohort": False,
            "next_required_evidence": [
                "publication-specific Or56a reconstruction/root-ID list or equivalent frozen supplementary authority",
                "adjudication of the ORN_DA2 row with side='na'",
                "identity of the publication's additional afferent beyond the FlyWire-annotated cohort",
                "reconciliation of current pinned annotation count with the publication's 40 annotated cells",
            ],
        },
        "claim_boundary": (
            "This receipt freezes exact v783 annotation candidates and the discrepancy against the "
            "published Or56a count constraints. It is discovery/adjudication evidence only. It may not "
            "silently infer missing root IDs, coerce side='na', promote all ORN_DA2 rows to the final "
            "Or56a cohort, or qualify E002."
        ),
    }
    report["receipt_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output / "e002-flywire-v783-discovery.json", report)
    return report


def freeze_e001_paper(paper_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    paper = Path(paper_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    raw = paper.read_bytes()
    if not raw.startswith(b"%PDF-"):
        raise ValueError(
            "E001 paper file is not a PDF. This commonly means the download returned an HTML "
            "bot/error page; refusing to freeze it."
        )
    report: dict[str, Any] = {
        "schema": "fly-sniff-e001-primary-paper-byte-freeze-v1",
        "status": "primary_paper_bytes_frozen_content_adjudication_pending",
        "authority_id": "E001_geosmin_receptor_physiology",
        "doi": E001_DOI,
        "pii": E001_PII,
        "source_url": E001_OPEN_ACCESS_URL,
        "filename": paper.name,
        "byte_count": len(raw),
        "sha256": _sha256_bytes(raw),
        "pdf_magic_verified": True,
        "numeric_parameterization_allowed": False,
        "claim_boundary": (
            "This receipt freezes the exact local primary-paper PDF bytes. It resolves only the "
            "primary-paper byte-provenance portion of E001. It does not establish that supplemental "
            "files or raw trials are present, does not convert published summaries into raw data, and "
            "does not authorize numerical calibration or O003."
        ),
    }
    report["receipt_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output / "e001-primary-paper-byte-freeze.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and audit the next olfactory evidence needed to unlock O003"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    e002 = sub.add_parser(
        "scan-e002",
        help="download the exact pinned FlyWire annotation table and freeze DA2 candidate discovery",
    )
    e002.add_argument("--out", required=True)

    e001 = sub.add_parser(
        "freeze-e001-paper",
        help="hash and freeze a locally downloaded Stensmyr 2012 primary-paper PDF",
    )
    e001.add_argument("--paper", required=True)
    e001.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "scan-e002":
        report = scan_e002(args.out)
    else:
        report = freeze_e001_paper(args.paper, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
