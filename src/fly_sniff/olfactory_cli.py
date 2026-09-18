from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .olfactory_door import ingest_door, validate_source_authority
from .olfactory_e006_audit import audit_e006
from .olfactory_geosmin import validate_geosmin_evidence
from .olfactory_o002 import run_o002_development
from .olfactory_o002_robustness import run_o002_robustness
from .olfactory_program import validate_study
from .olfactory_structure import validate_da2_authority

PROGRAM = "authority/olfactory-computation-program-v0.json"
EVIDENCE = "authority/olfactory-evidence-requirements-v0.json"
SOURCE_REGISTRY = "authority/olfactory-source-registry-v0.json"
O001 = "authority/geosmin-o001-development-v0.json"
E001 = "authority/geosmin-e001-evidence-v0.json"
E002 = "authority/flywire-da2-e002-v0.json"
E006_SOURCE = "authority/door-e006-source-v0.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required olfactory study artifact is missing: {path}")
    return path


def study_status(root: str | Path = ".") -> dict[str, Any]:
    repo = Path(root).resolve()
    study = validate_study(
        _path(repo, PROGRAM),
        _path(repo, EVIDENCE),
        source_registry_path=_path(repo, SOURCE_REGISTRY),
        o001_path=_path(repo, O001),
    )
    e001 = validate_geosmin_evidence(_load(_path(repo, E001)))
    e002 = validate_da2_authority(_load(_path(repo, E002)))
    e006_source = _load(_path(repo, E006_SOURCE))
    validate_source_authority(e006_source)

    readiness = {
        "O001_geosmin_calibration": {
            "status": "blocked_pending_qualified_evidence",
            "E001": e001["status"],
            "E002": e002["status"],
        },
        "O003_flagship": {
            "status": "blocked_pending_evidence_and_benchmark_lock",
            "confirmatory_allowed": False,
        },
        "E006_door_ingestion": {
            "status": e006_source["status"],
            "next_action": "run ingest-door against the exact pinned DoOR checkout",
        },
    }
    return {
        "program_id": study["program_id"],
        "study_status": study["status"],
        "unresolved_count": study["unresolved_count"],
        "unresolved_authorities": study["unresolved_authorities"],
        "readiness": readiness,
        "claim_boundary": (
            "Development/evidence acquisition may proceed. Confirmatory O003/O004 execution remains "
            "blocked until all required authorities are qualified and the experiment lock is frozen."
        ),
    }


def _status_command(args: argparse.Namespace) -> int:
    print(json.dumps(study_status(args.root), indent=2, sort_keys=True))
    return 0


def _e001_command(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    path = Path(args.path).resolve() if args.path else _path(root, E001)
    print(json.dumps(validate_geosmin_evidence(_load(path)), indent=2, sort_keys=True))
    return 0


def _e002_command(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    path = Path(args.path).resolve() if args.path else _path(root, E002)
    print(json.dumps(validate_da2_authority(_load(path)), indent=2, sort_keys=True))
    return 0


def _door_command(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    authority = Path(args.authority).resolve() if args.authority else _path(root, E006_SOURCE)
    report = ingest_door(args.door_checkout, args.output, authority_path=authority)
    print(
        json.dumps(
            {
                "status": "ingested_not_qualified",
                "authority_id": report["authority_id"],
                "responding_units": report["responding_unit_count"],
                "studies": report["study_column_count"],
                "response_cells": report["response_cells"],
                "observed_cells": report["observed_response_cells"],
                "missing_cells": report["missing_response_cells"],
                "geosmin_observed_cells": report["geosmin_observed_cells"],
                "receipt_sha256": report["receipt_sha256"],
                "output": str(Path(args.output).resolve()),
                "claim_boundary": report["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _e006_audit_command(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    report = audit_e006(
        args.artifact_dir,
        door_checkout=args.door_checkout,
        repo_root=root,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "status": report["gate"]["status"],
                "audit_sha256": report["audit_sha256"],
                "summary": report["summary"],
                "blockers": report["gate"]["blockers"],
                "output": str(Path(args.output).expanduser().resolve()),
                "claim_boundary": report["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _o002_command(args: argparse.Namespace) -> int:
    report = run_o002_development(
        args.artifact_dir,
        audit_path=args.audit,
        output_dir=args.output,
        study_id=args.study,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "study": report["selected_study"]["study_id"],
                "receipt_sha256": report["receipt_sha256"],
                "matrix": report["matrix"],
                "pca": report["pca"],
                "geometry_stability": {
                    key: value
                    for key, value in report["geometry_stability"].items()
                    if key != "per_unit"
                },
                "chemical_class_probe": report["chemical_class_probe"],
                "output": str(Path(args.output).expanduser().resolve()),
                "claim_boundary": report["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _o002_robustness_command(args: argparse.Namespace) -> int:
    report = run_o002_robustness(
        args.o002_dir,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "receipt_sha256": report["receipt_sha256"],
                "frozen_question": report["frozen_question"],
                "sample": report["sample"],
                "full_pattern": report["full_pattern"],
                "amplitude_only": report["amplitude_only"],
                "direction_only": report["direction_only"],
                "identity_erased_sorted_profile": report["identity_erased_sorted_profile"],
                "channel_identity_shuffle_null": report["channel_identity_shuffle_null"],
                "leave_one_unit": report["leave_one_unit"],
                "feature_subset_robustness": report["feature_subset_robustness"],
                "contrasts": report["contrasts"],
                "output": str(Path(args.output).expanduser().resolve()),
                "claim_boundary": report["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fly-sniff-olfactory",
        description="Local orchestration CLI for the preregistered olfactory-computation study",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="fly-sniff repository root (default: current directory)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="validate the study contract and print current readiness")
    status.set_defaults(func=_status_command)

    e001 = sub.add_parser("validate-e001", help="validate frozen geosmin evidence constraints")
    e001.add_argument("--path")
    e001.set_defaults(func=_e001_command)

    e002 = sub.add_parser("validate-e002", help="validate frozen FlyWire Or56a/DA2 identity authority")
    e002.add_argument("--path")
    e002.set_defaults(func=_e002_command)

    door = sub.add_parser(
        "ingest-door",
        help="ingest the exact pinned DoOR checkout into lossless E006 evidence",
    )
    door.add_argument("door_checkout")
    door.add_argument("--output", required=True)
    door.add_argument("--authority")
    door.set_defaults(func=_door_command)

    e006 = sub.add_parser(
        "audit-e006",
        help="run the complete fail-closed E006 provenance, coverage, scale, and identity audit",
    )
    e006.add_argument("artifact_dir", help="directory containing door-e006-receipt.json")
    e006.add_argument("--door-checkout", required=True, help="exact pinned clean DoOR checkout")
    e006.add_argument("--output", required=True, help="directory for audit outputs")
    e006.set_defaults(func=_e006_audit_command)

    o002 = sub.add_parser(
        "run-o002-dev",
        help="run frozen performance-blind within-study O002 representation development",
    )
    o002.add_argument("artifact_dir", help="directory containing the audited E006 ingestion")
    o002.add_argument("--audit", required=True, help="path to e006-audit.json")
    o002.add_argument("--output", required=True, help="new output directory")
    o002.add_argument(
        "--study",
        help="optional frozen candidate study id; defaults to audit-selected development subset",
    )
    o002.set_defaults(func=_o002_command)

    o002_v2 = sub.add_parser(
        "run-o002-robustness",
        help="decompose the frozen O002 class signal with magnitude, identity, and redundancy controls",
    )
    o002_v2.add_argument("o002_dir", help="directory containing O002 v1 development outputs")
    o002_v2.add_argument("--output", required=True, help="new robustness output directory")
    o002_v2.set_defaults(func=_o002_robustness_command)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
