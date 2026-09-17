from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .olfactory_door import EXPECTED_COMMIT, EXPECTED_TREE, sha256_file, verify_checkout

SENSITIVE_PREFIXES = ("authority/", "src/", "tests/", "scripts/", ".github/")
SENSITIVE_FILES = {"pyproject.toml"}


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo_provenance(root: Path) -> dict[str, Any]:
    head = _git(root, "rev-parse", "HEAD^{commit}")
    branch = _git(root, "branch", "--show-current") or "(detached)"
    tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    sensitive = sorted(
        path
        for path in untracked
        if path in SENSITIVE_FILES or any(path.startswith(prefix) for prefix in SENSITIVE_PREFIXES)
    )
    return {
        "head": head,
        "branch": branch,
        "tracked_tree_clean": not bool(tracked),
        "tracked_status": tracked.splitlines() if tracked else [],
        "sensitive_untracked": sensitive,
        "scientific_tree_clean": not bool(tracked) and not bool(sensitive),
    }


def _validate_receipt(artifact_dir: Path) -> dict[str, Any]:
    receipt_path = artifact_dir / "door-e006-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing E006 receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("authority_id") != "E006_odor_panel_receptor_responses":
        raise ValueError("unexpected E006 authority id")
    if receipt.get("protocol") != "door-e006-source-resolved-ingestion-v0":
        raise ValueError("unexpected E006 ingestion protocol")
    if receipt.get("source_commit") != EXPECTED_COMMIT:
        raise ValueError("E006 receipt source commit does not match frozen source")
    if receipt.get("source_tree") != EXPECTED_TREE:
        raise ValueError("E006 receipt source tree does not match frozen source")
    observed = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if observed != _canonical_sha(unhashed):
        raise ValueError("E006 canonical receipt hash mismatch")
    for field, expected in (
        ("normalization_applied", False),
        ("aggregation_applied", False),
        ("missingness_preserved", True),
        ("study_identity_preserved", True),
    ):
        if receipt.get(field) is not expected:
            raise ValueError(f"E006 receipt violates frozen ingestion invariant: {field}")

    long_path = artifact_dir / "door-responses-long.csv"
    mapping_path = artifact_dir / "door-unit-mappings.json"
    if sha256_file(long_path) != str(receipt["long_form_artifact"]["sha256"]):
        raise ValueError("E006 long-form artifact hash mismatch")
    if sha256_file(mapping_path) != str(receipt["mapping_artifact"]["sha256"]):
        raise ValueError("E006 mapping artifact hash mismatch")
    return receipt


def _coverage_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed = df["response_status"].eq("observed")
    study = (
        df.assign(observed=observed)
        .groupby("study_id")
        .agg(
            cells=("observed", "size"),
            observed_cells=("observed", "sum"),
            observed_fraction=("observed", "mean"),
            responding_units=("responding_unit", "nunique"),
            odor_names=("odor_name", "nunique"),
        )
        .sort_values(["observed_cells", "observed_fraction"], ascending=False)
    )
    units = (
        df.assign(observed=observed)
        .groupby("responding_unit")
        .agg(
            cells=("observed", "size"),
            observed_cells=("observed", "sum"),
            observed_fraction=("observed", "mean"),
            studies=("study_id", "nunique"),
            odor_names=("odor_name", "nunique"),
        )
        .sort_values("observed_cells", ascending=False)
    )
    odors = (
        df[df["odor_name"].notna()]
        .assign(observed=lambda x: x["response_status"].eq("observed"))
        .groupby("odor_name")
        .agg(
            cells=("observed", "size"),
            observed_cells=("observed", "sum"),
            observed_fraction=("observed", "mean"),
            responding_units=("responding_unit", "nunique"),
            studies=("study_id", "nunique"),
        )
        .sort_values("observed_cells", ascending=False)
    )
    return study, units, odors


def _study_scales(df: pd.DataFrame) -> pd.DataFrame:
    obs = df[df["response_status"].eq("observed")].copy()
    obs["response_value"] = pd.to_numeric(obs["response_value"], errors="coerce")
    return (
        obs.groupby("study_id")
        .agg(
            n=("response_value", "count"),
            responding_units=("responding_unit", "nunique"),
            odors=("odor_name", "nunique"),
            minimum=("response_value", "min"),
            q25=("response_value", lambda x: x.quantile(0.25)),
            median=("response_value", "median"),
            q75=("response_value", lambda x: x.quantile(0.75)),
            maximum=("response_value", "max"),
            mean=("response_value", "mean"),
            std=("response_value", "std"),
        )
        .sort_index()
    )


def _mapping_summary(mapping: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = {unit: len(rows) for unit, rows in mapping.items()}
    zero = sorted(unit for unit, count in counts.items() if count == 0)
    multiple = sorted(unit for unit, count in counts.items() if count > 1)
    histogram: dict[str, int] = {}
    for count in counts.values():
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    return {
        "candidate_count_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "zero_mapping_units": zero,
        "multiple_mapping_units": multiple,
        "multiple_mapping_count": len(multiple),
        "multiple_mapping_records": {unit: mapping[unit] for unit in multiple},
    }


def audit_e006(
    artifact_dir: str | Path,
    *,
    door_checkout: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    artifact = Path(artifact_dir).expanduser().resolve()
    door = Path(door_checkout).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    receipt = _validate_receipt(artifact)
    source_identity = verify_checkout(door)
    provenance = _repo_provenance(repo)

    long_path = artifact / "door-responses-long.csv"
    mapping_path = artifact / "door-unit-mappings.json"
    df = pd.read_csv(long_path)
    required = {
        "responding_unit", "odor_name", "study_id", "response_status", "response_value"
    }
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"E006 long-form artifact lacks required columns: {missing_columns}")

    study, units, odors = _coverage_tables(df)
    scales = _study_scales(df)
    mapping = json.loads(mapping_path.read_text())
    mapping_report = _mapping_summary(mapping)

    observed = df["response_status"].eq("observed")
    geosmin = df[
        df["odor_name"].fillna("").astype(str).str.casefold().eq("geosmin") & observed
    ].copy()

    study.to_csv(output / "study-coverage.csv")
    units.to_csv(output / "responding-unit-coverage.csv")
    odors.to_csv(output / "odor-coverage.csv")
    scales.to_csv(output / "study-response-scales.csv")
    geosmin.to_csv(output / "geosmin-observations.csv", index=False)
    (output / "ambiguous-unit-mappings.json").write_text(
        json.dumps(mapping_report["multiple_mapping_records"], indent=2, sort_keys=True) + "\n"
    )

    response_cells = int(len(df))
    observed_cells = int(observed.sum())
    missing_cells = response_cells - observed_cells
    observed_fraction = float(observed.mean()) if response_cells else 0.0

    blockers: list[dict[str, str]] = [
        {
            "id": "cross_study_assay_comparability_unqualified",
            "reason": (
                "The ingestion preserves study-specific raw response scales; no authority currently "
                "establishes that all study columns are numerically commensurable."
            ),
        }
    ]
    if mapping_report["multiple_mapping_count"]:
        blockers.append(
            {
                "id": "one_to_many_responding_unit_identity",
                "reason": (
                    f"{mapping_report['multiple_mapping_count']} responding units have multiple receptor/"
                    "OSN/glomerulus mapping records; receptor-level collapse requires a frozen identity policy."
                ),
            }
        )
    if not provenance["scientific_tree_clean"]:
        blockers.append(
            {
                "id": "scientific_worktree_not_clean",
                "reason": (
                    "Tracked changes or untracked files under scientific code/authority paths are present. "
                    "Diagnostics remain usable for development, but claim-bearing qualification requires "
                    "an isolated clean worktree."
                ),
            }
        )

    gate = {
        "status": "BLOCKED_METADATA_ADJUDICATION" if blockers else "READY_FOR_E006_QUALIFICATION",
        "development_analysis_allowed": True,
        "global_cross_study_matrix_allowed": False,
        "receptor_level_identity_collapse_allowed": False,
        "confirmatory_use_allowed": False,
        "blockers": blockers,
        "allowed_next_actions": [
            "adjudicate study/assay metadata and concentration provenance",
            "freeze a responding-unit identity policy without downstream model performance",
            "define within-study or explicitly commensurable development subsets",
        ],
        "forbidden_next_actions": [
            "average raw responses across studies without assay comparability authority",
            "fill source missingness with zero and treat it as measured non-response",
            "choose ambiguous mappings using decoding or navigation performance",
            "promote this audit to confirmatory O003 evidence",
        ],
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "door-e006-audit-v1",
        "authority_id": "E006_odor_panel_receptor_responses",
        "source": source_identity,
        "repository": provenance,
        "input_receipt": {
            "canonical_receipt_sha256": receipt["receipt_sha256"],
            "receipt_file_sha256": sha256_file(artifact / "door-e006-receipt.json"),
            "long_form_sha256": sha256_file(long_path),
            "mapping_sha256": sha256_file(mapping_path),
        },
        "summary": {
            "responding_units": int(df["responding_unit"].nunique()),
            "studies": int(df["study_id"].nunique()),
            "unique_odor_names": int(df["odor_name"].nunique(dropna=True)),
            "response_cells": response_cells,
            "observed_cells": observed_cells,
            "missing_cells": missing_cells,
            "observed_fraction": observed_fraction,
            "missing_fraction": 1.0 - observed_fraction,
            "geosmin_observed_cells": int(len(geosmin)),
            "multiple_mapping_units": mapping_report["multiple_mapping_count"],
        },
        "mapping": {
            key: value for key, value in mapping_report.items()
            if key != "multiple_mapping_records"
        },
        "gate": gate,
        "claim_boundary": (
            "This audit characterizes E006 provenance, missingness, study coverage, numerical scale "
            "heterogeneity, and identity ambiguity. It does not establish cross-study assay comparability, "
            "dose response, receptor-level identity for ambiguous units, or biological mechanism."
        ),
    }
    report["audit_sha256"] = _canonical_sha(report)
    (output / "e006-audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "E006 AUDIT V1",
        f"status: {gate['status']}",
        f"responding_units: {report['summary']['responding_units']}",
        f"studies: {report['summary']['studies']}",
        f"unique_odor_names: {report['summary']['unique_odor_names']}",
        f"observed_cells: {observed_cells}",
        f"missing_cells: {missing_cells}",
        f"missing_fraction: {report['summary']['missing_fraction']:.6f}",
        f"geosmin_observed_cells: {len(geosmin)}",
        f"multiple_mapping_units: {mapping_report['multiple_mapping_count']}",
        f"scientific_tree_clean: {provenance['scientific_tree_clean']}",
        f"audit_sha256: {report['audit_sha256']}",
        "",
        "BLOCKERS",
        *[f"- {item['id']}: {item['reason']}" for item in blockers],
        "",
        "NEXT ACTIONS",
        *[f"- {item}" for item in gate["allowed_next_actions"]],
        "",
        "CLAIM BOUNDARY",
        report["claim_boundary"],
    ]
    (output / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    return report
