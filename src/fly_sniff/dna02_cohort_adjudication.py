from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .freeze import canonical_sha256

SCHEMA = "fly-sniff-dna02-cohort-adjudication-v1"
STATUS = "RESOLVED_COHORT_ONLY"
EXPECTED_COHORT = ("a2_d_08", "a2_d_12", "a2_d_13", "a2_d_14")
EXPECTED_METADATA_SHA256 = "1bc78258d1ec4ffab4a80ac30a99bb35aee46a3b8a9c057ddf5464fe667c57fe"
EXPECTED_README_SHA256 = "91409a03266dbddb44035df3177440ba0acc2a67aa75118a3d97c267038c9681"
EXPECTED_CODE_COMMIT = "7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8"
EXPECTED_FILES = {
    "a2_d_08": (
        11634638,
        "rayshubskiy_elife_102230/ephys_data_a2_d_08",
        "180410_gfp_3G_ss730_dual_08_data_for_SH_with_lat_vel.mat",
        "f3d0d40d7435af8f3d4e73d004bd3f9d",
    ),
    "a2_d_12": (
        11634639,
        "rayshubskiy_elife_102230/ephys_data_a2_d_12",
        "180430_gfp_3G_ss730_dual_12_data_for_SH_with_lat_vel.mat",
        "71ed13ab29ecd3b3abcf7bf6b77940e4",
    ),
    "a2_d_13": (
        11634640,
        "rayshubskiy_elife_102230/ephys_data_a2_d_13",
        "180501_gfp_3G_ss730_dual_13_data_for_SH_with_lat_vel.mat",
        "c3f95c76e9efc1a5b2277e7ee958143e",
    ),
    "a2_d_14": (
        11634641,
        "rayshubskiy_elife_102230/ephys_data_a2_d_14",
        "180517_gfp_3G_ss730_dual_14_data_for_SH_with_lat_vel.mat",
        "eb0950ad281d61c4473404ea126b386a",
    ),
}


def authority_ref(payload: dict[str, Any]) -> str:
    validate_adjudication(payload)
    return f"sha256:{payload['adjudication_sha256']}"


def _payload_without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "adjudication_sha256"}


def validate_adjudication(payload: dict[str, Any]) -> None:
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported DNa02 cohort adjudication schema")
    if payload.get("status") != STATUS:
        raise ValueError("DNa02 cohort adjudication must resolve cohort only")

    paper = payload["paper_authority"]
    if paper.get("doi") != "10.7554/eLife.102230.3":
        raise ValueError("cohort adjudication must bind the version-of-record DOI")
    if paper.get("figure") != "Figure 3C" or int(paper.get("published_n_flies", -1)) != 4:
        raise ValueError("cohort adjudication must remain scoped to Figure 3C n=4")
    definition = str(paper.get("cohort_definition", "")).lower()
    if "right" not in definition or "left" not in definition or "dna02" not in definition:
        raise ValueError("paper authority must explicitly define bilateral DNa02 recordings")

    dataverse = payload["dataverse_authority"]
    if dataverse.get("doi") != "10.7910/DVN/0NCLP1":
        raise ValueError("cohort adjudication must bind the published Dataverse DOI")
    if int(dataverse.get("dataset_id", -1)) != 11618096:
        raise ValueError("unexpected Dataverse dataset id")
    if str(dataverse.get("version")) != "1.2" or dataverse.get("version_state") != "RELEASED":
        raise ValueError("cohort adjudication must bind released Dataverse version 1.2")
    if int(dataverse.get("file_count", -1)) != 380:
        raise ValueError("Dataverse file inventory count changed")
    if dataverse.get("raw_metadata_sha256") != EXPECTED_METADATA_SHA256:
        raise ValueError("Dataverse metadata SHA-256 does not match reviewed evidence")
    readme = dataverse["readme"]
    if int(readme.get("file_id", -1)) != 11634832:
        raise ValueError("unexpected Dataverse README file id")
    if readme.get("observed_sha256") != EXPECTED_README_SHA256:
        raise ValueError("Dataverse README SHA-256 does not match reviewed evidence")

    rows = dataverse.get("bilateral_dna02_raw_folders", [])
    if int(dataverse.get("exact_bilateral_dna02_raw_folder_count", -1)) != 4 or len(rows) != 4:
        raise ValueError("released dataset must contain exactly four bilateral DNa02 raw folders")
    observed_aliases: set[str] = set()
    for row in rows:
        alias = str(row["fly_alias"])
        if alias in observed_aliases:
            raise ValueError(f"duplicate bilateral DNa02 alias: {alias}")
        if alias not in EXPECTED_FILES:
            raise ValueError(f"unexpected bilateral DNa02 alias: {alias}")
        expected_id, expected_dir, expected_name, expected_md5 = EXPECTED_FILES[alias]
        if (
            int(row["file_id"]) != expected_id
            or row["directory_label"] != expected_dir
            or row["filename"] != expected_name
            or row["checksum_type"] != "MD5"
            or row["checksum_value"] != expected_md5
            or bool(row["restricted"])
        ):
            raise ValueError(f"Dataverse identity mismatch for {alias}")
        observed_aliases.add(alias)
    if observed_aliases != set(EXPECTED_COHORT):
        raise ValueError("Dataverse bilateral DNa02 alias set is incomplete")

    code = payload["code_authority"]
    if code.get("repository") != "wilson-lab/rayshubskiy_elife_102230_secondary_analysis_code":
        raise ValueError("unexpected secondary-analysis repository")
    if code.get("commit") != EXPECTED_CODE_COMMIT:
        raise ValueError("secondary-analysis authority must use the pinned commit")
    completed_aliases = tuple(code.get("completed_bilateral_aliases", []))
    if completed_aliases != EXPECTED_COHORT:
        raise ValueError("completed bilateral alias set does not match reviewed cohort")
    by_path = {item["path"]: item for item in code.get("sources", [])}
    expected_blobs = {
        "import_preprocess_data.ipynb": "0da2089b468c172f700881b714bfdda99a6fe424",
        "kv_linrel_scatters.ipynb": "693bbdd8b60083744239ad049dd39f4d988418a4",
    }
    if set(by_path) != set(expected_blobs):
        raise ValueError("cohort adjudication must retain both pinned secondary-analysis sources")
    for path, blob in expected_blobs.items():
        if by_path[path].get("git_blob_sha1") != blob:
            raise ValueError(f"secondary-analysis blob changed: {path}")

    conflict = payload["conflicting_evidence"]
    if conflict.get("commit") != EXPECTED_CODE_COMMIT:
        raise ValueError("conflicting QC evidence must be pinned to the same reviewed commit")
    if conflict.get("path") != "physiology_quant_analysis_figures.ipynb":
        raise ValueError("unexpected conflicting-evidence path")
    if conflict.get("git_blob_sha1") != "384f5d342e089e101600955820fa46306b1c2593":
        raise ValueError("conflicting QC evidence blob changed")
    scope = str(conflict.get("scope_decision", "")).lower()
    if "not panel-level" not in scope:
        raise ValueError("a2_d_14 low-SNR note must remain scoped away from Figure 3C exclusion")

    decision = payload["decision_rule"]
    if decision.get("all_conditions_met") is not True:
        raise ValueError("cohort may not resolve unless all frozen evidence conditions are met")
    if decision.get("navigation_performance_used") is not False:
        raise ValueError("navigation performance may not enter cohort adjudication")
    if tuple(payload.get("resolved_figure3c_cohort", [])) != EXPECTED_COHORT:
        raise ValueError("resolved Figure 3C cohort must equal the four-source identity set")

    remaining = str(payload.get("remaining_blocker", "")).lower()
    if "sha-256" not in remaining or "does not authorize" not in remaining:
        raise ValueError("cohort adjudication must preserve the unresolved byte-identity boundary")
    forbidden = " ".join(str(value).lower() for value in payload.get("forbidden_interpretations", []))
    if "md5" not in forbidden or "ready_for_extraction" not in forbidden or "a2_d_14" not in forbidden:
        raise ValueError("cohort adjudication is missing required interpretation guards")

    claimed = str(payload.get("adjudication_sha256", ""))
    computed = canonical_sha256(_payload_without_hash(payload))
    if claimed != computed:
        raise ValueError("DNa02 cohort adjudication hash mismatch")


def load_adjudication(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DNa02 cohort adjudication must be a JSON object")
    validate_adjudication(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen DNa02 Figure 3C cohort adjudication")
    parser.add_argument("manifest")
    args = parser.parse_args(argv)
    payload = load_adjudication(args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
