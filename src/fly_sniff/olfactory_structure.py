from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_ANNOTATION = {
    "name": "FlyWire FAFB",
    "materialization": 783,
    "sex": "female",
    "annotation_repository": "flyconnectome/flywire_annotations",
    "annotation_release": "v2.1.0",
    "annotation_commit": "ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0",
    "annotation_tree": "0487505ae0638638b8871ed120d74b56c339bdc6",
    "annotation_table_path": "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    "annotation_table_git_blob": "1a3168731618ee62a47392252d3af7664e739e9e",
}
KNOWN_DA2_LPN_ANCHOR = "720575940624106442"
ALLOWED_SIDES = {"left", "right", "unresolved"}
ALLOWED_IDENTITY_GRADES = {
    "direct_public_cross_reference",
    "publication_matched_annotation",
    "manual_morphology_adjudication",
}


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError("E002 authority must be a JSON object")
    return payload


def _validate_body_records(records: Any, *, population_id: str) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise TypeError(f"{population_id}: body_records must be a list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"{population_id}: body record must be an object")
        root_id = record.get("root_id")
        if not isinstance(root_id, str) or not root_id.isdigit():
            raise ValueError(f"{population_id}: body root_id must be a decimal string")
        if root_id in seen:
            raise ValueError(f"{population_id}: duplicate root_id {root_id}")
        seen.add(root_id)
        if record.get("side") not in ALLOWED_SIDES:
            raise ValueError(f"{population_id}: every body requires left/right or explicit unresolved side")
        if record.get("identity_grade") not in ALLOWED_IDENTITY_GRADES:
            raise ValueError(f"{population_id}: unsupported identity grade")
        authority = record.get("identity_authority")
        if not isinstance(authority, str) or not authority.strip():
            raise ValueError(f"{population_id}: every body requires an identity authority")
        normalized.append(record)
    return normalized


def validate_da2_authority(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("E002 authority requires schema_version=1")
    if payload.get("authority_id") != "E002_da2_pathway_identity":
        raise ValueError("E002 authority id changed")
    if payload.get("program_id") != "olfactory-computation-v0":
        raise ValueError("E002 program id changed")
    if payload.get("dataset") != EXPECTED_ANNOTATION:
        raise ValueError("E002 must use the publication-matched FlyWire v783 annotation authority")

    anchors = payload.get("known_anchors")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("E002 requires at least one independently anchored identity")
    anchor = next((row for row in anchors if row.get("root_id") == KNOWN_DA2_LPN_ANCHOR), None)
    if anchor is None:
        raise ValueError("E002 lost the independently cross-referenced DA2_lPN anchor")
    if anchor.get("cell_type") != "DA2_lPN" or anchor.get("side") != "right":
        raise ValueError("known DA2_lPN anchor identity changed")
    if anchor.get("identity_grade") != "direct_public_cross_reference":
        raise ValueError("known DA2_lPN anchor must retain direct-public-cross-reference grade")

    constraints = payload.get("population_constraints")
    if not isinstance(constraints, list):
        raise TypeError("E002 population_constraints must be a list")
    by_id = {row.get("population_id"): row for row in constraints if isinstance(row, dict)}
    required = {"or56a_osn_to_right_DA2", "DA2_lPN_publication_matched"}
    if set(by_id) != required:
        raise ValueError("E002 requires exactly the frozen Or56a OSN and DA2_lPN populations")

    osn = by_id["or56a_osn_to_right_DA2"]
    if osn.get("expected_total") != 41:
        raise ValueError("Or56a OSN total-count constraint changed")
    if osn.get("expected_ipsilateral") != 22 or osn.get("expected_contralateral") != 19:
        raise ValueError("Or56a OSN side-count constraint changed")
    if osn["expected_ipsilateral"] + osn["expected_contralateral"] != osn["expected_total"]:
        raise ValueError("Or56a OSN side counts do not sum to total")
    if osn.get("publication_reported_annotated_count") != 40:
        raise ValueError("Or56a publication-reported annotated count changed")
    if osn.get("publication_reported_manual_additional_count") != 1:
        raise ValueError("Or56a publication-reported manual-additional count changed")
    if osn.get("frozen_annotation_count") != 39:
        raise ValueError("Or56a exact v2.1.0 frozen annotation count changed")
    if osn.get("candidate_discovery_rule") != (
        "exact frozen v2.1.0 FlyWire hemibrain_type/cell_type ORN_DA2 only; "
        "no connectivity- or outcome-based expansion"
    ):
        raise ValueError("Or56a OSN candidate discovery rule changed")
    osn_records = _validate_body_records(
        osn.get("body_records", []), population_id=osn["population_id"]
    )
    frozen_osn_records = [
        row
        for row in osn_records
        if "exact blob 1a3168731618ee62a47392252d3af7664e739e9e"
        in row.get("identity_authority", "")
    ]
    if len(frozen_osn_records) != 39:
        raise ValueError("E002 must retain all 39 exact v2.1.0 ORN_DA2 annotation records")
    unresolved_osn = [row for row in osn_records if row["side"] == "unresolved"]

    pn = by_id["DA2_lPN_publication_matched"]
    if pn.get("candidate_discovery_rule") != (
        "exact publication-matched annotation hemibrain_type/cell_type DA2_lPN only"
    ):
        raise ValueError("DA2_lPN candidate discovery rule changed")
    if pn.get("frozen_annotation_count") != 11:
        raise ValueError("DA2_lPN exact v2.1.0 frozen annotation count changed")
    pn_records = _validate_body_records(
        pn.get("body_records", []), population_id=pn["population_id"]
    )
    if len(pn_records) != 11:
        raise ValueError("E002 must retain all 11 exact v2.1.0 DA2_lPN records")
    if pn.get("enumeration_complete") is not True:
        raise ValueError("DA2_lPN exact annotation cohort must remain complete")
    if any(row["side"] == "unresolved" for row in pn_records):
        raise ValueError("DA2_lPN cohort may not contain unresolved sides")
    if KNOWN_DA2_LPN_ANCHOR not in {row["root_id"] for row in pn_records}:
        raise ValueError("frozen DA2_lPN candidate records must retain the known anchor")

    discrepancies = payload.get("unresolved_discrepancies")
    if not isinstance(discrepancies, list):
        raise TypeError("E002 unresolved_discrepancies must be a list")
    discrepancy_ids = {
        row.get("id") for row in discrepancies if isinstance(row, dict)
    }
    required_discrepancies = {
        "orn_da2_annotation_count_gap",
        "orn_da2_manual_additional_root_id_missing",
        "orn_da2_side_unresolved_fw044213",
    }
    if payload.get("status") != "qualified_identity_cohort" and not required_discrepancies.issubset(
        discrepancy_ids
    ):
        raise ValueError("blocked E002 authority must retain all unresolved cohort discrepancies")

    rules = payload.get("qualification_rules")
    if not isinstance(rules, dict) or any(value is not True for value in rules.values()):
        raise ValueError("E002 qualification rules may not be weakened")

    osn_complete = (
        osn.get("enumeration_complete") is True
        and len(osn_records) == 41
        and not unresolved_osn
        and sum(row["side"] == "right" for row in osn_records) == 22
        and sum(row["side"] == "left" for row in osn_records) == 19
    )
    pn_complete = pn.get("enumeration_complete") is True and len(pn_records) == 11
    expected_status = "qualified_identity_cohort" if osn_complete and pn_complete else "blocked_incomplete_body_id_adjudication"
    if payload.get("status") != expected_status:
        raise ValueError("E002 status does not match body-ID adjudication completeness")

    return {
        "status": payload["status"],
        "osn_body_records": len(osn_records),
        "da2_lpn_body_records": len(pn_records),
        "known_anchor_present": True,
        "unresolved_osn_side_records": len(unresolved_osn),
        "unresolved_discrepancies": len(discrepancies),
        "confirmatory_usable": expected_status == "qualified_identity_cohort",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen FlyWire Or56a/DA2 identity authority")
    parser.add_argument("authority")
    args = parser.parse_args()
    print(json.dumps(validate_da2_authority(_load(args.authority)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
