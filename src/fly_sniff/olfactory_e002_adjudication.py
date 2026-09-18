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


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError("E002 adjudication must be a JSON object")
    return payload


def _records(payload: Any, *, kind: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise TypeError(f"{kind} body_records must be a list")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            raise TypeError(f"{kind} body record must be an object")
        root_id = row.get("root_id")
        if not isinstance(root_id, str) or not root_id.isdigit():
            raise ValueError(f"{kind} root_id must be a decimal string")
        if root_id in seen:
            raise ValueError(f"{kind} duplicate root_id {root_id}")
        seen.add(root_id)
        rows.append(row)
    return rows


def validate_da2_adjudication(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("E002 adjudication requires schema_version=1")
    if payload.get("authority_id") != "E002_da2_pathway_identity_adjudication":
        raise ValueError("E002 adjudication authority id changed")
    if payload.get("program_id") != "olfactory-computation-v0":
        raise ValueError("E002 adjudication program id changed")
    if payload.get("dataset") != EXPECTED_ANNOTATION:
        raise ValueError("E002 adjudication must retain the exact FlyWire annotation authority")

    publication = payload.get("publication_authority")
    if not isinstance(publication, dict):
        raise TypeError("E002 adjudication requires publication_authority")
    expected_publication = {
        "reported_annotated_Or56a_OSNs": 40,
        "reported_additional_DA2_afferents": 1,
        "reported_total_Or56a_OSNs": 41,
        "reported_right_DA2_ipsilateral": 22,
        "reported_right_DA2_contralateral": 19,
    }
    for key, expected in expected_publication.items():
        if publication.get(key) != expected:
            raise ValueError(f"E002 publication count changed: {key}")

    sensory = payload.get("exact_annotation_adjudication")
    if not isinstance(sensory, dict):
        raise TypeError("E002 adjudication requires exact_annotation_adjudication")
    if sensory.get("selector") != "hemibrain_type == ORN_DA2":
        raise ValueError("E002 sensory selector changed")
    osn = _records(sensory.get("body_records"), kind="ORN_DA2")
    if sensory.get("table_resolved_count") != 39 or len(osn) != 39:
        raise ValueError("E002 adjudication must retain exactly 39 table-resolved ORN_DA2 rows")
    if any(row.get("hemibrain_type") != "ORN_DA2" for row in osn):
        raise ValueError("E002 adjudication contains a non-ORN_DA2 sensory row")
    source_sides = {
        side: sum(row.get("source_side") == side for row in osn)
        for side in ("left", "right", "na")
    }
    if source_sides != {"left": 16, "right": 22, "na": 1}:
        raise ValueError("E002 frozen ORN_DA2 source-side counts changed")

    projection = payload.get("projection_neuron_adjudication")
    if not isinstance(projection, dict):
        raise TypeError("E002 adjudication requires projection_neuron_adjudication")
    if projection.get("selector") != "hemibrain_type == DA2_lPN":
        raise ValueError("E002 projection selector changed")
    pn = _records(projection.get("body_records"), kind="DA2_lPN")
    if projection.get("table_resolved_count") != 11 or len(pn) != 11:
        raise ValueError("E002 adjudication must retain exactly 11 table-resolved DA2_lPN rows")
    if any(row.get("hemibrain_type") != "DA2_lPN" for row in pn):
        raise ValueError("E002 adjudication contains a non-DA2_lPN projection row")
    if KNOWN_DA2_LPN_ANCHOR not in {row["root_id"] for row in pn}:
        raise ValueError("E002 adjudication lost the independently anchored DA2_lPN")

    discrepancy = payload.get("discrepancy")
    if not isinstance(discrepancy, dict):
        raise TypeError("E002 adjudication requires discrepancy")
    if discrepancy.get("publication_total_Or56a_OSNs") != 41:
        raise ValueError("E002 publication total changed")
    if discrepancy.get("exact_annotation_table_ORN_DA2_rows") != 39:
        raise ValueError("E002 annotation-table count changed")
    if discrepancy.get("unresolved_body_count") != 2:
        raise ValueError("E002 unresolved body count must remain 2")
    unresolved_side = discrepancy.get("unresolved_side_rows")
    if not isinstance(unresolved_side, list) or len(unresolved_side) != 1:
        raise ValueError("E002 must retain the one unresolved source-side row")
    if unresolved_side[0] not in {row["root_id"] for row in osn if row.get("source_side") == "na"}:
        raise ValueError("E002 unresolved-side root ID changed")

    if payload.get("status") != "blocked_2_osn_root_ids_and_1_side_unresolved":
        raise ValueError("E002 adjudication may not be promoted before discrepancy resolution")
    if payload.get("confirmatory_usable") is not False:
        raise ValueError("partial E002 adjudication cannot be confirmatory-usable")

    return {
        "status": payload["status"],
        "table_resolved_osn_body_records": len(osn),
        "publication_total_osns": 41,
        "unresolved_osn_body_records": 2,
        "unresolved_osn_side_records": 1,
        "da2_lpn_body_records": len(pn),
        "known_anchor_present": True,
        "confirmatory_usable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen partial FlyWire Or56a/DA2 cohort adjudication"
    )
    parser.add_argument("authority")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_da2_adjudication(_load(args.authority)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
