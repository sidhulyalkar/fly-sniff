from __future__ import annotations

import argparse
import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import ipc

from .public_data import sha256_file

PROTOCOL = "olfactory-motion-structural-audit-v1"
DEFAULT_CONFIG_RESOURCE = "configs/olfactory_motion_audit_v1.json"


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _default_config_text() -> str:
    resource = files("fly_sniff").joinpath(DEFAULT_CONFIG_RESOURCE)
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    return (Path(__file__).resolve().parents[2] / "configs" / "olfactory_motion_audit_v1.json").read_text()


def load_audit_config(path: str | Path | None = None) -> dict[str, Any]:
    document = json.loads(Path(path).read_text() if path is not None else _default_config_text())
    if document.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported olfactory-motion audit protocol {document.get('protocol')!r}")
    if document.get("dataset") != "male-cns:v1.0":
        raise ValueError("olfactory-motion audit requires male-cns:v1.0")
    if document.get("controller_access") is not False or document.get("navigation_performance_used") is not False:
        raise ValueError("structural audit must remain independent of controller/navigation performance")
    if document.get("may_import_body_ids_from_other_connectomes") is not False:
        raise ValueError("cross-connectome body IDs cannot be selection authority")
    return document


def _configured_text_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    present = [column for column in columns if column in frame.columns]
    if not present:
        raise ValueError(f"no configured annotation text columns found; columns={list(frame.columns)}")
    return present


def _row_matches(
    frame: pd.DataFrame,
    columns: list[str],
    compiled: list[re.Pattern[str]],
    raw_patterns: list[str],
) -> tuple[pd.Series, dict[int, list[dict[str, str]]]]:
    matches: dict[int, list[dict[str, str]]] = {}
    mask = pd.Series(False, index=frame.index)
    text = frame[columns].fillna("").astype(str)
    for row_index, row in text.iterrows():
        row_matches: list[dict[str, str]] = []
        for column in columns:
            value = row[column]
            for pattern_index, pattern in enumerate(compiled):
                if pattern.search(value):
                    row_matches.append({"column": column, "pattern": raw_patterns[pattern_index]})
        if row_matches:
            mask.loc[row_index] = True
            matches[int(row_index)] = row_matches
    return mask, matches


def resolve_candidate_families(
    annotations: pd.DataFrame,
    document: dict[str, Any],
) -> pd.DataFrame:
    if "bodyId" not in annotations.columns:
        raise ValueError("MaleCNS annotations must contain bodyId")
    text_columns = _configured_text_columns(annotations, list(document["annotation_text_columns"]))
    pieces: list[pd.DataFrame] = []
    identity = [
        column
        for column in document["identity_columns_if_available"]
        if column in annotations.columns
    ]
    for family, patterns in document["candidate_families"].items():
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        mask, match_details = _row_matches(annotations, text_columns, compiled, patterns)
        if not bool(mask.any()):
            continue
        part = annotations.loc[mask, identity].copy()
        part["candidate_family"] = family
        part["matched_patterns"] = [match_details[int(row_index)] for row_index in part.index]
        pieces.append(part)
    if not pieces:
        return pd.DataFrame(columns=[*identity, "candidate_family", "matched_patterns"])
    resolved = pd.concat(pieces, ignore_index=True)
    resolved["bodyId"] = resolved["bodyId"].astype(int)
    ambiguous = resolved.groupby("bodyId")["candidate_family"].nunique()
    ambiguous_ids = sorted(int(body) for body in ambiguous[ambiguous > 1].index)
    if ambiguous_ids:
        raise ValueError(f"candidate regexes assign bodies to multiple families: {ambiguous_ids[:20]}")
    return resolved.drop_duplicates(subset=["bodyId", "candidate_family"]).sort_values(
        ["candidate_family", "bodyId"]
    ).reset_index(drop=True)


def _stream_selected_edges(weights_path: str | Path, selected: set[int]) -> pd.DataFrame:
    if not selected:
        return pd.DataFrame(columns=["source", "target", "weight"])
    values = pa.array(sorted(selected), type=pa.int64())
    reader = ipc.open_file(pa.memory_map(str(weights_path)))
    tables: list[pa.Table] = []
    for index in range(reader.num_record_batches):
        batch = reader.get_batch(index)
        missing = {"body_pre", "body_post", "weight"} - set(batch.schema.names)
        if missing:
            raise ValueError(f"weights file missing columns: {sorted(missing)}")
        mask = pc.and_(
            pc.is_in(batch["body_pre"], value_set=values),
            pc.is_in(batch["body_post"], value_set=values),
        )
        if bool(pc.any(mask).as_py()):
            tables.append(
                pa.Table.from_batches([batch.filter(mask)]).select(
                    ["body_pre", "body_post", "weight"]
                )
            )
    if not tables:
        return pd.DataFrame(columns=["source", "target", "weight"])
    frame = pa.concat_tables(tables).to_pandas().rename(
        columns={"body_pre": "source", "body_post": "target"}
    )
    if frame.duplicated(["source", "target"]).any():
        raise ValueError("MaleCNS weight table contains duplicate selected body pairs")
    frame[["source", "target", "weight"]] = frame[["source", "target", "weight"]].astype(int)
    return frame.sort_values(["source", "target"]).reset_index(drop=True)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = json.loads(frame.to_json(orient="records"))
    return records if isinstance(records, list) else []


def build_olfactory_motion_audit(
    annotations_path: str | Path,
    weights_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    document = load_audit_config(config_path)
    annotations_path = Path(annotations_path)
    annotations = pd.read_feather(annotations_path)
    candidates = resolve_candidate_families(annotations, document)
    counts = {
        family: int((candidates["candidate_family"] == family).sum())
        for family in document["candidate_families"]
    }
    missing_candidate = [
        family for family in document["required_families_for_candidate_audit"] if counts.get(family, 0) == 0
    ]
    missing_complete = [
        family for family in document["required_families_for_complete_motif"] if counts.get(family, 0) == 0
    ]
    if missing_candidate:
        status = "blocked_missing_required_candidate_family"
    elif missing_complete:
        status = "candidate_audit_pass_incomplete_motif"
    else:
        status = "candidate_audit_pass_complete_structural_motif"

    selected = set(candidates["bodyId"].astype(int)) if len(candidates) else set()
    edges = pd.DataFrame(columns=["source", "target", "weight", "source_family", "target_family"])
    weights_sha = None
    family_summaries: list[dict[str, Any]] = []
    if weights_path is not None:
        weights_path = Path(weights_path)
        weights_sha = sha256_file(weights_path)
        edges = _stream_selected_edges(weights_path, selected)
        family_by_body = dict(zip(candidates["bodyId"].astype(int), candidates["candidate_family"]))
        if len(edges):
            edges["source_family"] = edges["source"].map(family_by_body)
            edges["target_family"] = edges["target"].map(family_by_body)
            grouped = (
                edges.groupby(["source_family", "target_family"], dropna=False)
                .agg(edge_count=("weight", "size"), aggregate_weight=("weight", "sum"))
                .reset_index()
            )
            family_summaries = _records(grouped)

    candidate_records = {
        family: _records(candidates[candidates["candidate_family"] == family])
        for family in document["candidate_families"]
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": "male-cns:v1.0",
        "status": status,
        "annotation_sha256": sha256_file(annotations_path),
        "weights_sha256": weights_sha,
        "selection_authority": "exact male-cns:v1.0 annotation table; no cross-connectome body IDs",
        "candidate_counts": counts,
        "missing_required_candidate_families": missing_candidate,
        "missing_complete_motif_families": missing_complete,
        "candidates": candidate_records,
        "selected_body_count": len(selected),
        "selected_edge_count": len(edges),
        "family_connectivity": family_summaries,
        "literature_hypothesis": document["literature_hypothesis"],
        "claim_boundary": document["claim_boundary"],
        "controller_access": False,
        "navigation_performance_used": False,
        "functional_motion_claim_allowed": False,
        "navigation_claim_allowed": False,
    }
    report["report_sha256"] = _canonical_sha(report)
    return report, edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MaleCNS v1.0 olfactory-motion candidate structure")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--edges-output")
    args = parser.parse_args()
    report, edges = build_olfactory_motion_audit(args.annotations, args.weights, args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.edges_output:
        edge_path = Path(args.edges_output)
        edge_path.parent.mkdir(parents=True, exist_ok=True)
        edges.to_parquet(edge_path, index=False)
    print(output)
    print(report["report_sha256"])
    print(report["status"])


if __name__ == "__main__":
    main()
