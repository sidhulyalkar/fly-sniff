from __future__ import annotations

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from .olfactory_motion_audit import _stream_selected_edges
from .public_data import sha256_file

PROTOCOL = "olfactory-motion-connectivity-audit-v1"
DEFAULT_CONFIG_RESOURCE = "configs/olfactory_motion_connectivity_v1.json"


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _default_config_text() -> str:
    resource = files("fly_sniff").joinpath(DEFAULT_CONFIG_RESOURCE)
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    return (Path(__file__).resolve().parents[2] / "configs" / "olfactory_motion_connectivity_v1.json").read_text()


def load_connectivity_config(path: str | Path | None = None) -> dict[str, Any]:
    document = json.loads(Path(path).read_text() if path is not None else _default_config_text())
    if document.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported connectivity protocol {document.get('protocol')!r}")
    if document.get("dataset") != "male-cns:v1.0":
        raise ValueError("connectivity audit requires male-cns:v1.0")
    for field in (
        "controller_access",
        "navigation_performance_used",
        "may_reselect_candidates",
        "may_infer_neurotransmitter_sign",
        "may_infer_delay",
    ):
        if document.get(field) is not False:
            raise ValueError(f"connectivity contract requires {field}=false")
    return document


def _normalized(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def _gate_a_hash(report: dict[str, Any]) -> str:
    payload = dict(report)
    claimed = payload.pop("report_sha256", None)
    calculated = _canonical_sha(payload)
    if claimed != calculated:
        raise ValueError(f"Gate A self-hash mismatch: claimed {claimed}, calculated {calculated}")
    return calculated


def _flatten_candidates(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, records in report.get("candidates", {}).items():
        for record in records:
            row = dict(record)
            if row.get("candidate_family") != family:
                raise ValueError("Gate A candidate family key/record mismatch")
            rows.append(row)
    if not rows:
        raise ValueError("Gate A contains no candidates")
    frame = pd.DataFrame(rows)
    if frame["bodyId"].duplicated().any():
        duplicates = sorted(frame.loc[frame["bodyId"].duplicated(False), "bodyId"].astype(int).unique())
        raise ValueError(f"Gate A contains duplicate candidate body IDs: {duplicates[:20]}")
    frame["bodyId"] = frame["bodyId"].astype(int)
    return frame.sort_values(["candidate_family", "bodyId"]).reset_index(drop=True)


def verify_gate_a_receipt(
    gate_a: dict[str, Any],
    annotations_path: str | Path,
    document: dict[str, Any],
) -> pd.DataFrame:
    expected = document["gate_a"]
    gate_hash = _gate_a_hash(gate_a)
    if gate_hash != expected["report_sha256"]:
        raise ValueError("Gate A report SHA-256 differs from frozen Gate B contract")
    if gate_a.get("protocol") != expected["required_protocol"]:
        raise ValueError("Gate A protocol mismatch")
    if gate_a.get("status") != expected["required_status"]:
        raise ValueError("Gate A did not reach the required complete-candidate status")
    if gate_a.get("annotation_sha256") != expected["annotation_sha256"]:
        raise ValueError("Gate A annotation SHA-256 differs from frozen Gate B contract")
    if gate_a.get("weights_sha256") is not None or gate_a.get("selected_edge_count") != 0:
        raise ValueError("Gate B requires the annotation-only Gate A receipt")
    if gate_a.get("controller_access") is not False or gate_a.get("navigation_performance_used") is not False:
        raise ValueError("Gate A claim boundary was weakened")
    if gate_a.get("functional_motion_claim_allowed") is not False or gate_a.get("navigation_claim_allowed") is not False:
        raise ValueError("Gate A contains an impermissible functional/navigation promotion")
    if gate_a.get("candidate_counts") != expected["candidate_counts"]:
        raise ValueError("Gate A candidate counts differ from frozen Gate B contract")
    candidates = _flatten_candidates(gate_a)
    if len(candidates) != int(expected["selected_body_count"]):
        raise ValueError("Gate A selected-body count differs from frozen Gate B contract")

    annotations_path = Path(annotations_path)
    actual_annotation_sha = sha256_file(annotations_path)
    if actual_annotation_sha != gate_a["annotation_sha256"]:
        raise ValueError("current annotation bytes do not match frozen Gate A annotation SHA-256")
    annotations = pd.read_feather(annotations_path)
    if "bodyId" not in annotations.columns:
        raise ValueError("MaleCNS annotation table is missing bodyId")
    if annotations["bodyId"].duplicated().any():
        raise ValueError("MaleCNS annotation table contains duplicate bodyId rows")
    indexed = annotations.set_index("bodyId", drop=False)
    identity_fields = ("bodyId", "instance", "type", "class", "subclass", "somaSide", "rootSide")
    for record in candidates.to_dict(orient="records"):
        body_id = int(record["bodyId"])
        if body_id not in indexed.index:
            raise ValueError(f"Gate A body {body_id} is absent from current annotations")
        current = indexed.loc[body_id]
        for field in identity_fields:
            if field not in record or field not in annotations.columns:
                continue
            expected_value = _normalized(record[field])
            actual_value = _normalized(current[field])
            if expected_value != actual_value:
                raise ValueError(
                    f"Gate A identity mismatch for body {body_id} field {field}: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )
    return candidates


def _side(record: dict[str, Any], document: dict[str, Any]) -> tuple[str, str]:
    family = str(record["candidate_family"])
    field = document["side_authority"][family]
    value = _normalized(record.get(field))
    return (value if value in {"L", "R"} else "unknown", field)


def _relation(source_side: str, target_side: str) -> str:
    if "unknown" in {source_side, target_side}:
        return "unknown"
    return "same_annotation_side" if source_side == target_side else "opposite_annotation_side"


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    value = json.loads(frame.to_json(orient="records"))
    return value if isinstance(value, list) else []


def build_connectivity_audit(
    gate_a_path: str | Path,
    annotations_path: str | Path,
    weights_path: str | Path,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    document = load_connectivity_config(config_path)
    gate_a = json.loads(Path(gate_a_path).read_text())
    candidates = verify_gate_a_receipt(gate_a, annotations_path, document)
    selected = set(candidates["bodyId"].astype(int))
    edges = _stream_selected_edges(weights_path, selected)

    candidate_records = {
        int(record["bodyId"]): record for record in candidates.to_dict(orient="records")
    }
    family_by_body = {body: str(record["candidate_family"]) for body, record in candidate_records.items()}
    if not edges.empty:
        edges["source_family"] = edges["source"].map(family_by_body)
        edges["target_family"] = edges["target"].map(family_by_body)
        source_side = edges["source"].map(lambda body: _side(candidate_records[int(body)], document))
        target_side = edges["target"].map(lambda body: _side(candidate_records[int(body)], document))
        edges["source_side"] = source_side.map(lambda value: value[0])
        edges["source_side_basis"] = source_side.map(lambda value: value[1])
        edges["target_side"] = target_side.map(lambda value: value[0])
        edges["target_side_basis"] = target_side.map(lambda value: value[1])
        edges["side_relation"] = [
            _relation(source, target)
            for source, target in zip(edges["source_side"], edges["target_side"])
        ]

    if edges.empty:
        family_summary = pd.DataFrame()
        side_summary = pd.DataFrame()
    else:
        family_summary = (
            edges.groupby(["source_family", "target_family"], dropna=False)
            .agg(edge_count=("weight", "size"), aggregate_weight=("weight", "sum"))
            .reset_index()
        )
        side_summary = (
            edges.groupby(
                ["source_family", "target_family", "source_side", "target_side", "side_relation"],
                dropna=False,
            )
            .agg(edge_count=("weight", "size"), aggregate_weight=("weight", "sum"))
            .reset_index()
        )

    path_results: list[dict[str, Any]] = []
    required_present: list[bool] = []
    for specification in document["prespecified_family_edges"]:
        source_family = specification["source"]
        target_family = specification["target"]
        subset = edges[
            (edges.get("source_family", pd.Series(index=edges.index, dtype=str)) == source_family)
            & (edges.get("target_family", pd.Series(index=edges.index, dtype=str)) == target_family)
        ]
        laterality = []
        if not subset.empty:
            grouped = (
                subset.groupby("side_relation", dropna=False)
                .agg(edge_count=("weight", "size"), aggregate_weight=("weight", "sum"))
                .reset_index()
            )
            laterality = _records(grouped)
        result = {
            **specification,
            "present": bool(len(subset)),
            "edge_count": len(subset),
            "aggregate_weight": int(subset["weight"].sum()) if len(subset) else 0,
            "laterality": laterality,
        }
        path_results.append(result)
        if specification["required_for_candidate_motif"]:
            required_present.append(result["present"])

    candidate_identity = [
        {
            "bodyId": int(record["bodyId"]),
            "candidate_family": record["candidate_family"],
            "side": _side(record, document)[0],
            "side_basis": _side(record, document)[1],
        }
        for record in candidates.to_dict(orient="records")
    ]
    motif_present = bool(required_present) and all(required_present)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": "male-cns:v1.0",
        "status": "candidate_structural_motif_present" if motif_present else "candidate_structural_motif_incomplete",
        "gate_a_report_sha256": gate_a["report_sha256"],
        "annotation_sha256": gate_a["annotation_sha256"],
        "weights_sha256": sha256_file(weights_path),
        "frozen_candidate_identity_sha256": _canonical_sha(candidate_identity),
        "selected_body_count": len(candidates),
        "selected_edge_count": len(edges),
        "family_connectivity": _records(family_summary),
        "side_connectivity": _records(side_summary),
        "prespecified_paths": path_results,
        "all_required_structural_paths_present": motif_present,
        "side_authority": document["side_authority"],
        "laterality_contract": document["laterality_contract"],
        "claim_boundary": document["claim_boundary"],
        "controller_access": False,
        "navigation_performance_used": False,
        "neurotransmitter_sign_inferred": False,
        "delay_inferred": False,
        "functional_motion_claim_allowed": False,
        "navigation_claim_allowed": False,
    }
    report["report_sha256"] = _canonical_sha(report)
    return report, edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Gate A olfactory candidate connectivity")
    parser.add_argument("--gate-a", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--edges-output")
    args = parser.parse_args()
    report, edges = build_connectivity_audit(
        args.gate_a,
        args.annotations,
        args.weights,
        args.config,
    )
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
