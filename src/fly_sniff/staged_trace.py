from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .trace import body_ids_matching, trace_corridor
from .trace_audit import audit_corridor

SEED_COLUMNS = ("bodyId", "type", "instance", "class", "subclass", "side")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {
        "name": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _seed_table(annotations: pd.DataFrame, body_ids: set[int]) -> pd.DataFrame:
    columns = [column for column in SEED_COLUMNS if column in annotations.columns]
    if "bodyId" not in columns:
        raise ValueError("annotations require bodyId")
    if not body_ids:
        return annotations.loc[annotations.index[:0], columns].copy()
    body_id_numeric = pd.to_numeric(annotations.bodyId, errors="coerce")
    table = annotations.loc[body_id_numeric.isin(body_ids), columns].copy()
    table["bodyId"] = table.bodyId.astype(int)
    return table.drop_duplicates("bodyId").sort_values("bodyId").reset_index(drop=True)


def _stage_report(
    *,
    stage: dict[str, Any],
    source_count: int,
    target_count: int,
    retained_source_count: int,
    retained_target_count: int,
    node_count: int,
    edge_count: int,
) -> dict[str, Any]:
    if source_count == 0:
        status = "empty_source_seed"
    elif target_count == 0:
        status = "empty_target_seed"
    elif node_count == 0:
        status = "no_corridor"
    else:
        status = "candidate_corridor"
    return {
        "name": stage["name"],
        "status": status,
        "hypothesis_class": stage.get("hypothesis_class", "primary_navigation"),
        "required_for_primary_hypothesis": bool(
            stage.get("required_for_primary_hypothesis", True)
        ),
        "source_regex": stage["source"],
        "target_regex": stage["target"],
        "source_seed_count": source_count,
        "target_seed_count": target_count,
        "input_source_seed_count": source_count,
        "input_target_seed_count": target_count,
        "retained_source_seed_count": retained_source_count,
        "retained_target_seed_count": retained_target_count,
        "source_seed_artifact": "source_seeds.csv",
        "target_seed_artifact": "target_seeds.csv",
        "corridor_nodes": node_count,
        "corridor_edges": edge_count,
        "max_hops": stage["max_hops"],
        "min_weight": stage["min_weight"],
        "fanout": stage["fanout"],
        "rationale": stage.get("rationale", ""),
        "evidence_scope": stage.get(
            "evidence_scope",
            "structural discovery only; functional identity is not established",
        ),
    }


def _retained_seed_sets(provenance: pd.DataFrame) -> dict[str, set[int]]:
    if provenance.empty:
        return {"source": set(), "target": set()}
    source_mask = provenance.is_source_seed.astype(bool)
    target_mask = provenance.is_target_seed.astype(bool)
    return {
        "source": set(provenance.loc[source_mask, "bodyId"].astype(int)),
        "target": set(provenance.loc[target_mask, "bodyId"].astype(int)),
    }


def _audit_handoffs(
    handoffs: list[dict[str, Any]],
    retained_seeds: dict[str, dict[str, set[int]]],
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for handoff in handoffs:
        name = str(handoff["name"])
        members = handoff.get("members", [])
        if len(members) < 2:
            raise ValueError(f"handoff {name!r} requires at least two members")
        member_sets: list[set[int]] = []
        member_counts: list[dict[str, Any]] = []
        for member in members:
            stage_name = str(member["stage"])
            role = str(member["role"])
            if stage_name not in retained_seeds:
                raise ValueError(f"handoff {name!r} references unknown stage {stage_name!r}")
            if role not in {"source", "target"}:
                raise ValueError(
                    f"handoff {name!r} role must be 'source' or 'target', found {role!r}"
                )
            values = retained_seeds[stage_name][role]
            member_sets.append(values)
            member_counts.append(
                {"stage": stage_name, "role": role, "retained_seed_count": len(values)}
            )
        shared = set.intersection(*member_sets)
        minimum = int(handoff.get("min_shared_body_ids", 1))
        if minimum < 1:
            raise ValueError(f"handoff {name!r} min_shared_body_ids must be >= 1")
        required = bool(handoff.get("required_for_primary_hypothesis", True))
        reports.append(
            {
                "name": name,
                "required_for_primary_hypothesis": required,
                "members": member_counts,
                "min_shared_body_ids": minimum,
                "shared_body_id_count": len(shared),
                "shared_body_ids": sorted(shared),
                "passed": len(shared) >= minimum,
                "rationale": handoff.get("rationale", ""),
                "warning": (
                    "Body-ID continuity is structural evidence only; sharing a retained seed across "
                    "stages does not establish physiological integration or information flow."
                ),
            }
        )

    required_reports = [
        report for report in reports if report["required_for_primary_hypothesis"]
    ]
    return {
        "protocol": "staged-handoff-audit-v1",
        "handoff_count": len(reports),
        "required_handoff_count": len(required_reports),
        "required_handoff_pass_count": sum(report["passed"] for report in required_reports),
        "all_required_handoffs_pass": all(report["passed"] for report in required_reports),
        "handoffs": reports,
    }


def run_staged_trace(
    annotations: pd.DataFrame,
    weights: pd.DataFrame,
    config: dict[str, Any],
    output: str | Path,
    *,
    input_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run staged structural searches plus exact body-ID handoff checks.

    Each stage is evaluated independently so a missing upstream stage cannot hide
    downstream structural information. Exact regex-matched seed identities are
    persisted, every candidate corridor is structurally audited, and configured
    handoffs require exact retained body-ID overlap between stages. None of these
    checks promote structural discovery into physiological or functional evidence.
    """
    stages = config.get("stages", [])
    if not stages:
        raise ValueError("staged trace requires at least one stage")
    names = [stage["name"] for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError("stage names must be unique")
    if any(not name or Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("stage names must be simple directory names")
    required_stage_count = sum(
        bool(stage.get("required_for_primary_hypothesis", True)) for stage in stages
    )
    if required_stage_count == 0:
        raise ValueError("staged trace requires at least one primary required stage")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    aggregate_nodes: list[pd.DataFrame] = []
    aggregate_edges: list[pd.DataFrame] = []
    retained_seeds: dict[str, dict[str, set[int]]] = {}

    for stage in stages:
        name = stage["name"]
        stage_dir = output / name
        stage_dir.mkdir(parents=True, exist_ok=True)
        source_ids = body_ids_matching(annotations, stage["source"])
        target_ids = body_ids_matching(annotations, stage["target"])

        source_seeds = _seed_table(annotations, source_ids)
        target_seeds = _seed_table(annotations, target_ids)
        source_seeds.to_csv(stage_dir / "source_seeds.csv", index=False)
        target_seeds.to_csv(stage_dir / "target_seeds.csv", index=False)

        if source_ids and target_ids:
            nodes, edges, provenance = trace_corridor(
                annotations,
                weights,
                source_ids,
                target_ids,
                max_hops=int(stage["max_hops"]),
                min_weight=float(stage["min_weight"]),
                fanout_per_node=int(stage["fanout"]),
            )
        else:
            nodes = annotations.iloc[0:0].copy()
            edges = pd.DataFrame(columns=["source", "target", "weight"])
            provenance = pd.DataFrame(
                columns=[
                    "bodyId",
                    "forward_depth",
                    "reverse_depth",
                    "is_source_seed",
                    "is_target_seed",
                ]
            )

        nodes = nodes.copy()
        edges = edges.copy()
        provenance = provenance.copy()
        nodes["trace_stage"] = name
        edges["trace_stage"] = name
        provenance["trace_stage"] = name
        nodes.to_parquet(stage_dir / "nodes.parquet", index=False)
        edges.to_parquet(stage_dir / "edges.parquet", index=False)
        provenance.to_csv(stage_dir / "path_provenance.csv", index=False)

        stage_retained = _retained_seed_sets(provenance)
        retained_seeds[name] = stage_retained
        retained_source_count = len(stage_retained["source"])
        retained_target_count = len(stage_retained["target"])

        report = _stage_report(
            stage=stage,
            source_count=len(source_ids),
            target_count=len(target_ids),
            retained_source_count=retained_source_count,
            retained_target_count=retained_target_count,
            node_count=len(nodes),
            edge_count=len(edges),
        )

        audit: dict[str, Any] | None = None
        if report["status"] == "candidate_corridor":
            audit = audit_corridor(nodes, edges, provenance, report)
            (stage_dir / "structural_audit.json").write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n"
            )
        else:
            (stage_dir / "structural_audit.json").unlink(missing_ok=True)
        report["structural_audit_artifact"] = (
            "structural_audit.json" if audit is not None else None
        )
        report["structural_audit_passed"] = (
            bool(audit["passed"]) if audit is not None else None
        )

        (stage_dir / "trace_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        reports.append(report)
        if not nodes.empty:
            aggregate_nodes.append(nodes)
        if not edges.empty:
            aggregate_edges.append(edges)

    # Clear only aggregate products from a previous run when no corridors remain.
    for filename, tables in (
        ("all_stage_nodes.parquet", aggregate_nodes),
        ("all_stage_edges.parquet", aggregate_edges),
    ):
        if not tables:
            (output / filename).unlink(missing_ok=True)
    if aggregate_nodes:
        pd.concat(aggregate_nodes, ignore_index=True).to_parquet(
            output / "all_stage_nodes.parquet",
            index=False,
        )
    if aggregate_edges:
        pd.concat(aggregate_edges, ignore_index=True).to_parquet(
            output / "all_stage_edges.parquet",
            index=False,
        )

    handoff_audit = _audit_handoffs(config.get("handoffs", []), retained_seeds)
    (output / "handoff_audit.json").write_text(
        json.dumps(handoff_audit, indent=2, sort_keys=True) + "\n"
    )

    candidate_count = sum(
        report["status"] == "candidate_corridor" for report in reports
    )
    required_reports = [
        report for report in reports if report["required_for_primary_hypothesis"]
    ]
    required_candidate_count = sum(
        report["status"] == "candidate_corridor" for report in required_reports
    )
    required_audit_pass_count = sum(
        report["status"] == "candidate_corridor"
        and report["structural_audit_passed"] is True
        for report in required_reports
    )
    required_stages_ready = required_audit_pass_count == len(required_reports)
    primary_ready = required_stages_ready and handoff_audit["all_required_handoffs_pass"]

    summary = {
        "protocol": "staged-structural-discovery-v3",
        "dataset": config.get("dataset", "unspecified"),
        "dataset_release": config.get("dataset_release"),
        "purpose": config.get("purpose", "structural discovery"),
        "literature_authority": config.get("literature_authority"),
        "input_provenance": input_provenance,
        "stage_count": len(reports),
        "candidate_stage_count": candidate_count,
        "all_stages_have_candidate_corridors": candidate_count == len(reports),
        "required_stage_count": len(required_reports),
        "required_candidate_stage_count": required_candidate_count,
        "required_stage_audit_pass_count": required_audit_pass_count,
        "all_required_stages_have_audited_candidate_corridors": required_stages_ready,
        "handoff_audit_artifact": "handoff_audit.json",
        "required_handoff_count": handoff_audit["required_handoff_count"],
        "required_handoff_pass_count": handoff_audit["required_handoff_pass_count"],
        "all_required_handoffs_pass": handoff_audit["all_required_handoffs_pass"],
        "primary_structural_hypothesis_passed": primary_ready,
        "handoffs": handoff_audit["handoffs"],
        "stages": reports,
        "warning": (
            "Structural discovery only. Candidate corridors, passing audits, and exact body-ID "
            "handoffs do not establish physiological influence or a qualified MaleCNS controller."
        ),
    }
    (output / "staged_trace_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a staged MaleCNS structural route audit"
    )
    parser.add_argument("annotations", help="MaleCNS body annotation Feather file")
    parser.add_argument("weights", help="MaleCNS connection-weight Feather file")
    parser.add_argument(
        "--config",
        default="configs/staged_route_v0.json",
        help="staged route JSON configuration",
    )
    parser.add_argument("--output", default="data/cache/staged-route-v0")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "exit non-zero unless every primary required stage passes its structural audit "
            "and every required cross-stage body-ID handoff passes"
        ),
    )
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    weights_path = Path(args.weights)
    config_path = Path(args.config)
    annotations = pd.read_feather(annotations_path)
    weights = pd.read_feather(weights_path)
    config = json.loads(config_path.read_text())
    input_provenance = {
        "annotations": _file_provenance(annotations_path),
        "weights": _file_provenance(weights_path),
        "config": _file_provenance(config_path),
    }
    summary = run_staged_trace(
        annotations,
        weights,
        config,
        args.output,
        input_provenance=input_provenance,
    )
    print(json.dumps(summary, indent=2))
    if args.strict and not summary["primary_structural_hypothesis_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
