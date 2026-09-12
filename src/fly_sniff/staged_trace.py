from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .trace import body_ids_matching, trace_corridor


def _stage_report(
    *,
    stage: dict[str, Any],
    source_count: int,
    target_count: int,
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
        "source_regex": stage["source"],
        "target_regex": stage["target"],
        "source_seed_count": source_count,
        "target_seed_count": target_count,
        "corridor_nodes": node_count,
        "corridor_edges": edge_count,
        "max_hops": stage["max_hops"],
        "min_weight": stage["min_weight"],
        "fanout": stage["fanout"],
        "rationale": stage.get("rationale", ""),
    }


def run_staged_trace(
    annotations: pd.DataFrame,
    weights: pd.DataFrame,
    config: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    """Run independent structural corridor searches for an interpretable route audit.

    Each stage is deliberately evaluated independently. An empty upstream stage
    therefore does not prevent downstream structural questions from being
    inspected, and no stage is promoted from structural discovery to functional
    evidence by this utility.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    aggregate_nodes: list[pd.DataFrame] = []
    aggregate_edges: list[pd.DataFrame] = []

    for stage in config.get("stages", []):
        name = stage["name"]
        stage_dir = output / name
        stage_dir.mkdir(parents=True, exist_ok=True)
        source_ids = body_ids_matching(annotations, stage["source"])
        target_ids = body_ids_matching(annotations, stage["target"])

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

        report = _stage_report(
            stage=stage,
            source_count=len(source_ids),
            target_count=len(target_ids),
            node_count=len(nodes),
            edge_count=len(edges),
        )
        (stage_dir / "trace_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        reports.append(report)
        if not nodes.empty:
            aggregate_nodes.append(nodes)
        if not edges.empty:
            aggregate_edges.append(edges)

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

    candidate_count = sum(
        report["status"] == "candidate_corridor" for report in reports
    )
    summary = {
        "dataset": config.get("dataset", "unspecified"),
        "purpose": config.get("purpose", "structural discovery"),
        "stage_count": len(reports),
        "candidate_stage_count": candidate_count,
        "all_stages_have_candidate_corridors": candidate_count == len(reports),
        "stages": reports,
        "warning": (
            "Structural discovery only. A candidate corridor is not evidence of "
            "functional influence or a qualified MaleCNS controller."
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
        help="exit non-zero unless every configured stage yields a candidate corridor",
    )
    args = parser.parse_args()

    annotations = pd.read_feather(args.annotations)
    weights = pd.read_feather(args.weights)
    config = json.loads(Path(args.config).read_text())
    summary = run_staged_trace(annotations, weights, config, args.output)
    print(json.dumps(summary, indent=2))
    if args.strict and not summary["all_stages_have_candidate_corridors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
