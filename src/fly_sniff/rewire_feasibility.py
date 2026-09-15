from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx

from .graph import GraphBundle

PROTOCOL = "E004a-rewire-feasibility-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_changed_edge_fraction(bundle: GraphBundle) -> dict[str, Any]:
    """Compute the exact maximum edge distance under the frozen simple-digraph degrees.

    The problem is a bipartite minimum-cost flow. Source-side copies of neurons
    supply their exact out-degree; target-side copies consume their exact in-degree.
    Candidate self-loops are omitted and every source-target pair has capacity one,
    enforcing a simple directed graph. Original edges cost one and all other legal
    edges cost zero, so the minimum flow cost is the fewest original edges any
    degree-preserving realization must retain.
    """
    bundle.validate(require_sign="sign" in bundle.edges.columns, require_qualified=False)
    edges = bundle.edges
    edge_count = len(edges)
    if edge_count == 0:
        return {
            "edge_count": 0,
            "minimum_unavoidable_original_edges": 0,
            "maximum_changed_edge_count": 0,
            "maximum_changed_edge_fraction": 0.0,
        }

    original = set(zip(edges.source.astype(int), edges.target.astype(int), strict=True))
    if len(original) != edge_count:
        raise ValueError("feasibility bound requires a simple graph with no duplicate edges")
    if any(source == target for source, target in original):
        raise ValueError("feasibility bound assumes self-loops are forbidden")

    out_degree = edges.source.astype(int).value_counts().to_dict()
    in_degree = edges.target.astype(int).value_counts().to_dict()
    source_ids = sorted(int(x) for x in out_degree)
    target_ids = sorted(int(x) for x in in_degree)

    flow = nx.DiGraph()
    root = ("root", -1)
    sink = ("sink", -1)
    flow.add_node(root, demand=-edge_count)
    flow.add_node(sink, demand=edge_count)

    for body_id in source_ids:
        node = ("out", body_id)
        flow.add_node(node, demand=0)
        flow.add_edge(root, node, capacity=int(out_degree[body_id]), weight=0)
    for body_id in target_ids:
        node = ("in", body_id)
        flow.add_node(node, demand=0)
        flow.add_edge(node, sink, capacity=int(in_degree[body_id]), weight=0)

    for source in source_ids:
        out_node = ("out", source)
        for target in target_ids:
            if source == target:
                continue
            in_node = ("in", target)
            flow.add_edge(
                out_node,
                in_node,
                capacity=1,
                weight=1 if (source, target) in original else 0,
            )

    try:
        cost, flow_dict = nx.network_simplex(flow)
    except nx.NetworkXUnfeasible as exc:
        raise RuntimeError("degree sequence unexpectedly has no legal simple-digraph realization") from exc

    realized_edges = 0
    overlap = 0
    for source in source_ids:
        for target, value in flow_dict[("out", source)].items():
            if not isinstance(target, tuple) or target[0] != "in" or value <= 0:
                continue
            realized_edges += int(value)
            overlap += int(value) * int((source, int(target[1])) in original)
    if realized_edges != edge_count or overlap != int(cost):
        raise RuntimeError("minimum-cost-flow accounting mismatch")

    changed = edge_count - overlap
    return {
        "edge_count": int(edge_count),
        "minimum_unavoidable_original_edges": int(overlap),
        "maximum_changed_edge_count": int(changed),
        "maximum_changed_edge_fraction": float(changed / edge_count),
    }


def run_feasibility(bundle: GraphBundle, config: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol") != "E004a-rewire-prequalification-v1":
        raise ValueError("unexpected E004a config protocol")
    bound = maximum_changed_edge_fraction(bundle)
    floor = float(config["rewire"]["minimum_changed_edge_fraction"])
    return {
        "protocol": PROTOCOL,
        "dataset": config["dataset"],
        "frozen_minimum_changed_edge_fraction": floor,
        **bound,
        "frozen_floor_is_degree_feasible": bool(bound["maximum_changed_edge_fraction"] >= floor),
        "interpretation": (
            "This is a topology-only feasibility bound, not a randomization result. If the frozen floor is infeasible, "
            "the null must move to a less constrained/larger graph rather than lowering the floor after behavior. If it "
            "is feasible, an under-mixed swap chain is an implementation/null-generation issue."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute exact E004a degree-constrained mixing feasibility")
    parser.add_argument("bundle")
    parser.add_argument("--config", default="configs/e004a_rewire_prequalification_v1.json")
    parser.add_argument("--output", default="results/e004/rewire-feasibility-v1.json")
    args = parser.parse_args()

    bundle = GraphBundle.load(args.bundle)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    report = run_feasibility(bundle, config)
    report["input_sha256"] = {
        "config": _sha256(config_path),
        "bundle_manifest": _sha256(Path(args.bundle) / "manifest.json"),
        "bundle_edges": _sha256(Path(args.bundle) / "edges.parquet"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    print(
        "max_changed="
        f"{report['maximum_changed_edge_fraction']:.3f} "
        f"frozen_floor={report['frozen_minimum_changed_edge_fraction']:.3f} "
        f"feasible={report['frozen_floor_is_degree_feasible']}"
    )


if __name__ == "__main__":
    main()
