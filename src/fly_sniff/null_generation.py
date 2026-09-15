from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .graph import GraphBundle
from .rewire import save_bundle

GENERATOR = "exact-minimum-overlap-degree-rewire-v1"


def _edge_pairs(bundle: GraphBundle) -> set[tuple[int, int]]:
    return set(
        zip(
            bundle.edges.source.astype(int),
            bundle.edges.target.astype(int),
            strict=True,
        )
    )


def changed_edge_fraction(intact: GraphBundle, rewired: GraphBundle) -> float:
    intact_pairs = _edge_pairs(intact)
    rewired_pairs = _edge_pairs(rewired)
    if not intact_pairs:
        return 0.0
    return float(1.0 - len(intact_pairs & rewired_pairs) / len(intact_pairs))


def audit_degree_null(intact: GraphBundle, rewired: GraphBundle) -> dict[str, Any]:
    intact.validate(require_sign="sign" in intact.edges.columns)
    rewired.validate(require_sign="sign" in rewired.edges.columns)

    def degrees(bundle: GraphBundle, field: str) -> Counter[int]:
        return Counter(int(x) for x in bundle.edges[field].astype(int))

    pair_rows = list(
        zip(
            rewired.edges.source.astype(int),
            rewired.edges.target.astype(int),
            strict=True,
        )
    )
    changed = changed_edge_fraction(intact, rewired)
    weight_identity = sorted(intact.edges.weight.astype(float)) == sorted(
        rewired.edges.weight.astype(float)
    )
    sign_identity = True
    if "sign" in intact.edges.columns or "sign" in rewired.edges.columns:
        sign_identity = (
            "sign" in intact.edges.columns
            and "sign" in rewired.edges.columns
            and sorted(intact.edges.sign.astype(int)) == sorted(rewired.edges.sign.astype(int))
        )

    gates = {
        "node_identity_preserved": set(intact.nodes.bodyId.astype(int))
        == set(rewired.nodes.bodyId.astype(int)),
        "edge_count_preserved": len(intact.edges) == len(rewired.edges),
        "exact_out_degree_identity": degrees(intact, "source") == degrees(rewired, "source"),
        "exact_in_degree_identity": degrees(intact, "target") == degrees(rewired, "target"),
        "weight_multiset_identity": weight_identity,
        "sign_multiset_identity": sign_identity,
        "no_self_loops": all(source != target for source, target in pair_rows),
        "no_duplicate_edges": len(pair_rows) == len(set(pair_rows)),
    }
    return {
        "generator": GENERATOR,
        "passed": all(gates.values()),
        "gates": gates,
        "changed_edge_fraction": changed,
        "intact_overlap_fraction": 1.0 - changed,
        "edge_count": len(intact.edges),
    }


def exact_max_distance_degree_rewire(
    bundle: GraphBundle,
    *,
    seed: int,
    minimum_changed_edge_fraction: float = 0.80,
    max_candidate_pairs: int = 2_000_000,
) -> GraphBundle:
    """Construct a simple directed graph with identical in/out degrees and minimum overlap.

    The topology problem is solved as a bipartite min-cost flow. Existing edges carry
    a dominating overlap penalty, while seeded integer tie costs select among equally
    distant solutions. Edge attributes remain attached to their original presynaptic
    records, so each source preserves its outgoing weight/sign multiset as well as the
    global attribute multisets.

    This exact solver is intended for candidate circuits where the source-target
    candidate product is tractable. It fails closed instead of silently falling back
    to a weaker null on very large graphs.
    """
    bundle.validate(require_sign="sign" in bundle.edges.columns)
    if not 0.0 <= minimum_changed_edge_fraction <= 1.0:
        raise ValueError("minimum_changed_edge_fraction must be in [0, 1]")

    edges = bundle.edges.copy().reset_index(drop=True)
    original_rows = list(
        zip(edges.source.astype(int), edges.target.astype(int), strict=True)
    )
    if len(original_rows) != len(set(original_rows)):
        raise ValueError("exact degree null requires a simple input digraph without duplicate edges")
    if any(source == target for source, target in original_rows):
        raise ValueError("exact degree null requires an input digraph without self-loops")
    if not original_rows:
        raise ValueError("cannot rewire an empty graph")

    out_degree = Counter(source for source, _ in original_rows)
    in_degree = Counter(target for _, target in original_rows)
    sources = sorted(out_degree)
    targets = sorted(in_degree)
    candidate_count = sum(1 for source in sources for target in targets if source != target)
    if candidate_count > int(max_candidate_pairs):
        raise ValueError(
            f"exact degree-null candidate graph is too large ({candidate_count} pairs > "
            f"{max_candidate_pairs}); use a separately qualified scalable generator"
        )

    original = set(original_rows)
    rng = np.random.default_rng(int(seed))
    tie_max = 999
    overlap_penalty = len(original_rows) * tie_max + 1

    flow_graph = nx.DiGraph()
    for source in sources:
        flow_graph.add_node(("source", source), demand=-int(out_degree[source]))
    for target in targets:
        flow_graph.add_node(("target", target), demand=int(in_degree[target]))

    for source in sources:
        for target in targets:
            if source == target:
                continue
            tie_cost = int(rng.integers(0, tie_max + 1))
            overlap_cost = overlap_penalty if (source, target) in original else 0
            flow_graph.add_edge(
                ("source", source),
                ("target", target),
                capacity=1,
                weight=int(overlap_cost + tie_cost),
            )

    flow = nx.min_cost_flow(flow_graph)
    new_targets_by_source: dict[int, list[int]] = {source: [] for source in sources}
    for source in sources:
        outgoing = flow[("source", source)]
        for node, value in outgoing.items():
            if int(value) <= 0:
                continue
            _, target = node
            new_targets_by_source[source].append(int(target))
        if len(new_targets_by_source[source]) != out_degree[source]:
            raise RuntimeError("min-cost flow returned an invalid source degree")
        new_targets_by_source[source].sort()

    rewired_frames: list[pd.DataFrame] = []
    for source in sources:
        source_rows = edges.loc[edges.source.astype(int) == source].copy()
        source_rows = source_rows.sort_values(["target"], kind="stable").reset_index(drop=True)
        targets_for_source = new_targets_by_source[source]
        if len(source_rows) != len(targets_for_source):
            raise RuntimeError("attribute assignment source degree mismatch")
        source_rows.loc[:, "target"] = targets_for_source
        rewired_frames.append(source_rows)

    rewired_edges = pd.concat(rewired_frames, ignore_index=True)
    rewired_edges = rewired_edges.sort_values(["source", "target"], kind="stable").reset_index(
        drop=True
    )
    manifest = copy.deepcopy(bundle.manifest) if bundle.manifest else {}
    rewired = GraphBundle(
        nodes=bundle.nodes.copy(),
        edges=rewired_edges,
        roles={key: list(values) for key, values in bundle.roles.items()},
        manifest=manifest,
    )
    audit = audit_degree_null(bundle, rewired)
    if not audit["passed"]:
        failed = [name for name, passed in audit["gates"].items() if not passed]
        raise RuntimeError(f"exact degree null failed invariants: {failed}")
    if float(audit["changed_edge_fraction"]) + 1e-12 < minimum_changed_edge_fraction:
        raise ValueError(
            "maximum-distance degree-preserving null cannot satisfy frozen distance floor: "
            f"changed={audit['changed_edge_fraction']:.6f} "
            f"required={minimum_changed_edge_fraction:.6f}"
        )

    rewired.manifest["graph_role"] = "degree-preserving-confirmatory-null"
    rewired.manifest["rewire"] = {
        "generator": GENERATOR,
        "seed": int(seed),
        "minimum_changed_edge_fraction": float(minimum_changed_edge_fraction),
        "changed_edge_fraction": float(audit["changed_edge_fraction"]),
        "intact_overlap_fraction": float(audit["intact_overlap_fraction"]),
        "exact_in_out_degree_preserved": True,
        "weight_multiset_preserved": True,
        "sign_multiset_preserved": True,
        "selection_used_behavior_performance": False,
    }
    return rewired


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a confirmatory maximum-distance degree null")
    parser.add_argument("bundle")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--minimum-changed", type=float, default=0.80)
    parser.add_argument("--max-candidate-pairs", type=int, default=2_000_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    intact = GraphBundle.load(args.bundle)
    rewired = exact_max_distance_degree_rewire(
        intact,
        seed=args.seed,
        minimum_changed_edge_fraction=args.minimum_changed,
        max_candidate_pairs=args.max_candidate_pairs,
    )
    save_bundle(rewired, args.output)
    report = audit_degree_null(intact, rewired)
    report["seed"] = int(args.seed)
    report["minimum_changed_edge_fraction"] = float(args.minimum_changed)
    report_path = Path(args.report) if args.report else Path(args.output) / "null-audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{args.output}")
    print(
        f"passed={report['passed']} changed={report['changed_edge_fraction']:.6f} "
        f"overlap={report['intact_overlap_fraction']:.6f}"
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
