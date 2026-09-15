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

GENERATOR = "exact-minimum-overlap-constrained-rewire-v1"


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


def _node_metadata(
    bundle: GraphBundle,
    columns: tuple[str, ...],
) -> dict[int, tuple[str, ...]]:
    if not columns:
        return {int(body_id): () for body_id in bundle.nodes.bodyId.astype(int)}
    missing = [name for name in columns if name not in bundle.nodes.columns]
    if missing:
        raise ValueError(f"node metadata columns missing for constrained null: {missing}")
    metadata: dict[int, tuple[str, ...]] = {}
    for row in bundle.nodes[["bodyId", *columns]].itertuples(index=False, name=None):
        body_id = int(row[0])
        values = tuple(str(value) for value in row[1:])
        if any(value in {"", "None", "nan", "<NA>"} for value in values):
            raise ValueError(
                f"node {body_id} has unresolved metadata required by constrained null: {columns}"
            )
        metadata[body_id] = values
    return metadata


def _source_target_metadata_counts(
    bundle: GraphBundle,
    columns: tuple[str, ...],
) -> Counter[tuple[int, tuple[str, ...]]]:
    metadata = _node_metadata(bundle, columns)
    return Counter(
        (int(source), metadata[int(target)])
        for source, target in zip(
            bundle.edges.source.astype(int),
            bundle.edges.target.astype(int),
            strict=True,
        )
    )


def audit_degree_null(
    intact: GraphBundle,
    rewired: GraphBundle,
    *,
    target_metadata_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
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
    metadata_identity = _source_target_metadata_counts(
        intact, target_metadata_columns
    ) == _source_target_metadata_counts(rewired, target_metadata_columns)

    gates = {
        "node_identity_preserved": set(intact.nodes.bodyId.astype(int))
        == set(rewired.nodes.bodyId.astype(int)),
        "edge_count_preserved": len(intact.edges) == len(rewired.edges),
        "exact_out_degree_identity": degrees(intact, "source") == degrees(rewired, "source"),
        "exact_in_degree_identity": degrees(intact, "target") == degrees(rewired, "target"),
        "weight_multiset_identity": weight_identity,
        "sign_multiset_identity": sign_identity,
        "target_metadata_constraint_identity": metadata_identity,
        "no_self_loops": all(source != target for source, target in pair_rows),
        "no_duplicate_edges": len(pair_rows) == len(set(pair_rows)),
    }
    return {
        "generator": GENERATOR,
        "passed": all(gates.values()),
        "gates": gates,
        "target_metadata_columns": list(target_metadata_columns),
        "changed_edge_fraction": changed,
        "intact_overlap_fraction": 1.0 - changed,
        "edge_count": len(intact.edges),
    }


def exact_max_distance_constrained_rewire(
    bundle: GraphBundle,
    *,
    seed: int,
    minimum_changed_edge_fraction: float = 0.80,
    target_metadata_columns: tuple[str, ...] = (),
    max_candidate_pairs: int = 2_000_000,
) -> GraphBundle:
    """Minimize intact edge overlap under exact degree and metadata constraints.

    Each source preserves its exact out-degree and the exact multiset of target
    metadata classes defined by ``target_metadata_columns``. Each target preserves
    its exact in-degree. Because edge attribute records remain attached to their
    original presynaptic source rows, source-specific outgoing weight/sign multisets
    are also preserved. Self-loops and duplicate directed edges are forbidden.

    Empty metadata constraints produce the degree-preserving null. Adding target
    metadata successively creates stricter nulls, such as hemisphere-, cell-type-,
    or spatial-bin-preserving controls. The solver fails closed when constraints are
    infeasible or the exact candidate graph is too large.
    """
    bundle.validate(require_sign="sign" in bundle.edges.columns)
    if not 0.0 <= minimum_changed_edge_fraction <= 1.0:
        raise ValueError("minimum_changed_edge_fraction must be in [0, 1]")

    edges = bundle.edges.copy().reset_index(drop=True)
    original_rows = list(zip(edges.source.astype(int), edges.target.astype(int), strict=True))
    if len(original_rows) != len(set(original_rows)):
        raise ValueError("exact constrained null requires a simple input digraph")
    if any(source == target for source, target in original_rows):
        raise ValueError("exact constrained null requires an input digraph without self-loops")
    if not original_rows:
        raise ValueError("cannot rewire an empty graph")

    metadata = _node_metadata(bundle, target_metadata_columns)
    in_degree = Counter(target for _, target in original_rows)
    group_supply = Counter((source, metadata[target]) for source, target in original_rows)
    targets_by_class: dict[tuple[str, ...], list[int]] = {}
    for target in sorted(in_degree):
        targets_by_class.setdefault(metadata[target], []).append(target)

    candidate_count = sum(
        sum(target != source for target in targets_by_class.get(target_class, []))
        for source, target_class in group_supply
    )
    if candidate_count > int(max_candidate_pairs):
        raise ValueError(
            f"exact constrained-null candidate graph is too large ({candidate_count} pairs > "
            f"{max_candidate_pairs}); use a separately qualified scalable generator"
        )

    original = set(original_rows)
    rng = np.random.default_rng(int(seed))
    tie_max = 999
    overlap_penalty = len(original_rows) * tie_max + 1
    flow_graph = nx.DiGraph()

    for group, supply in group_supply.items():
        source, target_class = group
        node = ("group", source, *target_class)
        flow_graph.add_node(node, demand=-int(supply))
        candidates = targets_by_class.get(target_class, [])
        for target in candidates:
            if target == source:
                continue
            tie_cost = int(rng.integers(0, tie_max + 1))
            overlap_cost = overlap_penalty if (source, target) in original else 0
            flow_graph.add_edge(
                node,
                ("target", target),
                capacity=1,
                weight=int(overlap_cost + tie_cost),
            )

    for target, degree in in_degree.items():
        flow_graph.add_node(("target", target), demand=int(degree))

    try:
        flow = nx.min_cost_flow(flow_graph)
    except (nx.NetworkXUnfeasible, nx.NetworkXError) as exc:
        raise ValueError(
            "no simple degree-preserving graph satisfies the requested target metadata constraints"
        ) from exc

    new_targets_by_group: dict[tuple[int, tuple[str, ...]], list[int]] = {
        group: [] for group in group_supply
    }
    for group in group_supply:
        source, target_class = group
        node = ("group", source, *target_class)
        for target_node, value in flow[node].items():
            if int(value) <= 0:
                continue
            _, target = target_node
            new_targets_by_group[group].append(int(target))
        if len(new_targets_by_group[group]) != group_supply[group]:
            raise RuntimeError("min-cost flow returned an invalid constrained source degree")
        new_targets_by_group[group].sort()

    rewired_frames: list[pd.DataFrame] = []
    for source, target_class in sorted(group_supply, key=lambda item: (item[0], item[1])):
        source_rows = edges.loc[edges.source.astype(int) == source].copy()
        mask = source_rows.target.astype(int).map(metadata).eq(target_class)
        group_rows = source_rows.loc[mask].copy()
        group_rows = group_rows.sort_values(["target"], kind="stable").reset_index(drop=True)
        targets_for_group = new_targets_by_group[(source, target_class)]
        if len(group_rows) != len(targets_for_group):
            raise RuntimeError("attribute assignment constrained-group degree mismatch")
        group_rows.loc[:, "target"] = targets_for_group
        rewired_frames.append(group_rows)

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
    audit = audit_degree_null(
        bundle,
        rewired,
        target_metadata_columns=target_metadata_columns,
    )
    if not audit["passed"]:
        failed = [name for name, passed in audit["gates"].items() if not passed]
        raise RuntimeError(f"exact constrained null failed invariants: {failed}")
    if float(audit["changed_edge_fraction"]) + 1e-12 < minimum_changed_edge_fraction:
        raise ValueError(
            "maximum-distance constrained null cannot satisfy frozen distance floor: "
            f"changed={audit['changed_edge_fraction']:.6f} "
            f"required={minimum_changed_edge_fraction:.6f}"
        )

    rewired.manifest["graph_role"] = "confirmatory-topology-null"
    rewired.manifest["rewire"] = {
        "generator": GENERATOR,
        "seed": int(seed),
        "minimum_changed_edge_fraction": float(minimum_changed_edge_fraction),
        "changed_edge_fraction": float(audit["changed_edge_fraction"]),
        "intact_overlap_fraction": float(audit["intact_overlap_fraction"]),
        "target_metadata_columns": list(target_metadata_columns),
        "exact_in_out_degree_preserved": True,
        "weight_multiset_preserved": True,
        "sign_multiset_preserved": True,
        "selection_used_behavior_performance": False,
    }
    return rewired


def exact_max_distance_degree_rewire(
    bundle: GraphBundle,
    *,
    seed: int,
    minimum_changed_edge_fraction: float = 0.80,
    max_candidate_pairs: int = 2_000_000,
) -> GraphBundle:
    return exact_max_distance_constrained_rewire(
        bundle,
        seed=seed,
        minimum_changed_edge_fraction=minimum_changed_edge_fraction,
        target_metadata_columns=(),
        max_candidate_pairs=max_candidate_pairs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a confirmatory maximum-distance topology null")
    parser.add_argument("bundle")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--minimum-changed", type=float, default=0.80)
    parser.add_argument("--target-metadata", action="append", default=[])
    parser.add_argument("--max-candidate-pairs", type=int, default=2_000_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    intact = GraphBundle.load(args.bundle)
    metadata_columns = tuple(str(x) for x in args.target_metadata)
    rewired = exact_max_distance_constrained_rewire(
        intact,
        seed=args.seed,
        minimum_changed_edge_fraction=args.minimum_changed,
        target_metadata_columns=metadata_columns,
        max_candidate_pairs=args.max_candidate_pairs,
    )
    save_bundle(rewired, args.output)
    report = audit_degree_null(
        intact,
        rewired,
        target_metadata_columns=metadata_columns,
    )
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
    print(f"metadata={','.join(metadata_columns) if metadata_columns else 'degree-only'}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
