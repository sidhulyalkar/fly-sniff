from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .graph import GraphBundle


def _edge_pairs(edges: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in zip(edges.source, edges.target, strict=True)
    }


def _body_id_column(nodes: pd.DataFrame) -> str:
    if "bodyId" in nodes.columns:
        return "bodyId"
    if "body_id" in nodes.columns:
        return "body_id"
    raise ValueError("constrained rewiring requires nodes.bodyId (or legacy body_id)")


def _seeded_tie_ranks(
    candidates: list[tuple[tuple[Any, ...], int]],
    *,
    seed: int,
) -> dict[tuple[tuple[Any, ...], int], int]:
    """Assign unique deterministic seed-specific ranks to otherwise equivalent edges."""
    if not candidates:
        return {}
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(candidates))
    return {
        candidate: int(permutation[index])
        for index, candidate in enumerate(candidates)
    }


def changed_edge_fraction(original: GraphBundle, rewired: GraphBundle) -> float:
    original_pairs = _edge_pairs(original.edges)
    rewired_pairs = _edge_pairs(rewired.edges)
    if len(original_pairs) != len(rewired_pairs):
        raise ValueError("changed-edge comparison requires equal edge counts")
    if not original_pairs:
        return 0.0
    return float(len(original_pairs - rewired_pairs) / len(original_pairs))


def _edge_invariants(bundle: GraphBundle) -> dict[str, Any]:
    edges = bundle.edges
    return {
        "edge_count": int(len(edges)),
        "out_degree": Counter(int(x) for x in edges.source),
        "in_degree": Counter(int(x) for x in edges.target),
        "weight_multiset": Counter(float(x) for x in edges.weight),
        "sign_multiset": Counter(int(x) for x in edges.sign) if "sign" in edges.columns else None,
    }


def validate_rewire(original: GraphBundle, rewired: GraphBundle) -> dict[str, Any]:
    before = _edge_invariants(original)
    after = _edge_invariants(rewired)
    pairs = list(zip(rewired.edges.source.astype(int), rewired.edges.target.astype(int), strict=True))
    duplicate_count = len(pairs) - len(set(pairs))
    self_loop_count = sum(int(source == target) for source, target in pairs)
    checks = {
        "edge_count_preserved": before["edge_count"] == after["edge_count"],
        "out_degree_preserved": before["out_degree"] == after["out_degree"],
        "in_degree_preserved": before["in_degree"] == after["in_degree"],
        "weight_multiset_preserved": before["weight_multiset"] == after["weight_multiset"],
        "sign_multiset_preserved": before["sign_multiset"] == after["sign_multiset"],
        "no_duplicate_edges": duplicate_count == 0,
        "no_self_loops": self_loop_count == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "changed_edge_fraction": changed_edge_fraction(original, rewired),
        "duplicate_edge_count": int(duplicate_count),
        "self_loop_count": int(self_loop_count),
    }


def degree_preserving_rewire(
    bundle: GraphBundle,
    *,
    seed: int,
    swaps_per_edge: int = 8,
) -> GraphBundle:
    """Directed simple-graph endpoint swaps preserving exact in/out degree.

    This is an engineering kernel. Confirmatory use must separately verify final graph
    distance, invariants, frozen seed provenance and the relevant NullFactory contract.
    """
    rng = np.random.default_rng(seed)
    edges = bundle.edges.copy().reset_index(drop=True)
    pairs = [(int(s), int(t)) for s, t in zip(edges.source, edges.target, strict=True)]
    occupied = set(pairs)
    target_swaps = int(swaps_per_edge) * len(edges)
    accepted = 0
    attempts = 0
    max_attempts = max(1000, target_swaps * 100)
    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(edges), size=2, replace=False)
        source_a, target_a = pairs[int(i)]
        source_b, target_b = pairs[int(j)]
        if source_a == source_b or target_a == target_b:
            continue
        new_a = (source_a, target_b)
        new_b = (source_b, target_a)
        if source_a == target_b or source_b == target_a:
            continue
        if new_a in occupied or new_b in occupied:
            continue
        occupied.remove((source_a, target_a))
        occupied.remove((source_b, target_b))
        occupied.add(new_a)
        occupied.add(new_b)
        pairs[int(i)] = new_a
        pairs[int(j)] = new_b
        accepted += 1
    if accepted != target_swaps:
        raise RuntimeError(
            f"degree-preserving rewire under-mixed during generation: "
            f"accepted={accepted} requested={target_swaps}"
        )
    edges.loc[:, "target"] = [target for _, target in pairs]
    manifest = copy.deepcopy(bundle.manifest) if bundle.manifest else {}
    manifest.update(
        {
            "rewire_family": "degree_preserving-engineering-swap-chain",
            "rewire_seed": int(seed),
            "rewire_swaps_per_edge": int(swaps_per_edge),
            "rewire_accepted_swaps": int(accepted),
            "qualification_status": "candidate",
        }
    )
    return GraphBundle(
        nodes=bundle.nodes.copy(),
        edges=edges,
        roles={key: list(values) for key, values in bundle.roles.items()},
        manifest=manifest,
    )


def _metadata_lookup(
    bundle: GraphBundle,
    columns: tuple[str, ...],
) -> dict[int, tuple[str, ...]]:
    id_column = _body_id_column(bundle.nodes)
    missing = [column for column in columns if column not in bundle.nodes.columns]
    if missing:
        raise ValueError(f"missing constrained-rewire metadata columns: {missing}")
    lookup: dict[int, tuple[str, ...]] = {}
    for row in bundle.nodes[[id_column, *columns]].itertuples(index=False, name=None):
        body_id = int(row[0])
        values = tuple("" if value is None or pd.isna(value) else str(value) for value in row[1:])
        if any(not value for value in values):
            raise ValueError(
                f"constrained rewiring requires complete metadata {columns}; body {body_id} is missing a value"
            )
        lookup[body_id] = values
    return lookup


def _attribute_preserving_target_rewire(
    bundle: GraphBundle,
    *,
    metadata_columns: tuple[str, ...],
    seed: int,
    family: str = "metadata_constrained_minimum_overlap",
) -> GraphBundle:
    """Construct a seeded maximum-distance degree-preserving target reassignment.

    The min-cost flow uses a lexicographic objective. One additional intact edge costs
    more than the maximum possible total seed-specific tie cost, so the primary optimum
    is always minimum overlap. The seed only selects among equally distant solutions.
    """
    metadata = _metadata_lookup(bundle, metadata_columns)
    edges = bundle.edges.copy().reset_index(drop=True)
    for body_id in set(edges.source.astype(int)) | set(edges.target.astype(int)):
        if body_id not in metadata:
            raise ValueError(f"body {body_id} missing from constrained-rewire metadata lookup")

    group_supply: Counter[tuple[int, tuple[str, ...]]] = Counter()
    target_demand: Counter[int] = Counter()
    for source, target in zip(edges.source.astype(int), edges.target.astype(int), strict=True):
        group_supply[(source, metadata[target])] += 1
        target_demand[target] += 1

    targets_by_class: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for target in sorted(target_demand):
        targets_by_class[metadata[target]].append(target)

    candidates: list[tuple[tuple[Any, ...], int]] = []
    for source, target_class in sorted(group_supply, key=lambda item: (item[0], item[1])):
        group_key: tuple[Any, ...] = (source, *target_class)
        for target in targets_by_class[target_class]:
            if source != target:
                candidates.append((group_key, int(target)))
    tie_ranks = _seeded_tie_ranks(candidates, seed=seed)
    overlap_scale = len(edges) * max(len(candidates), 1) + 1

    flow_graph = nx.DiGraph()
    total_edges = len(edges)
    flow_graph.add_node("source", demand=-total_edges)
    flow_graph.add_node("sink", demand=total_edges)
    for target in sorted(target_demand):
        flow_graph.add_node(("target", target), demand=0)
        flow_graph.add_edge(("target", target), "sink", capacity=target_demand[target], weight=0)

    intact_pairs = _edge_pairs(bundle.edges)
    for source, target_class in sorted(group_supply, key=lambda item: (item[0], item[1])):
        group = (source, target_class)
        group_key = (source, *target_class)
        group_node = ("group", source, *target_class)
        flow_graph.add_node(group_node, demand=0)
        flow_graph.add_edge("source", group_node, capacity=group_supply[group], weight=0)
        for target in targets_by_class[target_class]:
            if source == target:
                continue
            overlap_cost = overlap_scale if (source, target) in intact_pairs else 0
            tie_cost = tie_ranks[(group_key, int(target))]
            flow_graph.add_edge(
                group_node,
                ("target", target),
                capacity=1,
                weight=int(overlap_cost + tie_cost),
            )

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
    manifest.update(
        {
            "rewire_family": family,
            "rewire_seed": int(seed),
            "rewire_objective": "minimum-intact-overlap-with-seeded-optimum-tie-break",
            "qualification_status": "candidate",
        }
    )
    rewired = GraphBundle(
        nodes=bundle.nodes.copy(),
        edges=rewired_edges,
        roles={key: list(values) for key, values in bundle.roles.items()},
        manifest=manifest,
    )
    validation = validate_rewire(bundle, rewired)
    if not validation["passed"]:
        raise RuntimeError(f"constrained rewire violated graph invariants: {validation}")
    return rewired


def minimum_overlap_degree_rewire(bundle: GraphBundle, *, seed: int) -> GraphBundle:
    return _attribute_preserving_target_rewire(
        bundle,
        metadata_columns=(),
        seed=seed,
        family="degree_preserving",
    )


def sign_preserving_max_distance_rewire(bundle: GraphBundle, *, seed: int) -> GraphBundle:
    if "sign" not in bundle.edges.columns:
        raise ValueError("sign-preserving rewiring requires an edge sign column")
    return _attribute_preserving_target_rewire_with_edge_group(
        bundle,
        edge_group=lambda row: (str(int(row.sign)),),
        metadata_columns=(),
        seed=seed,
        nodes_override=bundle.nodes,
        family="degree_sign_preserving",
    )


def _attribute_preserving_target_rewire_with_edge_group(
    bundle: GraphBundle,
    *,
    edge_group: Callable[[Any], tuple[str, ...]],
    metadata_columns: tuple[str, ...],
    seed: int,
    nodes_override: pd.DataFrame | None = None,
    family: str,
) -> GraphBundle:
    """General seeded constrained min-cost target assignment with source-row groups."""
    nodes = bundle.nodes if nodes_override is None else nodes_override
    id_column = _body_id_column(nodes)
    if metadata_columns:
        metadata = _metadata_lookup(
            GraphBundle(nodes, bundle.edges, bundle.roles, bundle.manifest),
            metadata_columns,
        )
    else:
        metadata = {int(body_id): () for body_id in nodes[id_column].astype(int)}
    edges = bundle.edges.copy().reset_index(drop=True)
    intact_pairs = _edge_pairs(edges)
    target_demand = Counter(int(x) for x in edges.target)
    group_rows: dict[tuple[int, tuple[str, ...], tuple[str, ...]], list[int]] = defaultdict(list)
    for idx, row in enumerate(edges.itertuples(index=False)):
        source = int(row.source)
        target = int(row.target)
        group_rows[(source, edge_group(row), metadata[target])].append(idx)

    targets_by_metadata: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for target in sorted(target_demand):
        targets_by_metadata[metadata[target]].append(target)

    candidates: list[tuple[tuple[Any, ...], int]] = []
    for key in sorted(group_rows, key=str):
        source, edge_class, target_class = key
        group_key: tuple[Any, ...] = (source, *edge_class, "targetclass", *target_class)
        for target in targets_by_metadata[target_class]:
            if target != source:
                candidates.append((group_key, int(target)))
    tie_ranks = _seeded_tie_ranks(candidates, seed=seed)
    overlap_scale = len(edges) * max(len(candidates), 1) + 1

    flow_graph = nx.DiGraph()
    total_edges = len(edges)
    flow_graph.add_node("source", demand=-total_edges)
    flow_graph.add_node("sink", demand=total_edges)
    for target in sorted(target_demand):
        target_node = ("target", target)
        flow_graph.add_node(target_node, demand=0)
        flow_graph.add_edge(target_node, "sink", capacity=target_demand[target], weight=0)

    for key, row_indices in group_rows.items():
        source, edge_class, target_class = key
        group_key = (source, *edge_class, "targetclass", *target_class)
        group_node = ("group", source, *edge_class, "targetclass", *target_class)
        flow_graph.add_node(group_node, demand=0)
        flow_graph.add_edge("source", group_node, capacity=len(row_indices), weight=0)
        for target in targets_by_metadata[target_class]:
            if target == source:
                continue
            overlap_cost = overlap_scale if (source, target) in intact_pairs else 0
            tie_cost = tie_ranks[(group_key, int(target))]
            flow_graph.add_edge(
                group_node,
                ("target", target),
                capacity=1,
                weight=int(overlap_cost + tie_cost),
            )

    try:
        flow = nx.min_cost_flow(flow_graph)
    except (nx.NetworkXUnfeasible, nx.NetworkXError) as exc:
        raise ValueError("no constrained simple degree-preserving rewire exists") from exc

    assigned_targets: dict[tuple[int, tuple[str, ...], tuple[str, ...]], list[int]] = {}
    for key in group_rows:
        source, edge_class, target_class = key
        group_node = ("group", source, *edge_class, "targetclass", *target_class)
        targets = [
            int(target_node[1])
            for target_node, value in flow[group_node].items()
            if int(value) > 0
        ]
        targets.sort()
        if len(targets) != len(group_rows[key]):
            raise RuntimeError("constrained flow assignment cardinality mismatch")
        assigned_targets[key] = targets

    for key, row_indices in group_rows.items():
        targets = assigned_targets[key]
        ordered_rows = sorted(row_indices, key=lambda idx: int(edges.iloc[idx].target))
        for idx, target in zip(ordered_rows, targets, strict=True):
            edges.at[idx, "target"] = int(target)

    edges = edges.sort_values(["source", "target"], kind="stable").reset_index(drop=True)
    manifest = copy.deepcopy(bundle.manifest) if bundle.manifest else {}
    manifest.update(
        {
            "rewire_family": family,
            "rewire_seed": int(seed),
            "rewire_objective": "minimum-intact-overlap-with-seeded-optimum-tie-break",
            "qualification_status": "candidate",
        }
    )
    rewired = GraphBundle(
        nodes=nodes.copy(),
        edges=edges,
        roles={key: list(values) for key, values in bundle.roles.items()},
        manifest=manifest,
    )
    validation = validate_rewire(bundle, rewired)
    if not validation["passed"]:
        raise RuntimeError(f"constrained rewire violated graph invariants: {validation}")
    return rewired


def sign_and_metadata_preserving_rewire(
    bundle: GraphBundle,
    *,
    metadata_columns: tuple[str, ...],
    seed: int,
    family: str,
) -> GraphBundle:
    if "sign" not in bundle.edges.columns:
        raise ValueError("sign-constrained rewiring requires an edge sign column")
    return _attribute_preserving_target_rewire_with_edge_group(
        bundle,
        edge_group=lambda row: (str(int(row.sign)),),
        metadata_columns=metadata_columns,
        seed=seed,
        nodes_override=bundle.nodes,
        family=family,
    )


def hemisphere_preserving_rewire(bundle: GraphBundle, *, seed: int) -> GraphBundle:
    return sign_and_metadata_preserving_rewire(
        bundle,
        metadata_columns=("hemisphere_class",),
        seed=seed,
        family="degree_sign_hemisphere_preserving",
    )


def type_preserving_rewire(bundle: GraphBundle, *, seed: int) -> GraphBundle:
    return sign_and_metadata_preserving_rewire(
        bundle,
        metadata_columns=("cell_type",),
        seed=seed,
        family="within_cell_type_rewire",
    )


def spatially_constrained_rewire(
    bundle: GraphBundle,
    *,
    seed: int,
    spatial_column: str = "spatial_bin",
) -> GraphBundle:
    return sign_and_metadata_preserving_rewire(
        bundle,
        metadata_columns=("cell_type", spatial_column),
        seed=seed,
        family="spatially_constrained_rewire",
    )
