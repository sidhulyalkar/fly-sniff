from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .freeze import canonical_sha256
from .graph import GraphBundle
from .rewire import save_bundle


class NullFamily(str, Enum):
    """Frozen null families for graph-level topology experiments."""

    DIRECTED_DEGREE = "directed_degree"
    SOURCE_ATTRIBUTE = "source_attribute"
    HEMISPHERE = "hemisphere"
    CELL_TYPE = "cell_type"
    CONSTRAINED = "constrained"


def _python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def graph_fingerprint(bundle: GraphBundle) -> str:
    """Content-address graph topology, edge attributes, node IDs, and roles."""

    bundle.validate(require_sign="sign" in bundle.edges.columns)
    node_ids = sorted(int(value) for value in bundle.nodes["bodyId"].tolist())
    edge_columns = sorted(str(column) for column in bundle.edges.columns)
    edge_rows = []
    for _, row in bundle.edges[edge_columns].iterrows():
        edge_rows.append({column: _python_scalar(row[column]) for column in edge_columns})
    edge_rows.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    roles = {
        str(role): sorted(int(body_id) for body_id in body_ids)
        for role, body_ids in sorted(bundle.roles.items())
    }
    return canonical_sha256(
        {
            "node_ids": node_ids,
            "edge_columns": edge_columns,
            "edges": edge_rows,
            "roles": roles,
        }
    )


def _degree_signature(edges: pd.DataFrame) -> dict[str, dict[int, int]]:
    out_counts = Counter(int(value) for value in edges["source"].tolist())
    in_counts = Counter(int(value) for value in edges["target"].tolist())
    return {
        "out": dict(sorted(out_counts.items())),
        "in": dict(sorted(in_counts.items())),
    }


def _source_attribute_signature(
    edges: pd.DataFrame,
    *,
    attribute_columns: tuple[str, ...] = ("weight", "sign"),
) -> dict[int, tuple[tuple[Any, ...], ...]]:
    available = tuple(column for column in attribute_columns if column in edges.columns)
    result: dict[int, tuple[tuple[Any, ...], ...]] = {}
    for source, group in edges.groupby("source", sort=True):
        rows = [
            tuple(_python_scalar(row[column]) for column in available)
            for _, row in group.iterrows()
        ]
        rows.sort(key=repr)
        result[int(source)] = tuple(rows)
    return result


def _resolve_constraint_columns(
    bundle: GraphBundle,
    family: NullFamily,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    if family in {NullFamily.DIRECTED_DEGREE, NullFamily.SOURCE_ATTRIBUTE}:
        if requested:
            raise ValueError(f"{family.value} does not accept constraint columns")
        return ()
    if family is NullFamily.CONSTRAINED:
        if not requested:
            raise ValueError("constrained null requires at least one --constraint-column")
        missing = [column for column in requested if column not in bundle.nodes.columns]
        if missing:
            raise ValueError(f"constraint columns missing from nodes: {missing}")
        return requested

    candidates = {
        NullFamily.HEMISPHERE: ("somaSide", "hemisphere", "side"),
        NullFamily.CELL_TYPE: ("type", "cell_type", "cellType"),
    }[family]
    if requested:
        if len(requested) != 1:
            raise ValueError(f"{family.value} accepts at most one explicit constraint column")
        if requested[0] not in bundle.nodes.columns:
            raise ValueError(f"constraint column missing from nodes: {requested[0]}")
        return requested
    for column in candidates:
        if column in bundle.nodes.columns:
            return (column,)
    raise ValueError(
        f"{family.value} null requires one of node columns {list(candidates)} or an explicit column"
    )


def _node_labels(bundle: GraphBundle, columns: tuple[str, ...]) -> dict[int, tuple[Any, ...]]:
    if not columns:
        return {int(body_id): () for body_id in bundle.nodes["bodyId"].tolist()}
    labels: dict[int, tuple[Any, ...]] = {}
    for _, row in bundle.nodes[["bodyId", *columns]].iterrows():
        body_id = int(row["bodyId"])
        label = tuple(_python_scalar(row[column]) for column in columns)
        if any(value is None or value == "" for value in label):
            raise ValueError(
                f"bodyId {body_id} lacks complete metadata for constraints {list(columns)}"
            )
        labels[body_id] = label
    return labels


def _edge_block_signature(
    edges: pd.DataFrame,
    labels: dict[int, tuple[Any, ...]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for source, target in zip(edges["source"], edges["target"], strict=True):
        source_label = labels[int(source)]
        target_label = labels[int(target)]
        key = json.dumps([source_label, target_label], sort_keys=True, separators=(",", ":"))
        counts[key] += 1
    return dict(sorted(counts.items()))


def _edge_overlap(source: pd.DataFrame, candidate: pd.DataFrame) -> tuple[int, float]:
    source_pairs = set(zip(source["source"].astype(int), source["target"].astype(int), strict=True))
    candidate_pairs = set(
        zip(candidate["source"].astype(int), candidate["target"].astype(int), strict=True)
    )
    overlap = len(source_pairs & candidate_pairs)
    fraction = overlap / len(source_pairs) if source_pairs else 1.0
    return overlap, float(fraction)


@dataclass(frozen=True)
class NullReceipt:
    family: NullFamily
    seed: int
    swaps_per_edge: int
    accepted_swaps: int
    attempted_swaps: int
    target_swaps: int
    constraint_columns: tuple[str, ...]
    source_graph_sha256: str
    output_graph_sha256: str
    edge_count: int
    edge_overlap_count: int
    edge_overlap_fraction: float
    exact_directed_degree_preserved: bool
    source_weight_sign_multiset_preserved: bool
    constraint_block_counts_preserved: bool
    complete_requested_swap_budget: bool
    stationarity_claimed: bool = False
    schema: str = "fly-sniff-topology-null-receipt-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-topology-null-receipt-v1":
            raise ValueError(f"unsupported null receipt schema: {self.schema}")
        if self.seed < 0:
            raise ValueError("null seed must be non-negative")
        if self.swaps_per_edge <= 0:
            raise ValueError("swaps_per_edge must be positive")
        if self.edge_count < 0 or self.target_swaps < 0:
            raise ValueError("edge and swap counts must be non-negative")
        if not 0.0 <= self.edge_overlap_fraction <= 1.0:
            raise ValueError("edge_overlap_fraction must lie in [0, 1]")
        if not self.exact_directed_degree_preserved:
            raise ValueError("qualified null receipt requires exact directed degree preservation")
        if not self.source_weight_sign_multiset_preserved:
            raise ValueError("qualified null receipt requires presynaptic weight/sign preservation")
        if self.constraint_columns and not self.constraint_block_counts_preserved:
            raise ValueError("constrained null changed frozen metadata block counts")
        if self.stationarity_claimed:
            raise ValueError("fixed swap budgets do not establish Markov-chain stationarity")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "schema": self.schema,
            "family": self.family.value,
            "seed": self.seed,
            "swaps_per_edge": self.swaps_per_edge,
            "accepted_swaps": self.accepted_swaps,
            "attempted_swaps": self.attempted_swaps,
            "target_swaps": self.target_swaps,
            "constraint_columns": list(self.constraint_columns),
            "source_graph_sha256": self.source_graph_sha256,
            "output_graph_sha256": self.output_graph_sha256,
            "edge_count": self.edge_count,
            "edge_overlap_count": self.edge_overlap_count,
            "edge_overlap_fraction": self.edge_overlap_fraction,
            "exact_directed_degree_preserved": self.exact_directed_degree_preserved,
            "source_weight_sign_multiset_preserved": self.source_weight_sign_multiset_preserved,
            "constraint_block_counts_preserved": self.constraint_block_counts_preserved,
            "complete_requested_swap_budget": self.complete_requested_swap_budget,
            "stationarity_claimed": self.stationarity_claimed,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class NullResult:
    bundle: GraphBundle
    receipt: NullReceipt


def generate_topology_null(
    bundle: GraphBundle,
    *,
    family: NullFamily,
    seed: int,
    swaps_per_edge: int = 8,
    constraint_columns: tuple[str, ...] = (),
    require_complete: bool = True,
    max_attempt_multiplier: int = 50,
) -> NullResult:
    """Generate one directed degree-preserving null with frozen metadata constraints.

    The algorithm swaps only edge targets. Every edge row therefore keeps the same
    presynaptic source and all row-level attributes, including structural weight and
    sign. For constrained families, swaps are proposed only between edge rows whose
    source metadata labels match and whose target metadata labels match. This preserves
    exact source→target metadata-block edge counts in addition to directed degree.
    """

    bundle.validate(require_sign="sign" in bundle.edges.columns)
    if swaps_per_edge <= 0:
        raise ValueError("swaps_per_edge must be positive")
    if max_attempt_multiplier <= 0:
        raise ValueError("max_attempt_multiplier must be positive")

    constraints = _resolve_constraint_columns(bundle, family, tuple(constraint_columns))
    labels = _node_labels(bundle, constraints)
    source_edges = bundle.edges.copy().reset_index(drop=True)
    edges = source_edges.copy()
    source_graph_sha256 = graph_fingerprint(bundle)
    source_degree = _degree_signature(source_edges)
    source_attributes = _source_attribute_signature(source_edges)
    source_blocks = _edge_block_signature(source_edges, labels)

    groups: dict[tuple[tuple[Any, ...], tuple[Any, ...]], list[int]] = defaultdict(list)
    for index, row in edges.iterrows():
        source = int(row["source"])
        target = int(row["target"])
        groups[(labels[source], labels[target])].append(int(index))
    eligible_groups = [indices for indices in groups.values() if len(indices) >= 2]
    if not eligible_groups and len(edges) >= 2:
        raise ValueError("no eligible edge pairs exist under the frozen null constraints")

    rng = np.random.default_rng(seed)
    occupied = set(zip(edges["source"].astype(int), edges["target"].astype(int), strict=True))
    target_swaps = int(swaps_per_edge * len(edges))
    max_attempts = max(target_swaps * max_attempt_multiplier, 1000)
    accepted = 0
    attempts = 0

    while accepted < target_swaps and attempts < max_attempts and eligible_groups:
        attempts += 1
        group = eligible_groups[int(rng.integers(0, len(eligible_groups)))]
        i, j = rng.choice(group, size=2, replace=False)
        a, b = int(edges.at[i, "source"]), int(edges.at[i, "target"])
        c, d = int(edges.at[j, "source"]), int(edges.at[j, "target"])
        if a == c or b == d:
            continue
        p1, p2 = (a, d), (c, b)
        if a == d or c == b or p1 == p2:
            continue
        if p1 in occupied or p2 in occupied:
            continue
        occupied.remove((a, b))
        occupied.remove((c, d))
        edges.at[i, "target"] = d
        edges.at[j, "target"] = b
        occupied.add(p1)
        occupied.add(p2)
        accepted += 1

    complete = accepted == target_swaps
    if require_complete and not complete:
        raise RuntimeError(
            "null generation exhausted its preregistered attempt budget: "
            f"accepted {accepted}/{target_swaps} swaps in {attempts} attempts"
        )

    manifest = copy.deepcopy(bundle.manifest) if bundle.manifest else {}
    manifest["graph_role"] = "topology-null"
    manifest["null_family"] = family.value
    manifest["null_seed"] = int(seed)
    manifest["constraint_columns"] = list(constraints)
    output = GraphBundle(
        nodes=bundle.nodes.copy(),
        edges=edges,
        roles={role: list(body_ids) for role, body_ids in bundle.roles.items()},
        manifest=manifest,
    )
    output.validate(require_sign="sign" in edges.columns)

    degree_preserved = _degree_signature(edges) == source_degree
    attributes_preserved = _source_attribute_signature(edges) == source_attributes
    blocks_preserved = _edge_block_signature(edges, labels) == source_blocks
    overlap_count, overlap_fraction = _edge_overlap(source_edges, edges)
    output_graph_sha256 = graph_fingerprint(output)

    receipt = NullReceipt(
        family=family,
        seed=int(seed),
        swaps_per_edge=int(swaps_per_edge),
        accepted_swaps=int(accepted),
        attempted_swaps=int(attempts),
        target_swaps=target_swaps,
        constraint_columns=constraints,
        source_graph_sha256=source_graph_sha256,
        output_graph_sha256=output_graph_sha256,
        edge_count=len(edges),
        edge_overlap_count=overlap_count,
        edge_overlap_fraction=overlap_fraction,
        exact_directed_degree_preserved=degree_preserved,
        source_weight_sign_multiset_preserved=attributes_preserved,
        constraint_block_counts_preserved=blocks_preserved,
        complete_requested_swap_budget=complete,
    )
    receipt.validate()
    return NullResult(bundle=output, receipt=receipt)


def _write_receipt(path: Path, receipt: NullReceipt) -> None:
    path.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a receipt-bearing topology null graph.")
    parser.add_argument("bundle", help="Source GraphBundle directory")
    parser.add_argument("output", help="Output GraphBundle directory")
    parser.add_argument("--family", choices=[family.value for family in NullFamily], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--swaps-per-edge", type=int, default=8)
    parser.add_argument("--constraint-column", action="append", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    source = GraphBundle.load(args.bundle)
    result = generate_topology_null(
        source,
        family=NullFamily(args.family),
        seed=args.seed,
        swaps_per_edge=args.swaps_per_edge,
        constraint_columns=tuple(args.constraint_column),
        require_complete=not args.allow_incomplete,
    )
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    save_bundle(result.bundle, output)
    _write_receipt(output / "null-receipt.json", result.receipt)
    print(json.dumps(result.receipt.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())